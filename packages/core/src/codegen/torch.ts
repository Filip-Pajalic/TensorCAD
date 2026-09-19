/**
 * PyTorch code generation.
 *
 * One generic emitter walks a graph and produces an `nn.Module`. Composites and
 * repeat containers become their own classes, named after the block, so the
 * output reads like code somebody wrote rather than a flattened trace. Because
 * there is a single code path, generated code cannot drift from the analysis:
 * both consume the same expansion.
 *
 * The emitted file has no dependency beyond PyTorch itself.
 */

import type { Doc, Graph, NodeDef, ParamValue, Resolved, SymbolTable } from "../ir/types.js";
import { joinPath, splitEndpoint } from "../ir/types.js";
import { CATALOG, catalogOf, isComposite, isContainer, type Catalog } from "../catalog/index.js";
import { portsOf, resolveNodeParams } from "../catalog/resolve.js";
import type { Ports } from "../catalog/types.js";
import { ex } from "../catalog/types.js";
import { resolveSymbols } from "../ir/symbols.js";
import { evalCtxFor } from "../shapes/infer.js";
import { parsePattern, type PatternAtom } from "../shapes/pattern.js";
import { evalExpr } from "../shapes/expr.js";
import type { Sym } from "../shapes/symexpr.js";
import { countParams } from "../analysis/params.js";
import { flatten } from "../analysis/flatten.js";

export interface TorchOptions {
  /** Class name for the top-level module. */
  className?: string;
  /** Emit a `__main__` block that instantiates the model and checks its size. */
  includeSmokeTest?: boolean;
  /**
   * How a mixture-of-experts layer routes tokens.
   *
   * `"sparse"` gathers the rows each expert was given, which is what you want
   * for speed but uses `nonzero`, whose output shape depends on the data. That
   * makes it untraceable by `torch.export`.
   *
   * `"dense"` runs every expert over every token and weights the results by
   * whether the expert was selected. It computes the same thing, costs
   * `experts / top_k` times as much, and traces cleanly. Use it to verify a
   * design, not to train one.
   */
  moeDispatch?: "sparse" | "dense";
  /**
   * Standard deviation for the weight initialization, or 0 to leave PyTorch's
   * defaults alone.
   *
   * This matters more than it looks. `nn.Embedding` defaults to a unit normal,
   * which starts a language model at a cross-entropy of a few hundred instead
   * of `ln(vocab)`, and it takes a long time to recover.
   */
  initStd?: number;
}

export interface GeneratedFile {
  path: string;
  contents: string;
}

export interface GeneratedCode {
  files: GeneratedFile[];
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Helpers emitted only when the design uses them
// ---------------------------------------------------------------------------

const HELPER_ROPE = `class RotaryEmbedding(nn.Module):
    """Rotary position embedding over the last dimension of (B, H, T, head_dim)."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[-2]
        pos = torch.arange(seq, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(x.device))
        cos = freqs.cos().to(x.dtype)[None, None, :, :]
        sin = freqs.sin().to(x.dtype)[None, None, :, :]
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
`;

const HELPER_WINDOW = `def sliding_window_mask(seq: int, window: int, device, dtype) -> torch.Tensor:
    """Causal mask that also forbids attending further back than \`window\` tokens."""
    i = torch.arange(seq, device=device)
    allowed = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)
    mask = torch.zeros(seq, seq, device=device, dtype=dtype)
    return mask.masked_fill(~allowed, float("-inf"))
`;

const HELPER_SSD = `class SSDScan(nn.Module):
    """Mamba-2 state-space scan.

    A readable sequential reference so the generated file runs unmodified. For a
    real training run, swap this for the fused kernel in \`mamba_ssm\`; it is the
    same recurrence but orders of magnitude faster.
    """

    def __init__(self, heads: int, head_dim: int, state: int, groups: int):
        super().__init__()
        self.heads, self.head_dim, self.state, self.groups = heads, head_dim, state, groups
        self.A_log = nn.Parameter(torch.zeros(heads))
        self.D = nn.Parameter(torch.ones(heads))
        self.dt_bias = nn.Parameter(torch.zeros(heads))

    def forward(self, xbc: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        b, t, _ = xbc.shape
        d_inner = self.heads * self.head_dim
        gs = self.groups * self.state
        x, bs, cs = torch.split(xbc, [d_inner, gs, gs], dim=-1)
        x = x.view(b, t, self.heads, self.head_dim)
        rep = self.heads // self.groups
        bs = bs.view(b, t, self.groups, self.state).repeat_interleave(rep, dim=2)
        cs = cs.view(b, t, self.groups, self.state).repeat_interleave(rep, dim=2)
        dt = F.softplus(dt + self.dt_bias)
        a = -torch.exp(self.A_log)
        h = torch.zeros(b, self.heads, self.head_dim, self.state, device=xbc.device, dtype=torch.float32)
        out = []
        for i in range(t):
            decay = torch.exp(dt[:, i] * a)[:, :, None, None].float()
            h = h * decay + (dt[:, i][:, :, None, None] * x[:, i][..., None] * bs[:, i][:, :, None, :]).float()
            y = (h * cs[:, i][:, :, None, :].float()).sum(-1).to(x.dtype) + self.D[None, :, None] * x[:, i]
            out.append(y)
        return torch.stack(out, dim=1).reshape(b, t, d_inner)
`;

const ACTIVATION_CALL: Record<string, (v: string) => string> = {
  silu: (v) => `F.silu(${v})`,
  swish: (v) => `F.silu(${v})`,
  gelu: (v) => `F.gelu(${v})`,
  gelu_tanh: (v) => `F.gelu(${v}, approximate="tanh")`,
  relu: (v) => `F.relu(${v})`,
  relu2: (v) => `F.relu(${v}).square()`,
  tanh: (v) => `torch.tanh(${v})`,
  sigmoid: (v) => `torch.sigmoid(${v})`,
  identity: (v) => v,
};

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

interface Emitted {
  name: string;
  code: string;
}

interface Ctx {
  doc: Doc;
  /** The document's catalog, so blocks it defines itself generate too. */
  catalog: Catalog;
  symbols: SymbolTable;
  warnings: string[];
  /** Deduplicated classes, in dependency order. */
  classes: Emitted[];
  byKey: Map<string, string>;
  usedNames: Set<string>;
  needsRope: boolean;
  needsWindow: boolean;
  needsSsd: boolean;
  moeDispatch: "sparse" | "dense";
}

function pyName(raw: string): string {
  const cleaned = raw.replace(/[^A-Za-z0-9_]/g, "_");
  return /^[0-9]/.test(cleaned) ? `_${cleaned}` : cleaned;
}

function className(base: string, ctx: Ctx): string {
  const pascal = base
    .split(/[^A-Za-z0-9]+/)
    .filter(Boolean)
    .map((p) => p[0].toUpperCase() + p.slice(1))
    .join("");
  let name = pascal || "Block";
  let n = 2;
  while (ctx.usedNames.has(name)) name = `${pascal}${n++}`;
  ctx.usedNames.add(name);
  return name;
}

function pyBool(v: unknown): string {
  return v === true ? "True" : "False";
}

function pyNum(v: number): string {
  if (Number.isInteger(v)) return String(v);
  return String(v);
}

/** Topological order of a graph's nodes. */
function order(graph: Graph): NodeDef[] {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const deps = new Map<string, Set<string>>();
  for (const n of graph.nodes) deps.set(n.id, new Set());
  for (const [from, to] of graph.edges) {
    const f = splitEndpoint(from);
    const t = splitEndpoint(to);
    if (byId.has(f.node) && byId.has(t.node)) deps.get(t.node)!.add(f.node);
  }
  const out: NodeDef[] = [];
  const state = new Map<string, boolean>();
  const visit = (id: string): void => {
    if (state.get(id)) return;
    state.set(id, true);
    for (const d of deps.get(id) ?? []) visit(d);
    const n = byId.get(id);
    if (n) out.push(n);
  };
  for (const n of graph.nodes) visit(n.id);
  return out;
}

/** Python code for a rearrange, as a list of statements acting on `src`. */
function emitRearrange(
  resolved: Resolved,
  ctx: Ctx,
  symbols: SymbolTable,
  src: string,
  dst: string,
  path: string,
): string[] {
  const evalCtx = evalCtxFor(resolved, symbols);
  const from = parsePattern(String(resolved.p.from));
  const to = parsePattern(String(resolved.p.to));

  if (from.atoms.some((a) => a.kind === "ellipsis") || to.atoms.some((a) => a.kind === "ellipsis")) {
    ctx.warnings.push(`${path}: cannot generate code for a rearrange that uses "...".`);
    return [`${dst} = ${src}  # TODO: unsupported rearrange ${resolved.p.from} -> ${resolved.p.to}`];
  }

  // A pattern atom is an expression, not a bare name: composite expansion
  // substitutes parameters in, so the atom may read `((H))` or `2*D`.
  const symOf = (expr: string): Sym | null => {
    try {
      return evalExpr(expr, evalCtx);
    } catch {
      return null;
    }
  };
  /** The runtime symbol this atom *is*, or null when it is a fixed size. */
  const runtimeName = (expr: string): string | null => {
    const atom = symOf(expr)?.asAtom() ?? null;
    return atom && symbols.runtime.has(atom) ? atom : null;
  };
  const isRuntime = (expr: string): boolean => runtimeName(expr) !== null;
  const sizeOf = (expr: string): number | null => symOf(expr)?.toNumber(evalCtx.values) ?? null;
  /** Canonical form, so `((H))` on one side matches `(H)` on the other. */
  const canonical = (expr: string): string => symOf(expr)?.toString() ?? expr.trim();

  const lines: string[] = [];

  // Name each incoming dimension; runtime dimensions are read from the tensor.
  const groupVars: string[] = from.atoms.map((atom, i) => {
    const a = atom as Extract<PatternAtom, { kind: "dims" }>;
    if (a.parts.length === 1) {
      const rt = runtimeName(a.parts[0]);
      if (rt) return pyName(rt);
    }
    return `_d${i}`;
  });
  lines.push(`${groupVars.join(", ")} = ${src}.shape`);

  // Flatten both sides to atomic axes keyed by their expression.
  interface Axis {
    key: string;
    code: string;
  }
  const fromAxes: Axis[] = [];
  for (let i = 0; i < from.atoms.length; i++) {
    const a = from.atoms[i] as Extract<PatternAtom, { kind: "dims" }>;
    for (const part of a.parts) {
      const key = part.trim();
      if (isRuntime(key)) {
        if (a.parts.length > 1) {
          ctx.warnings.push(`${path}: a runtime dimension inside a group is not supported.`);
        }
        fromAxes.push({ key: canonical(key), code: groupVars[i] });
      } else {
        const n = sizeOf(key);
        if (n === null) {
          ctx.warnings.push(`${path}: could not evaluate "${key}" while generating a rearrange.`);
          return [`${dst} = ${src}  # TODO: unsupported rearrange`];
        }
        fromAxes.push({ key: canonical(key), code: pyNum(n) });
      }
    }
  }

  const needsView = from.atoms.some((a) => a.kind === "dims" && a.parts.length > 1);
  let expr = src;
  if (needsView) {
    expr = `${expr}.view(${fromAxes.map((a) => a.code).join(", ")})`;
  }

  // Permutation, matched by axis expression.
  const toFlat: string[] = [];
  for (const atom of to.atoms) {
    const a = atom as Extract<PatternAtom, { kind: "dims" }>;
    for (const part of a.parts) toFlat.push(canonical(part.trim()));
  }
  const used = new Set<number>();
  const perm: number[] = [];
  let permutable = toFlat.length === fromAxes.length;
  for (const key of toFlat) {
    const idx = fromAxes.findIndex((a, i) => a.key === key && !used.has(i));
    if (idx < 0) {
      permutable = false;
      break;
    }
    used.add(idx);
    perm.push(idx);
  }
  if (!permutable) {
    ctx.warnings.push(
      `${path}: the dimensions of "${resolved.p.from}" and "${resolved.p.to}" do not correspond, so the generated reshape may be wrong.`,
    );
  } else if (!perm.every((v, i) => v === i)) {
    expr = `${expr}.permute(${perm.join(", ")})`;
  }

  // Regroup into the target shape.
  // Only a grouped target dimension needs a reshape: when every target dim is a
  // single axis, the view and permute above already produced that shape.
  const needsReshape = to.atoms.some((a) => a.kind === "dims" && a.parts.length > 1);
  if (needsReshape) {
    const sizes = to.atoms.map((atom) => {
      const a = atom as Extract<PatternAtom, { kind: "dims" }>;
      const parts = a.parts.map((part) => {
        const key = part.trim();
        const rt = runtimeName(key);
        if (rt) return pyName(rt);
        const n = sizeOf(key);
        return n === null ? "-1" : pyNum(n);
      });
      if (parts.length === 1) return parts[0];
      // Fold the constant factors so the emitted code stays readable.
      const nums = parts.filter((p) => /^[0-9]+$/.test(p)).map(Number);
      const rest = parts.filter((p) => !/^[0-9]+$/.test(p));
      const folded = nums.length ? String(nums.reduce((x, y) => x * y, 1)) : null;
      return [...rest, folded].filter(Boolean).join(" * ");
    });
    expr = `${expr}.reshape(${sizes.join(", ")})`;
  }

  lines.push(`${dst} = ${expr}`);
  return lines;
}

interface GraphEmit {
  init: string[];
  forward: string[];
  outputs: Record<string, string>;
}

/**
 * Emit the body of one graph. `inputs` maps the boundary port names to the
 * variable names they arrive in.
 */
function emitGraph(graph: Graph, prefix: string, inputs: Record<string, string>, ctx: Ctx): GraphEmit {
  const init: string[] = [];
  const forward: string[] = [];
  const outputs: Record<string, string> = {};

  const producers = new Map<string, string>();
  for (const [from, to] of graph.edges) producers.set(to, from);

  /** Variable holding the value on a given output endpoint. */
  const vars = new Map<string, string>();
  const valueOf = (nodeId: string, port: string): string => {
    const v = vars.get(`${nodeId}:${port}`);
    if (v) return v;
    ctx.warnings.push(`${joinPath(prefix, nodeId)}: no value for port "${port}".`);
    return "None";
  };
  const inputVar = (nodeId: string, port: string): string => {
    const from = producers.get(`${nodeId}:${port}`);
    if (!from) {
      ctx.warnings.push(`${joinPath(prefix, nodeId)}: input "${port}" is not connected.`);
      return "None";
    }
    const f = splitEndpoint(from);
    return valueOf(f.node, f.port);
  };

  for (const node of order(graph)) {
    const path = joinPath(prefix, node.id);
    const def = ctx.catalog[node.type];
    if (!def) {
      ctx.warnings.push(`${path}: unknown block type "${node.type}".`);
      continue;
    }
    const r = resolveNodeParams(def, node.params, ctx.symbols);
    const attr = pyName(node.id);
    const out = (port: string): string => pyName(`${node.id}_${port}`);

    let ports: Ports;
    try {
      ports = isContainer(def) ? { in: {}, out: {} } : portsOf(def.ports, r);
    } catch {
      ports = { in: {}, out: {} };
    }

    switch (node.type) {
      case "input": {
        vars.set(`${node.id}:x`, "ids");
        break;
      }
      case "output": {
        outputs.x = inputVar(node.id, "x");
        break;
      }
      case "boundary_in": {
        for (const port of Object.keys((r.p.ports as Record<string, string>) ?? {})) {
          vars.set(`${node.id}:${port}`, inputs[port] ?? "None");
        }
        break;
      }
      case "boundary_out": {
        for (const port of Object.keys((r.p.ports as Record<string, string>) ?? {})) {
          outputs[port] = inputVar(node.id, port);
        }
        break;
      }
      case "embedding": {
        init.push(`self.${attr} = nn.Embedding(${r.p.vocab}, ${r.p.dim})`);
        forward.push(`${out("y")} = self.${attr}(${inputVar(node.id, "ids")})`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "pos_embedding": {
        init.push(`self.${attr} = nn.Embedding(${r.p.max_seq}, ${r.p.dim})`);
        const src = inputVar(node.id, "x");
        forward.push(
          `${out("y")} = ${src} + self.${attr}(torch.arange(${src}.shape[1], device=${src}.device))`,
        );
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "conv2d": {
        const groups = Number(r.p.groups) > 1 ? `, groups=${r.p.groups}` : "";
        init.push(
          `self.${attr} = nn.Conv2d(${r.p.in_channels}, ${r.p.out_channels}, ` +
            `kernel_size=${r.p.kernel}, stride=${r.p.stride}, padding=${r.p.padding}` +
            `${groups}, bias=${pyBool(r.p.bias)})`,
        );
        const src = inputVar(node.id, "x");
        const act = ACTIVATION_CALL[String(r.p.act ?? "identity")] ?? ACTIVATION_CALL.identity;
        forward.push(`${out("y")} = ${act(`self.${attr}(${src})`)}`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "maxpool2d": {
        init.push(
          `self.${attr} = nn.MaxPool2d(kernel_size=${r.p.kernel}, stride=${r.p.stride}, padding=${r.p.padding})`,
        );
        forward.push(`${out("y")} = self.${attr}(${inputVar(node.id, "x")})`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "flatten2d": {
        forward.push(`${out("y")} = torch.flatten(${inputVar(node.id, "x")}, 1)`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "learned_tokens": {
        // A parameter, not a module: registered so it trains and moves with the
        // model, expanded to the batch at use. `expand` rather than `repeat`
        // keeps it a view, which is what the reference relies on.
        init.push(
          `self.${attr} = nn.Parameter(torch.zeros(1, ${r.p.count}, ${r.p.dim}))`,
        );
        // Expanded against the model's own input, which is the only tensor a
        // block with no inputs can learn the batch size from. `ids` is the
        // forward's parameter, whatever the design feeds it.
        forward.push(
          `${out("y")} = self.${attr}.expand(ids.shape[0], ${r.p.tokens ? r.p.tokens : -1}, -1)`,
        );
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "linear": {
        init.push(`self.${attr} = nn.Linear(${r.p.in_features}, ${r.p.out_features}, bias=${pyBool(r.p.bias)})`);
        forward.push(`${out("y")} = self.${attr}(${inputVar(node.id, "x")})`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "lm_head": {
        init.push(`self.${attr} = nn.Linear(${r.p.dim}, ${r.p.vocab}, bias=${pyBool(r.p.bias)})`);
        forward.push(`${out("y")} = self.${attr}(${inputVar(node.id, "x")})`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "rmsnorm": {
        init.push(`self.${attr} = nn.RMSNorm(${r.p.dim}, eps=${r.p.eps})`);
        forward.push(`${out("y")} = self.${attr}(${inputVar(node.id, "x")})`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "layernorm": {
        init.push(`self.${attr} = nn.LayerNorm(${r.p.dim}, eps=${r.p.eps}, bias=${pyBool(r.p.bias)})`);
        forward.push(`${out("y")} = self.${attr}(${inputVar(node.id, "x")})`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "activation": {
        const fn = ACTIVATION_CALL[String(r.p.kind)] ?? ACTIVATION_CALL.silu;
        forward.push(`${out("y")} = ${fn(inputVar(node.id, "x"))}`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "add": {
        forward.push(`${out("y")} = ${inputVar(node.id, "a")} + ${inputVar(node.id, "b")}`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "mul": {
        forward.push(`${out("y")} = ${inputVar(node.id, "a")} * ${inputVar(node.id, "b")}`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "rearrange": {
        forward.push(...emitRearrange(r, ctx, ctx.symbols, inputVar(node.id, "x"), out("y"), path));
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "rope": {
        ctx.needsRope = true;
        init.push(`self.${attr} = RotaryEmbedding(${r.p.head_dim}, theta=${r.p.theta})`);
        forward.push(`${out("y")} = self.${attr}(${inputVar(node.id, "x")})`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "conv1d": {
        const src = inputVar(node.id, "x");
        init.push(
          `self.${attr} = nn.Conv1d(${r.p.channels}, ${r.p.channels}, ${r.p.kernel}, ` +
            `groups=${r.p.channels}, padding=${r.p.kernel - 1}, bias=${pyBool(r.p.bias)})`,
        );
        // Convolve over time, then drop the padding the causal kernel added.
        forward.push(
          `${out("y")} = self.${attr}(${src}.transpose(1, 2))[..., : ${src}.shape[1]].transpose(1, 2)`,
        );
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "ssd_scan": {
        ctx.needsSsd = true;
        init.push(
          `self.${attr} = SSDScan(${r.p.heads}, ${r.p.head_dim}, ${r.p.state}, ${r.p.groups})`,
        );
        forward.push(
          `${out("y")} = self.${attr}(${inputVar(node.id, "xbc")}, ${inputVar(node.id, "dt")})`,
        );
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "kv_latent_cache": {
        // Analysis only: it marks what an inference engine would cache. In a
        // forward pass the compressed vector simply flows through.
        forward.push(`${out("y")} = ${inputVar(node.id, "x")}  # cached latent`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "split": {
        // Sizes are expressions over the design symbols, not plain numbers.
        const splitCtx = evalCtxFor(r, ctx.symbols);
        const sizes = ((r.p.sizes as (string | number)[]) ?? []).map((v) => {
          if (typeof v === "number") return v;
          try {
            return evalExpr(String(v), splitCtx).toNumber(splitCtx.values) ?? Number.NaN;
          } catch {
            return Number.NaN;
          }
        });
        if (sizes.some((n) => !Number.isFinite(n))) {
          ctx.warnings.push(`${path}: could not evaluate the split widths.`);
        }
        const names = sizes.map((_, i) => out(`y${i}`));
        forward.push(
          `${names.join(", ")} = torch.split(${inputVar(node.id, "x")}, [${sizes.join(", ")}], dim=-1)`,
        );
        sizes.forEach((_, i) => vars.set(`${node.id}:y${i}`, names[i]));
        break;
      }
      case "concat": {
        const sizes = ((r.p.sizes as (string | number)[]) ?? []).map((_, i) => inputVar(node.id, `y${i}`));
        const axis = typeof r.p.axis === "number" ? r.p.axis : -1;
        forward.push(`${out("y")} = torch.cat([${sizes.join(", ")}], dim=${axis})`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "expand_heads": {
        const src = inputVar(node.id, "x");
        forward.push(`${out("y")} = ${src}.unsqueeze(1).expand(-1, ${r.p.heads}, -1, -1)`);
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      case "sdpa": {
        const q = inputVar(node.id, "q");
        const k = inputVar(node.id, "k");
        const v = inputVar(node.id, "v");
        const gqa = r.p.heads !== r.p.kv_heads ? ", enable_gqa=True" : "";
        // A narrower value head means the kernel cannot infer the scale from
        // the query width, so it is passed explicitly.
        const scale =
          r.p.v_head_dim && r.p.v_head_dim !== r.p.head_dim
            ? `, scale=${(1 / Math.sqrt(r.p.head_dim)).toPrecision(12)}`
            : "";
        if (typeof r.p.window === "number" && r.p.window > 0) {
          ctx.needsWindow = true;
          forward.push(
            `${out("y")}_mask = sliding_window_mask(${q}.shape[-2], ${r.p.window}, ${q}.device, ${q}.dtype)`,
          );
          forward.push(
            `${out("y")} = F.scaled_dot_product_attention(${q}, ${k}, ${v}, attn_mask=${out("y")}_mask${gqa}${scale})`,
          );
        } else {
          forward.push(
            `${out("y")} = F.scaled_dot_product_attention(${q}, ${k}, ${v}, is_causal=${pyBool(r.p.causal)}${gqa}${scale})`,
          );
        }
        vars.set(`${node.id}:y`, out("y"));
        break;
      }
      default: {
        if (isContainer(def) && node.graph) {
          const inner = emitClassForGraph(node.graph, path, "Layer", ctx);
          init.push(
            `self.${attr} = nn.ModuleList([${inner.name}() for _ in range(${r.p.count})])`,
          );
          const portNames = inner.inputs;
          const seed = portNames.map((p) => inputVar(node.id, p));
          const loopVars = portNames.map((p) => out(p));
          forward.push(`${loopVars.join(", ")} = ${seed.join(", ")}`);
          forward.push(`for layer in self.${attr}:`);
          forward.push(`    ${loopVars.join(", ")} = layer(${loopVars.join(", ")})`);
          for (const p of inner.outputs) vars.set(`${node.id}:${p}`, out(p));
          break;
        }
        if (isComposite(def) && def.type === "moe_layer") {
          const inner = emitMoeClass(r, path, ctx);
          init.push(`self.${attr} = ${inner.name}()`);
          forward.push(`${out("y")} = self.${attr}(${inputVar(node.id, "x")})`);
          vars.set(`${node.id}:y`, out("y"));
          break;
        }
        if (isComposite(def)) {
          let expansion: Graph;
          try {
            expansion = def.expand(r.rawFull, r);
          } catch (e) {
            ctx.warnings.push(`${path}: expansion failed: ${(e as Error).message}`);
            break;
          }
          const inner = emitClassForGraph(expansion, path, def.type, ctx, r);
          init.push(`self.${attr} = ${inner.name}()`);
          const args = inner.inputs.map((p) => inputVar(node.id, p));
          const rets = inner.outputs.map((p) => out(p));
          forward.push(`${rets.join(", ")} = self.${attr}(${args.join(", ")})`);
          for (const p of inner.outputs) vars.set(`${node.id}:${p}`, out(p));
          break;
        }
        ctx.warnings.push(`${path}: no code generator for block type "${node.type}".`);
        for (const port of Object.keys(ports.out)) vars.set(`${node.id}:${port}`, "None");
      }
    }
  }

  return { init, forward, outputs };
}

/**
 * Mixture-of-experts dispatch.
 *
 * This is the one block whose runtime behaviour is not a dataflow graph of our
 * primitives: routing sends different tokens to different modules. The expert
 * body still comes from the generic emitter, so only the dispatch is bespoke.
 */
function emitMoeClass(r: Resolved, path: string, ctx: Ctx): ClassInfo {
  const raw = r.rawFull;
  const D = ex(raw.d_model);
  const Fe = ex(raw.expert_hidden);
  const shared = typeof r.p.shared_experts === "number" ? r.p.shared_experts : 0;

  const mlpDef = CATALOG.gated_mlp;
  const expertParams: Record<string, ParamValue> = {
    d_model: D,
    hidden: Fe,
    act: r.p.act,
    bias: r.p.bias === true,
  };
  const expertResolved = resolveNodeParams(mlpDef, expertParams, ctx.symbols);
  const expertGraph = (mlpDef as { expand: (raw: Record<string, ParamValue>, r: Resolved) => Graph }).expand(
    expertResolved.rawFull,
    expertResolved,
  );
  const expert = emitClassForGraph(expertGraph, `${path}/expert`, "Expert", ctx, expertResolved);

  let sharedClass: string | null = null;
  if (shared > 0) {
    const sharedParams: Record<string, ParamValue> = {
      d_model: D,
      hidden: `${shared}*${Fe}`,
      act: r.p.act,
      bias: r.p.bias === true,
    };
    const sharedResolved = resolveNodeParams(mlpDef, sharedParams, ctx.symbols);
    const sharedGraph = (mlpDef as { expand: (raw: Record<string, ParamValue>, r: Resolved) => Graph }).expand(
      sharedResolved.rawFull,
      sharedResolved,
    );
    sharedClass = emitClassForGraph(sharedGraph, `${path}/shared`, "SharedExpert", ctx, sharedResolved).name;
  }

  const key = JSON.stringify({ moe: r.p, expert: expert.name, sharedClass });
  const existing = ctx.byKey.get(key);
  if (existing) return { name: existing, inputs: ["x"], outputs: ["y"] };

  const name = className("MoeLayer", ctx);
  ctx.byKey.set(key, name);

  const lines: string[] = [];
  lines.push(`class ${name}(nn.Module):`);
  lines.push(
    `    """${r.p.experts} experts, top ${r.p.top_k} per token` +
      (shared > 0 ? `, plus ${shared} shared expert(s)` : "") +
      `.` +
      (ctx.moeDispatch === "dense"
        ? ` Dense dispatch: traceable, but ${Math.round(r.p.experts / r.p.top_k)}x the work.`
        : "") +
      `"""`,
  );
  lines.push("");
  lines.push("    def __init__(self):");
  lines.push("        super().__init__()");
  lines.push(`        self.top_k = ${r.p.top_k}`);
  lines.push(`        self.router = nn.Linear(${r.p.d_model}, ${r.p.experts}, bias=${pyBool(r.p.router_bias)})`);
  lines.push(`        self.experts = nn.ModuleList([${expert.name}() for _ in range(${r.p.experts})])`);
  if (sharedClass) lines.push(`        self.shared = ${sharedClass}()`);
  lines.push("");
  lines.push("    def forward(self, x):");
  lines.push("        shape = x.shape");
  lines.push(`        flat = x.reshape(-1, ${r.p.d_model})`);
  lines.push("        scores = F.softmax(self.router(flat).float(), dim=-1)");
  lines.push("        weight, index = torch.topk(scores, self.top_k, dim=-1)");
  if (r.p.normalize !== false) {
    lines.push("        weight = weight / weight.sum(dim=-1, keepdim=True)");
  }
  lines.push("        weight = weight.to(x.dtype)");
  if (ctx.moeDispatch === "dense") {
    // Every expert sees every token, weighted by whether it was chosen. Same
    // result, no data-dependent shapes, so `torch.export` can trace it.
    lines.push(`        gate = torch.zeros(flat.shape[0], ${r.p.experts}, device=x.device, dtype=x.dtype)`);
    lines.push("        gate = gate.scatter(1, index, weight)");
    lines.push("        out = torch.zeros_like(flat)");
    lines.push("        for e, expert in enumerate(self.experts):");
    lines.push("            out = out + expert(flat) * gate[:, e : e + 1]");
    lines.push("        y = out.reshape(shape)");
  } else {
    lines.push("        out = torch.zeros_like(flat)");
    lines.push("        for e, expert in enumerate(self.experts):");
    lines.push("            rows, slot = (index == e).nonzero(as_tuple=True)");
    lines.push("            if rows.numel() == 0:");
    lines.push("                continue");
    lines.push("            out.index_add_(0, rows, expert(flat[rows]) * weight[rows, slot, None])");
    lines.push("        y = out.reshape(shape)");
  }
  if (sharedClass) lines.push("        y = y + self.shared(x)");
  lines.push("        return y");
  lines.push("");

  ctx.classes.push({ name, code: lines.join("\n") });
  return { name, inputs: ["x"], outputs: ["y"] };
}

interface ClassInfo {
  name: string;
  inputs: string[];
  outputs: string[];
}

/** Generate (or reuse) a class for a graph, returning its name and signature. */
function emitClassForGraph(
  graph: Graph,
  path: string,
  baseName: string,
  ctx: Ctx,
  resolved?: Resolved,
): ClassInfo {
  const bIn = graph.nodes.find((n) => n.type === "boundary_in");
  const bOut = graph.nodes.find((n) => n.type === "boundary_out");
  const inputs = Object.keys((bIn?.params?.ports as Record<string, string>) ?? { x: "" });
  const outputs = Object.keys((bOut?.params?.ports as Record<string, string>) ?? { y: "" });

  // Two blocks of the same type with the same numbers share a class.
  const key = JSON.stringify({ baseName, params: resolved ? resolved.p : null, graph });
  const existing = ctx.byKey.get(key);
  if (existing) return { name: existing, inputs, outputs };

  const name = className(baseName, ctx);
  ctx.byKey.set(key, name);

  const argMap: Record<string, string> = {};
  for (const p of inputs) argMap[p] = pyName(p);

  const body = emitGraph(graph, path, argMap, ctx);
  const returned = outputs.map((p) => body.outputs[p] ?? "None");

  const lines: string[] = [];
  lines.push(`class ${name}(nn.Module):`);
  if (resolved) {
    const summary = Object.entries(resolved.p)
      .filter(([, v]) => typeof v === "number" || typeof v === "boolean" || typeof v === "string")
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");
    if (summary) lines.push(`    """${baseName}: ${summary}"""`);
  }
  lines.push("");
  lines.push("    def __init__(self):");
  lines.push("        super().__init__()");
  for (const l of body.init) lines.push(`        ${l}`);
  if (body.init.length === 0) lines.push("        pass");
  lines.push("");
  lines.push(`    def forward(self, ${inputs.map((p) => pyName(p)).join(", ")}):`);
  for (const l of body.forward) lines.push(`        ${l}`);
  lines.push(`        return ${returned.join(", ")}`);
  lines.push("");

  ctx.classes.push({ name, code: lines.join("\n") });
  return { name, inputs, outputs };
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/** Total stacked layers, used to scale the residual initialization by depth. */
function totalLayers(flat: ReturnType<typeof flatten>): number {
  let n = 0;
  for (const rep of flat.repeats) {
    if (rep.type === "repeat") n += rep.count;
  }
  return n || 1;
}

export function generateTorch(doc: Doc, options: TorchOptions = {}): GeneratedCode {
  const symbols = resolveSymbols(doc);
  const ctx: Ctx = {
    doc,
    catalog: catalogOf(doc),
    symbols,
    warnings: [...symbols.errors],
    classes: [],
    byKey: new Map(),
    usedNames: new Set(),
    needsRope: false,
    needsWindow: false,
    needsSsd: false,
    moeDispatch: options.moeDispatch ?? "sparse",
  };

  const modelName = options.className ?? className(doc.meta.name || "Model", ctx);

  const body = emitGraph(doc.graph, "", {}, ctx);

  // Weight tying, which the graph expresses as a flag rather than an edge.
  const tie: string[] = [];
  const head = doc.graph.nodes.find((n) => n.type === "lm_head");
  const embed = doc.graph.nodes.find((n) => n.type === "embedding");
  if (head && embed) {
    const r = resolveNodeParams(ctx.catalog[head.type], head.params, symbols);
    if (r.p.tied === true) {
      tie.push(`self.${pyName(head.id)}.weight = self.${pyName(embed.id)}.weight`);
    }
  }

  const flatResult = flatten(doc, symbols);
  const params = countParams(doc, symbols, flatResult);

  const header: string[] = [
    `"""${doc.meta.name} — generated by TensorCAD.`,
    "",
    "Do not edit by hand: change the design and regenerate.",
    "",
    "Symbols:",
    ...symbols.order
      .filter((n) => !symbols.runtime.has(n))
      .map((n) => `    ${n} = ${symbols.values[n]}`),
    "",
    `Parameters: ${params.total.toLocaleString("en-US")}`,
    '"""',
    "",
    "import torch",
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "",
    "",
  ];

  const helpers: string[] = [];
  if (ctx.needsRope) helpers.push(HELPER_ROPE, "");
  if (ctx.needsWindow) helpers.push(HELPER_WINDOW, "");
  if (ctx.needsSsd) helpers.push(HELPER_SSD, "");

  const modelLines: string[] = [];
  modelLines.push(`class ${modelName}(nn.Module):`);
  modelLines.push("");
  modelLines.push("    def __init__(self):");
  modelLines.push("        super().__init__()");
  for (const l of body.init) modelLines.push(`        ${l}`);
  for (const l of tie) modelLines.push(`        ${l}`);
  modelLines.push("");
  modelLines.push("    def forward(self, ids):");
  for (const l of body.forward) modelLines.push(`        ${l}`);
  modelLines.push(`        return ${body.outputs.x ?? "None"}`);
  modelLines.push("");

  // Weight initialization. Residual projections are scaled down by the depth so
  // the residual stream does not grow as layers are added, which is what GPT-2
  // does and what every model since has kept.
  const initStd = options.initStd ?? 0.02;
  const initLines: string[] = [];
  if (initStd > 0) {
    const layers = totalLayers(flatResult);
    const residualStd = initStd / Math.sqrt(2 * Math.max(1, layers));
    modelLines.splice(modelLines.indexOf("    def forward(self, ids):"), 0,
      ...[
        "    @torch.no_grad()",
        "    def init_weights(self):",
        `        """Normal(0, ${initStd}), with residual projections scaled by depth.`,
        "",
        "        PyTorch's defaults leave nn.Embedding at a unit normal, which starts",
        `        a language model near a cross-entropy of a few hundred rather than`,
        "        ln(vocab). Call this after moving the model to its device.",
        '        """',
        "        for module in self.modules():",
        "            if isinstance(module, (nn.Linear, nn.Embedding)):",
        `                nn.init.normal_(module.weight, mean=0.0, std=${initStd})`,
        '                bias = getattr(module, "bias", None)',
        "                if bias is not None:",
        "                    nn.init.zeros_(bias)",
        "        for name, param in self.named_parameters():",
        `            if name.endswith(RESIDUAL_PROJECTIONS):`,
        `                nn.init.normal_(param, mean=0.0, std=${residualStd.toPrecision(8)})`,
        "        return self",
        "",
      ]);
    initLines.push(
      "",
      "# Projections that write back into the residual stream. Their initial scale",
      "# is divided by sqrt(2 * layers) so depth does not inflate the stream.",
      'RESIDUAL_PROJECTIONS = ("o_proj.weight", "down.weight", "out_proj.weight")',
      "",
    );
  }

  const smoke: string[] = [];
  if (options.includeSmokeTest !== false) {
    smoke.push(
      "",
      'if __name__ == "__main__":',
      `    expected = ${params.total}`,
      `    with torch.device("meta"):`,
      `        model = ${modelName}()`,
      "    actual = sum(p.numel() for p in model.parameters())",
      `    print(f"parameters: {actual:,} (design says {expected:,})")`,
      "    assert actual == expected, f\"parameter count differs: {actual} vs {expected}\"",
      "",
    );
  }

  const contents = [
    header.join("\n"),
    initLines.join("\n"),
    helpers.join("\n"),
    ctx.classes.map((c) => c.code).join("\n"),
    modelLines.join("\n"),
    smoke.join("\n"),
  ]
    .filter((s) => s.trim().length > 0)
    .join("\n");

  return {
    files: [
      { path: "model.py", contents: contents.endsWith("\n") ? contents : `${contents}\n` },
      { path: "design.tensorcad.json", contents: `${JSON.stringify(doc, null, 2)}\n` },
    ],
    warnings: ctx.warnings,
  };
}
