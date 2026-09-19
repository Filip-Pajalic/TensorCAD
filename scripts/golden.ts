/**
 * Golden files for the Go port.
 *
 * The Go engine is replacing this one, and "it looks right" is not a migration
 * plan. So the TypeScript writes down, for every preset, the document it builds
 * and the answers it gets; the Go tests read the same documents and have to
 * produce the same answers. A stage of the port is done when its golden file
 * matches, and not before.
 *
 *   bun run scripts/golden.ts
 *
 * Regenerate deliberately: a diff here is either a bug being fixed or a
 * behaviour being changed, and both want to be seen in review.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  PRESET_NAMES,
  Sym,
  evalExpr,
  getPreset,
  instantiate,
  matchPattern,
  parsePattern,
  resolveSymbols,
  CATALOG,
  resolveNodeParams,
  portsOf,
  shapeToString,
} from "@tensorcad/core";

const root = join(import.meta.dir, "..", "packages", "core-go", "testdata");
const docsDir = join(root, "presets");
const goldenDir = join(root, "golden");
mkdirSync(docsDir, { recursive: true });
mkdirSync(goldenDir, { recursive: true });

/** JSON has no NaN, and a NaN here would mean the preset itself is broken. */
function finite(label: string, values: Record<string, number>): Record<string, number> {
  for (const [k, v] of Object.entries(values)) {
    if (!Number.isFinite(v)) throw new Error(`${label}: symbol ${k} is ${v}`);
  }
  return values;
}

const index: string[] = [];

for (const name of PRESET_NAMES) {
  const doc = getPreset(name);
  writeFileSync(join(docsDir, `${name}.json`), JSON.stringify(doc, null, 2) + "\n");

  const table = resolveSymbols(doc);
  const golden = {
    preset: name,
    symbols: {
      order: table.order,
      values: finite(name, table.values),
      designValues: finite(name, table.designValues),
      runtime: [...table.runtime].sort(),
      docs: table.docs,
      errors: table.errors,
    },
  };
  writeFileSync(join(goldenDir, `${name}.json`), JSON.stringify(golden, null, 2) + "\n");
  index.push(name);
}

writeFileSync(join(root, "presets.json"), JSON.stringify(index, null, 2) + "\n");
/**
 * The expression algebra, pinned separately.
 *
 * A symbol table only exercises the paths its presets happen to take. These are
 * the corners: rational coefficients that do not divide, exact monomial
 * division, eager folding of functions, and the printed form of a polynomial.
 * The printed form is what a shape label on the canvas says, so a divergence
 * there is visible rather than theoretical.
 */
const ENV: Record<string, number> = {
  B: 4,
  T: 2048,
  D: 4096,
  H: 32,
  Hkv: 8,
  dh: 128,
  F: 14336,
  V: 128256,
  L: 32,
};
const KNOWN = new Set(Object.keys(ENV));

const CASES = [
  "D",
  "-D",
  "D + 1",
  "4*D",
  "D*4",
  "H*dh",
  "B*T*D",
  "D/H",
  "D/4",
  "D/3",
  "1/3*D",
  "1.3*8/3*D",
  "ceil_mult(1.3*8/3*D, 1024)",
  "(H + 2*Hkv)*dh",
  "H*dh - D",
  "D^2",
  "2^10",
  "(D + D)/2",
  "D*dh/dh",
  "B*T*(H*dh)/H",
  "floor(D/H)",
  "ceil(D/3)",
  "round(D/3)",
  "min(D, F)",
  "max(D, F)",
  "floor_mult(F, 1024)",
  "round_mult(D*2/3, 128)",
  "log2(D)",
  "sqrt(D)",
  "abs(0 - D)",
  "1_000 + D",
  "1e3 + D",
  "0.5*D",
  "D - D",
  "3*D - 2*D",
];

const ERRORS = ["D +", "D $ 1", "nope", "floor(D, 2)", "notafunction(D)", "D^(0 - 1)", "D/0", "B/T"];

const expressions = {
  env: ENV,
  ok: CASES.map((src) => {
    const sym = evalExpr(src, { values: ENV, known: KNOWN });
    return { src, text: sym.toString(), value: sym.toNumber(ENV), symbols: sym.symbols() };
  }),
  errors: ERRORS.map((src) => {
    try {
      const sym = evalExpr(src, { values: ENV, known: KNOWN });
      return { src, message: "", text: sym.toString() };
    } catch (e) {
      return { src, message: (e as Error).message, text: "" };
    }
  }),
};
writeFileSync(join(root, "expressions.json"), JSON.stringify(expressions, null, 2) + "\n");

/**
 * Shape patterns, pinned the same way.
 *
 * A pattern is where a shape stops being a string and becomes something the
 * engine can disagree with. The cases below are the corners: groups, ellipses
 * that bind nothing, ellipses that bind three dimensions, rank mismatches, and
 * the message a mismatch produces — which is what a person reads off the canvas
 * when a design is wrong, so it has to be identical in both engines.
 */
const PATTERNS = [
  "B T D",
  "B T (H dh)",
  "... D",
  "...",
  "B H T dh",
  "... 2*D",
  "B (H dh) T",
  "B 3 224 224",
  "B T (H dh) D",
];

const PATTERN_ERRORS = ["B T (H dh", "B () D", "... T ... D"];

/** actual shape (as patterns, instantiated), checked against a pattern. */
const MATCHES: [string, string][] = [
  ["B T D", "B T D"],
  ["B T D", "... D"],
  ["B T D", "B T (H dh)"],
  ["B H T dh", "... dh"],
  ["B T D", "B T D D"],
  ["B T D", "B H T dh"],
  ["D", "... D"],
  ["B T F", "... D"],
];

const patternCtx = { values: ENV, known: KNOWN };

const patterns = {
  parse: PATTERNS.map((src) => {
    const p = parsePattern(src);
    const shape = instantiate(p, patternCtx, [Sym.v("B"), Sym.v("T")]);
    return {
      src,
      atoms: p.atoms.map((a) => (a.kind === "ellipsis" ? "..." : a.parts.join(" "))),
      instantiated: shape.shape ? shapeToString(shape.shape) : null,
      errors: shape.errors,
    };
  }),
  parseErrors: PATTERN_ERRORS.map((src) => {
    try {
      parsePattern(src);
      return { src, message: "" };
    } catch (e) {
      return { src, message: (e as Error).message };
    }
  }),
  matches: MATCHES.map(([actualSrc, patternSrc]) => {
    const actual = instantiate(parsePattern(actualSrc), patternCtx, []).shape ?? [];
    const r = matchPattern(actual, parsePattern(patternSrc), patternCtx, ENV);
    return {
      actual: actualSrc,
      pattern: patternSrc,
      ok: r.ok,
      batch: shapeToString(r.batch),
      errors: r.errors,
    };
  }),
};
writeFileSync(join(root, "patterns.json"), JSON.stringify(patterns, null, 2) + "\n");

console.log(
  `wrote ${index.length} presets, their golden symbol tables, ` +
    `${CASES.length + ERRORS.length} expression cases, and ` +
    `${PATTERNS.length + PATTERN_ERRORS.length + MATCHES.length} pattern cases`,
);

/**
 * The primitive formulas, pinned one block at a time.
 *
 * A preset exercises the primitives its architecture happens to use, at the
 * sizes that architecture happens to pick. This walks every primitive in the
 * catalog at a fixed set of parameters and writes down what it says: its
 * resolved parameters, its ports, its parameter count, its FLOPs, what it
 * retains, and its cache state. A port of a formula is done when this matches.
 */
const PRIMITIVE_CASES: { type: string; params: Record<string, unknown> }[] = [
  { type: "input", params: { shape: "B T", dtype: "int64" } },
  { type: "output", params: {} },
  { type: "boundary_in", params: { ports: { x: "B T D" } } },
  { type: "boundary_out", params: { ports: { x: "B T D" } } },
  { type: "embedding", params: { vocab: 128256, dim: 4096 } },
  { type: "pos_embedding", params: { max_seq: 1024, dim: 768 } },
  { type: "learned_tokens", params: { count: 1, dim: 384, tokens: 0 } },
  { type: "learned_tokens", params: { count: 1, dim: 384, tokens: 256 } },
  { type: "linear", params: { in_features: 4096, out_features: 14336, bias: false } },
  { type: "linear", params: { in_features: 4096, out_features: 14336, bias: true } },
  { type: "lm_head", params: { vocab: 128256, dim: 4096, tied: false, bias: false } },
  { type: "lm_head", params: { vocab: 128256, dim: 4096, tied: true, bias: false } },
  { type: "conv2d", params: { in_channels: 3, out_channels: 64, kernel: 11, stride: 4, padding: 2, in_h: 224, in_w: 224, act: "relu" } },
  { type: "conv2d", params: { in_channels: 192, out_channels: 384, kernel: 3, stride: 1, padding: 1, in_h: 13, in_w: 13, act: "relu" } },
  { type: "maxpool2d", params: { channels: 64, kernel: 3, stride: 2, in_h: 55, in_w: 55 } },
  { type: "flatten2d", params: { channels: 256, in_h: 6, in_w: 6 } },
  { type: "rmsnorm", params: { dim: 4096 } },
  { type: "rmsnorm", params: { dim: 4096, scale: false } },
  { type: "layernorm", params: { dim: 768, bias: true } },
  { type: "layernorm", params: { dim: 768, bias: false } },
  { type: "activation", params: { kind: "silu", dim: 14336 } },
  { type: "activation", params: { kind: "gelu", dim: 3072 } },
  { type: "add", params: { dim: 4096 } },
  { type: "mul", params: { dim: 14336 } },
  { type: "rearrange", params: { from: "B T (H dh)", to: "B H T dh" } },
  { type: "rope", params: { heads: 32, head_dim: 128, theta: 500000 } },
  { type: "rope", params: { heads: 32, head_dim: 127, theta: 10000 } },
  { type: "sdpa", params: { heads: 32, kv_heads: 8, head_dim: 128, causal: true } },
  { type: "sdpa", params: { heads: 32, kv_heads: 8, head_dim: 128, causal: false } },
  { type: "sdpa", params: { heads: 32, kv_heads: 8, head_dim: 128, window: 4096 } },
  { type: "sdpa", params: { heads: 32, kv_heads: 7, head_dim: 128 } },
  { type: "sdpa", params: { heads: 32, kv_heads: 8, head_dim: 128, cache: false } },
  { type: "topk_router", params: { d_model: 7168, experts: 256, top_k: 8, bias: true } },
  { type: "topk_router", params: { d_model: 7168, experts: 8, top_k: 9 } },
  { type: "weighted_sum", params: { dim: 4096, n: 8 } },
  { type: "split", params: { from: "B T D", sizes: [512, 64] } },
  { type: "concat", params: { to: "B T D", sizes: [512, 64] } },
  { type: "concat", params: { to: "B T Dp", sizes: ["Tc", "T-Tc"], axis: 1 } },
  { type: "expand_heads", params: { heads: 32, dim: 64 } },
  { type: "kv_latent_cache", params: { dim: 576 } },
  { type: "conv1d", params: { channels: 8192, kernel: 4, bias: true } },
  { type: "ssd_scan", params: { d_inner: 8192, heads: 128, head_dim: 64, state: 128, groups: 8, xbc_width: 10240 } },
  { type: "ssd_scan", params: { d_inner: 8192, heads: 128, head_dim: 65, state: 128, groups: 7, xbc_width: 10240 } },
];

const primCtx = { T: 4096, B: 1, bytes: 2, flash: true };
const primSymbols = resolveSymbols({
  version: 1,
  meta: { name: "primitive-cases" },
  symbols: { ...ENV, Tc: 218, Dp: 384 },
  graph: { nodes: [], edges: [] },
} as never);

const primitives = PRIMITIVE_CASES.map(({ type, params }) => {
  const def = CATALOG[type];
  const r = resolveNodeParams(def, params as never, primSymbols);
  let ports: unknown = null;
  let portErr = "";
  try {
    const p = portsOf(def.ports, r);
    ports = {
      in: Object.fromEntries(Object.entries(p.in).map(([k, v]) => [k, v.shape])),
      out: Object.fromEntries(Object.entries(p.out).map(([k, v]) => [k, v.shape])),
      anchors: Object.fromEntries(
        [...Object.entries(p.in), ...Object.entries(p.out)]
          .filter(([, v]) => v.anchor !== "flow" || v.dtype !== "inherit")
          .map(([k, v]) => [k, `${v.anchor}/${v.dtype}`]),
      ),
    };
  } catch (e) {
    portErr = (e as Error).message;
  }
  const prim = def as never as {
    paramCount?: (r: unknown) => number;
    flops?: (r: unknown, c: unknown) => Record<string, number>;
    retains?: (r: unknown) => string[];
    stateBytes?: (r: unknown, c: unknown) => Record<string, number>;
    constraints?: (r: unknown) => { id: string; message: string }[];
  };
  return {
    type,
    params,
    resolved: r.p,
    errors: r.errors,
    ports,
    portError: portErr,
    paramCount: prim.paramCount ? prim.paramCount(r) : null,
    flops: prim.flops ? prim.flops(r, primCtx) : null,
    retains: prim.retains ? prim.retains(r) : null,
    stateBytes: prim.stateBytes ? prim.stateBytes(r, primCtx) : null,
    constraints: prim.constraints ? prim.constraints(r).map((c) => `${c.id}: ${c.message}`) : [],
  };
});
writeFileSync(join(root, "primitives.json"), JSON.stringify({ ctx: primCtx, cases: primitives }, null, 2) + "\n");
console.log(`  and ${primitives.length} primitive cases`);
