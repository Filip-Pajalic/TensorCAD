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
  analyze,
  evalExpr,
  getPreset,
  inferShapes,
  instantiate,
  matchPattern,
  parsePattern,
  resolveSymbols,
  CATALOG,
  resolveNodeParams,
  portsOf,
  shapeToString,
  type AnalysisOptions,
  type InferResult,
} from "@tensorcad/core";

const root = join(import.meta.dir, "..", "packages", "core-go", "testdata");
const docsDir = join(root, "presets");
const goldenDir = join(root, "golden");
const analysisDir = join(root, "analysis");
mkdirSync(docsDir, { recursive: true });
mkdirSync(goldenDir, { recursive: true });
mkdirSync(analysisDir, { recursive: true });

/** JSON has no NaN, and a NaN here would mean the preset itself is broken. */
function finite(label: string, values: Record<string, number>): Record<string, number> {
  for (const [k, v] of Object.entries(values)) {
    if (!Number.isFinite(v)) throw new Error(`${label}: symbol ${k} is ${v}`);
  }
  return values;
}

const index: string[] = [];

/** Sorts entries by key, because neither engine's map order is meaningful. */
function byKey(entries: [string, string][]): [string, string][] {
  return entries.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
}

/**
 * What shape inference decided, written down so the Go walk has to agree.
 *
 * Shapes are compared as text: a shape is a polynomial, and its printed form is
 * what the editor puts on the wire, so agreeing on the string is agreeing on
 * both the value and how it reads. `... D` arriving as `B T 4096` rather than
 * `B T D` would be a correct number and a wrong answer.
 *
 * The ports map is deliberately absent: it is already pinned block by block in
 * primitives.json, and a container's derived ports show up here anyway, as the
 * shapes of everything downstream of them.
 */
function inferGolden(res: InferResult) {
  return {
    outputs: byKey([...res.outputs].map(([k, v]) => [k, shapeToString(v)])),
    inputs: byKey([...res.inputs].map(([k, v]) => [k, shapeToString(v)])),
    producerOf: byKey([...res.producerOf]),
    // Issues sorted, not in report order: the TypeScript visits a block's ports
    // in declaration order and a Go map has none, so the set is the contract
    // and the sequence is not.
    issues: res.issues
      .map((i) => ({
        path: i.path,
        port: i.port,
        message: i.message,
        severity: i.severity,
        rule: i.rule,
        param: i.param,
      }))
      .sort((a, b) => (JSON.stringify(a) < JSON.stringify(b) ? -1 : 1)),
  };
}

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
    // Both modes, because both are used: the rule engine and the canvas walk
    // the graph as written, and the analysis walks it expanded.
    infer: inferGolden(inferShapes(doc, table)),
    inferExpanded: inferGolden(inferShapes(doc, table, { expandComposites: true })),
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
  `wrote ${index.length} presets, their golden symbol tables and inferred shapes, ` +
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
    extraActivationBytes?: (r: unknown, c: unknown) => number;
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
    extraActivationBytes: prim.extraActivationBytes ? prim.extraActivationBytes(r, primCtx) : null,
    stateBytes: prim.stateBytes ? prim.stateBytes(r, primCtx) : null,
    constraints: prim.constraints ? prim.constraints(r).map((c) => `${c.id}: ${c.message}`) : [],
  };
});
writeFileSync(join(root, "primitives.json"), JSON.stringify({ ctx: primCtx, cases: primitives }, null, 2) + "\n");
console.log(`  and ${primitives.length} primitive cases`);

/**
 * Composite expansions, pinned node for node.
 *
 * A composite is nothing but the subgraph it stands for, so that subgraph *is*
 * the specification. Comparing parameter totals would let a wrong expansion
 * pass whenever two wrong numbers happened to cancel; comparing the graph
 * cannot. Each case records every node's id, type and parameters, and every
 * edge, in order.
 */
const COMPOSITE_CASES: { type: string; params: Record<string, unknown> }[] = [
  { type: "gqa_attention", params: { d_model: "D", heads: "H", kv_heads: "Hkv", head_dim: "dh" } },
  { type: "gqa_attention", params: { d_model: "D", heads: "H", kv_heads: "Hkv", head_dim: "dh", bias: true, o_bias: false } },
  { type: "gqa_attention", params: { d_model: "D", heads: "H", kv_heads: "Hkv", head_dim: "dh", qk_norm: true, rope: { theta: 500000 } } },
  { type: "gqa_attention", params: { d_model: "D", heads: "H", kv_heads: "Hkv", head_dim: "dh", window: 4096, causal: false } },
  { type: "gated_mlp", params: { d_model: "D", hidden: "F", act: "silu" } },
  { type: "gated_mlp", params: { d_model: "D", hidden: "F", act: "gelu", bias: true } },
  { type: "dense_mlp", params: { d_model: "D", hidden: "4*D", act: "gelu", bias: true } },
  { type: "mla_attention", params: { d_model: 7168, heads: 128, q_lora: 1536, kv_lora: 512, nope_dim: 128, rope_dim: 64, v_dim: 128, rope: { theta: 10000 } } },
  { type: "moe_layer", params: { d_model: "D", experts: 8, top_k: 2, expert_hidden: "F" } },
  { type: "moe_layer", params: { d_model: "D", experts: 256, top_k: 8, expert_hidden: 2048, shared_experts: 1, router_bias: true } },
  { type: "mamba2_block", params: { d_model: "D", expand: 2, head_dim: 64, state: 128, groups: 8, conv_kernel: 4 } },
  { type: "transformer_block", params: { d_model: "D", heads: "H", kv_heads: "Hkv", head_dim: "dh", ffn_hidden: "F" } },
  { type: "transformer_block", params: { d_model: "D", heads: "H", kv_heads: "Hkv", head_dim: "dh", ffn_hidden: "F", norm: "layernorm", norm_bias: true, mlp: "dense", act: "gelu", attn_bias: true, mlp_bias: true, rope: null } },
  { type: "transformer_block", params: { d_model: "D", heads: "H", kv_heads: "Hkv", head_dim: "dh", ffn_hidden: "F", post_norm: true } },
  { type: "transformer_block", params: { d_model: "D", heads: "H", kv_heads: "Hkv", head_dim: "dh", ffn_hidden: "F", mlp: "moe", experts: 8, top_k: 2, expert_hidden: 2048 } },
  { type: "transformer_block", params: { d_model: 7168, heads: 128, kv_heads: 128, head_dim: 128, ffn_hidden: 18432, attention: "mla", q_lora: 1536, kv_lora: 512, nope_dim: 128, rope_dim: 64, v_dim: 128 } },
];

const flatNode = (n: Record<string, unknown>): unknown => ({
  id: n.id,
  type: n.type,
  params: n.params ?? null,
  graph: n.graph
    ? {
        nodes: ((n.graph as { nodes: Record<string, unknown>[] }).nodes ?? []).map(flatNode),
        edges: (n.graph as { edges: unknown }).edges ?? [],
      }
    : null,
});

const composites = COMPOSITE_CASES.map(({ type, params }) => {
  const def = CATALOG[type] as never as {
    expand: (raw: unknown, r: unknown) => { nodes: Record<string, unknown>[]; edges: unknown[] };
    constraints?: (r: unknown) => { id: string; message: string }[];
  };
  const r = resolveNodeParams(CATALOG[type], params as never, primSymbols);
  const g = def.expand({ ...(r as never as { rawFull: Record<string, unknown> }).rawFull, ...params }, r);
  return {
    type,
    params,
    nodes: g.nodes.map(flatNode),
    edges: g.edges,
    constraints: def.constraints ? def.constraints(r).map((c) => `${c.id}: ${c.message}`) : [],
  };
});
writeFileSync(join(root, "composites.json"), JSON.stringify({ cases: composites }, null, 2) + "\n");
console.log(`  and ${composites.length} composite expansions`);

/**
 * The analysis, at three operating points per preset.
 *
 * One would not be enough: the default point never exercises sharding, full
 * recomputation or an eager attention kernel, and those are three of the places
 * the arithmetic is easiest to get subtly wrong. Each case records the resolved
 * options as well as the answers, so if the Go test's own copy of a variant
 * ever drifts from this one, the resolved options disagree and say so in the
 * first line of the failure rather than as a wrong number somewhere downstream.
 */
const ANALYSIS_VARIANTS: { label: string; options: AnalysisOptions }[] = [
  { label: "default", options: {} },
  {
    label: "sharded",
    options: {
      dtype: "fp8",
      inferenceDtype: "fp8",
      recompute: "full",
      optimizer: "adamw8bit",
      parallel: { tp: 8, pp: 2, dp: 4, zero: 3, sequenceParallel: true },
      gpus: 64,
      concurrency: 32,
      tokens: 15e12,
      mfu: 0.4,
    },
  },
  {
    label: "eager",
    options: { T: 8192, B: 4, flash: false, recompute: "selective", kvDtype: "fp8", gpus: 8 },
  },
];

/** A map as a sorted list of pairs, so neither engine's key order matters. */
function pairs(m: Record<string, number>): [string, number][] {
  return Object.entries(m).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
}

let analysisCases = 0;
for (const name of PRESET_NAMES) {
  const doc = getPreset(name);
  const cases = ANALYSIS_VARIANTS.map(({ label, options }) => {
    const a = analyze(doc, options);
    analysisCases++;
    return {
      label,
      options: {
        T: a.options.T,
        B: a.options.B,
        dtype: a.options.dtype,
        inferenceDtype: a.options.inferenceDtype,
        kvDtype: a.options.kvDtype,
        hardware: a.options.hardware.id,
        gpus: a.options.gpus,
        parallel: a.options.parallel,
        optimizer: a.options.optimizer,
        recompute: a.options.recompute,
        flash: a.options.flash,
        tokens: a.options.tokens,
        tokensWereDefaulted: a.options.tokensWereDefaulted,
        mfu: a.options.mfu,
        decodeEfficiency: a.options.decodeEfficiency,
        concurrency: a.options.concurrency,
      },
      params: {
        total: a.params.total,
        active: a.params.active,
        embedding: a.params.embedding,
        head: a.params.head,
        nonEmbedding: a.params.nonEmbedding,
        nonEmbeddingActive: a.params.nonEmbeddingActive,
        byPath: pairs(a.params.byPath),
        byCategory: pairs(a.params.byCategory),
        byType: pairs(a.params.byType),
      },
      flops: {
        fwdDense: a.flops.fwdDense,
        fwdAttention: a.flops.fwdAttention,
        fwdAttentionUnmasked: a.flops.fwdAttentionUnmasked,
        fwdTotal: a.flops.fwdTotal,
        fwdTotalUnmasked: a.flops.fwdTotalUnmasked,
        elementwise: a.flops.elementwise,
        trainPerToken: a.flops.trainPerToken,
        attentionShare: a.flops.attentionShare,
        ruleOfThumb2N: a.flops.ruleOfThumb2N,
        ruleOfThumb6N: a.flops.ruleOfThumb6N,
        byPath: pairs(a.flops.byPath),
        byCategory: pairs(a.flops.byCategory),
      },
      kv: {
        bytesPerToken: a.kv.bytesPerToken,
        bytesPerSequenceFixed: a.kv.bytesPerSequenceFixed,
        byPath: pairs(a.kv.byPath),
      },
      memory: {
        weightsBytes: a.memory.weightsBytes,
        train: {
          weights: a.memory.train.weights,
          grads: a.memory.train.grads,
          optimizer: a.memory.train.optimizer,
          activations: a.memory.train.activations,
          logits: a.memory.train.logits,
          total: a.memory.train.total,
          perGpu: a.memory.train.perGpu,
          activationsByPath: pairs(a.memory.train.activationsByPath),
        },
        infer: a.memory.infer,
        optimizerLabel: a.memory.optimizerLabel,
        notes: a.memory.notes,
      },
      throughput: {
        ridgePoint: a.throughput.ridgePoint,
        decodeBytesPerStep: a.throughput.decodeBytesPerStep,
        decodeFlopsPerStep: a.throughput.decodeFlopsPerStep,
        decodeSecondsPerStep: a.throughput.decodeSecondsPerStep,
        decodeTokensPerSecond: a.throughput.decodeTokensPerSecond,
        memoryBound: a.throughput.memoryBound,
        prefillSeconds: a.throughput.prefillSeconds,
        notes: a.throughput.notes,
      },
      cost: a.cost,
      chinchilla: {
        optimalTokens: a.chinchilla.optimalTokens,
        tokensPerParam: a.chinchilla.tokensPerParam,
        tokensPerActiveParam: a.chinchilla.tokensPerActiveParam,
        overTrainingRatio: a.chinchilla.overTrainingRatio,
        predictedLoss: pairs(a.chinchilla.predictedLoss),
        verdict: a.chinchilla.verdict,
      },
      errors: a.errors,
      flat: {
        nodes: a.flat.nodes.length,
        blocks: a.flat.blocks.length,
        repeats: a.flat.repeats.map((r) => ({ path: r.path, type: r.type, count: r.count, active: r.active })),
        errors: a.flat.errors,
      },
    };
  });
  writeFileSync(join(analysisDir, `${name}.json`), JSON.stringify({ preset: name, cases }, null, 2) + "\n");
}
console.log(`  and ${analysisCases} analysis cases`);
