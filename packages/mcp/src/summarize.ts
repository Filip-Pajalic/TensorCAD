import type { CatalogEntry as EngineCatalogEntry } from "@tensor-cad/engine";
import type { AnalysisResult, Doc, Graph, ParamSpec, SymbolTable, ValidationReport } from "@tensor-cad/engine";
import { formatBytes, formatCount, formatFlops, isComposite, isContainer, isPrimitive, joinPath, splitEndpoint } from "@tensor-cad/engine";
import { catalogByCategory, countParams, getBlock, inferShapes, resolveSymbols } from "@tensor-cad/engine/node";
/**
 * Projections of a design that are cheap for a model to read.
 *
 * `outline` is the one an assistant should reach for first: it is the whole
 * shape of the design in a few hundred tokens, where the full document is a few
 * thousand. The others mirror the core's results as plain JSON, because
 * `AnalysisResult` carries `Map`s that do not survive serialization.
 */


// ---------------------------------------------------------------------------
// Outline
// ---------------------------------------------------------------------------

export interface OutlineSymbol {
  name: string;
  kind: "design" | "runtime";
  value: string;
  resolved?: number;
  doc?: string;
}

export interface OutlineBlock {
  path: string;
  type: string;
  kind: string;
  depth: number;
  label?: string;
  /** Trainable parameters under this path, including every repeat instance. */
  params: number;
  /** Instance count for `repeat` containers. */
  repeat?: number;
}

export interface OutlineEdge {
  /** Container path the edge lives in; `""` is the root graph. */
  graph: string;
  from: string;
  to: string;
  /** Inferred shape on the wire, e.g. `"B T D"`. */
  shape?: string;
}

export interface Outline {
  name: string;
  family?: string;
  notes?: string;
  symbols: OutlineSymbol[];
  blocks: OutlineBlock[];
  edges: OutlineEdge[];
  params_total: number;
  params_active: number;
  issues: number;
}

export function outlineOf(doc: Doc): Outline {
  const symbols = resolveSymbols(doc);
  const infer = inferShapes(doc);
  const params = countParams(doc);

  const paramsAt = (path: string): number => {
    let sum = 0;
    for (const [p, v] of Object.entries(params.byPath)) {
      if (p === path || p.startsWith(`${path}/`)) sum += v;
    }
    return sum;
  };

  const blocks: OutlineBlock[] = [];
  const edges: OutlineEdge[] = [];

  const walk = (graph: Graph, prefix: string, depth: number): void => {
    for (const [from, to] of graph.edges) {
      const edge: OutlineEdge = { graph: prefix, from, to };
      const source = splitEndpoint(from);
      const shape = infer.outputs[`${joinPath(prefix, source.node)}:${source.port}`];
      if (shape) edge.shape = shape.symbolic;
      edges.push(edge);
    }

    for (const node of graph.nodes) {
      const path = joinPath(prefix, node.id);
      const def = getBlock(node.type);
      const block: OutlineBlock = {
        path,
        type: node.type,
        kind: def?.kind ?? "unknown",
        depth,
        params: paramsAt(path),
      };
      if (node.label) block.label = node.label;
      if (node.graph) {
        const count = infer.resolved[path]?.p?.count;
        if (typeof count === "number") block.repeat = count;
      }
      blocks.push(block);
      if (node.graph) walk(node.graph, path, depth + 1);
    }
  };
  walk(doc.graph, "", 0);

  const out: Outline = {
    name: doc.meta.name,
    symbols: outlineSymbols(doc, symbols),
    blocks,
    edges,
    params_total: params.total,
    params_active: params.active,
    issues: infer.issues.length,
  };
  if (doc.meta.family) out.family = doc.meta.family;
  if (doc.meta.notes) out.notes = doc.meta.notes;
  return out;
}

function outlineSymbols(doc: Doc, symbols: SymbolTable): OutlineSymbol[] {
  return Object.entries(doc.symbols ?? {}).map(([name, def]) => {
    const runtime = typeof def === "object" && def !== null && def.kind === "runtime";
    const raw =
      typeof def === "object" && def !== null
        ? runtime
          ? String((def as { default: number }).default)
          : String((def as { value: number | string }).value)
        : String(def);
    const s: OutlineSymbol = { name, kind: runtime ? "runtime" : "design", value: raw };
    const resolved = symbols.values[name];
    if (typeof resolved === "number") s.resolved = resolved;
    const docString = typeof def === "object" && def !== null ? def.doc : undefined;
    if (docString) s.doc = docString;
    return s;
  });
}

/** The outline rendered for a human (and as the text mirror of the tool result). */
export function outlineText(o: Outline): string {
  const lines: string[] = [
    `${o.name}${o.family ? ` (${o.family})` : ""}  ${formatCount(o.params_total)} parameters` +
      (o.params_active !== o.params_total ? `, ${formatCount(o.params_active)} active` : ""),
  ];
  if (o.notes) lines.push(o.notes);

  lines.push("", "symbols");
  for (const s of o.symbols) {
    lines.push(`  ${s.name} = ${s.value}${s.kind === "runtime" ? " (runtime)" : ""}${s.doc ? `  ${s.doc}` : ""}`);
  }

  lines.push("", "blocks");
  for (const b of o.blocks) {
    const indent = "  ".repeat(b.depth + 1);
    const repeat = b.repeat !== undefined ? ` x${b.repeat}` : "";
    const size = b.params > 0 ? `  ${formatCount(b.params)}` : "";
    lines.push(`${indent}${b.path.split("/").pop()}  ${b.type}${repeat}${size}`);
  }

  lines.push("", "edges");
  for (const e of o.edges) {
    const where = e.graph ? `${e.graph}/` : "";
    lines.push(`  ${where}${e.from} -> ${where}${e.to}${e.shape ? `   ${e.shape}` : ""}`);
  }
  if (o.issues > 0) lines.push("", `${o.issues} shape issue(s); run tensorcad_validate`);
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// One block
// ---------------------------------------------------------------------------

export interface BlockPort {
  name: string;
  pattern: string;
  shape?: string;
  /** Declared element type, when it is not inherited from the producer. */
  dtype?: string;
  /** True when this port may legitimately be left unwired. */
  optional?: boolean;
  /** The port on the other end of the wire, as a full path. */
  connected_to?: string[];
}

export interface BlockDetail {
  path: string;
  id: string;
  type: string;
  kind: string;
  category: string;
  label?: string;
  summary: string;
  formula?: string;
  params: Record<string, unknown>;
  resolved_params: Record<string, unknown>;
  param_errors: string[];
  inputs: BlockPort[];
  outputs: BlockPort[];
  params_count: number;
  /** Multiplied instances of this block, from the enclosing `repeat`s. */
  instances: number;
  children: string[];
}

export function blockDetail(doc: Doc, path: string): BlockDetail {
  const infer = inferShapes(doc);
  const params = countParams(doc);

  const segments = path.split("/").filter(Boolean);
  if (segments.length === 0) throw new Error(`"${path}" is not a block path.`);

  let graph: Graph = doc.graph;
  let node = undefined as ReturnType<typeof findNode>;
  for (let i = 0; i < segments.length; i++) {
    node = findNode(graph, segments[i]);
    if (!node) {
      const known = graph.nodes.map((n) => n.id).join(", ");
      throw new Error(`No block "${segments.slice(0, i + 1).join("/")}". Blocks here: ${known || "(none)"}.`);
    }
    if (i < segments.length - 1) {
      if (!node.graph) throw new Error(`Block "${segments.slice(0, i + 1).join("/")}" has no subgraph.`);
      graph = node.graph;
    }
  }
  if (!node) throw new Error(`No block "${path}".`);

  const def = getBlock(node.type);
  const ports = infer.ports[path] ?? { in: {}, out: {} };

  const consumers = new Map<string, string[]>();
  for (const [consumer, producer] of Object.entries(infer.producerOf)) {
    const list = consumers.get(producer);
    if (list) list.push(consumer);
    else consumers.set(producer, [consumer]);
  }

  const inputs: BlockPort[] = Object.entries(ports.in).map(([name, spec]) => {
    const port: BlockPort = { name, pattern: spec.shape };
    if (spec.dtype !== "inherit") port.dtype = spec.dtype;
    if (spec.optional) port.optional = true;
    const shape = infer.inputs[`${path}:${name}`];
    if (shape) port.shape = shape.symbolic;
    const producer = infer.producerOf[`${path}:${name}`];
    if (producer) port.connected_to = [producer];
    return port;
  });

  const outputs: BlockPort[] = Object.entries(ports.out).map(([name, spec]) => {
    const port: BlockPort = { name, pattern: spec.shape };
    if (spec.dtype !== "inherit") port.dtype = spec.dtype;
    const shape = infer.outputs[`${path}:${name}`];
    if (shape) port.shape = shape.symbolic;
    const to = consumers.get(`${path}:${name}`);
    if (to) port.connected_to = to;
    return port;
  });

  let paramsCount = 0;
  for (const [p, v] of Object.entries(params.byPath)) {
    if (p === path || p.startsWith(`${path}/`)) paramsCount += v;
  }

  // Whatever the analysis resolved for this path. A block it could not reach
  // has none, and the error below says so rather than this resolving a second
  // set that might disagree with the numbers beside it.
  const resolved = infer.resolved[path];

  const detail: BlockDetail = {
    path,
    id: node.id,
    type: node.type,
    kind: def?.kind ?? "unknown",
    category: def?.category ?? "unknown",
    summary: def?.docs.summary ?? "Unknown block type.",
    params: { ...(node.params ?? {}) },
    resolved_params: resolved ? { ...resolved.p } : {},
    param_errors: resolved ? [] : [`Unknown block type "${node.type}"`],
    inputs,
    outputs,
    params_count: paramsCount,
    instances: instancesOf(infer, segments),
    children: (node.graph?.nodes ?? []).map((n) => joinPath(path, n.id)),
  };
  if (node.label) detail.label = node.label;
  if (def?.docs.formula) detail.formula = def.docs.formula;
  return detail;
}

function findNode(graph: Graph, id: string) {
  return graph.nodes.find((n) => n.id === id);
}

/** How many copies of a block exist, from the `count` of each enclosing repeat. */
function instancesOf(infer: ReturnType<typeof inferShapes>, segments: string[]): number {
  let multiplier = 1;
  for (let i = 0; i < segments.length - 1; i++) {
    const prefix = segments.slice(0, i + 1).join("/");
    const count = infer.resolved[prefix]?.p?.count;
    if (typeof count === "number") multiplier *= count;
  }
  return multiplier;
}

export function blockText(b: BlockDetail): string {
  const lines = [
    `${b.path}  ${b.type} (${b.kind}/${b.category})${b.instances > 1 ? ` x${b.instances}` : ""}`,
    b.summary,
  ];
  if (b.formula) lines.push(`formula: ${b.formula}`);
  lines.push(`parameters: ${formatCount(b.params_count)}`);
  if (Object.keys(b.params).length > 0) {
    lines.push("", "params");
    for (const [k, v] of Object.entries(b.params)) lines.push(`  ${k} = ${JSON.stringify(v)}`);
  }
  if (b.inputs.length > 0) {
    lines.push("", "in");
    for (const p of b.inputs) {
      lines.push(`  ${p.name}: ${p.shape ?? p.pattern}${p.connected_to ? `  <- ${p.connected_to.join(", ")}` : ""}`);
    }
  }
  if (b.outputs.length > 0) {
    lines.push("", "out");
    for (const p of b.outputs) {
      lines.push(`  ${p.name}: ${p.shape ?? p.pattern}${p.connected_to ? `  -> ${p.connected_to.join(", ")}` : ""}`);
    }
  }
  if (b.param_errors.length > 0) lines.push("", ...b.param_errors.map((e) => `! ${e}`));
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

export interface CatalogParam {
  name: string;
  type: string;
  default?: string;
  doc?: string;
  values?: string[];
}

export interface CatalogEntry {
  type: string;
  kind: string;
  category: string;
  /**
   * What a drawing calls this block.
   *
   * The editor has printed it since M7 and an agent could not see it, so the
   * two were describing one design in two vocabularies: the human's screen
   * said "grouped-query attention" and the assistant only ever knew
   * `gqa_attention`. Falls back to the type, so it is always something to say.
   */
  name: string;
  summary: string;
  formula?: string;
  refs: string[];
  params: CatalogParam[];
  inputs: string[];
  outputs: string[];
  /** True when the port list depends on the block's parameters. */
  dynamic_ports: boolean;
}

export function catalogEntry(def: EngineCatalogEntry): CatalogEntry {
  const entry: CatalogEntry = {
    type: def.type,
    kind: def.kind,
    category: def.category,
    name: def.docs.name ?? def.type,
    summary: def.docs.summary ?? "",
    refs: def.docs.refs ?? [],
    // In the order the block declares them, which is the order a reader
    // expects to meet them in.
    params: def.paramOrder.map((name: string) => catalogParam(name, def.params[name])),
    inputs: [],
    outputs: [],
    dynamic_ports: false,
  };
  if (def.docs.formula) entry.formula = def.docs.formula;

  if (isPrimitive(def) || isComposite(def)) {
    // A block whose pins depend on its parameters declares none until it has
    // some, and the catalog reports it that way rather than inventing a set.
    const declared = Object.keys(def.ports.in).length + Object.keys(def.ports.out).length;
    entry.dynamic_ports = declared === 0;
    entry.inputs = Object.entries(def.ports.in).map(([n, shape]) => `${n}: ${shape}`);
    entry.outputs = Object.entries(def.ports.out).map(([n, shape]) => `${n}: ${shape}`);
  } else if (isContainer(def)) {
    entry.inputs = ["(from the container's boundary_in block)"];
    entry.outputs = ["(from the container's boundary_out block)"];
    entry.dynamic_ports = true;
  }

  return entry;
}

function catalogParam(name: string, spec: ParamSpec): CatalogParam {
  const p: CatalogParam = { name, type: spec.type };
  const dflt = (spec as { default?: unknown }).default;
  if (dflt !== undefined) p.default = JSON.stringify(dflt);
  if (spec.doc) p.doc = spec.doc;
  if (spec.type === "enum" && spec.values) p.values = [...spec.values];
  return p;
}

export function allCatalogEntries(): CatalogEntry[] {
  const byCategory = catalogByCategory();
  return Object.keys(byCategory)
    .sort()
    .flatMap((category) => byCategory[category].map(catalogEntry));
}

export function catalogText(entries: CatalogEntry[]): string {
  return entries
    .map((e) => {
      const lines = [
        e.name && e.name !== e.type
          ? `${e.type}  — ${e.name}  (${e.kind}/${e.category})`
          : `${e.type}  (${e.kind}/${e.category})`,
        `  ${e.summary}`,
      ];
      if (e.formula) lines.push(`  formula: ${e.formula}`);
      if (e.params.length > 0) {
        lines.push(
          `  params: ${e.params
            .map((p) => `${p.name}:${p.type}${p.default !== undefined ? `=${p.default}` : ""}`)
            .join(", ")}`,
        );
      }
      if (e.inputs.length > 0) lines.push(`  in: ${e.inputs.join(", ")}`);
      if (e.outputs.length > 0) lines.push(`  out: ${e.outputs.join(", ")}`);
      return lines.join("\n");
    })
    .join("\n\n");
}

// ---------------------------------------------------------------------------
// Validation and analysis
// ---------------------------------------------------------------------------

export interface FindingJson {
  rule: string;
  severity: "error" | "warning" | "info";
  path?: string;
  port?: string;
  message: string;
  hint?: string;
}

export function findingsJson(report: ValidationReport): FindingJson[] {
  return report.findings.map((f) => {
    const out: FindingJson = { rule: f.rule, severity: f.severity, message: f.message };
    if (f.path) out.path = f.path;
    if (f.port) out.port = f.port;
    if (f.hint) out.hint = f.hint;
    return out;
  });
}

export interface ValidationSummary {
  ok: boolean;
  counts: { error: number; warning: number; info: number };
  /** The first few findings, worst first. Call `tensorcad_validate` for all of them. */
  top_findings: FindingJson[];
}

export function validationSummary(report: ValidationReport, limit = 5): ValidationSummary {
  return {
    ok: report.ok,
    counts: report.counts,
    top_findings: findingsJson(report).slice(0, limit),
  };
}

export function findingsText(findings: FindingJson[]): string {
  if (findings.length === 0) return "No findings.";
  return findings
    .map((f) => {
      const where = f.path ? ` [${f.path}${f.port ? `:${f.port}` : ""}]` : "";
      return `${f.severity}: ${f.message}${where}${f.hint ? `\n  fix: ${f.hint}` : ""}  (${f.rule})`;
    })
    .join("\n");
}

/** JSON does not have Infinity or NaN. */
function n(v: number): number | null {
  return Number.isFinite(v) ? v : null;
}

export function analysisJson(a: AnalysisResult): Record<string, unknown> {
  const o = a.options;
  return {
    name: a.name,
    options: {
      T: o.T,
      B: o.B,
      // The source length, only for a design with a second sequence.
      ...(o.S !== undefined ? { S: o.S } : {}),
      dtype: o.dtype,
      hardware: o.hardware.id,
      gpus: o.gpus,
      parallel: { ...o.parallel },
      optimizer: o.optimizer,
      recompute: o.recompute,
      tokens: o.tokens,
      tokens_were_defaulted: o.tokensWereDefaulted,
      mfu: o.mfu,
      concurrency: o.concurrency,
      ...(o.packing ? { packing: { ...o.packing } } : {}),
    },
    params: {
      total: a.params.total,
      active: a.params.active,
      embedding: a.params.embedding,
      head: a.params.head,
      non_embedding: a.params.nonEmbedding,
      non_embedding_active: a.params.nonEmbeddingActive,
      by_category: a.params.byCategory,
      by_type: a.params.byType,
    },
    flops: {
      fwd_dense: n(a.flops.fwdDense),
      fwd_attention: n(a.flops.fwdAttention),
      fwd_total: n(a.flops.fwdTotal),
      elementwise: n(a.flops.elementwise),
      train_per_token: n(a.flops.trainPerToken),
      attention_share: n(a.flops.attentionShare),
      // Two sequences: each one's forward pass per token of its own, and one
      // example's. The figures above are then per target token.
      ...(a.flops.perStream
        ? {
            per_stream: a.flops.perStream.map((s) => ({ symbol: s.symbol, length: s.length, fwd: n(s.fwd) })),
            fwd_per_example: n(a.flops.fwdPerExample ?? 0),
          }
        : {}),
      ...(a.flops.packed
        ? {
            packed: {
              fwd_attention: n(a.flops.packed.fwdAttention),
              fwd_total: n(a.flops.packed.fwdTotal),
              train_per_token: n(a.flops.packed.trainPerToken),
              attention_share: n(a.flops.packed.attentionShare),
              fwd_attention_blocks: n(a.flops.packed.fwdAttentionBlocks),
            },
          }
        : {}),
    },
    kv: {
      bytes_per_token: n(a.kv.bytesPerToken),
      bytes_per_sequence: n(a.kv.bytesPerToken * o.T + a.kv.bytesPerSequenceFixed),
      // Latent attention alone has two answers: the compressed vector a kernel
      // that scores in latent space holds, and the multi-head cache one that
      // does not holds instead.
      bytes_per_token_decompressed: n(a.kv.bytesPerTokenDecompressed),
    },
    memory: {
      optimizer_label: a.memory.optimizerLabel,
      train_weights: n(a.memory.train.perGpu.weights),
      train_grads: n(a.memory.train.perGpu.grads),
      train_optimizer: n(a.memory.train.perGpu.optimizer),
      train_activations: n(a.memory.train.perGpu.activations),
      train_per_gpu: n(a.memory.train.perGpu.total),
      train_total: n(a.memory.train.total),
      infer_weights: n(a.memory.infer.weights),
      infer_kv: n(a.memory.infer.kv),
      infer_total: n(a.memory.infer.total),
      device_memory: n(o.hardware.memory),
      notes: a.memory.notes,
    },
    throughput: {
      decode_tokens_per_second: n(a.throughput.decodeTokensPerSecond),
      // For a mixture of experts these differ: a batch reads the union of what
      // its tokens routed to, not one token's share.
      decode_weight_bytes: n(a.throughput.decodeWeightBytes),
      resident_weight_bytes: n(a.throughput.residentWeightBytes),
      prefill_seconds: n(a.throughput.prefillSeconds),
      memory_bound: a.throughput.memoryBound,
      notes: a.throughput.notes,
    },
    cost: {
      total_flops: n(a.cost.totalFlops),
      gpu_hours: n(a.cost.gpuHours),
      wall_clock_hours: n(a.cost.wallClockHours),
      dollars: n(a.cost.dollars),
      tokens: a.cost.tokens,
    },
    chinchilla: {
      optimal_tokens: n(a.chinchilla.optimalTokens),
      tokens_per_param: n(a.chinchilla.tokensPerParam),
      over_training_ratio: n(a.chinchilla.overTrainingRatio),
      verdict: a.chinchilla.verdict,
    },
    errors: a.errors,
  };
}

export function analysisText(a: AnalysisResult): string {
  const o = a.options;
  const fits = a.memory.train.perGpu.total <= o.hardware.memory;
  return [
    `${a.name} at T=${o.T} B=${o.B} ${o.dtype} on ${o.gpus} x ${o.hardware.id}`,
    ``,
    `parameters      ${formatCount(a.params.total)} total, ${formatCount(a.params.active)} active, ` +
      `${formatCount(a.params.nonEmbedding)} non-embedding`,
    `flops/token     ${formatFlops(a.flops.fwdTotal)} forward, ${formatFlops(a.flops.trainPerToken)} training; ` +
      `attention ${(a.flops.attentionShare * 100).toFixed(1)}%`,
    ...(a.flops.packed && o.packing
      ? [
          `packed          ${formatFlops(a.flops.packed.trainPerToken)} training in documents of ${o.packing.mean} ` +
            `tokens (spread ${o.packing.spread}); attention ${(a.flops.packed.attentionShare * 100).toFixed(1)}%, ` +
            `${(a.flops.packed.fwdAttentionBlocks / a.flops.packed.fwdAttention).toFixed(2)}x that in whole blocks`,
        ]
      : []),
    `kv cache        ${formatBytes(a.kv.bytesPerToken)}/token, ` +
      `${formatBytes(a.kv.bytesPerToken * o.T + a.kv.bytesPerSequenceFixed)} at T=${o.T}`,
    `train memory    ${formatBytes(a.memory.train.perGpu.total)} per GPU ` +
      `(weights ${formatBytes(a.memory.train.perGpu.weights)}, optimizer ${formatBytes(a.memory.train.perGpu.optimizer)}, ` +
      `activations ${formatBytes(a.memory.train.perGpu.activations)}) - ` +
      `${fits ? "fits" : "does NOT fit"} ${formatBytes(o.hardware.memory)}`,
    `serve memory    ${formatBytes(a.memory.infer.total)} for ${o.concurrency} concurrent sequence(s)`,
    `throughput      ${a.throughput.decodeTokensPerSecond.toFixed(1)} tok/s decode ` +
      `(${a.throughput.memoryBound ? "memory" : "compute"} bound), prefill ${a.throughput.prefillSeconds.toFixed(2)} s`,
    `training cost   ${formatCount(a.cost.tokens)} tokens, ${a.cost.gpuHours.toFixed(0)} GPU-hours, ` +
      `$${a.cost.dollars.toFixed(0)}`,
    `chinchilla      ${a.chinchilla.tokensPerParam.toFixed(1)} tokens/param - ${a.chinchilla.verdict}`,
    ...(a.errors.length > 0 ? ["", ...a.errors.map((e) => `! ${e}`)] : []),
  ].join("\n");
}
