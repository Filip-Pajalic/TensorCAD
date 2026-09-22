/**
 * What the engine's JSON means.
 *
 * These are a description of the wire, not a second model of the domain: each
 * one names a shape the Go engine marshals, and nothing here computes. The Go
 * side is where a field's meaning is decided; this is how TypeScript reads it.
 */

// ---------------------------------------------------------------------------
// The document
// ---------------------------------------------------------------------------

export const DOC_VERSION = 1;

/** Symbols that stay indeterminate through shape checking. */
export const RUNTIME_SYMBOLS = ["B", "T"] as const;

/** The generated entry and exit nodes of a container's subgraph. */
export const BOUNDARY_IN = "_in";
export const BOUNDARY_OUT = "_out";

/** A parameter as the document writes it: a number, a flag, or an expression. */
export type ParamValue = number | string | boolean | null | ParamObject | ParamValue[];
export interface ParamObject {
  [key: string]: ParamValue;
}

/**
 * A symbol, in any of the four spellings a document uses: a bare number, a bare
 * expression, or an object tagged "runtime" or "design".
 */
export type SymbolDef =
  | number
  | string
  | { kind: "runtime"; default?: number; doc?: string }
  | { kind: "design"; value: number | string; doc?: string };

/** One block in a graph. */
export interface NodeDef {
  id: string;
  type: string;
  params?: Record<string, ParamValue>;
  /** The subgraph of a container such as `repeat`. */
  graph?: Graph;
  /** Named subgraph alternatives, for hybrid repeat patterns. */
  variants?: Record<string, Graph>;
  /** Free text shown on the canvas instead of the id. */
  label?: string;
}

/** An edge, `"nodeId:portName"` on both ends. */
export type Edge = [from: string, to: string];

export interface Graph {
  nodes: NodeDef[];
  edges: Edge[];
}

/** The reference numbers a preset is checked against. */
export interface Published {
  params?: number;
  activeParams?: number;
  kvBytesPerToken?: number;
  source?: string;
  /**
   * Allowed relative difference, defaulting to 0.5%. Set wider only where the
   * published figure is itself a rounded headline number.
   */
  tolerance?: number;
}

export interface DocMeta {
  name: string;
  family?: string;
  notes?: string;
  published?: Published;
}

/** The editor's view of a document, carried along with it. */
export interface UiState {
  positions?: Record<string, [number, number]>;
  collapsed?: string[];
}

/** A design. */
/** One named set of symbol values. */
export interface Configuration {
  doc?: string;
  /**
   * Overrides the document's own symbols, by name. A symbol it does not
   * mention keeps whatever the design says, so an expression over one that is
   * overridden follows it.
   */
  symbols: Record<string, SymbolDef>;
}

export interface Doc {
  version: number;
  meta: DocMeta;
  symbols: Record<string, SymbolDef>;
  graph: Graph;
  /**
   * What this design has decided the design rules mean to it, keyed by rule id.
   *
   * A rule that is right in general is sometimes wrong here, and the
   * alternative to recording that is people learning to read past a warning.
   * It lives in the document because it is a decision about the design, so it
   * travels with the file and shows up in review. Suppression is never silent:
   * `ValidationReport.overridden` says what it did.
   */
  rules?: Record<string, RuleSeverity>;
  /**
   * Named sets of symbol values this design can be built at.
   *
   * Four GPT-2 presets are the same architecture at four sizes, and holding
   * them apart means an architectural change has to be made four times. A
   * configuration overrides symbols and nothing else: a variant that changed
   * the graph would be a different design.
   */
  configurations?: Record<string, Configuration>;
  /**
   * The configuration in force. Absent or unknown means the symbols as
   * written, which is what a design without configurations always has.
   *
   * In the document rather than in the editor, because it changes what the
   * design *is*. The operating point is editor state for the opposite reason:
   * it only changes what the design is measured under.
   */
  active?: string;
  /** Blocks this design defines for itself, keyed by type name. */
  defs?: Record<string, UserBlockDef>;
  ui?: UiState;
}

/** What a document may ask a rule to be. `off` drops its findings. */
export type RuleSeverity = Severity | "off";

/** A composite a design defines for itself, written as data. */
export interface UserBlockDef {
  type?: string;
  category?: string;
  params?: Record<string, ParamSpec>;
  ports: { in: Record<string, string | PortSpec>; out: Record<string, string | PortSpec> };
  graph: Graph;
  docs?: BlockDocs;
}

// ---------------------------------------------------------------------------
// The catalog
// ---------------------------------------------------------------------------

export type ParamKind = "int" | "num" | "bool" | "enum" | "str" | "pattern" | "obj";

/** One declared parameter of a block. */
export interface ParamSpec {
  type: ParamKind;
  default?: ParamValue;
  min?: number;
  max?: number;
  /** The permitted strings of an enum. */
  values?: string[];
  doc?: string;
  /**
   * The heading this field belongs under, for a block with enough parameters
   * that one list is unreadable. Absent puts it with the rest.
   */
  group?: string;
  /**
   * The condition under which this parameter means anything.
   *
   * A dense block has no `expert_hidden`, grouped-query attention has no
   * `kv_lora`, an RMSNorm has no bias. The block carries a value either way —
   * this is what lets a panel say so rather than showing them all alike.
   */
  when?: ParamWhen;
}

/** One parameter's value deciding whether another is meaningful. */
export interface ParamWhen {
  /** The other parameter to look at. */
  param: string;
  /** The values of it under which this parameter is meaningful. */
  is: string[];
}

/** What a pin declares. See docs/reference/ports.md. */
export interface PortSpec {
  shape: string;
  /** What the tensor carries; "inherit" takes it from the producer. */
  dtype?: string;
  /** A port that may legitimately dangle. */
  optional?: boolean;
  /**
   * What the tensor carries, as a *kind* rather than a width: "real" for
   * anything a matmul can multiply, "int" for an index, "bool" for a mask,
   * "inherit" to take it from whatever arrives. Which real type — fp32,
   * bf16, fp8 — is a condition of the run and belongs to the operating
   * point (invariant 8), not to the block.
   */
  /** "flow" or "side": which edge of the symbol a wire leaves by. */
  anchor?: "flow" | "side";
  showName?: boolean;
  doc?: string;
}

export interface BlockDocs {
  /**
   * What a drawing calls this block: a short noun phrase in the words a
   * published figure would use, where `type` is the identifier the engine
   * dispatches on and a path is written with. Both are kept — the identifier is
   * what an MCP call names, and a reader who found a block by its drawing then
   * has to type it.
   */
  name?: string;
  summary?: string;
  formula?: string;
  refs?: string[];
}

/**
 * One block, as the palette and the inspector need it.
 *
 * Parameters by name with the order beside them: a panel looks one up far more
 * often than it walks them all, and a Go map has no order to inherit.
 */
export interface CatalogEntry {
  type: string;
  kind: "primitive" | "composite" | "container";
  category: string;
  docs: BlockDocs;
  params: Record<string, ParamSpec>;
  /** The order the block declares them in, which is how they are laid out. */
  paramOrder: string[];
  /**
   * The pins declared before any parameter is known. A block whose pins depend
   * on its parameters reports none here; ask the analysis for those.
   */
  ports: { in: Record<string, string>; out: Record<string, string> };
}

// ---------------------------------------------------------------------------
// The operating point
// ---------------------------------------------------------------------------

export type Dtype = "fp32" | "bf16" | "fp16" | "fp8";
export type OptimizerKind = "adamw" | "adamw8bit" | "muon" | "sgd_momentum" | "sgd" | "bf16_adam";
export type Recompute = "none" | "selective" | "full";

/** How the model is spread over the devices. */
export interface ParallelPlan {
  dp: number;
  tp: number;
  pp: number;
  ep: number;
  /** ZeRO/FSDP stage, 0 to 3. */
  zero: 0 | 1 | 2 | 3;
  /** Shard activations along the sequence across the tensor-parallel group. */
  sequenceParallel: boolean;
}

export const DEFAULT_PARALLEL: ParallelPlan = {
  dp: 1,
  tp: 1,
  pp: 1,
  ep: 1,
  zero: 0,
  sequenceParallel: false,
};

export const DEFAULT_HARDWARE = "h100-sxm";

export const DTYPE_BYTES: Record<Dtype, number> = { fp32: 4, bf16: 2, fp16: 2, fp8: 1 };

/**
 * The conditions a design is measured under.
 *
 * Not properties of the design: batch, sequence length, dtype, device and
 * sharding belong to the question, not to the model.
 */
export interface AnalysisOptions {
  T?: number;
  B?: number;
  dtype?: Dtype;
  inferenceDtype?: Dtype;
  kvDtype?: Dtype;
  /** A profile id from `hardware()`. */
  hardware?: string;
  gpus?: number;
  parallel?: Partial<ParallelPlan>;
  optimizer?: OptimizerKind;
  recompute?: Recompute;
  /** Assume a memory-efficient attention kernel. */
  flash?: boolean;
  tokens?: number;
  mfu?: number;
  decodeEfficiency?: number;
  concurrency?: number;
}

export interface HardwareProfile {
  id: string;
  name: string;
  peakBf16: number;
  peakFp8: number;
  memory: number;
  bandwidth: number;
  pricePerHour: number;
  mfuHint: [low: number, high: number];
  notes?: string;
}

// ---------------------------------------------------------------------------
// The analysis
// ---------------------------------------------------------------------------

export interface SymbolTable {
  order: string[];
  values: Record<string, number>;
  /** Concrete design symbols only; runtime symbols excluded. */
  designValues: Record<string, number>;
  docs: Record<string, string>;
  errors: string[];
}

export interface ParamsResult {
  total: number;
  /** Weights a single token uses. Differs from total only for a mixture. */
  active: number;
  embedding: number;
  head: number;
  nonEmbedding: number;
  /** The N in the 2N rule. */
  nonEmbeddingActive: number;
  byPath: Record<string, number>;
  byCategory: Record<string, number>;
  byType: Record<string, number>;
  errors: string[];
}

export interface FlopsResult {
  fwdDense: number;
  fwdAttention: number;
  /** The attention term as a profiler counts it, with nothing masked. */
  fwdAttentionUnmasked: number;
  fwdTotal: number;
  fwdTotalUnmasked: number;
  /** Norms, activations, RoPE and adds: memory-bound, excluded above. */
  elementwise: number;
  trainPerToken: number;
  attentionShare: number;
  ruleOfThumb2N: number;
  ruleOfThumb6N: number;
  byPath: Record<string, number>;
  byCategory: Record<string, number>;
  errors: string[];
}

export interface KvResult {
  bytesPerToken: number;
  bytesPerSequenceFixed: number;
  /**
   * The same cache under an engine that does not absorb the weights latent
   * attention compressed against. Equal to `bytesPerToken` for every design
   * that has no latent attention in it; 57x it for DeepSeek-V3.
   */
  bytesPerTokenDecompressed: number;
  byPath: Record<string, number>;
  errors: string[];
}

export interface MemoryResult {
  weightsBytes: number;
  train: {
    weights: number;
    grads: number;
    optimizer: number;
    activations: number;
    /** The part of `activations` that is the vocabulary logits. */
    logits: number;
    total: number;
    perGpu: TrainPerGpu;
    activationsByPath: Record<string, number>;
    /**
     * The same bytes charged to the tensor rather than to the block that
     * produced it, keyed `"path:port"`.
     *
     * Not a reformatting of the line above. They agree row for row in a plain
     * transformer, where every block that holds an activation holds exactly
     * one; they diverge wherever a block fans out. Nemotron-H's `split` holds
     * three tensors from 2 MiB to 167 MiB, and one number for the block
     * answers neither which of them is the big one nor what dropping one would
     * save.
     */
    activationsByTensor: Record<string, number>;
  };
  infer: { weights: number; kv: number; overhead: number; total: number };
  optimizerLabel: string;
  notes: string[];
  errors: string[];
}

export interface ThroughputResult {
  /** FLOP per byte above which the device is compute-bound. */
  ridgePoint: number;
  decodeBytesPerStep: number;
  /**
   * The weights of that, which for a mixture of experts is neither the active
   * count nor the resident one: a batch reads the union of what its tokens
   * routed to, and that reaches nearly every expert well before the batch
   * reaches the expert count.
   */
  decodeWeightBytes: number;
  /** Every weight the device holds, whether or not a given step reads it. */
  residentWeightBytes: number;
  decodeFlopsPerStep: number;
  decodeSecondsPerStep: number;
  decodeTokensPerSecond: number;
  memoryBound: boolean;
  prefillSeconds: number;
  notes: string[];
}

export interface CostResult {
  totalFlops: number;
  gpuHours: number;
  wallClockHours: number;
  dollars: number;
  tokens: number;
  mfu: number;
}

export interface ChinchillaResult {
  optimalTokens: number;
  tokensPerParam: number;
  /** The meaningful ratio for a sparse model. */
  tokensPerActiveParam: number;
  overTrainingRatio: number;
  predictedLoss: Record<string, number>;
  verdict: string;
}

/** The operating point with every default filled in. */
export interface ResolvedAnalysisOptions {
  T: number;
  B: number;
  dtype: Dtype;
  inferenceDtype: Dtype;
  kvDtype: Dtype;
  hardware: HardwareProfile;
  gpus: number;
  parallel: ParallelPlan;
  optimizer: OptimizerKind;
  recompute: Recompute;
  flash: boolean;
  tokens: number;
  tokensWereDefaulted: boolean;
  mfu: number;
  decodeEfficiency: number;
  concurrency: number;
}

export interface AnalysisResult {
  name: string;
  options: ResolvedAnalysisOptions;
  symbols: SymbolTable;
  params: ParamsResult;
  flops: FlopsResult;
  kv: KvResult;
  memory: MemoryResult;
  throughput: ThroughputResult;
  cost: CostResult;
  chinchilla: ChinchillaResult;
  errors: string[];
}

// ---------------------------------------------------------------------------
// The design rules
// ---------------------------------------------------------------------------

export type Severity = "error" | "warning" | "info";

export interface Finding {
  /** Stable rule identifier, e.g. "flash-head-dim". */
  rule: string;
  severity: Severity;
  path?: string;
  port?: string;
  /** The parameter that caused it, so the inspector can point at the field. */
  param?: string;
  message: string;
  hint?: string;
}

export interface ValidationReport {
  name: string;
  findings: Finding[];
  counts: Record<Severity, number>;
  /** True when nothing blocks building this design. */
  ok: boolean;
  analysis: AnalysisResult;
  /**
   * What the document's own severities did, so suppression is never silent: a
   * design cannot drop a finding without the report saying which and from what.
   */
  overridden: RuleOverride[];
}

/** One finding whose severity the document changed. */
export interface RuleOverride {
  rule: string;
  path?: string;
  /** What the rule produced, and what the document asked for. */
  from: Severity;
  /** `off` means the finding was dropped. An unreadable value arrives as `?x`. */
  to: string;
}

// ---------------------------------------------------------------------------
// Explain
// ---------------------------------------------------------------------------

export interface ExplainedParam {
  /** The expression as written, when it was an expression. */
  expression?: string;
  value: unknown;
  doc?: string;
}

export interface Explanation {
  path: string;
  type: string;
  kind: "primitive" | "composite" | "container";
  label?: string;
  docs: BlockDocs;
  /** How many copies exist, and how many a token passes through. */
  copies: { total: number; active: number };
  params: Record<string, ExplainedParam>;
  /** The order the block declares its parameters in. */
  paramOrder: string[];
  shapes: { in: Record<string, string>; out: Record<string, string> };
  contributes: {
    params: number;
    activeParams: number;
    shareOfParams: number;
    flopsPerToken: number;
    shareOfFlops: number;
    activationBytes: number;
    cacheBytesPerToken: number;
    cacheBytesPerSequence: number;
  };
  /** The primitives this block expands into, largest first. */
  breakdown: { path: string; type: string; params: number }[];
  notFound?: boolean;
}

// ---------------------------------------------------------------------------
// Code generation, scaling, import
// ---------------------------------------------------------------------------

export interface TorchOptions {
  className?: string;
  includeSmokeTest?: boolean;
  /**
   * How a mixture-of-experts layer routes. "sparse" is faster but cannot be
   * traced by `torch.export`; "dense" computes the same thing at
   * `experts / top_k` times the cost and traces cleanly.
   */
  moeDispatch?: "sparse" | "dense";
  /** 0 leaves PyTorch's own initialization alone. */
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

export interface ScaleOptions {
  targetParams: number;
  widthSymbols?: string[];
  depthSymbols?: string[];
  widthMultiple?: number;
  vocab?: number;
  /** Whether `targetParams` counts the embedding tables. */
  targetBasis?: "total" | "non-embedding";
  tieHead?: boolean;
  minHeads?: number;
  keepDepth?: boolean;
  maxIterations?: number;
}

/** The training footprint on one device. */
export interface TrainPerGpu {
  weights: number;
  grads: number;
  optimizer: number;
  activations: number;
  total: number;
}

/** One named value that moved between two designs. */
export interface DiffChange {
  name: string;
  from?: unknown;
  to?: unknown;
}

/** A block as it stands in one of the two designs. */
export interface DiffBlock {
  path: string;
  type: string;
  label?: string;
  params: Record<string, ParamValue>;
}

/** One parameter that moved on a block that exists in both. */
export interface DiffParamChange {
  key: string;
  from?: ParamValue;
  to?: ParamValue;
}

export interface DiffBlockChange {
  path: string;
  /** Set when the block became a different kind of block. */
  type?: DiffChange;
  label?: DiffChange;
  params: DiffParamChange[];
}

export interface DiffEdge {
  /** The graph the wire sits in; `<root>` is the top level. */
  graph: string;
  from: string;
  to: string;
}

/** One number that moved. */
export interface DiffDelta {
  metric: string;
  a: number;
  b: number;
  delta: number;
  /** Null when `a` is zero, because the ratio says nothing then. */
  ratio: number | null;
}

/**
 * What changed between two designs.
 *
 * Structure and numbers together, because either alone is misleading: that `F`
 * went from 11008 to 14336 does not tell you the model grew by 1.3B
 * parameters, and that it grew by 1.3B does not tell you where.
 */
export interface DesignDiff {
  a: string;
  b: string;
  symbols: { added: DiffChange[]; removed: DiffChange[]; changed: DiffChange[] };
  blocks: { added: DiffBlock[]; removed: DiffBlock[]; changed: DiffBlockChange[] };
  edges: { added: DiffEdge[]; removed: DiffEdge[] };
  metrics: DiffDelta[];
  /** The operating point both sides were measured under. */
  at: { T: number; B: number; hardware: string };
  /**
   * True when nothing structural moved. The numbers may still differ, because
   * they are measured at an operating point.
   */
  identical: boolean;
}

/** The cluster a design is being fitted to. */
export interface ClusterRequest {
  /** How many devices there are. */
  gpus: number;
  /**
   * Bounds the tensor-parallel degree: splitting a matrix across a slower link
   * than NVLink is rarely worth it. Defaults to 8.
   */
  gpusPerNode?: number;
  /**
   * The fraction of device memory left free for fragmentation, the allocator
   * and the communication buffers. Defaults to 0.1.
   */
  headroom?: number;
  /** Micro-batch sizes to try. Defaults to the one the analysis options give. */
  microBatch?: number[];
  /** Recompute settings to try. Defaults to all three. */
  recompute?: Recompute[];
  /** How many plans to return. Defaults to 8. */
  limit?: number;
}

/** One way of splitting the work, and what it costs to hold. */
export interface ClusterPlan {
  parallel: ParallelPlan;
  recompute: Recompute;
  microBatch: number;
  perGpu: TrainPerGpu;
  /** The fraction of the budget this plan takes; over 1 does not fit. */
  used: number;
  /** The plan as a person would say it: "DP 8 x TP 2, ZeRO-1". */
  summary: string;
  /** What this plan asks of whoever runs it. */
  notes: string[];
}

/** Every plan that fits, least demanding first. */
export interface ClusterResult {
  fits: ClusterPlan[];
  /** The nearest miss, when nothing fits. */
  closest?: ClusterPlan;
  /** How many combinations were priced. */
  considered: number;
  /** Bytes each device may use, after headroom. */
  budget: number;
  /** The device's own memory, before headroom. */
  memory: number;
  hardware: string;
  notes: string[];
}

export interface ScaleResult {
  doc: Doc;
  achieved: number;
  target: number;
  changes: Record<string, { from: number; to: number }>;
  notes: string[];
}

// ---------------------------------------------------------------------------
// The maximal-update-parametrization ladder
// ---------------------------------------------------------------------------

/** Which row of Tensor Programs V's Table 3 a weight belongs to. */
export type MupClass = "input" | "hidden" | "output";

/** What to multiply one class's settings by, against the base rung. */
export interface MupScaling {
  class: MupClass;
  /** Multiplies the base model's initialization standard deviation. */
  initStd: number;
  /** Multiplies the base model's learning rate. */
  adamLr: number;
  /** The blocks in this class, so the grouping can be checked against the design. */
  paths: string[];
  why: string;
}

/** One model in the ladder. */
export interface MupRung {
  /** What the width came out as: a width is held to a whole number of heads. */
  width: number;
  /** `width` over the base width: the m every rule is written in. */
  multiplier: number;
  heads: number;
  params: number;
  doc: Doc;
  /** The rung the hyperparameters are tuned at, where the multiplier is 1. */
  base: boolean;
  scaling: MupScaling[];
  notes: string[];
}

export interface MupLadder {
  /** What the ladder moved, normally D. */
  widthSymbol: string;
  baseWidth: number;
  /** What was held fixed while the width moved. */
  headDim: number;
  rungs: MupRung[];
  notes: string[];
}

export interface MupOptions {
  /** The rungs. Empty halves the design's own width down to a width worth sweeping at. */
  widths?: number[];
  /** The width the sweep happens at. Omitted takes the narrowest rung. */
  baseWidth?: number;
  /** Symbols that move with the width beyond D. Omitted takes the same set scaling uses. */
  widthSymbols?: string[];
}

export interface ImportResult {
  doc: Doc;
  /** What the import could not represent faithfully. */
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Shape inference
// ---------------------------------------------------------------------------

/**
 * One tensor shape, in both the forms the editor shows.
 *
 * Symbolic is the honest one: `B T D` says the residual stream is D wide
 * whatever D is. Numeric substitutes the design symbols and leaves the runtime
 * ones alone, which is the quickest way to watch a symbol edit travel through a
 * design. Both come from the engine, because only it holds the polynomial.
 */
export interface Shape {
  symbolic: string;
  numeric: string;
}

/** One pin, as the canvas draws it. */
export interface ResolvedPort {
  shape: string;
  dtype: string;
  anchor: "flow" | "side";
  optional?: boolean;
  showName?: boolean;
  doc?: string;
}

export interface ResolvedPorts {
  in: Record<string, ResolvedPort>;
  out: Record<string, ResolvedPort>;
}

/** A node's parameters after evaluation. */
export interface Resolved {
  type: string;
  /** The concrete value of each parameter. */
  p: Record<string, ParamValue>;
  /** The symbolic form, so a width can be labelled "D" rather than 4096. */
  s: Record<string, string>;
}

/** Something wrong with the wiring, addressed to a node. */
export interface InferIssue {
  path: string;
  port?: string;
  message: string;
  severity: "error" | "warning";
  /** The block's own rule id, when the issue came from a block constraint. */
  rule?: string;
  param?: string;
}

/** Every shape in a design, keyed by `"path:port"`. */
export interface Inference {
  outputs: Record<string, Shape>;
  inputs: Record<string, Shape>;
  /** Consumer `"path:port"` to producer `"path:port"`. */
  producerOf: Record<string, string>;
  ports: Record<string, ResolvedPorts>;
  resolved: Record<string, Resolved>;
  /**
   * The subgraph each composite stood for, by path.
   *
   * The walk builds these anyway, so the editor draws the inside of a block
   * from what the analysis already saw rather than expanding it a second time
   * with its own copy of the rules.
   */
  expansions: Record<string, Graph>;
  issues: InferIssue[];
}

/** Everything the editor needs for one document at one operating point. */
export interface Derived {
  report: ValidationReport;
  /** Shape inference with composites expanded, so interiors can be opened. */
  infer: Inference;
}
