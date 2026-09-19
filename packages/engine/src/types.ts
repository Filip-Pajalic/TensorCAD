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
export interface Doc {
  version: number;
  meta: DocMeta;
  symbols: Record<string, SymbolDef>;
  graph: Graph;
  /** Blocks this design defines for itself, keyed by type name. */
  defs?: Record<string, UserBlockDef>;
  ui?: UiState;
}

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
}

/** What a pin declares. See docs/reference/ports.md. */
export interface PortSpec {
  shape: string;
  /** What the tensor carries; "inherit" takes it from the producer. */
  dtype?: string;
  /** A port that may legitimately dangle. */
  optional?: boolean;
  /** "flow" or "side": which edge of the symbol a wire leaves by. */
  anchor?: "flow" | "side";
  showName?: boolean;
  doc?: string;
}

export interface BlockDocs {
  summary?: string;
  formula?: string;
  refs?: string[];
}

/** One block, as the palette and the inspector need it. */
export interface CatalogEntry {
  type: string;
  kind: "primitive" | "composite" | "container";
  category: string;
  summary: string;
  formula?: string;
  refs?: string[];
  /** In the order the block declares them, which is how they are laid out. */
  params: (ParamSpec & { name: string })[];
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
    perGpu: {
      weights: number;
      grads: number;
      optimizer: number;
      activations: number;
      total: number;
    };
    activationsByPath: Record<string, number>;
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

export interface ScaleResult {
  doc: Doc;
  achieved: number;
  target: number;
  changes: Record<string, { from: number; to: number }>;
  notes: string[];
}

export interface ImportResult {
  doc: Doc;
  /** What the import could not represent faithfully. */
  warnings: string[];
}
