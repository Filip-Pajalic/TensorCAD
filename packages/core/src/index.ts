/**
 * @tensorcad/core — the pure core of TensorCAD.
 *
 * Everything here is deterministic and free of I/O so the editor, the CLI and
 * the MCP server can share one implementation.
 */

// IR
export type {
  Doc,
  DocMeta,
  Graph,
  NodeDef,
  Edge,
  ParamValue,
  ParamObject,
  SymbolDef,
  SymbolTable,
  SymbolValues,
  Resolved,
  UiState,
  NodePath,
} from "./ir/types.js";
export { DOC_VERSION, RUNTIME_SYMBOLS, splitEndpoint, joinPath } from "./ir/types.js";
export { resolveSymbols, symbolCtx } from "./ir/symbols.js";

// Symbolic shapes
export { Sym, rat, type Rat } from "./shapes/symexpr.js";
export { parseExpr, evalExpr, exprSymbols, EXPR_FUNCTIONS, type EvalCtx, type Ast } from "./shapes/expr.js";
export {
  parsePattern,
  instantiate,
  matchPattern,
  shapeToString,
  type Pattern,
  type Shape,
} from "./shapes/pattern.js";
export { inferShapes, evalCtxFor, type InferResult, type InferIssue, type InferOptions } from "./shapes/infer.js";

// Catalog
export {
  CATALOG,
  PRIMITIVES,
  COMPOSITES,
  CONTAINERS,
  getBlock,
  requireBlock,
  isPrimitive,
  isComposite,
  isContainer,
  catalogByCategory,
  resolveNodeParams,
  portsOf,
} from "./catalog/index.js";
export type {
  BlockDef,
  PrimitiveDef,
  CompositeDef,
  ContainerDef,
  ParamSpec,
  Ports,
  PortsSpec,
  AnalysisCtx,
  FlopsPerToken,
  StateBytes,
  BlockDocs,
} from "./catalog/types.js";

// Analysis
export { flatten, type FlatNode, type FlatBlock, type FlatResult } from "./analysis/flatten.js";
export { countParams, formatCount, type ParamsResult } from "./analysis/params.js";
export {
  analyze,
  countFlops,
  countKvCache,
  analyzeMemory,
  analyzeThroughput,
  analyzeCost,
  analyzeChinchilla,
  kvBytesFor,
  peakFlops,
  resolveHardware,
  formatFlops,
  formatBytes,
  formatHours,
  formatDollars,
  HARDWARE,
  HARDWARE_BY_ID,
  DEFAULT_HARDWARE,
  DTYPE_BYTES,
  OPTIMIZER_BYTES,
  DEFAULT_PARALLEL,
  CHINCHILLA_FITS,
  type AnalysisOptions,
  type AnalysisResult,
  type ResolvedAnalysisOptions,
  type FlopsResult,
  type KvResult,
  type MemoryResult,
  type ParallelPlan,
  type ThroughputResult,
  type CostResult,
  type ChinchillaResult,
  type HardwareProfile,
  type Dtype,
  type OptimizerKind,
} from "./analysis/index.js";

// Code generation
export { generateTorch, type TorchOptions, type GeneratedCode, type GeneratedFile } from "./codegen/index.js";

// Explanations
export { explain, explainAll, type Explanation, type ExplainedParam } from "./explain.js";

// Scaling
export { scaleDesign, type ScaleOptions, type ScaleResult } from "./scale.js";

// Import
export {
  importHfConfig,
  importHfConfigJson,
  SUPPORTED_MODEL_TYPES,
  type HfConfig,
  type ImportResult,
} from "./import/hf.js";

// Design rules
export {
  catalogOf,
  catalogEntries,
  isUserBlock,
  compileUserBlock,
  validateUserBlock,
  BOUNDARY_IN,
  BOUNDARY_OUT,
  type Catalog,
  type UserBlockDef,
} from "./catalog/index.js";
export { validate, RULES, type ValidationReport, type Finding, type Rule, type Severity } from "./rules/index.js";

// Presets
export { getPreset, allPresets, PRESET_NAMES, decoderOnly, type DecoderSpec } from "./presets/index.js";
