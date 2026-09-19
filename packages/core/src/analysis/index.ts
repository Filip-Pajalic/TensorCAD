/**
 * The analysis entry point: one call that produces every number the editor,
 * the CLI and the MCP server report.
 *
 * It is pure and fast enough to re-run on every keystroke: the 405B preset
 * analyses in a couple of milliseconds because repeat containers contribute a
 * multiplier instead of being unrolled.
 */

import type { Doc, SymbolTable } from "../ir/types.js";
import { resolveSymbols } from "../ir/symbols.js";
import { inferShapes, type InferResult } from "../shapes/infer.js";
import type { AnalysisCtx } from "../catalog/types.js";
import { flatten, type FlatResult } from "./flatten.js";
import { countParams, type ParamsResult } from "./params.js";
import { countFlops, type FlopsResult } from "./flops.js";
import { countKvCache, type KvResult } from "./kvcache.js";
import { analyzeMemory, DEFAULT_PARALLEL, type MemoryResult, type ParallelPlan } from "./memory.js";
import {
  analyzeChinchilla,
  analyzeCost,
  analyzeThroughput,
  type ChinchillaResult,
  type CostResult,
  type ThroughputResult,
} from "./cost.js";
import {
  DTYPE_BYTES,
  peakFlops,
  resolveHardware,
  type Dtype,
  type HardwareProfile,
  type OptimizerKind,
} from "./hardware.js";

export interface AnalysisOptions {
  /** Sequence length. Defaults to the document's runtime T. */
  T?: number;
  /** Micro-batch size. Defaults to the document's runtime B. */
  B?: number;
  /** Training dtype for weights and activations. */
  dtype?: Dtype;
  /** Serving dtype, which is often smaller than the training dtype. */
  inferenceDtype?: Dtype;
  /** Cache dtype. Defaults to the serving dtype. */
  kvDtype?: Dtype;
  hardware?: string | HardwareProfile;
  /** Total GPUs used for training. */
  gpus?: number;
  parallel?: Partial<ParallelPlan>;
  optimizer?: OptimizerKind;
  recompute?: "none" | "selective" | "full";
  /** Assume a memory-efficient attention kernel. */
  flash?: boolean;
  /** Training token budget. Defaults to the Chinchilla-optimal budget. */
  tokens?: number;
  /** Model FLOPs utilization. Defaults to the midpoint of the device's band. */
  mfu?: number;
  /** Fraction of peak reached by decode's skinny matmuls. */
  decodeEfficiency?: number;
  /** Concurrent sequences when serving. */
  concurrency?: number;
}

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
  recompute: "none" | "selective" | "full";
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
  infer: InferResult;
  expanded: InferResult;
  flat: FlatResult;
  params: ParamsResult;
  flops: FlopsResult;
  kv: KvResult;
  memory: MemoryResult;
  throughput: ThroughputResult;
  cost: CostResult;
  chinchilla: ChinchillaResult;
  errors: string[];
}

export interface AnalyzeInputs {
  symbols?: SymbolTable;
  infer?: InferResult;
  flat?: FlatResult;
  /** Shape inference with `expandComposites: true`. */
  expanded?: InferResult;
}

/**
 * Per-token activation width of each repeat container's stream, needed to model
 * full activation recomputation. Derived from the container's input shape with
 * the runtime dimensions set to one.
 */
function streamWidths(infer: InferResult, symbols: SymbolTable, flat: FlatResult): Record<string, number> {
  const out: Record<string, number> = {};
  const env: Record<string, number> = { ...symbols.designValues };
  for (const name of symbols.runtime) env[name] = 1;

  for (const rep of flat.repeats) {
    const ports = infer.ports.get(rep.path);
    if (!ports) continue;
    let width = 0;
    for (const portName of Object.keys(ports.in)) {
      const shape = infer.inputs.get(`${rep.path}:${portName}`);
      if (!shape) continue;
      let product = 1;
      let ok = true;
      for (const dim of shape) {
        const v = dim.toNumber(env);
        if (v === null) {
          ok = false;
          break;
        }
        product *= v;
      }
      if (ok) width += product;
    }
    if (width > 0) out[rep.path] = width;
  }
  return out;
}

export function analyze(doc: Doc, options: AnalysisOptions = {}, pre: AnalyzeInputs = {}): AnalysisResult {
  const symbols = pre.symbols ?? resolveSymbols(doc);
  const infer = pre.infer ?? inferShapes(doc, symbols);
  const flat = pre.flat ?? flatten(doc, symbols);
  // Activation memory needs the shape of every tensor inside each composite, so
  // it runs against the fully expanded graph rather than the collapsed view.
  const expanded = pre.expanded ?? inferShapes(doc, symbols, { expandComposites: true });

  const hardware = resolveHardware(options.hardware);
  const dtype = options.dtype ?? "bf16";
  const inferenceDtype = options.inferenceDtype ?? dtype;
  const kvDtype = options.kvDtype ?? inferenceDtype;

  const T = options.T ?? symbols.values.T ?? 2048;
  const B = options.B ?? symbols.values.B ?? 1;

  const parallel: ParallelPlan = { ...DEFAULT_PARALLEL, ...options.parallel };
  const recompute = options.recompute ?? "none";
  const flash = options.flash ?? true;

  const params = countParams(doc, symbols, flat);

  const trainCtx: AnalysisCtx = {
    T,
    B,
    bytes: DTYPE_BYTES[dtype],
    flash: flash || recompute === "selective",
  };
  const cacheCtx: AnalysisCtx = { ...trainCtx, bytes: DTYPE_BYTES[kvDtype] };

  const flops = countFlops(flat, {
    ctx: trainCtx,
    recompute,
    nonEmbeddingActive: params.nonEmbeddingActive,
  });
  const kv = countKvCache(flat, cacheCtx);

  const memory = analyzeMemory(flat, {
    ctx: trainCtx,
    optimizer: options.optimizer ?? "adamw",
    recompute,
    parallel,
    inferenceDtypeBytes: DTYPE_BYTES[inferenceDtype],
    concurrency: options.concurrency ?? 1,
    streamWidths: streamWidths(infer, symbols, flat),
    expanded,
    symbols,
    params,
    kv,
  });

  const peak = peakFlops(hardware, dtype);
  const mfu = options.mfu ?? (hardware.mfuHint[0] + hardware.mfuHint[1]) / 2;
  const decodeEfficiency = options.decodeEfficiency ?? 0.3;

  const tokensWereDefaulted = options.tokens === undefined;
  const tokens = options.tokens ?? 20 * params.nonEmbeddingActive;

  const throughput = analyzeThroughput({
    hardware,
    peak,
    mfu,
    decodeEfficiency,
    batch: options.concurrency ?? 1,
    seq: T,
    activeWeightBytes: params.active * DTYPE_BYTES[inferenceDtype],
    kv,
    flops,
  });

  const cost = analyzeCost({
    trainFlopsPerToken: flops.trainPerToken,
    tokens,
    gpus: options.gpus ?? 1,
    peak,
    mfu,
    pricePerHour: hardware.pricePerHour,
  });

  const chinchilla = analyzeChinchilla({
    total: params.total,
    active: params.active,
    nonEmbedding: params.nonEmbeddingActive,
    tokens,
  });

  return {
    name: doc.meta.name,
    options: {
      T,
      B,
      dtype,
      inferenceDtype,
      kvDtype,
      hardware,
      gpus: options.gpus ?? 1,
      parallel,
      optimizer: options.optimizer ?? "adamw",
      recompute,
      flash,
      tokens,
      tokensWereDefaulted,
      mfu,
      decodeEfficiency,
      concurrency: options.concurrency ?? 1,
    },
    symbols,
    infer,
    expanded,
    flat,
    params,
    flops,
    kv,
    memory,
    throughput,
    cost,
    chinchilla,
    errors: [
      ...symbols.errors,
      ...params.errors,
      ...flops.errors,
      ...kv.errors,
      ...memory.errors,
    ],
  };
}

export { flatten, type FlatNode, type FlatResult } from "./flatten.js";
export { countParams, formatCount, type ParamsResult } from "./params.js";
export { countFlops, formatFlops, type FlopsResult } from "./flops.js";
export { countKvCache, kvBytesFor, formatBytes, type KvResult } from "./kvcache.js";
export {
  analyzeMemory,
  DEFAULT_PARALLEL,
  type MemoryResult,
  type MemoryOptions,
  type ParallelPlan,
} from "./memory.js";
export {
  analyzeThroughput,
  analyzeCost,
  analyzeChinchilla,
  CHINCHILLA_FITS,
  formatHours,
  formatDollars,
  type ThroughputResult,
  type CostResult,
  type ChinchillaResult,
} from "./cost.js";
export {
  HARDWARE,
  HARDWARE_BY_ID,
  DEFAULT_HARDWARE,
  DTYPE_BYTES,
  OPTIMIZER_BYTES,
  peakFlops,
  resolveHardware,
  type HardwareProfile,
  type Dtype,
  type OptimizerKind,
} from "./hardware.js";
