/**
 * Memory accounting for training and inference.
 *
 * Training memory is four things: weights, gradients, optimizer state and
 * activations, plus the logits buffer which is large enough at modern
 * vocabularies to deserve its own line. Sharding follows ZeRO: stage 1 shards
 * the optimizer, stage 2 adds gradients, stage 3 adds weights. Tensor
 * parallelism divides the weights, and divides activations only when sequence
 * parallelism is on.
 *
 * References: ZeRO (arXiv 1910.02054) and Korthikanti et al. on activation
 * recomputation (arXiv 2205.05198).
 */

import type { AnalysisCtx } from "../catalog/types.js";
import type { FlatResult } from "./flatten.js";
import type { InferResult } from "../shapes/infer.js";
import type { SymbolTable } from "../ir/types.js";
import type { ParamsResult } from "./params.js";
import { kvBytesFor, type KvResult } from "./kvcache.js";
import { OPTIMIZER_BYTES, type OptimizerKind } from "./hardware.js";

export interface ParallelPlan {
  /** Data-parallel replicas. */
  dp: number;
  /** Tensor-parallel degree. */
  tp: number;
  /** Pipeline stages. */
  pp: number;
  /** Expert-parallel degree. */
  ep: number;
  /** ZeRO/FSDP stage, 0 to 3. */
  zero: 0 | 1 | 2 | 3;
  /** Shard activations along the sequence dimension across the TP group. */
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

export interface MemoryOptions {
  ctx: AnalysisCtx;
  optimizer: OptimizerKind;
  recompute: "none" | "selective" | "full";
  parallel: ParallelPlan;
  /** Bytes per weight when serving. */
  inferenceDtypeBytes: number;
  /** Concurrent sequences when serving. */
  concurrency: number;
  /** Repeat-container path -> per-token activation width of its stream. */
  streamWidths: Record<string, number>;
  /** Shape inference with composites expanded, used to size retained tensors. */
  expanded: InferResult;
  symbols: SymbolTable;
  params: ParamsResult;
  kv: KvResult;
}

/** Elements per token in a tensor, with the runtime dimensions set to one. */
function elementsPerToken(shape: readonly { toNumber(env: Record<string, number>): number | null }[], env: Record<string, number>): number | null {
  let product = 1;
  for (const dim of shape) {
    const v = dim.toNumber(env);
    if (v === null) return null;
    product *= v;
  }
  return product;
}

export interface MemoryResult {
  /** Weight bytes at the training dtype, unsharded. */
  weightsBytes: number;
  train: {
    weights: number;
    grads: number;
    optimizer: number;
    activations: number;
    /** Part of `activations` attributable to the vocabulary logits. */
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
  infer: {
    weights: number;
    kv: number;
    overhead: number;
    total: number;
  };
  optimizerLabel: string;
  notes: string[];
  errors: string[];
}

export function analyzeMemory(flat: FlatResult, opts: MemoryOptions): MemoryResult {
  const notes: string[] = [];
  const errors: string[] = [];
  const { ctx, parallel: par } = opts;
  const tokens = ctx.B * ctx.T;

  // --- activations ---------------------------------------------------------
  // Memory is attributed to tensors rather than to blocks. A tensor that several
  // blocks read (the residual stream feeding the query, key and value
  // projections) is kept alive once, not once per reader.
  const activationsByPath: Record<string, number> = {};
  let activations = 0;
  let logits = 0;

  const fullRecompute = opts.recompute === "full";
  const env: Record<string, number> = { ...opts.symbols.designValues };
  for (const name of opts.symbols.runtime) env[name] = 1;

  const countedTensors = new Set<string>();
  const add = (path: string, bytes: number): void => {
    if (bytes === 0) return;
    activations += bytes;
    activationsByPath[path] = (activationsByPath[path] ?? 0) + bytes;
  };

  for (const node of flat.nodes) {
    if (fullRecompute && node.container) continue;

    const retained = node.def.retains ? node.def.retains(node.resolved) : [];
    for (const port of retained) {
      const consumerKey = `${node.path}:${port}`;
      const producer = opts.expanded.producerOf.get(consumerKey) ?? consumerKey;
      if (countedTensors.has(producer)) continue;
      countedTensors.add(producer);

      const shape = opts.expanded.inputs.get(consumerKey);
      if (!shape) {
        errors.push(`${node.path}: no shape for retained input "${port}", so its activation memory is missing`);
        continue;
      }
      const elements = elementsPerToken(shape, env);
      if (elements === null) {
        errors.push(`${node.path}: could not size the tensor on input "${port}"`);
        continue;
      }
      const owner = producer.slice(0, producer.lastIndexOf(":")) || node.path;
      add(owner, elements * ctx.bytes * node.activeMultiplier * tokens);
    }

    if (node.def.extraActivationBytes) {
      let extra = 0;
      try {
        extra = node.def.extraActivationBytes(node.resolved, ctx);
      } catch (e) {
        errors.push(`${node.path}: ${(e as Error).message}`);
      }
      const total = extra * node.activeMultiplier * tokens;
      add(node.path, total);
      if (node.type === "lm_head") logits += total;
    }
  }

  if (fullRecompute) {
    for (const rep of flat.repeats) {
      if (rep.type !== "repeat") continue;
      const width = opts.streamWidths[rep.path];
      if (width === undefined) {
        errors.push(`Could not determine the stream width of "${rep.path}" for full recomputation`);
        continue;
      }
      const total = width * ctx.bytes * rep.count * tokens;
      activations += total;
      activationsByPath[rep.path] = total;
    }
    notes.push(
      "Full recomputation keeps only each layer's input, at the cost of one extra forward pass (8N instead of 6N).",
    );
  } else if (opts.recompute === "selective") {
    notes.push(
      "Selective recomputation drops the attention score matrix. A memory-efficient attention kernel already does this, so the two coincide here.",
    );
  }

  // --- weights, gradients, optimizer ---------------------------------------
  const opt = OPTIMIZER_BYTES[opts.optimizer];
  if (!opt) errors.push(`Unknown optimizer "${opts.optimizer}"`);
  const bytesPer = opt ?? OPTIMIZER_BYTES.adamw;

  const total = opts.params.total;
  const weights = total * bytesPer.weights;
  const grads = total * bytesPer.grads;
  const optimizer = total * bytesPer.optimizer;

  const shardModel = Math.max(1, par.tp) * Math.max(1, par.pp);
  const dp = Math.max(1, par.dp);

  let wGpu = weights / shardModel;
  let gGpu = grads / shardModel;
  let oGpu = optimizer / shardModel;
  if (par.zero >= 1) oGpu /= dp;
  if (par.zero >= 2) gGpu /= dp;
  if (par.zero >= 3) wGpu /= dp;

  const actGpu = activations / (par.sequenceParallel ? Math.max(1, par.tp) : 1);

  if (par.pp > 1) {
    notes.push(
      "Pipeline parallelism divides the weights but not the activations of the first stage: under 1F1B it holds one micro-batch worth of the whole model.",
    );
  }
  if (par.tp > 1 && !par.sequenceParallel) {
    notes.push("Without sequence parallelism, tensor parallelism leaves the norm and dropout activations replicated.");
  }
  if (opts.optimizer === "bf16_adam") {
    notes.push("Pure bf16 Adam without master weights saves 8 bytes per parameter but is prone to divergence.");
  }

  const trainTotal = weights + grads + optimizer + activations;
  const trainPerGpuTotal = wGpu + gGpu + oGpu + actGpu;

  // --- inference -----------------------------------------------------------
  const inferWeights = total * opts.inferenceDtypeBytes;
  const inferKv = kvBytesFor(opts.kv, ctx.T, opts.concurrency);
  const overhead = 0.2 * (inferWeights + inferKv);

  return {
    weightsBytes: weights,
    train: {
      weights,
      grads,
      optimizer,
      activations,
      logits,
      total: trainTotal,
      perGpu: {
        weights: wGpu,
        grads: gGpu,
        optimizer: oGpu,
        activations: actGpu,
        total: trainPerGpuTotal,
      },
      activationsByPath,
    },
    infer: {
      weights: inferWeights,
      kv: inferKv,
      overhead,
      total: inferWeights + inferKv + overhead,
    },
    optimizerLabel: bytesPer.label,
    notes,
    errors,
  };
}
