/**
 * Throughput, training cost and scaling-law sanity checks.
 *
 * The throughput model is a roofline: a decode step moves the active weights
 * and the cache through memory, and does a little arithmetic on them, so it is
 * memory-bound until the batch exceeds the device's ridge point.
 *
 * References: kipply's transformer inference arithmetic for the roofline,
 * Hoffmann et al. (Chinchilla) and Epoch's refit for the scaling law.
 */

import type { FlopsResult } from "./flops.js";
import type { KvResult } from "./kvcache.js";
import { kvBytesFor } from "./kvcache.js";
import type { HardwareProfile } from "./hardware.js";

export interface ThroughputResult {
  /** FLOP per byte above which the device is compute-bound. */
  ridgePoint: number;
  /** Bytes read per decode step. */
  decodeBytesPerStep: number;
  /** FLOPs per decode step. */
  decodeFlopsPerStep: number;
  /** Seconds per decode step, the larger of the memory and compute times. */
  decodeSecondsPerStep: number;
  decodeTokensPerSecond: number;
  /** True when memory bandwidth, not arithmetic, sets the pace. */
  memoryBound: boolean;
  /** Seconds to prefill one full context of T tokens for the whole batch. */
  prefillSeconds: number;
  notes: string[];
}

export interface ThroughputOptions {
  hardware: HardwareProfile;
  peak: number;
  /** Fraction of peak achieved during prefill. */
  mfu: number;
  /** Fraction of peak achieved during decode's small matmuls. */
  decodeEfficiency: number;
  batch: number;
  seq: number;
  activeWeightBytes: number;
  kv: KvResult;
  flops: FlopsResult;
}

export function analyzeThroughput(o: ThroughputOptions): ThroughputResult {
  const notes: string[] = [];
  const ridgePoint = o.peak / o.hardware.bandwidth;

  const kvBytes = kvBytesFor(o.kv, o.seq, o.batch);
  const decodeBytesPerStep = o.activeWeightBytes + kvBytes;
  const decodeFlopsPerStep = o.batch * o.flops.fwdTotal;

  const tMem = decodeBytesPerStep / o.hardware.bandwidth;
  const tCompute = decodeFlopsPerStep / (o.peak * o.decodeEfficiency);
  const step = Math.max(tMem, tCompute);
  const memoryBound = tMem >= tCompute;

  if (memoryBound) {
    notes.push(
      `Decoding is memory-bound at batch ${o.batch}. It stays that way until the batch passes this device's ridge point of about ${Math.round(ridgePoint)} FLOP per byte.`,
    );
  } else {
    notes.push(`Decoding is compute-bound at batch ${o.batch}.`);
  }

  const prefillSeconds = (o.batch * o.seq * o.flops.fwdTotal) / (o.peak * o.mfu);

  return {
    ridgePoint,
    decodeBytesPerStep,
    decodeFlopsPerStep,
    decodeSecondsPerStep: step,
    decodeTokensPerSecond: step > 0 ? o.batch / step : 0,
    memoryBound,
    prefillSeconds,
    notes,
  };
}

export interface CostOptions {
  trainFlopsPerToken: number;
  tokens: number;
  gpus: number;
  peak: number;
  mfu: number;
  pricePerHour: number;
}

export interface CostResult {
  totalFlops: number;
  gpuHours: number;
  wallClockHours: number;
  dollars: number;
  tokens: number;
  mfu: number;
}

export function analyzeCost(o: CostOptions): CostResult {
  const totalFlops = o.trainFlopsPerToken * o.tokens;
  const gpuHours = totalFlops / (o.peak * o.mfu * 3600);
  return {
    totalFlops,
    gpuHours,
    wallClockHours: gpuHours / Math.max(1, o.gpus),
    dollars: gpuHours * o.pricePerHour,
    tokens: o.tokens,
    mfu: o.mfu,
  };
}

// ---------------------------------------------------------------------------
// Scaling laws
// ---------------------------------------------------------------------------

export interface ScalingLawFit {
  name: string;
  E: number;
  A: number;
  B: number;
  alpha: number;
  beta: number;
  source: string;
}

export const CHINCHILLA_FITS: Record<string, ScalingLawFit> = {
  hoffmann: {
    name: "Hoffmann et al. 2022",
    E: 1.69,
    A: 406.4,
    B: 410.7,
    alpha: 0.34,
    beta: 0.28,
    source: "https://arxiv.org/abs/2203.15556",
  },
  epoch: {
    name: "Epoch AI replication",
    E: 1.8172,
    A: 482.01,
    B: 2085.43,
    alpha: 0.3478,
    beta: 0.3658,
    source: "https://epoch.ai/blog/chinchilla-scaling-a-replication-attempt",
  },
};

export interface ChinchillaResult {
  /** Compute-optimal token budget, about 20 tokens per parameter. */
  optimalTokens: number;
  /** Tokens per parameter for the budget actually chosen. */
  tokensPerParam: number;
  /** Tokens per *active* parameter, the meaningful ratio for sparse models. */
  tokensPerActiveParam: number;
  /** How far the chosen budget is from compute-optimal. Above 1 is over-training. */
  overTrainingRatio: number;
  /** Predicted loss under each fit, when a token budget was given. */
  predictedLoss: Record<string, number>;
  verdict: string;
}

export function analyzeChinchilla(params: {
  total: number;
  active: number;
  nonEmbedding: number;
  tokens: number;
}): ChinchillaResult {
  const N = params.nonEmbedding;
  const optimalTokens = 20 * N;
  const tokensPerParam = params.total > 0 ? params.tokens / params.total : 0;
  const tokensPerActiveParam = params.active > 0 ? params.tokens / params.active : 0;
  const overTrainingRatio = optimalTokens > 0 ? params.tokens / optimalTokens : 0;

  const predictedLoss: Record<string, number> = {};
  if (params.tokens > 0 && N > 0) {
    for (const [key, f] of Object.entries(CHINCHILLA_FITS)) {
      predictedLoss[key] = f.E + f.A / Math.pow(N, f.alpha) + f.B / Math.pow(params.tokens, f.beta);
    }
  }

  let verdict: string;
  if (params.tokens <= 0) verdict = "No token budget given.";
  else if (overTrainingRatio < 0.5) verdict = "Under-trained relative to Chinchilla: the model is larger than the data supports.";
  else if (overTrainingRatio <= 2) verdict = "Close to compute-optimal.";
  else if (overTrainingRatio <= 20) verdict = "Over-trained, which is the norm for models meant to be served.";
  else verdict = "Heavily over-trained, in the range of small models trained on very large corpora.";

  return { optimalTokens, tokensPerParam, tokensPerActiveParam, overTrainingRatio, predictedLoss, verdict };
}

export function formatHours(h: number): string {
  if (h < 1) return `${(h * 60).toFixed(1)} min`;
  if (h < 48) return `${h.toFixed(1)} h`;
  return `${(h / 24).toFixed(1)} days`;
}

export function formatDollars(d: number): string {
  if (d >= 1e6) return `$${(d / 1e6).toFixed(2)}M`;
  if (d >= 1e3) return `$${(d / 1e3).toFixed(1)}k`;
  return `$${d.toFixed(2)}`;
}
