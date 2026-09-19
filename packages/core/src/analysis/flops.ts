/**
 * FLOPs accounting.
 *
 * Matmul FLOPs are counted the standard way (a multiply-accumulate is 2), and
 * the sequence-dependent attention work is kept separate from the rest because
 * the familiar 2N and 6N rules exclude it. Backward is twice forward; full
 * activation recomputation adds one more forward pass.
 *
 * Reference: PaLM appendix B gives the training form 6N + 12*L*H*Q*T.
 */

import type { AnalysisCtx } from "../catalog/types.js";
import type { FlatNode, FlatResult } from "./flatten.js";

export interface FlopsResult {
  /** Matmul FLOPs per token that do not depend on sequence length. */
  fwdDense: number;
  /** Attention score and value-product FLOPs per token at the given T. */
  fwdAttention: number;
  /**
   * The same term counted as if nothing were masked, which is what a profiler
   * reports. `fwdTotalUnmasked` is the number to compare against one.
   */
  fwdAttentionUnmasked: number;
  /** fwdDense + fwdAttention. */
  fwdTotal: number;
  /** fwdDense + fwdAttentionUnmasked. */
  fwdTotalUnmasked: number;
  /** Norms, activations, RoPE, residual adds. Memory-bound, excluded above. */
  elementwise: number;
  /** Forward plus backward, per token. */
  trainPerToken: number;
  /** Share of forward FLOPs spent inside attention, 0..1. */
  attentionShare: number;
  /** 2 * non-embedding active parameters, the usual inference approximation. */
  ruleOfThumb2N: number;
  /** 6 * non-embedding active parameters, the usual training approximation. */
  ruleOfThumb6N: number;
  byPath: Record<string, number>;
  byCategory: Record<string, number>;
  errors: string[];
}

export interface FlopsOptions {
  ctx: AnalysisCtx;
  /** Extra forward pass when activations are recomputed. */
  recompute: "none" | "selective" | "full";
  /** Non-embedding active parameter count, for the rule-of-thumb comparison. */
  nonEmbeddingActive: number;
}

export function countFlops(flat: FlatResult, opts: FlopsOptions): FlopsResult {
  const res: FlopsResult = {
    fwdDense: 0,
    fwdAttention: 0,
    fwdAttentionUnmasked: 0,
    fwdTotal: 0,
    fwdTotalUnmasked: 0,
    elementwise: 0,
    trainPerToken: 0,
    attentionShare: 0,
    ruleOfThumb2N: 2 * opts.nonEmbeddingActive,
    ruleOfThumb6N: 6 * opts.nonEmbeddingActive,
    byPath: {},
    byCategory: {},
    errors: [],
  };

  for (const node of flat.nodes) {
    const per = flopsOfNode(node, opts.ctx, res.errors);
    if (!per) continue;
    // FLOPs follow the active count: a token passes through top_k experts, not
    // through all of them.
    const dense = (per.fwd ?? 0) * node.activeMultiplier;
    const seq = (per.fwdSeq ?? 0) * node.activeMultiplier;
    const seqUnmasked = (per.fwdSeqUnmasked ?? per.fwdSeq ?? 0) * node.activeMultiplier;
    const elem = (per.elementwise ?? 0) * node.activeMultiplier;

    res.fwdDense += dense;
    res.fwdAttention += seq;
    res.fwdAttentionUnmasked += seqUnmasked;
    res.elementwise += elem;

    const total = dense + seq;
    if (total > 0) {
      res.byPath[node.path] = total;
      res.byCategory[node.category] = (res.byCategory[node.category] ?? 0) + total;
    }
  }

  res.fwdTotal = res.fwdDense + res.fwdAttention;
  res.fwdTotalUnmasked = res.fwdDense + res.fwdAttentionUnmasked;
  res.attentionShare = res.fwdTotal > 0 ? res.fwdAttention / res.fwdTotal : 0;

  // Backward costs two forwards. Recomputing activations adds a third.
  const passes = opts.recompute === "full" ? 4 : 3;
  res.trainPerToken = res.fwdTotal * passes;
  return res;
}

function flopsOfNode(node: FlatNode, ctx: AnalysisCtx, errors: string[]) {
  if (!node.def.flops) return null;
  try {
    return node.def.flops(node.resolved, ctx);
  } catch (e) {
    errors.push(`${node.path}: ${(e as Error).message}`);
    return null;
  }
}

/** Human-readable FLOPs: 4.53e21, 312 TFLOP, 1.2 PFLOP. */
export function formatFlops(n: number): string {
  const units: [number, string][] = [
    [1e18, "EFLOP"],
    [1e15, "PFLOP"],
    [1e12, "TFLOP"],
    [1e9, "GFLOP"],
    [1e6, "MFLOP"],
    [1e3, "kFLOP"],
  ];
  for (const [scale, unit] of units) {
    if (Math.abs(n) >= scale) return `${(n / scale).toFixed(2)} ${unit}`;
  }
  return `${n.toFixed(0)} FLOP`;
}
