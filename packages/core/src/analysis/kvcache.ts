/**
 * Inference cache accounting.
 *
 * Two kinds of state: what grows with every generated token (the KV cache of a
 * full-attention layer) and what is fixed per sequence (a sliding window's
 * bounded cache, and the recurrent state of a state-space layer).
 */

import type { AnalysisCtx } from "../catalog/types.js";
import type { FlatResult } from "./flatten.js";

export interface KvResult {
  /** Bytes added to the cache for each new token, across all layers. */
  bytesPerToken: number;
  /** Bytes held per sequence regardless of length. */
  bytesPerSequenceFixed: number;
  byPath: Record<string, number>;
  errors: string[];
}

export function countKvCache(flat: FlatResult, ctx: AnalysisCtx): KvResult {
  const res: KvResult = { bytesPerToken: 0, bytesPerSequenceFixed: 0, byPath: {}, errors: [] };

  for (const node of flat.nodes) {
    if (!node.def.stateBytes) continue;
    let s;
    try {
      s = node.def.stateBytes(node.resolved, ctx);
    } catch (e) {
      res.errors.push(`${node.path}: ${(e as Error).message}`);
      continue;
    }
    const perToken = s.perToken * node.multiplier;
    const perSeq = s.perSeq * node.multiplier;
    res.bytesPerToken += perToken;
    res.bytesPerSequenceFixed += perSeq;
    if (perToken > 0 || perSeq > 0) res.byPath[node.path] = perToken || perSeq;
  }
  return res;
}

/** Cache bytes for `sequences` concurrent sequences of `tokens` tokens each. */
export function kvBytesFor(kv: KvResult, tokens: number, sequences = 1): number {
  return (kv.bytesPerToken * tokens + kv.bytesPerSequenceFixed) * sequences;
}

export function formatBytes(n: number): string {
  const units: [number, string][] = [
    [1024 ** 5, "PiB"],
    [1024 ** 4, "TiB"],
    [1024 ** 3, "GiB"],
    [1024 ** 2, "MiB"],
    [1024, "KiB"],
  ];
  for (const [scale, unit] of units) {
    if (Math.abs(n) >= scale) return `${(n / scale).toFixed(2)} ${unit}`;
  }
  return `${Math.round(n)} B`;
}
