/**
 * Parameter counting.
 *
 * `total` is every trainable weight. `active` is what a single token actually
 * uses, which differs from `total` only for mixture-of-experts designs.
 * `nonEmbedding` is the number that the 2N/6N FLOPs rules are written against.
 */

import type { Doc, SymbolTable } from "../ir/types.js";
import { flatten, type FlatNode, type FlatResult } from "./flatten.js";

export interface ParamsResult {
  /** Every trainable weight. */
  total: number;
  /** Weights a single token actually uses. Differs from total only for MoE. */
  active: number;
  embedding: number;
  head: number;
  nonEmbedding: number;
  /** Active parameters excluding the embedding table: the N in the 2N rule. */
  nonEmbeddingActive: number;
  byPath: Record<string, number>;
  byCategory: Record<string, number>;
  byType: Record<string, number>;
  errors: string[];
}

const EMBEDDING_TYPES = new Set(["embedding", "pos_embedding"]);

export function countParams(doc: Doc, symbols: SymbolTable, flat?: FlatResult): ParamsResult {
  const f = flat ?? flatten(doc, symbols);
  const res: ParamsResult = {
    total: 0,
    active: 0,
    embedding: 0,
    head: 0,
    nonEmbedding: 0,
    nonEmbeddingActive: 0,
    byPath: {},
    byCategory: {},
    byType: {},
    errors: [...f.errors],
  };

  for (const node of f.nodes) {
    const per = paramsOfNode(node, res.errors);
    if (per === null) continue;
    const total = per * node.multiplier;
    const active = per * node.activeMultiplier;
    if (total === 0) continue;

    res.total += total;
    res.active += active;
    res.byPath[node.path] = total;
    res.byCategory[node.category] = (res.byCategory[node.category] ?? 0) + total;
    res.byType[node.type] = (res.byType[node.type] ?? 0) + total;

    if (EMBEDDING_TYPES.has(node.type)) res.embedding += total;
    else if (node.type === "lm_head") res.head += total;
  }

  res.nonEmbedding = res.total - res.embedding;
  res.nonEmbeddingActive = res.active - res.embedding;
  return res;
}

function paramsOfNode(node: FlatNode, errors: string[]): number | null {
  if (!node.def.paramCount) return 0;
  try {
    const n = node.def.paramCount(node.resolved);
    if (!Number.isFinite(n)) {
      errors.push(`${node.path}: parameter count is not a finite number`);
      return null;
    }
    return n;
  } catch (e) {
    errors.push(`${node.path}: ${(e as Error).message}`);
    return null;
  }
}

/** Human-readable parameter count: 8.03B, 124.4M, 12.9K. */
export function formatCount(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}
