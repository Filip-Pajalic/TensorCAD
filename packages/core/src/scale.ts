/**
 * Scale a design to a parameter budget while keeping its proportions.
 *
 * This is what makes a local test bench possible: take an architecture you care
 * about, shrink it until it trains in ten minutes on one GPU, and compare it
 * against a baseline shrunk the same way. Because parameters grow roughly as
 * layers times width squared, balanced scaling moves both by the cube root of
 * the ratio, which keeps the depth-to-width aspect close to the original.
 */

import type { Doc, SymbolDef } from "./ir/types.js";
import { resolveSymbols } from "./ir/symbols.js";
import { countParams } from "./analysis/params.js";

export interface ScaleOptions {
  /** Parameter count to aim for. */
  targetParams: number;
  /** Symbols scaled with the width. Expressions over `D` follow automatically. */
  widthSymbols?: string[];
  /** Symbols scaled with the depth. */
  depthSymbols?: string[];
  /** Keep the residual width a multiple of this. Defaults to the head dimension. */
  widthMultiple?: number;
  /** Replace the vocabulary, for a bench that uses a smaller tokenizer. */
  vocab?: number;
  /**
   * Whether `targetParams` counts the embedding tables. At bench sizes the
   * vocabulary dominates, so "non-embedding" is usually what you mean when you
   * say "a 30M model".
   */
  targetBasis?: "total" | "non-embedding";
  /** Share the output projection with the embedding, halving the vocabulary cost. */
  tieHead?: boolean;
  /**
   * Narrow the head dimension when the scaled model would otherwise have very
   * few heads. A two-head model is not a useful bench proxy for a 32-head one.
   */
  minHeads?: number;
  /** Hold the depth fixed and move only the width. */
  keepDepth?: boolean;
  maxIterations?: number;
}

export interface ScaleResult {
  doc: Doc;
  /** Parameter count actually reached. */
  achieved: number;
  target: number;
  /** Symbol changes, as `name: before -> after`. */
  changes: Record<string, { from: number; to: number }>;
  notes: string[];
}

const DEFAULT_WIDTH = ["D", "F", "Fe"];
const DEFAULT_DEPTH = ["L"];

function literal(doc: Doc, name: string): number | null {
  const def = doc.symbols[name];
  if (typeof def === "number") return def;
  if (def && typeof def === "object" && def.kind === "design" && typeof def.value === "number") {
    return def.value;
  }
  return null;
}

function setLiteral(doc: Doc, name: string, value: number): void {
  const def = doc.symbols[name];
  if (def && typeof def === "object" && def.kind === "design") {
    doc.symbols[name] = { ...def, value } as SymbolDef;
  } else {
    doc.symbols[name] = value;
  }
}

/** Round to the nearest positive multiple of `m`. */
function roundTo(value: number, m: number): number {
  return Math.max(m, Math.round(value / m) * m);
}

function applyScale(base: Doc, factor: number, opts: ScaleOptions): { doc: Doc; notes: string[] } {
  const doc = structuredClone(base);
  const notes: string[] = [];

  const widthNames = opts.widthSymbols ?? DEFAULT_WIDTH;
  const depthNames = opts.depthSymbols ?? DEFAULT_DEPTH;

  // Parameters grow as layers times width squared, so a balanced move takes the
  // cube root in each direction.
  const widthFactor = opts.keepDepth ? Math.sqrt(factor) : Math.cbrt(factor);
  const depthFactor = opts.keepDepth ? 1 : Math.cbrt(factor);

  let headDim = literal(doc, "dh");
  const originalD = literal(doc, "D");

  // A very narrow model with a wide head dimension ends up with two or three
  // heads, which is a poor proxy for the original. Narrowing the head instead
  // keeps the head count reasonable and the width granularity finer.
  const minHeads = opts.minHeads ?? 4;
  if (headDim !== null && originalD !== null && opts.widthMultiple === undefined) {
    const wouldBe = Math.round((originalD * widthFactor) / headDim);
    if (wouldBe < minHeads && headDim > 64) {
      headDim = 64;
      setLiteral(doc, "dh", headDim);
      notes.push(`Narrowed the head dimension to 64 so the scaled design keeps at least ${minHeads} heads.`);
    }
  }

  const widthMultiple = opts.widthMultiple ?? headDim ?? 64;

  let newD: number | null = null;
  if (originalD !== null) {
    newD = roundTo(originalD * widthFactor, widthMultiple);
    setLiteral(doc, "D", newD);
  }

  // Heads follow the width so the head dimension stays kernel-friendly.
  if (newD !== null && headDim !== null) {
    const heads = Math.max(1, Math.round(newD / headDim));
    setLiteral(doc, "H", heads);
    const originalH = literal(base, "H");
    const originalKv = literal(base, "Hkv");
    if (originalH !== null && originalKv !== null) {
      const ratio = originalH / originalKv;
      let kv = Math.max(1, Math.round(heads / ratio));
      while (kv > 1 && heads % kv !== 0) kv--;
      setLiteral(doc, "Hkv", kv);
      if (heads % kv !== 0) notes.push(`Could not keep the ${ratio}:1 query-to-key ratio at ${heads} heads.`);
    }
  }

  // Other width symbols move with the width, but only when they are literals:
  // an expression over D already follows it.
  const widthRatio = newD !== null && originalD !== null ? newD / originalD : widthFactor;
  for (const name of widthNames) {
    if (name === "D") continue;
    const v = literal(doc, name);
    if (v === null) continue;
    setLiteral(doc, name, roundTo(v * widthRatio, 64));
  }

  for (const name of depthNames) {
    const v = literal(doc, name);
    if (v === null) continue;
    setLiteral(doc, name, Math.max(1, Math.round(v * depthFactor)));
  }

  // A design with leading dense layers keeps at least one of each kind.
  const ld = literal(doc, "Ld");
  const l = literal(doc, "L");
  if (ld !== null && l !== null && ld >= l) {
    setLiteral(doc, "Ld", Math.max(1, l - 1));
    notes.push("Reduced the leading dense layers so at least one sparse layer remains.");
  }

  if (opts.vocab !== undefined) setLiteral(doc, "V", opts.vocab);

  if (opts.tieHead !== undefined) {
    const head = doc.graph.nodes.find((node) => node.type === "lm_head");
    if (head) {
      head.params = { ...head.params, tied: opts.tieHead };
    }
  }

  return { doc, notes };
}

export function scaleDesign(base: Doc, opts: ScaleOptions): ScaleResult {
  const basis = opts.targetBasis ?? "total";
  const measure = (doc: Doc): number => {
    const p = countParams(doc, resolveSymbols(doc));
    return basis === "total" ? p.total : p.nonEmbedding;
  };

  const startParams = measure(base);
  const notes: string[] = [];

  if (opts.targetParams <= 0) throw new Error("targetParams must be positive");

  // Binary search on the scale factor. The relationship is monotone but not
  // smooth, because widths are rounded to kernel-friendly multiples.
  let lo = 1e-8;
  let hi = Math.max(4, (opts.targetParams / Math.max(1, startParams)) * 4);
  let best = applyScale(base, 1, opts);
  let bestParams = measure(best.doc);
  let bestError = Math.abs(bestParams - opts.targetParams);

  const iterations = opts.maxIterations ?? 48;
  for (let i = 0; i < iterations; i++) {
    const mid = Math.sqrt(lo * hi);
    const candidate = applyScale(base, mid, opts);
    const params = measure(candidate.doc);
    const error = Math.abs(params - opts.targetParams);
    if (error < bestError) {
      best = candidate;
      bestParams = params;
      bestError = error;
    }
    if (params > opts.targetParams) hi = mid;
    else lo = mid;
    if (hi / lo < 1.0001) break;
  }

  notes.push(...best.notes);
  const relative = Math.abs(bestParams - opts.targetParams) / opts.targetParams;
  if (relative > 0.1) {
    notes.push(
      `The closest reachable size is ${(relative * 100).toFixed(0)}% from the target. ` +
        "Rounding the width to whole heads limits how finely the size can be tuned.",
    );
  }

  const finalCounts = countParams(best.doc, resolveSymbols(best.doc));
  const embeddingShare = finalCounts.total > 0 ? finalCounts.embedding / finalCounts.total : 0;
  if (embeddingShare > 0.4) {
    notes.push(
      `The embedding table is ${(embeddingShare * 100).toFixed(0)}% of this design's weights. ` +
        "Shrink the vocabulary or tie the output projection if you want the comparison to be about the transformer.",
    );
  }

  const changes: Record<string, { from: number; to: number }> = {};
  for (const name of Object.keys(base.symbols)) {
    const from = literal(base, name);
    const to = literal(best.doc, name);
    if (from !== null && to !== null && from !== to) changes[name] = { from, to };
  }

  const doc = best.doc;
  doc.meta = {
    ...doc.meta,
    name: `${base.meta.name}-${formatShort(finalCounts.total)}`,
    notes:
      `Scaled down from ${base.meta.name} (${formatShort(startParams)} parameters) for a local bench run. ` +
      (base.meta.notes ?? ""),
    // The published figure belongs to the original, not to this.
    published: undefined,
  };

  return { doc, achieved: bestParams, target: opts.targetParams, changes, notes };
}

function formatShort(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}b`;
  if (n >= 1e6) return `${Math.round(n / 1e6)}m`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return String(n);
}
