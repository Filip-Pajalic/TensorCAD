/**
 * Shape patterns, written in einops-flavoured notation.
 *
 *   "B T D"            three dims
 *   "B T (H dh)"       last dim is the product H*dh
 *   "... D"            leading batch dims are free, last dim is D
 *   "B H T dh"         explicit four dims
 *
 * Atoms are expressions over parameters and global symbols, so `"... 2*D"` and
 * `"B T (H dh)"` both work. At most one `...` may appear in a pattern.
 *
 * A grouped dimension keeps its factors rather than collapsing to a product,
 * because code generation needs them to emit the matching view and permute.
 */

import { Sym } from "./symexpr.js";
import { evalExpr, type EvalCtx } from "./expr.js";

export type PatternAtom =
  | { kind: "ellipsis" }
  /** One dimension. `parts.length > 1` means the dimension is their product. */
  | { kind: "dims"; parts: string[] };

export interface Pattern {
  readonly src: string;
  readonly atoms: readonly PatternAtom[];
}

const patternCache = new Map<string, Pattern>();

export function parsePattern(src: string): Pattern {
  const hit = patternCache.get(src);
  if (hit) return hit;

  const atoms: PatternAtom[] = [];
  let i = 0;
  const s = src.trim();
  while (i < s.length) {
    const c = s[i];
    if (c === " " || c === "\t") {
      i++;
      continue;
    }
    if (s.startsWith("...", i)) {
      atoms.push({ kind: "ellipsis" });
      i += 3;
      continue;
    }
    if (c === "(") {
      let depth = 0;
      let j = i;
      for (; j < s.length; j++) {
        if (s[j] === "(") depth++;
        else if (s[j] === ")") {
          depth--;
          if (depth === 0) break;
        }
      }
      if (depth !== 0) throw new Error(`Unbalanced parentheses in pattern "${src}"`);
      const inner = s.slice(i + 1, j).trim();
      // Inside a group, whitespace separates the factors of one dimension.
      const parts = inner.split(/\s+/).filter(Boolean);
      if (parts.length === 0) throw new Error(`Empty group in pattern "${src}"`);
      atoms.push({ kind: "dims", parts });
      i = j + 1;
      continue;
    }
    // A bare atom runs until whitespace.
    let j = i;
    while (j < s.length && !/\s/.test(s[j])) j++;
    atoms.push({ kind: "dims", parts: [s.slice(i, j)] });
    i = j;
  }

  if (atoms.filter((a) => a.kind === "ellipsis").length > 1) {
    throw new Error(`Pattern "${src}" has more than one "..."`);
  }
  const p: Pattern = { src, atoms };
  patternCache.set(src, p);
  return p;
}

export type Shape = readonly Sym[];

export function shapeToString(shape: Shape): string {
  if (shape.length === 0) return "scalar";
  return shape.map((d) => d.toString()).join(" ");
}

/** The value of one dimension: the product of its factors. */
export function atomValue(atom: Extract<PatternAtom, { kind: "dims" }>, ctx: EvalCtx): Sym {
  let acc = Sym.con(1);
  for (const part of atom.parts) acc = acc.mul(evalExpr(part, ctx));
  return acc;
}

export function atomToString(atom: PatternAtom): string {
  if (atom.kind === "ellipsis") return "...";
  return atom.parts.length === 1 ? atom.parts[0] : `(${atom.parts.join(" ")})`;
}

export interface InstantiateResult {
  shape: Shape | null;
  errors: string[];
}

/**
 * Turn a pattern into a concrete shape. `batch` supplies the dims bound to
 * `...`; it must be provided when the pattern contains an ellipsis.
 */
export function instantiate(pattern: Pattern, ctx: EvalCtx, batch: Shape | null): InstantiateResult {
  const errors: string[] = [];
  const dims: Sym[] = [];
  for (const a of pattern.atoms) {
    if (a.kind === "ellipsis") {
      if (!batch) {
        errors.push(`Pattern "${pattern.src}" needs batch dims for "..." but none were bound`);
        return { shape: null, errors };
      }
      dims.push(...batch);
      continue;
    }
    try {
      dims.push(atomValue(a, ctx));
    } catch (e) {
      errors.push(`In pattern "${pattern.src}": ${(e as Error).message}`);
      return { shape: null, errors };
    }
  }
  return { shape: dims, errors };
}

export interface MatchResult {
  ok: boolean;
  /** Dims bound to `...`, empty when the pattern has no ellipsis. */
  batch: Shape;
  errors: string[];
}

/**
 * Check an actual shape against a pattern, binding `...`.
 *
 * `values` holds the numeric value of every concrete (design) symbol; runtime
 * symbols are absent and therefore compared symbolically.
 */
export function matchPattern(
  actual: Shape,
  pattern: Pattern,
  ctx: EvalCtx,
  values: Readonly<Record<string, number>>,
): MatchResult {
  const errors: string[] = [];
  const idx = pattern.atoms.findIndex((a) => a.kind === "ellipsis");

  const compare = (dim: Sym, atom: PatternAtom, position: number): void => {
    if (atom.kind === "ellipsis") return;
    let want: Sym;
    try {
      want = atomValue(atom, ctx);
    } catch (e) {
      errors.push(`In pattern "${pattern.src}": ${(e as Error).message}`);
      return;
    }
    if (!dim.equalsUnder(want, values)) {
      errors.push(
        `dim ${position} is ${dim.toString()} but the port expects ${atomToString(atom)} = ${want.toString()}`,
      );
    }
  };

  if (idx === -1) {
    if (actual.length !== pattern.atoms.length) {
      errors.push(`rank ${actual.length} does not match pattern "${pattern.src}" (rank ${pattern.atoms.length})`);
      return { ok: false, batch: [], errors };
    }
    actual.forEach((d, k) => compare(d, pattern.atoms[k], k));
    return { ok: errors.length === 0, batch: [], errors };
  }

  const prefix = pattern.atoms.slice(0, idx);
  const suffix = pattern.atoms.slice(idx + 1);
  if (actual.length < prefix.length + suffix.length) {
    errors.push(
      `rank ${actual.length} is too small for pattern "${pattern.src}" (needs at least ${prefix.length + suffix.length})`,
    );
    return { ok: false, batch: [], errors };
  }
  prefix.forEach((a, k) => compare(actual[k], a, k));
  suffix.forEach((a, k) => {
    const pos = actual.length - suffix.length + k;
    compare(actual[pos], a, pos);
  });
  const batch = actual.slice(prefix.length, actual.length - suffix.length);
  return { ok: errors.length === 0, batch, errors };
}
