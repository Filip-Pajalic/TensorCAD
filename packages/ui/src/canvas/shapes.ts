/**
 * How a shape is written on an edge, a handle or in the inspector.
 *
 * Symbolic is the default and the honest one: `B T D` says the residual stream
 * is D wide whatever D happens to be. Numeric substitutes the design symbols
 * and leaves the runtime ones (`B`, `T`) alone, which is the quickest way to
 * watch a symbol edit propagate through the whole design. English is for a
 * reader who does not yet know what the letters are.
 *
 * The first two arrive already written: the polynomial lives inside the engine,
 * and substituting a symbol into a string is not something the editor could do
 * for itself. English is built here, out of the symbolic form and the symbol
 * table, because it needs what each letter *counts* and that is a fact about
 * the design rather than about the shape.
 */

import type { Shape, SymbolTable } from "@tensor-cad/engine";

export type ShapeMode = "symbolic" | "numeric" | "english";

export const SHAPE_MODES: ShapeMode[] = ["symbolic", "numeric", "english"];

/** What the toggle prints for each, and what it says on hover. */
export const SHAPE_MODE_LABEL: Record<ShapeMode, string> = {
  symbolic: "B T D",
  numeric: "B T 4096",
  english: "plain",
};

export const SHAPE_MODE_HINT: Record<ShapeMode, string> = {
  symbolic: "The symbol names, which is what the design is written in",
  numeric: "Design symbols substituted; batch and sequence stay symbolic",
  english: "What each axis counts, and what a reshape is doing",
};

/**
 * What an axis counts, from the symbol's own documentation.
 *
 * The documentation is the right source and the symbol name is not: `T` is
 * tokens in a language model, patches in a vision transformer and one image in
 * a convnet, and each of those designs already says so in its own symbol table.
 *
 * Only countable things get a word. A width is a number and calling it "wide"
 * adds nothing — `4096 wide` reads worse than `4096`, and the reader who wants
 * to know which width it is has the key.
 */
const COUNTS: [RegExp, string, string][] = [
  [/\bclass(es)?\b/i, "class", "classes"],
  [/\bexperts?\b/i, "expert", "experts"],
  [/\bgroups?\b/i, "group", "groups"],
  [/\bheads?\b/i, "head", "heads"],
  [/\blayers?\b/i, "layer", "layers"],
  [/\bpatch(es)?\b/i, "patch", "patches"],
  [/\bimages?\b/i, "image", "images"],
  [/\btokens?\b/i, "token", "tokens"],
  [/\bbatch\b/i, "batch", "batch"],
];

/**
 * A width is never a count of the thing it is a width *of*.
 *
 * "Head dimension" mentions heads and counts none of them. Nor does "Hidden
 * width of one expert", "Sliding-window width on the local attention layers" or
 * "Width of the fully-connected layers". Without this rule `dh` rendered as
 * "128 heads", which is not merely unhelpful — it is false, and it was false on
 * every attention block of every design.
 */
const IS_WIDTH = /\b(width|dimension|resolution|rank)\b/i;

/**
 * And a count of what is inside *one* of a thing is not a count of the thing.
 * I-JEPA's `P` is "Values in one patch": 588 values, one patch.
 */
const PER_ONE = /\b(one|each|per)\s+$/i;

/**
 * What an axis counts, from the symbol's own documentation.
 *
 * The earliest count word in the sentence wins, rather than a fixed order over
 * the table: "Groups of eight layers" counts groups and "Windowed layers past
 * the last whole group" counts layers, and the only thing that tells them apart
 * is which word the sentence leads with.
 */
function noun(doc: string | undefined, n: number): string | null {
  if (!doc || IS_WIDTH.test(doc)) return null;

  let best: { at: number; one: string; many: string } | null = null;
  for (const [pattern, one, many] of COUNTS) {
    const at = doc.search(pattern);
    if (at === -1) continue;
    if (PER_ONE.test(doc.slice(0, at))) continue;
    if (!best || at < best.at) best = { at, one, many };
  }
  if (!best) return null;
  return n === 1 ? best.one : best.many;
}

const group = (n: number): string => n.toLocaleString("en-US");

/** One axis of the symbolic form, which may be a product like `H*dh`. */
function axis(term: string, symbols: SymbolTable): string {
  const factors = term.split("*").map((f) => f.trim());
  const parts = factors.map((name) => {
    const value = symbols.values[name];
    if (value === undefined || !Number.isFinite(value)) {
      // A literal (a convnet writes `B 64 55 55`) is already a number, and a
      // symbol with no value is a design that has not resolved — either way,
      // printing what is written is the only true thing left to say.
      return { text: name, value: Number(name) };
    }
    const word = noun(symbols.docs[name], value);
    return { text: word ? `${group(value)} ${word}` : group(value), value };
  });

  if (parts.length === 1) return parts[0]!.text;

  // A product of symbols is one axis holding several things: the heads folded
  // into the stream. The product is what the tensor's side actually measures,
  // so it leads, and the factors say what it is made of.
  const total = parts.reduce((n, p) => n * (Number.isFinite(p.value) ? p.value : NaN), 1);
  const factorText = parts.map((p) => p.text).join(" × ");
  return Number.isFinite(total) ? `${group(total)} (${factorText})` : factorText;
}

/** Split a shape into its axes. Patterns here are space-separated. */
const axesOf = (shape: string): string[] => shape.trim().split(/\s+/).filter(Boolean);

export function toEnglish(symbolic: string, symbols: SymbolTable): string {
  const axes = axesOf(symbolic);
  if (axes.length === 0) return symbolic;
  return axes.map((a) => axis(a, symbols)).join(" × ");
}

export function formatShape(
  shape: Shape | undefined,
  mode: ShapeMode,
  symbols?: SymbolTable,
): string | null {
  if (!shape) return null;
  if (mode === "numeric") return shape.numeric;
  // Without a symbol table there is nothing to translate against, so English
  // degrades to the symbolic form rather than to a guess.
  if (mode === "english") return symbols ? toEnglish(shape.symbolic, symbols) : shape.symbolic;
  return shape.symbolic;
}

// ---------------------------------------------------------------------------
// Reshapes
// ---------------------------------------------------------------------------

/** `B T (H dh)` split into axes, keeping a bracketed group as one. */
function einopsAxes(pattern: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let current = "";
  for (const ch of pattern) {
    if (ch === "(") {
      depth++;
      current += ch;
    } else if (ch === ")") {
      depth--;
      current += ch;
    } else if (/\s/.test(ch) && depth === 0) {
      if (current) out.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  if (current) out.push(current);
  return out;
}

const isGroup = (a: string): boolean => a.startsWith("(") && a.endsWith(")");
const inner = (a: string): string[] => a.slice(1, -1).trim().split(/\s+/).filter(Boolean);

/**
 * A reshape in English, when the reshape is one of the two this catalog
 * actually produces.
 *
 * A transformer's reshapes are a pair: one axis is taken apart into heads on
 * the way into attention and put back together on the way out. Naming those two
 * is most of what an English rendering of this drawing is worth. Anything else
 * returns null and the einops pattern stands, because a reshape nobody can name
 * is better printed as notation than as a wrong sentence.
 */
export function reshapeInEnglish(
  from: string,
  to: string,
  symbols: SymbolTable,
): string | null {
  const a = einopsAxes(from);
  const b = einopsAxes(to);

  const grouped = (axes: string[]): number => axes.findIndex(isGroup);
  const ga = grouped(a);
  const gb = grouped(b);
  // Exactly one side groups, or there is nothing to say about the difference.
  if ((ga === -1) === (gb === -1)) return null;

  const side = ga !== -1 ? a : b;
  const at = ga !== -1 ? ga : gb;
  const factors = inner(side[at]!);
  if (factors.length !== 2) return null;

  const [count, width] = factors as [string, string];
  const n = symbols.values[count];
  const w = symbols.values[width];
  if (!Number.isFinite(n) || !Number.isFinite(w)) return null;

  const word = noun(symbols.docs[count], n!) ?? "parts";
  const whole = group(n! * w!);
  const many = `${group(n!)} ${word} of ${group(w!)}`;

  // Grouped on the left means the wide axis is being taken apart; grouped on
  // the right means it is being put back together.
  return ga !== -1 ? `split ${whole} into ${many}` : `fold ${many} back into ${whole}`;
}
