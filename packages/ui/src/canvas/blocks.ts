import type { NodeDef, ParamSpec, Resolved, SymbolTable } from "@tensor-cad/engine";
import { reshapeInEnglish, type ShapeMode } from "./shapes.js";
import type { BlockDef } from "../engine.js";
/**
 * Presentation helpers for catalog blocks: colours, one-line summaries and the
 * handle-id convention the canvas uses.
 */


/**
 * Part colours.
 *
 * These resolve to CSS variables rather than literals, so a category keeps its
 * meaning when the theme changes and nothing here has to know which theme is
 * active. `theme.css` holds the actual values for both.
 */
export interface PartColor {
  fill: string;
  edge: string;
  /** Text on the fill. Only a dark-filled part needs its own. */
  text: string;
}

/** Categories that have their own colour. Anything else falls back. */
const KNOWN_CATEGORIES = new Set([
  "io",
  "embedding",
  "linear",
  "head",
  "norm",
  "elementwise",
  "shape",
  "position",
  "attention",
  "mlp",
  "block",
  "moe",
  "ssm",
  "container",
]);

export function partColor(category: string | undefined): PartColor {
  const key = category && KNOWN_CATEGORIES.has(category) ? category : "unknown";
  return {
    fill: `var(--part-${key}-fill)`,
    edge: `var(--part-${key}-edge)`,
    // Most categories declare no text colour, so the fallback carries them.
    text: `var(--part-${key}-text, var(--part-text))`,
  };
}

/**
 * What a category is called, in the order the palette lists them.
 *
 * The engine's category is a key — `ssm`, `moe`, `io` — and the palette is
 * where somebody new meets it. Whole layers first, then what a layer is made
 * of, then the plumbing, because that is the order a design is built in.
 */
const CATEGORIES: [key: string, name: string][] = [
  ["block", "Layers"],
  ["attention", "Attention"],
  ["mlp", "Feed-forward"],
  ["moe", "Mixture of experts"],
  ["ssm", "State space"],
  ["norm", "Normalization"],
  ["embedding", "Embeddings"],
  ["position", "Positions"],
  ["head", "Output heads"],
  ["linear", "Linear"],
  ["elementwise", "Elementwise"],
  ["shape", "Reshaping"],
  ["container", "Containers"],
  ["io", "Inputs and outputs"],
];

export function categoryName(category: string | undefined): string {
  return CATEGORIES.find(([key]) => key === category)?.[1] ?? category ?? "Other";
}

/** Where a category comes in the palette; one it does not know goes last. */
export function categoryRank(category: string): number {
  const at = CATEGORIES.findIndex(([key]) => key === category);
  return at < 0 ? CATEGORIES.length : at;
}

/** The outline colour, for swatches and the minimap. */
export function categoryColor(category: string | undefined): string {
  return partColor(category).edge;
}

const SHORT: Record<string, string> = {
  d_model: "D",
  dim: "D",
  heads: "heads",
  kv_heads: "kv",
  head_dim: "dh",
  ffn_hidden: "ffn",
  hidden: "ffn",
  in_features: "in",
  out_features: "out",
  vocab: "vocab",
  max_seq: "max_seq",
  count: "×",
  kind: "",
  act: "",
  norm: "",
  mlp: "",
  dtype: "",
  shape: "",
  from: "",
  to: "",
  theta: "theta",
  window: "window",
  eps: "eps",
};

/** Which parameters say the most about a block, in the order to show them. */
const SUMMARY_KEYS: Record<string, string[]> = {
  input: ["shape", "dtype"],
  output: [],
  boundary_in: [],
  boundary_out: [],
  embedding: ["vocab", "dim"],
  pos_embedding: ["max_seq", "dim"],
  linear: ["in_features", "out_features", "bias"],
  lm_head: ["vocab", "dim", "tied"],
  rmsnorm: ["dim"],
  layernorm: ["dim", "bias"],
  activation: ["kind", "dim"],
  add: ["dim"],
  mul: ["dim"],
  rearrange: ["from", "to"],
  rope: ["heads", "head_dim", "theta"],
  sdpa: ["heads", "kv_heads", "head_dim", "causal"],
  gqa_attention: ["heads", "kv_heads", "head_dim"],
  mha_attention: ["heads", "head_dim"],
  gated_mlp: ["hidden", "act"],
  dense_mlp: ["hidden", "act"],
  transformer_block: ["d_model", "heads", "ffn_hidden"],
  repeat: ["count"],
};

function fmtValue(v: unknown): string | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)));
  if (typeof v === "string") return v;
  if (typeof v === "boolean") return v ? "yes" : null;
  return null;
}

/**
 * A reshape, written the way einops writes one.
 *
 * Two patterns separated by a space read as one long unparseable string; an
 * arrow between them reads as what it is. Groups are kept, because in a reshape
 * the grouping is the whole point: `B H T dh -> B T (H dh)` says the heads are
 * being folded back into the stream, and dropping the brackets would lose that.
 */
function rearrangeSummary(
  resolved: Resolved,
  mode: ShapeMode,
  symbols: SymbolTable | undefined,
): string {
  const from = typeof resolved.p.from === "string" ? resolved.p.from.trim() : "";
  const to = typeof resolved.p.to === "string" ? resolved.p.to.trim() : "";
  if (!from || !to) return "";
  if (mode === "english" && symbols) {
    // Named where it can be named, and the pattern where it cannot. A reshape
    // this catalog does not produce is better shown as notation than as a
    // sentence that might be describing something else.
    const said = reshapeInEnglish(from, to, symbols);
    if (said) return said;
  }
  return `${from} → ${to}`;
}

/** A short, information-dense parameter line for the node body. */
export function paramSummary(
  def: BlockDef | undefined,
  resolved: Resolved | undefined,
  mode: ShapeMode = "symbolic",
  symbols?: SymbolTable,
): string {
  if (!def || !resolved) return "";
  if (def.type === "rearrange") return rearrangeSummary(resolved, mode, symbols);
  const specs = def.params as Record<string, ParamSpec>;
  const wanted = SUMMARY_KEYS[def.type] ?? Object.keys(specs ?? {}).slice(0, 3);
  const parts: string[] = [];
  for (const key of wanted) {
    if (parts.length >= 3) break;
    const value = resolved.p[key];
    const text = fmtValue(value);
    if (text === null || text === "") continue;
    const spec = specs?.[key];
    if (spec && spec.type === "bool") {
      if (value === true) parts.push(SHORT[key] ?? key);
      continue;
    }
    const label = SHORT[key] ?? key;
    parts.push(label === "×" ? `×${text}` : label ? `${label} ${text}` : text);
  }
  return parts.join(" · ");
}

export type BlockKind = "primitive" | "composite" | "container" | "unknown";

/**
 * Which of the three kinds a block is.
 *
 * It takes the definition rather than a type to look up, and that is the whole
 * point: looking one up means choosing a catalog, and choosing the built-in one
 * answers "unknown" for a design's own block — which is the same shape as a
 * real answer and reads as one (invariant 1). Every caller has already had to
 * resolve the block in order to draw it, so there is nothing to look up here.
 */
export function kindOf(def: BlockDef | undefined): BlockKind {
  return def ? def.kind : "unknown";
}

export function labelOf(node: NodeDef): string {
  return node.label ?? node.id;
}

/**
 * What to print for a block's type.
 *
 * The catalog carries a name — the phrase a published figure would use — beside
 * the identifier the engine dispatches on. The drawing wants the first and a
 * path, an MCP call and the inspector want the second, so both are kept and
 * this decides which one a label gets.
 *
 * The identifier is the fallback rather than a blank, because a block the
 * catalog has never heard of still has to be drawn as something, and its type
 * is the only true thing left to say about it.
 */
export function typeName(def: BlockDef | undefined, type: string): string {
  return def?.docs.name || type;
}

/**
 * Element-type colours for pins, again as variables.
 *
 * Five distinctions, not fifteen: integer ids, activations at full or half
 * width, a narrow quantized type, and booleans. More than that stops being
 * readable at pin size.
 */
const DTYPE_VAR: Record<string, string> = {
  // The kinds a port declares.
  int: "int",
  bool: "bool",
  real: "float",
  // And the concrete names, for a port that pins one down and for the input
  // block, whose `dtype` parameter is a real choice about the data.
  int64: "int",
  int32: "int",
  fp32: "float",
  float: "float",
  bf16: "half",
  fp16: "half",
  half: "half",
  fp8: "fp8",
};

export function dtypeColor(dtype: string | null | undefined): string {
  const key = (dtype && DTYPE_VAR[dtype]) || "unknown";
  return `var(--dtype-${key})`;
}

/**
 * Elementwise operators drawn as a circled mark on the line rather than as a
 * labelled box. This is how every architecture figure draws a residual sum, and
 * it is the difference between a drawing that reads as a transformer and one
 * that reads as a flowchart of boxes.
 */
const GLYPHS: Record<string, string> = {
  add: "⊕",
  mul: "⊗",
};

export function glyphFor(type: string): string | null {
  return GLYPHS[type] ?? null;
}
