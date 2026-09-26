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

type Values = Record<string, unknown>;

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** A number as a figure prints one: 14,336 and 0.02, never 1.4336e4. */
function fig(v: number): string {
  return Number.isInteger(v) ? v.toLocaleString("en-US") : String(Number(v.toPrecision(4)));
}

/** Two numbers as a figure pairs them: "128,256 × 4,096", "4,096 → 1,024". */
function pair(a: unknown, b: unknown, by: string): string | null {
  const x = num(a);
  const y = num(b);
  return x !== null && y !== null ? `${fig(x)} ${by} ${fig(y)}` : null;
}

/** A number with its word after it: "128 experts". */
function count(v: unknown, what: string): string | null {
  const x = num(v);
  return x === null ? null : `${fig(x)} ${what}`;
}

/** A number with its word before it: "width 4,096". */
function named(what: string, v: unknown): string | null {
  const x = num(v);
  return x === null ? null : `${what} ${fig(x)}`;
}

/** Heads, and their width when there is one: "32 heads × 128". */
function heads(p: Values): string | null {
  const h = num(p.heads);
  if (h === null) return null;
  const dh = num(p.head_dim);
  return dh === null ? `${fig(h)} heads` : `${fig(h)} heads × ${fig(dh)}`;
}

/** Key/value heads, when fewer than the query heads: "8 kv". */
function kv(p: Values): string | null {
  const h = num(p.heads);
  const k = num(p.kv_heads);
  return h !== null && k !== null && k > 0 && k < h ? `${fig(k)} kv` : null;
}

type Phrase = (p: Values, word: (key: string) => string | null) => (string | null)[];

/**
 * What each kind of block says on its face, as a phrase.
 *
 * The line used to be the parameters' code names with their values — `D 48 ·
 * dh 16 · ffn 192` — which is what a document writes and not what anybody
 * says. A figure says "32 heads × 128" and "128,256 × 4,096", so the common
 * blocks say that. Short, because the line is one line of a part 216 pixels
 * wide and everything after the twenty-fifth character or so is an ellipsis.
 * A block with nothing here falls back to its parameters' plain labels.
 */
const PHRASES: Record<string, Phrase> = {
  input: (p) => [typeof p.shape === "string" ? p.shape : null, typeof p.dtype === "string" ? p.dtype : null],
  output: () => [],
  boundary_in: () => [],
  boundary_out: () => [],
  add: () => [],
  mul: () => [],
  embedding: (p) => [pair(p.vocab, p.dim, "×")],
  pos_embedding: (p) => [pair(p.max_seq, p.dim, "×")],
  learned_tokens: (p) => [pair(p.count, p.dim, "×")],
  linear: (p) => [pair(p.in_features, p.out_features, "→"), p.bias === true ? "bias" : null],
  lm_head: (p) => [pair(p.dim, p.vocab, "→"), p.tied === true ? "tied" : null],
  conv2d: (p) => [pair(p.in_channels, p.out_channels, "→"), pair(p.kernel, p.kernel, "×")],
  rmsnorm: (p) => [named("width", p.dim)],
  layernorm: (p) => [named("width", p.dim), p.bias === true ? "bias" : null],
  activation: (_p, word) => [word("kind")],
  rope: (p) => [heads(p), named("θ", p.theta)],
  sdpa: (p) => [heads(p), kv(p), num(p.window) ? named("window", p.window) : p.causal === true ? "causal" : null],
  gqa_attention: (p) => [heads(p), kv(p), num(p.window) ? named("window", p.window) : null],
  cross_attention: (p) => [heads(p), kv(p)],
  eager_attention: (p) => [heads(p), kv(p), "written out"],
  diff_attention: (p) => [heads(p), "differential"],
  mla_attention: (p) => [named("latent", p.kv_lora), count(p.heads, "heads")],
  gated_mlp: (p, word) => [count(p.hidden, "wide"), word("act")],
  dense_mlp: (p, word) => [count(p.hidden, "wide"), word("act")],
  moe_layer: (p) => [count(p.experts, "experts"), named("top", p.top_k)],
  moe_experts: (p) => [count(p.experts, "experts"), named("top", p.top_k)],
  topk_router: (p) => [count(p.experts, "experts"), named("top", p.top_k)],
  transformer_block: (p) => [
    count(p.heads, "heads"),
    p.mlp === "moe" ? count(p.experts, "experts") : named("ffn", p.ffn_hidden),
  ],
  mamba_block: (p) => [named("state", p.state), count(p.expand, "× wide")],
  mamba2_block: (p) => [named("state", p.state), count(p.expand, "× wide")],
  gated_deltanet_block: (p) => [heads(p)],
  gated_delta_scan: (p) => [heads(p)],
  selective_scan: (p) => [named("state", p.state)],
  ssd_scan: (p) => [heads(p), named("state", p.state)],
  position_bias: (p) => [count(p.buckets, "buckets"), count(p.heads, "heads")],
  kv_latent_cache: (p) => [named("width", p.dim)],
  mtp_head: (p) => [count(p.by, "ahead")],
  repeat: (p) => [num(p.count) === null ? null : `×${fig(num(p.count)!)}`],
};

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

/**
 * How many characters of summary a part has room for: its line is 10-pixel
 * monospace, about six pixels a character, across what the parameter count
 * beside it leaves. Measured on the sheet, not guessed, because a phrase cut
 * off with an ellipsis says less than a shorter one that is whole.
 */
export const SUMMARY_ROOM = { beside: 21, alone: 28 };

/** Parts of a phrase, in order of importance, as many as fit in `room`. */
function fitting(parts: string[], room: number): string {
  let line = "";
  for (const part of parts) {
    const next = line ? `${line} · ${part}` : part;
    // The first part always, cut off if it must: it is what the block is.
    if (line && next.length > room) break;
    line = next;
  }
  return line;
}

/**
 * A short, information-dense line for the node body, in words.
 *
 * `room` is in characters; a phrase drops its least important parts rather
 * than running into an ellipsis.
 */
export function paramSummary(
  def: BlockDef | undefined,
  resolved: Resolved | undefined,
  mode: ShapeMode = "symbolic",
  symbols?: SymbolTable,
  room: number = SUMMARY_ROOM.alone,
): string {
  if (!def || !resolved) return "";
  if (def.type === "rearrange") return rearrangeSummary(resolved, mode, symbols);
  const specs = (def.params ?? {}) as Record<string, ParamSpec>;
  const p = resolved.p as Values;
  // An enum's value as a paper writes it: SiLU, not silu.
  const word = (key: string): string | null => {
    const v = p[key];
    if (typeof v !== "string" || v === "") return null;
    return specs[key]?.valueLabels?.[v] ?? v;
  };
  const phrase = PHRASES[def.type];
  if (phrase) {
    return fitting(
      phrase(p, word).filter((x): x is string => !!x).slice(0, 3),
      room,
    );
  }
  // Anything else says its first few parameters, by what they are called.
  const parts: string[] = [];
  for (const key of def.paramOrder ?? Object.keys(specs)) {
    if (parts.length >= 3) break;
    const spec = specs[key];
    if (!spec || spec.advanced) continue;
    const v = p[key];
    const label = (spec.label ?? key).toLowerCase();
    if (typeof v === "boolean") {
      if (v) parts.push(label);
    } else if (typeof v === "number" && Number.isFinite(v)) {
      parts.push(`${label} ${fig(v)}`);
    } else if (typeof v === "string" && v !== "" && spec.type === "enum") {
      parts.push(word(key)!);
    }
  }
  return fitting(parts, room);
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
