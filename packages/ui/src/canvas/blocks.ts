/**
 * Presentation helpers for catalog blocks: colours, one-line summaries and the
 * handle-id convention the canvas uses.
 */

import type { BlockDef, NodeDef, ParamSpec, Resolved } from "@tensorcad/core";
import { CATALOG } from "@tensorcad/core";

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
function rearrangeSummary(resolved: Resolved): string {
  const from = typeof resolved.p.from === "string" ? resolved.p.from.trim() : "";
  const to = typeof resolved.p.to === "string" ? resolved.p.to.trim() : "";
  if (!from || !to) return "";
  return `${from} → ${to}`;
}

/** A short, information-dense parameter line for the node body. */
export function paramSummary(def: BlockDef | undefined, resolved: Resolved | undefined): string {
  if (!def || !resolved) return "";
  if (def.type === "rearrange") return rearrangeSummary(resolved);
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

export function kindOf(type: string): BlockKind {
  const def: BlockDef | undefined = CATALOG[type];
  return def ? def.kind : "unknown";
}

export function labelOf(node: NodeDef): string {
  return node.label ?? node.id;
}

/**
 * Element-type colours for pins, again as variables.
 *
 * Five distinctions, not fifteen: integer ids, activations at full or half
 * width, a narrow quantized type, and booleans. More than that stops being
 * readable at pin size.
 */
const DTYPE_VAR: Record<string, string> = {
  int64: "int",
  int32: "int",
  fp32: "float",
  bf16: "half",
  fp16: "half",
  fp8: "fp8",
  bool: "bool",
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
