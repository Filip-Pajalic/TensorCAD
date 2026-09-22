/**
 * Turning a catalog entry into a node the document can hold.
 *
 * Parameters that have a default are filled in; parameters that are required
 * but have no sensible default are deliberately left out, so the Inspector
 * shows them as missing and the analysis reports them.
 */

import { repeatSkeleton } from "./ops.js";
import type { Doc, NodeDef, ParamSpec, ParamValue } from "@tensor-cad/engine";
import { blockDef, type BlockDef } from "../engine.js";

/** Short, readable node ids instead of repeating the full type name. */
const BASE_ID: Record<string, string> = {
  boundary_in: "_in",
  boundary_out: "_out",
  embedding: "embed",
  pos_embedding: "pos",
  lm_head: "head",
  linear: "lin",
  activation: "act",
  rearrange: "reshape",
  rmsnorm: "norm",
  layernorm: "norm",
  gqa_attention: "attn",
  gated_mlp: "mlp",
  dense_mlp: "mlp",
  transformer_block: "block",
  repeat: "layers",
  sdpa: "attn_core",
  input: "tokens",
  output: "logits",
};

export function baseIdFor(type: string): string {
  return BASE_ID[type] ?? type;
}

export function defaultParams(def: BlockDef): Record<string, ParamValue> {
  const out: Record<string, ParamValue> = {};
  for (const [key, spec] of Object.entries((def.params ?? {}) as Record<string, ParamSpec>)) {
    const fallback = (spec as { default?: ParamValue }).default;
    if (fallback !== undefined) out[key] = structuredClone(fallback) as ParamValue;
  }
  return out;
}

/**
 * A new node of this type. Required parameters without a catalog default stay
 * absent on purpose: the Inspector marks them and the analysis reports them,
 * which is more honest than inventing a value.
 */
export function newNodeFor(type: string, doc: Doc | undefined): NodeDef | null {
  // The document's catalog, not the built-in one (invariant 1). A design's own
  // block is listed in the palette, so resolving it here against the built-ins
  // meant dragging one onto the sheet did nothing at all, silently: `null` is
  // how the drop handler is told there is no such block.
  const def = blockDef(type, doc);
  if (!def) return null;
  if (def.kind === "container") return repeatSkeleton(baseIdFor(type));
  return { id: baseIdFor(type), type, params: defaultParams(def) };
}
