/**
 * How a shape is written on an edge, a handle or in the inspector.
 *
 * Symbolic is the default and the honest one: `B T D` says the residual stream
 * is D wide whatever D happens to be. Numeric substitutes the design symbols
 * and leaves the runtime ones (`B`, `T`) alone, which is the quickest way to
 * watch a symbol edit propagate through the whole design.
 *
 * Both forms arrive already written. The polynomial lives inside the engine,
 * and substituting a symbol into a string is not something the editor could do
 * for itself.
 */

import type { Shape } from "@tensor-cad/engine";

export type ShapeMode = "symbolic" | "numeric";

export function formatShape(shape: Shape | undefined, mode: ShapeMode): string | null {
  if (!shape) return null;
  return mode === "symbolic" ? shape.symbolic : shape.numeric;
}
