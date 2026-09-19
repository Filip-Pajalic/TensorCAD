/**
 * How a shape is written on an edge, a handle or in the inspector.
 *
 * Symbolic is the default and the honest one: `B T D` says the residual stream
 * is D wide whatever D happens to be. Numeric substitutes the design symbols
 * and leaves the runtime ones (`B`, `T`) alone, which is the quickest way to
 * watch a symbol edit propagate through the whole design.
 */

import type { Shape, SymbolTable } from "@tensorcad/core";
import { shapeToString } from "@tensorcad/core";

export type ShapeMode = "symbolic" | "numeric";

export function formatShape(
  shape: Shape | undefined,
  symbols: SymbolTable,
  mode: ShapeMode,
): string | null {
  if (!shape) return null;
  if (mode === "symbolic") return shapeToString(shape);
  return shape
    .map((dim) => {
      const n = dim.toNumber(symbols.designValues);
      return n === null || !Number.isFinite(n) ? dim.toString() : String(n);
    })
    .join(" ");
}
