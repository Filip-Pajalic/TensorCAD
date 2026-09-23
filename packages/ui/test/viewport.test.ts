/**
 * A viewport with NaN in it is replaced, not kept.
 *
 * React Flow keeps whatever it is handed, and one NaN reaches the grid, every
 * wire and the minimap. The canvas asks `repairViewport` on every move and fits
 * again from whatever it returns.
 */

import { describe, expect, test } from "bun:test";
import { getViewportForBounds } from "@xyflow/react";
import { repairViewport } from "../src/canvas/viewport.js";

describe("repairViewport", () => {
  test("leaves a good viewport alone", () => {
    expect(repairViewport({ x: 211.6, y: -55.2, zoom: 0.6 })).toBeNull();
  });

  test("replaces one that cannot be drawn", () => {
    for (const bad of [
      { x: Number.NaN, y: 0, zoom: 1 },
      { x: 0, y: Number.NaN, zoom: 1 },
      { x: 0, y: 0, zoom: Number.NaN },
      { x: 0, y: 0, zoom: 0 },
      { x: Number.POSITIVE_INFINITY, y: 0, zoom: 1 },
    ]) {
      expect(repairViewport(bad)).toEqual({ x: 0, y: 0, zoom: 1 });
    }
  });

  test("including the one a fit makes of nothing, in a sheet of no size", () => {
    // The known way in: nothing measured to fit, and nowhere to fit it.
    const fitted = getViewportForBounds({ x: 0, y: 0, width: 0, height: 0 }, 0, 0, 0.6, 1.1, 0.1);
    expect(Number.isNaN(fitted.zoom)).toBe(true);
    expect(repairViewport(fitted)).toEqual({ x: 0, y: 0, zoom: 1 });
  });
});
