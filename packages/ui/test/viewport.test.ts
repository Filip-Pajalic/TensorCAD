/**
 * A viewport with NaN in it is replaced, not kept.
 *
 * React Flow keeps whatever it is handed, and one NaN reaches the grid, every
 * wire and the minimap. The canvas asks `repairViewport` on every move and fits
 * again from whatever it returns.
 */

import { describe, expect, test } from "bun:test";
import { getViewportForBounds } from "@xyflow/react";
import { READABLE_ZOOM, repairViewport, standFor } from "../src/canvas/viewport.js";

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

describe("standFor: where a sheet opens", () => {
  const pane = { width: 1000, height: 800 };
  const none = { top: 0, right: 0, bottom: 0, left: 0 };

  test("a drawing that fits readably is shown whole, centred", () => {
    const stand = standFor({ x: 0, y: 0, width: 800, height: 600 }, pane, none, READABLE_ZOOM, 1.1)!;
    // It would fit at 1.25, and opens no larger than the ceiling.
    expect(stand.zoom).toBe(1.1);
    // Centred: the drawing's middle at the pane's middle.
    expect(stand.x + 400 * stand.zoom).toBeCloseTo(500);
    expect(stand.y + 300 * stand.zoom).toBeCloseTo(400);
  });

  test("a tall one opens readable, at its top, not shrunk to fit", () => {
    // Three times the pane's height: whole, it would be at a third of full size.
    const stand = standFor({ x: 100, y: 50, width: 300, height: 2400 }, pane, none, READABLE_ZOOM, 1.1)!;
    expect(stand.zoom).toBe(READABLE_ZOOM);
    // Its top at the top of the pane, and it still centred across.
    expect(stand.y + 50 * stand.zoom).toBeCloseTo(0);
    expect(stand.x + (100 + 150) * stand.zoom).toBeCloseTo(500);
  });

  test("asked for the whole of it, it shrinks as far as that takes", () => {
    const stand = standFor({ x: 0, y: 0, width: 300, height: 2400 }, pane, none, 0.3, 1.1)!;
    expect(stand.zoom).toBeCloseTo(800 / 2400);
  });

  test("margins are kept clear", () => {
    const margins = { top: 0.1, right: 0.3, bottom: 0.2, left: 0.1 };
    const stand = standFor({ x: 0, y: 0, width: 300, height: 2400 }, pane, margins, READABLE_ZOOM, 1.1)!;
    expect(stand.y).toBeCloseTo(80);
  });

  test("nothing to stand in front of is not a place to stand", () => {
    expect(standFor({ x: 0, y: 0, width: 0, height: 0 }, pane, none, READABLE_ZOOM, 1.1)).toBeNull();
    expect(standFor({ x: 0, y: 0, width: 10, height: 10 }, { width: 0, height: 0 }, none, READABLE_ZOOM, 1.1)).toBeNull();
  });
});
