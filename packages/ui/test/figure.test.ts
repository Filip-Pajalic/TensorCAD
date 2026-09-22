/**
 * Figure mode: the drawing a paper would print.
 *
 * A reshape is real, necessary, and the thing that makes multi-head attention
 * multi-head. No published figure draws one, because it moves no data and costs
 * no parameters — the tensor that comes out holds exactly the numbers that went
 * in. So the mode leaves them out.
 *
 * Two things must hold, and both are ways a view that hides parts goes wrong:
 * the signal must still run end to end, and nothing the design rules found may
 * disappear with the block it was found on.
 */

import { describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();

const { getPreset, catalogOf } = await import("../src/engine.js");
const { unfold } = await import("../src/state/unfold.js");
const { resolveLevel } = await import("../src/state/level.js");
const { derive } = await import("../src/state/derive.js");
const { DEFAULT_OPERATING } = await import("../src/state/operating.js");

import type { Doc } from "@tensor-cad/engine";

function viewOf(doc: Doc, depth: number, figure: boolean) {
  const derived = derive(doc, DEFAULT_OPERATING);
  const level = resolveLevel(doc, [], derived);
  return unfold(doc, level, derived, depth, figure);
}

const doc = getPreset("llama-3-8b");

describe("what a figure leaves out", () => {
  test("the reshapes are drawn when it is not a figure", () => {
    const plain = viewOf(doc, 5, false);
    const types = plain.nodes.map((n) => n.def?.type);
    expect(types).toContain("rearrange");
  });

  test("and left out when it is", () => {
    const fig = viewOf(doc, 5, true);
    expect(fig.nodes.map((n) => n.def?.type)).not.toContain("rearrange");
  });

  test("the signal still runs through where they were", () => {
    const fig = viewOf(doc, 5, true);
    const drawn = new Set(fig.nodes.map((n) => n.path));

    // Every wire the figure draws lands on parts the figure draws. A trace that
    // stopped at a block it had hidden would leave a wire pointing at nothing.
    for (const e of fig.edges) {
      expect({ edge: e.id, source: drawn.has(e.source) }).toEqual({ edge: e.id, source: true });
      expect({ edge: e.id, target: drawn.has(e.target) }).toEqual({ edge: e.id, target: true });
    }

    // And the parts on either side of a dropped reshape are joined directly.
    // `q_proj -> q_heads -> rope_q`: the reshape goes and the projection now
    // feeds the rotary block. `attn -> o_merge -> o_proj` likewise.
    const from = (needle: string): string[] =>
      fig.edges.filter((e) => e.source.endsWith(needle)).map((e) => e.target);
    expect(from("/q_proj").some((t) => t.endsWith("/rope_q"))).toBe(true);
    expect(from("/attn").some((t) => t.endsWith("/o_proj"))).toBe(true);
    // And nothing is drawn leaving a block that is no longer drawn.
    expect(from("/o_merge")).toEqual([]);
    expect(from("/q_heads")).toEqual([]);
  });

  test("nothing is stranded: every drawn part but the ends is wired", () => {
    const fig = viewOf(doc, 5, true);
    const wired = new Set<string>();
    for (const e of fig.edges) {
      wired.add(e.source);
      wired.add(e.target);
    }
    const stranded = fig.nodes
      .filter((n) => !n.frame && !wired.has(n.path))
      .map((n) => n.path);
    expect(stranded).toEqual([]);
  });

  test("a left-out block reports where its findings went", () => {
    const fig = viewOf(doc, 5, true);
    // Every block the figure dropped names the drawn part that took its place,
    // which is what lets the canvas show a finding that would otherwise vanish.
    expect(fig.absorbed.size).toBeGreaterThan(0);
    const drawn = new Set(fig.nodes.map((n) => n.path));
    for (const [hidden, landed] of fig.absorbed) {
      expect({ hidden, drawn: drawn.has(hidden) }).toEqual({ hidden, drawn: false });
      expect({ hidden, landed: drawn.has(landed) }).toEqual({ hidden, landed: true });
    }
  });

  test("it says nothing was absorbed when it is not a figure", () => {
    expect(viewOf(doc, 5, false).absorbed.size).toBe(0);
  });
});

describe("the names it hides are names the catalog has", () => {
  test("every type in the plumbing list is a real block", () => {
    // The failure this guards against is the one `SIDEWAYS_IN` had: a table of
    // type strings written at a distance from the catalog, two of whose five
    // rows matched no block type and had therefore never done anything.
    const cat = catalogOf(doc);
    for (const type of ["rearrange", "expand_heads"]) {
      expect({ type, known: type in cat }).toEqual({ type, known: true });
      expect({ type, category: cat[type]?.category }).toEqual({ type, category: "shape" });
    }
  });
});
