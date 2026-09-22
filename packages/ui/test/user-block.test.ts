/**
 * A block the document defines for itself, through the editor's own doors.
 *
 * Invariant 1 says to resolve a type through `catalogOf(doc)` and never the
 * bare `CATALOG`, and the reason is exactly this: the built-in catalog is a
 * module-level record filled once at load, so a lookup against it answers
 * "unknown" for a design's own block — and every one of these functions then
 * fails *quietly*, returning a value that is the same shape as a real answer.
 *
 * The palette lists a design's own blocks, so each of these is reachable by
 * dragging one onto the sheet.
 */

import { describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();

const { catalogOf, catalogByCategory } = await import("../src/engine.js");
const { newNodeFor } = await import("../src/state/addBlock.js");
const { kindOf } = await import("../src/canvas/blocks.js");
const ops = await import("../src/state/ops.js");
const { withBlock } = await import("../src/state/blocks.js");

import type { Doc, UserBlockDef } from "@tensor-cad/engine";

const def: UserBlockDef = {
  type: "half_mlp",
  category: "mlp",
  params: {
    D: { type: "int", default: 512, min: 1, doc: "Width in" },
    F: { type: "int", default: 2048, min: 1, doc: "Width in the middle" },
  },
  ports: { in: { x: "... D" }, out: { y: "... F" } },
  graph: {
    nodes: [
      { id: "up", type: "linear", params: { in_features: "$D", out_features: "$F", bias: false } },
    ],
    edges: [
      ["_in:x", "up:x"],
      ["up:y", "_out:y"],
    ],
  },
  docs: { summary: "Half of a feed-forward, for testing." },
};

const base: Doc = {
  version: 1,
  meta: { name: "own-blocks" },
  symbols: { D: { kind: "design", value: 512 } },
  graph: { nodes: [], edges: [] },
};

const doc = withBlock(base, def);

describe("a block the document defines", () => {
  test("the palette offers it", () => {
    const listed = (catalogByCategory(doc).mlp ?? []).map((d) => d.type);
    expect(listed).toContain("half_mlp");
  });

  test("the document's catalog knows it", () => {
    expect(catalogOf(doc)["half_mlp"]?.kind).toBe("composite");
  });

  // Each of the three below went through the built-in catalog and so answered
  // as if the block did not exist. None of them said so.

  test("it can be placed on the sheet", () => {
    // `newNodeFor` returning null is how the drop handler decides to do
    // nothing, so this failing is a block you can see in the palette, drag
    // onto the drawing, and watch not appear.
    const node = newNodeFor("half_mlp", doc);
    expect(node).not.toBeNull();
    expect(node?.type).toBe("half_mlp");
    // The defaults the definition declares, so the placed block resolves.
    expect(node?.params).toMatchObject({ D: 512, F: 2048 });
  });

  test("it is drawn as the kind it is", () => {
    expect(kindOf(catalogOf(doc)["half_mlp"])).toBe("composite");
  });

  test("it can be opened", () => {
    const node = { id: "m1", type: "half_mlp", params: {} };
    expect(ops.isDrillable(node, catalogOf(doc)["half_mlp"])).toBe(true);
  });
});

describe("a type nothing has heard of", () => {
  test("still answers, and answers that it is unknown", () => {
    expect(newNodeFor("not_a_block", doc)).toBeNull();
    expect(kindOf(catalogOf(doc)["not_a_block"])).toBe("unknown");
    expect(ops.isDrillable({ id: "x", type: "not_a_block", params: {} }, undefined)).toBe(false);
  });
});
