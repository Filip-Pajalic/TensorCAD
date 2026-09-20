/**
 * Editing a block the document defines for itself.
 *
 * The template is written in the block's own parameters — `$D`, not 4096 — and
 * the canvas is given a preview with those bound to their declared defaults, so
 * shapes resolve and weights count. What matters is that the rewrite is total in
 * both directions: what is drawn is what is stored, and a value that names a
 * parameter comes back as a reference to it rather than as the number it
 * happened to resolve to.
 */

import { describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();
const { previewDoc, storeInDefinition, storeValue, DEF_PREFIX, defOfPath, isDefPath } = await import(
  "../src/state/definition.js"
);
const ops = await import("../src/state/ops.js");

import type { Doc, Graph, UserBlockDef } from "@tensorcad/engine";

const def: UserBlockDef = {
  type: "half_mlp",
  category: "block",
  params: {
    D: { type: "int", default: 512, min: 1, doc: "Width in" },
    F: { type: "int", default: 2048, min: 1, doc: "Width in the middle" },
  },
  ports: { in: { x: "... D" }, out: { y: "... F" } },
  graph: {
    nodes: [
      { id: "up", type: "linear", params: { in_features: "$D", out_features: "$F", bias: false } },
      { id: "act", type: "activation", params: { kind: "silu", dim: "$F" } },
      // A literal, which has to stay one: not everything inside a template is
      // a parameter, and 64 is 64 at every width.
      { id: "trim", type: "rmsnorm", params: { dim: 64 } },
    ],
    edges: [
      ["_in:x", "up:x"],
      ["up:y", "act:x"],
    ],
  } as unknown as Graph,
  docs: { summary: "Half a feed-forward." },
};

function doc(): Doc {
  return {
    version: 1,
    meta: { name: "d" },
    symbols: { B: 1, T: 8, D: 999 },
    graph: { nodes: [{ id: "one", type: "half_mlp", params: {} }], edges: [] },
    defs: { half_mlp: def },
  } as unknown as Doc;
}

describe("the preview a template is drawn as", () => {
  test("binds every parameter reference to a name the engine can resolve", () => {
    const p = previewDoc(doc(), "half_mlp")!;
    const up = p.graph.nodes.find((n) => n.id === "up")!;
    expect(up.params?.in_features).toBe("D");
    expect(up.params?.out_features).toBe("F");
    // And a literal is untouched.
    expect(p.graph.nodes.find((n) => n.id === "trim")?.params?.dim).toBe(64);
  });

  test("its symbols are the block's parameters, at their declared defaults", () => {
    const p = previewDoc(doc(), "half_mlp")!;
    // Not the design's `D` of 999: a template is measured in its own terms.
    expect((p.symbols as Record<string, { value?: number }>).D?.value).toBe(512);
    expect((p.symbols as Record<string, { value?: number }>).F?.value).toBe(2048);
  });

  test("carries the boundary nodes the expansion generates, from the declared ports", () => {
    const p = previewDoc(doc(), "half_mlp")!;
    const ids = p.graph.nodes.map((n) => n.id);
    expect(ids[0]).toBe("_in");
    expect(ids[ids.length - 1]).toBe("_out");
    expect(p.graph.nodes[0]?.params?.ports).toEqual({ x: "... D" });
  });

  test("its own defs come along, so a definition using another still resolves", () => {
    const d = doc();
    const other: UserBlockDef = { ...def, type: "other" };
    const p = previewDoc({ ...d, defs: { ...d.defs, other } } as Doc, "half_mlp")!;
    // Itself excluded: a template that contained itself would not expand.
    expect(Object.keys(p.defs as Record<string, unknown>)).toEqual(["other"]);
  });

  test("a block that is not defined has no preview", () => {
    expect(previewDoc(doc(), "no_such_block")).toBeNull();
  });
});

describe("storing what was edited", () => {
  const params = new Set(["D", "F"]);

  test("a name that is a parameter becomes a reference to it", () => {
    expect(storeValue("D", params)).toBe("$D");
    expect(storeValue("2*F", params)).toBe("2*$F");
    expect(storeValue("D + F", params)).toBe("$D + $F");
  });

  test("anything else is left exactly as written", () => {
    expect(storeValue("4096", params)).toBe("4096");
    expect(storeValue(64, params)).toBe(64);
    expect(storeValue(true, params)).toBe(true);
    expect(storeValue("silu", params)).toBe("silu");
    // A name that is not a parameter of this block is not one.
    expect(storeValue("V", params)).toBe("V");
  });

  test("a reference somebody already wrote is not doubled", () => {
    expect(storeValue("$D", params)).toBe("$D");
    expect(storeValue("$D * 2", params)).toBe("$D * 2");
  });

  test("and it is the block's own parameters that decide", () => {
    expect(storeInDefinition(doc(), "half_mlp", "D")).toBe("$D");
    // A block that does not exist cannot say, so nothing is rewritten.
    expect(storeInDefinition(doc(), "no_such_block", "D")).toBe("D");
  });
});

describe("the round trip through an edit", () => {
  test("a parameter typed by name is stored as a reference and drawn as the name", () => {
    const before = doc();
    // What the inspector sends when somebody types `F` into `dim`.
    const after = ops.setParam(before, [DEF_PREFIX, "half_mlp", "trim"], "dim", "F");
    const stored = (after.defs as Record<string, UserBlockDef>).half_mlp.graph.nodes.find(
      (n) => n.id === "trim",
    );
    expect(stored?.params?.dim).toBe("$F");
    // And it comes back as `F`, which is what was typed.
    expect(
      previewDoc(after, "half_mlp")!.graph.nodes.find((n) => n.id === "trim")?.params?.dim,
    ).toBe("F");
  });

  test("a literal stays a literal, at every width", () => {
    const after = ops.setParam(doc(), [DEF_PREFIX, "half_mlp", "trim"], "dim", 128);
    const stored = (after.defs as Record<string, UserBlockDef>).half_mlp.graph.nodes.find(
      (n) => n.id === "trim",
    );
    expect(stored?.params?.dim).toBe(128);
  });

  test("an edit outside a definition is not rewritten", () => {
    const after = ops.setParam(doc(), ["one"], "d_model", "D");
    expect(after.graph.nodes[0]?.params?.d_model).toBe("D");
  });

  test("adding and removing a block go to the template, not the design", () => {
    const added = ops.addNode(
      doc(),
      [DEF_PREFIX, "half_mlp"],
      { id: "extra", type: "rmsnorm", params: { dim: 8 } } as never,
    );
    const template = (added.defs as Record<string, UserBlockDef>).half_mlp.graph;
    expect(template.nodes.map((n) => n.id)).toContain("extra");
    // The design is untouched: it still holds exactly its one instance.
    expect(added.graph.nodes.map((n) => n.id)).toEqual(["one"]);

    const removed = ops.removeNode(added, [DEF_PREFIX, "half_mlp", "extra"]);
    expect(
      (removed.defs as Record<string, UserBlockDef>).half_mlp.graph.nodes.map((n) => n.id),
    ).not.toContain("extra");
  });
});

describe("recognising a definition path", () => {
  test("says which definition it is in, and when it is in none", () => {
    expect(isDefPath([DEF_PREFIX, "half_mlp", "up"])).toBe(true);
    expect(defOfPath([DEF_PREFIX, "half_mlp", "up"])).toBe("half_mlp");
    expect(isDefPath(["layers", "block"])).toBe(false);
    expect(defOfPath(["layers", "block"])).toBeNull();
  });
});
