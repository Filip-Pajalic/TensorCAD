/**
 * The design's own block library.
 *
 * `state/blocks.ts` is pure: a document in, a document out. That is where the
 * editor keeps the parts of itself that can be wrong in a way type-checking
 * would not catch — which type a rename left behind, which instance a delete
 * would orphan — so it is the part worth testing without a browser.
 */

import { beforeAll, describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";
import {
  blockFromGraph,
  defsOf,
  freeTypeName,
  mergeLibrary,
  renameBlock,
  toLibrary,
  usageOf,
  withBlock,
  withoutBlock,
} from "../src/state/blocks.js";
import type { Doc, Graph, UserBlockDef } from "@tensorcad/engine";

beforeAll(async () => {
  // `blocks.ts` reads the catalog to know which type names are taken.
  await loadEngine();
});

function graph(...nodes: { id: string; type: string; graph?: Graph }[]): Graph {
  return { nodes, edges: [] };
}

/** A design holding a definition, some instances of it, and a second one. */
function design(): Doc {
  const inner: UserBlockDef = {
    type: "inner",
    category: "block",
    params: {},
    ports: { in: { x: "... D" }, out: { y: "... D" } },
    graph: graph({ id: "n", type: "rmsnorm" }),
    docs: { summary: "" },
  };
  const outer: UserBlockDef = {
    ...inner,
    type: "outer",
    // A definition that uses the other one: this is the instance a delete has
    // to notice and a canvas cannot show.
    graph: graph({ id: "a", type: "inner" }),
  };
  return {
    version: 1,
    meta: { name: "d" },
    symbols: { D: 8 },
    graph: graph(
      { id: "one", type: "inner" },
      { id: "stack", type: "repeat", graph: graph({ id: "two", type: "inner" }) },
      { id: "other", type: "outer" },
    ),
    defs: { inner, outer },
  } as unknown as Doc;
}

describe("where a block is used", () => {
  test("finds every instance, at any depth, by path", () => {
    const { paths } = usageOf(design(), "inner");
    expect(paths).toEqual(["one", "stack/two"]);
  });

  test("counts the ones inside another definition separately", () => {
    // They are real uses and they are not on the canvas, so they are worth a
    // number of their own rather than a path nobody can follow.
    expect(usageOf(design(), "inner").inDefs).toBe(1);
    expect(usageOf(design(), "outer")).toEqual({ paths: ["other"], inDefs: 0 });
  });

  test("a definition is not a use of itself", () => {
    const doc = design();
    const defs = defsOf(doc);
    const selfish = { ...defs.inner!, graph: graph({ id: "me", type: "inner" }) };
    const used = usageOf({ ...doc, defs: { ...defs, inner: selfish } } as Doc, "inner");
    expect(used.inDefs).toBe(1); // the one in `outer`, and not the one in itself
  });

  test("says nothing is used when nothing is", () => {
    expect(usageOf(design(), "no_such_block")).toEqual({ paths: [], inDefs: 0 });
  });
});

describe("renaming a definition", () => {
  test("moves every instance with it, at any depth and inside other definitions", () => {
    const { doc, to } = renameBlock(design(), "inner", "Core Norm");
    expect(to).toBe("core_norm");
    expect(defsOf(doc).inner).toBeUndefined();
    expect(defsOf(doc).core_norm?.type).toBe("core_norm");
    // Nothing anywhere still names the old type, which is the whole point: a
    // design holding one would report an unknown block with no clue why.
    expect(JSON.stringify(doc)).not.toContain('"inner"');
    expect(usageOf(doc, "core_norm")).toEqual({ paths: ["one", "stack/two"], inDefs: 1 });
  });

  test("will not collide with a name already taken", () => {
    const { doc, to } = renameBlock(design(), "inner", "outer");
    expect(to).toBe("outer_2");
    expect(defsOf(doc).outer?.graph.nodes[0]?.type).toBe("outer_2");
  });

  test("will not collide with a built-in", () => {
    expect(renameBlock(design(), "inner", "linear").to).toBe("linear_2");
  });

  test("leaves the design alone when the name has not changed", () => {
    const before = design();
    expect(renameBlock(before, "inner", "inner").doc).toBe(before);
    expect(renameBlock(before, "no_such_block", "x").doc).toBe(before);
  });
});

describe("the library as a file", () => {
  test("round-trips through export and import", () => {
    const doc = design();
    const empty = { ...doc, defs: {} } as Doc;
    const { doc: back, added, errors } = mergeLibrary(empty, JSON.stringify(toLibrary(doc)));
    expect(errors).toEqual([]);
    expect(added.sort()).toEqual(["inner", "outer"]);
    expect(defsOf(back)).toEqual(defsOf(doc));
  });

  test("refuses anything that is not one, by name", () => {
    const doc = design();
    expect(mergeLibrary(doc, "{").errors[0]).toContain("Not JSON");
    expect(mergeLibrary(doc, "{}").errors[0]).toBe("Not a block library.");
  });

  test("keeps the blocks it can and says which it could not", () => {
    const lib = { kind: "tensorcad-blocks", version: 1, blocks: { bad: { graph: { nodes: [], edges: [] } } } };
    const { added, errors } = mergeLibrary(design(), JSON.stringify(lib));
    expect(added).toEqual([]);
    expect(errors.length).toBe(1);
    expect(errors[0]).toContain("bad");
  });
});

describe("adding and removing", () => {
  test("a free name avoids the built-ins and what is already defined", () => {
    const doc = design();
    expect(freeTypeName(doc, "linear")).toBe("linear_2");
    expect(freeTypeName(doc, "inner")).toBe("inner_2");
    expect(freeTypeName(doc, "  Fresh Name! ")).toBe("fresh_name");
    // A type name cannot start with a digit, and nothing at all is still a name.
    expect(freeTypeName(doc, "3rd")).toBe("b3rd");
    expect(freeTypeName(doc, "!!!")).toBe("block");
  });

  test("removing one leaves the rest", () => {
    const doc = withoutBlock(design(), "inner");
    expect(Object.keys(defsOf(doc))).toEqual(["outer"]);
  });

  test("adding one does not disturb the design", () => {
    const before = design();
    const after = withBlock(before, { ...defsOf(before).inner!, type: "third" });
    expect(after.graph).toBe(before.graph);
    expect(Object.keys(defsOf(after)).sort()).toEqual(["inner", "outer", "third"]);
  });
});

describe("making a block out of a level", () => {
  test("turns the symbols it uses into parameters, and references into $name", () => {
    const level: Graph = {
      nodes: [
        { id: "_in", type: "boundary_in", params: { ports: { x: "... D" } } },
        { id: "up", type: "linear", params: { in_features: "D", out_features: "F" } },
        { id: "_out", type: "boundary_out", params: { ports: { y: "... F" } } },
      ],
      edges: [["_in:x", "up:x"]],
    } as unknown as Graph;

    const { def, errors } = blockFromGraph(level, "mlp_half", { D: 8, F: 32, V: 100 }, "");
    expect(errors).toEqual([]);
    // Only the symbols this graph mentions, not every symbol in the document.
    expect(Object.keys(def.params ?? {})).toEqual(["D", "F"]);
    expect(def.params?.D?.default).toBe(8);
    // Node parameters take the `$` form; port patterns name them directly.
    expect(def.graph.nodes[0]?.params?.in_features).toBe("$D");
    expect(def.ports.in.x).toBe("... D");
    // The boundary nodes are regenerated on expansion, so they are not kept.
    expect(def.graph.nodes.map((n) => n.id)).toEqual(["up"]);
  });
});
