/**
 * The edit layer on its own, with no transport in the way.
 *
 * `applyOps` is the only thing that mutates a document, so its guarantees --
 * never touching the input, all-or-nothing batches, and edges that follow their
 * blocks -- are worth pinning down directly.
 */

import { describe, expect, test } from "bun:test";
import type { Doc } from "@tensor-cad/engine";
import { countParams, getPreset, loadEngine } from "@tensor-cad/engine/node";
import { applyOps, OpError, type Op } from "../src/ops.js";
import { FileStore } from "../src/store/file-store.js";
import { outlineOf } from "../src/summarize.js";

await loadEngine();

const params = (doc: Doc) => countParams(doc).total;

describe("applyOps", () => {
  test("leaves the input document untouched", () => {
    const before = getPreset("gpt2-small");
    const snapshot = JSON.stringify(before);
    const { doc } = applyOps(before, [{ op: "set_symbol", name: "L", value: 24 }]);
    expect(JSON.stringify(before)).toBe(snapshot);
    expect(params(doc)).toBeGreaterThan(params(before));
  });

  test("removing a block takes its edges with it", () => {
    const { doc, applied } = applyOps(getPreset("gpt2-small"), [{ op: "remove_node", path: "final_norm" }]);
    expect(applied[0]).toContain("2 edges");
    expect(doc.graph.nodes.some((n) => n.id === "final_norm")).toBe(false);
    expect(doc.graph.edges.flat().some((e) => e.startsWith("final_norm:"))).toBe(false);
  });

  test("disconnect then connect rewires a graph", () => {
    const start = getPreset("gpt2-small");
    const ops: Op[] = [
      { op: "disconnect", from: "final_norm:y", to: "head:x" },
      { op: "connect", from: "layers:x", to: "head:x" },
    ];
    const { doc } = applyOps(start, ops);
    expect(doc.graph.edges.some(([f, t]) => f === "layers:x" && t === "head:x")).toBe(true);
    expect(doc.graph.edges.some(([f]) => f === "final_norm:y")).toBe(false);
  });

  test("a connection into an occupied port is refused", () => {
    expect(() =>
      applyOps(getPreset("gpt2-small"), [{ op: "connect", from: "embed:y", to: "head:x" }]),
    ).toThrow(/already receives/);
  });

  test("blocks can be added inside a container", () => {
    const { doc } = applyOps(getPreset("gpt2-small"), [
      { op: "add_node", parent: "layers", id: "extra_norm", type: "rmsnorm", params: { dim: "D" } },
    ]);
    const inner = doc.graph.nodes.find((n) => n.id === "layers")!.graph!;
    expect(inner.nodes.map((n) => n.id)).toContain("extra_norm");
    // The new block is inside a 12-deep repeat, so it counts twelve times.
    expect(params(doc) - params(getPreset("gpt2-small"))).toBe(12 * 768);
  });

  test("a rejected operation aborts the whole batch", () => {
    const start = getPreset("gpt2-small");
    let thrown: unknown;
    try {
      applyOps(start, [
        { op: "set_symbol", name: "L", value: 24 },
        { op: "set_param", path: "embed", key: "not_a_param", value: 1 },
      ]);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(OpError);
    expect((thrown as OpError).index).toBe(1);
    expect((thrown as Error).message).toContain("has no parameter");
  });

  test("set_symbol keeps a runtime symbol runtime, and null deletes", () => {
    const { doc } = applyOps(getPreset("gpt2-small"), [{ op: "set_symbol", name: "T", value: 2048 }]);
    expect(doc.symbols.T).toEqual({ kind: "runtime", default: 2048, doc: "Sequence length in tokens" });

    const outline = outlineOf(doc);
    expect(outline.symbols.find((s) => s.name === "T")!.kind).toBe("runtime");

    const { doc: without } = applyOps(doc, [{ op: "set_symbol", name: "V", value: null }]);
    expect("V" in without.symbols).toBe(false);
  });

  test("a path through a non-container is refused clearly", () => {
    expect(() => applyOps(getPreset("gpt2-small"), [{ op: "remove_node", path: "embed/inner" }])).toThrow(
      /has no subgraph/,
    );
  });
});

describe("FileStore", () => {
  test("revisions advance and the op log drives undo", () => {
    const store = new FileStore();
    const record = store.create({ preset: "gpt2-small" });
    expect(record.revision).toBe(1);
    const original = params(record.doc);

    store.apply(record.design_id, [{ op: "set_symbol", name: "L", value: 24 }]);
    expect(store.get(record.design_id).revision).toBe(2);
    expect(params(store.get(record.design_id).doc)).toBeGreaterThan(original);

    store.restore(record.design_id);
    expect(params(store.get(record.design_id).doc)).toBe(original);
    // Undo is itself a change, so the revision keeps climbing.
    expect(store.get(record.design_id).revision).toBe(3);
  });

  test("a failed batch does not burn a revision", () => {
    const store = new FileStore();
    const record = store.create({ preset: "gpt2-small" });
    expect(() => store.apply(record.design_id, [{ op: "remove_node", path: "nope" }])).toThrow();
    expect(store.get(record.design_id).revision).toBe(1);
  });

  test("an empty design starts with only the runtime symbols", () => {
    const store = new FileStore();
    const record = store.create({ name: "blank" });
    expect(record.doc.graph.nodes).toEqual([]);
    expect(Object.keys(record.doc.symbols).sort()).toEqual(["B", "T"]);
    expect(record.source).toBe("empty");
  });
});
