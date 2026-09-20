/**
 * Saving a design and opening it again.
 *
 * There was no test here, and the round trip was lossy: `serializeDoc` wrote
 * every field faithfully and `parseDoc` rebuilt a five-field object out of a
 * nine-field type, so `defs`, `configurations`, `active` and `rules` went into
 * the file and never came out. The file on disk looked right, which is the
 * worst shape a bug of this kind can take.
 *
 * The property worth pinning is therefore not "it parses" but **what goes in
 * comes out**, field by field and including fields this module has never heard
 * of — because the next one added will not come back here to be added again.
 */

import { describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();
const { getPreset } = await import("../src/engine.js");
const { serializeDoc, parseDoc, fileNameFor } = await import("../src/state/serialize.js");
import type { Doc } from "@tensorcad/engine";

/** A design using every optional part of the format. */
function furnished(): Doc {
  const doc = getPreset("gpt2-small");
  doc.defs = {
    my_block: { params: {}, ports: { in: [], out: [] }, graph: { nodes: [], edges: [] } },
  } as never;
  doc.configurations = { small: { symbols: { L: { kind: "design", value: 4 } } } } as never;
  doc.active = "small";
  doc.rules = { "flash-head-dim": "info" } as never;
  doc.ui = { positions: { b: [10, 20], a: [0, 0] } };
  return doc;
}

describe("what goes in comes out", () => {
  test("every optional field survives the round trip", () => {
    const before = furnished();
    const after = parseDoc(serializeDoc(before));

    expect(after.defs).toEqual(before.defs!);
    expect(after.configurations).toEqual(before.configurations!);
    expect(after.active).toBe("small");
    expect(after.rules).toEqual(before.rules!);
  });

  test("and so does the design itself", () => {
    const before = furnished();
    const after = parseDoc(serializeDoc(before));

    expect(after.meta).toEqual(before.meta);
    expect(after.symbols).toEqual(before.symbols);
    expect(after.graph).toEqual(before.graph);
    expect(after.ui?.positions).toEqual(before.ui!.positions!);
  });

  test("a field this module has never heard of survives too", () => {
    const before = furnished() as Doc & { somethingNew?: unknown };
    before.somethingNew = { added: "in a later version" };

    const after = parseDoc(serializeDoc(before)) as Doc & { somethingNew?: unknown };
    // The next field added to Doc will not come back here to be added again.
    expect(after.somethingNew).toEqual({ added: "in a later version" });
  });

  test("twice is the same as once", () => {
    const once = serializeDoc(furnished());
    expect(serializeDoc(parseDoc(once))).toBe(once);
  });
});

describe("the written form", () => {
  test("keys come out in a fixed order, so a design diffs cleanly", () => {
    const text = serializeDoc(furnished());
    const keys = Object.keys(JSON.parse(text));
    expect(keys.slice(0, 4)).toEqual(["version", "meta", "symbols", "graph"]);
  });

  test("positions are sorted, so moving one block does not reorder the file", () => {
    const text = serializeDoc(furnished());
    const positions = (JSON.parse(text) as Doc).ui!.positions!;
    expect(Object.keys(positions)).toEqual(["a", "b"]);
  });

  test("the file is named after the design, safely", () => {
    const doc = getPreset("gpt2-small");
    doc.meta.name = "a design/with:awkward chars";
    expect(fileNameFor(doc)).toBe("a-design-with-awkward-chars.tensorcad.json");
  });
});

describe("what it refuses", () => {
  test("a version it does not know, rather than opening it anyway", () => {
    const text = serializeDoc(furnished()).replace('"version": 1', '"version": 2');
    expect(() => parseDoc(text)).toThrow(/Unsupported document version 2/);
  });

  test("something that is not a design", () => {
    expect(() => parseDoc(JSON.stringify({ version: 1 }))).toThrow(/no graph/);
    expect(() => parseDoc("[]")).toThrow();
  });
});
