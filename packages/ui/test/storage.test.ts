/**
 * The seam a store plugs into, exercised by a store.
 *
 * An interface with no implementation in the tree is an interface nobody has
 * checked, so this builds the smallest honest one — a Map — and drives a whole
 * session through it: save a design and where you were, load it back, and land
 * in the same place.
 *
 * The other half is what happens when the two drift. A view is stored beside a
 * document and the document can change without it, so a selection that is gone
 * has to be dropped without taking the rest of the restore with it.
 */

import { beforeEach, describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();
const { getPreset } = await import("../src/engine.js");
const { useEditor } = await import("../src/state/store.js");
const { serializeDoc, parseDoc } = await import("../src/state/serialize.js");
const { registerStorage, storage, subscribeStorage } = await import("../src/state/storage.js");
const { captureView, applyView } = await import("../src/state/session.js");
import type { StorageProvider, StoredDesign, ViewState } from "../src/state/storage.js";

/** The smallest thing that satisfies the interface. */
function fakeStore(): StorageProvider & { rows: Map<string, unknown> } {
  const rows = new Map<string, { name: string; body: string; view?: ViewState }>();
  let next = 1;
  return {
    label: "A Map",
    rows: rows as Map<string, unknown>,
    async list(): Promise<StoredDesign[]> {
      return [...rows.entries()].map(([id, r]) => ({ id, name: r.name, updatedAt: "2026-01-01T00:00:00Z" }));
    },
    async load(id) {
      const row = rows.get(id);
      if (!row) throw new Error(`no such design ${id}`);
      return row;
    },
    async save(id, name, body, view) {
      const key = id ?? `d${next++}`;
      rows.set(key, { name, body, view });
      return { id: key };
    },
    async remove(id) {
      rows.delete(id);
    },
  };
}

const state = () => useEditor.getState();

describe("registering a place to keep designs", () => {
  beforeEach(() => registerStorage(null));

  test("nothing is offering by default, which is the plain checkout", () => {
    expect(storage()).toBeNull();
  });

  test("a provider can be offered and withdrawn", () => {
    const fake = fakeStore();
    registerStorage(fake);
    expect(storage()?.label).toBe("A Map");
    registerStorage(null);
    expect(storage()).toBeNull();
  });

  test("and anyone watching is told", () => {
    let told = 0;
    const stop = subscribeStorage(() => told++);
    registerStorage(fakeStore());
    registerStorage(null);
    stop();
    registerStorage(fakeStore());
    expect(told).toBe(2);
  });
});

describe("a whole session through a provider", () => {
  beforeEach(() => {
    registerStorage(fakeStore());
    state().setDoc(getPreset("gpt2-small"), "test");
  });

  test("save where you were, come back to it", async () => {
    state().setDetail(3);
    state().setPath(["layers"]);
    state().setOperating({ batch: 4 });

    const view = captureView();
    const { id } = await storage()!.save(null, "mine", serializeDoc(state().doc), view);

    // A different design, a different place, as if the tab had been closed.
    state().setDoc(getPreset("llama-3-8b"), "elsewhere");
    expect(state().path).toEqual([]);

    const back = await storage()!.load(id);
    const doc = parseDoc(back.body);
    state().setDoc(doc, `Opened ${back.name}`);
    applyView(back.view, doc);

    expect(state().doc.meta.name).toBe("gpt2-small");
    expect(state().path).toEqual(["layers"]);
    expect(state().detail).toBe(3);
    expect(state().operating.batch).toBe(4);
  });

  test("the design itself survives, defs and all", async () => {
    const doc = state().doc;
    doc.defs = { mine: { params: {}, ports: { in: [], out: [] }, graph: { nodes: [], edges: [] } } } as never;

    const { id } = await storage()!.save(null, "with defs", serializeDoc(doc), captureView());
    expect(parseDoc((await storage()!.load(id)).body).defs).toEqual(doc.defs!);
  });

  test("a second save against the same id replaces it", async () => {
    const first = await storage()!.save(null, "one", serializeDoc(state().doc), {});
    const again = await storage()!.save(first.id, "two", serializeDoc(state().doc), {});
    expect(again.id).toBe(first.id);
    expect(await storage()!.list()).toHaveLength(1);
  });
});

describe("when the view and the document have drifted", () => {
  beforeEach(() => {
    registerStorage(fakeStore());
    state().setDoc(getPreset("gpt2-small"), "test");
  });

  test("a selection that is gone is dropped, and the rest still lands", () => {
    applyView({ selection: "a_block_that_never_existed", detail: 4, path: ["layers"] }, state().doc);
    // Landing on the right level with the right detail beats refusing to land.
    expect(state().selection).toBeNull();
    expect(state().detail).toBe(4);
    expect(state().path).toEqual(["layers"]);
  });

  test("a level that is gone is dropped too", () => {
    applyView({ path: ["nowhere", "at", "all"], detail: 2 }, state().doc);
    expect(state().path).toEqual([]);
    expect(state().detail).toBe(2);
  });

  test("a net whose producer is gone is dropped", () => {
    applyView({ selectedNet: "vanished:y" }, state().doc);
    expect(state().selectedNet).toBeNull();
  });

  test("no view at all is not an error", () => {
    expect(() => applyView(undefined, state().doc)).not.toThrow();
  });
});
