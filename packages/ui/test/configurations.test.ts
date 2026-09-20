/**
 * One design, several sizes.
 *
 * A configuration is a view of the design and not an edit to it, which is the
 * property everything else depends on: switching between two of them has to be
 * something you can do twice and end up where you started. These are the pure
 * functions that hold that, where an editing session cannot.
 */

import { describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();
const ops = await import("../src/state/ops.js");
import type { Doc, SymbolDef } from "@tensor-cad/engine";

const { getPreset, engine } = await import("../src/engine.js");

/** The whole analysis, which is how a configuration is checked: by its numbers. */
const analyze = (doc: Doc) => engine().analyze(doc);

const design = (value: number): SymbolDef => ({ kind: "design", value }) as SymbolDef;

function withSizes(): Doc {
  const doc = getPreset("gpt2-small");
  return {
    ...doc,
    configurations: {
      small: { symbols: {} },
      medium: { symbols: { D: design(1024), L: design(24), H: design(16), Hkv: design(16) } },
    },
  } as Doc;
}

describe("building at a configuration", () => {
  test("the numbers are the configuration's", () => {
    const base = analyze(withSizes()).params.total;
    const medium = analyze(ops.setActiveConfiguration(withSizes(), "medium")).params.total;
    expect(medium).toBeGreaterThan(base);
    // And it is exactly the preset of that size, not an approximation of it.
    expect(medium).toBe(analyze(getPreset("gpt2-medium")).params.total);
  });

  test("an expression follows an override rather than freezing", () => {
    // gpt2's F is written `4*D`, so overriding D has to move it.
    const doc = ops.setActiveConfiguration(withSizes(), "medium");
    expect(analyze(doc).symbols.designValues.F).toBe(4096);
    expect(analyze(withSizes()).symbols.designValues.F).toBe(3072);
  });

  test("going back gives the design as written", () => {
    const there = ops.setActiveConfiguration(withSizes(), "medium");
    const back = ops.setActiveConfiguration(there, null);
    expect(back.active).toBeUndefined();
    expect(analyze(back).params.total).toBe(analyze(withSizes()).params.total);
    // The design's own symbols were never touched on the way.
    expect(there.symbols).toEqual(withSizes().symbols);
  });

  test("a name nothing defines is the design as written", () => {
    const doc = ops.setActiveConfiguration(withSizes(), "no-such-size");
    expect(analyze(doc).params.total).toBe(analyze(withSizes()).params.total);
  });
});

describe("editing while one is in force", () => {
  test("the edit lands in the configuration, not in the design", () => {
    const at = ops.setActiveConfiguration(withSizes(), "medium");
    const edited = ops.setSymbolInConfiguration(at, "D", design(2048));
    expect(edited.configurations?.medium?.symbols?.D).toEqual(design(2048));
    // Untouched underneath, which is what makes switching back meaningful.
    expect(edited.symbols.D).toEqual(withSizes().symbols.D);
    expect(analyze(ops.setActiveConfiguration(edited, null)).params.total).toBe(
      analyze(withSizes()).params.total,
    );
  });

  test("with none in force it is an ordinary symbol edit", () => {
    const edited = ops.setSymbolInConfiguration(withSizes(), "D", design(2048));
    expect(edited.symbols.D).toEqual(design(2048));
    expect(edited.configurations).toEqual(withSizes().configurations);
  });

  test("removing a symbol from a configuration lets the design's own show through", () => {
    const at = ops.setActiveConfiguration(withSizes(), "medium");
    const cleared = ops.setSymbolInConfiguration(at, "D", undefined);
    expect(cleared.configurations?.medium?.symbols?.D).toBeUndefined();
    expect(analyze(cleared).symbols.designValues.D).toBe(768);
  });
});

describe("naming and forgetting", () => {
  test("a captured configuration says only what differs", () => {
    const at = ops.setActiveConfiguration(withSizes(), "medium");
    const named = ops.captureConfiguration(at, "24-layer");
    // Not every symbol: restating them all would stop `F` following `D`.
    expect(Object.keys(named.configurations!["24-layer"]!.symbols).sort()).toEqual([
      "D",
      "H",
      "Hkv",
      "L",
    ]);
    expect(named.active).toBe("24-layer");
  });

  test("capturing with nothing in force records nothing to override", () => {
    const named = ops.captureConfiguration(withSizes(), "plain");
    expect(named.configurations!.plain!.symbols).toEqual({});
    // Which still builds the design, because that is what it describes.
    expect(analyze(named).params.total).toBe(analyze(withSizes()).params.total);
  });

  test("forgetting one leaves the design, and stops building at it", () => {
    const at = ops.setActiveConfiguration(withSizes(), "medium");
    const gone = ops.removeConfiguration(at, "medium");
    expect(gone.configurations?.medium).toBeUndefined();
    expect(gone.active).toBeUndefined();
    expect(analyze(gone).params.total).toBe(analyze(withSizes()).params.total);
  });

  test("forgetting one that is not there changes nothing", () => {
    const before = withSizes();
    expect(ops.removeConfiguration(before, "no-such-size")).toBe(before);
  });
});
