/**
 * What a change did, said the way the rest of the editor says it.
 *
 * The strip under the parameter count lists what moved since the design was
 * opened. It reads a symbol by its own description and a block's parameter by
 * the catalog's label, so a change to `Hkv` says "key/value heads", and a
 * switch between activations names them as a paper does.
 */

import { beforeAll, describe, expect, test } from "bun:test";
import type { Doc } from "@tensor-cad/engine";
import { loadEngine } from "../src/engine.js";

let getPreset: typeof import("../src/engine.js").getPreset;
let diffDesigns: typeof import("../src/engine.js").diffDesigns;
let changesInWords: typeof import("../src/panels/SinceOpened.js").changesInWords;

beforeAll(async () => {
  await loadEngine();
  ({ getPreset, diffDesigns } = await import("../src/engine.js"));
  ({ changesInWords } = await import("../src/panels/SinceOpened.js"));
});

function words(before: Doc, after: Doc): string[] {
  return changesInWords(diffDesigns(before, after), before, after);
}

describe("what moved, in words", () => {
  test("a symbol by its description", () => {
    const before = getPreset("nano-sort");
    const after = structuredClone(before);
    (after.symbols.Hkv as { value: number }).value = 1;
    expect(words(before, after)).toEqual(["key/value heads 3 → 1"]);
  });

  test("a block's parameter by its label, and an enum in the catalog's words", () => {
    const before = getPreset("nano-sort");
    const after = structuredClone(before);
    const block = after.graph.nodes.find((n) => n.id === "layers")!.graph!.nodes.find((n) => n.id === "block")!;
    block.params = { ...block.params, act: "relu" };
    expect(words(before, after)).toEqual(["block: activation GELU → ReLU"]);
  });

  test("a block added, by what it is", () => {
    const before = getPreset("nano-sort");
    const after = structuredClone(before);
    after.graph.nodes.push({ id: "extra", type: "rmsnorm", params: { dim: "D" } });
    expect(words(before, after)).toContain("added RMS norm");
  });
});
