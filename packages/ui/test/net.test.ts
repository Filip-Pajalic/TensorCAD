/**
 * Selecting a net.
 *
 * A tensor is a thing in this design — it has a shape, a producer, consumers
 * and a share of the activation memory — and the drawing was the last place it
 * was not selectable. These are the rules that keeps it from becoming a second
 * kind of selection everything downstream has to know about.
 */

import { beforeEach, describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();
const { getPreset } = await import("../src/engine.js");
const { useEditor } = await import("../src/state/store.js");

const state = () => useEditor.getState();

describe("a net and a block are never both selected", () => {
  beforeEach(() => {
    state().setDoc(getPreset("gpt2-small"), "test");
  });

  test("selecting a net drops the block", () => {
    state().select("embed");
    expect(state().selection).toBe("embed");

    state().selectNet("embed:y");
    expect(state().selectedNet).toBe("embed:y");
    expect(state().selection).toBeNull();
  });

  test("selecting a block drops the net", () => {
    state().selectNet("embed:y");
    state().select("embed");
    expect(state().selection).toBe("embed");
    expect(state().selectedNet).toBeNull();
  });

  test("selecting several blocks drops it too", () => {
    state().selectNet("embed:y");
    state().selectPaths(["embed", "final_norm"]);
    expect(state().selectedNet).toBeNull();
  });

  test("changing level drops it, because a net belongs to a drawing", () => {
    state().selectNet("embed:y");
    state().setPath(["layers"]);
    expect(state().selectedNet).toBeNull();
  });
});

describe("a net that is not there any more", () => {
  beforeEach(() => {
    state().setDoc(getPreset("gpt2-small"), "test");
  });

  test("opening a different design drops it", () => {
    state().selectNet("embed:y");
    state().setDoc(getPreset("llama-3-8b"), "another");
    expect(state().selectedNet).toBeNull();
  });

  test("an edit that arrives from an agent keeps a net that survived it", () => {
    state().selectNet("embed:y");
    const edited = structuredClone(state().doc);
    edited.meta.name = "edited elsewhere";
    state().applyRemote(edited, "agent");
    expect(state().selectedNet).toBe("embed:y");
  });

  test("and drops one whose producer it deleted", () => {
    state().selectNet("embed:y");
    const edited = structuredClone(state().doc);
    edited.graph.nodes = edited.graph.nodes.filter((n) => n.id !== "embed");
    state().applyRemote(edited, "agent");
    // Describing a tensor that is not there is worse than describing nothing:
    // every field would be blank and the name would still look authoritative.
    expect(state().selectedNet).toBeNull();
  });
});

describe("what the analysis says about one", () => {
  test("a block that fans out charges each of its outputs separately", async () => {
    const { engine } = await import("../src/engine.js");
    const result = engine().analyze(getPreset("nemotron-h-8b"));
    const byTensor = result.memory.train.activationsByTensor;

    const split = Object.entries(byTensor).filter(([k]) => k.startsWith("layers/blk0/split:"));
    expect(split).toHaveLength(3);

    const sizes = split.map(([, bytes]) => bytes).sort((a, b) => a - b);
    // Two megabytes and a hundred and sixty: one number for the block answers
    // neither which of these is the big one nor what dropping one would save.
    expect(sizes[0]).toBeLessThan(sizes[2]! / 10);

    const byPath = result.memory.train.activationsByPath["layers/blk0/split"];
    expect(byPath).toBeCloseTo(
      sizes.reduce((a, b) => a + b, 0),
      0,
    );
  });
});
