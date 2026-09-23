/**
 * The drawing, without looking at it.
 *
 * Two things: what a block and a wire are called when they have focus, and
 * that the keyboard can go down the hierarchy and back up it. Double-click was
 * the only way into a block, and Escape — which React Flow also answers, by
 * deselecting the focused block — went up a level on the same press.
 */

import { describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();

const { getPreset } = await import("../src/engine.js");
const { blockLabel, frameLabel, wireLabel } = await import("../src/canvas/a11y.js");
const { beforeKey, runCommand, COMMANDS } = await import("../src/state/commands.js");
const { useEditor } = await import("../src/state/store.js");

import type { BlockNodeData } from "../src/canvas/BlockNode.js";
import type { FrameNodeData } from "../src/canvas/FrameNode.js";
import type { Doc } from "@tensor-cad/engine";

const block = (over: Partial<BlockNodeData>): BlockNodeData =>
  ({
    path: "embed",
    label: "embed",
    type: "embedding",
    typeName: "token embedding",
    params: 144,
    inPorts: [],
    outPorts: [{ name: "y", shape: "B T D", connected: true, dtype: null }],
    severity: null,
    findings: [],
    drillable: false,
    locked: false,
    ...over,
  }) as BlockNodeData;

describe("what a block is called", () => {
  test("its name, what it is, its size and what comes out", () => {
    expect(blockLabel(block({}))).toBe("embed, token embedding, 144 parameters, out B T D");
  });

  test("the type is not said twice when the block was never renamed", () => {
    expect(blockLabel(block({ label: "token embedding" }))).toBe("token embedding, 144 parameters, out B T D");
  });

  test("the checks, counted, and whether there is an inside", () => {
    const d = block({
      findings: [
        { severity: "warning", message: "a", rule: "r" },
        { severity: "warning", message: "b", rule: "r" },
        { severity: "error", message: "c", rule: "r" },
      ],
    });
    expect(blockLabel(d, true)).toBe(
      "embed, token embedding, 144 parameters, out B T D, 1 error and 2 warnings, press Enter to open",
    );
  });

  test("a marker from something folded inside still gets said", () => {
    expect(blockLabel(block({ severity: "info" }))).toContain("has notes");
  });

  test("a stack does not say its count twice", () => {
    const d = { label: "Transformer block x3", typeName: "stack", multiplier: 3, params: 84_768, severity: null } as FrameNodeData;
    expect(frameLabel(d)).toBe("Transformer block x3, stack, 84.8K parameters");
    expect(frameLabel({ ...d, label: "layers" })).toBe("layers, stack ×3, 84.8K parameters");
  });

  test("a wire says where it runs and what it carries", () => {
    expect(wireLabel("layers/block", "y", "final_norm", "x", "B T D")).toBe("block y to final_norm x, B T D");
  });
});

describe("down the hierarchy and back up it", () => {
  const state = () => useEditor.getState();
  const open = (doc: Doc) => {
    state().setDoc(doc);
    state().setDetail(0);
    state().setPath([]);
  };

  test("Enter opens the selected block, and only one that has an inside", () => {
    open(getPreset("nano-sort") as Doc);
    expect(COMMANDS.find((c) => c.id === "view.open")?.shortcut).toBe("Enter");

    state().select("embed");
    runCommand("view.open");
    expect(state().path).toEqual([]);

    state().select("layers");
    runCommand("view.open");
    expect(state().path).toEqual(["layers"]);
  });

  test("the Enter that selects a block does not also open it", () => {
    open(getPreset("nano-sort") as Doc);
    // React Flow selects the focused block on Enter before the command runs.
    // Nothing was selected when the key went down, so this press only selects.
    beforeKey();
    state().select("layers");
    runCommand("view.open");
    expect(state().path).toEqual([]);

    // The next Enter finds it selected, and opens it.
    beforeKey();
    runCommand("view.open");
    expect(state().path).toEqual(["layers"]);
  });

  test("Escape deselects first, and goes up only when nothing was selected", () => {
    open(getPreset("nano-sort") as Doc);
    state().enter("layers");
    state().select("layers/block");

    // React Flow deselects the focused block on Escape before the command runs.
    // What counts is what was selected when the key went down.
    beforeKey();
    state().select(null);
    runCommand("edit.deselect");
    expect(state().path).toEqual(["layers"]);

    beforeKey();
    runCommand("edit.deselect");
    expect(state().path).toEqual([]);
  });
});
