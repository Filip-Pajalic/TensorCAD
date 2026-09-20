/**
 * The timeline: edits as things, not as documents.
 *
 * The property everything here is about is that an edit survives being made.
 * A stack of documents can say what the design looked like before; only a list
 * of operations can say what it would look like *without* the third one, and
 * that is the difference between undo and a feature tree.
 *
 * So the tests that matter are the ones a stack could not pass: suppress a
 * step in the middle and watch the rest replay on top of what is left, and
 * watch what depended on it fail and say so rather than being dropped in
 * silence.
 */

import { beforeEach, describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();
const { getPreset } = await import("../src/engine.js");
const { useEditor } = await import("../src/state/store.js");
const ops = await import("../src/state/ops.js");

const state = () => useEditor.getState();
/** The list the panel draws: the base and then every step. */
const labels = (): string[] => [state().baseLabel, ...state().steps.map((s) => s.label)];
const symbolValue = (name: string): unknown =>
  (state().doc.symbols?.[name] as { value?: unknown } | undefined)?.value;

const design = (value: number) => ({ kind: "design", value }) as never;

function threeEdits(): void {
  state().setDoc(getPreset("gpt2-small"), "Opened GPT-2 Small");
  state().setSymbol("L", design(6));
  state().setSymbol("D", design(512));
  state().setMetaName("tiny");
}

describe("every step says what it is and what it did", () => {
  beforeEach(threeEdits);

  test("in the words the toolbar used at the time", () => {
    expect(labels()).toEqual(["Opened GPT-2 Small", "Set L", "Set D", `Named the design "tiny"`]);
    expect(state().at).toBe(3);
  });

  test("and keeps the operation, not the result", () => {
    // The thing that makes this a timeline. The first step is still an edit
    // with its arguments in it, a week and ninety-nine steps later.
    expect(state().steps[0]!.edit).toMatchObject({ kind: "setSymbol", name: "L" });
    expect(state().steps[2]!.edit).toMatchObject({ kind: "setMetaName", name: "tiny" });
  });

  test("an edit that changes nothing is not a step", () => {
    const before = state().steps.length;
    state().setMetaName("tiny");
    expect(state().steps).toHaveLength(before);
  });

  test("undo says what it took back", () => {
    state().undo();
    expect(state().status).toBe(`Undid named the design "tiny"`);
    expect(state().docLabel).toBe("Set D");
  });
});

describe("moving the mark", () => {
  beforeEach(threeEdits);

  test("is undo several times at once", () => {
    state().jumpTo(1);
    expect(state().at).toBe(1);
    expect(symbolValue("L")).toBe(6);
    expect(symbolValue("D")).not.toBe(512);
  });

  test("and forward again, because the steps are still there", () => {
    state().jumpTo(1);
    state().jumpTo(3);
    expect(state().doc.meta.name).toBe("tiny");
  });

  test("the list itself does not change length", () => {
    const before = labels();
    state().jumpTo(0);
    expect(labels()).toEqual(before);
    state().jumpTo(2);
    expect(labels()).toEqual(before);
  });

  test("pressing where you already are does nothing, and says nothing", () => {
    const before = { at: state().at, doc: state().doc, status: state().status };
    state().jumpTo(before.at);
    expect(state().at).toBe(before.at);
    expect(state().doc).toBe(before.doc);
    expect(state().status).toBe(before.status);
  });

  test("an index that is not there does nothing", () => {
    const before = state().at;
    state().jumpTo(99);
    state().jumpTo(-1);
    expect(state().at).toBe(before);
  });

  test("editing after a jump discards the steps that were ahead", () => {
    state().jumpTo(1);
    state().setMetaName("a different way");
    expect(labels()).toEqual(["Opened GPT-2 Small", "Set L", `Named the design "a different way"`]);
  });
});

describe("taking a step out of the middle", () => {
  beforeEach(threeEdits);

  test("replays everything after it without it", () => {
    // The whole of E7's third part in one assertion: D was set second and the
    // design is now named, so suppressing D has to leave the name alone.
    state().setSuppressed(1, true);
    expect(symbolValue("D")).not.toBe(512);
    expect(symbolValue("L")).toBe(6);
    expect(state().doc.meta.name).toBe("tiny");
  });

  test("and putting it back replays it again", () => {
    state().setSuppressed(1, true);
    state().setSuppressed(1, false);
    expect(symbolValue("D")).toBe(512);
    expect(state().doc.meta.name).toBe("tiny");
  });

  test("the step stays in the list, struck out rather than gone", () => {
    state().setSuppressed(1, true);
    expect(labels()).toHaveLength(4);
    expect(state().steps[1]!.suppressed).toBe(true);
  });

  test("removing it takes it out for good", () => {
    state().removeStep(1);
    expect(labels()).toEqual(["Opened GPT-2 Small", "Set L", `Named the design "tiny"`]);
    expect(symbolValue("D")).not.toBe(512);
    expect(state().doc.meta.name).toBe("tiny");
  });
});

describe("a step that cannot replay", () => {
  beforeEach(() => {
    state().setDoc(getPreset("gpt2-small"), "Opened GPT-2 Small");
    const node = ops.repeatSkeleton("extra");
    state().addNode([], node);
    state().renameNode("extra", "the one that matters");
  });

  test("says what it could not find rather than being dropped in silence", () => {
    expect(state().failures).toHaveLength(0);

    // Take out the step that added the block the next step renames.
    state().setSuppressed(0, true);

    expect(state().failures).toEqual([{ at: 1, reason: "there is no extra" }]);
  });

  test("and the fold carries on past it", () => {
    state().setMetaName("still here");
    state().setSuppressed(0, true);
    // The rename failed; the name after it did not.
    expect(state().doc.meta.name).toBe("still here");
    expect(state().failures.map((f) => f.at)).toEqual([1]);
  });

  test("putting the step back clears the failure", () => {
    state().setSuppressed(0, true);
    expect(state().failures).toHaveLength(1);
    state().setSuppressed(0, false);
    expect(state().failures).toHaveLength(0);
  });
});

describe("an edit that arrives from an agent", () => {
  beforeEach(threeEdits);

  test("is a step like any other", () => {
    const edited = structuredClone(state().doc);
    edited.meta.name = "the agent did this";
    state().applyRemote(edited, "Agent: set_symbol");

    expect(labels().at(-1)).toBe("Agent: set_symbol");
    expect(state().steps.at(-1)!.edit.kind).toBe("replaceDoc");
    expect(state().doc.meta.name).toBe("the agent did this");
  });

  test("so it can be taken back out", () => {
    const edited = structuredClone(state().doc);
    edited.meta.name = "the agent did this";
    state().applyRemote(edited, "Agent: set_symbol");

    state().setSuppressed(3, true);
    expect(state().doc.meta.name).toBe("tiny");
  });
});

describe("opening a different design", () => {
  test("starts a different timeline", () => {
    threeEdits();
    state().setDoc(getPreset("llama-3-8b"), "Opened Llama 3 8B");
    // Replaying edits made to another document is either meaningless or,
    // worse, occasionally works.
    expect(state().steps).toHaveLength(0);
    expect(state().at).toBe(0);
    expect(labels()).toEqual(["Opened Llama 3 8B"]);
  });
});
