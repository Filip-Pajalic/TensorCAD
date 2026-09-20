/**
 * The history: what you did, and going back to any of it.
 *
 * The states were always kept. What was missing was a way to read them, which
 * is what turns "press Ctrl+Z and watch" into "go back to the edit that broke
 * it". These are the rules the list depends on: every state carries the
 * sentence that produced it, jumping is undo several times at once, and
 * editing after a jump discards what was ahead — because this holds states and
 * not operations, and pretending otherwise would promise a replay it cannot do.
 */

import { beforeEach, describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();
const { getPreset } = await import("../src/engine.js");
const { useEditor } = await import("../src/state/store.js");

const state = () => useEditor.getState();
/** The list the panel draws: every state, oldest first. */
const labels = (): string[] => [
  ...state().past.map((p) => p.label),
  state().docLabel,
  ...state().future.map((f) => f.label),
];
const at = (): number => state().past.length;

function threeEdits(): void {
  // One store for the process, so each of these starts from an empty history
  // rather than from whatever the test before it left.
  useEditor.setState({ past: [], future: [] });
  state().setDoc(getPreset("gpt2-small"), "Opened GPT-2 Small");
  useEditor.setState({ past: [], future: [] });
  state().setSymbol("L", { kind: "design", value: 6 } as never);
  state().setSymbol("D", { kind: "design", value: 512 } as never);
  state().setMetaName("tiny");
}

describe("every state says what produced it", () => {
  beforeEach(threeEdits);

  test("in the words the toolbar used at the time", () => {
    expect(labels()).toEqual([
      "Opened GPT-2 Small",
      "Set L",
      "Set D",
      `Named the design "tiny"`,
    ]);
    expect(at()).toBe(3);
  });

  test("and an edit is announced as it happens", () => {
    expect(state().status).toBe(`Named the design "tiny"`);
  });

  test("undo says what it took back, not just that it did", () => {
    state().undo();
    expect(state().status).toBe(`Undid named the design "tiny"`);
    expect(state().docLabel).toBe("Set D");
  });
});

describe("going back to a point", () => {
  beforeEach(threeEdits);

  test("is undo several times at once", () => {
    state().jumpTo(1);
    expect(at()).toBe(1);
    expect(state().docLabel).toBe("Set L");
    expect(state().doc.symbols?.D).not.toMatchObject({ value: 512 });
    expect(state().doc.symbols?.L).toMatchObject({ value: 6 });
  });

  test("and forward again, because what is ahead is still there", () => {
    state().jumpTo(1);
    state().jumpTo(3);
    expect(state().doc.meta.name).toBe("tiny");
    expect(state().future).toHaveLength(0);
  });

  test("the list itself does not change length", () => {
    const before = labels();
    state().jumpTo(0);
    expect(labels()).toEqual(before);
    state().jumpTo(2);
    expect(labels()).toEqual(before);
  });

  test("pressing where you already are does nothing", () => {
    const before = { at: at(), doc: state().doc, status: state().status };
    state().jumpTo(at());
    expect(at()).toBe(before.at);
    expect(state().doc).toBe(before.doc);
    // And says nothing. "Back to: named the design" after pressing the row you
    // were already on is a message about something that did not happen.
    expect(state().status).toBe(before.status);
  });

  test("an index that is not there does nothing", () => {
    const before = at();
    state().jumpTo(99);
    state().jumpTo(-1);
    expect(at()).toBe(before);
  });
});

describe("what it is not", () => {
  beforeEach(threeEdits);

  test("editing after a jump discards what was ahead", () => {
    state().jumpTo(1);
    state().setMetaName("a different way");

    // States, not operations. A feature timeline would replay "Set D" and
    // "Named the design" on top of this; this cannot, and says so rather than
    // appearing to and then not.
    expect(labels()).toEqual([
      "Opened GPT-2 Small",
      "Set L",
      `Named the design "a different way"`,
    ]);
    expect(state().future).toHaveLength(0);
  });
});
