/**
 * The command list.
 *
 * It is the single list behind the keyboard, the menu, the shortcut sheet and
 * the native menu, so that a key can never be documented wrong. That holds only
 * as long as the list itself is consistent: `handleKey` looks a chord up in a
 * Map, so two commands claiming one chord means the loser silently stops
 * working while the menu goes on advertising it.
 */

import { describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

// Loaded before the import, not in a `beforeAll`: `commands.ts` reaches
// `store.ts`, which builds a starting design the moment its module runs, and
// that needs the engine. It is the same ordering `main.tsx` keeps for the same
// reason — which is worth a test noticing, since nothing else enforces it.
await loadEngine();
const { chordOf, COMMANDS, MOD, prettyShortcut } = await import("../src/state/commands.js");

/** Commands that share a chord on purpose: one action, two keys people reach for. */
const ALIASES = new Map<string, string>([
  ["edit.redo.alt", "edit.redo"],
  ["edit.delete.alt", "edit.delete"],
  ["view.zoomIn.alt", "view.zoomIn"],
]);

describe("shortcuts", () => {
  test("no two commands claim the same key", () => {
    const byChord = new Map<string, string[]>();
    for (const c of COMMANDS) {
      if (!c.shortcut) continue;
      byChord.set(c.shortcut, [...(byChord.get(c.shortcut) ?? []), c.id]);
    }
    const clashes = [...byChord.entries()]
      .filter(([, ids]) => ids.length > 1)
      // An alias is a second key for the same action, not a second action.
      .filter(([, ids]) => new Set(ids.map((id) => ALIASES.get(id) ?? id)).size > 1)
      .map(([chord, ids]) => `${chord}: ${ids.join(", ")}`);
    expect(clashes).toEqual([]);
  });

  test("an alias points at a command that exists and does the same thing", () => {
    const byId = new Map(COMMANDS.map((c) => [c.id, c]));
    for (const [alias, real] of ALIASES) {
      expect(byId.get(alias), alias).toBeDefined();
      expect(byId.get(real), real).toBeDefined();
      expect(byId.get(alias)!.label).toBe(byId.get(real)!.label);
    }
  });

  test("every id is unique", () => {
    const ids = COMMANDS.map((c) => c.id);
    expect(ids.length).toBe(new Set(ids).size);
  });

  test("a written shortcut is one a key event can actually produce", () => {
    for (const c of COMMANDS) {
      if (!c.shortcut) continue;
      // `+` is both the separator and a key people press, so a chord that ends
      // in one means the plus key rather than an empty one.
      const key = c.shortcut.endsWith("+") ? "+" : c.shortcut.slice(c.shortcut.lastIndexOf("+") + 1);
      const parts = c.shortcut.slice(0, c.shortcut.length - key.length).split("+");
      // A plain object rather than a real KeyboardEvent: there is no DOM here,
      // and `chordOf` reads five fields off whatever it is given.
      const event = {
        key: key === "Space" ? " " : key,
        ctrlKey: parts.includes(MOD) && MOD === "Ctrl",
        metaKey: parts.includes(MOD) && MOD === "Cmd",
        altKey: parts.includes("Alt"),
        shiftKey: parts.includes("Shift"),
      } as KeyboardEvent;
      expect(chordOf(event), `${c.id} claims ${c.shortcut}`).toBe(c.shortcut);
    }
  });

  test("every shortcut has a label a menu can print", () => {
    for (const c of COMMANDS) {
      expect(c.label.length, c.id).toBeGreaterThan(0);
      if (c.shortcut) expect(prettyShortcut(c.shortcut).length, c.id).toBeGreaterThan(0);
    }
  });
});
