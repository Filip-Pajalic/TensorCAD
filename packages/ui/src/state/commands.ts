/**
 * Commands.
 *
 * One list, three consumers: the keyboard, the menu, and the shortcut sheet.
 * Anything a user can do from more than one place is defined here once, so a
 * key and the menu item that teaches it can never drift apart, and so the
 * question "what can this thing do" has an answer that is data rather than a
 * tour of the source.
 */

import { blockFromGraph, defsOf, freeTypeName, mergeLibrary, toLibrary, withBlock } from "./blocks.js";
import { SHAPE_MODES } from "../canvas/shapes.js";
import { buildWalkthrough } from "./walkthrough.js";
import { resolveLevel } from "./level.js";
import { isDrillable, nodeAtPath, segmentsOf } from "./ops.js";
import { derive } from "./derive.js";
import { useEditor } from "./store.js";
import { downloadDoc, downloadText } from "./serialize.js";
import { resolvedTheme, setThemePreference } from "./theme.js";
import { sheetToSvg } from "../canvas/svg.js";
import { catalogOf, generateTorch, resolveSymbols } from "../engine.js";

export type CommandGroup = "file" | "edit" | "view" | "panel" | "blocks" | "help";

export interface Command {
  id: string;
  label: string;
  group: CommandGroup;
  /** Written the way a menu writes it, e.g. `Ctrl+Shift+S`. */
  shortcut?: string;
  /** One line for the shortcut sheet, when the label is not self-explanatory. */
  hint?: string;
  run: () => void;
  /** False greys the menu item out. */
  enabled?: () => boolean;
  /** Present for a command that toggles something, so the menu can tick it. */
  checked?: () => boolean;
}

const isMac = typeof navigator !== "undefined" && /Mac|iP(hone|ad)/.test(navigator.platform);
/** The platform's command key, spelled the way that platform spells it. */
export const MOD = isMac ? "Cmd" : "Ctrl";

/**
 * What a key event matches, normalised: modifiers in a fixed order, the key
 * lower-cased, and the platform's command key folded into one name.
 */
export function chordOf(e: KeyboardEvent): string {
  const parts: string[] = [];
  if (e.ctrlKey || e.metaKey) parts.push(MOD);
  if (e.altKey) parts.push("Alt");
  // Shift is only named when it is not already spelled by the character. `?`
  // is typed with shift but written `?`, never `Shift+/`, and a chord that
  // said otherwise would never match what the menu shows.
  const printable = e.key.length === 1;
  const letter = printable && /[a-z]/i.test(e.key);
  if (e.shiftKey && (!printable || letter)) parts.push("Shift");
  const key = printable ? e.key.toLowerCase() : e.key;
  parts.push(key === " " ? "Space" : key);
  return parts.join("+");
}

/** Normalise a written shortcut the same way, so the two can be compared. */
function normalise(shortcut: string): string {
  return shortcut
    .split("+")
    .map((part) => {
      const p = part.trim();
      if (p === "Ctrl" || p === "Cmd" || p === "Mod") return MOD;
      if (p.length === 1) return p.toLowerCase();
      return p;
    })
    .join("+");
}

/**
 * The viewport is owned by React Flow, which only exists inside the canvas, so
 * the canvas hands these up when it mounts. Commands that need the viewport
 * simply do nothing before it has.
 */
export interface ViewportApi {
  fit: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  zoomReset: () => void;
  duplicateSelection: () => void;
}

let viewport: ViewportApi | null = null;
export function setViewportApi(api: ViewportApi | null): void {
  viewport = api;
}

const editor = (): ReturnType<typeof useEditor.getState> => useEditor.getState();

function exportTorch(): void {
  const { doc } = editor();
  const generated = generateTorch(doc);
  for (const file of generated.files) {
    downloadText(file.contents, `${doc.meta.name || "model"}-${file.path.replace(/\//g, "-")}`);
  }
  editor().setStatus(
    generated.warnings.length > 0
      ? `Exported ${generated.files.length} file(s) with ${generated.warnings.length} warning(s)`
      : `Exported ${generated.files.length} file(s)`,
  );
}

export const COMMANDS: Command[] = [
  // --- file ---------------------------------------------------------------
  { id: "file.new", label: "New design", group: "file", shortcut: `${MOD}+n`, run: () => editor().newDoc() },
  {
    id: "file.open",
    label: "Open…",
    group: "file",
    shortcut: `${MOD}+o`,
    hint: "Read a design back from JSON",
    run: () => document.getElementById("tensorcad-open-input")?.click(),
  },
  {
    id: "file.save",
    label: "Save a copy",
    group: "file",
    shortcut: `${MOD}+s`,
    run: () => {
      const { doc } = editor();
      downloadDoc(doc);
      editor().setStatus(`Saved ${doc.meta.name}.json`);
    },
  },
  {
    id: "file.export",
    label: "Export PyTorch",
    group: "file",
    shortcut: `${MOD}+e`,
    hint: "Generate model.py for this design",
    run: exportTorch,
  },
  {
    id: "file.exportSvg",
    label: "Export the sheet as SVG",
    group: "file",
    shortcut: `${MOD}+Shift+e`,
    hint: "The drawing as a vector, for a paper or a slide",
    run: () => {
      // The sheet as it is drawn, so what is exported is what is on screen:
      // the level you have open, at the detail you have it open to.
      const root = document.querySelector<HTMLElement>(".react-flow");
      const { doc } = editor();
      const svg = root ? sheetToSvg(root, doc.meta.name) : undefined;
      if (!svg) {
        editor().setStatus("There is nothing on the sheet to export.");
        return;
      }
      downloadText(svg, `${doc.meta.name}.svg`, "image/svg+xml");
      editor().setStatus(`Exported ${doc.meta.name}.svg`);
    },
  },

  // --- edit ---------------------------------------------------------------
  {
    id: "edit.undo",
    label: "Undo",
    group: "edit",
    shortcut: `${MOD}+z`,
    run: () => editor().undo(),
    enabled: () => editor().at > 0,
  },
  {
    id: "edit.redo",
    label: "Redo",
    group: "edit",
    shortcut: `${MOD}+Shift+z`,
    run: () => editor().redo(),
    enabled: () => editor().at < editor().steps.length,
  },
  {
    id: "edit.redo.alt",
    label: "Redo",
    group: "edit",
    shortcut: `${MOD}+y`,
    run: () => editor().redo(),
    enabled: () => editor().at < editor().steps.length,
  },
  {
    id: "edit.delete",
    label: "Delete",
    group: "edit",
    shortcut: "Delete",
    run: () => {
      const state = editor();
      const chosen = state.selected();
      if (chosen.length === 0) return;
      if (state.detail > 0) {
        state.setStatus("The drawing is unfolded. Set detail to flat to edit it.");
        return;
      }
      const locked = chosen.filter((path) => state.isLocked(path));
      const free = chosen.filter((path) => !state.isLocked(path));
      if (free.length === 0) {
        state.setStatus(
          chosen.length === 1
            ? "That block is locked. Unlock it first."
            : "Every one of those is locked. Unlock them first.",
        );
        return;
      }
      // Deepest first, so removing one does not shift the path of the next.
      for (const path of [...free].sort((a, b) => b.split("/").length - a.split("/").length)) {
        editor().removeNode(path);
      }
      if (locked.length > 0) {
        editor().setStatus(
          `Deleted ${free.length}; left ${locked.length} locked block${locked.length === 1 ? "" : "s"} alone.`,
        );
      }
    },
    enabled: () => editor().selection !== null,
  },
  {
    id: "edit.delete.alt",
    label: "Delete",
    group: "edit",
    shortcut: "Backspace",
    run: () => runCommand("edit.delete"),
    enabled: () => editor().selection !== null,
  },
  {
    id: "edit.duplicate",
    label: "Duplicate",
    group: "edit",
    shortcut: `${MOD}+d`,
    run: () => viewport?.duplicateSelection(),
    enabled: () => editor().selection !== null && editor().detail === 0,
  },
  {
    id: "edit.lock",
    label: "Lock or unlock",
    group: "edit",
    shortcut: `${MOD}+l`,
    hint: "A locked block cannot be dragged and auto-layout leaves it alone",
    run: () => {
      const state = editor();
      const chosen = state.selected();
      if (chosen.length === 0) return;
      // What the primary is about to become is what they all become, so a
      // mixed selection ends up in one state rather than inverted piecemeal.
      const locking = !state.isLocked(chosen[chosen.length - 1]!);
      for (const path of chosen) {
        if (editor().isLocked(path) !== locking) editor().toggleLock(path);
      }
    },
    enabled: () => editor().selection !== null,
  },
  {
    id: "edit.grow",
    label: "Grow selection along the wires",
    group: "edit",
    shortcut: "Shift+g",
    hint: "Add everything directly wired to what is selected. Press again to go further.",
    enabled: () => editor().selected().length > 0,
    run: () => {
      const state = editor();
      const have = state.selected();
      if (have.length === 0) return;

      const derived = derive(state.doc, state.operating);
      const next = new Set(have);
      // One junction at a time, because that is the gesture: press again to go
      // further. Growing to the whole net in one press would make the common
      // case — "this block and what feeds it" — unreachable.
      for (const [consumer, producer] of derived.infer.producerOf) {
        const from = producer.slice(0, producer.lastIndexOf(":"));
        const to = consumer.slice(0, consumer.lastIndexOf(":"));
        if (next.has(from)) next.add(to);
        if (next.has(to)) next.add(from);
      }
      if (next.size === have.length) {
        state.setStatus("Nothing further is wired to this.");
        return;
      }
      // The primary stays the primary: the inspector should not jump to some
      // other block because the selection grew around it.
      const grown = [...next].filter((p) => p !== have[have.length - 1]);
      grown.push(have[have.length - 1]!);
      state.selectPaths(grown);
      state.setStatus(`Selection grew to ${grown.length} blocks.`);
    },
  },
  {
    id: "edit.deselect",
    label: "Deselect, or go up a level",
    group: "edit",
    shortcut: "Escape",
    run: () => {
      const state = editor();
      // What was selected when the key went down, not now: React Flow
      // deselects a focused block on Escape itself, before this runs, and
      // going by the state it leaves would take one press as two.
      const had = selectionAtKeyDown === undefined ? state.selection : selectionAtKeyDown;
      if (had || state.selection) state.select(null);
      else if (state.path.length > 0) {
        // Back out onto the block that was just left, as a file manager
        // puts you back on the folder you came up out of.
        const left = state.path.join("/");
        state.setPath(state.path.slice(0, -1));
        refocus(left);
      }
    },
  },

  // --- view ---------------------------------------------------------------
  { id: "view.fit", label: "Fit to window", group: "view", shortcut: "f", run: () => viewport?.fit() },
  { id: "view.zoomIn", label: "Zoom in", group: "view", shortcut: `${MOD}+=`, run: () => viewport?.zoomIn() },
  { id: "view.zoomIn.alt", label: "Zoom in", group: "view", shortcut: `${MOD}++`, run: () => viewport?.zoomIn() },
  { id: "view.zoomOut", label: "Zoom out", group: "view", shortcut: `${MOD}+-`, run: () => viewport?.zoomOut() },
  { id: "view.zoomReset", label: "Zoom to 100%", group: "view", shortcut: `${MOD}+0`, run: () => viewport?.zoomReset() },
  {
    id: "view.arrange",
    label: "Arrange",
    group: "view",
    shortcut: `${MOD}+r`,
    hint: "Re-run the automatic layout",
    run: () => editor().requestLayout(),
  },
  {
    id: "view.detailIn",
    label: "More detail",
    group: "view",
    shortcut: "]",
    hint: "Open one more level of container in place",
    run: () => editor().setDetail(editor().detail + 1),
  },
  {
    id: "view.detailOut",
    label: "Less detail",
    group: "view",
    shortcut: "[",
    run: () => editor().setDetail(editor().detail - 1),
  },
  {
    id: "view.open",
    label: "Open block",
    group: "view",
    shortcut: "Enter",
    hint: "Go inside the selected block. Escape comes back out.",
    // Double-click was the only way in, which left a keyboard with no way
    // down the hierarchy at all. Enter is what opens a thing everywhere else;
    // it counts only while the drawing has focus, because on a focused button
    // it already means press.
    enabled: () => {
      if (!drawingHasFocus()) return false;
      const { doc, selection } = editor();
      if (!selection) return false;
      const node = nodeAtPath(doc, segmentsOf(selection));
      return node !== null && isDrillable(node, catalogOf(doc)[node.type]);
    },
    run: () => {
      const state = editor();
      const path = state.selection;
      // In an unfolded drawing a block is opened in place, as a double-click does.
      if (state.detail > 0) {
        state.setDetail(state.detail + 1);
        refocus(path);
      } else if (path) {
        state.enter(path);
        refocus(null);
      }
    },
  },
  {
    id: "file.library",
    label: "Reference architectures\u2026",
    group: "file",
    hint: "The ported designs, and what each one is",
    run: () => editor().openDialog("library"),
  },
  {
    id: "view.walkthrough",
    label: "Walk me through it",
    group: "view",
    shortcut: "w",
    hint: "What each stage of this design does, on this design, with its numbers",
    run: () => {
      const state = editor();
      if (state.walkthrough !== null) return state.endWalkthrough();
      // A design with no blocks has no steps, and the panel renders nothing —
      // so the command reported itself as on and put nothing on screen, which
      // is the same silent success the rest of this pass is about.
      if (buildWalkthrough(state.doc, derive(state.doc, state.operating)).length === 0) {
        state.setStatus("Nothing to walk through yet: this design has no blocks.");
        return;
      }
      state.startWalkthrough();
    },
    checked: () => editor().walkthrough !== null,
  },
  {
    id: "view.figure",
    label: "Draw it as a figure",
    group: "view",
    shortcut: "g",
    hint: "Leave out the reshapes, the way a published figure does",
    run: () => editor().setFigure(!editor().figure),
    checked: () => editor().figure,
  },
  {
    id: "view.key",
    label: "Key",
    group: "view",
    shortcut: "k",
    hint: "What the letters and the marks on the sheet mean",
    run: () => editor().setShowKey(!editor().showKey),
    checked: () => editor().showKey,
  },
  {
    id: "view.callouts",
    label: "Annotations",
    group: "view",
    shortcut: "n",
    run: () => editor().toggleCallouts(),
    checked: () => editor().showCallouts,
  },
  {
    id: "view.shapes",
    label: "How wire labels are written",
    group: "view",
    shortcut: "s",
    hint: "Cycles: symbol names, sizes, then plain English",
    run: () => {
      const { shapeMode, setShapeMode } = editor();
      const at = SHAPE_MODES.indexOf(shapeMode);
      setShapeMode(SHAPE_MODES[(at + 1) % SHAPE_MODES.length]!);
    },
  },
  {
    id: "view.volume",
    label: "Volume view",
    group: "view",
    shortcut: "Shift+v",
    hint: "The design as boxes at their real proportions, instead of a schematic",
    run: () => {
      const { viewMode, setViewMode } = editor();
      setViewMode(viewMode === "volume" ? "sheet" : "volume");
    },
    checked: () => editor().viewMode === "volume",
  },
  {
    id: "view.dock.left",
    label: "Left dock",
    group: "view",
    shortcut: `${MOD}+b`,
    hint: "Collapse the model tree to a rail",
    run: () => editor().toggleDock("left"),
    checked: () => !editor().leftCollapsed,
  },
  {
    id: "view.dock.right",
    label: "Right dock",
    group: "view",
    shortcut: `${MOD}+Shift+b`,
    hint: "Collapse the readout to a rail",
    run: () => editor().toggleDock("right"),
    checked: () => !editor().rightCollapsed,
  },
  {
    id: "view.focus",
    label: "Both docks",
    group: "view",
    shortcut: `${MOD}+Alt+b`,
    hint: "Collapse or restore them together, for a clean sheet",
    run: () => {
      const s = editor();
      const collapse = !s.leftCollapsed || !s.rightCollapsed;
      if (s.leftCollapsed === collapse) s.toggleDock("left");
      if (s.rightCollapsed === collapse) s.toggleDock("right");
    },
  },
  {
    id: "view.palette",
    label: "Parts palette",
    group: "view",
    shortcut: "p",
    run: () => editor().togglePalette(),
    checked: () => editor().paletteOpen,
  },
  {
    id: "view.theme",
    label: "Dark mode",
    group: "view",
    shortcut: `${MOD}+Shift+d`,
    run: () => setThemePreference(resolvedTheme() === "dark" ? "light" : "dark"),
    checked: () => resolvedTheme() === "dark",
  },

  // --- panels -------------------------------------------------------------
  { id: "panel.inspector", label: "Inspector", group: "panel", shortcut: "1", run: () => editor().setRightTab("inspector") },
  { id: "panel.symbols", label: "Symbols", group: "panel", shortcut: "2", run: () => editor().setRightTab("symbols") },
  { id: "panel.cluster", label: "Cluster", group: "panel", shortcut: "3", run: () => editor().setRightTab("cluster") },
  { id: "panel.ladder", label: "Ladder", group: "panel", shortcut: "4", run: () => editor().setRightTab("ladder") },
  {
    id: "panel.findings",
    label: "Checks",
    group: "panel",
    shortcut: `${MOD}+Shift+f`,
    hint: "The design-rule list, along the bottom",
    run: () => editor().toggleFindings(),
    checked: () => editor().dockOpen,
  },

  // --- blocks -------------------------------------------------------------
  {
    id: "blocks.fromLevel",
    label: "Make a block from this level\u2026",
    group: "blocks",
    hint: "Turn what is on screen into a reusable block, with its symbols as parameters",
    run: () => {
      const state = editor();
      const { doc } = state;
      const derived = derive(doc, state.operating);
      const level = resolveLevel(doc, state.path, derived);
      if (level.graph.nodes.length === 0) {
        state.setStatus("Nothing on this level to make a block from.");
        return;
      }
      const suggested = level.owner?.type ?? doc.meta.name ?? "block";
      const wanted = window.prompt("Name for the new block", suggested);
      if (!wanted) return;

      const symbols = resolveSymbols(doc);
      const { def, errors } = blockFromGraph(
        level.graph,
        freeTypeName(doc, wanted),
        symbols.designValues,
        `From ${doc.meta.name}.`,
      );
      if (errors.length > 0) {
        state.setStatus(`Could not make a block: ${errors[0]}`);
        return;
      }
      state.setDoc(withBlock(doc, def), `Defined block "${def.type}"`);
      if (!editor().paletteOpen) editor().togglePalette();
    },
  },
  {
    id: "blocks.library",
    label: "Blocks this design defines…",
    group: "blocks",
    hint: "List them, open one for editing, rename or delete",
    run: () => editor().openDialog("definitions"),
  },
  {
    id: "blocks.import",
    label: "Import blocks\u2026",
    group: "blocks",
    hint: "Read a block library into this design",
    run: () => document.getElementById("tensorcad-blocks-input")?.click(),
  },
  {
    id: "blocks.export",
    label: "Export blocks",
    group: "blocks",
    hint: "Write this design's own blocks out as a library",
    run: () => {
      const { doc } = editor();
      const count = Object.keys(defsOf(doc)).length;
      if (count === 0) {
        editor().setStatus("This design defines no blocks of its own.");
        return;
      }
      downloadText(
        JSON.stringify(toLibrary(doc), null, 2),
        `${doc.meta.name || "design"}.blocks.json`,
        "application/json",
      );
      editor().setStatus(`Exported ${count} block${count === 1 ? "" : "s"}`);
    },
    enabled: () => Object.keys(defsOf(editor().doc)).length > 0,
  },

  {
    id: "view.compare",
    label: "Compare…",
    group: "view",
    // Not Mod+D, which is Duplicate and was here first. They both had it for a
    // while, and since the chord map keeps the last one written, Duplicate's
    // key quietly stopped working while the menu went on advertising it.
    shortcut: `${MOD}+Shift+c`,
    run: () => editor().openDialog("compare"),
  },
  {
    id: "view.mark-baseline",
    label: "Compare against this from now on",
    group: "view",
    run: () => editor().markOpened(),
  },

  // --- help ---------------------------------------------------------------
  {
    id: "help.palette",
    label: "Commands…",
    group: "help",
    shortcut: `${MOD}+k`,
    hint: "Every command by name, including the ones with no shortcut.",
    run: () => editor().openDialog("palette"),
  },
  {
    id: "help.settings",
    label: "Settings…",
    group: "help",
    shortcut: `${MOD}+,`,
    run: () => editor().openDialog("settings"),
  },
  {
    id: "help.shortcuts",
    label: "Keyboard shortcuts",
    group: "help",
    shortcut: "?",
    run: () => editor().openDialog("shortcuts"),
  },
];

export const COMMAND_BY_ID = new Map(COMMANDS.map((c) => [c.id, c]));

export function runCommand(id: string): void {
  const command = COMMAND_BY_ID.get(id);
  if (!command) return;
  if (command.enabled && !command.enabled()) return;
  command.run();
}

/** Chord to command, built once. Later entries do not overwrite earlier ones. */
const BY_CHORD = new Map<string, Command>();
for (const command of COMMANDS) {
  if (!command.shortcut) continue;
  const chord = normalise(command.shortcut);
  if (!BY_CHORD.has(chord)) BY_CHORD.set(chord, command);
}

/**
 * Handle a key press. Returns true when a command ran, so the caller knows
 * whether to swallow the event.
 *
 * Typing in a field is never a shortcut, with one exception: the modified
 * chords still work, because Ctrl+S while the cursor sits in a name field
 * should still save.
 */
/**
 * Put the keyboard back on a block once the drawing has redrawn around it.
 *
 * Opening a level replaces every node, and the one that had focus goes with
 * it: focus falls to the page, and the next Tab starts from the top of the
 * window. Null means the first block of the new level.
 */
function refocus(path: string | null): void {
  if (typeof document === "undefined") return;
  // Retried rather than timed. The new nodes are rendered at once but kept
  // invisible until React Flow has measured them, and an invisible element
  // refuses focus — how long that takes is up to the browser's frames, which
  // a background tab may not be drawing at all.
  let tries = 0;
  const attempt = (): void => {
    const nodes = document.querySelectorAll<HTMLElement>(".react-flow__node");
    const target = (path && [...nodes].find((n) => n.dataset.id === path)) || nodes[0];
    target?.focus();
    if ((!target || document.activeElement !== target) && ++tries < 40) setTimeout(attempt, 25);
  };
  setTimeout(attempt, 0);
}

/** Whether keys are the drawing's: nothing focused, or something inside the sheet. */
function drawingHasFocus(): boolean {
  if (typeof document === "undefined") return true;
  const active = document.activeElement;
  return !active || active === document.body || active.closest(".react-flow") !== null;
}

/** The selection as the key being handled went down, before the drawing saw it. Unset outside a key. */
let selectionAtKeyDown: string | null | undefined;

/**
 * Note what is selected before anything handles a key. Registered in the
 * capture phase, so it runs ahead of React Flow's own handlers on the node.
 */
export function beforeKey(): void {
  selectionAtKeyDown = editor().selection;
}

export function handleKey(e: KeyboardEvent): boolean {
  try {
    return dispatchKey(e);
  } finally {
    selectionAtKeyDown = undefined;
  }
}

function dispatchKey(e: KeyboardEvent): boolean {
  const target = e.target as HTMLElement | null;
  const typing =
    target !== null &&
    (/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) || target.isContentEditable);
  const modified = e.ctrlKey || e.metaKey || e.altKey;
  if (typing && !modified) return false;

  const command = BY_CHORD.get(chordOf(e));
  if (!command) return false;
  if (command.enabled && !command.enabled()) return true;
  e.preventDefault();
  command.run();
  return true;
}

/** Commands in a group, minus the duplicate chords that share a label. */
export function commandsIn(group: CommandGroup): Command[] {
  const seen = new Set<string>();
  return COMMANDS.filter((c) => {
    if (c.group !== group) return false;
    if (seen.has(c.label)) return false;
    seen.add(c.label);
    return true;
  });
}

/** How a shortcut is written for the reader, rather than for the matcher. */
export function prettyShortcut(shortcut: string | undefined): string {
  if (!shortcut) return "";
  return shortcut
    .split("+")
    .map((part) => {
      const p = part.trim();
      if (p.length === 1) return p.toUpperCase();
      if (p === "Delete") return "Del";
      if (p === "Escape") return "Esc";
      if (p === "Backspace") return "⌫";
      return p;
    })
    .join("+");
}
