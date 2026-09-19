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
import { resolveLevel } from "./level.js";
import { derive } from "./derive.js";
import { useEditor } from "./store.js";
import { downloadDoc, downloadText } from "./serialize.js";
import { resolvedTheme, setThemePreference } from "./theme.js";
import { generateTorch, resolveSymbols } from "../engine.js";

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

  // --- edit ---------------------------------------------------------------
  {
    id: "edit.undo",
    label: "Undo",
    group: "edit",
    shortcut: `${MOD}+z`,
    run: () => editor().undo(),
    enabled: () => editor().past.length > 0,
  },
  {
    id: "edit.redo",
    label: "Redo",
    group: "edit",
    shortcut: `${MOD}+Shift+z`,
    run: () => editor().redo(),
    enabled: () => editor().future.length > 0,
  },
  {
    id: "edit.redo.alt",
    label: "Redo",
    group: "edit",
    shortcut: `${MOD}+y`,
    run: () => editor().redo(),
    enabled: () => editor().future.length > 0,
  },
  {
    id: "edit.delete",
    label: "Delete",
    group: "edit",
    shortcut: "Delete",
    run: () => {
      const { selection, detail } = editor();
      if (!selection) return;
      if (detail > 0) {
        editor().setStatus("The drawing is unfolded. Set detail to flat to edit it.");
        return;
      }
      if (editor().isLocked(selection)) {
        editor().setStatus("That block is locked. Unlock it first.");
        return;
      }
      editor().removeNode(selection);
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
      const { selection } = editor();
      if (selection) editor().toggleLock(selection);
    },
    enabled: () => editor().selection !== null,
  },
  {
    id: "edit.deselect",
    label: "Deselect, or go up a level",
    group: "edit",
    shortcut: "Escape",
    run: () => {
      const state = editor();
      if (state.selection) state.select(null);
      else if (state.path.length > 0) state.setPath(state.path.slice(0, -1));
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
    id: "view.callouts",
    label: "Annotations",
    group: "view",
    shortcut: "n",
    run: () => editor().toggleCallouts(),
    checked: () => editor().showCallouts,
  },
  {
    id: "view.shapes",
    label: "Wire labels show sizes",
    group: "view",
    shortcut: "s",
    hint: "Otherwise they show symbol names",
    run: () => {
      const { shapeMode, setShapeMode } = editor();
      setShapeMode(shapeMode === "numeric" ? "symbolic" : "numeric");
    },
    checked: () => editor().shapeMode === "numeric",
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
  { id: "panel.rules", label: "Rules", group: "panel", shortcut: "3", run: () => editor().setRightTab("rules") },


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

  // --- help ---------------------------------------------------------------
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
export function handleKey(e: KeyboardEvent): boolean {
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
