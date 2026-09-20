/**
 * Editor state.
 *
 * The document is the single source of truth; the canvas is a view of it. Every
 * mutation goes through `commit`, which pushes the previous document onto the
 * undo stack, so undo/redo needs no per-operation inverse.
 */

import { create } from "zustand";
import type { ShapeMode } from "../canvas/shapes.js";
import * as ops from "./ops.js";
import type { Segments } from "./ops.js";
import { DEFAULT_OPERATING, loadOperating, saveOperating, type OperatingPoint } from "./operating.js";
import type { Doc, NodeDef, ParamValue, RuleSeverity, SymbolDef } from "@tensorcad/engine";
import { getPreset } from "../engine.js";

const UNDO_LIMIT = 100;

/** The last segment of a path, which is what a person calls the block. */
const lastOf = (path: string): string => path.split("/").pop() ?? path;

/**
 * Two levels open a `repeat` holding a `transformer_block`, which is exactly
 * the drawing every architecture figure shows: the stack, and inside it the
 * norms, the attention, the feed-forward and the two residual sums.
 */
const DEFAULT_DETAIL = 2;
const MAX_DETAIL = 5;
const DETAIL_KEY = "tensorcad.detail";

/** View preferences are remembered; losing them is harmless if storage fails. */
function loadFlag(name: string, fallback: boolean): boolean {
  try {
    const raw = localStorage.getItem(`tensorcad.${name}`);
    return raw === null ? fallback : raw === "1";
  } catch {
    return fallback;
  }
}

function saveFlag<T>(name: string, value: boolean, patch: T): T {
  try {
    localStorage.setItem(`tensorcad.${name}`, value ? "1" : "0");
  } catch {
    // Not remembering it is harmless.
  }
  return patch;
}

function loadDetail(): number {
  try {
    const raw = localStorage.getItem(DETAIL_KEY);
    if (raw === null) return DEFAULT_DETAIL;
    const n = Number(raw);
    return Number.isFinite(n) ? Math.max(0, Math.min(MAX_DETAIL, Math.floor(n))) : DEFAULT_DETAIL;
  } catch {
    return DEFAULT_DETAIL;
  }
}

/** One state in the history, and what was done to reach it. */
export interface Step {
  doc: Doc;
  label: string;
}

export type RightTab = "inspector" | "symbols" | "cluster" | "ladder" | "runs" | "history";
export type DialogId = "settings" | "shortcuts" | "compare" | "palette" | "definitions" | null;

/**
 * The active tool, the way any drawing program has one. Select edits, pan moves
 * the sheet, and wire is a dedicated connect mode so a long drag across a dense
 * drawing cannot accidentally move a block instead.
 */
export type Tool = "select" | "pan" | "wire";

/**
 * How the design is drawn. The sheet is the editable schematic; the volume is
 * the same design as boxes at their real proportions, which answers "where are
 * the parameters" in a way a table cannot.
 */
export type ViewMode = "sheet" | "volume";
export type { ShapeMode } from "../canvas/shapes.js";

export interface EditorState {
  doc: Doc;
  /**
   * The design as it stood when this one was opened, which is the other side of
   * "what have I changed". Set by loading, opening or starting a design, and by
   * asking for it; never by an edit.
   */
  opened: Doc;
  /** Take the design as it stands now as the baseline to compare against. */
  markOpened: () => void;
  /** Breadcrumb path of the graph level on screen. */
  path: string[];
  /**
   * Full path of the selected node, or null.
   *
   * The *primary* selection: the one the inspector edits, the one the volume
   * view highlights, the one a finding points at. It stays a single path
   * because each of those answers exactly one block, and asking them to mean
   * "one of several" would make every one of them worse.
   */
  selection: string | null;
  /**
   * The rest of a multiple selection, primary excluded.
   *
   * Kept beside `selection` rather than replacing it, so everything that wants
   * one block still gets one and only what can act on several has to know
   * there are several.
   */
  also: string[];
  /**
   * The selected net, as the producing pin `"path:port"`, or null.
   *
   * A tensor is a thing in this design, not a line between two things: it has
   * a shape, a dtype, one producer, however many consumers, and a share of the
   * activation memory that the analysis has always attributed to it rather
   * than to a block. Until now the drawing was the one place it was not
   * selectable, so the only way to ask what a wire cost was to guess which
   * block to click.
   *
   * Kept beside `selection` rather than folded into it: everything that acts
   * on a block takes a node path, and teaching all of it that a path might be
   * a wire would make every one of those worse. A net and a block are never
   * selected at once.
   */
  selectedNet: string | null;
  selectNet: (net: string | null) => void;
  /**
   * The history, as states rather than as a stack of documents alone.
   *
   * Each step carries the document *and* what was done to get it, because an
   * undo stack you cannot read is one you have to step through blindly: three
   * presses of Ctrl+Z to find out whether the thing you regret was four edits
   * ago. The label is the same sentence the toolbar shows when the edit
   * happens, so what you read then is what you read later.
   *
   * Still a pair of arrays and not a list with an index, because `past.length`
   * and `future.length` are what the toolbar and the command list ask, and a
   * refactor that made those two subtractions would be a refactor for its own
   * sake.
   */
  past: Step[];
  future: Step[];
  /** What produced the document on screen. The current step's label. */
  docLabel: string;
  /**
   * Go to a point in the history directly.
   *
   * The index is into `past.concat(current, future)`, which is the list the
   * panel shows. Several undos in one gesture and the drawing lands where the
   * row said it would — the property a list is worth having for.
   */
  jumpTo: (index: number) => void;
  rightTab: RightTab;
  /**
   * Whether the findings dock along the bottom is open.
   *
   * The findings were a tab, which meant they were never on screen while you
   * edited — you had to leave the inspector to find out what was wrong, and
   * then leave the findings to fix it. A DRC list belongs where a PCB tool puts
   * it: across the bottom, under the drawing it is about.
   *
   * Collapsed it is still a strip carrying the counts, so a design that is
   * broken always says so somewhere on screen.
   */
  dockOpen: boolean;
  toggleFindings: () => void;
  setDockOpen: (open: boolean) => void;
  /** Whether shape labels show symbol names or substituted design values. */
  shapeMode: ShapeMode;
  /** Bumped whenever something wants the canvas to re-run auto-layout. */
  layoutNonce: number;
  /** Bumped whenever something wants the canvas to centre on the selection. */
  focusNonce: number;
  /**
   * The block whose findings the rules list is showing, set by pressing a
   * marker on the drawing. Null means the list is showing everything, which is
   * what it does the rest of the time.
   */
  findingFocus: string | null;
  /** Bumped alongside it, so pressing the same marker twice scrolls again. */
  findingNonce: number;
  /** Transient message for the status bar. */
  status: string | null;
  /** The conditions every number past the parameter count is measured under. */
  operating: OperatingPoint;
  setOperating: (patch: Partial<OperatingPoint>) => void;
  resetOperating: () => void;
  /** Whether the parts palette is open in the left column. */
  paletteOpen: boolean;
  togglePalette: () => void;
  /**
   * How many levels of container to draw open. Zero is one flat, editable,
   * hand-placed graph. Above zero the drawing is generated: containers are
   * drawn as frames around their contents, the way every published figure
   * draws an architecture, and the layout engine places it.
   */
  detail: number;
  setDetail: (levels: number) => void;
  tool: Tool;
  setTool: (tool: Tool) => void;
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  /** Whether each side dock is collapsed to a rail, the way Fusion's is. */
  leftCollapsed: boolean;
  rightCollapsed: boolean;
  toggleDock: (side: "left" | "right") => void;
  /** Whether the canvas snaps positions to the grid. */
  snap: boolean;
  setSnap: (snap: boolean) => void;
  /** Whether the grid is drawn at all. */
  showGrid: boolean;
  setShowGrid: (show: boolean) => void;
  /** Whether the minimap is drawn. */
  showMinimap: boolean;
  setShowMinimap: (show: boolean) => void;
  /** Whether the title block is drawn on the sheet. */
  showTitleBlock: boolean;
  setShowTitleBlock: (show: boolean) => void;
  /** Which modal is open, if any. */
  dialog: DialogId;
  openDialog: (dialog: DialogId) => void;
  closeDialog: () => void;

  setDoc: (doc: Doc, status?: string) => void;
  /**
   * A document that arrived from somewhere else — the live agent bridge.
   *
   * Not `setDoc`, which is *opening* a design: that resets the baseline, the
   * level and the selection, all of which would be wrong here. Somebody is
   * watching an agent work, and having the view thrown back to the top sheet
   * every time it touches a symbol is how you stop watching.
   *
   * It goes on the undo stack, so an agent's edit can be taken back by the
   * person who saw it happen.
   */
  applyRemote: (doc: Doc, status?: string) => void;
  setPath: (path: string[]) => void;
  enter: (path: string) => void;
  select: (path: string | null) => void;
  /** Select several at once. The last is the primary. */
  selectPaths: (paths: string[]) => void;
  /** Everything selected, primary last. Empty when nothing is. */
  selected: () => string[];
  /** What the canvas is showing, for the status bar. */
  canvasStatus: CanvasStatus;
  setCanvasStatus: (patch: Partial<CanvasStatus>) => void;
  /** Lock or unlock a block's position. */
  toggleLock: (path: string) => void;
  /** True when this block's position is fixed. */
  isLocked: (path: string) => boolean;
  /** Open the level that owns `path`, select it and centre the viewport. */
  focusOn: (path: string) => void;
  /** Open the rules list on one block's findings. */
  showFindingsFor: (path: string) => void;
  clearFindingFocus: () => void;
  setRightTab: (tab: RightTab) => void;
  setShapeMode: (mode: ShapeMode) => void;
  /** Whether the drawing is annotated with callouts. */
  showCallouts: boolean;
  toggleCallouts: () => void;
  setStatus: (status: string | null) => void;
  requestLayout: () => void;

  undo: () => void;
  redo: () => void;

  addNode: (parent: Segments, node: NodeDef, xy?: [number, number]) => void;
  removeNode: (path: string) => void;
  setParam: (path: string, key: string, value: ParamValue | undefined) => void;
  connect: (parent: Segments, from: string, to: string) => void;
  disconnect: (parent: Segments, from: string, to: string) => void;
  /**
   * Move one end of an existing wire.
   *
   * One commit, not a disconnect followed by a connect, because dragging a wire
   * to a different pin is one thing a person did and undo should put it back in
   * one press.
   */
  reconnect: (
    parent: Segments,
    from: string,
    to: string,
    next: { from: string; to: string },
  ) => void;
  /**
   * What this design has decided a rule means to it. Undefined removes the
   * decision, putting the rule back to what it says about itself.
   */
  setRuleSeverity: (rule: string, severity: RuleSeverity | undefined) => void;
  setSymbol: (name: string, def: SymbolDef | undefined) => void;
  /** Build the design at a named configuration, or at its own symbols. */
  setActiveConfiguration: (name: string | null) => void;
  /** Name the symbols as they stand now. */
  captureConfiguration: (name: string, doc?: string) => void;
  removeConfiguration: (name: string) => void;
  renameSymbol: (from: string, to: string) => void;
  moveNode: (path: string, xy: [number, number]) => void;
  /**
   * `label` because the two callers are not the same event: one is a person
   * dragging blocks and the other is auto-layout placing them. A history that
   * called both "Moved 6 blocks" would credit the machine's work to the
   * person, who would then look for the drag they do not remember making.
   */
  moveNodes: (moves: { path: string; xy: [number, number] }[], label?: string) => void;
  renameNode: (path: string, label: string | undefined) => void;
  setNodeId: (path: string, id: string) => void;
  setMetaName: (name: string) => void;
  loadPreset: (name: string) => void;
  newDoc: () => void;
}

/** Live canvas state that is view-only: never part of the document. */
export interface CanvasStatus {
  cursor: { x: number; y: number } | null;
  zoom: number;
  gridMinor: number;
  gridMajor: number;
  nodeCount: number;
  edgeCount: number;
}

const INITIAL_STATUS: CanvasStatus = {
  cursor: null,
  zoom: 1,
  gridMinor: 8,
  gridMajor: 80,
  nodeCount: 0,
  edgeCount: 0,
};

const initialDoc = getPreset("llama-3-8b");

export const useEditor = create<EditorState>((set, get) => {
  /**
   * Apply a pure document operation and record it for undo.
   *
   * `label` is not optional any more. It is what the history list shows and
   * what the toolbar says while the edit is fresh, and an unlabelled row in a
   * list of edits is a row you have to reconstruct from the drawing.
   */
  const commit = (fn: (doc: Doc) => Doc, label: string): void => {
    const { doc, past, docLabel } = get();
    const next = fn(doc);
    if (next === doc) return;
    set({
      doc: next,
      past: [...past.slice(-(UNDO_LIMIT - 1)), { doc, label: docLabel }],
      future: [],
      docLabel: label,
      status: label,
    });
  };

  return {
    doc: initialDoc,
    opened: initialDoc,
    path: [],
    selection: null,
    also: [],
    selectedNet: null,
    past: [],
    future: [],
    docLabel: "Opened Llama 3 8B",
    rightTab: "inspector",
    dockOpen: false,
    shapeMode: "symbolic",
    layoutNonce: 0,
    focusNonce: 0,
    findingFocus: null,
    findingNonce: 0,
    status: null,

    operating: loadOperating(),
    setOperating: (patch) =>
      set((s) => {
        const operating = { ...s.operating, ...patch };
        saveOperating(operating);
        return { operating };
      }),
    resetOperating: () => {
      const operating = { ...DEFAULT_OPERATING };
      saveOperating(operating);
      set({ operating });
    },

    paletteOpen: false,
    togglePalette: () => set((s) => ({ paletteOpen: !s.paletteOpen })),

    tool: "select",
    setTool: (tool) => set({ tool }),
    viewMode: loadFlag("volume", false) ? "volume" : "sheet",
    setViewMode: (viewMode) => set(saveFlag("volume", viewMode === "volume", { viewMode })),
    leftCollapsed: loadFlag("dock.left", false),
    rightCollapsed: loadFlag("dock.right", false),
    toggleDock: (side) =>
      set((s) =>
        side === "left"
          ? saveFlag("dock.left", !s.leftCollapsed, { leftCollapsed: !s.leftCollapsed })
          : saveFlag("dock.right", !s.rightCollapsed, { rightCollapsed: !s.rightCollapsed }),
      ),
    snap: loadFlag("snap", true),
    setSnap: (snap) => set(saveFlag("snap", snap, { snap })),
    showGrid: loadFlag("grid", true),
    setShowGrid: (showGrid) => set(saveFlag("grid", showGrid, { showGrid })),
    showMinimap: loadFlag("minimap", true),
    setShowMinimap: (showMinimap) => set(saveFlag("minimap", showMinimap, { showMinimap })),
    showTitleBlock: loadFlag("titleblock", true),
    setShowTitleBlock: (showTitleBlock) =>
      set(saveFlag("titleblock", showTitleBlock, { showTitleBlock })),

    dialog: null,
    openDialog: (dialog) => set({ dialog }),
    closeDialog: () => set({ dialog: null }),

    detail: loadDetail(),
    setDetail: (levels) => {
      const detail = Math.max(0, Math.min(MAX_DETAIL, Math.floor(levels)));
      try {
        localStorage.setItem(DETAIL_KEY, String(detail));
      } catch {
        // Not remembering it is harmless.
      }
      set({ detail, selection: null });
    },

    setDoc: (doc, status) =>
      set((s) => ({
        doc,
        docLabel: status ?? "Opened a design",
        // Replacing the whole document is opening a different design, so it
        // becomes its own baseline. An edit does not: that is the point.
        opened: doc,
        past: [...s.past.slice(-(UNDO_LIMIT - 1)), { doc: s.doc, label: s.docLabel }],
        future: [],
        path: [],
        selection: null,
        selectedNet: null,
        status: status ?? null,
      })),
    applyRemote: (doc, status) =>
      set((s) => {
        // The level survives unless the agent removed it from under us, and
        // so does the selection unless the block it names is gone.
        const path = ops.graphAtPath(doc, s.path) ? s.path : [];
        const selection =
          s.selection && path === s.path && ops.nodeAtPath(doc, ops.segmentsOf(s.selection))
            ? s.selection
            : null;
        // A net whose producer the agent deleted describes a tensor that is
        // not there any more, which is worse than describing nothing.
        const net = s.selectedNet;
        const producer = net ? net.slice(0, net.lastIndexOf(":")) : "";
        const selectedNet =
          net && producer && ops.nodeAtPath(doc, ops.segmentsOf(producer)) ? net : null;
        return {
          doc,
          docLabel: status ?? "An edit from elsewhere",
          past: [...s.past.slice(-(UNDO_LIMIT - 1)), { doc: s.doc, label: s.docLabel }],
          future: [],
          path,
          selection,
          selectedNet,
          also: [],
          status: status ?? null,
        };
      }),
    markOpened: () => set((s) => ({ opened: s.doc, status: "Comparing against this design" })),
    setPath: (path) => set({ path, selection: null, also: [], findingFocus: null, selectedNet: null }),
    enter: (path) =>
      set({ path: ops.segmentsOf(path), selection: null, also: [], findingFocus: null, selectedNet: null }),
    // Selecting something else puts the rules list back to showing
    // everything: a highlight that outlives what it pointed at is worse
    // than no highlight.
    select: (selection) => set({ selection, also: [], findingFocus: null, selectedNet: null }),
    selectNet: (selectedNet) => set({ selectedNet, selection: null, also: [], findingFocus: null }),
    selectPaths: (paths) => {
      const unique = [...new Set(paths)];
      // The last one is the primary, which is what clicking one more makes it.
      set({
        selection: unique[unique.length - 1] ?? null,
        also: unique.slice(0, -1),
        findingFocus: null,
        selectedNet: null,
      });
    },
    selected: () => {
      const { selection, also } = get();
      return selection === null ? [] : [...also, selection];
    },

    showCallouts: true,
    toggleCallouts: () => set((state) => ({ showCallouts: !state.showCallouts })),

    canvasStatus: INITIAL_STATUS,

    setCanvasStatus: (patch) =>
      set((state) => {
        const next = { ...state.canvasStatus, ...patch };
        // Pointer moves fire constantly; only re-render when something changed.
        const same =
          next.zoom === state.canvasStatus.zoom &&
          next.nodeCount === state.canvasStatus.nodeCount &&
          next.edgeCount === state.canvasStatus.edgeCount &&
          next.cursor?.x === state.canvasStatus.cursor?.x &&
          next.cursor?.y === state.canvasStatus.cursor?.y;
        return same ? {} : { canvasStatus: next };
      }),

    toggleLock: (path) =>
      set((state) => {
        const doc = structuredClone(state.doc);
        const ui = (doc.ui ??= {}) as { locked?: string[] };
        const locked = new Set(ui.locked ?? []);
        if (locked.has(path)) locked.delete(path);
        else locked.add(path);
        ui.locked = [...locked].sort();
        return { doc };
      }),

    isLocked: (path) => {
      const ui = get().doc.ui as { locked?: string[] } | undefined;
      const locked = ui?.locked ?? [];
      // Locking inherits down the hierarchy: locking a stack locks what it
      // stacks, the way a locked group locks its members.
      return locked.some((l) => path === l || path.startsWith(`${l}/`));
    },
    focusOn: (target) => {
      const segs = ops.segmentsOf(target);
      set((s) => ({
        path: segs.slice(0, -1),
        selection: target,
        focusNonce: s.focusNonce + 1,
        findingFocus: null,
        rightTab: "inspector",
      }));
    },
    showFindingsFor: (target) =>
      set((s) => ({
        selection: target,
        also: [],
        // Pressing a marker on the drawing asks what is wrong with this block,
        // so the dock opens whether or not it was.
        dockOpen: true,
        findingFocus: target,
        findingNonce: s.findingNonce + 1,
      })),
    clearFindingFocus: () => set({ findingFocus: null }),
    setRightTab: (rightTab) => set({ rightTab }),
    toggleFindings: () => set((s) => ({ dockOpen: !s.dockOpen })),
    setDockOpen: (dockOpen) => set({ dockOpen }),
    setShapeMode: (shapeMode) => set({ shapeMode }),
    setStatus: (status) => set({ status }),
    requestLayout: () => set((s) => ({ layoutNonce: s.layoutNonce + 1 })),

    undo: () =>
      set((s) => {
        const previous = s.past[s.past.length - 1];
        if (!previous) return s;
        return {
          doc: previous.doc,
          docLabel: previous.label,
          past: s.past.slice(0, -1),
          future: [{ doc: s.doc, label: s.docLabel }, ...s.future].slice(0, UNDO_LIMIT),
          status: `Undid ${s.docLabel.toLowerCase()}`,
        };
      }),
    redo: () =>
      set((s) => {
        const next = s.future[0];
        if (!next) return s;
        return {
          doc: next.doc,
          docLabel: next.label,
          past: [...s.past.slice(-(UNDO_LIMIT - 1)), { doc: s.doc, label: s.docLabel }],
          future: s.future.slice(1),
          status: next.label,
        };
      }),
    jumpTo: (index) =>
      set((s) => {
        const steps = [...s.past, { doc: s.doc, label: s.docLabel }, ...s.future];
        const target = steps[index];
        if (!target || index === s.past.length) return s;
        return {
          doc: target.doc,
          docLabel: target.label,
          past: steps.slice(0, index),
          future: steps.slice(index + 1),
          // Not "undid five edits": which five is the question, and the row
          // you pressed is the answer.
          status: `Back to: ${target.label}`,
          selection: null,
          also: [],
          selectedNet: null,
        };
      }),

    addNode: (parent, node, xy) => commit((d) => ops.addNode(d, parent, node, xy), `Added ${node.type}`),
    removeNode: (path) => {
      commit((d) => ops.removeNode(d, ops.segmentsOf(path)), `Deleted ${lastOf(path)}`);
      const { selection, also } = get();
      if (selection === path) set({ selection: null });
      if (also.includes(path)) set({ also: also.filter((p) => p !== path) });
    },
    setParam: (path, key, value) => commit((d) => ops.setParam(d, ops.segmentsOf(path), key, value), `Set ${lastOf(path)}.${key}`),
    connect: (parent, from, to) => commit((d) => ops.connect(d, parent, from, to), `Wired ${from} to ${to}`),
    disconnect: (parent, from, to) =>
      commit((d) => ops.disconnect(d, parent, from, to), `Unwired ${from} from ${to}`),
    reconnect: (parent, from, to, next) =>
      commit(
        (d) => ops.connect(ops.disconnect(d, parent, from, to), parent, next.from, next.to),
        `Moved a wire to ${next.to}`,
      ),
    setRuleSeverity: (rule, severity) => commit((d) => ops.setRuleSeverity(d, rule, severity), `Set the rule "${rule}" to ${severity}`),
    // Through the configuration in force, so an edit made while one is selected
    // lands in it rather than in the design underneath.
    setSymbol: (name, def) => commit((d) => ops.setSymbolInConfiguration(d, name, def), `Set ${name}`),
    setActiveConfiguration: (name) =>
      commit(
        (d) => ops.setActiveConfiguration(d, name),
        name === null ? "Building the design as written" : `Building at "${name}"`,
      ),
    captureConfiguration: (name, doc) =>
      commit((d) => ops.captureConfiguration(d, name, doc), `Saved the configuration "${name}"`),
    removeConfiguration: (name) =>
      commit((d) => ops.removeConfiguration(d, name), `Removed the configuration "${name}"`),
    renameSymbol: (from, to) => commit((d) => ops.renameSymbol(d, from, to), `Renamed ${from} to ${to}`),
    moveNode: (path, xy) => commit((d) => ops.moveNode(d, ops.segmentsOf(path), xy), `Moved ${lastOf(path)}`),
    moveNodes: (moves, label) =>
      commit(
        (d) => ops.moveNodes(d, moves),
        label ?? (moves.length === 1 ? `Moved ${lastOf(moves[0]!.path)}` : `Moved ${moves.length} blocks`),
      ),
    renameNode: (path, label) => commit((d) => ops.renameNode(d, ops.segmentsOf(path), label), `Labelled ${lastOf(path)}`),
    setNodeId: (path, id) => {
      const segs = ops.segmentsOf(path);
      commit((d) => ops.setNodeId(d, segs, id), `Renamed ${lastOf(path)} to ${id.trim()}`);
      const renamed = [...segs.slice(0, -1), id.trim()].join("/");
      if (get().selection === path) set({ selection: renamed });
    },
    setMetaName: (name) => commit((d) => ops.setMetaName(d, name), `Named the design "${name}"`),
    loadPreset: (name) => {
      const doc = getPreset(name);
      get().setDoc(doc, `Loaded preset ${name}`);
    },
    newDoc: () => get().setDoc(ops.emptyDoc(), "New document"),
  };
});
