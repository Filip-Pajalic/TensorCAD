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
import { applyEdit, labelAt, reasonFor, type Edit, type Failure, type Step } from "./edits.js";
export type { Edit, Failure, Step } from "./edits.js";
import type { Segments } from "./ops.js";
import { DEFAULT_OPERATING, loadOperating, saveOperating, type OperatingPoint } from "./operating.js";
import type { Doc, NodeDef, ParamValue, RuleSeverity, SymbolDef } from "@tensor-cad/engine";
import { getPreset, PRESET_NAMES } from "../engine.js";

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

export type RightTab = "inspector" | "symbols" | "cluster" | "ladder" | "runs" | "history" | "designs";
export type DialogId =
  | "settings"
  | "shortcuts"
  | "compare"
  | "palette"
  | "definitions"
  | "library"
  | "new"
  | null;

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
   * The history, as operations rather than as documents.
   *
   * `base` is where the design started — opened, loaded or empty — and `steps`
   * is everything done to it since. The document on screen is what you get by
   * folding the first `at` of them over the base, skipping the ones switched
   * off. That is what makes this a timeline rather than an undo stack: a stack
   * can say what it looked like before, and only a list of operations can say
   * what it would look like *without* the third one.
   *
   * `cache[i]` is the document after `i` steps, so the ordinary edit — append
   * at the end — is one operation and not a replay of a hundred. It is
   * truncated from the first index whose meaning changed, which is the only
   * thing that has to be right for the cache to be invisible.
   */
  base: Doc;
  /** How the base arrived, which is the first row of the list. */
  baseLabel: string;
  steps: Step[];
  /** How many steps are in force. The mark in the list; undo moves it back. */
  at: number;
  /** Materialised documents, `cache[i]` after `i` steps. Never read by a panel. */
  cache: Doc[];
  /** Steps that could not replay, and what they could not find. */
  failures: Failure[];
  /** What produced the document on screen. The current step's label. */
  docLabel: string;
  /**
   * Go to a point in the history directly.
   *
   * The index is how many steps are in force, so 0 is the design as opened.
   * Several undos in one gesture and the drawing lands where the row said it
   * would — the property a list is worth having for.
   */
  jumpTo: (index: number) => void;
  /**
   * Switch a step off, or back on, and replay everything after it.
   *
   * The thing a stack of documents cannot do. A suppressed step stays in the
   * list — it is a step you have taken out, not one you never made — and
   * whatever depended on it fails and says so rather than being silently
   * dropped.
   */
  setSuppressed: (index: number, suppressed: boolean) => void;
  /** Take a step out of the history for good. */
  removeStep: (index: number) => void;
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
  /**
   * Whether the minimap is drawn. Off until asked for: the drawing fits the
   * window when it opens, and a map of a sheet you can already see is one
   * more thing over it.
   */
  showMinimap: boolean;
  setShowMinimap: (show: boolean) => void;
  /**
   * Whether the title block is drawn on the sheet. Off until asked for: every
   * number on it is in the readout beside the sheet, and on a laptop it sat
   * over a third of the drawing.
   */
  showTitleBlock: boolean;
  setShowTitleBlock: (show: boolean) => void;
  /**
   * Whether the key is open on the sheet.
   *
   * Shut to a tab rather than away: the thing it explains is on every wire of
   * every drawing, so the tab stays on the sheet for the reader who does not
   * know what a dotted line means. Shut, not open, on a first visit, because
   * open it covered most of the drawing it was explaining — and the drawing is
   * the first thing anybody should see.
   */
  showKey: boolean;
  setShowKey: (show: boolean) => void;
  /**
   * Figure mode: draw what a paper draws.
   *
   * The reshapes on either side of attention are real and necessary and no
   * published figure draws them, because they move no data and cost no
   * parameters. Off by default: this is a CAD tool before it is a figure, and
   * a view that leaves parts out should be asked for.
   */
  figure: boolean;
  setFigure: (on: boolean) => void;
  /**
   * Which step of the walkthrough is open, or null when it is not running.
   *
   * The steps themselves are derived from the document on every render and are
   * not state: a walkthrough that had been computed once would go stale the
   * moment somebody changed `D`, and running on the design as it is now is the
   * one thing this can do that a recorded explanation cannot.
   */
  walkthrough: number | null;
  startWalkthrough: () => void;
  setWalkthroughStep: (at: number) => void;
  endWalkthrough: () => void;
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

/**
 * What the editor opens on.
 *
 * A toy, not an eight-billion-parameter Llama. Opening a CAD tool on the
 * hardest thing it can draw is a real cost for a reader who has never seen it
 * and no cost at all for somebody who loads a preset in the first five seconds
 * — and `nano-sort` is GPT-2 exactly, at a size where every weight fits on the
 * screen at a readable zoom. That is the whole argument for it.
 *
 * Which is also why the last preset loaded is remembered: somebody who works on
 * Llama every day should land on Llama, and only the first visit should land on
 * the toy. A name, not a document — an edited design is the storage provider's
 * job, and this is one line of localStorage.
 */
const DEFAULT_PRESET = "nano-sort";
const OPENED_KEY = "tensorcad.opened";

function loadOpened(): string {
  try {
    const name = localStorage.getItem(OPENED_KEY);
    return name && PRESET_NAMES.includes(name) ? name : DEFAULT_PRESET;
  } catch {
    return DEFAULT_PRESET;
  }
}

function rememberOpened(name: string): void {
  try {
    localStorage.setItem(OPENED_KEY, name);
  } catch {
    // Not remembering it is harmless.
  }
}

/**
 * A preset that will not load must not take the editor with it. The library is
 * embedded in the engine, so this can only happen when a remembered name has
 * been withdrawn between one release and the next.
 */
function openingDoc(): Doc {
  const wanted = loadOpened();
  try {
    return getPreset(wanted);
  } catch {
    return getPreset(DEFAULT_PRESET);
  }
}

const initialDoc = openingDoc();

export const useEditor = create<EditorState>((set, get) => {
  /**
   * Record an edit and perform it.
   *
   * It takes a value rather than a function now, because the whole of E7's
   * third part turns on an edit being something you can keep: a closure can be
   * called and a value can be replayed, suppressed, and read back out of the
   * list a week later.
   *
   * An edit that changes nothing is not recorded. Dragging a block one pixel
   * and back is not two rows in a timeline.
   */
  const commit = (edit: Edit, label: string): void => {
    const { doc, steps, at, cache } = get();
    const next = applyEdit(doc, edit);
    if (next === doc) return;

    // Everything ahead of the mark goes. Editing after an undo has always
    // discarded the redo; what is new is that the *steps* go rather than the
    // documents, which is the same thing said about a list instead of a stack.
    const kept = steps.slice(0, at).slice(-(UNDO_LIMIT - 1));
    const dropped = Math.max(0, Math.min(at, steps.length) - kept.length);
    set({
      doc: next,
      base: dropped > 0 ? cache[dropped]! : get().base,
      steps: [...kept, { edit, label }],
      at: kept.length + 1,
      cache: [...cache.slice(dropped, dropped + kept.length + 1), next],
      failures: get().failures.filter((f) => f.at < kept.length),
      docLabel: label,
      status: label,
    });
  };

  /**
   * Replay from the first step whose meaning changed.
   *
   * Nothing before `from` can have moved, so the cache up to there stands. A
   * suppressed step in the middle is the case this exists for: the fold has to
   * run again from there, and the steps after it are being applied to a
   * document they have not seen before, which is where a failure comes from.
   */
  const rebuild = (from: number): void => {
    const { base, steps, at, cache } = get();
    const start = Math.max(0, Math.min(from, at));
    const head = cache.slice(0, start + 1);
    let doc = head[start] ?? base;
    const failures = get().failures.filter((f) => f.at < start);

    for (let i = start; i < at; i++) {
      const step = steps[i]!;
      if (step.suppressed) {
        head.push(doc);
        continue;
      }
      const next = applyEdit(doc, step.edit);
      if (next === doc) failures.push({ at: i, reason: reasonFor(step.edit) });
      doc = next;
      head.push(doc);
    }
    set({ doc, cache: head, failures, docLabel: labelAt(steps, at, get().baseLabel) });
  };

  return {
    doc: initialDoc,
    opened: initialDoc,
    path: [],
    selection: null,
    also: [],
    selectedNet: null,
    base: initialDoc,
    baseLabel: `Opened ${initialDoc.meta.name}`,
    steps: [],
    at: 0,
    cache: [initialDoc],
    failures: [],
    docLabel: `Opened ${initialDoc.meta.name}`,
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
    showMinimap: loadFlag("minimap", false),
    setShowMinimap: (showMinimap) => set(saveFlag("minimap", showMinimap, { showMinimap })),
    showTitleBlock: loadFlag("titleblock", false),
    setShowTitleBlock: (showTitleBlock) =>
      set(saveFlag("titleblock", showTitleBlock, { showTitleBlock })),
    showKey: loadFlag("key", false),
    setShowKey: (showKey) => set(saveFlag("key", showKey, { showKey })),
    figure: loadFlag("figure", false),
    setFigure: (figure) => set(saveFlag("figure", figure, { figure })),

    walkthrough: null,
    // Opening it goes back to the top level and clears the selection. The
    // steps name blocks of the whole design, so starting one while drilled
    // into a container left every step with nothing to light and the entire
    // sheet dimmed — the drawing greyed out and no part of it lit.
    startWalkthrough: () =>
      set({ walkthrough: 0, path: [], selection: null, also: [], selectedNet: null, findingFocus: null }),
    setWalkthroughStep: (at) => set({ walkthrough: Math.max(0, at) }),
    endWalkthrough: () => set({ walkthrough: null }),

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

    // Opening a different design starts a different timeline. Keeping the old
    // steps would offer to replay edits made to another document, which is
    // either meaningless or, worse, occasionally works.
    setDoc: (doc, status) =>
      set({
        doc,
        base: doc,
        baseLabel: status ?? "Opened a design",
        docLabel: status ?? "Opened a design",
        steps: [],
        at: 0,
        cache: [doc],
        failures: [],
        // Replacing the whole document is opening a different design, so it
        // becomes its own baseline. An edit does not: that is the point.
        opened: doc,
        path: [],
        selection: null,
        selectedNet: null,
        status: status ?? null,
      }),
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
        // A step like any other, so an agent's edit is in the timeline: it
        // can be jumped past, suppressed, and read a week later. It carries
        // the document because that is what arrived — what the agent did is
        // expressible as operations, but what comes over the wire is a result.
        const label = status ?? "An edit from elsewhere";
        const kept = s.steps.slice(0, s.at).slice(-(UNDO_LIMIT - 1));
        const dropped = Math.max(0, Math.min(s.at, s.steps.length) - kept.length);
        return {
          doc,
          base: dropped > 0 ? s.cache[dropped]! : s.base,
          steps: [...kept, { edit: { kind: "replaceDoc", doc } as Edit, label }],
          at: kept.length + 1,
          cache: [...s.cache.slice(dropped, dropped + kept.length + 1), doc],
          failures: s.failures.filter((f) => f.at < kept.length),
          docLabel: label,
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

    // Moving the mark, not unwinding a stack. Every document between here and
    // the base is already in the cache, so an undo is a lookup.
    undo: () =>
      set((s) => {
        if (s.at === 0) return s;
        const at = s.at - 1;
        return {
          at,
          doc: s.cache[at]!,
          docLabel: labelAt(s.steps, at, s.baseLabel),
          status: `Undid ${s.steps[at]!.label.toLowerCase()}`,
        };
      }),
    redo: () =>
      set((s) => {
        if (s.at >= s.steps.length) return s;
        const at = s.at + 1;
        return {
          at,
          doc: s.cache[at]!,
          docLabel: labelAt(s.steps, at, s.baseLabel),
          status: s.steps[s.at]!.label,
        };
      }),
    jumpTo: (index) =>
      set((s) => {
        if (index < 0 || index > s.steps.length || index === s.at) return s;
        return {
          at: index,
          doc: s.cache[index]!,
          docLabel: labelAt(s.steps, index, s.baseLabel),
          // Not "undid five edits": which five is the question, and the row
          // you pressed is the answer.
          status: `Back to: ${labelAt(s.steps, index, s.baseLabel)}`,
          selection: null,
          also: [],
          selectedNet: null,
        };
      }),

    setSuppressed: (index, suppressed) => {
      const { steps } = get();
      const step = steps[index];
      if (!step || Boolean(step.suppressed) === suppressed) return;
      set({
        steps: steps.map((st, i) => (i === index ? { ...st, suppressed } : st)),
        selection: null,
        also: [],
        selectedNet: null,
        status: `${suppressed ? "Suppressed" : "Restored"}: ${step.label}`,
      });
      rebuild(index);
    },

    removeStep: (index) => {
      const { steps, at } = get();
      const step = steps[index];
      if (!step) return;
      set({
        steps: steps.filter((_, i) => i !== index),
        at: index < at ? at - 1 : at,
        selection: null,
        also: [],
        selectedNet: null,
        status: `Removed: ${step.label}`,
      });
      rebuild(index);
    },

    addNode: (parent, node, xy) => commit({ kind: "addNode", parent, node, xy }, `Added ${node.type}`),
    removeNode: (path) => {
      commit({ kind: "removeNode", path }, `Deleted ${lastOf(path)}`);
      const { selection, also } = get();
      if (selection === path) set({ selection: null });
      if (also.includes(path)) set({ also: also.filter((p) => p !== path) });
    },
    setParam: (path, key, value) => commit({ kind: "setParam", path, key, value }, `Set ${lastOf(path)}.${key}`),
    connect: (parent, from, to) => commit({ kind: "connect", parent, from, to }, `Wired ${from} to ${to}`),
    disconnect: (parent, from, to) =>
      commit({ kind: "disconnect", parent, from, to }, `Unwired ${from} from ${to}`),
    reconnect: (parent, from, to, next) =>
      commit({ kind: "reconnect", parent, from, to, next }, `Moved a wire to ${next.to}`),
    setRuleSeverity: (rule, severity) => commit({ kind: "setRuleSeverity", rule, severity }, `Set the rule "${rule}" to ${severity}`),
    // Through the configuration in force, so an edit made while one is selected
    // lands in it rather than in the design underneath.
    setSymbol: (name, def) => commit({ kind: "setSymbol", name, def }, `Set ${name}`),
    setActiveConfiguration: (name) =>
      commit(
        { kind: "setActiveConfiguration", name },
        name === null ? "Building the design as written" : `Building at "${name}"`,
      ),
    captureConfiguration: (name, doc) =>
      commit({ kind: "captureConfiguration", name, doc }, `Saved the configuration "${name}"`),
    removeConfiguration: (name) =>
      commit({ kind: "removeConfiguration", name }, `Removed the configuration "${name}"`),
    renameSymbol: (from, to) => commit({ kind: "renameSymbol", from, to }, `Renamed ${from} to ${to}`),
    moveNode: (path, xy) => commit({ kind: "moveNode", path, xy }, `Moved ${lastOf(path)}`),
    moveNodes: (moves, label) =>
      commit(
        { kind: "moveNodes", moves },
        label ?? (moves.length === 1 ? `Moved ${lastOf(moves[0]!.path)}` : `Moved ${moves.length} blocks`),
      ),
    renameNode: (path, label) => commit({ kind: "renameNode", path, label }, `Labelled ${lastOf(path)}`),
    setNodeId: (path, id) => {
      const segs = ops.segmentsOf(path);
      commit({ kind: "setNodeId", path, id }, `Renamed ${lastOf(path)} to ${id.trim()}`);
      const renamed = [...segs.slice(0, -1), id.trim()].join("/");
      if (get().selection === path) set({ selection: renamed });
    },
    setMetaName: (name) => commit({ kind: "setMetaName", name }, `Named the design "${name}"`),
    loadPreset: (name) => {
      const doc = getPreset(name);
      rememberOpened(name);
      get().setDoc(doc, `Loaded preset ${name}`);
    },
    newDoc: () => get().setDoc(ops.emptyDoc(), "New document"),
  };
});
