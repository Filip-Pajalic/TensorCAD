/**
 * The canvas: exactly one graph level, never a nested sub-flow.
 *
 * Nodes and edges are a projection of the document plus the analysis. Dragging
 * is the only interaction that keeps local state, and it is written back to the
 * document when the drag ends so undo sees one step per drag.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  applyNodeChanges,
  useNodesInitialized,
  useReactFlow,
  type Connection,
  type Edge as FlowEdge,
  type IsValidConnection,
  type NodeChange,
  type EdgeTypes,
  type NodeTypes,
} from "@xyflow/react";
import BlockNodeView, { type BlockFlowNode, type BlockNodeData, type PortView } from "./BlockNode.js";
import FrameNodeView, { type FrameFlowNode, type FrameNodeData } from "./FrameNode.js";
import { categoryColor, glyphFor, kindOf, labelOf, paramSummary, typeName } from "./blocks.js";
import { chooseSides, handleId, portOfHandle, sideOfHandle, wireKind, type Box } from "./wiring.js";
import WireEdge from "./WireEdge.js";
import { formatShape, type ShapeMode } from "./shapes.js";
import { createConnectionChecker } from "./validate.js";
import { layoutGraph, layoutTree, type LayoutBox } from "./layout.js";
import CalloutLayer from "./CalloutLayer.js";
import PartDoc from "./PartDoc.js";
import Key from "./Key.js";
import Walkthrough from "../panels/Walkthrough.js";
import { buildWalkthrough } from "../state/walkthrough.js";
import TitleBlock from "../panels/TitleBlock.js";
import { onThemeChange, resolvedTheme, themeValue } from "../state/theme.js";
import { calloutFor, type Callout } from "./callouts.js";
import { useEditor } from "../state/store.js";
import { setViewportApi } from "../state/commands.js";
import { type Derived } from "../state/derive.js";
import { useLevel } from "../state/hooks.js";
import { type Level } from "../state/level.js";
import { unfold, type Unfolded } from "../state/unfold.js";
import { newNodeFor } from "../state/addBlock.js";
import * as ops from "../state/ops.js";
import type { Doc, NodeDef } from "@tensor-cad/engine";
import { joinPath, splitEndpoint } from "@tensor-cad/engine";
import { blockDef, catalogOf, type BlockDef } from "../engine.js";

const nodeTypes: NodeTypes = {
  block: BlockNodeView as unknown as NodeTypes[string],
  frame: FrameNodeView as unknown as NodeTypes[string],
};

const edgeTypes: EdgeTypes = {
  wire: WireEdge as unknown as EdgeTypes[string],
};

/** Both kinds of thing the sheet can carry. */
export type CanvasNode = BlockFlowNode | FrameFlowNode;

/**
 * Grid pitch, in canvas units. Positions snap to the minor grid.
 *
 * KiCad ships four schematic grids and defaults to the second; 16 over 80 is
 * the same idea at screen scale: fine enough to place precisely, coarse enough
 * that two hand-placed blocks line up.
 */
export const GRID_MINOR = 16;
export const GRID_MAJOR = 80;

export const DRAG_MIME = "application/tensorcad-block";

const NODE_WIDTH = 216;

/** Shared, so a block the catalog does not know still compares equal. */
const EMPTY_DOCS: { summary?: string; formula?: string } = {};

/** A circled operator is a junction on the line, not a part with a caption. */
const GLYPH_SIZE = 38;

/** Two text rows and a rule, which is what every part on the sheet carries. */
const ESTIMATED_PART_HEIGHT = 74;

const READ_ONLY_REASON = "This level is generated; edit the block's parameters instead";

function estimateHeight(data: BlockNodeData | FrameNodeData): number {
  // A frame is sized by the layout engine from what it holds; only a part has
  // a height that can be guessed from its own contents. Both data types index
  // to `unknown`, so the narrowing is explicit.
  const part = data as Partial<BlockNodeData>;
  if (!part.inPorts || !part.outPorts) return 120;
  const rows = Math.max(part.inPorts.length, part.outPorts.length, 1);
  return 30 + 16 + (part.summary ? 16 : 0) + rows * 18 + 14;
}

type Positions = Record<string, [number, number]>;

/** Stable identity matters: this feeds a memo that rebuilds every node. */
const NO_POSITIONS: Positions = {};

/**
 * The element type on a port.
 *
 * Only a few blocks declare one: the model input carries token ids, and
 * everything downstream carries activations at the training dtype. Rather than
 * invent a type system, report what the block actually says and leave the rest
 * unlabelled.
 */
function dtypeOf(
  node: { type: string },
  resolved: { p: Record<string, unknown> } | undefined,
  port: string,
  side: "in" | "out",
): string | null {
  if (node.type === "input") return String(resolved?.p.dtype ?? "int64");
  if (node.type === "embedding" && side === "in" && port === "ids") return "int64";
  return null;
}

/**
 * A drawn thing, before it becomes a React Flow node.
 *
 * Both views — the flat editable graph and the unfolded drawing — reduce to
 * this, so the wiring is computed once for both rather than twice, slightly
 * differently.
 */
interface Item {
  path: string;
  node: NodeDef;
  def: BlockDef | undefined;
  parent: string | null;
  depth: number;
  frame: boolean;
  multiplier: number | null;
  mergedType?: string;
  mergedDef?: BlockDef;
}

interface EdgeSpec {
  id: string;
  source: string;
  sourcePort: string;
  target: string;
  targetPort: string;
  /** Local endpoints, for the document ops that delete an edge. */
  from?: string;
  to?: string;
}

interface Wiring {
  edges: FlowEdge[];
  /**
   * How many wires land on each handle, per node path.
   *
   * A count rather than a set, because the count is what tells a junction from
   * an ordinary connection: a net that branches is drawn with a dot and one
   * that does not is drawn with nothing at all.
   */
  live: Map<string, Map<string, number>>;
}

const NO_PINS: ReadonlyMap<string, number> = new Map();

/** Absolute box of a node, summing the frames it sits inside. */
function absoluteBox(
  path: string,
  boxes: Record<string, Box>,
  parentOf: Map<string, string | null>,
): Box | undefined {
  const own = boxes[path];
  if (!own) return undefined;
  let x = own.x;
  let y = own.y;
  let up = parentOf.get(path) ?? null;
  const guard = new Set<string>([path]);
  while (up && !guard.has(up)) {
    guard.add(up);
    const box = boxes[up];
    if (!box) break;
    x += box.x;
    y += box.y;
    up = parentOf.get(up) ?? null;
  }
  return { x, y, width: own.width, height: own.height };
}

/**
 * Turn edge specs into drawn wires.
 *
 * The side each end uses comes from where the two boxes actually are, so a wire
 * that has to climb back up leaves sideways instead of doubling back through
 * the symbol it just left. The line style says what the wire carries.
 */
function wireUp(
  specs: EdgeSpec[],
  items: Map<string, Item>,
  boxes: Record<string, Box>,
  parentOf: Map<string, string | null>,
  derived: Derived,
  shapeMode: ShapeMode,
  deletable: boolean,
): Wiring {
  const live = new Map<string, Map<string, number>>();
  const mark = (path: string, id: string): void => {
    let held = live.get(path);
    if (!held) {
      held = new Map();
      live.set(path, held);
    }
    held.set(id, (held.get(id) ?? 0) + 1);
  };

  const edges = specs.map((spec) => {
    const fromItem = items.get(spec.source);
    const toItem = items.get(spec.target);
    // The ports themselves say which side they leave by and what they carry.
    // This used to be two `type:port` lookups in the renderer.
    const fromPort = derived.infer.ports.get(spec.source)?.out[spec.sourcePort];
    const toPort = derived.infer.ports.get(spec.target)?.in[spec.targetPort];
    const from = { side: fromPort?.anchor === "side", dtype: fromPort?.dtype ?? "inherit" };
    const to = { side: toPort?.anchor === "side", dtype: toPort?.dtype ?? "inherit" };

    const { sourceSide, targetSide } = chooseSides(
      absoluteBox(spec.source, boxes, parentOf),
      absoluteBox(spec.target, boxes, parentOf),
      from,
      to,
    );
    const sourceHandle = handleId("o", spec.sourcePort, sourceSide);
    const targetHandle = handleId("i", spec.targetPort, targetSide);
    mark(spec.source, sourceHandle);
    mark(spec.target, targetHandle);

    const shape = derived.infer.outputs.get(`${spec.source}:${spec.sourcePort}`);
    // A declared dtype wins; otherwise fall back to what the block's own
    // parameters imply, which is all an inherited port can offer.
    const declared =
      from.dtype !== "inherit"
        ? from.dtype
        : (dtypeOf(
            fromItem?.node ?? { type: "" },
            derived.infer.resolved.get(spec.source),
            spec.sourcePort,
            "out",
          ) ?? "inherit");
    const kind = wireKind({ ...from, dtype: declared.startsWith("int") ? "int" : declared }, to);

    return {
      id: spec.id,
      source: spec.source,
      target: spec.target,
      sourceHandle,
      targetHandle,
      // "step" is smoothstep with no corner rounding: true right angles, which
      // is how a schematic draws a net.
      type: "wire",
      // A bypass carries the same tensor as the line it rejoins; labelling both
      // just doubles the ink.
      label:
        kind === "bypass"
          ? undefined
          : (formatShape(shape, shapeMode, derived.symbols) ?? undefined),
      labelShowBg: true,
      className: `flow-edge flow-edge--${kind}`,
      deletable,
      // The net a wire belongs to is its source pin: every wire leaving one
      // output carries the same tensor, which is what makes highlighting them
      // together truthful rather than decorative.
      data: { from: spec.from, to: spec.to, net: `${spec.source}:${spec.sourcePort}` },
    } satisfies FlowEdge;
  });

  return { edges, live };
}

/** Ports, shapes and connection state for one drawn part. */
function portViews(
  item: Item,
  derived: Derived,
  shapeMode: ShapeMode,
  connectedIn: ReadonlySet<string>,
  connectedOut: ReadonlySet<string>,
): { inPorts: PortView[]; outPorts: PortView[] } {
  const { path, node } = item;
  const ports = derived.infer.ports.get(path);
  const resolved = derived.infer.resolved.get(path);

  const view = (name: string, side: "in" | "out"): PortView => {
    const map = side === "in" ? derived.infer.inputs : derived.infer.outputs;
    const shape = map.get(`${path}:${name}`);
    const set = side === "in" ? connectedIn : connectedOut;
    return {
      name,
      shape: formatShape(shape, shapeMode, derived.symbols),
      connected: set.has(`${path}:${name}`),
      dtype: dtypeOf(node, resolved, name, side),
    };
  };

  return {
    inPorts: Object.keys(ports?.in ?? {}).map((n) => view(n, "in")),
    outPorts: Object.keys(ports?.out ?? {}).map((n) => view(n, "out")),
  };
}

interface BuildOptions {
  items: Item[];
  specs: EdgeSpec[];
  derived: Derived;
  selection: string | null;
  /** The rest of a multiple selection, primary excluded. */
  alsoSelected: ReadonlySet<string>;
  boxes: Record<string, Box>;
  shapeMode: ShapeMode;
  lockedPaths: ReadonlySet<string>;
  /** True for the flat view, where the document owns positions and edits. */
  editable: boolean;
  /** Drawn path -> the left-out blocks whose findings it carries. */
  absorbedBy: ReadonlyMap<string, string[]>;
  /**
   * What the open walkthrough step is about, or null when none is running.
   * Everything outside it is dimmed, which is what makes the narration point.
   */
  lit: ReadonlySet<string> | null;
}

/** True when a path is one the step named, or sits inside one. */
function isLit(lit: ReadonlySet<string>, path: string): boolean {
  if (lit.has(path)) return true;
  for (const one of lit) {
    if (path.startsWith(`${one}/`)) return true;
  }
  return false;
}

/** The last segment of a path: the node's own id within its level. */
function idOf(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1);
}

/**
 * Both views, built the same way: wire first so every part knows which pins
 * carry a line, then draw the parts.
 */
function buildView(opts: BuildOptions): { nodes: CanvasNode[]; edges: FlowEdge[] } {
  const {
    items,
    specs,
    derived,
    selection,
    alsoSelected,
    boxes,
    shapeMode,
    lockedPaths,
    editable,
    absorbedBy,
    lit,
  } = opts;
  const byPath = new Map(items.map((i) => [i.path, i]));
  const parentOf = new Map(items.map((i) => [i.path, i.parent]));

  const { edges, live } = wireUp(specs, byPath, boxes, parentOf, derived, shapeMode, editable);

  const connectedIn = new Set(specs.map((s) => `${s.target}:${s.targetPort}`));
  const connectedOut = new Set(specs.map((s) => `${s.source}:${s.sourcePort}`));

  // Frames must precede the children that name them.
  const ordered = [...items].sort((a, b) => a.depth - b.depth);

  const nodes = ordered.map((item): CanvasNode => {
    const { path, node, def } = item;
    const box = boxes[path];
    const shared = {
      id: path,
      position: { x: box?.x ?? 0, y: box?.y ?? 0 },
      parentId: item.parent ?? undefined,
      // A step names a block, and everything under it belongs to that step too:
      // lighting `attn` without lighting what it expands into would dim the
      // very thing the sentence is describing.
      className: lit && !isLit(lit, path) ? "is-dimmed" : undefined,
      selected: selection === path || alsoSelected.has(path),
      draggable: editable && !lockedPaths.has(path),
      connectable: editable,
      deletable: editable && !lockedPaths.has(path),
    };

    if (item.frame) {
      // A merged frame shows the inner block's identity throughout: its type,
      // its name and its colour. `def` here is still the outer one.
      const frameType = item.mergedType ?? node.type;
      const frameDef = item.mergedDef ?? def;
      const data: FrameNodeData = {
        path,
        label: labelOf(node),
        type: frameType,
        typeName: typeName(frameDef, frameType),
        docs: frameDef?.docs ?? EMPTY_DOCS,
        category: frameDef?.category ?? "unknown",
        params: derived.paramsByPath.get(path) ?? 0,
        multiplier: item.multiplier,
        severity: derived.severityByPath.get(path) ?? null,
        locked: lockedPaths.has(path),
        depth: item.depth,
      };
      return {
        ...shared,
        type: "frame",
        // A frame is sized by what it holds, which only ELK knows.
        width: box?.width,
        height: box?.height,
        style: {
          width: box?.width,
          height: box?.height,
          // A frame is picked by its outline, not its interior: the rule every
          // vector editor uses for a shape with no fill. Its interior belongs
          // to the parts and the wires inside it, and while it was a solid
          // target every wire crossing a frame — nearly all of them, in an
          // unfolded drawing — was covered by it. The outline and the caption
          // re-enable themselves in CSS.
          //
          // Inline because React Flow sets `pointer-events: all` on every node
          // in its own stylesheet, and this has to win that outright.
          pointerEvents: "none",
        },
        data,
      } as FrameFlowNode;
    }

    const resolved = derived.infer.resolved.get(path);
    const { inPorts, outPorts } = portViews(item, derived, shapeMode, connectedIn, connectedOut);
    const data: BlockNodeData = {
      path,
      label: labelOf(node),
      type: node.type,
      typeName: typeName(def, node.type),
      docs: def?.docs ?? EMPTY_DOCS,
      category: def?.category ?? "unknown",
      kind: kindOf(def),
      summary: paramSummary(def, resolved, shapeMode, derived.symbols),
      params: derived.paramsByPath.get(path) ?? 0,
      inPorts,
      outPorts,
      severity: derived.severityByPath.get(path) ?? null,
      // Attributed to this block exactly, not rolled up from its interior: a
      // marker that fired because of something three levels down would be
      // pointing at the wrong part.
      // A block the figure left out still has its findings, and they appear on
      // the part that took its place. Naming the block in the message is what
      // keeps the marker honest — it is pointing at a consequence, not a cause.
      findings: [
        ...(derived.findingsByPath.get(path) ?? []).map((f) => ({
          severity: f.severity,
          message: f.message,
          rule: f.rule,
          port: f.port,
        })),
        ...(absorbedBy.get(path) ?? []).flatMap((hidden) =>
          (derived.findingsByPath.get(hidden) ?? []).map((f) => ({
            severity: f.severity,
            message: `${idOf(hidden)}: ${f.message}`,
            rule: f.rule,
            port: undefined,
          })),
        ),
      ],
      drillable: editable && ops.isDrillable(node, def),
      readOnly: !editable,
      locked: lockedPaths.has(path),
      repeat: editable && typeof resolved?.p.count === "number" ? resolved.p.count : null,
      glyph: glyphFor(node.type),
      livePins: live.get(path) ?? NO_PINS,
    };
    return { ...shared, type: "block", width: NODE_WIDTH, data } as BlockFlowNode;
  });

  return { nodes, edges };
}

/**
 * The flat, editable view: one graph level, hand-placed.
 *
 * Resolved through the document's own catalog rather than the built-in one
 * (invariant 1): a design that defines its own blocks would otherwise draw them
 * as an unknown category with no parameter summary and no way in, and only on
 * this view — the unfolded one has always gone through `catalogOf`.
 */
function flatItems(
  doc: Doc,
  level: Level,
  derived: Derived,
): { items: Item[]; specs: EdgeSpec[] } {
  const cat = catalogOf(doc);
  const items: Item[] = level.graph.nodes.map((node) => ({
    path: joinPath(level.prefix, node.id),
    node,
    def: cat[node.type],
    parent: null,
    depth: 0,
    frame: false,
    multiplier: null,
  }));

  const ids = new Set(level.graph.nodes.map((n) => n.id));
  const specs: EdgeSpec[] = [];
  for (const [from, to] of level.graph.edges) {
    let f, t;
    try {
      f = splitEndpoint(from);
      t = splitEndpoint(to);
    } catch {
      continue;
    }
    if (!ids.has(f.node) || !ids.has(t.node)) continue;
    specs.push({
      id: `${from}->${to}`,
      source: joinPath(level.prefix, f.node),
      sourcePort: f.port,
      target: joinPath(level.prefix, t.node),
      targetPort: t.port,
      from,
      to,
    });
  }
  void derived;
  return { items, specs };
}

/** The unfolded drawing: containers opened in place. */
function unfoldedItems(view: Unfolded): { items: Item[]; specs: EdgeSpec[] } {
  const items: Item[] = view.nodes.map((n) => ({
    path: n.path,
    node: n.node,
    def: n.def,
    parent: n.parent,
    depth: n.depth,
    frame: n.frame,
    multiplier: n.multiplier,
    mergedType: n.mergedType,
    mergedDef: n.mergedDef,
  }));
  const specs: EdgeSpec[] = view.edges.map((e) => ({
    id: e.id,
    source: e.source,
    sourcePort: e.sourcePort,
    target: e.target,
    targetPort: e.targetPort,
  }));
  return { items, specs };
}


export default function Canvas(): React.ReactElement {
  const doc = useEditor((s) => s.doc);
  const path = useEditor((s) => s.path);
  const selection = useEditor((s) => s.selection);
  const also = useEditor((s) => s.also);
  // A Set once per change rather than per node: `buildView` asks for every
  // node it draws, and a design can hold a few hundred.
  const alsoSelected = useMemo(() => new Set(also), [also]);
  const layoutNonce = useEditor((s) => s.layoutNonce);
  const focusNonce = useEditor((s) => s.focusNonce);
  const shapeMode = useEditor((s) => s.shapeMode);
  const showCallouts = useEditor((s) => s.showCallouts);
  const detail = useEditor((s) => s.detail);
  const tool = useEditor((s) => s.tool);
  const snap = useEditor((s) => s.snap);
  const showGrid = useEditor((s) => s.showGrid);
  const showMinimap = useEditor((s) => s.showMinimap);
  const showTitleBlock = useEditor((s) => s.showTitleBlock);

  // `useLevel`, not `useDerived`: inside a definition the numbers and shapes on
  // screen are the template's, analysed as a design of its own. The readout
  // goes on showing the design's, which is what it is for.
  const { level, derived } = useLevel();
  const { screenToFlowPosition, fitView, setCenter, getZoom, zoomIn, zoomOut, zoomTo } =
    useReactFlow();

  /**
   * Which blocks on this level are locked, counting inheritance: a block inside
   * a locked container is locked too.
   */
  // React Flow's background and minimap want literal colours, so they are read
  // from the theme tokens and re-read whenever the theme changes.
  const [theme, setTheme] = useState(resolvedTheme);
  useEffect(() => onThemeChange((next) => setTheme(next)), []);
  const palette = useMemo(
    () => ({
      gridMinor: themeValue("--sheet-grid-minor", "#e2e0d8"),
      gridMajor: themeValue("--sheet-grid-major", "#cfccc0"),
      sheet: themeValue("--sheet", "#f7f6f2"),
      border: themeValue("--border", "#b3b8bf"),
      accent: themeValue("--accent", "#1a66b3"),
    }),
    [theme],
  );

  const lockedPaths = useMemo(() => {
    const roots = (doc.ui as { locked?: string[] } | undefined)?.locked ?? [];
    const out = new Set<string>();
    for (const node of level.graph.nodes) {
      const path = joinPath(level.prefix, node.id);
      if (roots.some((l) => path === l || path.startsWith(`${l}/`) || level.prefix.startsWith(l))) {
        out.add(path);
      }
    }
    return out;
  }, [doc.ui, level]);

  /** Positions for read-only levels live here rather than in the document. */
  const [ephemeral, setEphemeral] = useState<Positions>({});
  /** True while a wire is being dragged, so every pin shows itself. */
  const [connecting, setConnecting] = useState(false);
  const positions: Positions = level.editable ? (doc.ui?.positions ?? NO_POSITIONS) : ephemeral;

  /**
   * The unfolded view, when the detail control asks for one. Zero detail keeps
   * the flat, editable, hand-placed graph; anything above it draws containers
   * open, which is how every published figure shows an architecture.
   */
  const figure = useEditor((s) => s.figure);
  const walkAt = useEditor((s) => s.walkthrough);
  const view = useMemo(
    () => (detail > 0 ? unfold(doc, level, derived, detail, figure) : null),
    [detail, doc, level, derived, figure],
  );

  /** Frame and leaf boxes for the unfolded view, from the layout engine. */
  const [boxes, setBoxes] = useState<Record<string, LayoutBox>>({});

  /**
   * What the pointer is over, and what the catalog says about it.
   *
   * One card owned by the sheet rather than one tooltip per part: a dense
   * drawing is a few hundred parts, and a floating-element instance on each of
   * them is machinery in the way of the one gesture — dragging — that the
   * canvas exists for. React Flow already reports which node the pointer
   * entered, so the card only has to be told where to sit.
   */
  const [hover, setHover] = useState<
    { x: number; y: number; name: string; type: string; docs: { summary?: string; formula?: string }; drillable: boolean } | null
  >(null);

  const built = useMemo(() => {
    const { items, specs } = view ? unfoldedItems(view) : flatItems(doc, level, derived);
    // The flat view stores only a position; a box still needs a size for the
    // geometry that decides which side a wire leaves by.
    const flatBoxes: Record<string, Box> = {};
    if (!view) {
      for (const item of items) {
        const xy = positions[item.path];
        flatBoxes[item.path] = {
          x: xy?.[0] ?? 0,
          y: xy?.[1] ?? 0,
          width: NODE_WIDTH,
          height: ESTIMATED_PART_HEIGHT,
        };
      }
    }
    // What the open walkthrough step is about. Built here rather than in the
    // store because the steps are derived from the document and are not state.
    const lit =
      walkAt === null
        ? null
        : new Set(buildWalkthrough(doc, derived)[walkAt]?.paths ?? []);

    // Inverted: unfold says which drawn part each left-out block landed on,
    // and the canvas needs the other direction to gather findings onto a part.
    const absorbedBy = new Map<string, string[]>();
    for (const [hidden, landed] of view?.absorbed ?? []) {
      const held = absorbedBy.get(landed);
      if (held) held.push(hidden);
      else absorbedBy.set(landed, [hidden]);
    }

    return buildView({
      items,
      specs,
      derived,
      selection,
      alsoSelected,
      boxes: view ? boxes : flatBoxes,
      shapeMode,
      lockedPaths,
      editable: !view && level.editable,
      absorbedBy,
      lit,
    });
  }, [view, boxes, level, derived, selection, alsoSelected, positions, shapeMode, lockedPaths, doc, walkAt]);

  const builtNodes = built.nodes;

/**
   * The net under the cursor.
   *
   * eeschema highlights a whole net rather than the one segment you happen to
   * be over, because a net is the thing that means something — a wire that
   * branches is still one signal. Here a net is everything leaving the same
   * output pin, which is exactly the set the analysis treats as one tensor.
   */
  const [litNet, setLitNet] = useState<string | null>(null);
  const selectedNet = useEditor((s) => s.selectedNet);

  /**
   * The drawn wires, with the net under the cursor marked.
   *
   * Applied here rather than in the build so that moving the cursor along a
   * wire does not re-derive every node on the sheet.
   */
  const edges = useMemo<FlowEdge[]>(() => {
    // The selected net stays lit; the cursor lights one on top of that. Hover
    // wins, so running along a second wire shows you *that* one rather than
    // leaving the old highlight to answer for it.
    const marked = litNet ?? selectedNet;
    if (!marked) return built.edges;
    return built.edges.map((e) => {
      const lit = (e.data as { net?: string } | undefined)?.net === marked;
      if (!lit) return e;
      return {
        ...e,
        data: { ...e.data, lit: true },
        className: `${e.className ?? ""} flow-edge--lit`,
        // A lit net is drawn over the blocks it runs between, which is the
        // point of lighting it.
        zIndex: 1,
      };
    });
  }, [built.edges, litNet, selectedNet]);

  const callouts = useMemo<Callout[]>(() => {
    if (!showCallouts) return [];
    const out: Callout[] = [];
    // An unfolded drawing annotates what it actually shows, which is deeper
    // than the top level: the attention block's head count belongs on the
    // attention block, wherever in the nesting it ended up.
    const targets = view
      ? view.nodes.filter((n) => !n.frame).map((n) => ({ node: n.node, path: n.path }))
      : level.graph.nodes.map((node) => ({ node, path: joinPath(level.prefix, node.id) }));
    for (const { node, path } of targets) {
      const c = calloutFor(node, derived.infer.resolved.get(path), derived.symbols, path);
      if (c) out.push(c);
    }
    return out;
  }, [view, level, derived, showCallouts]);

  useEffect(() => {
    useEditor.getState().setCanvasStatus({
      nodeCount: builtNodes.length,
      edgeCount: edges.length,
      gridMinor: GRID_MINOR,
      gridMajor: GRID_MAJOR,
    });
  }, [builtNodes.length, edges.length]);

  const [nodes, setNodes] = useState<CanvasNode[]>(builtNodes);

  /**
   * Rebuild from the document, but carry each node's measured box across.
   * React Flow derives its handle positions from that measurement, and an
   * unmeasured node has no handles to hang an edge on -- dropping it would make
   * every edge vanish for a frame (or longer) after each edit.
   */
  useEffect(() => {
    setNodes((previous) => {
      if (previous.length === 0) return builtNodes;
      const before = new Map(previous.map((n) => [n.id, n]));
      return builtNodes.map((n) => {
        const old = before.get(n.id);
        return old?.measured ? { ...n, measured: old.measured } : n;
      });
    });
  }, [builtNodes]);

  const nodesRef = useRef<CanvasNode[]>(builtNodes);
  nodesRef.current = nodes;

  const onNodesChange = useCallback((changes: NodeChange<CanvasNode>[]) => {
    // Removal is driven by the document, not by React Flow: a node goes when
    // the graph says it has, not when a key was pressed over it.
    //
    // Selection used to be dropped here for the same reason, which cost the
    // box-drag: React Flow reports a rubber-band selection as one `select`
    // change per node and nothing was listening. They are applied to the local
    // nodes and then reported up, so the store holds the set and the store is
    // still the one that says which nodes are drawn selected.
    const selects = changes.filter((c) => c.type === "select");
    const local = changes.filter((c) => c.type !== "select" && c.type !== "remove");
    if (local.length > 0) setNodes((current) => applyNodeChanges(local, current));
    if (selects.length === 0) return;

    const after = new Set(nodesRef.current.filter((n) => n.selected).map((n) => n.id));
    for (const change of selects) {
      if (change.type !== "select") continue;
      if (change.selected) after.add(change.id);
      else after.delete(change.id);
    }
    const state = useEditor.getState();
    const ordered = [...after];
    // Adding one block makes that one primary: a modified click is a click,
    // and the inspector should follow it. Adding several is a box drag, where
    // there is no one block the gesture named, so whatever was primary stays
    // primary rather than the inspector jumping to an arbitrary corner.
    const added = selects.filter((c) => c.type === "select" && c.selected);
    const primary =
      added.length === 1 ? added[0]!.id : state.selection !== null && after.has(state.selection) ? state.selection : null;
    if (primary !== null && after.has(primary)) {
      ordered.splice(ordered.indexOf(primary), 1);
      ordered.push(primary);
    }
    if (ordered.length === state.selected().length && ordered.every((id, i) => id === state.selected()[i])) {
      return;
    }
    state.selectPaths(ordered);
  }, []);

  // --- auto layout ---------------------------------------------------------

  /**
   * Ask for a fit. React Flow only knows a node's box after it has measured it
   * in the DOM, so the request is parked and honoured once `nodesInitialized`
   * says the measurements are in.
   */
  const pendingFit = useRef(true);
  const fitSoon = useCallback(() => {
    pendingFit.current = true;
  }, []);

  const nodesInitialized = useNodesInitialized();
  useEffect(() => {
    if (!nodesInitialized || !pendingFit.current) return;
    pendingFit.current = false;
    // Never open at a zoom where the symbols cannot be read. Showing part of a
    // large design is better than showing all of it illegibly.
    // Asymmetric padding: the title block sits in the bottom-right corner of
    // the sheet, the way it does on a real drawing, so the drawing is fitted
    // clear of it rather than under it.
    void fitView({
      padding: { top: 0.1, right: 0.3, bottom: 0.22, left: 0.14 },
      minZoom: 0.6,
      maxZoom: 1.1,
      duration: 220,
    });
  }, [nodesInitialized, nodes, fitView]);

  const runLayout = useCallback(async () => {
    const current = nodesRef.current;
    if (current.length === 0) return;
    const sizes = current.map((n) => ({
      id: n.id,
      width: n.measured?.width ?? NODE_WIDTH,
      height: n.measured?.height ?? estimateHeight(n.data),
    }));
    const placed = await layoutGraph(
      sizes,
      edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
    );
    if (Object.keys(placed).length === 0) return;

    // A lock has to stop automatic edits as well as dragging, otherwise it is
    // only a hint. Auto-layout leaves locked blocks where they are and says so.
    const moved = Object.entries(placed).filter(([p]) => !lockedPaths.has(p));
    const skipped = Object.keys(placed).length - moved.length;

    if (level.editable) {
      useEditor.getState().moveNodes(moved.map(([p, xy]) => ({ path: p, xy })), "Laid out the sheet");
    } else {
      setEphemeral((prev) => ({ ...prev, ...Object.fromEntries(moved) }));
    }
    if (skipped > 0) {
      useEditor
        .getState()
        .setStatus(`Auto-layout skipped ${skipped} locked block${skipped === 1 ? "" : "s"}.`);
    }
    fitSoon();
  }, [edges, level.editable, fitSoon, lockedPaths]);

  /**
   * Lay out the unfolded view.
   *
   * Leaf sizes are estimated rather than measured, because measuring them would
   * need them rendered, and rendering them needs the layout that measuring is
   * meant to produce. The estimate is exact for a part of fixed width and close
   * enough for the rest; the frames around them are sized by ELK from the
   * result, which is the part that actually has to be right.
   */
  useEffect(() => {
    if (!view) {
      setBoxes({});
      return;
    }
    let live = true;
    const sizes = view.nodes.map((n) => {
      const glyph = glyphFor(n.node.type) !== null;
      return {
        id: n.path,
        width: glyph ? GLYPH_SIZE : NODE_WIDTH,
        height: glyph ? GLYPH_SIZE : ESTIMATED_PART_HEIGHT,
        parent: n.parent,
      };
    });
    layoutTree(
      sizes,
      view.edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
    )
      .then((placed) => {
        if (!live) return;
        setBoxes(placed);
        fitSoon();
      })
      .catch((e: unknown) => {
        // A layout that throws leaves every frame without a size, which looks
        // like a drawing bug rather than a failure. Say so in both places.
        console.error("tensorcad: unfolded layout failed", e);
        if (live) useEditor.getState().setStatus(`Layout failed: ${(e as Error).message}`);
      });
    return () => {
      live = false;
    };
  }, [view, fitSoon]);

  // Lay out automatically the first time a level is opened without positions.
  const autoLaidOut = useRef(new Set<string>());
  useEffect(() => {
    if (view) return;
    const key = `${level.prefix}|${level.graph.nodes.map((n) => n.id).join(",")}`;
    if (autoLaidOut.current.has(key)) return;
    const missing = level.graph.nodes.some((n) => !positions[joinPath(level.prefix, n.id)]);
    if (!missing) return;
    autoLaidOut.current.add(key);
    runLayout().catch((e: unknown) => {
      autoLaidOut.current.delete(key);
      useEditor.getState().setStatus(`Auto-layout failed: ${(e as Error).message}`);
    });
  }, [view, level, positions, runLayout]);

  const firstLayoutRun = useRef(layoutNonce);
  useEffect(() => {
    if (firstLayoutRun.current === layoutNonce) return;
    firstLayoutRun.current = layoutNonce;
    void runLayout();
  }, [layoutNonce, runLayout]);

  // Fit the viewport whenever the breadcrumb changes level.
  useEffect(() => {
    fitSoon();
  }, [level.prefix, fitSoon]);

  /**
   * Hand the viewport to the command list. These live inside React Flow's
   * provider and nothing outside the canvas can reach them, so the canvas
   * pushes them up when it mounts and takes them back when it goes.
   */
  useEffect(() => {
    setViewportApi({
      fit: fitSoon,
      zoomIn: () => void zoomIn({ duration: 140 }),
      zoomOut: () => void zoomOut({ duration: 140 }),
      zoomReset: () => void zoomTo(1, { duration: 140 }),
      duplicateSelection: () => {
        const state = useEditor.getState();
        if (state.detail > 0 || !level.editable) return;
        // Only what is on this level: a selection can outlive a level change,
        // and copying a block into a graph it does not belong to is worse than
        // copying nothing.
        const here = state.selected().filter((p) => level.graph.nodes.some((n) => n.id === idOf(p)));
        if (here.length === 0) return;

        // The names have to be chosen against each other as well as against
        // the level, or two copies of the same block both take `_2`.
        const taken = new Set(level.graph.nodes.map((x) => x.id));
        const made: string[] = [];
        for (const path of here) {
          const source = level.graph.nodes.find((n) => n.id === idOf(path));
          if (!source) continue;
          const copy = structuredClone(source);
          let n = 2;
          while (taken.has(`${source.id}_${n}`)) n++;
          copy.id = `${source.id}_${n}`;
          taken.add(copy.id);
          const at = nodesRef.current.find((x) => x.id === path)?.position;
          useEditor
            .getState()
            .addNode(level.segments, copy, [(at?.x ?? 0) + 48, (at?.y ?? 0) + 48]);
          made.push(joinPath(level.prefix, copy.id));
        }
        useEditor.getState().selectPaths(made);
      },
    });
    return () => setViewportApi(null);
  }, [fitSoon, zoomIn, zoomOut, zoomTo, level]);

  // Centre on a node when something (the Rules panel) asks for it.
  const lastFocus = useRef(focusNonce);
  useEffect(() => {
    if (lastFocus.current === focusNonce) return;
    lastFocus.current = focusNonce;
    if (!selection) return;
    const node = nodesRef.current.find((n) => n.id === selection);
    if (!node) return;
    const w = node.measured?.width ?? NODE_WIDTH;
    const h = node.measured?.height ?? estimateHeight(node.data);
    void setCenter(node.position.x + w / 2, node.position.y + h / 2, {
      zoom: Math.max(getZoom(), 0.85),
      duration: 400,
    });
  }, [focusNonce, selection, setCenter, getZoom]);

  // --- interaction ---------------------------------------------------------

  const checker = useMemo(
    () => createConnectionChecker(doc, derived.symbols, level.segments),
    [doc, derived.symbols, level.segments],
  );

  const toLocal = useCallback(
    (nodeId: string): string => (level.prefix ? nodeId.slice(level.prefix.length + 1) : nodeId),
    [level.prefix],
  );

  const endpoints = useCallback(
    (c: Connection | FlowEdge): { from: string; to: string } | null => {
      if (!c.source || !c.target || !c.sourceHandle || !c.targetHandle) return null;
      return {
        from: `${toLocal(c.source)}:${portOfHandle(c.sourceHandle)}`,
        to: `${toLocal(c.target)}:${portOfHandle(c.targetHandle)}`,
      };
    },
    [toLocal],
  );

  /**
   * Why the wire under the cursor would be refused, or null when it is fine.
   *
   * Refusing silently is the worst of both worlds: the wire does not appear and
   * nothing says why. Blender turns the link red and names the reason, which is
   * what this drives.
   */
  const [rejection, setRejection] = useState<string | null>(null);

  const isValidConnection: IsValidConnection = useCallback(
    (c) => {
      if (!level.editable) {
        setRejection((prev) => (prev === READ_ONLY_REASON ? prev : READ_ONLY_REASON));
        return false;
      }
      const e = endpoints(c);
      if (!e) return false;
      const verdict = checker(e.from, e.to);
      const reason = verdict.ok ? null : (verdict.reason ?? "invalid connection");
      setRejection((prev) => (prev === reason ? prev : reason));
      return verdict.ok;
    },
    [checker, endpoints, level.editable],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      const e = endpoints(c);
      if (!e) return;
      const verdict = checker(e.from, e.to);
      if (!verdict.ok) {
        useEditor.getState().setStatus(`Refused: ${verdict.reason ?? "invalid connection"}`);
        return;
      }
      useEditor.getState().connect(level.segments, e.from, e.to);
    },
    [checker, endpoints, level.segments],
  );

  const clearRejection = useCallback(() => setRejection(null), []);

  /**
   * Moving one end of an existing wire.
   *
   * Without this React Flow's `edgesReconnectable` does nothing, and changing
   * where a wire goes means deleting it and drawing it again — which no
   * schematic editor has ever asked anyone to do. `held` records whether the
   * drop landed on a pin: `onReconnectEnd` fires either way, and a drop into
   * empty space has to leave the wire where it was rather than delete it,
   * because an edge with one end attached is not something this IR can hold.
   */
  const reconnected = useRef(false);

  const onReconnectStart = useCallback(() => {
    reconnected.current = false;
    setConnecting(true);
    clearRejection();
  }, [clearRejection]);

  const onReconnect = useCallback(
    (oldEdge: FlowEdge, c: Connection) => {
      if (!level.editable) return;
      const was = oldEdge.data as { from?: string; to?: string } | undefined;
      const now = endpoints(c);
      if (!was?.from || !was?.to || !now) return;
      if (was.from === now.from && was.to === now.to) {
        reconnected.current = true;
        return;
      }
      const verdict = checker(now.from, now.to);
      if (!verdict.ok) {
        useEditor.getState().setStatus(`Refused: ${verdict.reason ?? "invalid connection"}`);
        return;
      }
      reconnected.current = true;
      useEditor.getState().reconnect(level.segments, was.from, was.to, now);
    },
    [checker, endpoints, level.editable, level.segments],
  );

  const onReconnectEnd = useCallback(() => {
    setConnecting(false);
    clearRejection();
    // Dropped in space: the wire stays as it was. Nothing to undo, nothing to
    // explain, which is what every drafting program does with a cancelled drag.
    reconnected.current = false;
  }, [clearRejection]);


  const onEdgesDelete = useCallback(
    (deleted: FlowEdge[]) => {
      if (!level.editable) return;
      const act = useEditor.getState();
      for (const e of deleted) {
        const data = e.data as { from?: string; to?: string } | undefined;
        if (data?.from && data?.to) act.disconnect(level.segments, data.from, data.to);
      }
    },
    [level.editable, level.segments],
  );

  const onNodesDelete = useCallback(
    (deleted: CanvasNode[]) => {
      if (!level.editable) return;
      const act = useEditor.getState();
      for (const n of deleted) act.removeNode(n.id);
    },
    [level.editable],
  );

  const onNodeDragStop = useCallback(
    (_: unknown, _node: CanvasNode, dragged: CanvasNode[]) => {
      const moves = dragged.map((n) => ({
        path: n.id,
        xy: [n.position.x, n.position.y] as [number, number],
      }));
      if (level.editable) useEditor.getState().moveNodes(moves);
      else {
        setEphemeral((prev) => {
          const next = { ...prev };
          for (const m of moves) next[m.path] = [Math.round(m.xy[0]), Math.round(m.xy[1])];
          return next;
        });
      }
    },
    [level.editable],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const type = event.dataTransfer.getData(DRAG_MIME);
      if (!type) return;
      if (!level.editable) {
        useEditor.getState().setStatus("This level is read-only");
        return;
      }
      const node = newNodeFor(type, doc);
      if (!node) return;
      const p = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      useEditor.getState().addNode(level.segments, node, [p.x - NODE_WIDTH / 2, p.y - 30]);
    },
    [level.editable, level.segments, screenToFlowPosition],
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  return (
    <div className="canvas" onDrop={onDrop} onDragOver={onDragOver}>
      {!level.editable && (
        <div className="strip strip--readonly">
          <strong>Read-only</strong>
          <span>
            The catalog expansion of{" "}
            <strong>{typeName(blockDef(level.owner?.type ?? "", doc), level.owner?.type ?? "")}</strong>. Composite interiors are
            generated, not stored &mdash; edit the block&rsquo;s parameters instead.
          </span>
        </div>
      )}
      {level.error && (
        <div className="strip strip--error">
          <strong>Cannot open</strong>
          <span>{level.error}</span>
        </div>
      )}
      <div
        className={`canvas__flow tool-${tool}${rejection ? " is-rejecting" : ""}`}
        onPointerMove={(e) => {
          useEditor
            .getState()
            .setCanvasStatus({ cursor: screenToFlowPosition({ x: e.clientX, y: e.clientY }) });
        }}
        onPointerLeave={() => useEditor.getState().setCanvasStatus({ cursor: null })}
      >
      <ReactFlow<CanvasNode>
        className={connecting || tool === "wire" ? "is-connecting" : undefined}
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onConnect={onConnect}
        isValidConnection={isValidConnection}
        onConnectStart={(e, params) => {
          setConnecting(true);
          clearRejection();
          void e;
          void params;
        }}
        onConnectEnd={() => {
          setConnecting(false);
          clearRejection();
        }}
        onReconnect={onReconnect}
        onReconnectStart={onReconnectStart}
        onReconnectEnd={onReconnectEnd}
        onEdgeMouseEnter={(_, e) => setLitNet((e.data as { net?: string } | undefined)?.net ?? null)}
        onEdgeMouseLeave={() => setLitNet(null)}
        // What this part is, where the pointer already is. The card is placed
        // in the canvas's own coordinates so it does not move with the drawing:
        // it annotates the pointer, not the sheet.
        onNodeMouseEnter={(e, n) => {
          const d = n.data as BlockNodeData | FrameNodeData;
          if (!d.docs?.summary && !d.docs?.formula) return setHover(null);
          const host = e.currentTarget.closest(".canvas__flow") ?? e.currentTarget;
          const box = host.getBoundingClientRect();
          setHover({
            x: e.clientX - box.left,
            y: e.clientY - box.top,
            name: d.typeName,
            type: d.type,
            docs: d.docs,
            drillable: (d as BlockNodeData).drillable ?? false,
          });
        }}
        onNodeMouseLeave={() => setHover(null)}
        // Anything that moves the drawing under the pointer invalidates where
        // the card was put, so it goes rather than lags.
        onNodeDragStart={() => setHover(null)}
        onMoveStart={() => setHover(null)}
        // A wire is a tensor, and a tensor is selectable. It goes to the
        // inspector like a block does, and every other segment of the same net
        // lights with it, because they are one tensor and not several.
        onEdgeClick={(_, e) => {
          const net = (e.data as { net?: string } | undefined)?.net;
          if (net) useEditor.getState().selectNet(net);
        }}
        onEdgesDelete={onEdgesDelete}
        onNodesDelete={onNodesDelete}
        onNodeDragStop={onNodeDragStop}
        // A plain click replaces the selection. A modified one adds to it, and
        // that is React Flow's own gesture: it reports the whole set through
        // `onNodesChange`, so claiming the selection here as well would undo
        // what it just said.
        onNodeClick={(e, n) => {
          if (e.ctrlKey || e.metaKey || e.shiftKey) return;
          useEditor.getState().select(n.id);
        }}
        onNodeDoubleClick={(_, n) => {
          // In an unfolded drawing, opening a block means showing one more
          // level of it in place rather than replacing the view.
          if (view) {
            useEditor.getState().setDetail(detail + 1);
            return;
          }
          if (n.data.drillable) useEditor.getState().enter(n.id);
        }}
        onPaneClick={() => useEditor.getState().select(null)}
        onMove={(_, viewport) => useEditor.getState().setCanvasStatus({ zoom: viewport.zoom })}
        nodesConnectable={level.editable && !view && tool !== "pan"}
        nodesDraggable={!view && tool === "select"}
        elementsSelectable={tool !== "pan"}
        // Pan is a mode; with any other tool the middle and right buttons still
        // pan, which is what every drawing program does and what a trackpad
        // user expects.
        panOnDrag={tool === "pan" ? true : [1, 2]}
        selectionOnDrag={tool === "select"}
        panOnScroll={false}
        zoomOnDoubleClick={false}
        // A wire can be re-pointed by dragging either end, which is the one
        // editing gesture a schematic cannot do without. The radius is the
        // distance from an endpoint at which the drag grabs the wire rather
        // than starting a box selection; ten is too mean to find by feel.
        edgesReconnectable={level.editable && !view}
        reconnectRadius={18}
        // Where a dropped connection line still counts as landing on a pin.
        // Twenty is React Flow's default and assumes a mouse on a small graph;
        // a sheet this dense is worked at low zoom with a trackpad.
        connectionRadius={34}
        // Zero means a single pixel of travel between press and release turns a
        // click into a drag and the click is lost. On a trackpad that is most
        // clicks. Four pixels is the slop every desktop toolkit allows.
        paneClickDistance={4}
        nodeClickDistance={4}
        // Delete is the key on the keyboard that says delete. Backspace stays
        // because it is what React Flow documents and what muscle memory from
        // the browser expects.
        deleteKeyCode={["Delete", "Backspace"]}
        // A selected wire has to be visible over the blocks it runs between,
        // or selecting it tells you nothing.
        elevateEdgesOnSelect
        proOptions={{ hideAttribution: true }}
        // Below about a third scale the labels are sub-pixel and the browser
        // starts dropping them unevenly, which reads as flicker. Stopping there
        // keeps a big design legible rather than letting it turn into confetti.
        minZoom={0.3}
        maxZoom={2.5}
        snapToGrid={snap}
        snapGrid={[GRID_MINOR, GRID_MINOR]}
        connectionLineStyle={{ stroke: palette.accent, strokeWidth: 2 }}
        fitView
      >
        {/* Two grids, as every CAD canvas has: a fine one to snap to and a
            coarse one to judge distance by. The fine one is dropped once it is
            too dense to read, which is also where it starts to shimmer. */}
        {showGrid && (
          <>
            <Background
              id="minor"
              variant={BackgroundVariant.Lines}
              gap={GRID_MINOR}
              lineWidth={1}
              color={palette.gridMinor}
            />
            <Background
              id="major"
              variant={BackgroundVariant.Lines}
              gap={GRID_MAJOR}
              lineWidth={1}
              color={palette.gridMajor}
            />
          </>
        )}
        <CalloutLayer callouts={callouts} />
        <Panel position="top-left">
          <Key />
        </Panel>
        <Panel position="top-right">
          <Walkthrough />
        </Panel>
        {showTitleBlock && (
          <Panel position="bottom-right">
            <TitleBlock />
          </Panel>
        )}
        <Controls showInteractive={false} />
        {rejection && (
          <div className="reject-chip" role="alert">
            <span className="reject-chip__glyph">&#9679;</span>
            {rejection}
          </div>
        )}
        {showMinimap && (
          <MiniMap
            position="bottom-left"
            pannable
            zoomable
            nodeColor={(n) => categoryColor((n.data as BlockNodeData | undefined)?.category)}
            maskColor="rgba(120, 128, 138, 0.28)"
            style={{
              background: palette.sheet,
              border: `1px solid ${palette.border}`,
              width: 128,
              height: 84,
              // Sits beside the tool strip rather than over the drawing.
              marginBottom: 8,
              marginLeft: 8,
            }}
          />
        )}
      </ReactFlow>
        {hover && (
          <div
            className="partdoc-card"
            style={{ left: hover.x, top: hover.y }}
            role="tooltip"
            aria-hidden
          >
            <PartDoc
              name={hover.name}
              type={hover.type}
              docs={hover.docs}
              drillable={hover.drillable}
            />
          </div>
        )}
        {level.graph.nodes.length === 0 && (
          <div className="banner banner--empty">
            <strong>Empty graph</strong>
            <span>Drag a block from the palette, or load a preset from the toolbar.</span>
          </div>
        )}
      </div>
    </div>
  );
}
