import type { Doc, Edge, Graph, NodeDef, ParamValue, SymbolDef } from "@tensorcad/engine";
import { splitEndpoint } from "@tensorcad/engine";
import { CATALOG } from "../engine.js";
/**
 * Document operations.
 *
 * Every mutation the editor performs goes through one of these pure functions:
 * they take a document and return a new one. The store wraps them so undo/redo
 * is uniform and nothing can edit the document behind the store's back.
 */


export type Segments = string[];

export function cloneDoc(doc: Doc): Doc {
  return structuredClone(doc);
}

export function pathOf(segments: Segments): string {
  return segments.join("/");
}

export function segmentsOf(path: string): Segments {
  return path ? path.split("/") : [];
}

/** The graph that owns the nodes at `segments`, walking container subgraphs. */
export function graphAtPath(doc: Doc, segments: Segments): Graph | null {
  let graph: Graph = doc.graph;
  for (const seg of segments) {
    const node = graph.nodes.find((n) => n.id === seg);
    if (!node || !node.graph) return null;
    graph = node.graph;
  }
  return graph;
}

export function nodeAtPath(doc: Doc, segments: Segments): NodeDef | null {
  if (segments.length === 0) return null;
  const parent = graphAtPath(doc, segments.slice(0, -1));
  if (!parent) return null;
  return parent.nodes.find((n) => n.id === segments[segments.length - 1]) ?? null;
}

/** True when the node can be opened as its own canvas level. */
export function isDrillable(node: NodeDef): boolean {
  const def = CATALOG[node.type];
  if (!def) return false;
  if (def.kind === "container") return Boolean(node.graph);
  return def.kind === "composite";
}

export function uniqueId(graph: Graph, base: string): string {
  const taken = new Set(graph.nodes.map((n) => n.id));
  if (!taken.has(base)) return base;
  for (let i = 2; ; i++) {
    const candidate = `${base}_${i}`;
    if (!taken.has(candidate)) return candidate;
  }
}

function positions(doc: Doc): Record<string, [number, number]> {
  doc.ui ??= {};
  doc.ui.positions ??= {};
  return doc.ui.positions;
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

export function addNode(doc: Doc, parent: Segments, node: NodeDef, xy?: [number, number]): Doc {
  const next = cloneDoc(doc);
  const graph = graphAtPath(next, parent);
  if (!graph) return doc;
  const id = uniqueId(graph, node.id);
  graph.nodes.push({ ...node, id });
  if (xy) positions(next)[pathOf([...parent, id])] = [Math.round(xy[0]), Math.round(xy[1])];
  return next;
}

export function removeNode(doc: Doc, segments: Segments): Doc {
  if (segments.length === 0) return doc;
  const next = cloneDoc(doc);
  const parent = segments.slice(0, -1);
  const id = segments[segments.length - 1];
  const graph = graphAtPath(next, parent);
  if (!graph) return doc;
  graph.nodes = graph.nodes.filter((n) => n.id !== id);
  graph.edges = graph.edges.filter(
    ([from, to]) => splitEndpoint(from).node !== id && splitEndpoint(to).node !== id,
  );
  const prefix = pathOf(segments);
  const pos = positions(next);
  for (const key of Object.keys(pos)) {
    if (key === prefix || key.startsWith(prefix + "/")) delete pos[key];
  }
  return next;
}

export function setParam(
  doc: Doc,
  segments: Segments,
  key: string,
  value: ParamValue | undefined,
): Doc {
  const next = cloneDoc(doc);
  const node = nodeAtPath(next, segments);
  if (!node) return doc;
  node.params ??= {};
  if (value === undefined) delete node.params[key];
  else node.params[key] = value;
  return next;
}

function hasEdge(graph: Graph, from: string, to: string): boolean {
  return graph.edges.some(([f, t]) => f === from && t === to);
}

export function connect(doc: Doc, parent: Segments, from: string, to: string): Doc {
  const next = cloneDoc(doc);
  const graph = graphAtPath(next, parent);
  if (!graph) return doc;
  if (hasEdge(graph, from, to)) return doc;
  // An input port takes exactly one producer.
  graph.edges = graph.edges.filter(([, t]) => t !== to);
  graph.edges.push([from, to] as Edge);
  return next;
}

export function disconnect(doc: Doc, parent: Segments, from: string, to: string): Doc {
  const next = cloneDoc(doc);
  const graph = graphAtPath(next, parent);
  if (!graph) return doc;
  graph.edges = graph.edges.filter(([f, t]) => !(f === from && t === to));
  return next;
}

export function setSymbol(doc: Doc, name: string, def: SymbolDef | undefined): Doc {
  const next = cloneDoc(doc);
  if (def === undefined) delete next.symbols[name];
  else next.symbols[name] = def;
  return next;
}

export function renameSymbol(doc: Doc, from: string, to: string): Doc {
  if (from === to || !to) return doc;
  const next = cloneDoc(doc);
  const rebuilt: Record<string, SymbolDef> = {};
  for (const [k, v] of Object.entries(next.symbols)) rebuilt[k === from ? to : k] = v;
  next.symbols = rebuilt;
  return next;
}

export function moveNode(doc: Doc, segments: Segments, xy: [number, number]): Doc {
  return moveNodes(doc, [{ path: pathOf(segments), xy }]);
}

export function moveNodes(doc: Doc, moves: { path: string; xy: [number, number] }[]): Doc {
  if (moves.length === 0) return doc;
  const next = cloneDoc(doc);
  const pos = positions(next);
  for (const m of moves) pos[m.path] = [Math.round(m.xy[0]), Math.round(m.xy[1])];
  return next;
}

export function renameNode(doc: Doc, segments: Segments, label: string | undefined): Doc {
  const next = cloneDoc(doc);
  const node = nodeAtPath(next, segments);
  if (!node) return doc;
  if (label === undefined || label.trim() === "") delete node.label;
  else node.label = label;
  return next;
}

/** Change a node's id, rewriting the edges and ui keys that mention it. */
export function setNodeId(doc: Doc, segments: Segments, rawId: string): Doc {
  if (segments.length === 0) return doc;
  const id = rawId.trim();
  if (!id || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(id)) return doc;
  const oldId = segments[segments.length - 1];
  if (id === oldId) return doc;
  const next = cloneDoc(doc);
  const parent = segments.slice(0, -1);
  const graph = graphAtPath(next, parent);
  if (!graph) return doc;
  if (graph.nodes.some((n) => n.id === id)) return doc;
  const node = graph.nodes.find((n) => n.id === oldId);
  if (!node) return doc;
  node.id = id;
  const swap = (endpoint: string): string => {
    const parts = splitEndpoint(endpoint);
    return parts.node === oldId ? id + ":" + parts.port : endpoint;
  };
  graph.edges = graph.edges.map(([f, t]) => [swap(f), swap(t)] as Edge);
  const oldPrefix = pathOf([...parent, oldId]);
  const newPrefix = pathOf([...parent, id]);
  const pos = positions(next);
  for (const key of Object.keys(pos)) {
    if (key === oldPrefix || key.startsWith(oldPrefix + "/")) {
      pos[newPrefix + key.slice(oldPrefix.length)] = pos[key];
      delete pos[key];
    }
  }
  return next;
}

export function setMetaName(doc: Doc, name: string): Doc {
  const next = cloneDoc(doc);
  next.meta = { ...next.meta, name };
  return next;
}

/** A blank document, used by "New". */
export function emptyDoc(): Doc {
  return {
    version: 1,
    meta: { name: "untitled" },
    symbols: {
      B: { kind: "runtime", default: 1, doc: "Batch size" },
      T: { kind: "runtime", default: 2048, doc: "Sequence length in tokens" },
      L: { kind: "design", value: 12, doc: "Number of transformer layers" },
      D: { kind: "design", value: 768, doc: "Residual stream width (d_model)" },
      H: { kind: "design", value: 12, doc: "Query heads" },
      Hkv: { kind: "design", value: 12, doc: "Key/value heads" },
      dh: { kind: "design", value: 64, doc: "Head dimension" },
      F: { kind: "design", value: "4*D", doc: "Feed-forward hidden width" },
      V: { kind: "design", value: 50257, doc: "Vocabulary size" },
    },
    graph: { nodes: [], edges: [] },
    ui: { positions: {} },
  };
}

/** Container helper: a `repeat` node pre-filled with its boundary nodes. */
export function repeatSkeleton(id: string): NodeDef {
  return {
    id,
    type: "repeat",
    params: { count: "L" },
    graph: {
      nodes: [
        { id: "_in", type: "boundary_in", params: { ports: { x: "B T D" } } },
        { id: "_out", type: "boundary_out", params: { ports: { x: "B T D" } } },
      ],
      edges: [],
    },
  };
}
