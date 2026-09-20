import type {
  Doc,
  Edge,
  Graph,
  NodeDef,
  ParamValue,
  RuleSeverity,
  SymbolDef,
  UserBlockDef,
} from "@tensor-cad/engine";
import { splitEndpoint } from "@tensor-cad/engine";
import { CATALOG } from "../engine.js";
import { DEF_PREFIX, storeInDefinition } from "./definition.js";
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

/**
 * The graph that owns the nodes at `segments`, walking container subgraphs.
 *
 * A path beginning `@def` addresses a block the document defines for itself, and
 * hands over that definition's stored template. Every operation below reaches
 * its graph through here, so one function is what makes a definition editable
 * with the tools already written rather than a second set that writes to `defs`.
 * `@` is not a legal node id, so a path can never mean this by accident.
 */
export function graphAtPath(doc: Doc, segments: Segments): Graph | null {
  let graph: Graph;
  let rest = segments;
  if (segments[0] === DEF_PREFIX) {
    const def = (doc.defs as Record<string, UserBlockDef> | undefined)?.[segments[1] ?? ""];
    if (!def?.graph) return null;
    graph = def.graph;
    rest = segments.slice(2);
  } else {
    graph = doc.graph;
  }
  for (const seg of rest) {
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
  // Nothing to remove is not the same as removing nothing. The timeline reads
  // an unchanged document as "this step could not replay", and a filter that
  // matched nothing would have said it succeeded.
  if (!graph.nodes.some((n) => n.id === id)) return doc;
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
  if (value === undefined) {
    delete node.params[key];
    return next;
  }
  // Inside a definition the template is written in the block's own parameters,
  // so a value naming one is stored as a reference to it rather than as the
  // number it happens to resolve to today. That is the whole point of a
  // definition: `D` means the width, not 4096.
  const inDef = segments[0] === DEF_PREFIX ? (segments[1] ?? "") : null;
  node.params[key] = inDef === null ? value : (storeInDefinition(next, inDef, value) as ParamValue);
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
  if (!hasEdge(graph, from, to)) return doc;
  graph.edges = graph.edges.filter(([f, t]) => !(f === from && t === to));
  return next;
}

export function setSymbol(doc: Doc, name: string, def: SymbolDef | undefined): Doc {
  const next = cloneDoc(doc);
  if (def === undefined) delete next.symbols[name];
  else next.symbols[name] = def;
  return next;
}

/**
 * Record what a rule means to this design, or stop recording it.
 *
 * The object goes away entirely when the last decision is removed, so a design
 * that has decided nothing does not carry an empty `rules: {}` in its file.
 */
export function setRuleSeverity(doc: Doc, rule: string, severity: RuleSeverity | undefined): Doc {
  const next = cloneDoc(doc);
  const rules = { ...(next.rules ?? {}) };
  if (severity === undefined) delete rules[rule];
  else rules[rule] = severity;
  if (Object.keys(rules).length === 0) delete next.rules;
  else next.rules = rules;
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
  // A position for a block that is not there is a stale key in the ui map and,
  // on a replay, a step reporting success for work it did not do.
  const real = moves.filter((m) => nodeAtPath(doc, segmentsOf(m.path)));
  if (real.length === 0) return doc;
  const next = cloneDoc(doc);
  const pos = positions(next);
  for (const m of real) pos[m.path] = [Math.round(m.xy[0]), Math.round(m.xy[1])];
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

/**
 * Build the design at a named configuration, or at its own symbols.
 *
 * A configuration is a view of the design and not an edit to it, so this only
 * moves which one is in force. Switching twice ends up where it started, which
 * is what makes it safe to click through them.
 */
export function setActiveConfiguration(doc: Doc, name: string | null): Doc {
  const next = cloneDoc(doc);
  if (name === null || name === "") delete next.active;
  else next.active = name;
  return next;
}

/**
 * Name the symbols as they stand now, as a configuration.
 *
 * Only the design symbols, and only the ones that differ from what the document
 * itself says: a configuration that restated every number would stop following
 * an expression, which is the thing that makes one worth having.
 */
export function captureConfiguration(doc: Doc, name: string, docText?: string): Doc {
  const next = cloneDoc(doc);
  const base = doc.symbols;
  const active = doc.active ? (doc.configurations?.[doc.active]?.symbols ?? {}) : {};
  const symbols: Record<string, SymbolDef> = {};
  for (const [key, def] of Object.entries(active)) {
    if (JSON.stringify(def) !== JSON.stringify(base[key])) symbols[key] = def;
  }
  next.configurations = { ...next.configurations, [name]: { ...(docText ? { doc: docText } : {}), symbols } };
  next.active = name;
  return next;
}

/** Forget a configuration. The design's own symbols are never touched. */
export function removeConfiguration(doc: Doc, name: string): Doc {
  const next = cloneDoc(doc);
  if (!next.configurations?.[name]) return doc;
  const rest = { ...next.configurations };
  delete rest[name];
  next.configurations = rest;
  if (next.active === name) delete next.active;
  return next;
}

/**
 * Change a symbol while a configuration is in force.
 *
 * The edit lands in the configuration, not in the design: that is what it means
 * for one to be in force. With none in force it is an ordinary symbol edit.
 */
export function setSymbolInConfiguration(
  doc: Doc,
  name: string,
  def: SymbolDef | undefined,
): Doc {
  const active = doc.active;
  if (!active || !doc.configurations?.[active]) return setSymbol(doc, name, def);
  const next = cloneDoc(doc);
  const config = next.configurations![active]!;
  const symbols = { ...config.symbols };
  if (def === undefined) delete symbols[name];
  else symbols[name] = def;
  next.configurations = { ...next.configurations, [active]: { ...config, symbols } };
  return next;
}

export function setMetaName(doc: Doc, name: string): Doc {
  if (doc.meta.name === name) return doc;
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
