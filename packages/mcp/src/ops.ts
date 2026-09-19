/**
 * The edit operations an agent can apply to a design.
 *
 * One patch tool beats thirty setters: it keeps the tool count under the caps
 * some clients impose, and a batch either lands whole or not at all, so a
 * half-applied edit can never be observed.
 */

import {
  getBlock,
  joinPath,
  splitEndpoint,
  type Doc,
  type Graph,
  type NodeDef,
  type ParamValue,
  type SymbolDef,
} from "@tensorcad/core";

export type Op =
  | {
      op: "add_node";
      /** Container node whose subgraph receives the block. Empty or absent means the root graph. */
      parent?: string;
      id: string;
      type: string;
      params?: Record<string, ParamValue>;
      label?: string;
    }
  | { op: "remove_node"; path: string }
  | { op: "set_param"; path: string; key: string; value: ParamValue }
  | { op: "connect"; graph?: string; from: string; to: string }
  | { op: "disconnect"; graph?: string; from: string; to: string }
  | { op: "set_symbol"; name: string; value: number | string | null; doc?: string; runtime?: boolean }
  | { op: "rename"; path: string; id: string }
  | { op: "set_label"; path: string; label?: string | null };

/** A rejected operation. The message is meant to be read by a model. */
export class OpError extends Error {
  constructor(
    public readonly index: number,
    public readonly op: Op,
    message: string,
  ) {
    super(`op ${index} (${op.op}): ${message}`);
    this.name = "OpError";
  }
}

// ---------------------------------------------------------------------------
// Path navigation
// ---------------------------------------------------------------------------

/** The graph a node path lives in, plus the node itself. */
function locate(doc: Doc, path: string): { graph: Graph; node: NodeDef; parent: string } {
  const segments = path.split("/").filter(Boolean);
  if (segments.length === 0) throw new Error(`"${path}" is not a node path`);

  let graph: Graph = doc.graph;
  for (let i = 0; i < segments.length - 1; i++) {
    const id = segments[i];
    const node = graph.nodes.find((n) => n.id === id);
    if (!node) throw new Error(`no block "${segments.slice(0, i + 1).join("/")}"`);
    if (!node.graph) throw new Error(`block "${segments.slice(0, i + 1).join("/")}" has no subgraph`);
    graph = node.graph;
  }

  const id = segments[segments.length - 1];
  const node = graph.nodes.find((n) => n.id === id);
  if (!node) throw new Error(`no block "${path}"`);
  return { graph, node, parent: segments.slice(0, -1).join("/") };
}

/** The graph identified by a container path; the empty string is the root. */
export function graphAt(doc: Doc, path: string | undefined): Graph {
  if (!path) return doc.graph;
  const { node } = locate(doc, path);
  if (!node.graph) throw new Error(`block "${path}" is not a container and has no subgraph`);
  return node.graph;
}

/** Every node path in the document, parents before children. */
export function allPaths(graph: Graph, prefix = "", out: string[] = []): string[] {
  for (const node of graph.nodes) {
    const path = joinPath(prefix, node.id);
    out.push(path);
    if (node.graph) allPaths(node.graph, path, out);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

const edgeEq = (a: readonly [string, string], from: string, to: string) => a[0] === from && a[1] === to;

function applyOne(doc: Doc, op: Op): string {
  switch (op.op) {
    case "add_node": {
      if (!op.id) throw new Error("id is required");
      const def = getBlock(op.type);
      if (!def) throw new Error(`unknown block type "${op.type}"`);
      const graph = graphAt(doc, op.parent);
      if (graph.nodes.some((n) => n.id === op.id)) {
        throw new Error(`"${joinPath(op.parent ?? "", op.id)}" already exists`);
      }
      const node: NodeDef = { id: op.id, type: op.type };
      if (op.params) node.params = { ...op.params };
      if (op.label) node.label = op.label;
      // A container is useless without somewhere to put things, so give it an
      // empty subgraph that later ops can add into.
      if (def.kind === "container") node.graph = { nodes: [], edges: [] };
      graph.nodes.push(node);
      return `added ${joinPath(op.parent ?? "", op.id)} (${op.type})`;
    }

    case "remove_node": {
      const { graph, node } = locate(doc, op.path);
      const index = graph.nodes.indexOf(node);
      graph.nodes.splice(index, 1);
      const before = graph.edges.length;
      graph.edges = graph.edges.filter(
        ([from, to]) => splitEndpoint(from).node !== node.id && splitEndpoint(to).node !== node.id,
      );
      const dropped = before - graph.edges.length;
      return `removed ${op.path}${dropped > 0 ? ` and ${dropped} edge${dropped === 1 ? "" : "s"}` : ""}`;
    }

    case "set_param": {
      const { node } = locate(doc, op.path);
      const def = getBlock(node.type);
      if (def && !(op.key in (def.params ?? {}))) {
        const known = Object.keys(def.params ?? {}).join(", ");
        throw new Error(`"${node.type}" has no parameter "${op.key}". Known: ${known || "(none)"}`);
      }
      node.params ??= {};
      const previous = node.params[op.key];
      if (op.value === undefined) delete node.params[op.key];
      else node.params[op.key] = op.value;
      return `${op.path}.${op.key}: ${JSON.stringify(previous ?? null)} -> ${JSON.stringify(op.value ?? null)}`;
    }

    case "connect": {
      const graph = graphAt(doc, op.graph);
      assertEndpoint(graph, op.from, "from");
      assertEndpoint(graph, op.to, "to");
      if (graph.edges.some((e) => edgeEq(e, op.from, op.to))) {
        throw new Error(`${op.from} -> ${op.to} already exists`);
      }
      const occupied = graph.edges.find((e) => e[1] === op.to);
      if (occupied) throw new Error(`${op.to} already receives ${occupied[0]}; disconnect it first`);
      graph.edges.push([op.from, op.to]);
      return `connected ${op.from} -> ${op.to}`;
    }

    case "disconnect": {
      const graph = graphAt(doc, op.graph);
      const index = graph.edges.findIndex((e) => edgeEq(e, op.from, op.to));
      if (index < 0) throw new Error(`no edge ${op.from} -> ${op.to}`);
      graph.edges.splice(index, 1);
      return `disconnected ${op.from} -> ${op.to}`;
    }

    case "set_symbol": {
      if (!op.name) throw new Error("name is required");
      const existing = doc.symbols[op.name];
      if (op.value === null) {
        if (existing === undefined) throw new Error(`no symbol "${op.name}"`);
        delete doc.symbols[op.name];
        return `removed symbol ${op.name}`;
      }

      const wasRuntime =
        op.runtime ?? (typeof existing === "object" && existing !== null && existing.kind === "runtime");

      let next: SymbolDef;
      if (wasRuntime) {
        if (typeof op.value !== "number") throw new Error(`runtime symbol "${op.name}" needs a number default`);
        next = { kind: "runtime", default: op.value };
      } else {
        next = { kind: "design", value: op.value };
      }
      const docString =
        op.doc ?? (typeof existing === "object" && existing !== null ? existing.doc : undefined);
      if (docString) (next as { doc?: string }).doc = docString;

      doc.symbols[op.name] = next;
      return `${op.name}: ${describeSymbol(existing)} -> ${describeSymbol(next)}`;
    }

    case "rename": {
      const { graph, node, parent } = locate(doc, op.path);
      if (!op.id) throw new Error("id is required");
      if (op.id === node.id) return `${op.path} unchanged`;
      if (graph.nodes.some((n) => n.id === op.id)) {
        throw new Error(`"${joinPath(parent, op.id)}" already exists`);
      }
      const old = node.id;
      node.id = op.id;
      graph.edges = graph.edges.map(([from, to]) => [rewrite(from, old, op.id), rewrite(to, old, op.id)]);
      return `renamed ${op.path} -> ${joinPath(parent, op.id)}`;
    }

    case "set_label": {
      const { node } = locate(doc, op.path);
      if (op.label === null || op.label === undefined || op.label === "") delete node.label;
      else node.label = op.label;
      return `${op.path} label -> ${op.label ?? "(none)"}`;
    }

    default: {
      const bad = op as { op: string };
      throw new Error(`unknown operation "${bad.op}"`);
    }
  }
}

function describeSymbol(s: SymbolDef | undefined): string {
  if (s === undefined) return "(unset)";
  if (typeof s === "number" || typeof s === "string") return String(s);
  return s.kind === "runtime" ? `runtime(${s.default})` : String(s.value);
}

function rewrite(endpoint: string, from: string, to: string): string {
  const { node, port } = splitEndpoint(endpoint);
  return node === from ? `${to}:${port}` : endpoint;
}

function assertEndpoint(graph: Graph, endpoint: string, which: string): void {
  let split: { node: string; port: string };
  try {
    split = splitEndpoint(endpoint);
  } catch {
    throw new Error(`${which} endpoint "${endpoint}" is not "blockId:port"`);
  }
  if (!graph.nodes.some((n) => n.id === split.node)) {
    const known = graph.nodes.map((n) => n.id).join(", ");
    throw new Error(`${which} endpoint "${endpoint}" names no block in this graph. Blocks: ${known}`);
  }
}

export interface ApplyResult {
  doc: Doc;
  /** One human-readable line per operation, in order. */
  applied: string[];
}

/**
 * Apply a batch to a copy of the document. The original is never touched, and
 * the first failing operation aborts the whole batch.
 */
export function applyOps(doc: Doc, ops: Op[]): ApplyResult {
  const next = structuredClone(doc);
  const applied: string[] = [];
  for (let i = 0; i < ops.length; i++) {
    try {
      applied.push(applyOne(next, ops[i]));
    } catch (e) {
      throw new OpError(i, ops[i], (e as Error).message);
    }
  }
  return { doc: next, applied };
}
