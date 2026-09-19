/**
 * `.tensorcad.json` read and write.
 *
 * Keys come out in a fixed order so a saved design diffs cleanly in git.
 */

import type { Doc, Graph, NodeDef } from "@tensorcad/core";
import { DOC_VERSION } from "@tensorcad/core";

const DOC_KEYS = ["version", "meta", "symbols", "graph", "ui"] as const;
const META_KEYS = ["name", "family", "notes", "published"] as const;
const NODE_KEYS = ["id", "type", "label", "params", "graph", "variants"] as const;
const GRAPH_KEYS = ["nodes", "edges"] as const;

function pick<T extends object>(src: T, keys: readonly (keyof T & string)[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const k of keys) if (src[k] !== undefined) out[k] = src[k];
  for (const k of Object.keys(src)) {
    if (!keys.includes(k as keyof T & string) && (src as Record<string, unknown>)[k] !== undefined) {
      out[k] = (src as Record<string, unknown>)[k];
    }
  }
  return out;
}

function orderNode(node: NodeDef): Record<string, unknown> {
  const out = pick(node, NODE_KEYS as unknown as (keyof NodeDef & string)[]);
  if (node.graph) out.graph = orderGraph(node.graph);
  if (node.variants) {
    out.variants = Object.fromEntries(
      Object.entries(node.variants).map(([k, g]) => [k, orderGraph(g)]),
    );
  }
  return out;
}

function orderGraph(graph: Graph): Record<string, unknown> {
  const out = pick(graph, GRAPH_KEYS as unknown as (keyof Graph & string)[]);
  out.nodes = graph.nodes.map(orderNode);
  out.edges = graph.edges;
  return out;
}

function orderDoc(doc: Doc): Record<string, unknown> {
  const out = pick(doc, DOC_KEYS as unknown as (keyof Doc & string)[]);
  out.meta = pick(doc.meta, META_KEYS as unknown as (keyof typeof doc.meta & string)[]);
  out.graph = orderGraph(doc.graph);
  if (doc.ui) {
    const positions = doc.ui.positions ?? {};
    const sorted: Record<string, [number, number]> = {};
    for (const key of Object.keys(positions).sort()) sorted[key] = positions[key];
    out.ui = { positions: sorted, ...(doc.ui.collapsed ? { collapsed: doc.ui.collapsed } : {}) };
  }
  return out;
}

export function serializeDoc(doc: Doc): string {
  return JSON.stringify(orderDoc(doc), null, 2) + "\n";
}

export function parseDoc(text: string): Doc {
  const raw = JSON.parse(text) as Partial<Doc>;
  if (typeof raw !== "object" || raw === null) throw new Error("Not a JSON object");
  if (raw.version !== DOC_VERSION) {
    throw new Error(`Unsupported document version ${String(raw.version)}, expected ${DOC_VERSION}`);
  }
  if (!raw.graph || !Array.isArray(raw.graph.nodes) || !Array.isArray(raw.graph.edges)) {
    throw new Error("Document has no graph");
  }
  return {
    version: DOC_VERSION,
    meta: raw.meta ?? { name: "untitled" },
    symbols: raw.symbols ?? {},
    graph: raw.graph,
    ui: raw.ui ?? { positions: {} },
  };
}

export function fileNameFor(doc: Doc): string {
  const base = (doc.meta.name || "design").replace(/[^A-Za-z0-9._-]+/g, "-");
  return `${base}.tensorcad.json`;
}

/** Hand a file to the browser. The desktop build routes through Go instead. */
export function downloadText(contents: string, name: string, type = "text/plain"): void {
  const blob = new Blob([contents], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadDoc(doc: Doc): void {
  downloadText(serializeDoc(doc), fileNameFor(doc), "application/json");
}
