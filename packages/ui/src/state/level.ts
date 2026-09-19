/**
 * One graph level at a time.
 *
 * The canvas never nests sub-flows. It shows exactly one `Graph`, and the
 * breadcrumb path says which one. A `repeat` container hands over its stored
 * subgraph, which stays editable. A composite has no stored subgraph, so the
 * level is produced by the catalog's own expansion and is read-only.
 */

import type { Doc, Graph, NodeDef } from "@tensorcad/core";
import { catalogOf, isComposite, isContainer } from "@tensorcad/core";
import type { Derived } from "./derive.js";

export type LevelKind = "root" | "container" | "composite";

export interface Crumb {
  label: string;
  sub?: string;
  segments: string[];
}

export interface Level {
  segments: string[];
  /** `""` at the root, otherwise the `/`-joined path of the owning node. */
  prefix: string;
  graph: Graph;
  editable: boolean;
  kind: LevelKind;
  /** The node this level belongs to, when it is not the root. */
  owner: NodeDef | null;
  crumbs: Crumb[];
  /** Why the requested path could not be opened, if it could not. */
  error: string | null;
}

function crumb(node: NodeDef, segments: string[]): Crumb {
  return { label: node.label ?? node.id, sub: node.type, segments };
}

export function resolveLevel(doc: Doc, path: string[], derived: Derived): Level {
  const crumbs: Crumb[] = [{ label: doc.meta.name || "design", sub: "model", segments: [] }];
  let graph: Graph = doc.graph;
  let editable = true;
  let kind: LevelKind = "root";
  let owner: NodeDef | null = null;
  const walked: string[] = [];

  for (const seg of path) {
    const node: NodeDef | undefined = graph.nodes.find((n) => n.id === seg);
    if (!node) {
      return {
        segments: walked,
        prefix: walked.join("/"),
        graph,
        editable,
        kind,
        owner,
        crumbs,
        error: `No node "${seg}" in ${walked.join("/") || "the top level"}`,
      };
    }
    walked.push(seg);
    const def = catalogOf(doc)[node.type];
    if (def && isContainer(def)) {
      if (!node.graph) {
        return {
          segments: walked.slice(0, -1),
          prefix: walked.slice(0, -1).join("/"),
          graph,
          editable,
          kind,
          owner,
          crumbs,
          error: `Container "${seg}" has no subgraph`,
        };
      }
      graph = node.graph;
      kind = "container";
      owner = node;
      crumbs.push(crumb(node, [...walked]));
      continue;
    }
    if (def && isComposite(def)) {
      const resolved = derived.infer.resolved.get(walked.join("/"));
      if (!resolved) {
        return {
          segments: walked.slice(0, -1),
          prefix: walked.slice(0, -1).join("/"),
          graph,
          editable,
          kind,
          owner,
          crumbs,
          error: `"${seg}" has not been analysed, so its interior cannot be shown`,
        };
      }
      try {
        graph = def.expand(resolved.rawFull, resolved);
      } catch (e) {
        return {
          segments: walked.slice(0, -1),
          prefix: walked.slice(0, -1).join("/"),
          graph,
          editable,
          kind,
          owner,
          crumbs,
          error: `Expanding "${seg}" failed: ${(e as Error).message}`,
        };
      }
      editable = false;
      kind = "composite";
      owner = node;
      crumbs.push(crumb(node, [...walked]));
      continue;
    }
    return {
      segments: walked.slice(0, -1),
      prefix: walked.slice(0, -1).join("/"),
      graph,
      editable,
      kind,
      owner,
      crumbs,
      error: `"${seg}" is a ${node.type} block and has no interior`,
    };
  }

  return {
    segments: walked,
    prefix: walked.join("/"),
    graph,
    editable,
    kind,
    owner,
    crumbs,
    error: null,
  };
}
