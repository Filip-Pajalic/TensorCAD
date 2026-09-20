/**
 * One graph level at a time.
 *
 * The canvas never nests sub-flows. It shows exactly one `Graph`, and the
 * breadcrumb path says which one. A `repeat` container hands over its stored
 * subgraph, which stays editable. A composite has no stored subgraph, so the
 * level is produced by the catalog's own expansion and is read-only.
 *
 * A block the *document* defines is the third case, and the only editable one
 * that is not a container: its template is stored, so a path beginning `@def`
 * opens it directly rather than through any instance. Which instance would have
 * been the wrong question — an edit to a definition changes all of them.
 *
 * What is drawn there is the preview from `definition.ts`, not the stored
 * template: `$d_model` is not a symbol and nothing about it would resolve. What
 * is *edited* is the template, because `graphAtPath` follows the same prefix.
 */

import type { Derived } from "./derive.js";
import type { Doc, Graph, NodeDef } from "@tensor-cad/engine";
import { catalogOf, isComposite, isContainer } from "../engine.js";
import { DEF_PREFIX, previewDoc } from "./definition.js";

export type LevelKind = "root" | "container" | "composite" | "definition";

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

  let rest = path;
  if (path[0] === DEF_PREFIX) {
    const type = path[1] ?? "";
    const preview = previewDoc(doc, type);
    if (!preview) {
      return {
        segments: [], prefix: "", graph, editable, kind, owner, crumbs,
        error: `This design defines no block called "${type}".`,
      };
    }
    graph = preview.graph;
    kind = "definition";
    walked.push(DEF_PREFIX, type);
    crumbs.push({ label: type, sub: "definition", segments: [DEF_PREFIX, type] });
    rest = path.slice(2);
  }

  for (const seg of rest) {
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
      // The analysis already expanded this one; drawing it means reading what
      // it saw, not unfolding the composite a second time here.
      const expansion = derived.infer.expansions.get(walked.join("/"));
      if (!expansion) {
        return {
          segments: walked.slice(0, -1),
          prefix: walked.slice(0, -1).join("/"),
          graph,
          editable,
          kind,
          owner,
          crumbs,
          error: `"${seg}" did not expand, so its interior cannot be shown`,
        };
      }
      graph = expansion;
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
