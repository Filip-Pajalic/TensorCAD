/**
 * Editing a block the document defines for itself.
 *
 * A definition's template is written in terms of the block's parameters: a
 * `linear` inside it has `in_features: "$D"`, not 4096, because that is what
 * makes it reusable at another width. Nothing can draw that directly — `$D` is
 * not a symbol, so no shape resolves and no number can be counted.
 *
 * So the canvas is given a *preview*: the same graph with every `$name`
 * rewritten to `name`, over a symbol table made of the block's own parameters at
 * their declared defaults. That is not a trick — it is exactly how the built-in
 * composites are written, where `gqa_attention`'s expansion names `d_model`
 * directly. Inside a definition, a bare identifier that names a declared
 * parameter *is* that parameter.
 *
 * Which answers the question the editor would otherwise have to guess at. A
 * `linear` dragged in with `in_features` of 4096 means 4096, and stays 4096 at
 * every width. One with `in_features` of `D` means the parameter, and is stored
 * back as `$D`. What you see is what is kept, and the rewrite is total in both
 * directions.
 */

import type {
  Doc,
  Graph,
  NodeDef,
  ParamSpec,
  ParamValue,
  SymbolDef,
  UserBlockDef,
} from "@tensor-cad/engine";
import { BOUNDARY_IN, BOUNDARY_OUT } from "@tensor-cad/engine";
import { defsOf } from "./blocks.js";

/** The first segment of a path into a definition rather than into the design. */
export const DEF_PREFIX = "@def";

/** True for a path that addresses a definition's template. */
export function isDefPath(path: readonly string[]): boolean {
  return path[0] === DEF_PREFIX;
}

/** Which definition a path is in, or null when it is not in one. */
export function defOfPath(path: readonly string[]): string | null {
  return isDefPath(path) ? (path[1] ?? null) : null;
}

/** Rewrite one direction: `$name` to `name`, for the parameters declared. */
function bindValue(value: unknown, params: ReadonlySet<string>): unknown {
  if (typeof value !== "string") return value;
  return value.replace(/\$([A-Za-z_][A-Za-z0-9_]*)/g, (whole, name: string) =>
    params.has(name) ? name : whole,
  );
}

/** And the other: `name` back to `$name`, for the parameters declared. */
export function storeValue(value: unknown, params: ReadonlySet<string>): unknown {
  if (typeof value !== "string") return value;
  // Identifiers only, so `4096` stays a number and a `$D` somebody typed is not
  // turned into `$$D`.
  return value.replace(/\$?[A-Za-z_][A-Za-z0-9_]*/g, (token) => {
    if (token.startsWith("$")) return token;
    return params.has(token) ? `$${token}` : token;
  });
}

function mapParams(nodes: NodeDef[], fn: (v: unknown) => unknown): NodeDef[] {
  return nodes.map((node) => {
    const out: NodeDef = { ...node };
    if (node.params) {
      out.params = Object.fromEntries(
        Object.entries(node.params).map(([k, v]) => [k, fn(v) as ParamValue]),
      );
    }
    if (node.graph) out.graph = { nodes: mapParams(node.graph.nodes, fn), edges: node.graph.edges };
    return out;
  });
}

/** The parameter names a definition declares. */
export function paramsOf(def: UserBlockDef): Set<string> {
  return new Set(Object.keys(def.params ?? {}));
}

/**
 * Store a value the way the template holds it.
 *
 * This is what every edit inside a definition goes through, which is why it is
 * one function rather than a rule applied in several places.
 */
export function storeInDefinition(doc: Doc, type: string, value: unknown): unknown {
  const def = defsOf(doc)[type];
  return def ? storeValue(value, paramsOf(def)) : value;
}

/** A default for a parameter, for the preview's symbol table. */
function defaultOf(spec: ParamSpec): number {
  const raw = spec.default;
  if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  // A width nobody gave a default for still has to be a number, or nothing in
  // the preview resolves. One head's worth is a shape, which is what matters.
  return spec.min ?? 1;
}

/**
 * The design a definition's template is drawn as.
 *
 * Its symbols are the block's parameters, so a shape written `... d_model`
 * resolves the way it would in any design, and its graph carries the boundary
 * nodes the expansion generates — declared rather than written, so a definition
 * cannot show one interface here and present another when used.
 */
export function previewDoc(doc: Doc, type: string): Doc | null {
  const def = defsOf(doc)[type];
  if (!def?.graph) return null;
  const params = paramsOf(def);

  const symbols: Record<string, SymbolDef> = {
    B: { kind: "runtime", default: 1 } as unknown as SymbolDef,
    T: { kind: "runtime", default: 128 } as unknown as SymbolDef,
  };
  const order = ["B", "T"];
  for (const [name, spec] of Object.entries(def.params ?? {})) {
    symbols[name] = { kind: "design", value: defaultOf(spec), doc: spec.doc ?? "" } as unknown as SymbolDef;
    order.push(name);
  }

  const inner = mapParams(def.graph.nodes, (v) => bindValue(v, params));
  const nodes: NodeDef[] = [
    { id: BOUNDARY_IN, type: "boundary_in", params: { ports: def.ports.in } } as NodeDef,
    ...inner,
    { id: BOUNDARY_OUT, type: "boundary_out", params: { ports: def.ports.out } } as NodeDef,
  ];

  return {
    version: 1,
    meta: { name: type, notes: def.docs?.summary ?? "" },
    symbols,
    symbolOrder: order,
    graph: { nodes, edges: [...def.graph.edges] } as Graph,
    // Its own defs, so a definition that uses another one still resolves.
    defs: Object.fromEntries(Object.entries(defsOf(doc)).filter(([name]) => name !== type)),
    ui: doc.ui,
  } as unknown as Doc;
}
