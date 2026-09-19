/**
 * The design document: our own JSON IR.
 *
 * The editor's node/edge state is a *view* of this document, never the source
 * of truth. Everything downstream (validation, analysis, codegen, MCP) reads
 * this type.
 */

import type { Sym } from "../shapes/symexpr.js";

export const DOC_VERSION = 1 as const;

/** Reserved symbols that stay indeterminate through analysis. */
export const RUNTIME_SYMBOLS = ["B", "T"] as const;

export type ParamValue = number | string | boolean | null | ParamObject | ParamValue[];
export interface ParamObject {
  [key: string]: ParamValue;
}

/**
 * A symbol is either a literal number, an expression over earlier symbols, or a
 * runtime dimension that stays symbolic (batch, sequence length).
 */
export type SymbolDef =
  | number
  | string
  | { kind: "runtime"; default: number; doc?: string }
  | { kind: "design"; value: number | string; doc?: string };

export interface NodeDef {
  id: string;
  type: string;
  params?: Record<string, ParamValue>;
  /** Subgraph for container nodes (`repeat`). */
  graph?: Graph;
  /** Named subgraph variants for hybrid repeat patterns. */
  variants?: Record<string, Graph>;
  /** Free-text label shown on the canvas instead of the id. */
  label?: string;
}

/** `"nodeId:portName"` on both ends. */
export type Edge = [from: string, to: string];

export interface Graph {
  nodes: NodeDef[];
  edges: Edge[];
}

export interface DocMeta {
  name: string;
  family?: string;
  notes?: string;
  /** Published reference numbers, used by the regression suite. */
  published?: {
    params?: number;
    activeParams?: number;
    kvBytesPerToken?: number;
    source?: string;
    /**
     * Allowed relative difference. Defaults to 0.5%. Set it wider when the
     * published figure is itself a rounded headline number such as "22B active".
     */
    tolerance?: number;
  };
}

export interface UiState {
  positions?: Record<string, [number, number]>;
  collapsed?: string[];
}

export interface Doc {
  version: typeof DOC_VERSION;
  meta: DocMeta;
  symbols: Record<string, SymbolDef>;
  graph: Graph;
  /**
   * Blocks this design defines for itself, keyed by type name.
   *
   * The built-in catalog is fixed, but a design can carry its own composites —
   * a new attention variant, a different block arrangement — the way a KiCad
   * project carries its own symbol library. They resolve exactly like built-in
   * composites, so shape checking, parameter counting and code generation need
   * to know nothing about where a block came from.
   *
   * Typed loosely here because the definition lives in the catalog, which sits
   * above the IR.
   */
  defs?: Record<string, unknown>;
  ui?: UiState;
}

// ---------------------------------------------------------------------------
// Resolved values
// ---------------------------------------------------------------------------

/** Numeric value of every symbol, including runtime defaults. */
export type SymbolValues = Readonly<Record<string, number>>;

export interface SymbolTable {
  /** Evaluation order actually used (dependency order). */
  order: string[];
  /** Numeric value of every symbol (runtime symbols use their default). */
  values: Record<string, number>;
  /** Numeric value of concrete design symbols only. Runtime symbols excluded. */
  designValues: Record<string, number>;
  /** Symbols that stay indeterminate. */
  runtime: Set<string>;
  /** Documentation string per symbol. */
  docs: Record<string, string>;
  errors: string[];
}

/** A node's parameters after expression evaluation. */
export interface Resolved {
  type: string;
  /** Concrete values: numbers, booleans, strings, objects. */
  p: Record<string, any>;
  /** Symbolic form of every numeric parameter, for shape display. */
  s: Record<string, Sym>;
  /** Raw, unevaluated parameter values as stored in the document. */
  raw: Record<string, ParamValue>;
  /**
   * Raw values with catalog defaults filled in. Composite expansion uses this
   * so expressions such as `"D"` survive into the inner graph and inner shapes
   * keep showing symbol names.
   */
  rawFull: Record<string, ParamValue>;
  errors: string[];
}

/** Dotted path identifying a node inside nested graphs: `"layers/block/attn"`. */
export type NodePath = string;

export function splitEndpoint(endpoint: string): { node: string; port: string } {
  const i = endpoint.lastIndexOf(":");
  if (i < 0) throw new Error(`Malformed endpoint "${endpoint}", expected "node:port"`);
  return { node: endpoint.slice(0, i), port: endpoint.slice(i + 1) };
}

export function joinPath(prefix: string, id: string): string {
  return prefix ? `${prefix}/${id}` : id;
}
