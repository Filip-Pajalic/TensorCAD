/**
 * Blocks defined by a design rather than by this package.
 *
 * The built-in catalog is TypeScript: a primitive carries formulas, so it has
 * to be. A *composite* does not — it is parameters, ports and a subgraph, and
 * the only reason `gqa_attention` is code is that its expansion interpolates
 * parameters into the graph it builds. That interpolation is a string
 * substitution, so it can be written down instead.
 *
 * So a design can carry its own block library, the way a KiCad project carries
 * its own symbols. A user block is a template subgraph whose node parameters
 * may refer to the block's own parameters as `$name`; expanding it substitutes
 * the values the instance was given and hands the result to the same shape
 * inference, parameter counting and code generation as everything else.
 *
 * What this does not give you is a new *primitive*: anything that needs its own
 * parameter-count or FLOP formula still belongs in `primitives.ts`. In practice
 * a new architecture is almost always a new arrangement of existing primitives,
 * which is exactly what this covers.
 */

import type { Graph, NodeDef, ParamValue } from "../ir/types.js";
import type { BlockDocs, CompositeDef, ParamSpec, Ports } from "./types.js";
import { ex } from "./types.js";

export interface UserBlockDef {
  /** Unique type name. Must not collide with a built-in. */
  type: string;
  category: string;
  params: Record<string, ParamSpec>;
  /**
   * Declared ports, as shape patterns over this block's own parameter names —
   * the same notation the built-in blocks use, e.g. `"... d_model"`.
   */
  ports: Ports;
  /**
   * The subgraph, without boundary nodes: those are generated from `ports` so a
   * definition cannot declare one set and wire another.
   */
  graph: Graph;
  docs: BlockDocs;
}

/** `$name`, not bare `name`, so a substitution can never be a symbol by accident. */
const REF = /\$([A-Za-z_][A-Za-z0-9_]*)/g;

/**
 * Substitute the instance's parameters into one template value.
 *
 * Only strings carry references; a number or a flag in the template is already
 * the value it will have. `ex` decides the bracketing, so `$d_model` given the
 * expression `H*dh` lands as `(H*dh)` and stays one term inside a larger one.
 */
function fill(value: ParamValue | undefined, raw: Record<string, ParamValue>): ParamValue | undefined {
  if (typeof value !== "string") return value;
  return value.replace(REF, (_, name: string) => ex(raw[name]));
}

function fillNode(node: NodeDef, raw: Record<string, ParamValue>): NodeDef {
  const out: NodeDef = { ...node };
  if (node.params) {
    const params: Record<string, ParamValue> = {};
    for (const [key, value] of Object.entries(node.params)) {
      const filled = fill(value, raw);
      if (filled !== undefined) params[key] = filled;
    }
    out.params = params;
  }
  // A template may nest a container with its own stored subgraph; the same
  // substitution has to reach into it.
  if (node.graph) {
    out.graph = {
      nodes: node.graph.nodes.map((n) => fillNode(n, raw)),
      edges: [...node.graph.edges],
    };
  }
  return out;
}

/** Problems that would make a definition unusable, in plain words. */
export function validateUserBlock(def: UserBlockDef, taken: ReadonlySet<string>): string[] {
  const errors: string[] = [];
  if (!/^[a-z][a-z0-9_]*$/.test(def.type)) {
    errors.push(`"${def.type}" is not a usable type name: lower case, digits and underscores.`);
  }
  if (taken.has(def.type)) {
    errors.push(`"${def.type}" is already a built-in block.`);
  }
  const ins = Object.keys(def.ports.in);
  const outs = Object.keys(def.ports.out);
  if (ins.length === 0) errors.push("A block needs at least one input port.");
  if (outs.length === 0) errors.push("A block needs at least one output port.");
  if (def.graph.nodes.length === 0) errors.push("A block needs at least one node.");

  const ids = new Set(def.graph.nodes.map((n) => n.id));
  if (ids.has(BOUNDARY_IN) || ids.has(BOUNDARY_OUT)) {
    errors.push(`"${BOUNDARY_IN}" and "${BOUNDARY_OUT}" are generated; do not define them.`);
  }

  // Every `$ref` has to name a parameter, or the expansion produces a "0" that
  // silently counts as a valid expression.
  const known = new Set(Object.keys(def.params));
  const seen = new Set<string>();
  const scan = (nodes: NodeDef[]): void => {
    for (const node of nodes) {
      for (const value of Object.values(node.params ?? {})) {
        if (typeof value !== "string") continue;
        for (const m of value.matchAll(REF)) if (!known.has(m[1])) seen.add(m[1]);
      }
      if (node.graph) scan(node.graph.nodes);
    }
  };
  scan(def.graph.nodes);
  for (const name of seen) {
    errors.push(`$${name} is not a parameter of this block.`);
  }
  return errors;
}

export const BOUNDARY_IN = "_in";
export const BOUNDARY_OUT = "_out";

/** Turn a stored definition into a composite the rest of the core can use. */
export function compileUserBlock(def: UserBlockDef): CompositeDef {
  return {
    kind: "composite",
    type: def.type,
    category: def.category,
    params: def.params,
    ports: def.ports,
    expand: (raw): Graph => ({
      nodes: [
        { id: BOUNDARY_IN, type: "boundary_in", params: { ports: { ...def.ports.in } } },
        ...def.graph.nodes.map((n) => fillNode(n, raw)),
        { id: BOUNDARY_OUT, type: "boundary_out", params: { ports: { ...def.ports.out } } },
      ],
      edges: [...def.graph.edges],
    }),
    docs: def.docs,
  };
}
