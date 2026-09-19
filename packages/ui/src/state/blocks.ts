/**
 * The design's own block library.
 *
 * A design carries its composites in `doc.defs`, and these are the three things
 * you do with them: make one out of a level you have already drawn, bring a
 * library in, and send one out. That is deliberately the same shape as KiCad's
 * symbol libraries — a project has its own, and they travel as a file.
 */

import {
  CATALOG,
  BOUNDARY_IN,
  BOUNDARY_OUT,
  validateUserBlock,
  type Doc,
  type Graph,
  type ParamSpec,
  type UserBlockDef,
} from "@tensorcad/core";

/** A block library on disk: just the definitions, with a version to check. */
export interface BlockLibrary {
  kind: "tensorcad-blocks";
  version: 1;
  blocks: Record<string, UserBlockDef>;
}

const BUILT_IN = new Set(Object.keys(CATALOG));

export function defsOf(doc: Doc): Record<string, UserBlockDef> {
  return (doc.defs as Record<string, UserBlockDef> | undefined) ?? {};
}

/** A type name not already taken, derived from what the user typed. */
export function freeTypeName(doc: Doc, wanted: string): string {
  const base =
    wanted
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9_]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .replace(/^([0-9])/, "b$1") || "block";
  const taken = new Set([...BUILT_IN, ...Object.keys(defsOf(doc))]);
  if (!taken.has(base)) return base;
  let n = 2;
  while (taken.has(`${base}_${n}`)) n++;
  return `${base}_${n}`;
}

/**
 * Turn a graph into a block definition.
 *
 * The boundary nodes become the declared ports and are dropped from the
 * template, because `compileUserBlock` regenerates them — a definition that
 * declared one set of ports and wired another would be a trap.
 *
 * Symbols the graph uses become the block's parameters, defaulting to what they
 * currently resolve to, and every reference to them is rewritten to `$name`.
 * That is what turns a drawing into something reusable: a block extracted from
 * a 4,096-wide design should work at 2,048 without being rebuilt.
 */
export function blockFromGraph(
  graph: Graph,
  type: string,
  symbols: Record<string, number>,
  summary: string,
): { def: UserBlockDef; errors: string[] } {
  const boundaryIn = graph.nodes.find((n) => n.type === "boundary_in");
  const boundaryOut = graph.nodes.find((n) => n.type === "boundary_out");

  const ports = {
    in: (boundaryIn?.params?.ports as Record<string, string> | undefined) ?? { x: "... D" },
    out: (boundaryOut?.params?.ports as Record<string, string> | undefined) ?? { y: "... D" },
  };

  // Which symbols this graph actually mentions, so a block does not acquire a
  // parameter for every symbol in the document.
  const used = new Set<string>();
  const mention = (value: unknown): void => {
    if (typeof value !== "string") return;
    for (const m of value.matchAll(/[A-Za-z_][A-Za-z0-9_]*/g)) {
      if (m[0] in symbols) used.add(m[0]);
    }
  };
  const scan = (nodes: Graph["nodes"]): void => {
    for (const node of nodes) {
      for (const v of Object.values(node.params ?? {})) mention(v);
      if (node.graph) scan(node.graph.nodes);
    }
  };
  scan(graph.nodes);
  for (const pattern of [...Object.values(ports.in), ...Object.values(ports.out)]) mention(pattern);

  const params: Record<string, ParamSpec> = {};
  for (const name of [...used].sort()) {
    params[name] = { type: "int", default: symbols[name], min: 1, doc: `Was ${symbols[name]}` };
  }

  /** Rewrite a symbol reference into a parameter reference. */
  const parameterise = (value: unknown): unknown => {
    if (typeof value !== "string") return value;
    return value.replace(/[A-Za-z_][A-Za-z0-9_]*/g, (name) =>
      used.has(name) ? `$${name}` : name,
    );
  };
  const rewrite = (nodes: Graph["nodes"]): Graph["nodes"] =>
    nodes.map((node) => {
      const out = { ...node };
      if (node.params) {
        out.params = Object.fromEntries(
          Object.entries(node.params).map(([k, v]) => [k, parameterise(v) as never]),
        );
      }
      if (node.graph) out.graph = { nodes: rewrite(node.graph.nodes), edges: [...node.graph.edges] };
      return out;
    });

  const def: UserBlockDef = {
    type,
    category: "block",
    params,
    // Port patterns name the block's parameters directly, the way a built-in
    // block's do; only node parameters take the `$` form.
    ports,
    graph: {
      nodes: rewrite(graph.nodes.filter((n) => n.id !== BOUNDARY_IN && n.id !== BOUNDARY_OUT)),
      edges: graph.edges.map(([from, to]) => [from, to] as [string, string]),
    },
    docs: { summary: summary || `Defined in this design from ${graph.nodes.length} blocks.` },
  };

  return { def, errors: validateUserBlock(def, BUILT_IN) };
}

export function withBlock(doc: Doc, def: UserBlockDef): Doc {
  return { ...doc, defs: { ...defsOf(doc), [def.type]: def } };
}

export function withoutBlock(doc: Doc, type: string): Doc {
  const next = { ...defsOf(doc) };
  delete next[type];
  return { ...doc, defs: next };
}

export function toLibrary(doc: Doc): BlockLibrary {
  return { kind: "tensorcad-blocks", version: 1, blocks: defsOf(doc) };
}

/** Read a library file, keeping whatever is already defined unless it clashes. */
export function mergeLibrary(doc: Doc, text: string): { doc: Doc; added: string[]; errors: string[] } {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    return { doc, added: [], errors: [`Not JSON: ${(e as Error).message}`] };
  }
  const lib = parsed as Partial<BlockLibrary>;
  if (lib.kind !== "tensorcad-blocks" || typeof lib.blocks !== "object" || lib.blocks === null) {
    return { doc, added: [], errors: ["Not a block library."] };
  }

  const defs = { ...defsOf(doc) };
  const added: string[] = [];
  const errors: string[] = [];
  for (const [type, raw] of Object.entries(lib.blocks)) {
    const def = { ...(raw as UserBlockDef), type };
    const problems = validateUserBlock(def, BUILT_IN);
    if (problems.length > 0) {
      errors.push(`${type}: ${problems[0]}`);
      continue;
    }
    defs[type] = def;
    added.push(type);
  }
  return { doc: { ...doc, defs }, added, errors };
}
