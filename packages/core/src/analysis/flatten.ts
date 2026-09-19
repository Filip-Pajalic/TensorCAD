/**
 * Flatten a design to the primitive nodes that carry the formulas.
 *
 * Composites are expanded; `repeat` containers contribute a multiplier instead
 * of being unrolled, so a 126-layer model still flattens to a few dozen nodes.
 */

import type { Doc, Graph, Resolved, SymbolTable } from "../ir/types.js";
import { joinPath } from "../ir/types.js";
import { catalogOf, isComposite, isContainer, isPrimitive } from "../catalog/index.js";
import { resolveNodeParams } from "../catalog/resolve.js";
import type { BlockDef, PrimitiveDef } from "../catalog/types.js";

export interface FlatNode {
  /** Full path, e.g. `"layers/block/attn/q_proj"`. */
  path: string;
  type: string;
  category: string;
  def: PrimitiveDef;
  resolved: Resolved;
  /** Product of the enclosing container counts: how many copies exist. */
  multiplier: number;
  /** Product of the enclosing active counts: how many a token passes through. */
  activeMultiplier: number;
  /** Path of the nearest enclosing repeat container, if any. */
  container: string | null;
}

export interface FlatBlock {
  path: string;
  type: string;
  kind: "primitive" | "composite" | "container";
  category: string;
  def: BlockDef;
  resolved: Resolved;
  multiplier: number;
  activeMultiplier: number;
}

export interface FlatResult {
  /** Primitive nodes only: these carry the formulas. */
  nodes: FlatNode[];
  /** Every node at every level, including composites and containers. */
  blocks: FlatBlock[];
  errors: string[];
  /** Layer and expert counts contributed by containers, for reporting. */
  repeats: { path: string; type: string; count: number; active: number }[];
}

export function flatten(doc: Doc, symbols: SymbolTable): FlatResult {
  // A design may define blocks of its own, so the catalog is the document's.
  const cat = catalogOf(doc);
  const out: FlatResult = { nodes: [], blocks: [], errors: [], repeats: [] };
  const seen = new Set<string>();

  const walk = (
    graph: Graph,
    prefix: string,
    multiplier: number,
    activeMultiplier: number,
    container: string | null,
    depth: number,
  ): void => {
    if (depth > 32) {
      out.errors.push(`Graph nesting deeper than 32 levels at "${prefix}"; is a composite expanding into itself?`);
      return;
    }
    for (const node of graph.nodes) {
      const path = joinPath(prefix, node.id);
      if (seen.has(path)) {
        out.errors.push(`Duplicate node id "${node.id}" in "${prefix || "<root>"}"`);
        continue;
      }
      seen.add(path);

      const def = cat[node.type];
      if (!def) {
        out.errors.push(`Unknown block type "${node.type}" at "${path}"`);
        continue;
      }
      const resolved = resolveNodeParams(def, node.params, symbols);
      for (const e of resolved.errors) out.errors.push(`${path}: ${e}`);
      out.blocks.push({
        path,
        type: def.type,
        kind: def.kind,
        category: def.category,
        def,
        resolved,
        multiplier,
        activeMultiplier,
      });

      if (isPrimitive(def)) {
        out.nodes.push({
          path,
          type: def.type,
          category: def.category,
          def,
          resolved,
          multiplier,
          activeMultiplier,
          container,
        });
        continue;
      }

      if (isComposite(def)) {
        let inner: Graph;
        try {
          inner = def.expand(resolved.rawFull, resolved);
        } catch (e) {
          out.errors.push(`${path}: expansion failed: ${(e as Error).message}`);
          continue;
        }
        walk(inner, path, multiplier, activeMultiplier, container, depth + 1);
        continue;
      }

      if (isContainer(def)) {
        let counts;
        try {
          counts = def.multipliers(resolved);
        } catch (e) {
          out.errors.push(`${path}: ${(e as Error).message}`);
          continue;
        }
        if (!Number.isFinite(counts.total) || counts.total < 0 || !Number.isFinite(counts.active) || counts.active < 0) {
          out.errors.push(`${path}: container counts must be non-negative numbers`);
          continue;
        }
        out.repeats.push({ path, type: def.type, count: counts.total, active: counts.active });
        if (node.graph) {
          walk(
            node.graph,
            path,
            multiplier * counts.total,
            activeMultiplier * counts.active,
            // Only a layer stack owns its activations for recomputation.
            def.type === "repeat" ? path : container,
            depth + 1,
          );
        } else {
          out.errors.push(`${path}: container "${def.type}" has no subgraph`);
        }
        continue;
      }
    }
  };

  walk(doc.graph, "", 1, 1, null, 0);
  return out;
}
