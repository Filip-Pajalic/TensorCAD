import type { Doc } from "../ir/types.js";
import type { BlockDef, CompositeDef, ContainerDef, PrimitiveDef } from "./types.js";
import { PRIMITIVES } from "./primitives.js";
import { COMPOSITES, CONTAINERS } from "./composites.js";
import { compileUserBlock, type UserBlockDef } from "./user.js";

const ALL: BlockDef[] = [...PRIMITIVES, ...COMPOSITES, ...CONTAINERS];

/** The built-in blocks. A design may add its own; see `catalogOf`. */
export const CATALOG: Record<string, BlockDef> = Object.fromEntries(ALL.map((d) => [d.type, d]));

export type Catalog = Record<string, BlockDef>;

const cache = new WeakMap<Doc, Catalog>();

/**
 * The catalog a document sees: the built-ins, plus whatever it defines itself.
 *
 * Cached on the document object, which is immutable per edit, so the whole
 * analysis pipeline can call this freely. A definition that fails to compile is
 * dropped rather than thrown: the design rules report it, and half a catalog is
 * more useful than an exception.
 */
export function catalogOf(doc: Doc | undefined | null): Catalog {
  if (!doc) return CATALOG;
  const hit = cache.get(doc);
  if (hit) return hit;

  const defs = doc.defs as Record<string, UserBlockDef> | undefined;
  let out: Catalog = CATALOG;
  if (defs && Object.keys(defs).length > 0) {
    out = { ...CATALOG };
    for (const [type, def] of Object.entries(defs)) {
      if (CATALOG[type]) continue; // never shadow a built-in
      try {
        out[type] = compileUserBlock({ ...def, type });
      } catch {
        // Reported by the design rules; not a reason to lose the rest.
      }
    }
  }
  cache.set(doc, out);
  return out;
}

/** Every block a document can place, grouped for the palette. */
export function catalogEntries(doc?: Doc): BlockDef[] {
  return Object.values(catalogOf(doc));
}

/** True when this type is defined by the document rather than built in. */
export function isUserBlock(doc: Doc | undefined, type: string): boolean {
  const defs = doc?.defs as Record<string, unknown> | undefined;
  return Boolean(defs && type in defs && !CATALOG[type]);
}

export function getBlock(type: string): BlockDef | undefined {
  return CATALOG[type];
}

export function requireBlock(type: string): BlockDef {
  const d = CATALOG[type];
  if (!d) throw new Error(`Unknown block type "${type}"`);
  return d;
}

export function isPrimitive(d: BlockDef): d is PrimitiveDef {
  return d.kind === "primitive";
}
export function isComposite(d: BlockDef): d is CompositeDef {
  return d.kind === "composite";
}
export function isContainer(d: BlockDef): d is ContainerDef {
  return d.kind === "container";
}

/** Catalog entries grouped by category, for the editor palette. */
export function catalogByCategory(doc?: Doc): Record<string, BlockDef[]> {
  const out: Record<string, BlockDef[]> = {};
  for (const d of catalogEntries(doc)) {
    (out[d.category] ??= []).push(d);
  }
  return out;
}

export { PRIMITIVES, COMPOSITES, CONTAINERS };
export * from "./types.js";
export { resolveNodeParams, portsOf } from "./resolve.js";
export {
  compileUserBlock,
  validateUserBlock,
  BOUNDARY_IN,
  BOUNDARY_OUT,
  type UserBlockDef,
} from "./user.js";
