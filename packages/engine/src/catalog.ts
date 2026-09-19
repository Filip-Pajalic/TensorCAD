/**
 * Asking the catalog about a block.
 *
 * The engine ships the blocks it knows; a design can define more. Every client
 * needs the same two things from that — one block by name, and all of them
 * grouped — and needs them to agree with what the analysis counted. So the
 * folding happens here rather than three times over.
 */

import type { CatalogEntry, Doc, ParamSpec, PortSpec, UserBlockDef } from "./types.js";

/**
 * A block a design defined for itself, described the way a built-in is.
 *
 * The engine only ships the built-in catalog, because a user block lives in the
 * document. Drawing one needs the same fields, so this fills them in from the
 * definition rather than teaching every panel about a second kind of block.
 */
function fromUserBlock(type: string, def: UserBlockDef): CatalogEntry {
  const params: Record<string, ParamSpec> = { ...(def.params ?? {}) };
  const shapes = (side: Record<string, string | PortSpec> | undefined): Record<string, string> => {
    const out: Record<string, string> = {};
    for (const [name, port] of Object.entries(side ?? {})) {
      out[name] = typeof port === "string" ? port : port.shape;
    }
    return out;
  };
  return {
    type,
    kind: "composite",
    category: def.category ?? "custom",
    docs: {
      summary: def.docs?.summary ?? "A block this design defines for itself.",
      formula: def.docs?.formula,
      refs: def.docs?.refs,
    },
    params,
    paramOrder: Object.keys(params),
    ports: { in: shapes(def.ports?.in), out: shapes(def.ports?.out) },
  };
}

/** The blocks an engine knows, and the ones a document adds to them. */
export class Catalog {
  private readonly builtIn: Map<string, CatalogEntry>;

  constructor(entries: readonly CatalogEntry[]) {
    this.builtIn = new Map(entries.map((entry) => [entry.type, entry]));
  }

  /** Every built-in block, in the order the engine declares them. */
  get builtInEntries(): CatalogEntry[] {
    return [...this.builtIn.values()];
  }

  /** True when the engine itself knows this type. */
  isBuiltIn(type: string): boolean {
    return this.builtIn.has(type);
  }

  /**
   * One block's definition.
   *
   * A design never shadows a built-in, which is what the engine's own resolver
   * does; agreeing here keeps a palette showing what the analysis counted.
   */
  get(type: string, doc?: Doc): CatalogEntry | undefined {
    if (this.builtIn.has(type)) return this.builtIn.get(type);
    const own = doc?.defs?.[type];
    return own ? fromUserBlock(type, own) : undefined;
  }

  /** Every block a document can use, built-ins first. */
  entries(doc?: Doc): CatalogEntry[] {
    const out = this.builtInEntries;
    for (const [type, def] of Object.entries(doc?.defs ?? {})) {
      if (!this.builtIn.has(type)) out.push(fromUserBlock(type, def));
    }
    return out;
  }

  byCategory(doc?: Doc): Record<string, CatalogEntry[]> {
    const out: Record<string, CatalogEntry[]> = {};
    for (const entry of this.entries(doc)) {
      (out[entry.category] ??= []).push(entry);
    }
    return out;
  }

  /** True when this type came from the design rather than from the engine. */
  isUserBlock(doc: Doc | undefined, type: string): boolean {
    return Boolean(doc?.defs && type in doc.defs && !this.builtIn.has(type));
  }

  /** A block's parameter spec, by name. */
  param(type: string, name: string, doc?: Doc): ParamSpec | undefined {
    return this.get(type, doc)?.params[name];
  }
}

export function isPrimitive(def: CatalogEntry | undefined): boolean {
  return def?.kind === "primitive";
}
export function isComposite(def: CatalogEntry | undefined): boolean {
  return def?.kind === "composite";
}
export function isContainer(def: CatalogEntry | undefined): boolean {
  return def?.kind === "container";
}
