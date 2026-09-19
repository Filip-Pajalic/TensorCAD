/**
 * The engine, as the editor sees it.
 *
 * The analysis is Go compiled to WebAssembly, loaded once before the first
 * frame. Everything below the load is synchronous, so the editor keeps the
 * shape it had when the analysis was TypeScript running in the same window: a
 * keystroke changes the document, the document is re-derived, the panels read
 * the result.
 *
 * The catalog is fetched once at load and held here. It cannot change while the
 * app is running — a block a *design* defines can, and that is handled by
 * folding `doc.defs` in at each lookup.
 */

import {
  createEngine,
  type CatalogEntry,
  type Doc,
  type Engine,
  type HardwareProfile,
  type ParamSpec,
  type PortSpec,
  type GeneratedCode,
  type Inference,
  type RuleInfo,
  type SymbolTable,
  type TorchOptions,
  type UserBlockDef,
} from "@tensorcad/engine";
import "@tensorcad/engine/wasm_exec";

let loaded: Engine | null = null;
let builtIn = new Map<string, CatalogEntry>();

/**
 * The built-in catalog, by type.
 *
 * Filled in by the load and never replaced, so a module can hold a reference to
 * it at import time. A block a *design* defines is not in here; use `blockDef`
 * for that, which folds `doc.defs` in.
 */
export const CATALOG: Record<string, CatalogEntry> = {};

/** The design library, the hardware profiles and the rules: fixed after load. */
export const PRESET_NAMES: string[] = [];
export const HARDWARE: HardwareProfile[] = [];
export const HARDWARE_BY_ID: Record<string, HardwareProfile> = {};
export const RULES: RuleInfo[] = [];

/**
 * Loads the engine and the things that never change while it runs.
 *
 * Called once, before the first render. The editor has nothing to draw without
 * a catalog, so there is no useful partially-loaded state to design for.
 */
export async function loadEngine(): Promise<void> {
  if (loaded) return;
  const engine = await createEngine();
  builtIn = new Map(engine.catalog().map((entry) => [entry.type, entry]));
  for (const [type, entry] of builtIn) CATALOG[type] = entry;
  PRESET_NAMES.push(...engine.presets());
  HARDWARE.push(...engine.hardware());
  for (const profile of HARDWARE) HARDWARE_BY_ID[profile.id] = profile;
  RULES.push(...engine.rules());
  loaded = engine;
}

/** The engine. Throws if the editor got here before the load finished. */
export function engine(): Engine {
  if (!loaded) throw new Error("The engine is not loaded yet.");
  return loaded;
}

export function isEngineLoaded(): boolean {
  return loaded !== null;
}

export function hardwareById(id: string): HardwareProfile | undefined {
  return HARDWARE_BY_ID[id];
}

// ---------------------------------------------------------------------------
// The calls, as the editor makes them
// ---------------------------------------------------------------------------

/** One preset's document. */
export function getPreset(name: string): Doc {
  return engine().preset(name);
}

/** PyTorch for a design. */
export function generateTorch(doc: Doc, options?: TorchOptions): GeneratedCode {
  return engine().generateTorch(doc, options);
}

/** The symbol table alone, without the rest of the analysis. */
export function resolveSymbols(doc: Doc): SymbolTable {
  return engine().analyze(doc).symbols;
}

/** Shape inference alone, for the wire the pointer is over. */
export function inferShapes(doc: Doc, mode: "flat" | "expanded" = "flat"): Inference {
  return engine().infer(doc, mode);
}

/**
 * What is wrong with a block the design defines for itself.
 *
 * Empty means it compiles. The block editor asks on every keystroke, which is
 * why it is a call and not a finding from the rule engine.
 */
export function validateUserBlock(def: UserBlockDef, name: string): string[] {
  return engine().checkUserBlock(def, name);
}

// ---------------------------------------------------------------------------
// The catalog
// ---------------------------------------------------------------------------

/**
 * A block a design defined for itself, described the way a built-in is.
 *
 * The engine only ships the built-in catalog, because a user block lives in the
 * document. Rendering one needs the same fields, so this fills them in from the
 * definition rather than teaching every panel about a second kind of block.
 */
function fromUserBlock(type: string, def: UserBlockDef): CatalogEntry {
  const params = { ...(def.params ?? {}) };
  const shapes = (side: Record<string, string | PortSpec>): Record<string, string> => {
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
    ports: { in: shapes(def.ports?.in ?? {}), out: shapes(def.ports?.out ?? {}) },
  };
}

/** One block's definition, built-in or the design's own. */
export function blockDef(type: string, doc?: Doc): CatalogEntry | undefined {
  const own = doc?.defs?.[type];
  // A design never shadows a built-in, which is what the engine's own resolver
  // does; agreeing here keeps the palette showing what the analysis counted.
  if (builtIn.has(type)) return builtIn.get(type);
  if (own) return fromUserBlock(type, own);
  return undefined;
}

/** Every block a document can use, built-ins first. */
export function catalogEntries(doc?: Doc): CatalogEntry[] {
  const out = [...builtIn.values()];
  for (const [type, def] of Object.entries(doc?.defs ?? {})) {
    if (!builtIn.has(type)) out.push(fromUserBlock(type, def));
  }
  return out;
}

export function catalogByCategory(doc?: Doc): Record<string, CatalogEntry[]> {
  const out: Record<string, CatalogEntry[]> = {};
  for (const entry of catalogEntries(doc)) {
    (out[entry.category] ??= []).push(entry);
  }
  return out;
}

/** True when this type came from the design rather than from the engine. */
export function isUserBlock(doc: Doc | undefined, type: string): boolean {
  return Boolean(doc?.defs && type in doc.defs && !builtIn.has(type));
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

/** A block's parameter spec, by name, in declaration order. */
export function paramSpec(def: CatalogEntry | undefined, name: string): ParamSpec | undefined {
  return def?.params[name];
}


/**
 * A block's definition, as the editor's own name for it.
 *
 * The engine calls this a catalog entry. The editor has always called it a
 * block definition, and renaming every call site would be churn for nothing.
 */
export type BlockDef = CatalogEntry;

/** One block's definition, built-in only. */
export function getBlock(type: string): BlockDef | undefined {
  return builtIn.get(type);
}

/** Every block a document can use, built-ins first. */
export function catalogOf(doc?: Doc): Record<string, BlockDef> {
  const out: Record<string, BlockDef> = { ...CATALOG };
  for (const [type, def] of Object.entries(doc?.defs ?? {})) {
    if (!builtIn.has(type)) out[type] = fromUserBlock(type, def);
  }
  return out;
}
