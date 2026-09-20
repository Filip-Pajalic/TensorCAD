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
  type AnalysisOptions,
  type ClusterRequest,
  type ClusterResult,
  type DesignDiff,
  createEngine,
  isComposite,
  isContainer,
  isPrimitive,
  type CatalogEntry,
  type Doc,
  type Engine,
  type HardwareProfile,
  type GeneratedCode,
  type Inference,
  type ParamSpec,
  type RuleInfo,
  type SymbolTable,
  type TorchOptions,
  type UserBlockDef,
} from "@tensorcad/engine";
import "@tensorcad/engine/wasm_exec";

// Re-exported so a panel asking what kind of block it has does not have to
// reach past this module for it.
export { isComposite, isContainer, isPrimitive };

let loaded: Engine | null = null;

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
  for (const entry of engine.blocks.builtInEntries) CATALOG[entry.type] = entry;
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

/**
 * Every way of splitting the training across the cluster, and which of them fit.
 *
 * Not part of `derive`: this is a few hundred analyses and the answer only
 * changes when the cluster or the design does, where `derive` runs on every
 * keystroke.
 */
export function planCluster(
  doc: Doc,
  options: AnalysisOptions,
  cluster: ClusterRequest,
): ClusterResult {
  return engine().plan(doc, options, cluster);
}

/** What changed between two designs, structurally and numerically. */
export function diffDesigns(a: Doc, b: Doc, options?: AnalysisOptions): DesignDiff {
  return engine().diff(a, b, options);
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
//
// The folding of a design's own blocks into the engine's lives in the engine
// package, because the command line and the MCP server need exactly the same
// answers. What is here is the editor's names for it.

/**
 * A block's definition, as the editor's own name for it.
 *
 * The engine calls this a catalog entry. The editor has always called it a
 * block definition, and renaming every call site would be churn for nothing.
 */
export type BlockDef = CatalogEntry;

/** One block's definition, built-in or the design's own. */
export function blockDef(type: string, doc?: Doc): BlockDef | undefined {
  return engine().blocks.get(type, doc);
}

/** One block's definition, built-in only. */
export function getBlock(type: string): BlockDef | undefined {
  return engine().blocks.isBuiltIn(type) ? engine().blocks.get(type) : undefined;
}

/** Every block a document can use, built-ins first. */
export function catalogEntries(doc?: Doc): BlockDef[] {
  return engine().blocks.entries(doc);
}

export function catalogByCategory(doc?: Doc): Record<string, BlockDef[]> {
  return engine().blocks.byCategory(doc);
}

/** Every block a document can use, by type. */
export function catalogOf(doc?: Doc): Record<string, BlockDef> {
  const out: Record<string, BlockDef> = {};
  for (const entry of catalogEntries(doc)) out[entry.type] = entry;
  return out;
}

/** True when this type came from the design rather than from the engine. */
export function isUserBlock(doc: Doc | undefined, type: string): boolean {
  return engine().blocks.isUserBlock(doc, type);
}

/** A block's parameter spec, by name. */
export function paramSpec(def: BlockDef | undefined, name: string): ParamSpec | undefined {
  return def?.params[name];
}
