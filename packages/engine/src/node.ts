/**
 * Loading the engine outside a browser.
 *
 * The command line and the MCP server are Node processes with a filesystem, so
 * they read the module rather than fetching it, and they hold it in a module
 * singleton the way the editor does: one engine per process, loaded before the
 * first command runs.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { createEngine, EngineError, type Engine } from "./index.js";
import "../vendor/wasm_exec.js";

let loaded: Engine | null = null;

/** Where the compiled module sits, next to this package's source. */
function modulePath(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "wasm", "tensorcad.wasm");
}

/**
 * Loads the engine, once per process.
 *
 * A missing module is the one failure worth naming precisely: it means the
 * repository was cloned but not built, and the fix is one command.
 */
export async function loadEngine(): Promise<Engine> {
  if (loaded) return loaded;
  const path = modulePath();
  let bytes: ArrayBuffer;
  try {
    // Copied out of the Buffer rather than handed over: a Buffer is a view on
    // a pool shared with other reads, and WebAssembly takes the whole backing
    // store.
    const file = readFileSync(path);
    bytes = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength) as ArrayBuffer;
  } catch {
    throw new EngineError(
      `The analysis engine is not built (${path}). Run: bun run build:wasm`,
    );
  }
  loaded = await createEngine({ wasm: bytes });
  PRESET_NAMES.push(...loaded.presets());
  HARDWARE.push(...loaded.hardware());
  for (const entry of loaded.blocks.builtInEntries) CATALOG[entry.type] = entry;
  return loaded;
}

/** The engine. Throws if something asked before the load finished. */
export function engine(): Engine {
  if (!loaded) throw new EngineError("The engine is not loaded yet; await loadEngine() first.");
  return loaded;
}

export * from "./index.js";

// ---------------------------------------------------------------------------
// The engine's calls, as a process uses them
// ---------------------------------------------------------------------------
//
// Free functions over the singleton rather than methods on a handle. A command
// is a function that takes a document, and threading an engine through every
// one of them would say nothing: there is one engine per process and it is
// already loaded by the time any command runs.

import type {
  AnalysisOptions,
  AnalysisResult,
  CatalogEntry,
  ClusterRequest,
  ClusterResult,
  Derived,
  DesignDiff,
  Doc,
  Dtype,
  Explanation,
  GeneratedCode,
  HardwareProfile,
  ImportResult,
  Inference,
  ParamsResult,
  MupLadder,
  MupOptions,
  ScaleOptions,
  ScaleResult,
  SymbolTable,
  TorchOptions,
  ValidationReport,
} from "./index.js";

/** Filled in by the load, so a module can hold a reference at import time. */
export const PRESET_NAMES: string[] = [];
export const HARDWARE: HardwareProfile[] = [];
export const CATALOG: Record<string, CatalogEntry> = {};

export function analyze(doc: Doc, options?: AnalysisOptions): AnalysisResult {
  return engine().analyze(doc, options);
}

export function validate(doc: Doc, options?: AnalysisOptions): ValidationReport {
  return engine().validate(doc, options);
}

/** The findings and every shape, from one walk of the graph. */
export function derive(doc: Doc, options?: AnalysisOptions): Derived {
  return engine().derive(doc, options);
}

export function inferShapes(doc: Doc, mode: "flat" | "expanded" = "flat"): Inference {
  return engine().infer(doc, mode);
}

export function explain(doc: Doc, path: string, options?: AnalysisOptions): Explanation {
  return engine().explain(doc, path, options);
}

export function explainAll(doc: Doc, options?: AnalysisOptions): Explanation[] {
  return engine().explainAll(doc, options);
}

export function generateTorch(doc: Doc, options?: TorchOptions): GeneratedCode {
  return engine().generateTorch(doc, options);
}

export function scaleDesign(doc: Doc, options: ScaleOptions): ScaleResult {
  return engine().scale(doc, options);
}

export function mupLadder(doc: Doc, options?: MupOptions): MupLadder {
  return engine().mup(doc, options);
}

export function planCluster(
  doc: Doc,
  options: AnalysisOptions,
  cluster: ClusterRequest,
): ClusterResult {
  return engine().plan(doc, options, cluster);
}

export function diffDesigns(a: Doc, b: Doc, options?: AnalysisOptions): DesignDiff {
  return engine().diff(a, b, options);
}

export function importHfConfig(configText: string, name?: string): ImportResult {
  return engine().importHuggingFace(configText, name);
}

export function getPreset(name: string): Doc {
  return engine().preset(name);
}

/** The parameter counts alone. */
export function countParams(doc: Doc): ParamsResult {
  return engine().analyze(doc).params;
}

/** The symbol table alone. */
export function resolveSymbols(doc: Doc): SymbolTable {
  return engine().analyze(doc).symbols;
}

/** One block's definition, built-in or the design's own. */
export function getBlock(type: string, doc?: Doc): CatalogEntry | undefined {
  return engine().blocks.get(type, doc);
}

export function catalogByCategory(doc?: Doc): Record<string, CatalogEntry[]> {
  return engine().blocks.byCategory(doc);
}

export function hardwareById(id: string): HardwareProfile | undefined {
  return HARDWARE.find((h) => h.id === id);
}

/**
 * A device's throughput for a dtype, falling back to BF16 where FP8 is
 * unsupported. Arithmetic on a profile the caller already has, so it stays on
 * this side of the boundary.
 */
export function peakFlops(hw: HardwareProfile, dtype: Dtype): number {
  return dtype === "fp8" && hw.peakFp8 > 0 ? hw.peakFp8 : hw.peakBf16;
}
