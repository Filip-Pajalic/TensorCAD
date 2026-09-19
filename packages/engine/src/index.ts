/**
 * The engine, loaded into whatever is running the editor.
 *
 * TensorCAD's analysis is Go, compiled to WebAssembly. That is not an
 * optimisation: it is what keeps there from being two of it. The desktop shell,
 * a browser tab and the command line all ask the same code the same questions,
 * so an answer cannot depend on where it was asked.
 *
 * Everything crosses the boundary as JSON text. A design *is* JSON and so is
 * every report, so serialising costs a copy and buys a boundary with nothing
 * clever in it — no object graph to keep in step, no handles to leak.
 */

import type {
  AnalysisOptions,
  AnalysisResult,
  CatalogEntry,
  Doc,
  Explanation,
  GeneratedCode,
  HardwareProfile,
  ImportResult,
  ScaleOptions,
  ScaleResult,
  TorchOptions,
  ValidationReport,
} from "./types.js";

export * from "./types.js";
export * from "./format.js";

/** What the engine says about itself. */
export interface EngineVersion {
  engine: string;
  target: string;
}

/**
 * The engine's whole surface.
 *
 * Synchronous, because the calls run on the thread that asked and the largest
 * design in the library analyses in a couple of milliseconds. Loading is the
 * asynchronous part, and it happens once.
 */
export interface Engine {
  version(): EngineVersion;
  analyze(doc: Doc, options?: AnalysisOptions): AnalysisResult;
  /** Runs the design rules and returns the analysis they were drawn from. */
  validate(doc: Doc, options?: AnalysisOptions): ValidationReport;
  explain(doc: Doc, path: string, options?: AnalysisOptions): Explanation;
  /** Every block, largest contribution first. */
  explainAll(doc: Doc, options?: AnalysisOptions): Explanation[];
  generateTorch(doc: Doc, options?: TorchOptions): GeneratedCode;
  scale(doc: Doc, options: ScaleOptions): ScaleResult;
  /** The design library this engine ships with. */
  presets(): string[];
  preset(name: string): Doc;
  /** Reads a Hugging Face `config.json`. */
  importHuggingFace(configText: string, name?: string): ImportResult;
  /** Every block the engine knows, for the palette. */
  catalog(): CatalogEntry[];
  hardware(): HardwareProfile[];
}

/** The shape the WebAssembly module publishes on `globalThis`. */
interface Exports {
  version(): string;
  analyze(doc: string, options: string): string;
  validate(doc: string, options: string): string;
  explain(doc: string, path: string, options: string): string;
  explainAll(doc: string, options: string): string;
  generateTorch(doc: string, options: string): string;
  scale(doc: string, options: string): string;
  presets(): string;
  preset(name: string): string;
  importHf(configText: string, name: string): string;
  catalog(): string;
  hardware(): string;
}

/**
 * Raised when the engine refuses a call.
 *
 * A distinct type because a refusal is not a bug: an unknown hardware profile
 * or a design with no version is something a person can fix, and the editor
 * shows it rather than logging it.
 */
export class EngineError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EngineError";
  }
}

/** Where to find the compiled engine, and what to run it with. */
export interface LoadOptions {
  /**
   * The `.wasm` file. A URL is fetched; bytes are used as they are. Omitted,
   * it is resolved next to this module, which is what a bundler produces.
   */
  wasm?: string | URL | BufferSource;
}

declare global {
  // eslint-disable-next-line no-var
  var Go: (new () => {
    importObject: WebAssembly.Imports;
    run(instance: WebAssembly.Instance): Promise<void>;
  }) | undefined;
  // eslint-disable-next-line no-var
  var __tensorcad: Exports | undefined;
}

function unwrap(text: string): unknown {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new EngineError(`The engine returned something that is not JSON: ${text.slice(0, 200)}`);
  }
  if (parsed && typeof parsed === "object" && "error" in parsed && typeof parsed.error === "string") {
    throw new EngineError(parsed.error);
  }
  return parsed;
}

/** The operating point, as text. An absent one means "use the defaults". */
function point(options: AnalysisOptions | undefined): string {
  return options ? JSON.stringify(options) : "";
}

async function bytesOf(wasm: LoadOptions["wasm"]): Promise<BufferSource> {
  if (wasm && typeof wasm !== "string" && !(wasm instanceof URL)) return wasm;
  const url = wasm ?? new URL("../wasm/tensorcad.wasm", import.meta.url);
  const response = await fetch(url);
  if (!response.ok) {
    throw new EngineError(`Could not load the engine from ${String(url)}: ${response.status}`);
  }
  return await response.arrayBuffer();
}

/**
 * Loads the engine.
 *
 * `globalThis.Go` has to exist first: it comes from the Go toolchain's
 * `wasm_exec.js`, which is vendored beside this file. A bundler wants
 * `import "@tensorcad/engine/wasm_exec"` before the first call.
 */
export async function createEngine(options: LoadOptions = {}): Promise<Engine> {
  if (typeof globalThis.Go !== "function") {
    throw new EngineError(
      'The Go WebAssembly runtime is missing. Import "@tensorcad/engine/wasm_exec" before creating the engine.',
    );
  }

  const go = new globalThis.Go();
  // instantiate answers differently for bytes than for an already-compiled
  // module, and the ambient types only describe one of the two. A caller may
  // hand over either, so both shapes are handled.
  const compiled = (await WebAssembly.instantiate(await bytesOf(options.wasm), go.importObject)) as
    | WebAssembly.Instance
    | WebAssembly.WebAssemblyInstantiatedSource;
  const instance = "instance" in compiled ? compiled.instance : compiled;

  // The Go program does not return: it publishes its API and then parks, so
  // this promise settles only when the engine shuts down. Awaiting it would
  // wait forever.
  void go.run(instance);

  const api = globalThis.__tensorcad;
  if (!api) {
    throw new EngineError("The engine started but published no API.");
  }
  return wrap(api);
}

function wrap(api: Exports): Engine {
  return {
    version: () => unwrap(api.version()) as EngineVersion,
    analyze: (doc, options) => unwrap(api.analyze(JSON.stringify(doc), point(options))) as AnalysisResult,
    validate: (doc, options) => unwrap(api.validate(JSON.stringify(doc), point(options))) as ValidationReport,
    explain: (doc, path, options) =>
      unwrap(api.explain(JSON.stringify(doc), path, point(options))) as Explanation,
    explainAll: (doc, options) =>
      unwrap(api.explainAll(JSON.stringify(doc), point(options))) as Explanation[],
    generateTorch: (doc, options) =>
      unwrap(api.generateTorch(JSON.stringify(doc), options ? JSON.stringify(options) : "")) as GeneratedCode,
    scale: (doc, options) => unwrap(api.scale(JSON.stringify(doc), JSON.stringify(options))) as ScaleResult,
    presets: () => unwrap(api.presets()) as string[],
    preset: (name) => unwrap(api.preset(name)) as Doc,
    importHuggingFace: (configText, name) =>
      unwrap(api.importHf(configText, name ?? "")) as ImportResult,
    catalog: () => unwrap(api.catalog()) as CatalogEntry[],
    hardware: () => unwrap(api.hardware()) as HardwareProfile[],
  };
}
