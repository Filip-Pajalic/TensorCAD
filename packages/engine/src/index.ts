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
  Derived,
  Doc,
  Explanation,
  GeneratedCode,
  HardwareProfile,
  ImportResult,
  Inference,
  ScaleOptions,
  ScaleResult,
  TorchOptions,
  UserBlockDef,
  ValidationReport,
} from "./types.js";

export * from "./types.js";
export * from "./format.js";
export * from "./ir.js";

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
  /**
   * The editor's entry point: the findings and the shapes together.
   *
   * One call rather than two, because both come from the same walk of the same
   * graph and asking separately would walk it twice on every keystroke.
   */
  derive(doc: Doc, options?: AnalysisOptions): Derived;
  /**
   * Shape inference alone, which is what answers "would this wire type-check"
   * for every handle the pointer passes over.
   */
  infer(doc: Doc, mode?: "flat" | "expanded"): Inference;
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
  /** The design rules, for the panel that lists what is being checked. */
  rules(): RuleInfo[];
  /**
   * What is wrong with a block a design defines for itself, so the block editor
   * can say so while it is being written rather than after it is saved.
   */
  checkUserBlock(def: UserBlockDef, name: string): string[];
  hardware(): HardwareProfile[];
}

/** The shape the WebAssembly module publishes on `globalThis`. */
interface Exports {
  version(): string;
  analyze(doc: string, options: string): string;
  validate(doc: string, options: string): string;
  derive(doc: string, options: string): string;
  infer(doc: string, mode: string): string;
  explain(doc: string, path: string, options: string): string;
  explainAll(doc: string, options: string): string;
  generateTorch(doc: string, options: string): string;
  scale(doc: string, options: string): string;
  presets(): string;
  preset(name: string): string;
  importHf(configText: string, name: string): string;
  catalog(): string;
  rules(): string;
  checkUserBlock(def: string, name: string): string;
  hardware(): string;
}

/** One design rule, as the rules panel lists it. */
export interface RuleInfo {
  id: string;
  title: string;
  /** One line on what the rule protects against. */
  description: string;
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
    derive: (doc, options) => unwrap(api.derive(JSON.stringify(doc), point(options))) as Derived,
    infer: (doc, mode) => unwrap(api.infer(JSON.stringify(doc), mode ?? "flat")) as Inference,
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
    rules: () => unwrap(api.rules()) as RuleInfo[],
    checkUserBlock: (def, name) => unwrap(api.checkUserBlock(JSON.stringify(def), name)) as string[],
    hardware: () => unwrap(api.hardware()) as HardwareProfile[],
  };
}
