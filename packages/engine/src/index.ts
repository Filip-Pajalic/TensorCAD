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

import { Catalog } from "./catalog.js";
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
  MaskView,
  ModelFamily,
  MupLadder,
  MupOptions,
  NewDesignRequest,
  NewDesignResult,
  ScaleOptions,
  ScaleResult,
  ClusterRequest,
  ClusterResult,
  DesignDiff,
  TorchOptions,
  UserBlockDef,
  ValidationReport,
} from "./types.js";

export * from "./types.js";
export * from "./format.js";
export * from "./ir.js";
export * from "./catalog.js";

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
  /**
   * The attention mask of the block at `path` — itself if it is an `sdpa`,
   * otherwise the first attention inside it — drawn at the operating point's
   * sequence length, for one head.
   */
  attentionMask(doc: Doc, path: string, options?: AnalysisOptions, head?: number): MaskView;
  generateTorch(doc: Doc, options?: TorchOptions): GeneratedCode;
  scale(doc: Doc, options: ScaleOptions): ScaleResult;
  /** The kinds of model a new design can start as. */
  families(): ModelFamily[];
  /**
   * A new design of a kind, at a size or as large as trains on one device:
   * the kind's reference design, scaled with its proportions kept, and what
   * training it would take on one device under `options`.
   */
  newDesign(request: NewDesignRequest, options?: AnalysisOptions): NewDesignResult;
  /**
   * The same design at several widths, with what to multiply the
   * initialization and the learning rate by at each one.
   *
   * It is what makes a sweep affordable: tune at a width that fits on one
   * device, and carry the answer up the ladder. What it does not do is decide
   * the base learning rate, which is what the sweep is for.
   */
  mup(doc: Doc, options?: MupOptions): MupLadder;
  /**
   * Every way of splitting the work across a cluster, and which of them fit.
   *
   * Memory is the claim, and it is arithmetic. Which plan is fastest is not:
   * that turns on the interconnect and the kernels, so each plan carries a
   * note about what it costs to run rather than a number pretending to.
   */
  plan(doc: Doc, options: AnalysisOptions, cluster: ClusterRequest): ClusterResult;
  /** What changed between two designs, structurally and numerically. */
  diff(a: Doc, b: Doc, options?: AnalysisOptions): DesignDiff;
  /** The design library this engine ships with. */
  presets(): string[];
  preset(name: string): Doc;
  /** Reads a Hugging Face `config.json`. */
  importHuggingFace(configText: string, name?: string): ImportResult;
  /** Every block the engine knows, for the palette. */
  catalog(): CatalogEntry[];
  /**
   * The same blocks, ready to be asked about one at a time and to have a
   * design's own definitions folded in. Fetched once when the engine loads.
   */
  readonly blocks: Catalog;
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
  attentionMask(doc: string, path: string, options: string, head: string): string;
  generateTorch(doc: string, options: string): string;
  scale(doc: string, options: string): string;
  families(): string;
  newDesign(request: string, options: string): string;
  mup(doc: string, options: string): string;
  plan(doc: string, options: string, cluster: string): string;
  diff(a: string, b: string, options: string): string;
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

/**
 * The first four bytes of every WebAssembly module: `\0asm`.
 *
 * Checked rather than trusted, because the thing that goes wrong here does not
 * look like an error. A development server answering an unknown path with its
 * single-page fallback returns **200 OK** and `index.html`, so `response.ok`
 * is true and the failure surfaces much later as
 * "expected magic word 00 61 73 6d, found 3c 21 64 6f" — `3c 21 64 6f` being
 * `<!do`. Four bytes here turn that into a sentence naming the file.
 */
const WASM_MAGIC = [0x00, 0x61, 0x73, 0x6d];

function looksLikeWasm(bytes: ArrayBuffer): boolean {
  if (bytes.byteLength < 4) return false;
  const head = new Uint8Array(bytes, 0, 4);
  return WASM_MAGIC.every((b, i) => head[i] === b);
}

async function bytesOf(wasm: LoadOptions["wasm"]): Promise<BufferSource> {
  if (wasm && typeof wasm !== "string" && !(wasm instanceof URL)) return wasm;

  // **This literal must stay a literal.** A bundler rewrites
  // `new URL("…", import.meta.url)` to the asset it emitted, and it can only
  // do that when it can read the path without running anything. Built from a
  // variable — an array of candidates to try in turn, say — the rewrite
  // silently stops happening, the URL is resolved at runtime against the
  // deployed module, and the site serves its own index.html in place of the
  // engine. That shipped for eleven minutes.
  //
  // The published package keeps its module beside the entry rather than a
  // directory up, and `scripts/build-dist.ts` edits this one path when it
  // writes that package out — where the difference between the two layouts is
  // already recorded, and asserted so it cannot quietly stop matching.
  const urls = wasm ? [wasm] : [new URL("../wasm/tensorcad.wasm", import.meta.url)];

  const tried: string[] = [];
  for (const url of urls) {
    // `try`, not `fetch(...).catch(...)`. A missing `file://` makes Bun's
    // fetch throw *synchronously* rather than return a rejected promise, so a
    // `.catch()` is never attached in time and the first candidate takes the
    // whole load down with it — which is how this was written first, and what
    // the editor's own tests caught.
    try {
      const response = await fetch(url);
      if (!response.ok) {
        tried.push(`${String(url)} (${response.status})`);
        continue;
      }
      const bytes = await response.arrayBuffer();
      if (looksLikeWasm(bytes)) return bytes;
      tried.push(`${String(url)} (not WebAssembly — ${bytes.byteLength} bytes)`);
    } catch (error) {
      tried.push(`${String(url)} (${(error as Error).message})`);
    }
  }

  throw new EngineError(
    `Could not load the engine. Tried ${tried.join(", ")}. ` +
      `Pass \`wasm\` to createEngine() if it lives somewhere else.`,
  );
}

/**
 * Loads the engine.
 *
 * `globalThis.Go` has to exist first: it comes from the Go toolchain's
 * `wasm_exec.js`, which is vendored beside this file. A bundler wants
 * `import "@tensor-cad/engine/wasm_exec"` before the first call.
 */
export async function createEngine(options: LoadOptions = {}): Promise<Engine> {
  if (typeof globalThis.Go !== "function") {
    throw new EngineError(
      'The Go WebAssembly runtime is missing. Import "@tensor-cad/engine/wasm_exec" before creating the engine.',
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
  const entries = JSON.parse(api.catalog()) as CatalogEntry[];
  return {
    blocks: new Catalog(entries),
    version: () => unwrap(api.version()) as EngineVersion,
    analyze: (doc, options) => unwrap(api.analyze(JSON.stringify(doc), point(options))) as AnalysisResult,
    validate: (doc, options) => unwrap(api.validate(JSON.stringify(doc), point(options))) as ValidationReport,
    derive: (doc, options) => unwrap(api.derive(JSON.stringify(doc), point(options))) as Derived,
    infer: (doc, mode) => unwrap(api.infer(JSON.stringify(doc), mode ?? "flat")) as Inference,
    explain: (doc, path, options) =>
      unwrap(api.explain(JSON.stringify(doc), path, point(options))) as Explanation,
    explainAll: (doc, options) =>
      unwrap(api.explainAll(JSON.stringify(doc), point(options))) as Explanation[],
    attentionMask: (doc, path, options, head) =>
      unwrap(
        api.attentionMask(JSON.stringify(doc), path, point(options), String(head ?? 0)),
      ) as MaskView,
    generateTorch: (doc, options) =>
      unwrap(api.generateTorch(JSON.stringify(doc), options ? JSON.stringify(options) : "")) as GeneratedCode,
    scale: (doc, options) => unwrap(api.scale(JSON.stringify(doc), JSON.stringify(options))) as ScaleResult,
    families: () => unwrap(api.families()) as ModelFamily[],
    newDesign: (request, options) =>
      unwrap(api.newDesign(JSON.stringify(request), point(options))) as NewDesignResult,
    mup: (doc, options) =>
      unwrap(api.mup(JSON.stringify(doc), options ? JSON.stringify(options) : "")) as MupLadder,
    plan: (doc, options, cluster) =>
      unwrap(
        api.plan(JSON.stringify(doc), JSON.stringify(options ?? {}), JSON.stringify(cluster)),
      ) as ClusterResult,
    diff: (a, b, options) =>
      unwrap(
        api.diff(JSON.stringify(a), JSON.stringify(b), JSON.stringify(options ?? {})),
      ) as DesignDiff,
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
