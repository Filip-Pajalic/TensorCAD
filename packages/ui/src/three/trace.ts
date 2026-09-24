/**
 * What a design actually computes, when somebody has run it.
 *
 * Everything else the editor shows is a design's structure or its cost. A trace
 * is its behaviour: one model, trained by `tensorcad-runtime trace`, run on one
 * input, with every weight and every activation recorded under the block path
 * the editor already uses. The volume view puts those numbers in its cells.
 *
 * One trace is committed, for `nano-sort`: the editor is a static site and the
 * runtime is Python, so what it shows on its own can only have been made ahead
 * of time — `bun run trace` remakes it, deliberately, like the goldens. Any
 * other trace is *added*: a file `tensorcad-runtime trace` wrote, opened with
 * File > Load a trace, or handed over by the desktop shell after it ran the
 * runtime itself. Both kinds are loaded only when something asks for them.
 *
 * A trace describes the design it was made from and nothing else, so it is
 * shown only on a design that still generates the same `model.py`. That file
 * is the whole computation: an edit that changes what the model does changes
 * its bytes, and an edit that does not — a note, a layout — leaves the numbers
 * true. Anything cheaper than that check would sometimes put one model's
 * numbers on another model's drawing, which is worse than showing none.
 */

import { useEffect, useState } from "react";
import type { Doc } from "@tensor-cad/engine";
import { generateTorch } from "../engine.js";

interface Encoded {
  shape: number[];
  /** Little-endian float32, base64. */
  data: string;
}

interface TraceEntry extends Partial<Encoded> {
  path: string;
  layer: number | null;
  role?: "in" | "out";
  param?: string;
  /** An input that is another module's output is stored once, under that name. */
  same_as?: string;
}

export interface TraceFile {
  version: 1;
  design: string;
  model_sha256: string;
  params: number;
  symbols: Record<string, number>;
  /**
   * What the model was run as. `sort` was trained until it sorted every
   * held-out input; `untrained` is the model exactly as initialised, which is
   * what a design whose vocabulary is too large to sort gets.
   */
  task: { name: "sort" | "untrained"; length: number | null; vocab: number; symbols: string[] };
  training: {
    seed: number;
    steps: number;
    final_loss: number | null;
    held_out_accuracy: number | null;
  };
  input: number[];
  sequence: number[];
  answer: number[] | null;
  predicted: number[];
  checks: {
    sorted_correctly: boolean | null;
    attention_max_abs_error: number | null;
    attention_note: string | null;
  };
  weights: Record<string, TraceEntry>;
  activations: Record<string, TraceEntry>;
  attention: { path: string; layer: number | null; scores: Encoded; weights: Encoded }[];
}

/** A tensor, decoded: row-major, PyTorch's own axis order. */
export interface Tensor {
  shape: number[];
  data: Float32Array;
}

/**
 * What a block of the view is a picture of, in the trace's terms.
 *
 * Kept in the design's vocabulary — a block path, a layer and what of it —
 * rather than in PyTorch module names, so the layout can say what it draws
 * without knowing how a model is generated.
 */
export type Role =
  | "in"
  | "out"
  | "weight"
  | "bias"
  /** Attention before the softmax: `[heads, query, key]`, masked cells NaN. */
  | "scores"
  /** Attention after it. */
  | "probs"
  /** The input sequence itself, one symbol per position. */
  | "tokens"
  /** A layer norm's statistics over its input, one per position. */
  | "mean"
  | "std"
  /** The softmax of the output over its last axis: the next-token distribution. */
  | "softmax";

export interface CellSource {
  /** The primitive's path in the design, e.g. `layers/block/attn/q_proj`. */
  path: string;
  /** Which repeat of the stack, or -1 outside it. */
  layer: number;
  role: Role;
  /** Which of the tensor's two axes runs across the block; the other runs down it. */
  across: 0 | 1;
  /** One head's share of an axis. */
  slice?: { axis: 0 | 1; start: number; size: number };
  /** One head of a per-head tensor. */
  head?: number;
}

function decode(e: Encoded): Tensor {
  const bin = atob(e.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  // The runtime writes little-endian, and so is every machine this runs on;
  // a DataView would say so at the cost of a copy per value.
  return { shape: e.shape, data: new Float32Array(bytes.buffer) };
}

const key = (path: string, layer: number | null, what: string): string => `${path}#${layer ?? -1}#${what}`;

/** A trace, indexed by what the view asks for. */
export class Trace {
  readonly file: TraceFile;
  private readonly entries = new Map<string, Encoded>();
  private readonly decoded = new Map<string, Tensor>();

  constructor(file: TraceFile) {
    this.file = file;
    for (const w of Object.values(file.weights)) {
      if (w.shape && w.data && w.param) this.entries.set(key(w.path, w.layer, w.param), w as Encoded);
    }
    for (const a of Object.values(file.activations)) {
      const source = a.same_as ? file.activations[a.same_as] : a;
      if (source?.shape && source.data && a.role) this.entries.set(key(a.path, a.layer, a.role), source as Encoded);
    }
    for (const a of file.attention) {
      this.entries.set(key(a.path, a.layer, "scores"), a.scores);
      this.entries.set(key(a.path, a.layer, "probs"), a.weights);
    }
  }

  /** The number of positions the trace ran; a drawing of more than this has no values past it. */
  get positions(): number {
    return this.file.sequence.length;
  }

  /** The input, as the task's own symbols: letters for sort, token ids otherwise. */
  get letters(): string[] {
    return this.file.sequence.map((t) => this.file.task.symbols[t] ?? String(t));
  }

  /** The model as initialised, never trained. Everything that shows it says so. */
  get untrained(): boolean {
    return this.file.task.name === "untrained";
  }

  /**
   * One phrase for what these numbers are, for the legend, the picture's label
   * and anywhere else that has room for a phrase. Long inputs are elided: the
   * point is to say which run it is, not to print it.
   */
  get summary(): string {
    const shown = this.letters.length > 12 ? [...this.letters.slice(0, 12), "\u2026"] : this.letters;
    return this.untrained
      ? `untrained, as initialised, on ${this.positions} token ids: ${shown.join(" ")}`
      : `trained to sort, reading ${shown.join(" ")}`;
  }

  tensor(path: string, layer: number, what: string): Tensor | null {
    const k = key(path, layer < 0 ? null : layer, what);
    const hit = this.decoded.get(k);
    if (hit) return hit;
    const e = this.entries.get(k);
    if (!e) return null;
    const t = decode(e);
    this.decoded.set(k, t);
    return t;
  }

  /** The tensor a block of the view draws, as it names it; derived ones computed here. */
  resolve(src: CellSource): Tensor | null {
    switch (src.role) {
      case "tokens":
        return { shape: [this.positions], data: Float32Array.from(this.file.sequence) };
      case "mean":
      case "std": {
        const x = this.tensor(src.path, src.layer, "in");
        return x && rowStats(x, src.role);
      }
      case "softmax": {
        const x = this.tensor(src.path, src.layer, "out");
        return x && softmaxRows(x);
      }
      default:
        return this.tensor(src.path, src.layer, src.role);
    }
  }
}

function rowStats(x: Tensor, which: "mean" | "std"): Tensor {
  const [rows, cols] = [x.shape[0]!, x.shape[1] ?? 1];
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let c = 0; c < cols; c++) sum += x.data[r * cols + c]!;
    const mean = sum / cols;
    if (which === "mean") {
      out[r] = mean;
      continue;
    }
    let sq = 0;
    for (let c = 0; c < cols; c++) sq += (x.data[r * cols + c]! - mean) ** 2;
    out[r] = Math.sqrt(sq / cols);
  }
  return { shape: [rows], data: out };
}

function softmaxRows(x: Tensor): Tensor {
  const [rows, cols] = [x.shape[0]!, x.shape[1] ?? 1];
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    for (let c = 0; c < cols; c++) max = Math.max(max, x.data[r * cols + c]!);
    let sum = 0;
    for (let c = 0; c < cols; c++) sum += out[r * cols + c] = Math.exp(x.data[r * cols + c]! - max);
    for (let c = 0; c < cols; c++) out[r * cols + c]! /= sum;
  }
  return { shape: [rows, cols], data: out };
}

/**
 * A block's cells, `cx` across and `cy` down, row by row from the top.
 *
 * Null when the trace has no such tensor or it is smaller than the block — a
 * drawing at a longer sequence than the trace ran has positions nobody
 * computed. A *shorter* one is fine: the model is causal, so the first `T`
 * positions of a longer run are exactly what a run of `T` would have produced.
 * NaN survives, meaning a cell that was never visible.
 */
export function cellsFor(trace: Trace, src: CellSource, cx: number, cy: number): Float32Array | null {
  const t = trace.resolve(src);
  if (!t) return null;

  let shape = t.shape;
  let offset = 0;
  if (src.head !== undefined) {
    if (shape.length !== 3 || src.head >= shape[0]!) return null;
    offset = src.head * shape[1]! * shape[2]!;
    shape = shape.slice(1);
  }
  if (shape.length === 1) shape = [shape[0]!, 1];
  if (shape.length !== 2) return null;
  const [rows, cols] = shape as [number, number];

  // Where along each tensor axis the block's window starts, and how long it is.
  const start = [0, 0];
  const extent = [rows, cols];
  if (src.slice) {
    start[src.slice.axis] = src.slice.start;
    extent[src.slice.axis] = src.slice.size;
    if (src.slice.start + src.slice.size > shape[src.slice.axis]!) return null;
  }
  const down = src.across === 0 ? 1 : 0;
  if (extent[src.across]! < cx || extent[down]! < cy) return null;

  const out = new Float32Array(cx * cy);
  for (let y = 0; y < cy; y++) {
    for (let x = 0; x < cx; x++) {
      const i = [0, 0];
      i[src.across] = start[src.across]! + x;
      i[down] = start[down]! + y;
      out[y * cx + x] = t.data[offset + i[0]! * cols + i[1]!]!;
    }
  }
  return out;
}

// --- which designs a trace describes ---------------------------------------

/** The committed traces, by design name. Each is its own chunk, fetched on first use. */
const TRACES: Record<string, () => Promise<{ default: unknown }>> = {
  "nano-sort": () => import("./traces/nano-sort.json"),
};

const loaded = new Map<string, Promise<Trace | null>>();

/**
 * Traces added while the editor runs, newest first. Kept in memory for the
 * session: they are megabytes each, and a trace is cheap to make again.
 */
const added: Trace[] = [];
const listeners = new Set<() => void>();

/** Be told when a trace is added, so a view can look again. */
export function subscribeTraces(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/**
 * Read a trace file, refusing one that is not a trace rather than letting it
 * fail later as a picture of nothing.
 */
export function parseTrace(text: string): TraceFile {
  let file: unknown;
  try {
    file = JSON.parse(text);
  } catch (e) {
    throw new Error(`not JSON: ${(e as Error).message}`);
  }
  const f = file as Partial<TraceFile> | null;
  if (!f || typeof f !== "object") throw new Error("not a trace");
  if (f.version !== 1) throw new Error(`a trace of version ${String(f.version)}, and this editor reads version 1`);
  if (typeof f.model_sha256 !== "string" || !/^[0-9a-f]{64}$/.test(f.model_sha256)) {
    throw new Error("no fingerprint of the model it was made from, so it cannot be matched to a design");
  }
  if (!f.weights || !f.activations || !Array.isArray(f.sequence) || !f.task) {
    throw new Error("missing its weights, its activations or its input");
  }
  return f as TraceFile;
}

/**
 * Add a trace. It shows on whatever open design generates the model it was
 * made from, and on nothing else; a second trace of the same model replaces
 * the first.
 */
export function addTrace(file: TraceFile): Trace {
  const trace = new Trace(file);
  const same = added.findIndex((t) => t.file.model_sha256 === file.model_sha256);
  if (same >= 0) added.splice(same, 1);
  added.unshift(trace);
  for (const fn of listeners) fn();
  return trace;
}

/**
 * Load a trace file for the open design, and say what happened in one sentence.
 *
 * A trace of another model is still kept — it will show when that design is
 * opened — but the sentence says it is not this one, because a load that
 * silently changes nothing on screen reads as a load that failed.
 */
export async function loadTrace(text: string, doc: Doc): Promise<{ ok: boolean; message: string }> {
  let file: TraceFile;
  try {
    file = parseTrace(text);
  } catch (e) {
    return { ok: false, message: `That is not a trace TensorCAD can read: ${(e as Error).message}.` };
  }
  const trace = addTrace(file);
  const what = trace.untrained ? "untrained" : "trained to sort";
  const matches = (await traceFor(doc)) === trace;
  return {
    ok: true,
    message: matches
      ? `Loaded a trace of ${file.design} (${what}). Open the volume view to see its values.`
      : `Loaded a trace of ${file.design} (${what}), but the open design does not generate the model it was made from. It will show when that design is open.`,
  };
}

function load(name: string): Promise<Trace | null> {
  let hit = loaded.get(name);
  if (!hit) {
    const get = TRACES[name];
    hit = get
      ? get().then(
          (m) => new Trace(m.default as TraceFile),
          () => null,
        )
      : Promise.resolve(null);
    loaded.set(name, hit);
  }
  return hit;
}

async function sha256(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** The generated model's source, which is what a trace is fingerprinted by. */
export function modelSource(doc: Doc): string | null {
  try {
    return generateTorch(doc).files.find((f) => f.path === "model.py")?.contents ?? null;
  } catch {
    return null;
  }
}

/**
 * The trace of this design, if there is one and it still describes it.
 *
 * An added trace first, since it was made on purpose and more recently; then
 * the committed one, for which the name is only where to look. What decides,
 * either way, is the generated model.
 */
export async function traceFor(doc: Doc): Promise<Trace | null> {
  const name = doc.meta.name;
  const committed = name && name in TRACES;
  if (added.length === 0 && !committed) return null;
  const source = modelSource(doc);
  if (source === null) return null;
  const hash = await sha256(source);
  const mine = added.find((t) => t.file.model_sha256 === hash);
  if (mine) return mine;
  if (!committed) return null;
  const trace = await load(name);
  return trace && trace.file.model_sha256 === hash ? trace : null;
}

/**
 * The open design's trace, for a component: null until it has loaded, and null
 * for good on a design it does not describe.
 */
export function useTrace(doc: Doc): Trace | null {
  const [trace, setTrace] = useState<Trace | null>(null);
  // Bumped when a trace is added, so the design on screen is looked up again.
  const [additions, setAdditions] = useState(0);
  useEffect(() => subscribeTraces(() => setAdditions((n) => n + 1)), []);
  useEffect(() => {
    let live = true;
    traceFor(doc).then(
      (t) => live && setTrace(t),
      () => live && setTrace(null),
    );
    return () => {
      live = false;
    };
  }, [doc, additions]);
  return trace;
}
