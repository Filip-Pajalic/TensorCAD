/**
 * The run registry: training runs, kept so two designs can be compared.
 *
 * `tensorcad-runtime smoke-train` writes two files beside each other — a
 * `.jsonl` of one object per logged step, written as it goes so an interrupted
 * run still leaves something, and a `.json` record of the whole thing. This
 * reads either: the record when there is one, and the step log when there is
 * not, because the run that crashed is often the one you most want to look at.
 *
 * It lives in the browser because the editor does. A registry that indexed a
 * directory would need a server, and the thing being indexed is a handful of
 * files a person already knows where to find. What is kept here is the record,
 * not a path to it — `localStorage` survives a reload and a path does not
 * survive being on a different machine.
 */

import { create } from "zustand";

/** One logged step. */
export interface RunStep {
  step: number;
  loss: number;
  lr?: number;
  tokens?: number;
  tokens_per_second?: number;
  peak_memory_bytes?: number;
  seconds?: number;
}

/** A run as the trainer reported it, plus what this editor needs to show it. */
export interface RunRecord {
  id: string;
  /** What to call it in the list. The file's name unless the run named itself. */
  label: string;
  addedAt: string;

  params?: number;
  steps?: number;
  initial_loss?: number;
  final_loss?: number;
  best_loss?: number;
  tokens?: number;
  tokens_per_second?: number;
  peak_memory_bytes?: number;
  seconds?: number;
  device?: string;
  device_name?: string;
  dtype?: string;
  batch?: number;
  seq?: number;
  lr?: number;
  seed?: number;
  design_hash?: string;
  data_source?: string;
  torch_version?: string;
  model_path?: string;
  warnings?: string[];
  log: RunStep[];
}

const KEY = "tensorcad.runs";
/** Enough to compare a few designs; beyond that the chart is unreadable anyway. */
const LIMIT = 12;

export interface RunsState {
  runs: RunRecord[];
  /** Which runs are drawn. Everything loaded, until somebody says otherwise. */
  shown: string[];
  add: (runs: RunRecord[]) => void;
  remove: (id: string) => void;
  clear: () => void;
  toggle: (id: string) => void;
}

function load(): RunRecord[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as RunRecord[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function save(runs: RunRecord[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(runs));
  } catch {
    // A quota a dozen loss curves can exhaust is one there is nothing useful
    // to do about; the runs stay for this session.
  }
}

const initial = load();

export const useRuns = create<RunsState>((set, get) => ({
  runs: initial,
  shown: initial.map((r) => r.id),
  add: (incoming) => {
    const runs = [...get().runs.filter((r) => !incoming.some((i) => i.id === r.id)), ...incoming].slice(-LIMIT);
    save(runs);
    set({ runs, shown: [...new Set([...get().shown, ...incoming.map((r) => r.id)])] });
  },
  remove: (id) => {
    const runs = get().runs.filter((r) => r.id !== id);
    save(runs);
    set({ runs, shown: get().shown.filter((s) => s !== id) });
  },
  clear: () => {
    save([]);
    set({ runs: [], shown: [] });
  },
  toggle: (id) =>
    set((s) => ({ shown: s.shown.includes(id) ? s.shown.filter((x) => x !== id) : [...s.shown, id] })),
}));

/**
 * Read a run out of a file the person opened.
 *
 * Throws with something a person can act on rather than returning null: every
 * caller here is a file they chose deliberately, and "nothing happened" is the
 * worst possible answer to that.
 */
export function parseRun(name: string, text: string): RunRecord {
  const id = `${name}:${hash(text)}`;
  const base = { id, label: name.replace(/\.(json|jsonl)$/i, ""), addedAt: new Date().toISOString() };

  if (name.toLowerCase().endsWith(".jsonl") || text.trimStart().startsWith("{\"step\"")) {
    const log = text
      .split("\n")
      .filter((line) => line.trim().length > 0)
      .map((line) => JSON.parse(line) as RunStep);
    if (log.length === 0) throw new Error(`${name} has no steps in it.`);
    const last = log[log.length - 1]!;
    return {
      ...base,
      log,
      steps: log.length,
      initial_loss: log[0]!.loss,
      final_loss: last.loss,
      best_loss: Math.min(...log.map((s) => s.loss)),
      tokens: last.tokens,
      tokens_per_second: last.tokens_per_second,
      peak_memory_bytes: last.peak_memory_bytes,
      seconds: last.seconds,
      batch: (last as { batch?: number }).batch,
      seq: (last as { seq?: number }).seq,
    };
  }

  const record = JSON.parse(text) as Partial<RunRecord> & { ok?: boolean; error?: string };
  if (record.ok === false) throw new Error(`${name} is a failed run: ${record.error ?? "no reason given"}`);
  if (!Array.isArray(record.log)) throw new Error(`${name} has no "log" — is it a run record?`);
  return { ...base, ...record, log: record.log, id, addedAt: base.addedAt, label: labelOf(name, record) };
}

function labelOf(name: string, record: Partial<RunRecord>): string {
  const stem = name.replace(/\.(json|jsonl)$/i, "");
  // The model path says which design it was, which is the thing being compared;
  // the file name is usually that plus a timestamp, so prefer it when it says
  // more than the directory would.
  const model = record.model_path?.replace(/\\/g, "/").split("/").slice(-2, -1)[0];
  return model && !stem.includes(model) ? `${model} · ${stem}` : stem;
}

/** Enough to tell one file from another. Not a checksum. */
function hash(text: string): string {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(36);
}

/**
 * What makes two runs not comparable.
 *
 * A loss curve is only a comparison if both sides saw the same tokens under
 * the same conditions. Different sequence lengths mean different amounts of
 * context per step; a different corpus means different text; a different seed
 * means a different sample. Drawing them on one chart without saying so is how
 * an architecture gets credit for a batch size.
 */
export function incomparable(runs: RunRecord[]): string[] {
  if (runs.length < 2) return [];
  const notes: string[] = [];
  const differs = <K extends keyof RunRecord>(key: K, label: string): void => {
    const values = [...new Set(runs.map((r) => r[key]).filter((v) => v !== undefined))];
    if (values.length > 1) notes.push(`${label} differs: ${values.join(", ")}`);
  };
  differs("seq", "sequence length");
  differs("batch", "batch size");
  differs("data_source", "corpus");
  differs("seed", "seed");
  differs("lr", "learning rate");
  differs("dtype", "dtype");
  return notes;
}
