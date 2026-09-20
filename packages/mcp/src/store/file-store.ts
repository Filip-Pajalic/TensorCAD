/**
 * The headless document store: `.tensorcad.json` files plus the built-in presets,
 * held in memory with a revision counter and an operation log.
 *
 * The live editor bridge watches this store through `subscribe` rather than
 * replacing it. An earlier note here proposed a second `DocumentStore` that
 * proxied to the editor; that would have given a design two homes and no rule
 * for which one is right when they differ. See `src/bridge/`.
 */

import { readdir, readFile, stat, writeFile, mkdir } from "node:fs/promises";
import { dirname, extname, isAbsolute, join, resolve } from "node:path";
import { applyOps, type Op } from "../ops.js";
import type { Doc } from "@tensor-cad/engine";
import { PRESET_NAMES, getPreset } from "@tensor-cad/engine/node";
import {
  RevisionConflictError,
  UnknownDesignError,
  type ApplyOutcome,
  type CheckpointInfo,
  type DesignRecord,
  type DesignSummary,
  type DocumentStore,
  type NewDesignOptions,
  type StoreChange,
  type StoreListener,
} from "./types.js";

interface LogEntry {
  revision: number;
  at: string;
  ops: Op[];
  /** The document as it was *before* this batch, which is what undo restores. */
  before: Doc;
}

interface Entry {
  record: DesignRecord;
  log: LogEntry[];
  checkpoints: Map<string, CheckpointInfo & { doc: Doc }>;
}

const EMPTY_DOC = (name: string): Doc => ({
  version: 1,
  meta: { name },
  symbols: {
    B: { kind: "runtime", default: 1, doc: "Batch size" },
    T: { kind: "runtime", default: 2048, doc: "Sequence length in tokens" },
  },
  graph: { nodes: [], edges: [] },
});

export interface FileStoreOptions {
  /** Directory `listFiles` scans and relative save paths resolve against. */
  root?: string;
  /** How deep `listFiles` walks. */
  depth?: number;
}

export class FileStore implements DocumentStore {
  private readonly entries = new Map<string, Entry>();
  private readonly listeners = new Set<StoreListener>();
  private nextDesign = 1;
  private nextCheckpoint = 1;
  readonly root: string;
  private readonly depth: number;

  constructor(options: FileStoreOptions = {}) {
    this.root = resolve(options.root ?? process.cwd());
    this.depth = options.depth ?? 3;
  }

  // -- watching ------------------------------------------------------------

  subscribe(listener: StoreListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * A listener that throws must not take the edit down with it. The store's
   * job is the document; a mirror that has fallen over is the mirror's problem,
   * and it is reported where a server's diagnostics go.
   */
  private emit(change: StoreChange): void {
    for (const listener of this.listeners) {
      try {
        listener(change);
      } catch (e) {
        process.stderr.write(`tensorcad: store listener failed: ${(e as Error).message}\n`);
      }
    }
  }

  // -- reads ---------------------------------------------------------------

  list(): DesignSummary[] {
    return [...this.entries.values()]
      .map((e) => summaryOf(e.record))
      .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  }

  async listFiles(): Promise<string[]> {
    const out: string[] = [];
    const skip = new Set(["node_modules", ".git", "dist", "build", ".venv", "__pycache__"]);

    const walk = async (dir: string, left: number): Promise<void> => {
      let items;
      try {
        items = await readdir(dir, { withFileTypes: true });
      } catch {
        return;
      }
      for (const item of items) {
        const full = join(dir, item.name);
        if (item.isDirectory()) {
          if (left > 0 && !skip.has(item.name) && !item.name.startsWith(".")) await walk(full, left - 1);
        } else if (item.name.endsWith(".tensorcad.json")) {
          out.push(full);
        }
      }
    };

    await walk(this.root, this.depth);
    return out.sort();
  }

  get(id: string): DesignRecord {
    const entry = this.entries.get(id);
    if (!entry) throw new UnknownDesignError(id, [...this.entries.keys()]);
    return entry.record;
  }

  // -- creation ------------------------------------------------------------

  create(options: NewDesignOptions): DesignRecord {
    let doc: Doc;
    let source: DesignRecord["source"];

    if (options.preset) {
      if (!PRESET_NAMES.includes(options.preset)) {
        throw new Error(`Unknown preset "${options.preset}". Available: ${PRESET_NAMES.join(", ")}`);
      }
      doc = getPreset(options.preset);
      source = "preset";
    } else {
      doc = EMPTY_DOC(options.name ?? "untitled");
      source = "empty";
    }
    if (options.name) doc.meta.name = options.name;

    return this.register(doc, source, undefined, true);
  }

  adopt(doc: Doc): DesignRecord {
    return this.register(doc, "derived", undefined, true);
  }

  async open(path: string): Promise<DesignRecord> {
    const full = this.resolvePath(path);

    for (const entry of this.entries.values()) {
      if (entry.record.path === full) return entry.record;
    }

    const text = await readFile(full, "utf8");
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      throw new Error(`${full} is not valid JSON: ${(e as Error).message}`);
    }
    const doc = parsed as Doc;
    if (!doc || typeof doc !== "object" || !doc.graph || !doc.meta) {
      throw new Error(`${full} does not look like a design document: it has no "meta" and "graph".`);
    }
    doc.symbols ??= {};
    return this.register(doc, "file", full, false);
  }

  private register(doc: Doc, source: DesignRecord["source"], path: string | undefined, dirty: boolean): DesignRecord {
    const now = new Date().toISOString();
    const id = `dsn_${this.nextDesign++}`;
    const record: DesignRecord = {
      design_id: id,
      name: doc.meta.name,
      revision: 1,
      source,
      dirty,
      created_at: now,
      updated_at: now,
      doc,
    };
    if (path) record.path = path;
    this.entries.set(id, { record, log: [], checkpoints: new Map() });
    this.emit({ kind: "registered", record });
    return record;
  }

  // -- mutation ------------------------------------------------------------

  apply(id: string, ops: Op[], expectedRevision?: number): ApplyOutcome {
    const entry = this.entry(id);
    const { record } = entry;
    if (expectedRevision !== undefined && expectedRevision !== record.revision) {
      throw new RevisionConflictError(id, expectedRevision, record.revision);
    }

    const before = structuredClone(record.doc);
    const { doc, applied } = applyOps(record.doc, ops);

    const previousRevision = record.revision;
    entry.log.push({ revision: previousRevision, at: new Date().toISOString(), ops, before });
    record.doc = doc;
    record.name = doc.meta.name;
    record.revision = previousRevision + 1;
    record.dirty = true;
    record.updated_at = new Date().toISOString();

    this.emit({ kind: "applied", record, ops });
    return { record, applied, previousRevision };
  }

  replace(id: string, doc: Doc, expectedRevision?: number): ApplyOutcome {
    const entry = this.entry(id);
    const { record } = entry;
    if (expectedRevision !== undefined && expectedRevision !== record.revision) {
      throw new RevisionConflictError(id, expectedRevision, record.revision);
    }

    const previousRevision = record.revision;
    // No operations to log, but the document that was there is what undo
    // restores, and that is the half that matters.
    entry.log.push({ revision: previousRevision, at: new Date().toISOString(), ops: [], before: record.doc });
    record.doc = doc;
    record.name = doc.meta.name;
    record.revision = previousRevision + 1;
    record.dirty = true;
    record.updated_at = new Date().toISOString();

    this.emit({ kind: "replaced", record });
    return { record, applied: ["replaced the document"], previousRevision };
  }

  async save(id: string, path?: string): Promise<{ record: DesignRecord; path: string; bytes: number }> {
    const record = this.get(id);
    const target = path
      ? this.resolvePath(path)
      : (record.path ?? join(this.root, `${slug(record.name)}.tensorcad.json`));

    const text = `${JSON.stringify(record.doc, null, 2)}\n`;
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, text, "utf8");

    record.path = target;
    record.dirty = false;
    record.updated_at = new Date().toISOString();

    this.emit({ kind: "saved", record });
    return { record, path: target, bytes: Buffer.byteLength(text, "utf8") };
  }

  // -- history -------------------------------------------------------------

  checkpoint(id: string, label?: string): CheckpointInfo {
    const entry = this.entry(id);
    const info: CheckpointInfo = {
      checkpoint_id: `ckpt_${this.nextCheckpoint++}`,
      label: label ?? `revision ${entry.record.revision}`,
      revision: entry.record.revision,
      created_at: new Date().toISOString(),
    };
    entry.checkpoints.set(info.checkpoint_id, { ...info, doc: structuredClone(entry.record.doc) });
    return info;
  }

  checkpoints(id: string): CheckpointInfo[] {
    const entry = this.entry(id);
    return [...entry.checkpoints.values()].map(({ doc: _doc, ...info }) => info);
  }

  restore(id: string, checkpointId?: string): { record: DesignRecord; restoredFrom: string } {
    const entry = this.entry(id);
    const { record } = entry;

    let doc: Doc;
    let restoredFrom: string;

    if (checkpointId) {
      const saved = entry.checkpoints.get(checkpointId);
      if (!saved) {
        const known = [...entry.checkpoints.keys()];
        throw new Error(
          `Unknown checkpoint "${checkpointId}" for ${id}. ` +
            `Checkpoints: ${known.length > 0 ? known.join(", ") : "(none)"}.`,
        );
      }
      doc = structuredClone(saved.doc);
      restoredFrom = `checkpoint ${checkpointId} (${saved.label})`;
    } else {
      const last = entry.log.pop();
      if (!last) throw new Error(`Design ${id} has no edits to undo and no checkpoint was named.`);
      doc = last.before;
      restoredFrom = `undo of ${last.ops.length} op${last.ops.length === 1 ? "" : "s"} at revision ${last.revision}`;
    }

    record.doc = doc;
    record.name = doc.meta.name;
    record.revision += 1;
    record.dirty = true;
    record.updated_at = new Date().toISOString();

    this.emit({ kind: "restored", record });
    return { record, restoredFrom };
  }

  // -- helpers -------------------------------------------------------------

  private entry(id: string): Entry {
    const entry = this.entries.get(id);
    if (!entry) throw new UnknownDesignError(id, [...this.entries.keys()]);
    return entry;
  }

  private resolvePath(path: string): string {
    const full = isAbsolute(path) ? path : resolve(this.root, path);
    if (extname(full) === "") return `${full}.tensorcad.json`;
    return full;
  }

  /** Whether a path exists, used by tools that want a friendlier message. */
  static async exists(path: string): Promise<boolean> {
    try {
      await stat(path);
      return true;
    } catch {
      return false;
    }
  }
}

function summaryOf(record: DesignRecord): DesignSummary {
  const { doc: _doc, ...summary } = record;
  return summary;
}

function slug(name: string): string {
  return name.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "design";
}
