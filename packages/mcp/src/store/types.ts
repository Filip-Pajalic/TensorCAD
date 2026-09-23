/**
 * The document store the tools talk to.
 *
 * Designs are addressed by a server-minted `design_id` rather than by a
 * session, which is what the 2026-07-28 spec asks for: state crosses calls as
 * an explicit handle in the arguments, never as an implicit connection.
 *
 * The live editor bridge is *not* a second store: it observes this one through
 * `subscribe` and mirrors it to a running editor. Two stores could disagree
 * about what a design is; one store with a listener cannot.
 *
 * ## Everything returns a promise, including what a local store answers at once
 *
 * `FileStore` holds its designs in memory and could answer most of this
 * synchronously. A store behind a database cannot, and a caller that has to
 * know which kind it is holding is a caller that breaks when the kind changes.
 * So the signatures say promise throughout and the local implementation returns
 * ones that are already resolved.
 *
 * That costs `FileStore` nothing it cannot afford, and it does not cost the
 * bridge its contract either — see `subscribe`.
 */

import type { Op } from "../ops.js";
import type { Doc } from "@tensor-cad/engine";

/**
 * Where a design in this session came from. `derived` is one the engine
 * produced from another — scaled, or read out of a Hugging Face config — which
 * has no file behind it and is not a preset.
 */
/**
 * Where a design came from. One list, which the type and the output schema are
 * both built from: they were two once, and the schema's copy never learned
 * "derived" — so any session holding a design made by scale, mup or import
 * could no longer list its designs.
 */
export const DESIGN_SOURCES = ["preset", "file", "empty", "derived"] as const;
export type DesignSource = (typeof DESIGN_SOURCES)[number];

export interface DesignSummary {
  design_id: string;
  name: string;
  revision: number;
  /** Absolute path this design was opened from or last saved to. */
  path?: string;
  source: DesignSource;
  /** True when there are edits that `tensorcad_save_design` has not written yet. */
  dirty: boolean;
  created_at: string;
  updated_at: string;
}

export interface DesignRecord extends DesignSummary {
  doc: Doc;
}

export interface CheckpointInfo {
  checkpoint_id: string;
  label: string;
  revision: number;
  created_at: string;
}

export interface ApplyOutcome {
  record: DesignRecord;
  applied: string[];
  previousRevision: number;
}

export interface NewDesignOptions {
  preset?: string;
  name?: string;
}

/** What happened to a design, for anybody mirroring the store. */
export interface StoreChange {
  kind: "registered" | "applied" | "replaced" | "saved" | "restored";
  record: DesignRecord;
  /** The operations, when there were any — `applied` alone carries them. */
  ops?: Op[];
}

export type StoreListener = (change: StoreChange) => void;

export interface DocumentStore {
  /**
   * Watch every change to every design. Returns the function that stops
   * watching.
   *
   * Listeners are called *synchronously*, inside the call that changed
   * something, which is what lets the bridge attribute a change to the
   * connection that caused it without threading an origin through every
   * signature.
   *
   * Still true now that the mutating calls return promises, and it is worth
   * saying why: an `async` method runs its body synchronously until its first
   * `await`, and `FileStore` has none before it announces the change. So the
   * listener fires while the caller is still inside the call, exactly as
   * before, and only the settling of the promise happens later. A store that
   * *did* await before announcing would break the bridge's attribution rather
   * than its types, which is the kind of failure worth naming in advance.
   */
  subscribe(listener: StoreListener): () => void;

  /** Designs this server has open, newest first. */
  list(): Promise<DesignSummary[]>;
  /** `.tensorcad.json` files near the server's root that could be opened. */
  listFiles(): Promise<string[]>;

  create(options: NewDesignOptions): Promise<DesignRecord>;
  /**
   * Take a document the engine produced — scaled, imported, derived — as a new
   * design in this session.
   *
   * Separate from `create`, which builds one from a preset or from nothing:
   * these arrive whole and there is nothing to build.
   */
  adopt(doc: Doc): Promise<DesignRecord>;
  open(path: string): Promise<DesignRecord>;
  get(id: string): Promise<DesignRecord>;

  apply(id: string, ops: Op[], expectedRevision?: number): Promise<ApplyOutcome>;
  /**
   * The whole document, in place of a list of operations.
   *
   * The editor's edits are not all expressible as the eight operations a tool
   * call can send — a block moved on the sheet, a definition, a configuration —
   * so what the running editor sends back is the document it now has. It is
   * otherwise an `apply`: the revision is checked, the previous document goes
   * on the undo log, and the change is announced.
   */
  replace(id: string, doc: Doc, expectedRevision?: number): Promise<ApplyOutcome>;
  save(id: string, path?: string): Promise<{ record: DesignRecord; path: string; bytes: number }>;

  checkpoint(id: string, label?: string): Promise<CheckpointInfo>;
  checkpoints(id: string): Promise<CheckpointInfo[]>;
  /** Without a checkpoint id this undoes the most recent `apply`. */
  restore(id: string, checkpointId?: string): Promise<{ record: DesignRecord; restoredFrom: string }>;
}

/** Thrown when a mutating call names a revision that is no longer current. */
export class RevisionConflictError extends Error {
  constructor(
    public readonly designId: string,
    public readonly expected: number,
    public readonly actual: number,
  ) {
    super(
      `Design ${designId} is at revision ${actual}, not the expected ${expected}. ` +
        `Re-read it with tensorcad_get_design and rebuild the edit on the current document.`,
    );
    this.name = "RevisionConflictError";
  }
}

/** Thrown when a `design_id` is unknown. */
export class UnknownDesignError extends Error {
  constructor(id: string, known: string[]) {
    super(
      `Unknown design_id "${id}". Open designs: ${known.length > 0 ? known.join(", ") : "(none)"}. ` +
        `Use tensorcad_new_design or tensorcad_open_design first.`,
    );
    this.name = "UnknownDesignError";
  }
}
