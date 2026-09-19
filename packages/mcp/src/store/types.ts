/**
 * The document store the tools talk to.
 *
 * Designs are addressed by a server-minted `design_id` rather than by a
 * session, which is what the 2026-07-28 spec asks for: state crosses calls as
 * an explicit handle in the arguments, never as an implicit connection.
 *
 * `FileStore` is the headless implementation. A `LiveStore` that proxies to a
 * running editor over a local WebSocket is the other half of the design and is
 * not implemented yet; see the TODO in `file-store.ts`.
 */

import type { Doc } from "@tensorcad/core";
import type { Op } from "../ops.js";

export type DesignSource = "preset" | "file" | "empty";

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

export interface DocumentStore {
  /** Designs this server has open, newest first. */
  list(): DesignSummary[];
  /** `.tensorcad.json` files near the server's root that could be opened. */
  listFiles(): Promise<string[]>;

  create(options: NewDesignOptions): DesignRecord;
  open(path: string): Promise<DesignRecord>;
  get(id: string): DesignRecord;

  apply(id: string, ops: Op[], expectedRevision?: number): ApplyOutcome;
  save(id: string, path?: string): Promise<{ record: DesignRecord; path: string; bytes: number }>;

  checkpoint(id: string, label?: string): CheckpointInfo;
  checkpoints(id: string): CheckpointInfo[];
  /** Without a checkpoint id this undoes the most recent `apply`. */
  restore(id: string, checkpointId?: string): { record: DesignRecord; restoredFrom: string };
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
