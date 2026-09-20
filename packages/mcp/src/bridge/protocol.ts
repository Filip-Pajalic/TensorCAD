/**
 * What the agent and the editor say to each other.
 *
 * The bridge carries *documents*, not just operations. Forwarding the op
 * stream alone would make the canvas replay every edit the agent makes, which
 * sounds elegant until the two ends disagree about what an op means and
 * nothing notices: the editor would drift, silently, and the drawing would
 * stop being the design. So every change carries the whole document as the
 * truth, with the operations beside it as the *account* of what changed —
 * which is what a person watching wants to read.
 *
 * A document is not large next to what a browser already holds, and the engine
 * that would have to be asked to reconcile a divergence is the expensive part.
 */

import type { Doc } from "@tensorcad/engine";
import type { Op } from "../ops.js";
import type { DesignSummary } from "../store/types.js";

/**
 * Bumped when a message changes shape. The editor refuses a bridge whose
 * protocol it does not know rather than half-understanding it, because a
 * canvas that mirrors *some* of an edit is worse than one that mirrors none.
 */
export const BRIDGE_PROTOCOL = 1;

/** Why a design arrived. `requested` answers an `attach`; `published` answers a `publish`. */
export type ChangeReason =
  | "registered"
  | "applied"
  | "replaced"
  | "saved"
  | "restored"
  | "published"
  | "requested";

export interface HelloMessage {
  type: "hello";
  protocol: number;
  server: string;
  version: string;
  /** The directory the agent's relative paths resolve against, for the editor to show. */
  root: string;
  designs: DesignSummary[];
}

/** A design as it now stands. The one message that carries a document. */
export interface DesignMessage {
  type: "design";
  reason: ChangeReason;
  design: DesignSummary;
  doc: Doc;
  /** Present when `reason` is `applied`: what the agent did, in its own words. */
  ops?: Op[];
}

export interface ErrorMessage {
  type: "error";
  message: string;
  /** The message this is a reply to, when it was a reply to one. */
  about?: ClientMessage["type"];
}

export type ServerMessage = HelloMessage | DesignMessage | ErrorMessage;

/** "This is what I have open." The bridge takes it as a new design. */
export interface PublishMessage {
  type: "publish";
  doc: Doc;
}

/**
 * "I had this open before." Sent instead of `publish` after a reconnect, so a
 * dropped socket does not leave the agent looking at two copies of one design.
 * Answered with an error when the id is unknown — a restarted server has
 * forgotten it — and the editor then publishes.
 */
export interface AttachMessage {
  type: "attach";
  design_id: string;
}

/** The human edited, in a way that is one of the agent's own operations. */
export interface OpsMessage {
  type: "ops";
  design_id: string;
  /** Rejected when it is not the current revision, exactly as `tensorcad_apply_ops` is. */
  revision?: number;
  ops: Op[];
}

/**
 * The human edited, generally.
 *
 * Most of what an editor does is not one of the eight operations — a block
 * moved on the sheet, a definition written, a configuration switched — so the
 * editor's ordinary way of saying what it has is to say the whole thing. It is
 * an `ops` message in every other respect: revision-checked, undoable, and
 * announced to the agent.
 */
export interface ReplaceMessage {
  type: "replace";
  design_id: string;
  revision?: number;
  doc: Doc;
}

export type ClientMessage = PublishMessage | AttachMessage | OpsMessage | ReplaceMessage;

/** What `~/.tensorcad/session.json` holds while a bridge is listening. */
export interface SessionFile {
  version: number;
  protocol: number;
  port: number;
  token: string;
  pid: number;
  root: string;
  started_at: string;
}
