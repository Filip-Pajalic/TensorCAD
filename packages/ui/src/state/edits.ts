/**
 * An edit, as a thing rather than as a closure.
 *
 * This is the difference between an undo stack and a feature timeline. A stack
 * of documents can answer "what did it look like before"; it cannot answer
 * "what would it look like *without* the third one", because by the time the
 * third edit is a document the operation that made it is gone.
 *
 * So every edit is a value: a kind and its arguments, all of them JSON, and
 * `applyEdit` is the only thing that knows how to perform one. The history is
 * then a base document and a list of these, and the document on screen is what
 * you get by folding the list over the base. Suppress one and the fold skips
 * it; everything after replays on top of what is left.
 *
 * The fold itself lives in the store rather than here, because the one that
 * matters replays from the first step whose meaning changed rather than from
 * the base — an edit at the end must not cost a hundred applies. A second,
 * simpler fold here would be a second answer to the same question, and the two
 * would drift.
 *
 * ## Why every argument is a value and not a path into the old document
 *
 * `addNode` carries the node it adds rather than a reference to one, and
 * `captureConfiguration` carries the document it captured. An edit that
 * referred to state outside itself would replay differently depending on what
 * came before it, which is exactly the property a timeline cannot have.
 *
 * ## What happens when one cannot replay
 *
 * Suppressing the edit that added a block leaves the edit that wired it with
 * nothing to wire. That is not a bug to be prevented — it is the ordinary
 * consequence of taking a step out of the middle, and every parametric CAD
 * tool has the same condition. So `applyEdit` reports failure rather than
 * throwing, the fold skips a failed edit and carries on, and the panel says
 * which ones failed and why. The alternative — refusing to suppress anything
 * another edit might depend on — would refuse almost everything.
 */

import * as ops from "./ops.js";
import type { Doc, NodeDef, ParamValue, RuleSeverity, SymbolDef } from "@tensorcad/engine";
import type { Segments } from "./ops.js";

export type Edit =
  | { kind: "addNode"; parent: Segments; node: NodeDef; xy?: [number, number] }
  | { kind: "removeNode"; path: string }
  | { kind: "setParam"; path: string; key: string; value: ParamValue | undefined }
  | { kind: "connect"; parent: Segments; from: string; to: string }
  | { kind: "disconnect"; parent: Segments; from: string; to: string }
  | { kind: "reconnect"; parent: Segments; from: string; to: string; next: { from: string; to: string } }
  | { kind: "setRuleSeverity"; rule: string; severity: RuleSeverity | undefined }
  | { kind: "setSymbol"; name: string; def: SymbolDef | undefined }
  | { kind: "setActiveConfiguration"; name: string | null }
  | { kind: "captureConfiguration"; name: string; doc?: string }
  | { kind: "removeConfiguration"; name: string }
  | { kind: "renameSymbol"; from: string; to: string }
  | { kind: "moveNode"; path: string; xy: [number, number] }
  | { kind: "moveNodes"; moves: { path: string; xy: [number, number] }[] }
  | { kind: "renameNode"; path: string; label: string | undefined }
  | { kind: "setNodeId"; path: string; id: string }
  | { kind: "setMetaName"; name: string }
  /**
   * A whole document, in place of an operation.
   *
   * The live agent bridge hands over documents, not operations: what the agent
   * did is expressible as operations but what arrives is the result. Carrying
   * it as an edit keeps it in the timeline — suppressible, jumpable, and
   * labelled — rather than making it a second mechanism that resets one.
   */
  | { kind: "replaceDoc"; doc: Doc };

/** An edit in the timeline: what it is, what it is called, and whether it is on. */
export interface Step {
  edit: Edit;
  label: string;
  /** Switched off by hand. It stays in the list and is skipped by the fold. */
  suppressed?: boolean;
}

/** Why an edit did nothing when it was replayed. */
export interface Failure {
  /** Index into the step list. */
  at: number;
  reason: string;
}

/**
 * Perform one edit.
 *
 * Returns the same document it was given when the edit could not be made,
 * which the fold reads as failure — every `ops` function already returns its
 * argument unchanged when there is nothing to do, so this is the convention
 * the whole module is written in rather than a new one.
 */
export function applyEdit(doc: Doc, edit: Edit): Doc {
  const at = (path: string): Segments => ops.segmentsOf(path);
  switch (edit.kind) {
    case "addNode":
      return ops.addNode(doc, edit.parent, edit.node, edit.xy);
    case "removeNode":
      return ops.removeNode(doc, at(edit.path));
    case "setParam":
      return ops.setParam(doc, at(edit.path), edit.key, edit.value);
    case "connect":
      return ops.connect(doc, edit.parent, edit.from, edit.to);
    case "disconnect":
      return ops.disconnect(doc, edit.parent, edit.from, edit.to);
    case "reconnect":
      return ops.connect(
        ops.disconnect(doc, edit.parent, edit.from, edit.to),
        edit.parent,
        edit.next.from,
        edit.next.to,
      );
    case "setRuleSeverity":
      return ops.setRuleSeverity(doc, edit.rule, edit.severity);
    case "setSymbol":
      return ops.setSymbolInConfiguration(doc, edit.name, edit.def);
    case "setActiveConfiguration":
      return ops.setActiveConfiguration(doc, edit.name);
    case "captureConfiguration":
      return ops.captureConfiguration(doc, edit.name, edit.doc);
    case "removeConfiguration":
      return ops.removeConfiguration(doc, edit.name);
    case "renameSymbol":
      return ops.renameSymbol(doc, edit.from, edit.to);
    case "moveNode":
      return ops.moveNode(doc, at(edit.path), edit.xy);
    case "moveNodes":
      return ops.moveNodes(doc, edit.moves);
    case "renameNode":
      return ops.renameNode(doc, at(edit.path), edit.label);
    case "setNodeId":
      return ops.setNodeId(doc, at(edit.path), edit.id);
    case "setMetaName":
      return ops.setMetaName(doc, edit.name);
    case "replaceDoc":
      return edit.doc;
  }
}

/**
 * What an edit could not find, said in the words of the drawing.
 *
 * "addNode returned the same document" is true and useless. What the person
 * needs to know is that the block it was going to wire is not there any more,
 * because that tells them which earlier step they suppressed.
 */
export function reasonFor(edit: Edit): string {
  switch (edit.kind) {
    case "removeNode":
    case "setParam":
    case "moveNode":
    case "renameNode":
    case "setNodeId":
      return `there is no ${edit.path}`;
    case "connect":
    case "disconnect":
      return `${edit.from} or ${edit.to} is not there`;
    case "reconnect":
      return `${edit.next.from} or ${edit.next.to} is not there`;
    case "renameSymbol":
      return `there is no symbol ${edit.from}`;
    case "removeConfiguration":
    case "setActiveConfiguration":
      return `there is no configuration ${String(edit.name)}`;
    case "moveNodes":
      return "none of those blocks are there";
    default:
      return "it had nothing to change";
  }
}

/**
 * What produced the document at the mark.
 *
 * At zero it is how the design arrived — opened, loaded, started empty —
 * which is a real row in the list and not an absence of one.
 */
export function labelAt(steps: Step[], at: number, baseLabel: string): string {
  return at === 0 ? baseLabel : (steps[at - 1]?.label ?? baseLabel);
}
