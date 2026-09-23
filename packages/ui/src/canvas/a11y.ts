/**
 * What the sheet says to somebody who cannot see it.
 *
 * React Flow makes every block and wire focusable, and then — with nothing to
 * go on — announces a block as "node" and a wire as nothing at all, or reads
 * out every word on the card in the order the markup happens to hold them.
 * These are the one sentence each that a sighted reader takes from the
 * drawing at a glance: what the block is, what comes out of it, whether the
 * checks object to it, and whether there is anything inside to go into.
 *
 * Kept apart from the canvas so the words can be tested without a browser.
 */

import { formatCount } from "@tensor-cad/engine";
import type { BlockNodeData } from "./BlockNode.js";
import type { FrameNodeData } from "./FrameNode.js";

const leaf = (path: string): string => path.slice(path.lastIndexOf("/") + 1);

function findingsOf(d: Pick<BlockNodeData, "findings">): string | null {
  const errors = d.findings.filter((f) => f.severity === "error").length;
  const warnings = d.findings.filter((f) => f.severity === "warning").length;
  const parts = [
    errors ? `${errors} error${errors === 1 ? "" : "s"}` : null,
    warnings ? `${warnings} warning${warnings === 1 ? "" : "s"}` : null,
  ].filter(Boolean);
  return parts.length ? parts.join(" and ") : null;
}

const SEVERITY_WORDS = { error: "has errors", warning: "has warnings", info: "has notes" } as const;

/**
 * `embed, token embedding, 144 parameters, out B × T × 48, 1 warning, press Enter to open`
 *
 * `opens` is whether there is anything inside, which is not the same as the
 * card's own `drillable`: an unfolded drawing opens a block in place rather
 * than by going into it, and Enter does that too.
 */
export function blockLabel(d: BlockNodeData, opens: boolean = d.drillable): string {
  // The identifier says nothing more when it is the type's own name, which is
  // the common case for a block nobody renamed.
  const name = d.label === d.typeName ? d.label : `${d.label}, ${d.typeName}`;
  const outs = d.outPorts
    .filter((p) => p.shape)
    .map((p) => (d.outPorts.length > 1 ? `${p.name} ${p.shape}` : p.shape));
  return [
    name,
    d.params > 0 ? `${formatCount(d.params)} parameters` : null,
    outs.length ? `out ${outs.join(", ")}` : null,
    // A block's own findings when it has them; otherwise the marker the
    // drawing shows, which can come from something folded inside it.
    findingsOf(d) ?? (d.severity ? SEVERITY_WORDS[d.severity as keyof typeof SEVERITY_WORDS] ?? null : null),
    d.locked ? "locked" : null,
    opens ? "press Enter to open" : null,
  ]
    .filter(Boolean)
    .join(", ");
}

/** `layers, transformer block ×3, 84.8K parameters` */
export function frameLabel(d: FrameNodeData): string {
  // A stack's caption usually says its count already, as `Transformer block x3`.
  const counted = d.multiplier && d.multiplier > 1 && !d.label.includes(String(d.multiplier));
  const what = counted ? `${d.typeName} ×${d.multiplier}` : d.typeName;
  return [
    d.label === d.typeName ? what : `${d.label}, ${what}`,
    d.params > 0 ? `${formatCount(d.params)} parameters` : null,
    d.severity ? (SEVERITY_WORDS[d.severity as keyof typeof SEVERITY_WORDS] ?? null) : null,
  ]
    .filter(Boolean)
    .join(", ");
}

/** `embed y to pos x, B × T × 48` */
export function wireLabel(source: string, sourcePort: string, target: string, targetPort: string, shape?: string): string {
  const from = `${leaf(source)} ${sourcePort}`;
  const to = `${leaf(target)} ${targetPort}`;
  return shape ? `${from} to ${to}, ${shape}` : `${from} to ${to}`;
}

/**
 * What React Flow says on its own, reworded for what this editor actually does.
 *
 * Its two node descriptions are named the other way round from when it uses
 * them: `keyboardDisabled` is the one it reads out while the keyboard *works*,
 * and says the arrow keys move a block. That is the one this editor needs.
 */
export const ARIA_LABELS = {
  "node.a11yDescription.keyboardDisabled":
    "Press Enter or Space to select a block, then the arrow keys to move it. Enter again opens it, Escape comes back out, Delete removes it.",
  "node.a11yDescription.default":
    "Press Enter or Space to select a block. Enter again opens it, Escape comes back out, Delete removes it.",
  "node.a11yDescription.ariaLiveMessage": ({ direction }: { direction: string; x: number; y: number }) =>
    `Moved the block ${direction}.`,
  "edge.a11yDescription.default": "A wire. Press Enter or Space to select the tensor it carries, Delete to remove it.",
};
