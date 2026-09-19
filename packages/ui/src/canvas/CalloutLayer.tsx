/**
 * Draws the callouts in flow coordinates, each on a dashed leader line to the
 * part it annotates, the way a published figure does.
 *
 * They live inside the viewport transform so they pan and zoom with the
 * drawing, and they are not interactive: a note is something you read, not
 * something you click.
 *
 * Two things a figure does by hand and this has to do by arithmetic. A part
 * inside a frame is positioned relative to that frame, so the leader has to
 * start from the absolute position rather than the stored one. And notes on the
 * same side of a tall drawing collide, so they are pushed apart down the
 * column, which is what a draughtsman does with a stack of leader lines.
 */

import { memo, useMemo } from "react";
import { ViewportPortal, useNodes, type Node } from "@xyflow/react";
import type { Callout } from "./callouts.js";

const NOTE_WIDTH = 190;
/** Gap between the part's edge and the start of the leader line. */
const LEADER = 46;
/** Vertical room one note needs before the next may start. */
const NOTE_PITCH = 46;

interface Placed {
  callout: Callout;
  /** Where the leader touches the part. */
  anchorX: number;
  anchorY: number;
  /** Where the note sits, after collisions have been resolved. */
  noteY: number;
}

type AnyNode = Node<Record<string, unknown>>;

/** Absolute position of a node, summing the frames it sits inside. */
function absolute(node: AnyNode, byId: Map<string, AnyNode>): { x: number; y: number } {
  let x = node.position.x;
  let y = node.position.y;
  let parent = node.parentId;
  const guard = new Set<string>([node.id]);
  while (parent && !guard.has(parent)) {
    guard.add(parent);
    const up = byId.get(parent);
    if (!up) break;
    x += up.position.x;
    y += up.position.y;
    parent = up.parentId;
  }
  return { x, y };
}

function Note({ placed }: { placed: Placed }): React.ReactElement {
  const { callout, anchorX, anchorY, noteY } = placed;
  const onLeft = callout.side === "left";
  const tipX = onLeft ? anchorX - LEADER : anchorX + LEADER;
  const noteX = onLeft ? tipX - NOTE_WIDTH : tipX;

  // The leader runs from the part out to the note's own height, so a note that
  // was pushed down still points at the thing it describes.
  const left = Math.min(anchorX, tipX);
  const top = Math.min(anchorY, noteY);
  const height = Math.abs(noteY - anchorY) || 1;

  return (
    <>
      <svg
        className="callout__leader"
        style={{ position: "absolute", left, top, width: LEADER, height, overflow: "visible" }}
      >
        <polyline
          points={
            onLeft
              ? `${LEADER},${anchorY - top} ${LEADER / 2},${anchorY - top} ${LEADER / 2},${noteY - top} 0,${noteY - top}`
              : `0,${anchorY - top} ${LEADER / 2},${anchorY - top} ${LEADER / 2},${noteY - top} ${LEADER},${noteY - top}`
          }
          fill="none"
          stroke="var(--wire-label)"
          strokeWidth={1}
          strokeDasharray="3 2"
        />
        <circle
          cx={onLeft ? LEADER : 0}
          cy={anchorY - top}
          r={2}
          fill="var(--wire-label)"
        />
      </svg>

      <div
        className={`callout callout--${callout.side}`}
        style={{ position: "absolute", left: noteX, top: noteY, width: NOTE_WIDTH }}
      >
        {callout.lines.map((line, i) => (
          <div key={i} className={i === 0 ? "callout__lead" : "callout__sub"}>
            {line}
          </div>
        ))}
      </div>
    </>
  );
}

function CalloutsView({ callouts }: { callouts: Callout[] }): React.ReactElement {
  const nodes = useNodes();
  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n as AnyNode])), [nodes]);

  const placed = useMemo<Placed[]>(() => {
    const seen = new Set<string>();
    const rows: Placed[] = [];
    for (const callout of callouts) {
      const node = byId.get(callout.path);
      if (!node) continue;
      // The same sentence twice is worse than saying it once: two parts can
      // legitimately report the same number.
      const text = callout.lines.join("|");
      if (seen.has(text)) continue;
      seen.add(text);

      const pos = absolute(node, byId);
      const width = node.measured?.width ?? (typeof node.width === "number" ? node.width : 216);
      const height = node.measured?.height ?? (typeof node.height === "number" ? node.height : 74);
      const anchorY = pos.y + height / 2;
      rows.push({
        callout,
        anchorX: callout.side === "left" ? pos.x : pos.x + width,
        anchorY,
        noteY: anchorY,
      });
    }

    // Push overlapping notes down, one side at a time.
    for (const side of ["left", "right"] as const) {
      const column = rows.filter((r) => r.callout.side === side).sort((a, b) => a.noteY - b.noteY);
      let floor = -Infinity;
      for (const row of column) {
        row.noteY = Math.max(row.noteY, floor);
        floor = row.noteY + NOTE_PITCH;
      }
    }
    return rows;
  }, [callouts, byId]);

  return (
    <ViewportPortal>
      {placed.map((p) => (
        <Note key={p.callout.path} placed={p} />
      ))}
    </ViewportPortal>
  );
}

export default memo(CalloutsView);
