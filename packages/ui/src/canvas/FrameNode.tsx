/**
 * A frame: a container drawn around what it contains.
 *
 * This is the shape every architecture figure uses for a repeated block — a
 * rounded rectangle enclosing the sublayers, its name along the top and the
 * stack count written on the edge with a bracket, as those figures write
 * `32 ×`. The body is a tint rather than a fill so the parts inside stay the
 * brightest things on the sheet.
 *
 * It carries no pins. Once a container is open, every wire runs to the parts
 * inside it, so the frame's own ports have been short-circuited away by
 * `unfold` and there is nothing to attach to.
 */

import { memo } from "react";
import { type Node, type NodeProps } from "@xyflow/react";
import { partColor } from "./blocks.js";
import type { Severity } from "../state/derive.js";
import { formatCount } from "@tensor-cad/engine";

export interface FrameNodeData extends Record<string, unknown> {
  path: string;
  label: string;
  type: string;
  category: string;
  /** Parameters in everything the frame contains. */
  params: number;
  /** Stack count, written on the edge. */
  multiplier: number | null;
  severity: Severity | null;
  locked: boolean;
  /** Nesting level, so an inner frame reads as inside an outer one. */
  depth: number;
}

export type FrameFlowNode = Node<FrameNodeData, "frame">;

function FrameNodeView({ data, selected }: NodeProps<FrameFlowNode>): React.ReactElement {
  const part = partColor(data.category);
  return (
    <div
      className={
        "frame" +
        (selected ? " frame--selected" : "") +
        (data.locked ? " frame--locked" : "")
      }
      // Only the outline colour is taken from the category: a frame tints itself
      // from that, so a dark-filled leaf category cannot darken a frame.
      style={{ ["--edge" as string]: part.edge, ["--depth" as string]: data.depth }}
    >
      <div className="frame__caption">
        <span className="frame__name">{data.label}</span>
        <span className="frame__type">{data.type}</span>
        {data.severity && (
          <span className={`mark mark--${data.severity}`} title={`this block has ${data.severity}s`}>
            {data.severity === "error" ? "●" : "▲"}
          </span>
        )}
        {data.locked && (
          <span className="mark mark--lock" title="locked: position is fixed">
            &#128274;
          </span>
        )}
        {data.params > 0 && (
          <span className="frame__params" title={`${data.params.toLocaleString("en-US")} parameters`}>
            {formatCount(data.params)}
          </span>
        )}
      </div>

      {/* The stack count, on a bracket down the left edge, as the figures draw it. */}
      {data.multiplier && data.multiplier > 1 && (
        <div className="frame__stack" title={`stacked ${data.multiplier} times`}>
          <span className="frame__bracket" aria-hidden />
          <span className="frame__count">{data.multiplier}&times;</span>
        </div>
      )}
    </div>
  );
}

export default memo(FrameNodeView);
