/**
 * A wire.
 *
 * React Flow's built-in edges are fine lines with a fixed invisible band around
 * them for hit testing. Two things about that band are wrong for a CAD:
 *
 *   - It is measured in flow units, so it is drawn inside the viewport's
 *     transform and shrinks with the zoom. At the minimum zoom the default
 *     twenty units is six screen pixels, which is why picking a wire on a large
 *     sheet felt like threading a needle. Every drafting program keeps its pick
 *     aperture constant in *screen* space — you aim with the cursor, not with
 *     the drawing — so the band is divided by the zoom here.
 *
 *   - It gives no feedback before the click. eeschema highlights what the
 *     cursor is over before you commit to it, and shows the ends of a wire as
 *     grips once it is picked, because otherwise nothing tells you the ends can
 *     be dragged somewhere else.
 *
 * The path itself is `step`: true right angles, which is how a schematic draws
 * a net. Curves read as a dataflow toy.
 */

import { memo } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getSmoothStepPath,
  useStore,
  type EdgeProps,
} from "@xyflow/react";

/**
 * The pick aperture, in screen pixels.
 *
 * Eight either side of the line. Wide enough to hit without aiming, narrow
 * enough that two parallel wires a grid square apart stay separable.
 */
const APERTURE = 16;

/** Below this the grips are smaller than the wire is long; drawing them lies. */
const GRIP_MIN_ZOOM = 0.55;

export interface WireEdgeData extends Record<string, unknown> {
  from?: string;
  to?: string;
  /** Endpoint the whole net is identified by, for highlighting it as one. */
  net?: string;
  /** True while the net this belongs to is under the cursor. */
  lit?: boolean;
}

function WireEdgeView({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  label,
  labelShowBg,
  markerEnd,
  markerStart,
  selected,
  data,
  style,
}: EdgeProps): React.ReactElement {
  const zoom = useStore((s) => s.transform[2]);

  const [path, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    // Zero is `step`: true right angles rather than a rounded corner.
    borderRadius: 0,
  });

  // The aperture is in flow units because that is the space the path is drawn
  // in, so the zoom has to be divided out to keep it constant on screen.
  const interactionWidth = APERTURE / Math.max(zoom, 0.05);
  const grips = (selected || data?.lit) && zoom >= GRIP_MIN_ZOOM;

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={style}
        markerEnd={markerEnd}
        markerStart={markerStart}
        interactionWidth={interactionWidth}
        label={label}
        labelX={labelX}
        labelY={labelY}
        labelShowBg={labelShowBg}
      />
      {grips && (
        <EdgeLabelRenderer>
          {/* The two ends, as grips. They are not drag targets themselves —
              React Flow's own reconnect handling owns a radius around each
              endpoint — they are the thing that says the ends can be moved,
              which nothing on the sheet said before. */}
          <div
            className="wire-grip"
            style={{ transform: `translate(-50%, -50%) translate(${sourceX}px, ${sourceY}px)` }}
          />
          <div
            className="wire-grip"
            style={{ transform: `translate(-50%, -50%) translate(${targetX}px, ${targetY}px)` }}
          />
        </EdgeLabelRenderer>
      )}
    </>
  );
}

export default memo(WireEdgeView);
