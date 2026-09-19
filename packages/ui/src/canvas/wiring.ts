/**
 * Where a wire leaves a part, and what it looks like.
 *
 * A schematic does not route every net out of the bottom of a symbol. It leaves
 * by whichever side faces where it is going, and some pins only ever leave by
 * one side because that is what the symbol means — a residual sum takes its
 * bypass from the side, never from above, which is exactly how every
 * transformer figure draws it.
 *
 * So each port carries a handle on all four sides and the side is chosen from
 * the geometry once the layout is known, with a per-block override where the
 * block has an opinion. React Flow then renders the edge normally; nothing here
 * has to reimplement edge routing.
 */

export type Side = "top" | "right" | "bottom" | "left";

export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

export const SIDES: Side[] = ["top", "right", "bottom", "left"];

/** Handle ids carry the direction, the port and the side. */
export const handleId = (dir: "i" | "o", port: string, side: Side): string =>
  `${dir}:${port}@${side}`;

/** The port an edge endpoint refers to, whichever side it came out of. */
export const portOfHandle = (handle: string | null | undefined): string => {
  if (!handle) return "";
  const body = handle.slice(2);
  const at = body.lastIndexOf("@");
  return at === -1 ? body : body.slice(0, at);
};

export const sideOfHandle = (handle: string | null | undefined): Side | null => {
  if (!handle) return null;
  const at = handle.lastIndexOf("@");
  if (at === -1) return null;
  const side = handle.slice(at + 1);
  return (SIDES as string[]).includes(side) ? (side as Side) : null;
};

/**
 * One end of a wire, as the geometry needs to know it.
 *
 * This used to carry the block's type and the port's name so the renderer could
 * look both up in a table of `"type:port"` strings. The port now says these
 * things itself, so the hint carries the answers rather than the keys — and the
 * table, which had two entries matching no catalog type at all, is gone.
 */
export interface EndpointHint {
  /** True when this pin always leaves sideways, whatever the geometry. */
  side: boolean;
  /** Declared element type, or `inherit`. */
  dtype: string;
}

function centre(b: Box): { x: number; y: number } {
  return { x: b.x + b.width / 2, y: b.y + b.height / 2 };
}

/**
 * Which sides a wire should use, from the boxes it runs between.
 *
 * Down the sheet is the default reading direction, so a target below its source
 * leaves the bottom and arrives at the top. Anything else — a bypass rejoining
 * the line, a wire that has to climb back up, a block placed beside rather than
 * below — leaves by the side that faces where it is going.
 */
export function chooseSides(
  source: Box | undefined,
  target: Box | undefined,
  from: EndpointHint,
  to: EndpointHint,
): { sourceSide: Side; targetSide: Side } {
  const forcedIn = to.side;
  const forcedOut = from.side;

  if (!source || !target) {
    return {
      sourceSide: forcedOut ? "right" : "bottom",
      targetSide: forcedIn ? "left" : "top",
    };
  }

  const a = centre(source);
  const b = centre(target);
  const dx = b.x - a.x;
  const dy = b.y - a.y;

  // Which way round the two boxes actually are decides which side a forced
  // sideways pin uses, so a bypass on the left of the line stays on the left.
  const leftOfTarget = dx >= 0;

  let sourceSide: Side;
  let targetSide: Side;

  if (forcedIn) {
    targetSide = leftOfTarget ? "left" : "right";
    sourceSide = dy > 0 ? "bottom" : leftOfTarget ? "right" : "left";
  } else if (forcedOut) {
    sourceSide = leftOfTarget ? "right" : "left";
    targetSide = dy > 0 ? "top" : leftOfTarget ? "left" : "right";
  } else if (Math.abs(dy) >= Math.abs(dx)) {
    sourceSide = dy >= 0 ? "bottom" : "top";
    targetSide = dy >= 0 ? "top" : "bottom";
  } else {
    sourceSide = dx >= 0 ? "right" : "left";
    targetSide = dx >= 0 ? "left" : "right";
  }

  return { sourceSide, targetSide };
}

/**
 * What kind of line a wire is drawn with.
 *
 * Three, not ten. A schematic distinguishes a bus from a net and a net from a
 * no-connect, and nothing else. Here: the main path, the line that skips a
 * stage, and the one that carries something other than activations.
 */
export type WireKind = "signal" | "bypass" | "index";

export function wireKind(from: EndpointHint, to: EndpointHint): WireKind {
  if (to.side) return "bypass";
  // Declared, not sniffed. A port that carries indices says so; before this the
  // renderer decided by asking whether a dtype name began with "int".
  if (from.dtype === "int" || from.dtype === "bool") return "index";
  return "signal";
}
