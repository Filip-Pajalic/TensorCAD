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
 * Ports that always leave or arrive sideways, regardless of where the other end
 * is. The key is `type:port`.
 *
 * `add:b` is the residual bypass. `mul:b` is the gate in a gated feed-forward.
 * Both are the line a figure draws running alongside the main path, and both
 * look wrong entering from the top, because from the top they read as the main
 * path rather than the one that skips it.
 */
const SIDEWAYS_IN = new Set(["add:b", "mul:b"]);

/**
 * Ports whose block is an accessory hanging off the side of the main line,
 * rather than a stage on it: rotary tables, masks, routers.
 */
const SIDEWAYS_OUT = new Set(["rope:y", "causal_mask:mask", "router:weights"]);

export interface EndpointHint {
  type: string;
  port: string;
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
  const forcedIn = SIDEWAYS_IN.has(`${to.type}:${to.port}`);
  const forcedOut = SIDEWAYS_OUT.has(`${from.type}:${from.port}`);

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

export function wireKind(to: EndpointHint, dtype: string | null): WireKind {
  if (SIDEWAYS_IN.has(`${to.type}:${to.port}`)) return "bypass";
  if (dtype && dtype.startsWith("int")) return "index";
  return "signal";
}
