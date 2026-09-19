/**
 * A part on the drawing.
 *
 * Drawn the way every published architecture figure draws one: a pale body with
 * a thin outline, the name in the middle, the type beneath it, and the defining
 * numbers on a rule below that.
 *
 * Pins sit on all four sides. Only the ones a wire actually uses are drawn —
 * the rest stay as invisible targets so a connection can be started or dropped
 * anywhere on the symbol, which is what makes wiring feel like wiring rather
 * than like hitting a six-pixel dot on the bottom edge.
 *
 * Callouts are separate (see CalloutLayer): a drawing annotates its parts from
 * outside, on leader lines, rather than cramming everything into the symbol.
 */

import { memo } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { formatCount } from "@tensorcad/core";
import { dtypeColor, partColor, type BlockKind } from "./blocks.js";
import { SIDES, handleId, type Side } from "./wiring.js";
import type { Severity } from "../state/derive.js";

export interface PortView {
  name: string;
  shape: string | null;
  connected: boolean;
  /** Element type carried by this tensor, when the block declares one. */
  dtype: string | null;
}

export interface BlockNodeData extends Record<string, unknown> {
  path: string;
  label: string;
  type: string;
  category: string;
  kind: BlockKind;
  summary: string;
  params: number;
  inPorts: PortView[];
  outPorts: PortView[];
  severity: Severity | null;
  /**
   * What is actually wrong with this block, in its own words.
   *
   * `severity` rolls up everything beneath a container, which is right for the
   * model tree and wrong for a marker: a marker that fired because of something
   * three levels down would be pointing at the wrong part.
   */
  findings: { severity: Severity; message: string; rule: string; port?: string }[];
  drillable: boolean;
  readOnly: boolean;
  locked: boolean;
  /** Repeat count, shown against the part the way a figure writes "48x". */
  repeat: number | null;
  /**
   * A single character drawn in a circle instead of a labelled box. Elementwise
   * operators are junctions, not parts: every figure draws a residual sum as a
   * small circled plus sitting on the line, and drawing it as a box the size of
   * an attention block says it matters as much, which it does not.
   */
  glyph: string | null;
  /**
   * How many wires land on each handle. Absent is an unused side, one is an
   * ordinary connection, and more than one is a junction — the only one of the
   * three that a schematic actually marks.
   */
  livePins: ReadonlyMap<string, number>;
}

export type BlockFlowNode = Node<BlockNodeData, "block">;

const POSITION_OF: Record<Side, Position> = {
  top: Position.Top,
  right: Position.Right,
  bottom: Position.Bottom,
  left: Position.Left,
};

/**
 * Every side of every port, with the used ones visible.
 *
 * The idle handles are still full connection targets, so dragging a wire at a
 * block connects it rather than demanding the exact pin.
 */
function Pins({
  ports,
  dir,
  live,
  flagged,
}: {
  ports: PortView[];
  dir: "i" | "o";
  live: ReadonlyMap<string, number>;
  /** Ports a finding named, so the pin can carry the mark. */
  flagged: ReadonlySet<string>;
}): React.ReactElement | null {
  if (ports.length === 0) return null;
  const named = ports.length > 1;
  const home: Side = dir === "i" ? "top" : "bottom";

  return (
    <>
      {SIDES.map((side) => (
        <div className={`pins pins--${side}`} key={side}>
          {ports.map((port) => {
            const id = handleId(dir, port.name, side);
            const wires = live.get(id) ?? 0;
            const vertical = side === "top" || side === "bottom";
            // A port with nothing on it has no chosen side, because no wire
            // ever ran the geometry that picks one. It gets its mark on the
            // side it would have used by default — down the sheet, the way the
            // drawing reads — and on that side only, or an unwired port would
            // sprout four identical warnings around the symbol.
            const dangling = !port.connected && side === home;
            return (
              <div
                className={
                  `pin pin--${side}` +
                  (wires > 0 ? " pin--live" : " pin--idle") +
                  (dangling ? " pin--dangling" : "") +
                  (wires > 1 ? " pin--junction" : "") +
                  (flagged.has(port.name) ? " pin--flagged" : "")
                }
                key={id}
                style={{ ["--dtype" as string]: dtypeColor(port.dtype) }}
                title={
                  `${port.name}: ${port.shape ?? "shape not inferred"}` +
                  (port.dtype ? `\n${port.dtype}` : "") +
                  (wires > 1 ? `\nthe net branches here: ${wires} wires` : "") +
                  (port.connected ? "" : "\nnot connected")
                }
              >
                <Handle
                  type={dir === "i" ? "target" : "source"}
                  position={POSITION_OF[side]}
                  id={id}
                  className="pin__handle"
                />
                {named && wires > 0 && vertical && <span className="pin__name">{port.name}</span>}
              </div>
            );
          })}
        </div>
      ))}
    </>
  );
}

function BlockNodeView({ data, selected }: NodeProps<BlockFlowNode>): React.ReactElement {
  const part = partColor(data.category);
  const live = data.livePins;
  const findings = data.findings ?? [];
  const flaggedPorts = new Set(findings.map((f) => f.port).filter((p): p is string => !!p));

  if (data.glyph) {
    return (
      <div
        className={"glyph" + (selected ? " glyph--selected" : "")}
        style={{ ["--edge" as string]: part.edge }}
        title={`${data.label} — ${data.summary || data.type}`}
      >
        <Pins ports={data.inPorts} dir="i" live={live} flagged={flaggedPorts} />
        <span className="glyph__mark" aria-hidden>
          {data.glyph}
        </span>
        {data.severity && (
          <span className={`mark mark--${data.severity}`} title={`this part has ${data.severity}s`}>
            {data.severity === "error" ? "●" : "▲"}
          </span>
        )}
        <Pins ports={data.outPorts} dir="o" live={live} flagged={flaggedPorts} />
      </div>
    );
  }

  return (
    <div
      className={
        "part" +
        (selected ? " part--selected" : "") +
        (data.readOnly ? " part--readonly" : "") +
        (data.locked ? " part--locked" : "") +
        (data.drillable ? " part--drillable" : "")
      }
      style={{
        ["--fill" as string]: part.fill,
        ["--edge" as string]: part.edge,
        ["--part-ink" as string]: part.text,
      }}
    >
      <Pins ports={data.inPorts} dir="i" live={live} flagged={flaggedPorts} />

      <div
        className="part__body"
        title={data.drillable ? "double-click to open this block" : undefined}
      >
        <div className="part__name">
          {data.label}
          {data.severity && (
            <span
              className={`mark mark--${data.severity}`}
              title={`this part has ${data.severity}s`}
            >
              {data.severity === "error" ? "●" : "▲"}
            </span>
          )}
          {data.locked && (
            <span className="mark mark--lock" title="locked: position is fixed">
              &#128274;
            </span>
          )}
        </div>

        <div className="part__type">{data.type}</div>

        {(data.summary || data.params > 0) && (
          <div className="part__rule">
            {data.summary && <span className="part__values">{data.summary}</span>}
            {data.params > 0 && (
              <span
                className="part__params"
                title={`${data.params.toLocaleString("en-US")} parameters`}
              >
                {formatCount(data.params)}
              </span>
            )}
          </div>
        )}
      </div>

      {/* A repeated stack is marked beside the part, as a figure writes "48x". */}
      {/*
        A design-rule marker, as eeschema draws one: a mark placed on the thing
        that is wrong, not a note somewhere else. Seventeen rules run on every
        edit and until now none of them was visible where the work happens.
        The title carries every finding, so hovering answers "what is wrong
        with this" without a trip to the panel.
      */}
      {findings.length > 0 && (
        <div
          className={`drc drc--${findings.some((f) => f.severity === "error") ? "error" : findings.some((f) => f.severity === "warning") ? "warning" : "info"}`}
          title={findings.map((f) => `${f.rule}: ${f.message}`).join("\n")}
          aria-label={`${findings.length} finding${findings.length === 1 ? "" : "s"}`}
        >
          {findings.length > 1 ? findings.length : ""}
        </div>
      )}

      {data.repeat && data.repeat > 1 && (
        <div className="part__repeat" title={`this block is stacked ${data.repeat} times`}>
          {data.repeat}&times;
        </div>
      )}

      <Pins ports={data.outPorts} dir="o" live={live} flagged={flaggedPorts} />
    </div>
  );
}

export default memo(BlockNodeView);
