/**
 * The status bar.
 *
 * Every CAD application has one, and it always says the same kinds of thing:
 * where the cursor is, what the grid is, how far you are zoomed, what is
 * selected, and whether the design currently passes its checks. It is the one
 * place you can look to answer "what am I working on and is it valid".
 */

import { useEditor } from "../state/store.js";
import { useDerived } from "../state/hooks.js";
import { bridgeIsPossible, useBridge } from "../state/bridge.js";
import { formatCount } from "@tensor-cad/engine";

function Cell({
  label,
  value,
  tone,
  title,
  grow,
}: {
  label?: string;
  value: React.ReactNode;
  tone?: "ok" | "warning" | "error";
  title?: string;
  grow?: boolean;
}): React.ReactElement {
  return (
    <div className={`statusbar__cell${grow ? " statusbar__cell--grow" : ""}`} title={title}>
      {label && <span className="statusbar__key">{label}</span>}
      <span className={`statusbar__value${tone ? ` statusbar__value--${tone}` : ""}`}>{value}</span>
    </div>
  );
}

/**
 * Whether an agent is watching, and what it last did.
 *
 * Only where one could be: the hosted editor cannot reach a process on your
 * machine, and a cell that permanently said "no agent" there would be a cell
 * that taught people to stop reading the status bar.
 *
 * Pressing it detaches, and pressing it again looks for one — because the
 * thing a person wants when an agent starts editing under them is a way to
 * make it stop.
 */
function AgentCell(): React.ReactElement | null {
  const status = useBridge((s) => s.status);
  const enabled = useBridge((s) => s.enabled);
  const setEnabled = useBridge((s) => s.setEnabled);
  const agent = useBridge((s) => s.agent);
  const last = useBridge((s) => s.lastFromAgent);
  const detail = useBridge((s) => s.detail);

  if (!bridgeIsPossible()) return null;

  const value = !enabled
    ? "off"
    : status === "connected"
      ? (last ?? "attached")
      : status === "looking"
        ? "looking…"
        : "none";

  return (
    <button
      type="button"
      className="statusbar__cell statusbar__cell--button"
      onClick={() => setEnabled(!enabled)}
      title={
        status === "connected"
          ? `Attached to ${agent ?? "an agent"}. Its edits arrive here and land on the undo stack. Press to detach.`
          : (detail ?? "No agent is listening. Start one with TENSORCAD_BRIDGE=1. Press to look again.")
      }
    >
      <span className="statusbar__key">agent</span>
      <span className={`statusbar__value${status === "connected" ? " statusbar__value--ok" : ""}`}>{value}</span>
    </button>
  );
}

export default function StatusBar(): React.ReactElement {
  const derived = useDerived();
  const selection = useEditor((s) => s.selection);
  const also = useEditor((s) => s.also);
  const isLocked = useEditor((s) => s.isLocked);
  const path = useEditor((s) => s.path);
  const status = useEditor((s) => s.canvasStatus);
  const net = useEditor((s) => s.selectedNet);

  const errors = derived.issues.filter((i) => i.severity === "error").length;
  const warnings = derived.issues.filter((i) => i.severity === "warning").length;

  const selectionLabel = net
    ? `net ${net}`
    : !selection
    ? `${status.nodeCount} blocks, ${status.edgeCount} nets`
    : also.length > 0
      ? // Naming the primary as well as counting keeps the cell useful: it is
        // the one the inspector is showing.
        `${also.length + 1} selected, ${selection} last`
      : `${selection}${isLocked(selection) ? " (locked)" : ""}`;

  const checkTone = errors > 0 ? "error" : warnings > 0 ? "warning" : "ok";
  const checkText =
    errors > 0
      ? `${errors} error${errors === 1 ? "" : "s"}`
      : warnings > 0
        ? `${warnings} warning${warnings === 1 ? "" : "s"}`
        : "checks pass";

  return (
    <div className="statusbar" role="status">
      <Cell
        label="x y"
        value={
          status.cursor
            ? `${Math.round(status.cursor.x)}, ${Math.round(status.cursor.y)}`
            : "—"
        }
        title="Cursor position in canvas units"
      />
      <Cell
        label="grid"
        value={`${status.gridMinor} / ${status.gridMajor}`}
        title="Minor and major grid pitch. Positions snap to the minor grid."
      />
      <Cell label="zoom" value={`${Math.round(status.zoom * 100)}%`} />
      <Cell
        label="level"
        value={path.length === 0 ? "top" : path.join(" / ")}
        title="Which graph level is open"
      />
      <Cell label="sel" value={selectionLabel} grow title="Current selection" />
      <Cell
        label="params"
        value={formatCount(derived.params.total)}
        title={`${derived.params.total.toLocaleString("en-US")} parameters`}
      />
      <Cell label="drc" value={checkText} tone={checkTone} title="Design-rule check" />
      <AgentCell />
    </div>
  );
}
