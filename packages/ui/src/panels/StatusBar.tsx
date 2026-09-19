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
import { formatCount } from "@tensorcad/engine";

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

export default function StatusBar(): React.ReactElement {
  const derived = useDerived();
  const selection = useEditor((s) => s.selection);
  const isLocked = useEditor((s) => s.isLocked);
  const path = useEditor((s) => s.path);
  const status = useEditor((s) => s.canvasStatus);

  const errors = derived.issues.filter((i) => i.severity === "error").length;
  const warnings = derived.issues.filter((i) => i.severity === "warning").length;

  const selectionLabel = selection
    ? `${selection}${isLocked(selection) ? " (locked)" : ""}`
    : `${status.nodeCount} blocks, ${status.edgeCount} nets`;

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
    </div>
  );
}
