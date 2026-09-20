/**
 * The design-rule check, along the bottom.
 *
 * It was a tab, which meant it was never on screen while you worked: to find
 * out what was wrong you left the inspector, and to fix it you left the
 * findings. Eighteen rules run on every edit and the answer belongs where a PCB
 * tool puts it — across the bottom, under the drawing it is about, open while
 * you edit.
 *
 * Collapsed it is still a strip carrying the counts, because a design that is
 * broken should say so somewhere on screen whatever else is being looked at.
 * Pressing the strip opens it; pressing a marker on the canvas opens it too,
 * already filtered to that block.
 */

import { useEditor } from "../state/store.js";
import { useDerived } from "../state/hooks.js";
import Rules from "./Rules.js";
import { ChevronDown, ChevronUp } from "lucide-react";

/** The one-line summary, which is what the strip says either way. */
function Counts(): React.ReactElement {
  const derived = useDerived();
  const { error, warning, info } = derived.counts;
  if (error === 0 && warning === 0 && info === 0) {
    return <span className="findings__ok">checks pass</span>;
  }
  return (
    <>
      {error > 0 && (
        <span className="findings__count findings__count--error mono">
          {error} error{error === 1 ? "" : "s"}
        </span>
      )}
      {warning > 0 && (
        <span className="findings__count findings__count--warning mono">
          {warning} warning{warning === 1 ? "" : "s"}
        </span>
      )}
      {info > 0 && <span className="findings__count mono">{info} note{info === 1 ? "" : "s"}</span>}
    </>
  );
}

export default function FindingsDock(): React.ReactElement {
  const open = useEditor((s) => s.dockOpen);
  const focus = useEditor((s) => s.findingFocus);
  const toggle = useEditor((s) => s.toggleFindings);

  return (
    <section className={`findings${open ? " is-open" : ""}`}>
      <button
        className="findings__strip"
        onClick={toggle}
        title={open ? "Close the checks (Ctrl+Shift+F)" : "Open the checks (Ctrl+Shift+F)"}
        aria-expanded={open}
      >
        <span className="findings__label">Checks</span>
        <Counts />
        {focus !== null && <span className="findings__focus mono">{focus}</span>}
        <span className="findings__chevron" aria-hidden>
          {open ? <ChevronDown className="size-3.5" /> : <ChevronUp className="size-3.5" />}
        </span>
      </button>
      {open && (
        <div className="findings__body">
          <Rules />
        </div>
      )}
    </section>
  );
}
