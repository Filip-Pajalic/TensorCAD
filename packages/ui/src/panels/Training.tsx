/**
 * Training: three views of one question, behind one tab.
 *
 * How a design splits across a cluster, what it becomes at other widths, and
 * what it did when it was trained. They were three tabs of their own, which
 * with the inspector, the symbols and the history made six across a column
 * four hundred pixels wide — a row somebody new reads as six things to learn
 * before they can start. They are one tab now, with the three as a switch
 * inside it, and each still answers to its own command and shortcut.
 */

import { useEditor, type RightTab } from "../state/store.js";
import Cluster from "./Cluster.js";
import Ladder from "./Ladder.js";
import Runs from "./Runs.js";

export type TrainingView = Extract<RightTab, "cluster" | "ladder" | "runs">;

export const TRAINING_VIEWS: { id: TrainingView; label: string; hint: string }[] = [
  { id: "cluster", label: "Cluster", hint: "Every way of splitting training across the GPUs, and which fit" },
  { id: "ladder", label: "Width ladder", hint: "The same design at several widths, and what to scale by at each" },
  { id: "runs", label: "Runs", hint: "What the design did when it was trained: loss against tokens" },
];

export function isTrainingView(tab: RightTab): tab is TrainingView {
  return TRAINING_VIEWS.some((v) => v.id === tab);
}

export default function Training({ view }: { view: TrainingView }): React.ReactElement {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="subtabs" role="tablist" aria-label="Training">
        {TRAINING_VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            role="tab"
            aria-selected={v.id === view}
            className={"subtabs__tab" + (v.id === view ? " is-on" : "")}
            title={v.hint}
            onClick={() => useEditor.getState().setRightTab(v.id)}
          >
            {v.label}
          </button>
        ))}
      </div>
      {view === "cluster" ? <Cluster /> : view === "ladder" ? <Ladder /> : <Runs />}
    </div>
  );
}
