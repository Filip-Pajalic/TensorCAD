/**
 * The timeline: every edit, in order, and what the design is without any of
 * them.
 *
 * This is a feature tree, not an undo list. The rows are *operations* — an
 * `addNode` with the node it added, a `setParam` with the value it set — and
 * the drawing is what you get by replaying them over the design as it was
 * opened. So a row can be switched off and everything after it replays without
 * it, which is the thing a stack of documents could never do: by the time an
 * edit is a document, the operation that made it is gone.
 *
 * Three gestures, and they are different:
 *
 * - **Pressing a row** moves the mark there. Everything below is still in the
 *   list and still replayable; this is undo, several presses at once.
 * - **Suppressing** takes a step out of the middle and leaves it in the list.
 *   The design is what it would be without that edit.
 * - **Removing** takes it out for good.
 *
 * ## When a step cannot replay
 *
 * Suppress the edit that added a block and the edit that wired it has nothing
 * to wire. That is not a bug — it is the ordinary consequence of taking a step
 * out of the middle, and every parametric CAD tool has the condition. The step
 * is marked and says what it could not find, because that is what tells you
 * which earlier row to put back.
 */

import { useEditor } from "../state/store.js";

export default function History(): React.ReactElement {
  const steps = useEditor((s) => s.steps);
  const at = useEditor((s) => s.at);
  const baseLabel = useEditor((s) => s.baseLabel);
  const failures = useEditor((s) => s.failures);
  const jumpTo = useEditor((s) => s.jumpTo);
  const setSuppressed = useEditor((s) => s.setSuppressed);
  const removeStep = useEditor((s) => s.removeStep);

  const failureAt = new Map(failures.map((f) => [f.at, f.reason]));
  const ahead = steps.length - at;

  return (
    <div className="panel__body history">
      <ol className="history__list">
        {/* The base is a row. How the design arrived is a thing that happened,
            and a list that started at the first edit would have nowhere to put
            "opened Llama-3-8B". */}
        <li>
          <button
            type="button"
            className={"history__step" + (at === 0 ? " history__step--current" : "")}
            onClick={() => jumpTo(0)}
            disabled={at === 0}
          >
            <span className="history__mark" aria-hidden>
              {at === 0 ? "●" : "○"}
            </span>
            <span className="history__label">{baseLabel}</span>
          </button>
        </li>

        {steps.map((step, i) => {
          const index = i + 1;
          const failed = failureAt.get(i);
          return (
            <li key={`${i}-${step.label}`} className="history__row">
              <button
                type="button"
                className={
                  "history__step" +
                  (index === at ? " history__step--current" : "") +
                  (index > at ? " history__step--ahead" : "") +
                  (step.suppressed ? " history__step--off" : "") +
                  (failed ? " history__step--failed" : "")
                }
                onClick={() => jumpTo(index)}
                disabled={index === at}
                title={failed ? `Could not replay: ${failed}` : step.edit.kind}
              >
                <span className="history__mark" aria-hidden>
                  {index === at ? "●" : "○"}
                </span>
                <span className="history__label">{step.label}</span>
              </button>
              <button
                type="button"
                className="history__toggle"
                onClick={() => setSuppressed(i, !step.suppressed)}
                title={step.suppressed ? "Put this step back" : "Take this step out and replay the rest"}
                aria-pressed={Boolean(step.suppressed)}
              >
                {step.suppressed ? "off" : "on"}
              </button>
              <button
                type="button"
                className="history__drop"
                onClick={() => removeStep(i)}
                title="Remove this step for good"
              >
                ×
              </button>
            </li>
          );
        })}
      </ol>

      {failures.length > 0 && (
        <p className="history__note history__note--failed">
          {failures.length === 1 ? "One step" : `${failures.length} steps`} could not replay:{" "}
          {failures.map((f) => `${steps[f.at]?.label ?? "a step"} — ${f.reason}`).join("; ")}. Put
          back whatever it depended on, or remove it.
        </p>
      )}

      {ahead > 0 && (
        <p className="history__note">
          The {ahead === 1 ? "step" : `${ahead} steps`} below the mark {ahead === 1 ? "is" : "are"}{" "}
          ahead of where you are. Editing now discards {ahead === 1 ? "it" : "them"}.
        </p>
      )}

      {steps.length >= 99 && (
        <p className="history__note">
          A hundred steps is the limit; the oldest have been folded into the base.
        </p>
      )}
    </div>
  );
}
