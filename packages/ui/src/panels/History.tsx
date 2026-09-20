/**
 * The history: what you did, and a way back to any of it.
 *
 * Undo was a stack you could only walk one press at a time, which meant that
 * finding out whether the thing you regret was four edits ago or six was done
 * by pressing Ctrl+Z and watching. The states were always there — the store has
 * kept a hundred of them since the first pass — and what was missing was a way
 * to read them.
 *
 * So each one carries the sentence the toolbar showed when it happened. What
 * you read at the time is what you read here, which is the property that makes
 * a row recognisable rather than merely distinct.
 *
 * ## What this is not
 *
 * Not a feature timeline. A CAD timeline holds *operations*, and its point is
 * that you can suppress one in the middle and everything after it replays
 * without it. This holds states: jumping back and then editing discards what
 * was ahead, exactly as undo-then-edit always has. The difference matters and
 * is worth saying out loud, because the two look the same in a screenshot.
 */

import { useEditor } from "../state/store.js";

export default function History(): React.ReactElement {
  const past = useEditor((s) => s.past);
  const future = useEditor((s) => s.future);
  const docLabel = useEditor((s) => s.docLabel);
  const jumpTo = useEditor((s) => s.jumpTo);

  const steps = [...past.map((p) => p.label), docLabel, ...future.map((f) => f.label)];
  const current = past.length;

  return (
    <div className="panel__body history">
      <ol className="history__list">
        {steps.map((label, i) => (
          <li key={`${i}-${label}`}>
            <button
              type="button"
              className={
                "history__step" +
                (i === current ? " history__step--current" : "") +
                (i > current ? " history__step--ahead" : "")
              }
              onClick={() => jumpTo(i)}
              // The current row is where you are; pressing it would be a
              // no-op that still looked like it did something.
              disabled={i === current}
            >
              <span className="history__mark" aria-hidden>
                {i === current ? "●" : "○"}
              </span>
              <span className="history__label">{label}</span>
            </button>
          </li>
        ))}
      </ol>

      {future.length > 0 && (
        <p className="history__note">
          The {future.length === 1 ? "step" : `${future.length} steps`} below the mark{" "}
          {future.length === 1 ? "is" : "are"} ahead of where you are. Editing now discards{" "}
          {future.length === 1 ? "it" : "them"}.
        </p>
      )}

      {past.length >= 99 && (
        <p className="history__note">
          A hundred states is the limit; older ones have been dropped.
        </p>
      )}
    </div>
  );
}
