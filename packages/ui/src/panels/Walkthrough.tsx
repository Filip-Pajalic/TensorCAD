/**
 * The walkthrough, as a column beside the drawing.
 *
 * Deliberately a column and not a dialog. The point is that the text and the
 * drawing are one thing: the step lights the parts it is about and dims the
 * rest, and reading a sentence about attention while attention is the only lit
 * block on the sheet is the whole mechanism. A modal over the drawing would
 * cover exactly what it was describing.
 */

import { useEffect, useMemo } from "react";
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useEditor } from "../state/store.js";
import { useLevel } from "../state/hooks.js";
import { buildWalkthrough } from "../state/walkthrough.js";
import { useTrace } from "../three/trace.js";
import { Button } from "../ui/button.js";

export default function Walkthrough(): React.ReactElement | null {
  const doc = useEditor((s) => s.doc);
  const at = useEditor((s) => s.walkthrough);
  const act = useEditor.getState();
  const { derived } = useLevel();

  // The run of this design, if there is one: its steps then quote real numbers.
  const trace = useTrace(doc);
  const steps = useMemo(() => buildWalkthrough(doc, derived, trace), [doc, derived, trace]);

  const step = at === null ? null : steps[Math.min(at, steps.length - 1)];

  // The step decides how far the containers are open, so its blocks are on
  // screen to be lit. Done here rather than in the store because it is a
  // consequence of what is being shown, not part of the walkthrough's state.
  useEffect(() => {
    if (step) useEditor.getState().setDetail(step.detail);
  }, [step]);

  if (at === null || !step) return null;

  const last = steps.length - 1;
  const atStep = Math.min(at, last);

  return (
    <aside className="wt" aria-label="Walkthrough">
      <div className="wt__bar">
        <span className="wt__count">
          {atStep + 1} of {steps.length}
        </span>
        <button
          type="button"
          className="wt__shut"
          onClick={() => act.endWalkthrough()}
          aria-label="close the walkthrough"
        >
          <X size={13} />
        </button>
      </div>

      <h2 className="wt__title">{step.title}</h2>
      {step.body.map((paragraph, i) => (
        <p className="wt__para" key={i}>
          {paragraph}
        </p>
      ))}

      <div className="wt__dots" role="tablist" aria-label="Steps">
        {steps.map((s, i) => (
          <button
            key={s.id}
            type="button"
            role="tab"
            aria-selected={i === atStep}
            aria-label={s.title}
            title={s.title}
            className={`wt__dot${i === atStep ? " is-at" : ""}`}
            onClick={() => act.setWalkthroughStep(i)}
          />
        ))}
      </div>

      <div className="wt__nav">
        <Button
          variant="ghost"
          size="sm"
          disabled={atStep === 0}
          onClick={() => act.setWalkthroughStep(atStep - 1)}
        >
          <ChevronLeft size={14} />
          Back
        </Button>
        {atStep === last ? (
          <Button size="sm" onClick={() => act.endWalkthrough()}>
            Done
          </Button>
        ) : (
          <Button size="sm" onClick={() => act.setWalkthroughStep(atStep + 1)}>
            Next
            <ChevronRight size={14} />
          </Button>
        )}
      </div>
    </aside>
  );
}
