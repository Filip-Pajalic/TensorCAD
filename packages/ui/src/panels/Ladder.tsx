/**
 * The same design at several widths, for a sweep that transfers.
 *
 * A learning rate tuned on a narrow model is the right one for a wide model
 * too, provided the initialization and the per-parameter rates are scaled by
 * width the way Tensor Programs V's Table 3 says. This panel is that ladder:
 * what each rung costs, and what to multiply by at each.
 *
 * Pressing a rung opens it. A rung is a different design rather than a
 * different operating point — which is what distinguishes this from the Cluster
 * panel beside it, where pressing a plan changes only how the same design is
 * measured. It goes through `setDoc`, so it lands on the undo stack like any
 * other way of opening a design.
 *
 * Like Cluster, it does not read `derive()`: a ladder is one analysis per rung
 * and the answer only moves when the design does.
 */

import { useMemo } from "react";
import { useEditor } from "../state/store.js";
import { mupLadder } from "../engine.js";
import { formatCount } from "@tensorcad/engine";
import type { MupLadder, MupRung, MupScaling } from "@tensorcad/engine";
import Section from "./Section.js";

/** A multiplier, written so that 1 reads as "unchanged" rather than as 1.000. */
function times(v: number): string {
  if (v === 1) return "—";
  return `x${Number(v.toPrecision(4))}`;
}

function Row({ rung, widest }: { rung: MupRung; widest: number }): React.ReactElement {
  return (
    <button
      className={`rung${rung.base ? " rung--base" : ""}`}
      onClick={() =>
        useEditor
          .getState()
          .setDoc(rung.doc, `Opened the ${rung.width}-wide rung of ${rung.doc.meta.name}`)
      }
      title="Open this rung"
    >
      <span className="rung__width mono">{rung.width}</span>
      <span className="rung__heads">{rung.heads} heads</span>
      <span className="rung__bar" aria-hidden>
        <span
          className="rung__fill"
          style={{ width: `${Math.max(2, Math.round((rung.width / widest) * 100))}%` }}
        />
      </span>
      <span className="rung__params mono">{formatCount(rung.params)}</span>
    </button>
  );
}

/**
 * One class across the whole ladder.
 *
 * By class rather than by rung, because the classes are what a person acts on —
 * an optimizer is configured once with a parameter group per class — and the
 * rungs are the columns of that table.
 */
function ClassRow({
  klass,
  rungs,
  index,
}: {
  klass: MupScaling;
  rungs: MupRung[];
  index: number;
}): React.ReactElement {
  return (
    <div className="mup__class">
      <div className="mup__className">{klass.class}</div>
      <div className="mup__grid" style={{ gridTemplateColumns: `4rem repeat(${rungs.length}, 1fr)` }}>
        <div className="mup__rowLabel" />
        {rungs.map((r) => (
          <div className="mup__head mono" key={r.width}>
            {r.width}
          </div>
        ))}
        <div className="mup__rowLabel">init</div>
        {rungs.map((r) => (
          <div className="mup__cell mono" key={r.width}>
            {times(r.scaling[index]!.initStd)}
          </div>
        ))}
        <div className="mup__rowLabel">rate</div>
        {rungs.map((r) => (
          <div className="mup__cell mono" key={r.width}>
            {times(r.scaling[index]!.adamLr)}
          </div>
        ))}
      </div>
      <div className="mup__why">{klass.why}</div>
      <div className="mup__paths">
        {klass.paths.length === 0
          ? "nothing in this design"
          : `${klass.paths.length} ${klass.paths.length === 1 ? "weight" : "weights"}: ${klass.paths.join(", ")}`}
      </div>
    </div>
  );
}

export default function Ladder(): React.ReactElement {
  const doc = useEditor((s) => s.doc);

  const result = useMemo((): MupLadder | { error: string } => {
    try {
      return mupLadder(doc);
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) };
    }
  }, [doc]);

  if ("error" in result) {
    return (
      <div className="panel__body">
        <div className="empty">{result.error}</div>
      </div>
    );
  }

  const widest = Math.max(...result.rungs.map((r) => r.width));
  const classes = result.rungs[0]?.scaling ?? [];
  return (
    <div className="panel__body">
      <div className="cluster__head">
        <div className="cluster__device">
          {result.rungs.length} rungs on {result.widthSymbol}, tuned at {result.baseWidth}
        </div>
        <div className="cluster__budget">
          Heads of {result.headDim} throughout: the heads get more numerous, not wider.
        </div>
      </div>

      <div className="plans">
        {result.rungs.map((r) => (
          <Row key={r.width} rung={r} widest={widest} />
        ))}
      </div>

      <Section id="mup-scaling" title="Multiply by" note="against the base rung">
        {classes.map((c, i) => (
          <ClassRow key={c.class} klass={c} rungs={result.rungs} index={i} />
        ))}
      </Section>

      {result.rungs.flatMap((r) =>
        r.notes.map((n) => (
          <div className="cluster__note" key={`${r.width}:${n}`}>
            <b className="mono">{r.width}</b> {n}
          </div>
        )),
      )}
      {result.notes.map((n) => (
        <div className="cluster__note" key={n}>
          {n}
        </div>
      ))}
    </div>
  );
}
