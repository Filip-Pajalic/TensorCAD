/**
 * How to split the training across the cluster.
 *
 * The readout above answers "what does this design cost under the operating
 * point I set". This answers the question that comes first: what operating
 * point should it be. The engine prices every split the cluster admits — the
 * degrees have to multiply to the device count, a tensor-parallel group has to
 * fit in a node, a pipeline cannot have more stages than the stack has layers —
 * and returns the ones that fit.
 *
 * Clicking a plan applies it, which is the point: the parallelism is editor
 * state, so a plan is something the panel can hand to the operating point and
 * watch every other panel follow.
 *
 * What is claimed here is memory, which is arithmetic. Which plan is fastest is
 * not claimed; it turns on the interconnect and the kernels, so each plan says
 * what it costs to run instead.
 */

import { useMemo } from "react";
import { useEditor } from "../state/store.js";
import { toAnalysisOptions } from "../state/operating.js";
import { planCluster } from "../engine.js";
import { formatBytes } from "@tensor-cad/engine";
import type { ClusterPlan, ClusterResult } from "@tensor-cad/engine";
import Section from "./Section.js";

/** Where a fit stops being comfortable. Matches the engine's own threshold. */
const TIGHT = 0.85;

/** The operating-point fields a plan sets, and nothing else. */
function applyPlan(p: ClusterPlan): void {
  useEditor.getState().setOperating({
    tp: p.parallel.tp,
    pp: p.parallel.pp,
    ep: p.parallel.ep,
    zero: p.parallel.zero as 0 | 1 | 2 | 3,
    sequenceParallel: p.parallel.sequenceParallel,
    recompute: p.recompute,
    B: p.microBatch,
  });
}

/** True when the operating point is already this plan. */
function isCurrent(p: ClusterPlan, o: ReturnType<typeof useEditor.getState>["operating"]): boolean {
  return (
    p.parallel.tp === o.tp &&
    p.parallel.pp === o.pp &&
    p.parallel.ep === o.ep &&
    p.parallel.zero === o.zero &&
    p.parallel.sequenceParallel === (o.tp > 1 && o.sequenceParallel) &&
    p.recompute === o.recompute
  );
}

function Row({ plan, current }: { plan: ClusterPlan; current: boolean }): React.ReactElement {
  const pct = Math.round(plan.used * 100);
  return (
    <button
      className={`plan${current ? " plan--current" : ""}${plan.used > TIGHT ? " plan--tight" : ""}`}
      onClick={() => applyPlan(plan)}
      title={current ? "This is the operating point" : "Use this plan"}
    >
      <span className="plan__summary">{plan.summary}</span>
      <span className="plan__bytes mono">{formatBytes(plan.perGpu.total)}</span>
      <span className="plan__bar" aria-hidden>
        <span className="plan__fill" style={{ width: `${Math.min(100, pct)}%` }} />
      </span>
      <span className="plan__pct mono">{pct}%</span>
    </button>
  );
}

function Detail({ plan }: { plan: ClusterPlan }): React.ReactElement {
  const g = plan.perGpu;
  return (
    <div className="plan__detail">
      <div className="plan__parts">
        {(
          [
            ["weights", g.weights],
            ["gradients", g.grads],
            ["optimizer", g.optimizer],
            ["activations", g.activations],
          ] as const
        ).map(([label, bytes]) => (
          <div className="plan__part" key={label}>
            <span className="plan__partLabel">{label}</span>
            <span className="mono">{formatBytes(bytes)}</span>
          </div>
        ))}
      </div>
      {plan.notes.map((n) => (
        <div className="plan__note" key={n}>
          {n}
        </div>
      ))}
    </div>
  );
}

export default function Cluster(): React.ReactElement {
  const doc = useEditor((s) => s.doc);
  const operating = useEditor((s) => s.operating);

  // Only the cluster and the design change the answer. The parallelism and the
  // recompute setting are what is being searched for, so they are left out of
  // the key: applying a plan must not send the panel looking for another one.
  const key = useMemo(() => {
    const { parallel: _p, recompute: _r, ...rest } = toAnalysisOptions(operating);
    return JSON.stringify(rest);
  }, [operating]);

  const result = useMemo((): ClusterResult | { error: string } => {
    try {
      const options = JSON.parse(key) as ReturnType<typeof toAnalysisOptions>;
      return planCluster(doc, options, { gpus: operating.gpus, limit: 8 });
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) };
    }
    // `key` already carries everything from the operating point that matters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc, key]);

  if ("error" in result) {
    return (
      <div className="panel__body">
        <div className="empty">{result.error}</div>
      </div>
    );
  }

  const best = result.fits[0];
  return (
    <div className="panel__body">
      <div className="cluster__head">
        <div className="cluster__device">
          {operating.gpus} x {result.hardware}
        </div>
        <div className="cluster__budget">
          {formatBytes(result.budget)} usable of {formatBytes(result.memory)} each,{" "}
          {result.considered} plans priced
        </div>
      </div>

      {result.fits.length === 0 ? (
        <div className="empty">
          Nothing fits.
          {result.closest && (
            <>
              {" "}
              The nearest is <b>{result.closest.summary}</b> at{" "}
              {formatBytes(result.closest.perGpu.total)} per device, against a budget of{" "}
              {formatBytes(result.budget)}.
            </>
          )}
        </div>
      ) : (
        <>
          <div className="plans">
            {result.fits.map((p) => (
              <Row key={p.summary} plan={p} current={isCurrent(p, operating)} />
            ))}
          </div>
          <Section id="cluster-best" title={best.summary} note="least demanding">
            <Detail plan={best} />
          </Section>
        </>
      )}

      {result.notes.map((n) => (
        <div className="cluster__note" key={n}>
          {n}
        </div>
      ))}
    </div>
  );
}
