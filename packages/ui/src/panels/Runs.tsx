/**
 * Runs: training runs, on one chart.
 *
 * The rest of this editor answers what a design *would* cost. This is the one
 * panel that says what one actually did — which is the only way to find out
 * whether the architecture you drew is better than the one you drew before it.
 *
 * The chart is a log-scale loss curve against tokens rather than against
 * steps, because a step is not a fixed amount of work: two designs at the same
 * batch size and different sequence lengths see different numbers of tokens
 * per step, and plotting against steps quietly gives the longer one credit for
 * the extra text. Tokens is the axis the scaling laws are written in.
 *
 * What it refuses to do is pretend two runs are comparable when they are not.
 * Anything that differs between the runs on screen and would change the curve
 * — sequence length, batch, corpus, seed, learning rate, dtype — is named
 * above the chart. The chart is still drawn: the person asked for it, and a
 * comparison with a caveat is more use than a refusal.
 */

import { useCallback, useMemo, useRef, useState } from "react";
import { incomparable, parseRun, useRuns, type RunRecord } from "../state/runs.js";
import { formatCount } from "@tensor-cad/engine";
import Section from "./Section.js";

const SERIES = 6;
const seriesColor = (index: number): string => `var(--series-${(index % SERIES) + 1})`;

const bytes = (n: number | undefined): string =>
  n === undefined || n === 0 ? "—" : `${(n / 1024 ** 3).toFixed(2)} GiB`;

export default function Runs(): React.ReactElement {
  const runs = useRuns((s) => s.runs);
  const shown = useRuns((s) => s.shown);
  const add = useRuns((s) => s.add);
  const remove = useRuns((s) => s.remove);
  const toggle = useRuns((s) => s.toggle);
  const clear = useRuns((s) => s.clear);
  const [problem, setProblem] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const take = useCallback(
    async (files: FileList | null): Promise<void> => {
      if (!files || files.length === 0) return;
      const parsed: RunRecord[] = [];
      const failed: string[] = [];
      for (const file of Array.from(files)) {
        try {
          parsed.push(parseRun(file.name, await file.text()));
        } catch (e) {
          failed.push(`${file.name}: ${(e as Error).message}`);
        }
      }
      if (parsed.length > 0) add(parsed);
      setProblem(failed.length > 0 ? failed.join("; ") : null);
    },
    [add],
  );

  const drawn = useMemo(() => runs.filter((r) => shown.includes(r.id)), [runs, shown]);
  const caveats = useMemo(() => incomparable(drawn), [drawn]);

  return (
    <div
      className="panel__body runs"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        void take(e.dataTransfer.files);
      }}
    >
      <Section id="runs.list" title="Runs" note={runs.length || undefined}>
        {runs.length === 0 ? (
          <p className="empty">
            No runs yet. <code>tensorcad-runtime smoke-train out/&lt;design&gt;/model.py</code>{" "}
            writes one to <code>runs/</code>; open the <code>.json</code> beside the{" "}
            <code>.jsonl</code>, or drop it here.
          </p>
        ) : (
          <table className="runs__table">
            <thead>
              <tr>
                <th />
                <th>run</th>
                <th className="num">params</th>
                <th className="num">final</th>
                <th className="num">tok/s</th>
                <th className="num">peak</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => {
                const index = drawn.findIndex((d) => d.id === run.id);
                return (
                  <tr key={run.id} className={index < 0 ? "runs__row--hidden" : undefined}>
                    <td>
                      <input
                        type="checkbox"
                        checked={index >= 0}
                        onChange={() => toggle(run.id)}
                        aria-label={`Show ${run.label}`}
                      />
                    </td>
                    <td title={`${run.steps ?? run.log.length} steps on ${run.device_name ?? run.device ?? "an unknown device"}, seed ${run.seed ?? "?"}`}>
                      <span
                        className="runs__swatch"
                        style={{ background: index < 0 ? "var(--border-soft)" : seriesColor(index) }}
                        aria-hidden
                      />
                      {run.label}
                    </td>
                    <td className="num">{run.params ? formatCount(run.params) : "—"}</td>
                    <td className="num">{run.final_loss?.toFixed(3) ?? "—"}</td>
                    <td className="num">
                      {run.tokens_per_second ? Math.round(run.tokens_per_second).toLocaleString("en-US") : "—"}
                    </td>
                    <td className="num">{bytes(run.peak_memory_bytes)}</td>
                    <td>
                      <button type="button" className="runs__drop" onClick={() => remove(run.id)} title="Forget this run">
                        ×
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}

        <div className="runs__actions">
          <input
            ref={input}
            type="file"
            accept=".json,.jsonl"
            multiple
            className="sr-only"
            onChange={(e) => void take(e.target.files)}
          />
          <button type="button" onClick={() => input.current?.click()}>
            Open run files…
          </button>
          {runs.length > 0 && (
            <button type="button" onClick={clear}>
              Forget all
            </button>
          )}
        </div>

        {problem && <p className="runs__problem">{problem}</p>}
      </Section>

      {drawn.length > 0 && (
        <Section id="runs.loss" title="Loss">
          {caveats.length > 0 && (
            <p className="runs__caveat">
              Not a like-for-like comparison — {caveats.join("; ")}. The curves are drawn anyway,
              but the difference between them is not only the architecture.
            </p>
          )}
          <LossChart runs={drawn} />
        </Section>
      )}
    </div>
  );
}

/**
 * Loss against tokens, log-log.
 *
 * Log on both axes because that is where a power law is a straight line, which
 * is the whole reason anybody looks at one of these. A linear axis turns every
 * run into the same shape — a cliff and then a flat — and hides the part that
 * distinguishes them.
 */
function LossChart({ runs }: { runs: RunRecord[] }): React.ReactElement {
  const W = 320;
  const H = 180;
  const PAD = { top: 8, right: 8, bottom: 22, left: 34 };

  const series = runs.map((run) => ({
    run,
    points: run.log
      .map((s, i) => ({ x: s.tokens ?? (i + 1), y: s.loss }))
      .filter((p) => p.x > 0 && Number.isFinite(p.y) && p.y > 0),
  }));

  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  if (xs.length === 0) return <p className="empty">Nothing logged in these runs.</p>;

  const x0 = Math.log10(Math.min(...xs));
  const x1 = Math.log10(Math.max(...xs));
  const y0 = Math.log10(Math.min(...ys));
  const y1 = Math.log10(Math.max(...ys));
  // A single point, or every point identical, would divide by zero.
  const spanX = x1 - x0 || 1;
  const spanY = y1 - y0 || 1;

  const px = (x: number): number => PAD.left + ((Math.log10(x) - x0) / spanX) * (W - PAD.left - PAD.right);
  const py = (y: number): number => PAD.top + (1 - (Math.log10(y) - y0) / spanY) * (H - PAD.top - PAD.bottom);

  return (
    <figure className="runs__chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training loss against tokens seen, both axes logarithmic">
        <rect
          x={PAD.left}
          y={PAD.top}
          width={W - PAD.left - PAD.right}
          height={H - PAD.top - PAD.bottom}
          fill="none"
          stroke="var(--border-soft)"
        />
        {[y0, (y0 + y1) / 2, y1].map((l) => (
          <g key={l}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={py(10 ** l)}
              y2={py(10 ** l)}
              stroke="var(--border-soft)"
              strokeDasharray="2 3"
            />
            <text x={PAD.left - 4} y={py(10 ** l) + 3} textAnchor="end" className="runs__tick">
              {(10 ** l).toFixed(2)}
            </text>
          </g>
        ))}
        {series.map(({ run, points }, i) =>
          points.length < 2 ? null : (
            <path
              key={run.id}
              d={points.map((p, j) => `${j === 0 ? "M" : "L"}${px(p.x).toFixed(1)},${py(p.y).toFixed(1)}`).join(" ")}
              fill="none"
              stroke={seriesColor(i)}
              strokeWidth={1.5}
            />
          ),
        )}
        <text x={W - PAD.right} y={H - 6} textAnchor="end" className="runs__tick">
          tokens (log)
        </text>
      </svg>
      <figcaption className="runs__legend">
        {runs.map((run, i) => (
          <span key={run.id}>
            <span className="runs__swatch" style={{ background: seriesColor(i) }} aria-hidden />
            {run.label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
