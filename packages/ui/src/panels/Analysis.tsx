/**
 * The readout.
 *
 * Everything the analysis engine computes, in the order a designer asks for it:
 * how big is it, what does one token cost, will it fit, how fast does it serve,
 * what would training it take. Each section folds; none of them is behind a tab,
 * because the whole point of a CAD readout is that it is on screen while you
 * edit the thing it measures.
 */

import {
  DTYPE_BYTES,
  formatBytes,
  formatCount,
  formatDollars,
  formatFlops,
  formatHours,
} from "@tensorcad/core";
import { useEditor } from "../state/store.js";
import { useDerived } from "../state/hooks.js";
import { categoryColor } from "../canvas/blocks.js";
import Section from "./Section.js";

function pct(part: number, whole: number): string {
  if (!whole) return "—";
  return `${((part / whole) * 100).toFixed(1)}%`;
}

/** A label, a value, and an optional third column. */
function Row({
  label,
  value,
  aside,
  title,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  aside?: React.ReactNode;
  title?: string;
}): React.ReactElement {
  return (
    <tr title={title}>
      <td>{label}</td>
      <td className="num mono">{value}</td>
      <td className="num dim mono">{aside ?? ""}</td>
    </tr>
  );
}

function Breakdown({
  rows,
  total,
  colored = false,
  format = formatCount,
}: {
  rows: [string, number][];
  total: number;
  colored?: boolean;
  format?: (n: number) => string;
}): React.ReactElement {
  const sorted = [...rows].sort((a, b) => b[1] - a[1]);
  return (
    <table className="table">
      <tbody>
        {sorted.map(([name, value]) => (
          <tr key={name}>
            <td>
              {colored && (
                <span className="swatch" style={{ background: categoryColor(name) }} aria-hidden />
              )}
              <span className="mono">{name}</span>
            </td>
            <td className="num mono">{format(value)}</td>
            <td className="num dim mono">{pct(value, total)}</td>
            <td className="barcell">
              <span
                className="bar"
                style={{
                  width: `${total ? Math.max(1, (value / total) * 100) : 0}%`,
                  background: colored ? categoryColor(name) : "var(--accent)",
                }}
              />
            </td>
          </tr>
        ))}
        {sorted.length === 0 && (
          <tr>
            <td colSpan={4} className="dim">
              nothing counted yet
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

/** A bar split into named parts, for a budget against a capacity. */
function Budget({
  parts,
  capacity,
  capacityLabel,
}: {
  parts: { label: string; bytes: number; tone: string }[];
  capacity: number;
  capacityLabel: string;
}): React.ReactElement {
  const used = parts.reduce((a, p) => a + p.bytes, 0);
  const scale = Math.max(used, capacity);
  const over = used > capacity;
  return (
    <div className="budget">
      <div className="budget__bar">
        {parts.map((p) => (
          <span
            key={p.label}
            className="budget__part"
            style={{ width: `${(p.bytes / scale) * 100}%`, background: p.tone }}
            title={`${p.label}: ${formatBytes(p.bytes)}`}
          />
        ))}
        <span className="budget__mark" style={{ left: `${(capacity / scale) * 100}%` }} />
      </div>
      <div className="budget__legend">
        {parts.map((p) => (
          <span key={p.label} className="budget__key">
            <span className="swatch" style={{ background: p.tone }} aria-hidden />
            {p.label} <span className="mono dim">{formatBytes(p.bytes)}</span>
          </span>
        ))}
      </div>
      <div className={`budget__verdict ${over ? "error" : "ok"}`}>
        {formatBytes(used)} of {capacityLabel} &mdash;{" "}
        {over ? `over by ${formatBytes(used - capacity)}` : `${formatBytes(capacity - used)} spare`}
      </div>
    </div>
  );
}

export default function Analysis(): React.ReactElement {
  const doc = useEditor((s) => s.doc);
  const derived = useDerived();
  const a = derived.analysis;
  const p = a.params;
  const o = a.options;
  const hw = o.hardware;

  const published = doc.meta.published?.params;
  const delta = published ? p.total - published : 0;
  const exact = published ? delta === 0 : false;

  const topPaths = Object.entries(p.byPath)
    .sort((x, y) => y[1] - x[1])
    .slice(0, 10);

  const perGpu = a.memory.train.perGpu;
  const kvPer1k = a.kv.bytesPerToken * 1024;

  return (
    <div className="panel__body readout">
      <div className="headline">
        <div className="headline__value mono">{formatCount(p.total)}</div>
        <div className="headline__label">total parameters</div>
        <div className="headline__exact mono">{p.total.toLocaleString("en-US")}</div>
      </div>

      <Section id="params" title="Parameters" note={formatCount(p.total)}>
        <table className="table">
          <tbody>
            <Row
              label="non-embedding"
              value={formatCount(p.nonEmbedding)}
              aside={pct(p.nonEmbedding, p.total)}
            />
            <Row
              label="embedding"
              value={formatCount(p.embedding)}
              aside={pct(p.embedding, p.total)}
            />
            <Row label="head" value={formatCount(p.head)} aside={pct(p.head, p.total)} />
            <Row
              label="active per token"
              value={formatCount(p.active)}
              aside={pct(p.active, p.total)}
              title="What a single token actually multiplies against. Below the total only for a sparse model."
            />
          </tbody>
        </table>

        {published !== undefined && (
          <>
            <h4>Against the published figure</h4>
            <table className="table">
              <tbody>
                <Row label="published" value={published.toLocaleString("en-US")} />
                <Row label="this design" value={p.total.toLocaleString("en-US")} />
                <tr>
                  <td>difference</td>
                  <td className="num mono" colSpan={2}>
                    {exact ? (
                      <span className="badge badge--ok">exact</span>
                    ) : (
                      <span className={Math.abs(delta / published) < 0.005 ? "ok" : "warn"}>
                        {delta > 0 ? "+" : ""}
                        {delta.toLocaleString("en-US")} ({((delta / published) * 100).toFixed(3)}%)
                      </span>
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
            {doc.meta.published?.source && (
              <a
                className="source"
                href={doc.meta.published.source}
                target="_blank"
                rel="noreferrer"
              >
                {doc.meta.published.source.replace(/^https?:\/\//, "")}
              </a>
            )}
          </>
        )}

        <h4>By category</h4>
        <Breakdown rows={Object.entries(p.byCategory)} total={p.total} colored />
        <h4>By block type</h4>
        <Breakdown rows={Object.entries(p.byType)} total={p.total} />

        <h4>Heaviest blocks</h4>
        <table className="table">
          <tbody>
            {topPaths.map(([path, value]) => (
              <tr
                key={path}
                className="clickable"
                onClick={() => useEditor.getState().focusOn(path)}
                title={`Open ${path}`}
              >
                <td className="mono wrap">{path}</td>
                <td className="num mono">{formatCount(value)}</td>
                <td className="num dim mono">{pct(value, p.total)}</td>
              </tr>
            ))}
            {topPaths.length === 0 && (
              <tr>
                <td className="dim">no blocks with parameters</td>
              </tr>
            )}
          </tbody>
        </table>
      </Section>

      <Section id="compute" title="Compute" note={`${formatFlops(a.flops.fwdTotal)}/tok`}>
        <table className="table">
          <tbody>
            <Row
              label="forward, dense"
              value={formatFlops(a.flops.fwdDense)}
              aside={pct(a.flops.fwdDense, a.flops.fwdTotal)}
              title="Matmul work that does not grow with sequence length."
            />
            <Row
              label={`forward, attention @ T=${o.T.toLocaleString("en-US")}`}
              value={formatFlops(a.flops.fwdAttention)}
              aside={pct(a.flops.fwdAttention, a.flops.fwdTotal)}
              title="Scores and the value product, counted as a causal kernel actually does them."
            />
            <Row label="forward, total" value={formatFlops(a.flops.fwdTotal)} aside="per token" />
            <Row
              label="unmasked total"
              value={formatFlops(a.flops.fwdTotalUnmasked)}
              aside="profiler"
              title="What torch.utils.flop_counter reports: the attention operator's shape does not depend on the mask, so a profiler counts the half a causal kernel skips."
            />
            <Row
              label="elementwise"
              value={formatFlops(a.flops.elementwise)}
              aside="excluded"
              title="Norms, activations, RoPE and residual adds. Memory-bound, so they are reported but not added to the matmul totals."
            />
            <Row label="training" value={formatFlops(a.flops.trainPerToken)} aside="per token" />
          </tbody>
        </table>

        <h4>Against the rules of thumb</h4>
        <table className="table">
          <tbody>
            <Row
              label="2N inference"
              value={formatFlops(a.flops.ruleOfThumb2N)}
              aside={`${((a.flops.fwdTotal / a.flops.ruleOfThumb2N) * 100).toFixed(0)}%`}
              title="2 x non-embedding active parameters."
            />
            <Row
              label="6N training"
              value={formatFlops(a.flops.ruleOfThumb6N)}
              aside={`${((a.flops.trainPerToken / a.flops.ruleOfThumb6N) * 100).toFixed(0)}%`}
              title="6 x non-embedding active parameters."
            />
            <Row
              label="attention share"
              value={`${(a.flops.attentionShare * 100).toFixed(1)}%`}
              aside={a.flops.attentionShare > 0.3 ? "context-dominated" : "weight-dominated"}
            />
          </tbody>
        </table>
      </Section>

      <Section id="memory" title="Memory" note={formatBytes(perGpu.total)}>
        <h4>
          Training, per GPU &mdash; {o.parallel.dp}×DP {o.parallel.tp}×TP {o.parallel.pp}×PP, ZeRO-
          {o.parallel.zero}
        </h4>
        <Budget
          capacity={hw.memory}
          capacityLabel={`${formatBytes(hw.memory)} on ${hw.name}`}
          parts={[
            {
              label: "weights",
              bytes: perGpu.weights,
              tone: "var(--part-linear-edge)",
            },
            {
              label: "grads",
              bytes: perGpu.grads,
              tone: "var(--part-attention-edge)",
            },
            {
              label: "optimizer",
              bytes: perGpu.optimizer,
              tone: "var(--part-moe-edge)",
            },
            {
              label: "activations",
              bytes: perGpu.activations,
              tone: "var(--part-mlp-edge)",
            },
          ]}
        />
        <table className="table">
          <tbody>
            <Row label="optimizer" value={a.memory.optimizerLabel} />
            <Row label="recompute" value={o.recompute} />
            <Row
              label="activations, unsharded"
              value={formatBytes(a.memory.train.activations)}
              aside={`B=${o.B} T=${o.T}`}
            />
            <Row
              label="of which logits"
              value={formatBytes(a.memory.train.logits)}
              aside={pct(a.memory.train.logits, a.memory.train.activations)}
              title="The vocabulary projection's output. Often the single largest activation in a small model."
            />
          </tbody>
        </table>

        <h4>Serving, one replica</h4>
        <table className="table">
          <tbody>
            <Row
              label="weights"
              value={formatBytes(a.memory.infer.weights)}
              aside={o.inferenceDtype}
            />
            <Row
              label={`KV cache, ${o.concurrency}×${o.T.toLocaleString("en-US")}`}
              value={formatBytes(a.memory.infer.kv)}
              aside={o.kvDtype}
            />
            <Row label="overhead" value={formatBytes(a.memory.infer.overhead)} />
            <Row
              label="total"
              value={formatBytes(a.memory.infer.total)}
              aside={`of ${formatBytes(hw.memory)}`}
            />
          </tbody>
        </table>
        {a.memory.notes.map((n) => (
          <p className="hint" key={n}>
            {n}
          </p>
        ))}
      </Section>

      <Section
        id="serving"
        title="Serving"
        note={`${a.throughput.decodeTokensPerSecond.toFixed(0)} tok/s`}
      >
        <table className="table">
          <tbody>
            <Row
              label="KV per token"
              value={formatBytes(a.kv.bytesPerToken)}
              aside={`${formatBytes(kvPer1k)}/1k`}
            />
            {doc.meta.published?.kvBytesPerToken !== undefined && (
              <Row
                label="published KV"
                value={formatBytes(doc.meta.published.kvBytesPerToken)}
                aside={
                  a.kv.bytesPerToken === doc.meta.published.kvBytesPerToken ? (
                    <span className="ok">exact</span>
                  ) : (
                    <span className="warn">differs</span>
                  )
                }
              />
            )}
            <Row
              label="decode"
              value={`${a.throughput.decodeTokensPerSecond.toFixed(1)} tok/s`}
              aside="per step"
            />
            <Row
              label="bound by"
              value={a.throughput.memoryBound ? "memory bandwidth" : "arithmetic"}
              aside={`ridge ${a.throughput.ridgePoint.toFixed(0)} F/B`}
              title="Above the ridge point the device is compute-bound; below it, bandwidth sets the pace."
            />
            <Row label="bytes per step" value={formatBytes(a.throughput.decodeBytesPerStep)} />
            <Row
              label="prefill"
              value={`${(a.throughput.prefillSeconds * 1000).toFixed(1)} ms`}
              aside={`B=${o.B} T=${o.T}`}
            />
          </tbody>
        </table>
        {a.throughput.notes.map((n) => (
          <p className="hint" key={n}>
            {n}
          </p>
        ))}
      </Section>

      <Section
        id="training"
        title="Training"
        defaultOpen={false}
        note={formatHours(a.cost.wallClockHours)}
      >
        <table className="table">
          <tbody>
            <Row
              label="token budget"
              value={formatCount(a.cost.tokens)}
              aside={o.tokensWereDefaulted ? "Chinchilla" : "chosen"}
            />
            <Row label="tokens per parameter" value={a.chinchilla.tokensPerParam.toFixed(1)} />
            {p.active !== p.total && (
              <Row
                label="per active parameter"
                value={a.chinchilla.tokensPerActiveParam.toFixed(1)}
                title="The meaningful ratio for a sparse model."
              />
            )}
            <Row label="total FLOPs" value={formatFlops(a.cost.totalFlops)} />
            <Row
              label="GPU-hours"
              value={formatHours(a.cost.gpuHours)}
              aside={`MFU ${(o.mfu * 100).toFixed(0)}%`}
            />
            <Row
              label="wall clock"
              value={formatHours(a.cost.wallClockHours)}
              aside={`${o.gpus} GPUs`}
            />
            <Row
              label="indicative cost"
              value={formatDollars(a.cost.dollars)}
              aside={`$${hw.pricePerHour}/h`}
            />
          </tbody>
        </table>
        <p className="hint">{a.chinchilla.verdict}</p>
        {Object.keys(a.chinchilla.predictedLoss).length > 0 && (
          <>
            <h4>Predicted loss</h4>
            <table className="table">
              <tbody>
                {Object.entries(a.chinchilla.predictedLoss).map(([fit, loss]) => (
                  <Row key={fit} label={fit} value={loss.toFixed(3)} aside="nats/token" />
                ))}
              </tbody>
            </table>
          </>
        )}
      </Section>

      <p className="hint">
        Measured at B={o.B}, T={o.T.toLocaleString("en-US")}, {o.dtype} ({DTYPE_BYTES[o.dtype]}{" "}
        B/param) on {hw.name}. Analysis took {derived.elapsedMs.toFixed(1)} ms.
      </p>
    </div>
  );
}
