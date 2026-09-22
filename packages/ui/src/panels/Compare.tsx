/**
 * What changed, and what it cost.
 *
 * Two questions, and the answer to one without the other is misleading: that
 * `F` went from 11008 to 14336 does not tell you the model grew by 1.3B
 * parameters, and that it grew by 1.3B does not tell you where. The engine
 * reports both, measured at one operating point so the attention terms and the
 * activation memory are comparable, and this draws them side by side.
 *
 * A dialog rather than a tab, because comparing is something you open, read and
 * close, and a permanent fifth tab would cost the four that are used constantly.
 */

import { useMemo, useState } from "react";
import { useEditor } from "../state/store.js";
import { toAnalysisOptions } from "../state/operating.js";
import { blockDef, diffDesigns, getPreset, PRESET_NAMES } from "../engine.js";
import { typeName } from "../canvas/blocks.js";
import { formatBytes, formatCount, formatFlops } from "@tensor-cad/engine";
import type { DesignDiff, DiffDelta } from "@tensor-cad/engine";

/** The other side of the comparison: where this design started, or a preset. */
const AS_OPENED = "\u0000opened";

/**
 * How to read a number, from its name.
 *
 * The engine returns bytes, FLOPs and counts in one list, and a byte count
 * shown as "2,621,440,000" is a number nobody can weigh against another.
 */
function scale(metric: string, v: number): string {
  if (metric.includes("memory") || metric.includes("bytes")) return formatBytes(v);
  if (metric.includes("FLOPs")) return formatFlops(v);
  return formatCount(v);
}

function Row({ m }: { m: DiffDelta }): React.ReactElement {
  const moved = m.delta !== 0;
  const pct = m.ratio === null ? null : (m.ratio - 1) * 100;
  return (
    <tr className={moved ? "" : "cmp__still"}>
      <td>{m.metric}</td>
      <td className="mono cmp__num">{scale(m.metric, m.a)}</td>
      <td className="mono cmp__num">{scale(m.metric, m.b)}</td>
      <td className={`mono cmp__num${moved ? (m.delta > 0 ? " cmp__up" : " cmp__down") : ""}`}>
        {moved ? `${m.delta > 0 ? "+" : "−"}${scale(m.metric, Math.abs(m.delta))}` : "—"}
      </td>
      <td className={`mono cmp__num${moved ? (m.delta > 0 ? " cmp__up" : " cmp__down") : ""}`}>
        {pct === null || !moved ? "" : `${pct > 0 ? "+" : "−"}${Math.abs(pct).toFixed(1)}%`}
      </td>
    </tr>
  );
}

/** A value as the document writes it, shortened. */
function short(v: unknown): string {
  if (v === undefined || v === null) return "—";
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  if (typeof v === "string") return v;
  const o = v as Record<string, unknown>;
  const n = o.value ?? o.expr ?? o.default;
  if (typeof n === "number" || typeof n === "string") return String(n);
  const s = JSON.stringify(v);
  return s.length > 40 ? `${s.slice(0, 37)}…` : s;
}

function Moved({ from, to }: { from: unknown; to: unknown }): React.ReactElement {
  return (
    <span className="mono">
      <span className="cmp__from">{short(from)}</span>
      <span className="cmp__arrow"> → </span>
      {short(to)}
    </span>
  );
}

export default function Compare(): React.ReactElement {
  const doc = useEditor((s) => s.doc);
  const opened = useEditor((s) => s.opened);
  const operating = useEditor((s) => s.operating);
  const [against, setAgainst] = useState<string>(AS_OPENED);

  const result = useMemo((): DesignDiff | { error: string } => {
    try {
      const other = against === AS_OPENED ? opened : getPreset(against);
      // `a` is what it was and `b` is what it is, so a positive delta reads as
      // growth in the design on screen.
      return diffDesigns(other, doc, toAnalysisOptions(operating));
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) };
    }
  }, [doc, opened, against, operating]);

  const picker = (
    <label className="cmp__pick">
      against
      <select
        className="field"
        value={against}
        onChange={(e) => setAgainst(e.target.value)}
        title="What to compare this design with"
      >
        <option value={AS_OPENED}>{opened.meta.name} (as opened)</option>
        {PRESET_NAMES.map((n) => (
          <option key={n} value={n}>
            {n}
          </option>
        ))}
      </select>
    </label>
  );

  if ("error" in result) {
    return (
      <div className="cmp">
        {picker}
        <div className="empty">{result.error}</div>
      </div>
    );
  }

  const { symbols, blocks, edges } = result;
  const structural =
    symbols.added.length +
    symbols.removed.length +
    symbols.changed.length +
    blocks.added.length +
    blocks.removed.length +
    blocks.changed.length +
    edges.added.length +
    edges.removed.length;

  return (
    <div className="cmp">
      {picker}
      <div className="cmp__head mono">
        {result.a} <span className="cmp__arrow">→</span> {result.b}
        <span className="dim">
          {" "}
          at T={result.at.T}, B={result.at.B}
        </span>
      </div>

      {result.identical ? (
        <div className="empty">
          Structurally identical: the same symbols, the same blocks, the same wiring.
        </div>
      ) : (
        <div className="cmp__structure">
          {symbols.changed.map((s) => (
            <div className="cmp__line" key={`sc${s.name}`}>
              <span className="cmp__mark cmp__mark--change">~</span>
              <span className="mono">{s.name}</span>
              <Moved from={s.from} to={s.to} />
            </div>
          ))}
          {symbols.added.map((s) => (
            <div className="cmp__line" key={`sa${s.name}`}>
              <span className="cmp__mark cmp__mark--add">+</span>
              <span className="mono">{s.name}</span>
              <span className="mono">{short(s.to)}</span>
            </div>
          ))}
          {symbols.removed.map((s) => (
            <div className="cmp__line" key={`sr${s.name}`}>
              <span className="cmp__mark cmp__mark--remove">−</span>
              <span className="mono">{s.name}</span>
              <span className="mono">{short(s.from)}</span>
            </div>
          ))}
          {blocks.added.map((b) => (
            <div className="cmp__line" key={`ba${b.path}`}>
              <span className="cmp__mark cmp__mark--add">+</span>
              <button className="cmp__path mono" onClick={() => useEditor.getState().focusOn(b.path)}>
                {b.path}
              </button>
              <span className="dim" title={b.type}>
                {typeName(blockDef(b.type, doc), b.type)}
              </span>
            </div>
          ))}
          {blocks.removed.map((b) => (
            <div className="cmp__line" key={`br${b.path}`}>
              <span className="cmp__mark cmp__mark--remove">−</span>
              <span className="mono">{b.path}</span>
              <span className="dim" title={b.type}>
                {typeName(blockDef(b.type, doc), b.type)}
              </span>
            </div>
          ))}
          {blocks.changed.map((c) => (
            <div className="cmp__line cmp__line--block" key={`bc${c.path}`}>
              <span className="cmp__mark cmp__mark--change">~</span>
              <button className="cmp__path mono" onClick={() => useEditor.getState().focusOn(c.path)}>
                {c.path}
              </button>
              <div className="cmp__params">
                {c.type && (
                  <div>
                    <span className="dim">type </span>
                    <Moved from={c.type.from} to={c.type.to} />
                  </div>
                )}
                {c.label && (
                  <div>
                    <span className="dim">label </span>
                    <Moved from={c.label.from} to={c.label.to} />
                  </div>
                )}
                {c.params.map((p) => (
                  <div key={p.key}>
                    <span className="dim">{p.key} </span>
                    <Moved from={p.from} to={p.to} />
                  </div>
                ))}
              </div>
            </div>
          ))}
          {(edges.added.length > 0 || edges.removed.length > 0) && (
            <div className="cmp__line">
              <span className="cmp__mark cmp__mark--change">~</span>
              <span className="dim">
                wiring: {edges.added.length} added, {edges.removed.length} removed
              </span>
            </div>
          )}
        </div>
      )}

      <table className="cmp__table">
        <thead>
          <tr>
            <th>{structural === 0 ? "measured" : "cost"}</th>
            <th className="cmp__num">{result.a}</th>
            <th className="cmp__num">{result.b}</th>
            <th className="cmp__num">change</th>
            <th className="cmp__num" />
          </tr>
        </thead>
        <tbody>
          {result.metrics.map((m) => (
            <Row key={m.metric} m={m} />
          ))}
        </tbody>
      </table>
    </div>
  );
}
