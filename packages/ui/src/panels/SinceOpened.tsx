/**
 * What your changes did, where you are looking.
 *
 * The editor has always kept the design as it was opened, and the Compare
 * dialog has always diffed against it — two menus deep, where nobody trying
 * "what if this were grouped-query" goes. This is the same diff, reduced to
 * one strip under the parameter count: what moved, in the words the inspector
 * uses, and what that did to the four numbers a change is usually made for.
 *
 * It appears only once the design differs from where it started, and it is
 * measured a moment after the last edit rather than on every keystroke: a diff
 * analyses both designs, and the readout has already analysed one of them.
 */

import { useEffect, useState } from "react";
import type { DesignDiff, DiffDelta, Doc } from "@tensor-cad/engine";
import { useEditor } from "../state/store.js";
import { toAnalysisOptions } from "../state/operating.js";
import * as ops from "../state/ops.js";
import { blockDef, diffDesigns } from "../engine.js";
import { typeName } from "../canvas/blocks.js";

/** The numbers worth a glance, by the engine's name, with what to call them here. */
const HEADLINE: [metric: string, name: string][] = [
  ["parameters", "parameters"],
  ["training FLOPs/token", "compute"],
  ["KV bytes/token", "cache"],
  ["training memory/GPU", "memory/GPU"],
];

/** A value as a person reads it: 14,336, true, gqa. */
function shown(v: unknown): string {
  if (v === undefined || v === null) return "unset";
  if (typeof v === "number") return v.toLocaleString("en-US");
  if (typeof v === "string" || typeof v === "boolean") return String(v);
  return "…";
}

/**
 * A symbol's value from its definition, which a document writes as a number,
 * an expression, or an object carrying either with a description.
 */
function symbolValue(def: unknown): string {
  if (def && typeof def === "object") {
    const o = def as { value?: unknown; expr?: unknown; default?: unknown };
    return shown(o.value ?? o.expr ?? o.default);
  }
  return shown(def);
}

/**
 * What a symbol is called, from its own description when that is short:
 * "key/value heads" rather than `Hkv`. The description is the design's, so a
 * symbol nobody described keeps its name.
 */
function symbolName(name: string, def: unknown): string {
  const doc = def && typeof def === "object" ? (def as { doc?: unknown }).doc : undefined;
  if (typeof doc === "string" && doc.length > 0 && doc.length <= 28) {
    return doc[0]!.toLowerCase() + doc.slice(1);
  }
  return name;
}

/** The block a diff path names, in the design that has it. */
function nodeIn(doc: Doc, path: string): { type: string; label?: string; id: string } | null {
  const node = ops.nodeAtPath(doc, ops.segmentsOf(path));
  return node ? { type: node.type, label: node.label, id: node.id } : null;
}

/**
 * What moved, as sentences a person would say, most specific first.
 *
 * A parameter on a block reads by its label — "block: key/value heads 8 → 2" —
 * because the inspector says it that way and this is the same change.
 */
export function changesInWords(diff: DesignDiff, before: Doc, after: Doc): string[] {
  const out: string[] = [];
  for (const change of diff.blocks.changed) {
    const node = nodeIn(after, change.path) ?? nodeIn(before, change.path);
    const name = node ? (node.label ?? node.id) : change.path;
    const def = node ? blockDef(node.type, after) : undefined;
    if (change.type) {
      out.push(`${name} became ${shown(change.type.to)}`);
    }
    for (const p of change.params) {
      const label = (def?.params[p.key]?.label ?? p.key).toLowerCase();
      const words = def?.params[p.key]?.valueLabels ?? {};
      const say = (v: unknown) => (typeof v === "string" && words[v] ? words[v]! : shown(v));
      out.push(`${name}: ${label} ${say(p.from)} → ${say(p.to)}`);
    }
  }
  for (const s of diff.symbols.changed) {
    out.push(`${symbolName(s.name, s.to ?? s.from)} ${symbolValue(s.from)} → ${symbolValue(s.to)}`);
  }
  for (const s of diff.symbols.added) out.push(`added ${s.name}`);
  for (const s of diff.symbols.removed) out.push(`removed ${s.name}`);
  for (const b of diff.blocks.added) out.push(`added ${typeName(blockDef(b.type, after), b.type)}`);
  for (const b of diff.blocks.removed) out.push(`removed ${typeName(blockDef(b.type, before), b.type)}`);
  const wires = diff.edges.added.length + diff.edges.removed.length;
  if (wires > 0 && diff.blocks.added.length + diff.blocks.removed.length === 0) {
    out.push(`rewired ${wires} connection${wires === 1 ? "" : "s"}`);
  }
  return out;
}

/** A percentage the way a change is talked about: +12%, −75%, +0.4%. */
function percent(m: DiffDelta): string | null {
  if (m.delta === 0 || m.ratio === null) return null;
  const pct = (m.ratio - 1) * 100;
  const size = Math.abs(pct);
  return `${pct > 0 ? "+" : "−"}${size >= 10 ? size.toFixed(0) : size.toFixed(1)}%`;
}

export default function SinceOpened(): React.ReactElement | null {
  const doc = useEditor((s) => s.doc);
  const opened = useEditor((s) => s.opened);
  const operating = useEditor((s) => s.operating);
  const [diff, setDiff] = useState<DesignDiff | null>(null);
  const [stale, setStale] = useState(false);

  useEffect(() => {
    if (doc === opened) {
      setDiff(null);
      return;
    }
    setStale(true);
    const id = setTimeout(() => {
      try {
        setDiff(diffDesigns(opened, doc, toAnalysisOptions(operating)));
      } catch {
        // A design mid-edit that will not analyse has nothing to compare; the
        // readout below it is already saying why.
        setDiff(null);
      }
      setStale(false);
    }, 200);
    return () => clearTimeout(id);
  }, [doc, opened, operating]);

  if (!diff || diff.identical) return null;

  const changes = changesInWords(diff, opened, doc);
  const moved = HEADLINE.flatMap(([metric, name]) => {
    const m = diff.metrics.find((x) => x.metric === metric);
    const p = m ? percent(m) : null;
    return m && p ? [{ name, p, up: m.delta > 0 }] : [];
  });

  const act = useEditor.getState();
  return (
    <section className={"since" + (stale ? " is-stale" : "")} data-testid="since-opened" aria-live="polite">
      <div className="since__head">
        <span className="since__title">Since you opened it</span>
        <button type="button" className="linkish" onClick={() => act.openDialog("compare")}>
          Compare…
        </button>
        <button
          type="button"
          className="linkish"
          title="Measure later changes against the design as it is now"
          onClick={() => act.markOpened()}
        >
          Compare from here
        </button>
      </div>
      <p className="since__what">
        {changes.slice(0, 2).join("; ")}
        {changes.length > 2 && <span className="muted"> and {changes.length - 2} more</span>}
      </p>
      {moved.length > 0 ? (
        <div className="since__moved" data-testid="since-moved">
          {moved.map((m) => (
            <span key={m.name} className={"since__delta" + (m.up ? " is-up" : " is-down")}>
              {m.name} <b className="mono">{m.p}</b>
            </span>
          ))}
        </div>
      ) : (
        <p className="since__what muted">None of parameters, compute, cache or memory moved.</p>
      )}
    </section>
  );
}
