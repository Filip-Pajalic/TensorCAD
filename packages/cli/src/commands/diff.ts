/**
 * `tensorcad diff <a> <b>` — what changed between two designs, structurally and
 * numerically.
 */

import { bool, UsageError, type Args } from "../args.js";
import { loadDesign } from "../load.js";
import { analysisOptions, impliedGpus } from "../options.js";
import { bold, dim, green, heading, pad, red, writeOut, yellow } from "../format.js";
import type { DiffDelta } from "@tensor-cad/engine";
import { diffDesigns } from "@tensor-cad/engine/node";
import { formatBytes, formatCount, formatFlops } from "@tensor-cad/engine";

const FORMATTERS: Record<string, (n: number) => string> = {
  parameters: formatCount,
  "active parameters": formatCount,
  "non-embedding parameters": formatCount,
  "training FLOPs/token": formatFlops,
  "forward FLOPs/token": formatFlops,
  "KV bytes/token": formatBytes,
  "training memory/GPU": formatBytes,
  "training memory total": formatBytes,
  "serving memory": formatBytes,
};

function fmt(m: DiffDelta, v: number): string {
  const f = FORMATTERS[m.metric] ?? ((n: number) => String(n));
  return f(v);
}

function short(v: unknown): string {
  const s = JSON.stringify(v ?? null);
  return s.length > 60 ? `${s.slice(0, 57)}...` : s;
}

/**
 * A symbol as a reader wants to see it.
 *
 * `D` going from 768 to 1024 is the whole change; printing the documentation
 * string and the kind either side of it buries that in sixty characters of
 * things that did not move. When something other than the value *did* move, the
 * whole definition is what to show.
 */
function symbolText(before: unknown, after: unknown): [string, string] {
  const value = (v: unknown): string | null => {
    if (typeof v === "number" || typeof v === "string") return String(v);
    if (v && typeof v === "object") {
      const o = v as Record<string, unknown>;
      const n = o.value ?? o.expr ?? o.default;
      if (typeof n === "number" || typeof n === "string") return String(n);
    }
    return null;
  };
  const rest = (v: unknown): string => {
    if (!v || typeof v !== "object") return "";
    const { value: _v, expr: _e, default: _d, ...other } = v as Record<string, unknown>;
    return JSON.stringify(other);
  };
  const a = value(before);
  const b = value(after);
  if (a !== null && b !== null && rest(before) === rest(after)) return [a, b];
  return [short(before), short(after)];
}

export function cmdDiff(args: Args): number {
  if (args._.length < 2) throw new UsageError("diff needs two designs: tensorcad diff <a> <b>");
  const a = loadDesign(args._[0]);
  const b = loadDesign(args._[1]);
  const options = impliedGpus(analysisOptions(args));
  const d = diffDesigns(a.doc, b.doc, options);

  if (bool(args, "json")) {
    writeOut(JSON.stringify(d, null, 2));
    return 0;
  }

  const out: string[] = [`${bold(d.a)} ${dim("->")} ${bold(d.b)}`];

  const { symbols, blocks, edges } = d;
  if (symbols.added.length || symbols.removed.length || symbols.changed.length) {
    out.push(heading("Symbols"));
    for (const s of symbols.changed) {
      const [was, now] = symbolText(s.from, s.to);
      out.push(`  ${yellow("~")} ${s.name}  ${dim(was)} ${dim("->")} ${now}`);
    }
    // An addition has only a `to` and a removal only a `from`: one shape for
    // every kind of change rather than a second one for the two that are
    // one-sided.
    for (const s of symbols.added) out.push(`  ${green("+")} ${s.name} = ${symbolText(s.to, s.to)[0]}`);
    for (const s of symbols.removed) out.push(`  ${red("-")} ${s.name} = ${symbolText(s.from, s.from)[0]}`);
  }

  if (blocks.added.length || blocks.removed.length || blocks.changed.length) {
    out.push(heading("Blocks"));
    for (const blk of blocks.added) out.push(`  ${green("+")} ${blk.path}  ${dim(blk.type)}`);
    for (const blk of blocks.removed) out.push(`  ${red("-")} ${blk.path}  ${dim(blk.type)}`);
    for (const c of blocks.changed) {
      out.push(`  ${yellow("~")} ${c.path}`);
      if (c.type) out.push(`      ${dim("type")}  ${c.type.from} ${dim("->")} ${c.type.to}`);
      if (c.label) out.push(`      ${dim("label")}  ${short(c.label.from)} ${dim("->")} ${short(c.label.to)}`);
      for (const p of c.params) {
        out.push(`      ${dim(p.key)}  ${short(p.from)} ${dim("->")} ${short(p.to)}`);
      }
    }
  }

  if (edges.added.length || edges.removed.length) {
    out.push(heading("Edges"));
    for (const e of edges.added) out.push(`  ${green("+")} ${dim(e.graph)}  ${e.from} -> ${e.to}`);
    for (const e of edges.removed) out.push(`  ${red("-")} ${dim(e.graph)}  ${e.from} -> ${e.to}`);
  }

  if (d.identical) out.push(dim("\nThe two documents are structurally identical."));

  out.push(heading(`Numbers  ${dim(`T=${d.at.T} B=${d.at.B} ${d.at.hardware}`)}`));
  const nameWidth = Math.max(...d.metrics.map((m) => m.metric.length));
  const aWidth = Math.max(...d.metrics.map((m) => fmt(m, m.a).length));
  const bWidth = Math.max(...d.metrics.map((m) => fmt(m, m.b).length));
  for (const m of d.metrics) {
    const changed = m.delta !== 0;
    const paint = !changed ? dim : m.delta > 0 ? green : red;
    const sign = m.delta > 0 ? "+" : "";
    const rel = m.ratio === null ? "" : dim(`  ${sign}${((m.ratio - 1) * 100).toFixed(1)}%`);
    out.push(
      `  ${pad(m.metric, nameWidth)}  ${pad(fmt(m, m.a), aWidth)} ${dim("->")} ${pad(fmt(m, m.b), bWidth)}` +
        `  ${paint(`${sign}${fmt(m, m.delta)}`)}${rel}`,
    );
  }

  writeOut(out.join("\n"));
  return 0;
}
