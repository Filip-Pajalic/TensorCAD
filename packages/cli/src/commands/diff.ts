/**
 * `tensorcad diff <a> <b>` — what changed between two designs, structurally and
 * numerically.
 */

import { bool, UsageError, type Args } from "../args.js";
import { loadDesign } from "../load.js";
import { analysisOptions, impliedGpus } from "../options.js";
import { bold, dim, green, heading, pad, red, writeOut, yellow } from "../format.js";
import { diffDesigns, type NumericDelta } from "../diff.js";
import { formatBytes, formatCount, formatFlops } from "@tensorcad/engine";

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

function fmt(m: NumericDelta, v: number): string {
  const f = FORMATTERS[m.metric] ?? ((n: number) => String(n));
  return f(v);
}

function short(v: unknown): string {
  const s = JSON.stringify(v ?? null);
  return s.length > 60 ? `${s.slice(0, 57)}...` : s;
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
      out.push(`  ${yellow("~")} ${s.name}  ${dim(short(s.from))} ${dim("->")} ${short(s.to)}`);
    }
    for (const s of symbols.added) out.push(`  ${green("+")} ${s.name} = ${short(s.value)}`);
    for (const s of symbols.removed) out.push(`  ${red("-")} ${s.name} = ${short(s.value)}`);
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
