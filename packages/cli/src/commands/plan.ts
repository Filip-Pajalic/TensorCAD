/**
 * `tensorcad plan <file|preset> --gpus n` — how to split the work.
 *
 * The analysis prices one way of sharding a design. This asks the engine to
 * price every way the cluster admits and prints the ones that fit, least
 * demanding first. Exits 1 when nothing fits, which makes it usable as a check
 * before a run is queued.
 */

import { bool, num, str, type Args } from "../args.js";
import { loadDesign } from "../load.js";
import { analysisOptions } from "../options.js";
import { bold, dim, heading, pad, padStart, red, writeOut, yellow } from "../format.js";
import type { ClusterPlan, ClusterRequest, Recompute } from "@tensorcad/engine";
import { formatBytes } from "@tensorcad/engine";
import { planCluster } from "@tensorcad/engine/node";

/** A comma-separated list of numbers, for `--micro-batch 1,2,4`. */
function numbers(args: Args, name: string): number[] | undefined {
  const raw = str(args, name);
  if (raw === undefined) return undefined;
  const out = raw
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && n > 0);
  return out.length > 0 ? out : undefined;
}

function line(p: ClusterPlan, width: number): string {
  const used = `${Math.round(p.used * 100)}%`;
  const paint = p.used > 0.85 ? yellow : (s: string) => s;
  return (
    `  ${pad(p.summary, width)}  ${padStart(formatBytes(p.perGpu.total), 10)}` +
    `  ${paint(padStart(used, 4))} ${dim("of budget")}`
  );
}

export function cmdPlan(args: Args): number {
  const { doc } = loadDesign(args._[0]);
  const gpus = num(args, "gpus");
  if (gpus === undefined) {
    writeOut(red("plan needs --gpus: how many devices there are."));
    return 1;
  }
  const cluster: ClusterRequest = {
    gpus,
    gpusPerNode: num(args, "gpus-per-node"),
    headroom: num(args, "headroom"),
    microBatch: numbers(args, "micro-batch"),
    limit: num(args, "limit"),
  };
  const recompute = str(args, "recompute");
  if (recompute) cluster.recompute = [recompute as Recompute];

  // The operating point without the GPU count: that is what is being searched
  // for, not something to be told.
  const options = analysisOptions(args);
  const result = planCluster(doc, options, cluster);

  if (bool(args, "json")) {
    writeOut(JSON.stringify(result, null, 2));
    return result.fits.length > 0 ? 0 : 1;
  }

  const out: string[] = [
    heading(`${bold(doc.meta.name)} on ${gpus} x ${result.hardware}`),
    dim(
      `  ${formatBytes(result.budget)} per device after headroom, ` +
        `of ${formatBytes(result.memory)}. ${result.considered} plans priced.`,
    ),
    "",
  ];

  if (result.fits.length === 0) {
    out.push(red("  Nothing fits."));
    if (result.closest) {
      out.push(
        `  ${dim("closest:")} ${result.closest.summary} at ${formatBytes(result.closest.perGpu.total)}`,
      );
    }
  } else {
    const width = Math.max(...result.fits.map((p) => p.summary.length));
    for (const p of result.fits) out.push(line(p, width));
    out.push("");
    out.push(heading(bold(result.fits[0].summary)));
    const g = result.fits[0].perGpu;
    out.push(
      `  ${dim("weights")} ${formatBytes(g.weights)}   ${dim("gradients")} ${formatBytes(g.grads)}` +
        `   ${dim("optimizer")} ${formatBytes(g.optimizer)}   ${dim("activations")} ${formatBytes(g.activations)}`,
    );
    for (const n of result.fits[0].notes) out.push(`  ${dim("*")} ${n}`);
  }

  for (const n of result.notes) out.push(`  ${dim("note:")} ${n}`);
  writeOut(out.join("\n"));
  return result.fits.length > 0 ? 0 : 1;
}
