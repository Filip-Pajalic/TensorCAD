/**
 * `tensorcad mup <file|preset>` — the same design at several widths.
 *
 * A learning rate tuned on a narrow model is the right one for a wide model
 * too, provided the initialization and the per-parameter rates are scaled by
 * width the way Tensor Programs V's Table 3 says. This prints the ladder: what
 * each rung costs, and what to multiply by at each.
 *
 * What it does not print is a learning rate. That is what the sweep at the base
 * rung is for.
 */

import { bool, num, str, type Args } from "../args.js";
import { loadDesign } from "../load.js";
import { bold, dim, heading, pad, padStart, writeOut } from "../format.js";
import type { MupOptions, MupRung } from "@tensor-cad/engine";
import { formatCount } from "@tensor-cad/engine";
import { mupLadder } from "@tensor-cad/engine/node";

/** A comma-separated list of widths, for `--widths 256,512,1024`. */
function widths(args: Args): number[] | undefined {
  const raw = str(args, "widths");
  if (raw === undefined) return undefined;
  const out = raw
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && n > 0);
  return out.length > 0 ? out : undefined;
}

/** A multiplier, printed so that 1 reads as "unchanged" rather than as 1.000. */
function times(v: number): string {
  if (v === 1) return dim("unchanged");
  return `x${v.toPrecision(4).replace(/\.?0+$/, "")}`;
}

/** The blocks in a class, named while the list is short enough to read. */
function weights(paths: string[]): string {
  if (paths.length === 0) return "nothing in this design";
  const shown = paths.slice(0, 4).join(", ");
  const rest = paths.length > 4 ? ` and ${paths.length - 4} more` : "";
  return `${paths.length} ${paths.length === 1 ? "weight" : "weights"}: ${shown}${rest}`;
}

function rungLine(r: MupRung, width: number): string {
  const mark = r.base ? bold("base") : "    ";
  return (
    `  ${mark} ${padStart(String(r.width), width)} ${dim("wide")}` +
    `  ${padStart(String(r.heads), 3)} ${dim("heads")}` +
    `  ${padStart(formatCount(r.params), 10)}` +
    `  ${dim("m =")} ${r.multiplier}`
  );
}

export function cmdMup(args: Args): number {
  const { doc } = loadDesign(args._[0]);
  const options: MupOptions = { widths: widths(args), baseWidth: num(args, "base-width") };
  const ladder = mupLadder(doc, options);

  if (bool(args, "json")) {
    writeOut(JSON.stringify(ladder, null, 2));
    return 0;
  }

  const out: string[] = [
    heading(`${bold(doc.meta.name)} laddered by ${ladder.widthSymbol}`),
    dim(
      `  ${ladder.rungs.length} rungs, tuned at ${ladder.baseWidth}, ` +
        `heads of ${ladder.headDim} throughout.`,
    ),
    "",
  ];

  const w = Math.max(...ladder.rungs.map((r) => String(r.width).length));
  for (const r of ladder.rungs) out.push(rungLine(r, w));
  out.push("");

  // One table for the whole ladder would repeat the class names at every rung.
  // The rungs differ only in their multipliers, so the classes are named once
  // and the rungs are the columns.
  const classes = ladder.rungs[0]?.scaling ?? [];
  const nameWidth = Math.max(...classes.map((s) => s.class.length), 6);
  const cells = (v: (r: (typeof ladder.rungs)[number]) => string) =>
    ladder.rungs.map((r) => padStart(v(r), 14)).join("");
  out.push(heading("Multiply the base model's settings by"));
  out.push(`  ${pad("", nameWidth)}${cells((r) => String(r.width))}`);
  for (const [i, c] of classes.entries()) {
    out.push(`  ${bold(c.class)}`);
    out.push(`  ${pad("  init", nameWidth)}${cells((r) => times(r.scaling[i]!.initStd))}`);
    out.push(`  ${pad("  rate", nameWidth)}${cells((r) => times(r.scaling[i]!.adamLr))}`);
    out.push(`  ${dim(c.why)}`);
    out.push(`  ${dim(weights(c.paths))}`);
    out.push("");
  }

  // A rung's own notes are about that rung, so they are printed with it.
  for (const r of ladder.rungs) {
    for (const n of r.notes) out.push(`  ${dim(`${r.width}:`)} ${n}`);
  }
  for (const n of ladder.notes) out.push(`  ${dim("note:")} ${n}`);
  writeOut(out.join("\n"));
  return 0;
}
