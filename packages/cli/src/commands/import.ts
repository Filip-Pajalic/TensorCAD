/**
 * `tensorcad import <config.json>` — read a Hugging Face config into a design.
 *
 * The fastest honest way to get a design for a published model that is not one
 * of the twenty presets: point this at the `config.json` from its model card.
 * Anything the importer cannot model faithfully comes back as a warning rather
 * than being approximated silently, and those warnings are printed to stderr so
 * a redirected `--json` stays machine-readable.
 */

import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, resolve } from "node:path";
import { bool, str, type Args } from "../args.js";
import { UsageError } from "../args.js";
import { bold, dim, writeErr, writeOut, yellow } from "../format.js";
import { formatCount } from "@tensorcad/engine";
import { analyze, importHfConfig, validate } from "@tensorcad/engine/node";

/** A name for the design, from `--name`, the config's own, or the directory. */
function nameFor(args: Args, path: string): string | undefined {
  const given = str(args, "name");
  if (given) return given;
  // `.../Meta-Llama-3-8B/config.json` names the model in its directory, which
  // is more use than "config".
  const stem = basename(path, extname(path));
  if (stem !== "config") return stem;
  const dir = basename(dirname(resolve(path)));
  return dir && dir !== "." ? dir : undefined;
}

export function cmdImport(args: Args): number {
  const path = args._[0];
  if (!path) throw new UsageError("import needs a config: tensorcad import <config.json>");

  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (e) {
    writeOut(`Could not read ${path}: ${(e as Error).message}`);
    return 1;
  }

  const { doc, warnings } = importHfConfig(text, nameFor(args, path));
  const report = validate(doc);
  const params = analyze(doc).params;

  const out = str(args, "out");
  if (out) {
    const target = out.endsWith(".json") ? out : `${out}/${doc.meta.name}.tensorcad.json`;
    mkdirSync(dirname(resolve(target)), { recursive: true });
    writeFileSync(target, `${JSON.stringify(doc, null, 2)}\n`, "utf8");
    writeErr(`wrote ${target}`);
  }

  if (bool(args, "json")) {
    writeOut(JSON.stringify({ doc, warnings, params: params.total }, null, 2));
  } else {
    writeOut(
      [
        `${bold(doc.meta.name)}  ${formatCount(params.total)} parameters`,
        `  ${dim("non-embedding")} ${formatCount(params.nonEmbedding)}` +
          (params.active !== params.total ? `  ${dim("active")} ${formatCount(params.active)}` : ""),
        `  ${dim("rules")} ${report.counts.error} errors, ${report.counts.warning} warnings`,
        out ? "" : dim("  (--out <dir|file> to save it, --json for the document)"),
      ]
        .filter(Boolean)
        .join("\n"),
    );
  }

  // To stderr, so a redirected --json is still a document and nothing else.
  for (const w of warnings) writeErr(yellow(`warning: ${w}`));

  // An import that could not model the architecture faithfully is not a
  // success, whatever the parameter count says.
  return warnings.length > 0 || report.counts.error > 0 ? 1 : 0;
}
