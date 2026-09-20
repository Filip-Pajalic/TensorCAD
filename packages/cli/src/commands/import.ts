/**
 * `tensorcad import <config.json>` — a Hugging Face config as a design.
 *
 * The importer reproduces a config's layout, not the authors' arithmetic, so
 * the two things this prints beside the file it wrote are the warnings about
 * what it could not represent and the parameter count the analysis gets. Both
 * are there to be checked against the model card before anyone trusts the
 * document.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { bool, str, UsageError, type Args } from "../args.js";
import { bold, dim, green, writeOut, yellow } from "../format.js";
import { formatCount } from "@tensor-cad/engine";
import { countParams, importHfConfig } from "@tensor-cad/engine/node";

export function cmdImport(args: Args): number {
  const ref = args._[0];
  if (!ref) {
    throw new UsageError("Missing <config.json>: the path to a Hugging Face model config.");
  }

  const source = resolve(process.cwd(), ref);
  let text: string;
  try {
    text = readFileSync(source, "utf8");
  } catch {
    throw new UsageError(`Could not read "${ref}". It should be a Hugging Face config.json.`);
  }

  // An absent --name leaves the choice to the importer, which takes the
  // config's own `_name_or_path` and falls back to the model type.
  const { doc, warnings } = importHfConfig(text, str(args, "name"));
  const params = countParams(doc);

  if (bool(args, "json")) {
    writeOut(JSON.stringify({ name: doc.meta.name, doc, warnings, params: params.total }, null, 2));
    return 0;
  }

  const out = resolve(process.cwd(), str(args, "out") ?? join("out", `${slug(doc.meta.name)}.tensorcad.json`));
  // Two spaces and a trailing newline: the shape the documents in
  // `packages/core-go/presets/data` are in, so an import can be dropped there
  // unedited.
  const body = JSON.stringify(doc, null, 2);
  const contents = `${body}\n`;
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, contents, "utf8");

  const lines: string[] = [
    `${bold(doc.meta.name)}  ${dim(`<- ${ref}`)}`,
    `  ${green("wrote")} ${out}  ${dim(`${body.split("\n").length} lines, ${Buffer.byteLength(contents, "utf8")} B`)}`,
    "",
    `  parameters  ${bold(formatCount(params.total))}  ${dim(params.total.toLocaleString("en-US"))}`,
    dim("  An import reproduces the config, not the model card. Check that count against it."),
  ];

  if (warnings.length > 0) {
    lines.push("");
    for (const w of warnings) lines.push(`  ${yellow("warning")} ${w}`);
  }

  writeOut(lines.join("\n"));
  return 0;
}

/** A model name can be a Hub path (`meta-llama/Llama-3-8B`); a file name cannot. */
function slug(name: string): string {
  return name.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "imported";
}
