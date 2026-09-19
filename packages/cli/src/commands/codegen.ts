/**
 * `tensorcad codegen <file|preset> [--out dir]` — write the PyTorch module.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { bool, str, type Args } from "../args.js";
import { loadDesign } from "../load.js";
import { bold, dim, green, writeOut, yellow } from "../format.js";
import { generateTorch } from "@tensorcad/engine/node";

export function cmdCodegen(args: Args): number {
  const { doc } = loadDesign(args._[0]);
  const className = str(args, "class-name");
  const generated = generateTorch(doc, {
    ...(className ? { className } : {}),
    includeSmokeTest: bool(args, "smoke-test"),
  });

  if (bool(args, "json")) {
    writeOut(JSON.stringify({ name: doc.meta.name, files: generated.files, warnings: generated.warnings }, null, 2));
    return 0;
  }

  const outDir = resolve(process.cwd(), str(args, "out") ?? join("out", doc.meta.name));

  const lines: string[] = [bold(`${doc.meta.name} -> ${outDir}`)];
  for (const file of generated.files) {
    const target = join(outDir, file.path);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, file.contents, "utf8");
    const bytes = Buffer.byteLength(file.contents, "utf8");
    const lineCount = file.contents.split("\n").length;
    lines.push(`  ${green("wrote")} ${file.path}  ${dim(`${lineCount} lines, ${bytes} B`)}`);
  }

  if (generated.warnings.length > 0) {
    lines.push("");
    for (const w of generated.warnings) lines.push(`  ${yellow("warning")} ${w}`);
  }

  writeOut(lines.join("\n"));
  return 0;
}
