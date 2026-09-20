/**
 * `tensorcad analyze <file|preset>` — every number the core can produce.
 */

import { bool, type Args } from "../args.js";
import { loadDesign } from "../load.js";
import { analysisOptions, impliedGpus } from "../options.js";
import { analysisJson, analysisText } from "../report.js";
import { writeOut } from "../format.js";
import { analyze } from "@tensor-cad/engine/node";

export function cmdAnalyze(args: Args): number {
  const { doc, source, ref } = loadDesign(args._[0]);
  const options = impliedGpus(analysisOptions(args));
  const result = analyze(doc, options);

  if (bool(args, "json")) {
    writeOut(JSON.stringify({ source, ref, ...analysisJson(result) }, null, 2));
  } else {
    writeOut(analysisText(result, { title: `${doc.meta.name}${doc.meta.family ? `  (${doc.meta.family})` : ""}` }));
  }
  return result.errors.length > 0 ? 1 : 0;
}
