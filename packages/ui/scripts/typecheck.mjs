#!/usr/bin/env node
/**
 * Typecheck this package.
 *
 * `tsc` pulls @llmcad/core in as source (the workspace package points `types`
 * at `src/index.ts`), so diagnostics from the core package land in this run
 * too. Core is owned by another package and must not be edited from here, so
 * its diagnostics are reported as warnings and only diagnostics in this
 * package's own files fail the build. When core typechecks cleanly this script
 * is a plain `tsc --noEmit`.
 */

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const tsc = path.resolve(root, "../../node_modules/typescript/bin/tsc");

const run = spawnSync(process.execPath, [tsc, "--noEmit", "--pretty", "false"], {
  cwd: root,
  encoding: "utf8",
});

const lines = `${run.stdout ?? ""}${run.stderr ?? ""}`.split(/\r?\n/).filter((l) => l.trim() !== "");

const isDiagnosticStart = (line) => /^\S.*\(\d+,\d+\): (error|warning) TS\d+:/.test(line);
const fileOf = (line) => line.slice(0, line.indexOf("("));
const isForeign = (file) => file.replace(/\\/g, "/").includes("../core/");

const ours = [];
const foreign = [];
let sink = null;
for (const line of lines) {
  if (isDiagnosticStart(line)) sink = isForeign(fileOf(line)) ? foreign : ours;
  else if (sink === null) sink = ours;
  sink.push(line);
}

if (foreign.length > 0) {
  console.warn("warning: @llmcad/core has type errors; they are reported here but not owned by @llmcad/ui:");
  for (const line of foreign) console.warn(`  ${line}`);
}

if (ours.length > 0) {
  for (const line of ours) console.error(line);
  process.exit(1);
}

console.log("typecheck: @llmcad/ui is clean");
