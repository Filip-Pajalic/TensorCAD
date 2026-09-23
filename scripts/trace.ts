#!/usr/bin/env bun
/**
 * Regenerate the committed trace: `nano-sort`, trained and run on one input.
 *
 *   bun run trace
 *
 * Like the goldens, this is rewritten deliberately and by nothing else. The
 * editor shows the trace only on a design that still generates the `model.py`
 * it was made from, and `packages/ui/test/trace.test.ts` fails when the code
 * generator's output moves out from under it — which is the moment to run this.
 *
 * Needs PyTorch. Training takes a few seconds on a CPU; the file is a few
 * hundred kilobytes and the editor loads it only when the volume view opens.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { $ } from "bun";
import { generateTorch, getPreset, loadEngine } from "@tensor-cad/engine/node";

const ROOT = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const PRESET = "nano-sort";
const OUT = join(ROOT, "packages/ui/src/three/traces", `${PRESET}.json`);

await loadEngine();
const dir = join(ROOT, "out", PRESET);
mkdirSync(dir, { recursive: true });
for (const f of generateTorch(getPreset(PRESET)).files) writeFileSync(join(dir, f.path), f.contents);

const python = process.platform === "win32" ? "python" : "python3";
const result = await $`${python} -m tensorcad_runtime trace ${join(dir, "model.py")} --out ${OUT}`
  .cwd(ROOT)
  .nothrow();
if (result.exitCode !== 0) {
  console.error(`\n  The trace did not verify, and ${OUT} may be stale. See above.\n`);
  process.exit(1);
}
console.log(`\n  wrote ${OUT}`);
