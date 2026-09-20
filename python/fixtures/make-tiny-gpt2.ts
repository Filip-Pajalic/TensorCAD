/**
 * Generate the scaled-down GPT-2 used by `tensorcad-runtime smoke-train`.
 *
 * Run from the repo root:
 *   bun run python/fixtures/make-tiny-gpt2.ts
 *
 * Same emitter as the real presets, just a tiny configuration (~30M params) so
 * it trains in a couple of minutes on a single consumer GPU. The design sits
 * beside this file as a document, the way a preset does.
 */
import { generateTorch, loadEngine } from "@tensorcad/engine/node";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

await loadEngine();

const here = new URL(".", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const doc = JSON.parse(readFileSync(join(here, "tiny-gpt2.json"), "utf8"));
const out = generateTorch(doc);
const dir = join(here, "tiny-gpt2");
mkdirSync(dir, { recursive: true });
for (const file of out.files) writeFileSync(join(dir, file.path), file.contents);
for (const warning of out.warnings) console.error("warning: " + warning);
console.log(`wrote ${dir}/model.py`);
