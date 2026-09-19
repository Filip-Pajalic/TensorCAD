/**
 * Generate the scaled-down GPT-2 used by `tensorcad-runtime smoke-train`.
 *
 * Run from the repo root:
 *   bun run python/fixtures/make-tiny-gpt2.ts
 *
 * Same emitter as the real presets, just a tiny configuration (~30M params) so
 * it trains in a couple of minutes on a single consumer GPU.
 */
import { decoderOnly, generateTorch, type DecoderSpec } from "../../packages/core/src/index.js";
import { mkdirSync, writeFileSync } from "node:fs";

const spec: DecoderSpec = {
  name: "tiny-gpt2",
  family: "gpt2",
  notes: "Scaled-down GPT-2 for smoke training. Not a published model.",
  layers: 6,
  dModel: 384,
  heads: 6,
  ffnHidden: "4*D",
  vocab: 50257,
  maxSeq: 512,
  norm: "layernorm",
  mlp: "dense",
  act: "gelu",
  rope: null,
  tied: true,
  attnBias: true,
  mlpBias: true,
  defaultSeq: 512,
};

const out = generateTorch(decoderOnly(spec));
const dir = "python/fixtures/tiny-gpt2";
mkdirSync(dir, { recursive: true });
for (const file of out.files) writeFileSync(`${dir}/${file.path}`, file.contents);
for (const warning of out.warnings) console.error("warning: " + warning);
console.log(`wrote ${dir}/model.py`);
