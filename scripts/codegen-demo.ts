import { getPreset, generateTorch } from "../packages/core/src/index.js";
import { mkdirSync, writeFileSync } from "node:fs";
const name = process.argv[2] ?? "llama-3-8b";
const out = generateTorch(getPreset(name));
mkdirSync(`out/${name}`, { recursive: true });
for (const f of out.files) writeFileSync(`out/${name}/${f.path}`, f.contents);
if (out.warnings.length) { console.error("warnings:"); for (const w of out.warnings) console.error("  " + w); }
console.log(`wrote out/${name}/model.py`);
