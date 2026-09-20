import { generateTorch, getPreset, loadEngine } from "@tensor-cad/engine/node";
import { mkdirSync, writeFileSync } from "node:fs";

await loadEngine();

const name = process.argv[2] ?? "llama-3-8b";
const out = generateTorch(getPreset(name));
mkdirSync(`out/${name}`, { recursive: true });
for (const f of out.files) writeFileSync(`out/${name}/${f.path}`, f.contents);
if (out.warnings.length) { console.error("warnings:"); for (const w of out.warnings) console.error("  " + w); }
console.log(`wrote out/${name}/model.py`);
