import { generateTorch, getPreset, loadEngine, PRESET_NAMES } from "@tensor-cad/engine/node";
import { mkdirSync, writeFileSync } from "node:fs";

await loadEngine();

for (const name of PRESET_NAMES) {
  const doc = getPreset(name);
  const out = generateTorch(doc);
  const dir = `out/${doc.meta.name}`;
  mkdirSync(dir, { recursive: true });
  for (const f of out.files) writeFileSync(`${dir}/${f.path}`, f.contents);
  if (out.warnings.length) console.error(`${doc.meta.name}: ${out.warnings.length} warnings`);
}
console.log("generated", PRESET_NAMES.length, "models");
