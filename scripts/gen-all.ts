import { allPresets, generateTorch } from "../packages/core/src/index.js";
import { mkdirSync, writeFileSync } from "node:fs";
for (const doc of allPresets()) {
  const out = generateTorch(doc);
  const dir = `out/${doc.meta.name}`;
  mkdirSync(dir, { recursive: true });
  for (const f of out.files) writeFileSync(`${dir}/${f.path}`, f.contents);
  if (out.warnings.length) console.error(`${doc.meta.name}: ${out.warnings.length} warnings`);
}
console.log("generated", allPresets().length, "models");
