import { allPresets, generateTorch } from "../packages/core/src/index.js";
for (const doc of allPresets()) {
  const out = generateTorch(doc);
  if (out.warnings.length) {
    console.log(`\n== ${doc.meta.name}`);
    for (const w of out.warnings) console.log("  " + w);
  }
}
