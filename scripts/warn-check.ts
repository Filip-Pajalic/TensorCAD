import { generateTorch, getPreset, loadEngine, PRESET_NAMES } from "@tensorcad/engine/node";

await loadEngine();

for (const name of PRESET_NAMES) {
  const doc = getPreset(name);
  const out = generateTorch(doc);
  if (out.warnings.length) {
    console.log(`\n== ${doc.meta.name}`);
    for (const w of out.warnings) console.log("  " + w);
  }
}
