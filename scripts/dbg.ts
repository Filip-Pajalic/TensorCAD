import { getPreset, inferShapes, loadEngine, PRESET_NAMES } from "@tensorcad/engine/node";

await loadEngine();

for (const name of PRESET_NAMES) {
  const doc = getPreset(name);
  const r = inferShapes(doc, "expanded");
  const errs = r.issues.filter((i) => i.severity === "error");
  if (errs.length) {
    console.log(`\n== ${doc.meta.name}`);
    for (const e of errs) console.log(`  ${e.path}${e.port ? ":" + e.port : ""}  ${e.message}`);
  }
}
