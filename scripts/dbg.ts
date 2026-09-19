import { allPresets, resolveSymbols, inferShapes } from "../packages/core/src/index.js";
for (const doc of allPresets()) {
  const s = resolveSymbols(doc);
  const r = inferShapes(doc, s, { expandComposites: true });
  const errs = r.issues.filter((i) => i.severity === "error");
  if (errs.length) {
    console.log(`\n== ${doc.meta.name}`);
    for (const e of errs) console.log(`  ${e.path}${e.port ? ":" + e.port : ""}  ${e.message}`);
  }
}
