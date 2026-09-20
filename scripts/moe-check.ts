import { countParams, getPreset, loadEngine } from "@tensorcad/engine/node";

await loadEngine();

for (const n of ["mixtral-8x7b", "qwen3-30b-a3b", "qwen3-235b-a22b"]) {
  const doc = getPreset(n);
  const r = countParams(doc);
  const p = doc.meta.published!;
  console.log(
    n.padEnd(18),
    "total", String(r.total).padStart(13), "pub", String(p.params).padStart(13),
    r.total === p.params ? "exact" : ((r.total - p.params!) / p.params! * 100).toFixed(4) + "%",
    "| active", String(r.active).padStart(13), "pub", String(p.activeParams).padStart(13),
    r.active === p.activeParams ? "exact" : ((r.active - p.activeParams!) / p.activeParams! * 100).toFixed(4) + "%",
  );
  if (r.errors.length) console.log("  errors:", r.errors);
}
