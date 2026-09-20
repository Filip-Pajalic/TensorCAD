/** Prints the parameter regression table. Run with: bun run scripts/report.ts */
import { countParams, formatCount, getPreset, loadEngine, PRESET_NAMES } from "@tensorcad/engine/node";

await loadEngine();

const rows: string[] = [];
rows.push("preset".padEnd(16) + "calculated".padStart(14) + "published".padStart(14) + "   delta");
rows.push("-".repeat(60));
for (const name of PRESET_NAMES) {
  const doc = getPreset(name);
  const r = countParams(doc);
  const pub = doc.meta.published?.params ?? 0;
  const delta = pub ? ((r.total - pub) / pub) * 100 : 0;
  rows.push(
    doc.meta.name.padEnd(16) +
      formatCount(r.total).padStart(14) +
      formatCount(pub).padStart(14) +
      "   " + (r.total === pub ? "exact" : delta.toFixed(4) + "%"),
  );
}
console.log(rows.join("\n"));
