import {
  formatBytes, formatCount, getPreset, loadEngine, resolveSymbols, scaleDesign, validate,
} from "@tensor-cad/engine/node";

await loadEngine();

const r = scaleDesign(getPreset("llama-3-8b"), { targetParams: 30e6, vocab: 50304, targetBasis: "non-embedding", tieHead: true });
const s = resolveSymbols(r.doc);
console.log(`${r.doc.meta.name}: ${formatCount(r.achieved)} non-embedding params (target ${formatCount(r.target)})`);
console.log("changes:", Object.entries(r.changes).map(([k, v]) => `${k} ${v.from}->${v.to}`).join(", "));
console.log("symbols:", ["L","D","H","Hkv","dh","F","V"].map((k) => `${k}=${s.values[k]}`).join("  "));
for (const n of r.notes) console.log("note:", n);
const a = validate(r.doc, { T: 1024, B: 16, hardware: "rtx5080" }).analysis;
console.log(`train memory on an RTX 5080 at batch 16, 1024 tokens: ${formatBytes(a.memory.train.perGpu.total)} of ${formatBytes(a.options.hardware.memory)}`);
console.log(`tokens for Chinchilla-optimal: ${(a.chinchilla.optimalTokens/1e6).toFixed(0)}M`);
