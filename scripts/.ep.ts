import { analyze, getPreset, loadEngine, formatBytes, formatCount } from "@tensor-cad/engine/node";
await loadEngine();
const doc = getPreset("mixtral-8x7b");
const base = { T: 4096, hardware: "h100-sxm", gpus: 8 } as const;
const a0 = analyze(doc, base);
console.log("total params ", formatCount(a0.params.total), " experts", formatCount(a0.params.expert ?? 0));
for (const [label, parallel] of [
  ["1 GPU, nothing sharded   ", { dp: 1, tp: 1, pp: 1, ep: 1, zero: 0 }],
  ["8-way tensor parallel    ", { dp: 1, tp: 8, pp: 1, ep: 1, zero: 0 }],
  ["8-way expert parallel    ", { dp: 1, tp: 1, pp: 1, ep: 8, zero: 0 }],
  ["expert + tensor, 8 each  ", { dp: 1, tp: 8, pp: 1, ep: 8, zero: 0 }],
] as const) {
  const a = analyze(doc, { ...base, parallel: parallel as never });
  const g = a.memory.train.perGpu;
  console.log(`${label} weights ${formatBytes(g.weights).padStart(10)}  total/GPU ${formatBytes(g.total).padStart(10)}`);
}
for (const n of analyze(doc, { ...base, parallel: { dp: 1, tp: 1, pp: 1, ep: 8, zero: 0 } as never }).memory.notes) console.log("note:", n);
