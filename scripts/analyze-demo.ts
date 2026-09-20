import {
  getPreset, loadEngine, validate,
  formatCount, formatBytes, formatFlops, formatHours, formatDollars,
} from "@tensor-cad/engine/node";

await loadEngine();

const doc = getPreset("llama-3-8b");
const r = validate(doc, { T: 8192, B: 1, hardware: "h100-sxm", gpus: 8, tokens: 15e12 });
const a = r.analysis;

console.log(`== ${a.name} on ${a.options.hardware.name}, T=${a.options.T}, B=${a.options.B}`);
console.log(`params        ${formatCount(a.params.total)}  (non-embedding ${formatCount(a.params.nonEmbedding)})`);
console.log(`fwd/token     ${formatFlops(a.flops.fwdTotal)}   dense ${formatFlops(a.flops.fwdDense)} + attn ${formatFlops(a.flops.fwdAttention)}`);
console.log(`  2N rule     ${formatFlops(a.flops.ruleOfThumb2N)}   attention share ${(a.flops.attentionShare*100).toFixed(1)}%`);
console.log(`train/token   ${formatFlops(a.flops.trainPerToken)}   (6N rule ${formatFlops(a.flops.ruleOfThumb6N)})`);
console.log(`kv/token      ${formatBytes(a.kv.bytesPerToken)}`);
console.log(`weights bf16  ${formatBytes(a.memory.weightsBytes)}`);
console.log(`train total   ${formatBytes(a.memory.train.total)}  (activations ${formatBytes(a.memory.train.activations)}, logits ${formatBytes(a.memory.train.logits)})`);
console.log(`serve total   ${formatBytes(a.memory.infer.total)}  (kv ${formatBytes(a.memory.infer.kv)})`);
console.log(`decode        ${a.throughput.decodeTokensPerSecond.toFixed(0)} tok/s at batch ${a.options.concurrency}, ${a.throughput.memoryBound ? "memory" : "compute"}-bound, ridge ${a.throughput.ridgePoint.toFixed(0)}`);
console.log(`cost          ${a.cost.gpuHours.toExponential(2)} GPU-h, ${formatHours(a.cost.wallClockHours)} on ${a.options.gpus} GPUs, ${formatDollars(a.cost.dollars)} at MFU ${(a.options.mfu*100).toFixed(0)}%`);
console.log(`chinchilla    ${a.chinchilla.tokensPerParam.toFixed(0)} tokens/param, ${a.chinchilla.overTrainingRatio.toFixed(1)}x optimal`);
console.log(`\nfindings: ${r.counts.error} errors, ${r.counts.warning} warnings, ${r.counts.info} info`);
for (const f of r.findings) {
  console.log(`  [${f.severity}] ${f.rule}${f.path ? " @" + f.path : ""}: ${f.message}`);
  if (f.hint) console.log(`      -> ${f.hint}`);
}
