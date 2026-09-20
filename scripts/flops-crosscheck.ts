import { analyze, formatFlops, getPreset, loadEngine } from "@tensor-cad/engine/node";

await loadEngine();

const T = 128, B = 2;
const a = analyze(getPreset("gpt2-small"), { T, B });
const torchTotal = 64_456_359_936;       // measured by FlopCounterMode
const torchPerToken = torchTotal / (B * T);
const ours = a.flops.fwdTotal;
console.log(`sequence T=${T}, batch B=${B}`);
console.log(`ours   ${formatFlops(ours).padStart(14)}  per token  (dense ${formatFlops(a.flops.fwdDense)} + attention ${formatFlops(a.flops.fwdAttention)})`);
console.log(`torch  ${formatFlops(torchPerToken).padStart(14)}  per token  (FlopCounterMode over the real graph)`);
const diff = (torchPerToken - ours) / ours;
console.log(`difference ${(diff * 100).toFixed(2)}%`);
console.log(`\nattention term ours: ${formatFlops(a.flops.fwdAttention)} (causal, half of 4*T*H*dh per layer)`);
console.log(`if attention were counted without the causal halving: ${formatFlops(a.flops.fwdDense + a.flops.fwdAttention * 2)}`);
