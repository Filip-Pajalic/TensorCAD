/**
 * Analysis regression tests.
 *
 * Where a published figure exists (Llama 3's KV cache, DeepSeek's cache
 * compression, the 6N rule) the test pins it. Where the number is a model of
 * ours, the test pins the identity that makes it checkable by hand.
 */

import { describe, expect, it } from "bun:test";
import { analyze } from "../src/analysis/index.js";
import { validate } from "../src/rules/index.js";
import { allPresets, getPreset } from "../src/presets/index.js";

const KiB = 1024;
const GiB = 1024 ** 3;

describe("FLOPs", () => {
  it("differs from the 2N rule by exactly the norm parameters", () => {
    const a = analyze(getPreset("llama-3-8b"), { T: 8192 });
    // Every dense FLOP is a matmul over a non-embedding parameter, counted
    // twice. Norm gains are parameters that no matmul touches, which is the
    // entire gap between the rule of thumb and the real count.
    const normParams = a.params.byCategory.norm;
    expect(normParams).toBe(32 * 2 * 4096 + 4096);
    expect(a.flops.ruleOfThumb2N - a.flops.fwdDense).toBe(2 * normParams);
  });

  it("computes the attention term as 2*T*heads*head_dim per causal layer", () => {
    const T = 8192;
    const a = analyze(getPreset("llama-3-8b"), { T });
    const L = 32;
    const H = 32;
    const dh = 128;
    // 4*T*H*dh forward, halved because a causal kernel skips masked blocks.
    expect(a.flops.fwdAttention).toBe(L * 2 * T * H * dh);
  });

  it("makes attention's share grow with context", () => {
    const short = analyze(getPreset("llama-3-8b"), { T: 1024 });
    const long = analyze(getPreset("llama-3-8b"), { T: 32768 });
    expect(long.flops.attentionShare).toBeGreaterThan(short.flops.attentionShare * 8);
    expect(short.flops.attentionShare).toBeLessThan(0.05);
    expect(long.flops.attentionShare).toBeGreaterThan(0.3);
  });

  it("charges an extra forward pass for full recomputation", () => {
    const none = analyze(getPreset("llama-3-8b"), { T: 8192, recompute: "none" });
    const full = analyze(getPreset("llama-3-8b"), { T: 8192, recompute: "full" });
    expect(full.flops.trainPerToken / none.flops.trainPerToken).toBeCloseTo(4 / 3, 10);
  });
});

describe("KV cache", () => {
  it("matches the published per-token size for Llama 3 8B", () => {
    // 2 * 32 layers * 8 kv heads * 128 head dim * 2 bytes = 128 KiB.
    const a = analyze(getPreset("llama-3-8b"), { T: 8192 });
    expect(a.kv.bytesPerToken).toBe(128 * KiB);
  });

  it("matches the published per-token size for Llama 2 7B multi-head attention", () => {
    const a = analyze(getPreset("llama-2-7b"));
    expect(a.kv.bytesPerToken).toBe(512 * KiB);
  });

  it("matches the computed per-token size for Llama 3 70B", () => {
    // 2 * 80 layers * 8 kv heads * 128 head dim * 2 bytes = 320 KiB. Note this
    // corrects the 160 KiB quoted in docs/research/02-analysis-math.md.
    const a = analyze(getPreset("llama-3-70b"));
    expect(a.kv.bytesPerToken).toBe(320 * KiB);
  });

  it("turns a sliding window into a fixed per-sequence cost", () => {
    const a = analyze(getPreset("mistral-7b"), { T: 32768 });
    expect(a.kv.bytesPerToken).toBe(0);
    // 2 * 32 layers * 8 kv heads * 128 dim * 2 bytes * 4096 window.
    expect(a.kv.bytesPerSequenceFixed).toBe(2 * 32 * 8 * 128 * 2 * 4096);
  });

  it("halves with the cache dtype", () => {
    const bf16 = analyze(getPreset("llama-3-8b"), { kvDtype: "bf16" });
    const fp8 = analyze(getPreset("llama-3-8b"), { kvDtype: "fp8" });
    expect(fp8.kv.bytesPerToken * 2).toBe(bf16.kv.bytesPerToken);
  });
});

describe("memory", () => {
  it("charges 16 bytes per parameter for mixed-precision AdamW", () => {
    const a = analyze(getPreset("llama-3-8b"), { optimizer: "adamw" });
    const n = a.params.total;
    expect(a.memory.train.weights).toBe(2 * n);
    expect(a.memory.train.grads).toBe(2 * n);
    expect(a.memory.train.optimizer).toBe(12 * n);
  });

  it("counts a tensor once even when several blocks read it", () => {
    // The query, key and value projections all read the same normalized stream.
    // Counting per consumer would inflate the residual tensor threefold.
    const a = analyze(getPreset("llama-3-8b"), { T: 8192, B: 1 });
    const perTokenPerLayer =
      (a.memory.train.activations - a.memory.train.logits) / (32 * 8192);
    // Hand count for one pre-norm SwiGLU layer at D=4096, F=14336, 8 KV heads:
    // 4 stream tensors at 2D, q/k/v at 2D(1+2/g), the attention output and the
    // output projection's input at 4D, and four F-wide tensors in the MLP.
    expect(perTokenPerLayer).toBeGreaterThan(170_000);
    expect(perTokenPerLayer).toBeLessThan(182_000);
  });

  it("shrinks activations sharply under full recomputation", () => {
    const none = analyze(getPreset("llama-3-8b"), { T: 8192, recompute: "none" });
    const full = analyze(getPreset("llama-3-8b"), { T: 8192, recompute: "full" });
    expect(full.memory.train.activations).toBeLessThan(none.memory.train.activations * 0.25);
  });

  it("shards the optimizer under ZeRO 1 and everything under ZeRO 3", () => {
    const base = { T: 2048, parallel: { dp: 8 } };
    const z0 = analyze(getPreset("llama-3-8b"), { ...base, parallel: { dp: 8, zero: 0 } });
    const z1 = analyze(getPreset("llama-3-8b"), { ...base, parallel: { dp: 8, zero: 1 } });
    const z3 = analyze(getPreset("llama-3-8b"), { ...base, parallel: { dp: 8, zero: 3 } });

    expect(z1.memory.train.perGpu.optimizer).toBe(z0.memory.train.perGpu.optimizer / 8);
    expect(z1.memory.train.perGpu.weights).toBe(z0.memory.train.perGpu.weights);
    expect(z3.memory.train.perGpu.weights).toBe(z0.memory.train.perGpu.weights / 8);
    expect(z3.memory.train.perGpu.grads).toBe(z0.memory.train.perGpu.grads / 8);
  });

  it("reports the logits buffer separately", () => {
    const T = 8192;
    const a = analyze(getPreset("llama-3-8b"), { T, B: 1 });
    // 128256 vocab * (2 bf16 + 4 fp32) bytes per token.
    expect(a.memory.train.logits).toBe(128256 * 6 * T);
  });

  it("sizes the serving footprint from weights plus cache", () => {
    const a = analyze(getPreset("llama-3-8b"), { T: 8192, concurrency: 1 });
    expect(a.memory.infer.weights).toBe(a.params.total * 2);
    expect(a.memory.infer.kv).toBe(128 * KiB * 8192);
    expect(a.memory.infer.total).toBeGreaterThan(a.memory.infer.weights);
  });
});

describe("throughput and cost", () => {
  it("puts batch-1 decoding on the memory roof", () => {
    const a = analyze(getPreset("llama-3-8b"), { T: 8192, hardware: "h100-sxm", concurrency: 1 });
    expect(a.throughput.memoryBound).toBe(true);
    // Bounded above by bandwidth divided by the weight bytes that must be read.
    const ceiling = a.options.hardware.bandwidth / (a.params.total * 2);
    expect(a.throughput.decodeTokensPerSecond).toBeLessThanOrEqual(ceiling);
    expect(a.throughput.decodeTokensPerSecond).toBeGreaterThan(100);
  });

  it("reports the device ridge point", () => {
    const a = analyze(getPreset("llama-3-8b"), { hardware: "h100-sxm" });
    expect(a.throughput.ridgePoint).toBeGreaterThan(250);
    expect(a.throughput.ridgePoint).toBeLessThan(350);
  });

  it("scales training cost linearly with the token budget", () => {
    const one = analyze(getPreset("llama-3-8b"), { tokens: 1e12, gpus: 1024, mfu: 0.4 });
    const two = analyze(getPreset("llama-3-8b"), { tokens: 2e12, gpus: 1024, mfu: 0.4 });
    expect(two.cost.gpuHours / one.cost.gpuHours).toBeCloseTo(2, 10);
    expect(two.cost.wallClockHours).toBeCloseTo(two.cost.gpuHours / 1024, 6);
  });

  it("puts Chinchilla-optimal at about 20 tokens per parameter", () => {
    const a = analyze(getPreset("llama-3-8b"));
    expect(a.chinchilla.optimalTokens).toBe(20 * a.params.nonEmbedding);
    expect(a.chinchilla.predictedLoss.hoffmann).toBeGreaterThan(1.69);
    expect(a.chinchilla.predictedLoss.hoffmann).toBeLessThan(3);
  });

  it("flags Llama 3's over-training", () => {
    const a = analyze(getPreset("llama-3-8b"), { tokens: 15e12 });
    expect(a.chinchilla.overTrainingRatio).toBeGreaterThan(50);
    expect(a.chinchilla.tokensPerParam).toBeGreaterThan(1500);
  });
});

describe("design rules", () => {
  it("passes every preset with no errors and no dangling outputs", () => {
    for (const doc of allPresets()) {
      const r = validate(doc, { T: 2048, hardware: "h100-sxm" });
      const blocking = r.findings.filter((f) => f.severity === "error");
      expect({ name: doc.meta.name, blocking }).toEqual({ name: doc.meta.name, blocking: [] });
      expect(r.findings.filter((f) => f.rule === "dangling-output")).toEqual([]);
      expect(r.findings.filter((f) => f.rule === "published-drift")).toEqual([]);
      expect(r.ok).toBe(true);
    }
  });

  it("catches an unsupported head dimension", () => {
    const doc = getPreset("llama-3-8b");
    doc.symbols.dh = { kind: "design", value: 100 };
    doc.symbols.D = { kind: "design", value: 3200 };
    const r = validate(doc);
    expect(r.findings.some((f) => f.rule === "flash-head-dim")).toBe(true);
  });

  it("catches an unpadded vocabulary", () => {
    const r = validate(getPreset("gpt2-small"));
    const f = r.findings.find((x) => x.rule === "vocab-padding");
    expect(f).toBeDefined();
    expect(f!.hint).toMatch(/50304/);
  });

  it("warns when the design will not fit the chosen GPU", () => {
    const r = validate(getPreset("llama-3-70b"), { hardware: "rtx5080", T: 4096 });
    expect(r.findings.some((f) => f.rule === "training-fits")).toBe(true);
    expect(r.findings.some((f) => f.rule === "inference-fits")).toBe(true);
  });

  it("stays quiet about fit when the design does fit", () => {
    const r = validate(getPreset("gpt2-small"), { hardware: "rtx5080", T: 1024, B: 1 });
    expect(r.findings.some((f) => f.rule === "inference-fits")).toBe(false);
    expect(r.findings.some((f) => f.rule === "training-fits")).toBe(false);
  });

  it("notices a broken interface as an error", () => {
    const doc = getPreset("llama-3-8b");
    doc.graph.edges = doc.graph.edges.filter(([, to]) => to !== "head:x");
    const r = validate(doc);
    expect(r.ok).toBe(false);
    expect(r.findings.some((f) => f.severity === "error")).toBe(true);
  });

  it("notices when a design drifts from its published parameter count", () => {
    const doc = getPreset("llama-3-8b");
    doc.symbols.L = { kind: "design", value: 40 };
    const r = validate(doc);
    expect(r.findings.some((f) => f.rule === "published-drift")).toBe(true);
  });

  it("notices an unused symbol", () => {
    const doc = getPreset("llama-3-8b");
    doc.symbols.Unused = { kind: "design", value: 7 };
    const r = validate(doc);
    expect(r.findings.some((f) => f.rule === "unused-symbol" && f.message.includes("Unused"))).toBe(true);
  });
});

describe("performance", () => {
  it("analyses the 405B preset quickly enough to run on every keystroke", () => {
    const doc = getPreset("llama-3.1-405b");
    const start = performance.now();
    for (let i = 0; i < 20; i++) validate(doc, { T: 8192 });
    const perRun = (performance.now() - start) / 20;
    expect(perRun).toBeLessThan(50);
  });
});

describe("mixture of experts", () => {
  it("separates total from active parameters", () => {
    const a = analyze(getPreset("mixtral-8x7b"), { T: 4096 });
    expect(a.params.total).toBe(46_702_792_704);
    expect(a.params.active).toBe(12_879_925_248);
    // Eight experts, two of them used, so the sparse part is a quarter as costly.
    expect(a.params.total).toBeGreaterThan(a.params.active * 3);
  });

  it("charges FLOPs for the experts a token actually visits", () => {
    const a = analyze(getPreset("mixtral-8x7b"), { T: 4096 });
    // The 2N rule is written against active parameters for a sparse model.
    expect(a.flops.ruleOfThumb2N).toBe(2 * a.params.nonEmbeddingActive);
    // Dense forward FLOPs stay within a few percent of it; the gap is the norms
    // and the router, neither of which is a full matmul over its parameters.
    const ratio = a.flops.fwdDense / a.flops.ruleOfThumb2N;
    expect(ratio).toBeGreaterThan(0.95);
    expect(ratio).toBeLessThan(1.05);
  });

  it("holds every expert in memory but activates only some", () => {
    const a = analyze(getPreset("mixtral-8x7b"), { T: 4096, B: 1 });
    // All experts are resident when serving.
    expect(a.memory.infer.weights).toBe(a.params.total * 2);
    // A dense model of the same total size would keep far more activations.
    const dense = analyze(getPreset("llama-2-7b"), { T: 4096, B: 1 });
    expect(a.memory.train.activations).toBeLessThan(dense.memory.train.activations * 3);
  });

  it("leaves the KV cache untouched by expert count", () => {
    const moe = analyze(getPreset("mixtral-8x7b"));
    // 2 * 32 layers * 8 kv heads * 128 head dim * 2 bytes.
    expect(moe.kv.bytesPerToken).toBe(128 * 1024);
  });

  it("counts fine-grained experts and shared experts correctly", () => {
    const a = analyze(getPreset("qwen3-30b-a3b"));
    expect(a.params.total).toBe(30_532_122_624);
    expect(a.params.active).toBe(3_353_032_704);
  });

  it("rejects a top-k larger than the expert count", () => {
    const doc = getPreset("mixtral-8x7b");
    doc.symbols.K = { kind: "design", value: 16 };
    const r = validate(doc);
    expect(r.ok).toBe(false);
    expect(r.findings.some((f) => f.message.includes("cannot exceed"))).toBe(true);
  });

  it("reports a far higher tokens-per-active-parameter ratio than a dense model", () => {
    const a = analyze(getPreset("mixtral-8x7b"), { tokens: 1e12 });
    expect(a.chinchilla.tokensPerActiveParam).toBeGreaterThan(a.chinchilla.tokensPerParam * 3);
  });
});

describe("latent attention", () => {
  it("reproduces DeepSeek-V3's parameter counts", () => {
    const a = analyze(getPreset("deepseek-v3"), { T: 4096 });
    expect(a.params.total).toBe(671_026_419_200);
    expect(a.params.active).toBe(37_552_297_472);
  });

  it("caches one compressed vector per token per layer", () => {
    // 61 layers * (512 latent + 64 rotary) * 2 bytes = 70,272 bytes, the figure
    // the DeepSeek-V2 paper reports.
    const a = analyze(getPreset("deepseek-v3"));
    expect(a.kv.bytesPerToken).toBe(61 * (512 + 64) * 2);
    expect(a.kv.bytesPerToken).toBe(70_272);
  });

  it("caches far less than grouped-query attention of the same shape", () => {
    const mla = analyze(getPreset("deepseek-v3"));
    // Uncompressed, 128 heads of 192+128 wide across 61 layers would be enormous.
    const uncompressed = 61 * 128 * (192 + 128) * 2;
    expect(mla.kv.bytesPerToken * 30).toBeLessThan(uncompressed);
  });

  it("puts the attention parameters where the paper does", () => {
    const a = analyze(getPreset("deepseek-v3"));
    // 187.1M per layer across the 58 sparse layers, per the paper's accounting.
    const perLayer = Object.entries(a.params.byPath)
      .filter(([path]) => path.startsWith("layers/block/attn/"))
      .reduce((sum, [, v]) => sum + v, 0) / 58;
    expect(perLayer).toBeGreaterThan(187_000_000);
    expect(perLayer).toBeLessThan(187_200_000);
  });

  it("validates with no errors", () => {
    const r = validate(getPreset("deepseek-v3"), { T: 4096 });
    expect(r.findings.filter((f) => f.severity === "error")).toEqual([]);
    expect(r.ok).toBe(true);
  });
});

describe("state-space and hybrid stacks", () => {
  it("reproduces Nemotron-H 8B's parameter count", () => {
    const a = analyze(getPreset("nemotron-h-8b"), { T: 8192 });
    expect(a.params.total).toBe(8_100_852_736);
  });

  it("keeps a fixed state per sequence instead of a cache that grows", () => {
    const a = analyze(getPreset("nemotron-h-8b"), { T: 8192 });
    // Only the four attention layers contribute a per-token cache.
    expect(a.kv.bytesPerToken).toBe(2 * 4 * 8 * 128 * 2);
    expect(a.kv.bytesPerSequenceFixed).toBeGreaterThan(0);
  });

  it("caches far less than an all-attention model of the same size", () => {
    const hybrid = analyze(getPreset("nemotron-h-8b"), { T: 8192 });
    const dense = analyze(getPreset("llama-3-8b"), { T: 8192 });
    // Same parameter scale, but only 4 attention layers against Llama's 32,
    // so exactly one eighth of the per-token cache.
    expect(hybrid.params.total / dense.params.total).toBeGreaterThan(0.9);
    expect(dense.kv.bytesPerToken / hybrid.kv.bytesPerToken).toBe(8);
  });

  it("grows its cache sublinearly with context, unlike attention", () => {
    const short = analyze(getPreset("nemotron-h-8b"), { T: 4096, concurrency: 1 });
    const long = analyze(getPreset("nemotron-h-8b"), { T: 65536, concurrency: 1 });
    const denseShort = analyze(getPreset("llama-3-8b"), { T: 4096, concurrency: 1 });
    const denseLong = analyze(getPreset("llama-3-8b"), { T: 65536, concurrency: 1 });

    const hybridGrowth = long.memory.infer.kv / short.memory.infer.kv;
    const denseGrowth = denseLong.memory.infer.kv / denseShort.memory.infer.kv;
    expect(denseGrowth).toBeCloseTo(16, 1);
    expect(hybridGrowth).toBeLessThan(denseGrowth);
  });

  it("validates with no errors", () => {
    const r = validate(getPreset("nemotron-h-8b"), { T: 8192 });
    expect(r.findings.filter((f) => f.severity === "error")).toEqual([]);
  });

  it("rejects a head width that does not divide the state-space stream", () => {
    const doc = getPreset("nemotron-h-8b");
    const stack = doc.graph.nodes.find((n) => n.id === "layers")!;
    const blk = stack.graph!.nodes.find((n) => n.type === "mamba2_block")!;
    // 8192 inner width does not divide into 48-wide heads.
    blk.params!.head_dim = 48;
    const r = validate(doc);
    expect(r.ok).toBe(false);
    expect(r.findings.some((f) => f.message.includes("must be an integer"))).toBe(true);
  });
});

describe("agreement with a profiler", () => {
  it("matches what torch.utils.flop_counter reports for GPT-2 small", () => {
    // Measured on this machine with PyTorch 2.11:
    //   python -m tensorcad_runtime verify out/gpt2-small/model.py
    //   -> flops 64,456,359,936 over batch 2 x seq 128
    const measured = 64_456_359_936 / (2 * 128);
    const a = analyze(getPreset("gpt2-small"), { T: 128, B: 2 });

    // A profiler counts the attention operator as though nothing were masked,
    // because the operator's shape does not depend on the mask. Counted that
    // way, the two agree exactly.
    expect(a.flops.fwdTotalUnmasked).toBe(measured);

    // Our headline number is lower because a fused causal kernel skips the
    // masked blocks. The whole difference is that halving.
    expect(a.flops.fwdTotal).toBeLessThan(measured);
    expect(a.flops.fwdTotalUnmasked - a.flops.fwdTotal).toBe(a.flops.fwdAttention);
  });

  it("reports both attention counts for every design", () => {
    for (const doc of allPresets()) {
      const a = analyze(doc, { T: 2048 });
      expect(a.flops.fwdAttentionUnmasked).toBeGreaterThanOrEqual(a.flops.fwdAttention);
    }
  });
});
