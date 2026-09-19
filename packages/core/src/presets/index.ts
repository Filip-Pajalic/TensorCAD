/**
 * Reference architectures.
 *
 * Configuration numbers come from the model's `config.json` or its paper; see
 * docs/research/02-analysis-math.md for the source of each. `published.params`
 * is the figure the authors report and is asserted by the regression suite.
 */

import type { Doc } from "../ir/types.js";
import { decoderOnly, type DecoderSpec } from "./builders.js";
import { visionJepa, type JepaSpec } from "./jepa.js";
import { alexnet } from "./convnet.js";

const SPECS: DecoderSpec[] = [
  {
    name: "gpt2-small",
    family: "gpt2",
    notes: "The original 124M GPT-2. Learned positions, LayerNorm with bias, GELU MLP, tied head.",
    layers: 12,
    dModel: 768,
    heads: 12,
    ffnHidden: "4*D",
    vocab: 50257,
    maxSeq: 1024,
    norm: "layernorm",
    mlp: "dense",
    act: "gelu",
    rope: null,
    tied: true,
    attnBias: true,
    mlpBias: true,
    defaultSeq: 1024,
    published: { params: 124_439_808, source: "https://huggingface.co/openai-community/gpt2" },
  },
  {
    name: "gpt2-medium",
    family: "gpt2",
    layers: 24,
    dModel: 1024,
    heads: 16,
    ffnHidden: "4*D",
    vocab: 50257,
    maxSeq: 1024,
    norm: "layernorm",
    mlp: "dense",
    act: "gelu",
    rope: null,
    tied: true,
    attnBias: true,
    mlpBias: true,
    defaultSeq: 1024,
    published: { params: 354_823_168, source: "https://huggingface.co/openai-community/gpt2-medium" },
  },
  {
    name: "gpt2-large",
    family: "gpt2",
    layers: 36,
    dModel: 1280,
    heads: 20,
    ffnHidden: "4*D",
    vocab: 50257,
    maxSeq: 1024,
    norm: "layernorm",
    mlp: "dense",
    act: "gelu",
    rope: null,
    tied: true,
    attnBias: true,
    mlpBias: true,
    defaultSeq: 1024,
    published: { params: 774_030_080, source: "https://huggingface.co/openai-community/gpt2-large" },
  },
  {
    name: "gpt2-xl",
    family: "gpt2",
    layers: 48,
    dModel: 1600,
    heads: 25,
    ffnHidden: "4*D",
    vocab: 50257,
    maxSeq: 1024,
    norm: "layernorm",
    mlp: "dense",
    act: "gelu",
    rope: null,
    tied: true,
    attnBias: true,
    mlpBias: true,
    defaultSeq: 1024,
    published: { params: 1_557_611_200, source: "https://huggingface.co/openai-community/gpt2-xl" },
  },
  {
    name: "nanogpt",
    family: "gpt2",
    notes:
      "Karpathy's nanoGPT at its defaults: GPT-2 small, with the vocabulary padded from " +
      "50,257 up to 50,304 so it divides by 64. That padding is a speed decision rather " +
      "than a modelling one and it costs 36,096 parameters, which is the whole point of " +
      "having it beside gpt2-small. The repository's own figure, 123.69M, is the " +
      "non-embedding count: it subtracts the position table but keeps the token table, " +
      "because the tied head uses it.",
    layers: 12,
    dModel: 768,
    heads: 12,
    ffnHidden: "4*D",
    vocab: 50304,
    maxSeq: 1024,
    norm: "layernorm",
    mlp: "dense",
    act: "gelu",
    rope: null,
    tied: true,
    attnBias: true,
    mlpBias: true,
    defaultSeq: 1024,
    published: { params: 124_475_904, source: "https://github.com/karpathy/nanoGPT" },
  },
  {
    name: "llama-2-7b",
    family: "llama",
    notes: "Multi-head attention, SwiGLU, RMSNorm, RoPE theta 10000.",
    layers: 32,
    dModel: 4096,
    heads: 32,
    kvHeads: 32,
    headDim: 128,
    ffnHidden: 11008,
    vocab: 32000,
    rope: { theta: 10000 },
    defaultSeq: 4096,
    published: { params: 6_738_415_616, source: "https://huggingface.co/NousResearch/Llama-2-7b-hf" },
  },
  {
    name: "mistral-7b",
    family: "mistral",
    notes: "Grouped-query attention with 8 KV heads and a 4096-token sliding window (v0.1).",
    layers: 32,
    dModel: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    ffnHidden: 14336,
    vocab: 32000,
    rope: { theta: 10000 },
    window: 4096,
    defaultSeq: 8192,
    published: { params: 7_241_732_096, source: "https://huggingface.co/mistralai/Mistral-7B-v0.1" },
  },
  {
    name: "llama-3-8b",
    family: "llama",
    notes:
      "GQA with 8 KV heads, 128k vocabulary, RoPE theta 500000. The FFN width follows Llama 3's " +
      "ceil_mult(1.3 * 8/3 * D, 1024) rule, which is why F is written as an expression.",
    layers: 32,
    dModel: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    ffnHidden: "ceil_mult(1.3*8/3*D, 1024)",
    vocab: 128256,
    rope: { theta: 500000 },
    defaultSeq: 8192,
    published: { params: 8_030_261_248, source: "https://huggingface.co/NousResearch/Meta-Llama-3-8B" },
  },
  {
    name: "llama-3-70b",
    family: "llama",
    layers: 80,
    dModel: 8192,
    heads: 64,
    kvHeads: 8,
    headDim: 128,
    ffnHidden: 28672,
    vocab: 128256,
    rope: { theta: 500000 },
    defaultSeq: 8192,
    published: { params: 70_553_706_496, source: "https://arxiv.org/abs/2407.21783" },
  },
  {
    name: "llama-3.1-405b",
    family: "llama",
    layers: 126,
    dModel: 16384,
    heads: 128,
    kvHeads: 8,
    headDim: 128,
    ffnHidden: 53248,
    vocab: 128256,
    rope: { theta: 500000 },
    defaultSeq: 8192,
    published: { params: 405_853_388_800, source: "https://arxiv.org/abs/2407.21783" },
  },
  {
    name: "qwen2.5-7b",
    family: "qwen",
    notes: "Bias on the q/k/v projections but not on the output projection.",
    layers: 28,
    dModel: 3584,
    heads: 28,
    kvHeads: 4,
    headDim: 128,
    ffnHidden: 18944,
    vocab: 152064,
    rope: { theta: 1000000 },
    attnBias: true,
    attnOBias: false,
    defaultSeq: 32768,
    published: { params: 7_615_616_512, source: "https://huggingface.co/Qwen/Qwen2.5-7B" },
  },
  {
    name: "qwen3-8b",
    family: "qwen",
    notes: "QK-norm on each attention head.",
    layers: 36,
    dModel: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    ffnHidden: 12288,
    vocab: 151936,
    rope: { theta: 1000000 },
    qkNorm: true,
    defaultSeq: 32768,
    published: { params: 8_190_735_360, source: "https://huggingface.co/Qwen/Qwen3-8B" },
  },
  {
    name: "gemma-2-9b",
    family: "gemma",
    notes: "Tied embeddings, GeGLU, and a norm on both the input and the output of every sublayer.",
    layers: 42,
    dModel: 3584,
    heads: 16,
    kvHeads: 8,
    headDim: 256,
    ffnHidden: 14336,
    vocab: 256000,
    act: "gelu_tanh",
    postNorm: true,
    tied: true,
    rope: { theta: 10000 },
    defaultSeq: 8192,
    published: { params: 9_241_705_984, source: "https://huggingface.co/google/gemma-2-9b" },
  },
  {
    name: "mixtral-8x7b",
    family: "mistral",
    notes: "Eight experts per layer, two active per token. The first widely used sparse open model.",
    layers: 32,
    dModel: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    ffnHidden: 14336,
    vocab: 32000,
    rope: { theta: 1000000 },
    moe: { experts: 8, topK: 2, expertHidden: 14336 },
    defaultSeq: 32768,
    published: {
      params: 46_702_792_704,
      activeParams: 12_879_925_248,
      source: "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1",
    },
  },
  {
    name: "qwen3-30b-a3b",
    family: "qwen",
    notes: "128 fine-grained experts with eight active, and QK-norm on every head.",
    layers: 48,
    dModel: 2048,
    heads: 32,
    kvHeads: 4,
    headDim: 128,
    ffnHidden: 6144,
    vocab: 151936,
    rope: { theta: 1000000 },
    qkNorm: true,
    moe: { experts: 128, topK: 8, expertHidden: 768 },
    defaultSeq: 32768,
    published: {
      // Qwen publishes rounded headline figures for this family.
      params: 30_500_000_000,
      activeParams: 3_300_000_000,
      tolerance: 0.02,
      source: "https://huggingface.co/Qwen/Qwen3-30B-A3B",
    },
  },
  {
    name: "qwen3-235b-a22b",
    family: "qwen",
    layers: 94,
    dModel: 4096,
    heads: 64,
    kvHeads: 4,
    headDim: 128,
    ffnHidden: 12288,
    vocab: 151936,
    rope: { theta: 1000000 },
    qkNorm: true,
    moe: { experts: 128, topK: 8, expertHidden: 1536 },
    defaultSeq: 32768,
    published: {
      params: 235_000_000_000,
      activeParams: 22_000_000_000,
      tolerance: 0.02,
      source: "https://huggingface.co/Qwen/Qwen3-235B-A22B",
    },
  },
  {
    name: "deepseek-v3",
    family: "deepseek",
    notes:
      "Latent attention with a 576-wide cached vector per token per layer, three leading dense layers, " +
      "then 58 sparse layers with 256 routed experts, eight active, plus one shared expert. " +
      "The published figure excludes the multi-token-prediction module.",
    layers: 61,
    dModel: 7168,
    heads: 128,
    headDim: 192,
    ffnHidden: 18432,
    vocab: 129280,
    rope: { theta: 10000 },
    mla: { qLora: 1536, kvLora: 512, nopeDim: 128, ropeDim: 64, vDim: 128 },
    moe: { experts: 256, topK: 8, expertHidden: 2048, sharedExperts: 1, routerBias: true, denseLayers: 3 },
    defaultSeq: 4096,
    published: {
      params: 671_000_000_000,
      activeParams: 37_000_000_000,
      kvBytesPerToken: 70_272,
      tolerance: 0.02,
      source: "https://arxiv.org/abs/2412.19437",
    },
  },
  {
    name: "nemotron-h-8b",
    family: "nemotron",
    notes:
      "A hybrid stack: 24 Mamba-2 layers, 24 feed-forward layers and only 4 attention layers. " +
      "Almost all of the cache cost disappears because a state-space layer keeps a fixed state per " +
      "sequence rather than one that grows with every token.",
    layers: 52,
    dModel: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    ffnHidden: 21504,
    vocab: 131072,
    act: "relu2",
    mlp: "dense",
    rope: { theta: 10000 },
    hybrid: {
      pattern: "MFMFMFMAFMFMFMFMFMAFMFMFMFMFMAFMFMFMFMFMAFMFMFMFMFMF",
      mamba: { expand: 2, headDim: 64, state: 128, groups: 8, convKernel: 4 },
    },
    defaultSeq: 8192,
    published: {
      params: 8_100_852_736,
      source: "https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K",
    },
  },
];

/**
 * Designs that are not decoder-only language models.
 *
 * They are kept apart because the shape of the spec is different, not because
 * they are second class: everything downstream — shape checking, the design
 * rules, the analysis, code generation — treats them identically.
 */
const JEPA_SPECS: JepaSpec[] = [
  {
    name: "ijepa-vit-h14",
    family: "jepa",
    notes:
      "I-JEPA at the ImageNet-1k 300-epoch config: a ViT-H/14 context encoder at 224px, a " +
      "384-wide 12-layer predictor, and the exponential-moving-average target encoder drawn " +
      "as its own tower. Attention is bidirectional and positions are fixed 2-D sincos with " +
      "requires_grad=False, so — unlike the ViT-H usually quoted at 632M — neither the " +
      "position table nor a class token costs a parameter. The encoder alone is 630,434,560; " +
      "encoder and predictor together, which is what actually trains, are 652,713,984; the " +
      "figure pinned here includes the frozen target copy as well, because it is resident " +
      "the whole time you train and this drawing shows it.",
    layers: 32,
    dModel: 1280,
    heads: 16,
    ffnHidden: 5120,
    predLayers: 12,
    predDim: 384,
    predHeads: 16,
    patch: 14,
    imageSize: 224,
    emaTarget: true,
    published: {
      params: 1_283_148_544,
      source: "https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml",
    },
  },
];

const built = new Map<string, Doc>();

/** Designs written out block by block rather than generated from a spec. */
const LITERAL: Record<string, () => Doc> = { alexnet };

export const PRESET_NAMES: string[] = [
  ...SPECS.map((s) => s.name),
  ...JEPA_SPECS.map((s) => s.name),
  ...Object.keys(LITERAL),
];

export function getPreset(name: string): Doc {
  const hit = built.get(name);
  if (hit) return structuredClone(hit);
  const spec = SPECS.find((s) => s.name === name);
  const jepa = JEPA_SPECS.find((s) => s.name === name);
  const literal = LITERAL[name];
  if (!spec && !jepa && !literal) {
    throw new Error(`Unknown preset "${name}". Available: ${PRESET_NAMES.join(", ")}`);
  }
  const doc = spec ? decoderOnly(spec) : jepa ? visionJepa(jepa) : literal();
  built.set(name, doc);
  return structuredClone(doc);
}

export function allPresets(): Doc[] {
  return PRESET_NAMES.map(getPreset);
}

export { decoderOnly };
export type { DecoderSpec };