/**
 * Primitive blocks. These carry every formula in the system.
 *
 * Conventions:
 *  - `B` and `T` are reserved runtime symbols (batch, sequence length).
 *  - FLOPs are counted the standard way: multiply-accumulate counted as 2, and
 *    only matmuls counted in the headline number. Elementwise work is reported
 *    separately under `elementwise` because the 6N convention excludes it and
 *    because those ops are memory-bound.
 *  - Activation memory is attributed to tensors, not to blocks: a block lists
 *    the input ports it must keep alive (`retains`) and any buffer of its own
 *    (`extraActivationBytes`), so a tensor read by several blocks costs once.
 *
 * Sources for each formula are in docs/research/02-analysis-math.md.
 */

import type { PrimitiveDef } from "./types.js";
import { atomToString, parsePattern } from "../shapes/pattern.js";

const ELEMENTWISE_COST: Record<string, number> = {
  relu: 1,
  relu2: 2,
  gelu: 8,
  gelu_tanh: 8,
  silu: 5,
  swish: 5,
  tanh: 6,
  sigmoid: 4,
  identity: 0,
};

/** The spatial extent a convolution or pooling window leaves behind. */
function convOut(size: number, kernel: number, stride: number, padding: number): number {
  return Math.floor((size + 2 * padding - kernel) / stride) + 1;
}

/** Resolve a possibly-negative axis against a rank, the way NumPy does. */
function axisIndex(axis: unknown, rank: number): number {
  const n = typeof axis === "number" ? axis : -1;
  const i = n < 0 ? rank + n : n;
  return Math.min(Math.max(i, 0), Math.max(rank - 1, 0));
}

export const PRIMITIVES: PrimitiveDef[] = [
  // -------------------------------------------------------------------------
  // Graph boundaries
  // -------------------------------------------------------------------------
  {
    kind: "primitive",
    type: "input",
    category: "io",
    params: {
      shape: { type: "pattern", default: "B T", doc: "Shape pattern of the model input" },
      dtype: { type: "enum", values: ["int64", "int32", "bf16", "fp32"], default: "int64" },
    },
    ports: (r) => ({ in: {}, out: { x: String(r.p.shape ?? "B T") } }),
    docs: { summary: "Model input, usually a batch of token ids." },
  },
  {
    kind: "primitive",
    type: "output",
    category: "io",
    params: {},
    ports: { in: { x: "..." }, out: {} },
    docs: { summary: "Model output." },
  },
  {
    kind: "primitive",
    type: "boundary_in",
    category: "io",
    params: {
      ports: { type: "obj", default: { x: "B T D" }, doc: "Port name -> shape pattern" },
    },
    ports: (r) => ({ in: {}, out: { ...(r.p.ports as Record<string, string>) } }),
    docs: { summary: "Entry point of a container subgraph." },
  },
  {
    kind: "primitive",
    type: "boundary_out",
    category: "io",
    params: {
      ports: { type: "obj", default: { x: "B T D" }, doc: "Port name -> shape pattern" },
    },
    ports: (r) => ({ in: { ...(r.p.ports as Record<string, string>) }, out: {} }),
    docs: { summary: "Exit point of a container subgraph." },
  },

  // -------------------------------------------------------------------------
  // Embeddings
  // -------------------------------------------------------------------------
  {
    kind: "primitive",
    type: "embedding",
    category: "embedding",
    params: {
      vocab: { type: "int", min: 1, doc: "Vocabulary size" },
      dim: { type: "int", min: 1, doc: "Embedding width" },
    },
    ports: { in: { ids: "..." }, out: { y: "... dim" } },
    paramCount: (r) => r.p.vocab * r.p.dim,
    flops: () => ({ fwd: 0 }),
    retains: () => [],
    docs: {
      summary: "Token embedding table.",
      formula: "params = vocab * dim; FLOPs ~ 0 (a gather, not a matmul)",
    },
  },
  {
    kind: "primitive",
    type: "pos_embedding",
    category: "embedding",
    params: {
      max_seq: { type: "int", min: 1, doc: "Maximum position index" },
      dim: { type: "int", min: 1 },
    },
    ports: { in: { x: "... dim" }, out: { y: "... dim" } },
    paramCount: (r) => r.p.max_seq * r.p.dim,
    flops: (r) => ({ fwd: 0, elementwise: r.p.dim }),
    retains: () => [],
    docs: {
      summary: "Learned absolute position embedding, added to the token embedding (GPT-2 style).",
      formula: "params = max_seq * dim",
    },
  },

  {
    kind: "primitive",
    type: "learned_tokens",
    category: "embedding",
    params: {
      count: { type: "int", min: 1, default: 1, doc: "How many distinct vectors are learned" },
      dim: { type: "int", min: 1 },
      tokens: {
        type: "int",
        default: 0,
        min: 0,
        doc: "Sequence length it is broadcast to; 0 means one position per learned vector",
      },
    },
    // Nothing goes in. This is a weight that is also an activation: broadcast
    // across the batch, and often across the sequence as well — I-JEPA learns
    // one mask token and stands a copy of it in every position it has to
    // predict, so what it costs and how far it stretches are different numbers.
    ports: (r) => ({ in: {}, out: { y: `B ${r.p.tokens ? "tokens" : "count"} dim` } }),
    paramCount: (r) => r.p.count * r.p.dim,
    flops: () => ({ fwd: 0 }),
    retains: () => [],
    docs: {
      summary:
        "A learned tensor with no input: a mask token, a CLS token, register tokens, learned queries. " +
        "I-JEPA's predictor stands one of these in for every patch it has to predict.",
      formula: "params = count * dim",
      refs: ["https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py"],
    },
  },

  // -------------------------------------------------------------------------
  // Linear algebra
  // -------------------------------------------------------------------------
  {
    kind: "primitive",
    type: "linear",
    category: "linear",
    params: {
      in_features: { type: "int", min: 1 },
      out_features: { type: "int", min: 1 },
      bias: { type: "bool", default: false },
    },
    ports: { in: { x: "... in_features" }, out: { y: "... out_features" } },
    paramCount: (r) => r.p.in_features * r.p.out_features + (r.p.bias ? r.p.out_features : 0),
    flops: (r) => ({ fwd: 2 * r.p.in_features * r.p.out_features }),
    retains: () => ["x"],
    docs: {
      summary: "Dense projection.",
      formula: "params = in*out (+out with bias); FLOPs/token = 2*in*out; saves its input for the weight gradient",
    },
  },
  {
    kind: "primitive",
    type: "lm_head",
    category: "head",
    params: {
      vocab: { type: "int", min: 1 },
      dim: { type: "int", min: 1 },
      tied: { type: "bool", default: false, doc: "Share weights with the token embedding" },
      bias: { type: "bool", default: false },
    },
    ports: { in: { x: "... dim" }, out: { y: "... vocab" } },
    paramCount: (r) => (r.p.tied ? 0 : r.p.vocab * r.p.dim) + (r.p.bias ? r.p.vocab : 0),
    flops: (r) => ({ fwd: 2 * r.p.vocab * r.p.dim }),
    retains: () => ["x"],
    // bf16 logits plus the fp32 softmax/cross-entropy buffer.
    extraActivationBytes: (r, c) => r.p.vocab * (c.bytes + 4),
    docs: {
      summary: "Output projection to vocabulary logits.",
      formula: "params = 0 when tied, else vocab*dim; FLOPs/token = 2*vocab*dim; logits cost vocab*(bytes+4) per token",
      refs: ["https://blog.eleuther.ai/transformer-math/"],
    },
  },

  // -------------------------------------------------------------------------
  // Convolution
  //
  // A convnet is the one architecture here whose tensors are not a sequence of
  // vectors. Its shapes are `B C H W`, and its spatial extent shrinks layer by
  // layer, so each block is told the size of what arrives and works out what
  // leaves. The per-token FLOP convention still holds with one image as the
  // token, which is what `T = 1` means in these designs.
  // -------------------------------------------------------------------------
  {
    kind: "primitive",
    type: "conv2d",
    category: "linear",
    params: {
      in_channels: { type: "int", min: 1 },
      out_channels: { type: "int", min: 1 },
      kernel: { type: "int", min: 1, default: 3 },
      stride: { type: "int", min: 1, default: 1 },
      padding: { type: "int", min: 0, default: 0 },
      groups: { type: "int", min: 1, default: 1, doc: "Grouped convolution; equal to in_channels is depthwise" },
      bias: { type: "bool", default: true },
      in_h: { type: "int", min: 1, doc: "Height of the incoming feature map" },
      in_w: { type: "int", min: 1, doc: "Width of the incoming feature map" },
      act: {
        type: "enum",
        values: ["identity", "relu", "relu2", "gelu", "gelu_tanh", "silu"],
        default: "identity",
        doc: "Activation applied to the output, as a convnet always does",
      },
    },
    ports: (r) => ({
      in: { x: `B ${r.p.in_channels} ${r.p.in_h} ${r.p.in_w}` },
      out: {
        y: `B ${r.p.out_channels} ${convOut(r.p.in_h, r.p.kernel, r.p.stride, r.p.padding)} ${convOut(r.p.in_w, r.p.kernel, r.p.stride, r.p.padding)}`,
      },
    }),
    paramCount: (r) =>
      (r.p.in_channels / r.p.groups) * r.p.out_channels * r.p.kernel * r.p.kernel +
      (r.p.bias ? r.p.out_channels : 0),
    // Every output position is a dot product over one kernel window, so the
    // cost is the weight count times the positions it is applied at. This is
    // the whole image, not one token: a convnet's token is the image.
    flops: (r) => {
      const oh = convOut(r.p.in_h, r.p.kernel, r.p.stride, r.p.padding);
      const ow = convOut(r.p.in_w, r.p.kernel, r.p.stride, r.p.padding);
      const positions = oh * ow;
      const perPosition = 2 * (r.p.in_channels / r.p.groups) * r.p.out_channels * r.p.kernel * r.p.kernel;
      return {
        fwd: perPosition * positions,
        elementwise: (ELEMENTWISE_COST[String(r.p.act)] ?? 0) * r.p.out_channels * positions,
      };
    },
    retains: () => ["x"],
    docs: {
      summary: "Two-dimensional convolution, optionally with the activation a convnet always follows it with.",
      formula:
        "params = (in/groups)*out*k*k (+out with bias); FLOPs = 2*(in/groups)*out*k*k*H_out*W_out; " +
        "H_out = floor((H + 2*padding - k)/stride) + 1",
      refs: ["https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html"],
    },
  },
  {
    kind: "primitive",
    type: "maxpool2d",
    category: "shape",
    params: {
      channels: { type: "int", min: 1 },
      kernel: { type: "int", min: 1, default: 2 },
      stride: { type: "int", min: 1, default: 2 },
      padding: { type: "int", min: 0, default: 0 },
      in_h: { type: "int", min: 1 },
      in_w: { type: "int", min: 1 },
    },
    ports: (r) => ({
      in: { x: `B ${r.p.channels} ${r.p.in_h} ${r.p.in_w}` },
      out: {
        y: `B ${r.p.channels} ${convOut(r.p.in_h, r.p.kernel, r.p.stride, r.p.padding)} ${convOut(r.p.in_w, r.p.kernel, r.p.stride, r.p.padding)}`,
      },
    }),
    paramCount: () => 0,
    // A comparison per element of each window: no multiply-accumulates, so it
    // belongs with the memory-bound work rather than in the headline number.
    flops: (r) => {
      const oh = convOut(r.p.in_h, r.p.kernel, r.p.stride, r.p.padding);
      const ow = convOut(r.p.in_w, r.p.kernel, r.p.stride, r.p.padding);
      return { fwd: 0, elementwise: r.p.channels * oh * ow * r.p.kernel * r.p.kernel };
    },
    retains: () => ["x"],
    docs: {
      summary: "Spatial max pooling.",
      formula: "no parameters; H_out = floor((H + 2*padding - k)/stride) + 1",
    },
  },
  {
    kind: "primitive",
    type: "flatten2d",
    category: "shape",
    params: {
      channels: { type: "int", min: 1 },
      in_h: { type: "int", min: 1 },
      in_w: { type: "int", min: 1 },
    },
    // Where a convnet stops being spatial and becomes a vector, which in
    // AlexNet is where nine tenths of its parameters live.
    ports: (r) => ({
      in: { x: `B ${r.p.channels} ${r.p.in_h} ${r.p.in_w}` },
      out: { y: `B ${r.p.channels * r.p.in_h * r.p.in_w}` },
    }),
    paramCount: () => 0,
    flops: () => ({ fwd: 0 }),
    retains: () => [],
    docs: { summary: "Folds a feature map into one vector per sample." },
  },

  // -------------------------------------------------------------------------
  // Normalization and elementwise
  // -------------------------------------------------------------------------
  {
    kind: "primitive",
    type: "rmsnorm",
    category: "norm",
    params: {
      dim: { type: "int", min: 1 },
      eps: { type: "num", default: 1e-5 },
      scale: { type: "bool", default: true, doc: "Learned per-channel gain" },
    },
    ports: { in: { x: "... dim" }, out: { y: "... dim" } },
    paramCount: (r) => (r.p.scale ? r.p.dim : 0),
    flops: (r) => ({ fwd: 0, elementwise: 4 * r.p.dim }),
    retains: () => ["x"],
    docs: { summary: "Root-mean-square layer norm.", formula: "params = dim (scale only, no bias)" },
  },
  {
    kind: "primitive",
    type: "layernorm",
    category: "norm",
    params: {
      dim: { type: "int", min: 1 },
      eps: { type: "num", default: 1e-5 },
      bias: { type: "bool", default: true },
    },
    ports: { in: { x: "... dim" }, out: { y: "... dim" } },
    paramCount: (r) => r.p.dim + (r.p.bias ? r.p.dim : 0),
    flops: (r) => ({ fwd: 0, elementwise: 6 * r.p.dim }),
    retains: () => ["x"],
    docs: { summary: "Standard layer norm.", formula: "params = 2*dim with bias, dim without" },
  },
  {
    kind: "primitive",
    type: "activation",
    category: "elementwise",
    params: {
      kind: {
        type: "enum",
        values: ["silu", "gelu", "gelu_tanh", "relu", "relu2", "tanh", "sigmoid", "identity"],
        default: "silu",
      },
      dim: { type: "int", min: 1, doc: "Width, used for the memory and elementwise estimates" },
    },
    ports: { in: { x: "... dim" }, out: { y: "... dim" } },
    paramCount: () => 0,
    flops: (r) => ({ fwd: 0, elementwise: (ELEMENTWISE_COST[r.p.kind] ?? 4) * r.p.dim }),
    retains: () => ["x"],
    docs: { summary: "Pointwise nonlinearity." },
  },
  {
    kind: "primitive",
    type: "add",
    category: "elementwise",
    params: { dim: { type: "int", min: 1 } },
    ports: { in: { a: "... dim", b: "... dim" }, out: { y: "... dim" } },
    paramCount: () => 0,
    flops: (r) => ({ fwd: 0, elementwise: r.p.dim }),
    // The gradient of an add is the identity, so nothing needs to be saved.
    retains: () => [],
    docs: { summary: "Elementwise sum, the residual connection." },
  },
  {
    kind: "primitive",
    type: "mul",
    category: "elementwise",
    params: { dim: { type: "int", min: 1 } },
    ports: { in: { a: "... dim", b: "... dim" }, out: { y: "... dim" } },
    paramCount: () => 0,
    flops: (r) => ({ fwd: 0, elementwise: r.p.dim }),
    // Each operand is needed to differentiate the other.
    retains: () => ["a", "b"],
    docs: { summary: "Elementwise product, the gate in a gated MLP." },
  },
  {
    kind: "primitive",
    type: "rearrange",
    category: "shape",
    params: {
      from: { type: "pattern", default: "B T (H dh)" },
      to: { type: "pattern", default: "B H T dh" },
    },
    ports: (r) => ({ in: { x: String(r.p.from) }, out: { y: String(r.p.to) } }),
    paramCount: () => 0,
    flops: () => ({ fwd: 0 }),
    retains: () => [],
    docs: {
      summary: "Reshape or permute, written in einops notation.",
      formula: "No parameters and no FLOPs; the product of the dimensions must be preserved",
    },
  },

  // -------------------------------------------------------------------------
  // Attention
  // -------------------------------------------------------------------------
  {
    kind: "primitive",
    type: "rope",
    category: "position",
    params: {
      heads: { type: "int", min: 1 },
      head_dim: { type: "int", min: 2 },
      theta: { type: "num", default: 10000 },
      scaling: { type: "obj", default: null, doc: "Optional RoPE scaling spec (linear, NTK, YaRN)" },
    },
    ports: { in: { x: "B heads T head_dim" }, out: { y: "B heads T head_dim" } },
    paramCount: () => 0,
    flops: (r) => ({ fwd: 0, elementwise: 6 * r.p.heads * r.p.head_dim }),
    // The rotation is reconstructed from the position, so nothing is saved.
    retains: () => [],
    constraints: (r) =>
      r.p.head_dim % 2 !== 0 ? [`RoPE needs an even head_dim, got ${r.p.head_dim}`] : [],
    docs: {
      summary: "Rotary position embedding applied to queries or keys.",
      refs: ["https://arxiv.org/abs/2104.09864"],
    },
  },
  {
    kind: "primitive",
    type: "sdpa",
    category: "attention",
    params: {
      heads: { type: "int", min: 1, doc: "Query heads" },
      kv_heads: { type: "int", min: 1, doc: "Key/value heads; equal to heads for MHA, 1 for MQA" },
      head_dim: { type: "int", min: 1, doc: "Width of a query/key head" },
      v_head_dim: { type: "int", default: 0, min: 0, doc: "Width of a value head; 0 means the same as head_dim" },
      causal: { type: "bool", default: true },
      window: { type: "int", default: 0, doc: "Sliding-window width; 0 means full attention" },
      flash: { type: "bool", default: true, doc: "Memory-efficient kernel that never materializes the score matrix" },
      cache: {
        type: "bool",
        default: true,
        doc: "Whether this block owns the inference cache. Latent attention caches a compressed vector instead.",
      },
    },
    ports: (r) => {
      const v = r.p.v_head_dim ? "v_head_dim" : "head_dim";
      return {
        in: {
          q: "B heads T head_dim",
          k: "B kv_heads T head_dim",
          v: `B kv_heads T ${v}`,
        },
        out: { y: `B heads T ${v}` },
      };
    },
    paramCount: () => 0,
    flops: (r, c) => {
      const tEff = r.p.window > 0 ? Math.min(c.T, r.p.window) : c.T;
      // A causal kernel skips masked blocks, so on average each token attends to
      // half the window.
      const causalFactor = r.p.causal ? 0.5 : 1;
      const unmasked = 4 * tEff * r.p.heads * r.p.head_dim;
      return { fwd: 0, fwdSeq: unmasked * causalFactor, fwdSeqUnmasked: unmasked };
    },
    // q, k and v arrive on edges and are counted there; the output and the
    // kernel's own statistics are not on any edge the backward pass reads.
    retains: () => ["q", "k", "v"],
    extraActivationBytes: (r, c) => {
      const tEff = r.p.window > 0 ? Math.min(c.T, r.p.window) : c.T;
      const vDim = r.p.v_head_dim || r.p.head_dim;
      const output = r.p.heads * vDim * c.bytes;
      if (c.flash && r.p.flash) {
        // A fused kernel keeps the output and the log-sum-exp statistics only.
        return output + r.p.heads * 4;
      }
      // Otherwise the score matrix row and the softmax output are both kept.
      return output + 2 * r.p.heads * tEff * c.bytes;
    },
    stateBytes: (r, c) => {
      if (r.p.cache === false) return { perToken: 0, perSeq: 0 };
      const vDim = r.p.v_head_dim || r.p.head_dim;
      const perTokenFull = r.p.kv_heads * (r.p.head_dim + vDim) * c.bytes;
      if (r.p.window > 0) {
        return { perToken: 0, perSeq: perTokenFull * r.p.window };
      }
      return { perToken: perTokenFull, perSeq: 0 };
    },
    constraints: (r) => {
      const out: string[] = [];
      if (r.p.heads % r.p.kv_heads !== 0) {
        out.push(`heads (${r.p.heads}) must be divisible by kv_heads (${r.p.kv_heads})`);
      }
      if (r.p.window < 0) out.push("window must be non-negative");
      return out;
    },
    docs: {
      summary: "Scaled dot-product attention core. Covers MHA, GQA and MQA through kv_heads.",
      formula:
        "FLOPs/token = 4*T_eff*heads*head_dim (halved when causal); KV cache = 2*kv_heads*head_dim*bytes per token",
      refs: ["https://arxiv.org/abs/2305.13245", "https://arxiv.org/abs/2205.14135"],
    },
  },

  // -------------------------------------------------------------------------
  // Mixture of experts
  // -------------------------------------------------------------------------
  {
    kind: "primitive",
    type: "topk_router",
    category: "moe",
    params: {
      d_model: { type: "int", min: 1 },
      experts: { type: "int", min: 1, doc: "Routed experts to choose from" },
      top_k: { type: "int", min: 1, doc: "Experts each token is sent to" },
      bias: { type: "bool", default: false, doc: "Per-expert routing bias (DeepSeek's score correction)" },
      normalize: { type: "bool", default: true, doc: "Renormalize the chosen weights to sum to one" },
    },
    ports: { in: { x: "... d_model" }, out: { weights: "... top_k" } },
    paramCount: (r) => r.p.d_model * r.p.experts + (r.p.bias ? r.p.experts : 0),
    flops: (r) => ({ fwd: 2 * r.p.d_model * r.p.experts }),
    retains: () => ["x"],
    constraints: (r) =>
      r.p.top_k > r.p.experts
        ? [`top_k (${r.p.top_k}) cannot exceed the number of experts (${r.p.experts})`]
        : [],
    docs: {
      summary: "Chooses which experts each token is sent to.",
      formula: "params = d_model * experts (+ experts with a routing bias)",
      refs: ["https://arxiv.org/abs/2401.06066"],
    },
  },
  {
    kind: "primitive",
    type: "weighted_sum",
    category: "moe",
    params: {
      dim: { type: "int", min: 1 },
      n: { type: "int", min: 1, doc: "How many contributions are combined" },
    },
    ports: { in: { x: "... dim", weights: "... n" }, out: { y: "... dim" } },
    paramCount: () => 0,
    flops: (r) => ({ fwd: 0, elementwise: r.p.dim * r.p.n }),
    retains: () => [],
    docs: { summary: "Combines the chosen experts' outputs using the router's weights." },
  },

  // -------------------------------------------------------------------------
  // Tensor plumbing
  // -------------------------------------------------------------------------
  {
    kind: "primitive",
    type: "split",
    category: "shape",
    params: {
      from: { type: "pattern", default: "B T D", doc: "Shape of the incoming tensor" },
      sizes: { type: "obj", default: [], doc: "Widths of the pieces, along the last dimension" },
      axis: { type: "int", default: -1, doc: "Which dimension to cut. Only the last is supported." },
    },
    ports: (r) => {
      const atoms = parsePattern(String(r.p.from)).atoms.map(atomToString);
      const sizes = (r.p.sizes as (string | number)[]) ?? [];
      const out: Record<string, string> = {};
      sizes.forEach((size, i) => {
        const copy = [...atoms];
        copy[copy.length - 1] = `(${size})`;
        out[`y${i}`] = copy.join(" ");
      });
      return { in: { x: String(r.p.from) }, out };
    },
    paramCount: () => 0,
    flops: () => ({ fwd: 0 }),
    retains: () => [],
    docs: {
      summary: "Cuts a tensor into pieces along its last dimension.",
      formula: "No parameters and no FLOPs; the pieces must add up to the incoming width",
    },
  },
  {
    kind: "primitive",
    type: "concat",
    category: "shape",
    params: {
      to: { type: "pattern", default: "B T D", doc: "Shape of the combined tensor" },
      sizes: { type: "obj", default: [], doc: "Extents of the pieces, along `axis`" },
      axis: { type: "int", default: -1, doc: "Which dimension to join on; negative counts from the end" },
    },
    // Any axis, not only the last.
    //
    // Joining along the channel dimension is what a fused QKV projection does;
    // joining along the *sequence* is what a predictor does when it stands mask
    // tokens beside the context it was given. Both are concatenation and there
    // is no reason for the block to know the difference.
    ports: (r) => {
      const atoms = parsePattern(String(r.p.to)).atoms.map(atomToString);
      const sizes = (r.p.sizes as (string | number)[]) ?? [];
      const axis = axisIndex(r.p.axis, atoms.length);
      const inPorts: Record<string, string> = {};
      sizes.forEach((size, i) => {
        const copy = [...atoms];
        copy[axis] = `(${size})`;
        inPorts[`y${i}`] = copy.join(" ");
      });
      return { in: inPorts, out: { y: String(r.p.to) } };
    },
    paramCount: () => 0,
    flops: () => ({ fwd: 0 }),
    retains: () => [],
    docs: { summary: "Joins tensors along one dimension." },
  },
  {
    kind: "primitive",
    type: "expand_heads",
    category: "shape",
    params: {
      heads: { type: "int", min: 1 },
      dim: { type: "int", min: 1 },
    },
    ports: { in: { x: "B T dim" }, out: { y: "B heads T dim" } },
    paramCount: () => 0,
    flops: () => ({ fwd: 0 }),
    // A broadcast view costs nothing to keep.
    retains: () => [],
    docs: {
      summary: "Shares one tensor across every attention head, as latent attention does with its rotary key.",
    },
  },
  {
    kind: "primitive",
    type: "kv_latent_cache",
    category: "attention",
    params: {
      dim: { type: "int", min: 1, doc: "Width of the cached vector per token per layer" },
    },
    ports: { in: { x: "... dim" }, out: { y: "... dim" } },
    paramCount: () => 0,
    flops: () => ({ fwd: 0 }),
    retains: () => [],
    stateBytes: (r, c) => ({ perToken: r.p.dim * c.bytes, perSeq: 0 }),
    docs: {
      summary: "Marks the compressed vector that latent attention caches instead of keys and values.",
      formula: "cache = layers * dim * bytes per token, against 2 * layers * kv_heads * head_dim * bytes for GQA",
      refs: ["https://arxiv.org/abs/2405.04434"],
    },
  },

  // -------------------------------------------------------------------------
  // State-space models
  // -------------------------------------------------------------------------
  {
    kind: "primitive",
    type: "conv1d",
    category: "ssm",
    params: {
      channels: { type: "int", min: 1, doc: "Width of the stream, convolved per channel" },
      kernel: { type: "int", min: 1, default: 4, doc: "Kernel width" },
      bias: { type: "bool", default: true },
    },
    ports: { in: { x: "... channels" }, out: { y: "... channels" } },
    paramCount: (r) => r.p.channels * r.p.kernel + (r.p.bias ? r.p.channels : 0),
    flops: (r) => ({ fwd: 2 * r.p.channels * r.p.kernel }),
    retains: () => ["x"],
    // Generation keeps the last kernel-1 tokens per channel.
    stateBytes: (r, c) => ({ perToken: 0, perSeq: r.p.channels * (r.p.kernel - 1) * c.bytes }),
    docs: {
      summary: "Short depthwise convolution over time, used before a state-space scan.",
      formula: "params = channels * kernel (+ channels with bias)",
      refs: ["https://arxiv.org/abs/2405.21060"],
    },
  },
  {
    kind: "primitive",
    type: "ssd_scan",
    category: "ssm",
    params: {
      d_inner: { type: "int", min: 1, doc: "Width of the state-space stream" },
      heads: { type: "int", min: 1 },
      head_dim: { type: "int", min: 1, doc: "Width per state-space head (Mamba-2's P)" },
      state: { type: "int", min: 1, doc: "Recurrent state width per head (Mamba-2's N)" },
      groups: { type: "int", min: 1, doc: "How many heads share one B/C projection" },
      xbc_width: { type: "int", min: 1, doc: "Width of the combined x, B and C stream" },
      chunk: { type: "int", default: 256, doc: "Chunk length of the chunked scan" },
    },
    ports: (r) => ({
      in: { xbc: "... xbc_width", dt: "... heads" },
      out: { y: "... d_inner" },
    }),
    // Per-head decay, skip and timestep-bias scalars.
    paramCount: (r) => 3 * r.p.heads,
    flops: (r) => ({
      // The scan is linear in sequence length. The constants are approximate:
      // the Mamba-2 paper gives asymptotics rather than exact counts.
      fwd: 6 * r.p.d_inner * r.p.state + 4 * r.p.d_inner * r.p.chunk,
    }),
    retains: () => ["xbc", "dt"],
    // The recurrent state is fixed per sequence: this is why a state-space layer
    // has no cache that grows with context.
    stateBytes: (r, c) => ({
      perToken: 0,
      perSeq: r.p.heads * r.p.head_dim * r.p.state * c.bytes,
    }),
    constraints: (r) => {
      const out: string[] = [];
      if (r.p.heads * r.p.head_dim !== r.p.d_inner) {
        out.push(
          `heads (${r.p.heads}) times head_dim (${r.p.head_dim}) must equal d_inner (${r.p.d_inner})`,
        );
      }
      if (r.p.heads % r.p.groups !== 0) {
        out.push(`heads (${r.p.heads}) must be divisible by groups (${r.p.groups})`);
      }
      return out;
    },
    docs: {
      summary:
        "Mamba-2 state-space scan. Linear in sequence length, and its state is fixed per sequence rather than growing per token.",
      formula:
        "params = 3*heads; state = heads*head_dim*state*bytes per sequence; FLOPs are approximate",
      refs: ["https://arxiv.org/abs/2405.21060"],
    },
  },
];

export const PRIMITIVE_BY_TYPE: Record<string, PrimitiveDef> = Object.fromEntries(
  PRIMITIVES.map((p) => [p.type, p]),
);
