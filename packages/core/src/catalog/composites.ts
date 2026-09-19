/**
 * Composite blocks: named subgraphs of primitives.
 *
 * A composite carries no formulas. Analysis expands it and sums the primitives,
 * which is why adding a new attention variant or MLP shape never requires new
 * math. Each expansion is a self-contained graph with `boundary_in` and
 * `boundary_out` nodes matching the composite's declared ports, so the very
 * same shape inference runs inside it.
 */

import type { Graph, NodeDef, ParamValue } from "../ir/types.js";
import type { CompositeDef, ContainerDef, BlockFinding } from "./types.js";
import { ex } from "./types.js";

const ROPE_SPEC = { type: "obj", default: null, doc: "RoPE settings, or null for no rotary embedding" } as const;

/**
 * Boundary nodes for a composite expansion.
 *
 * The port names must match the composite's own declared ports, otherwise the
 * inner shapes never reach the outer graph.
 */
function boundary(
  inPorts: Record<string, string>,
  outPorts: Record<string, string>,
): { inNode: NodeDef; outNode: NodeDef } {
  return {
    inNode: { id: "_in", type: "boundary_in", params: { ports: inPorts } },
    outNode: { id: "_out", type: "boundary_out", params: { ports: outPorts } },
  };
}

/** The standard single-stream composite boundary: input `x`, output `y`. */
function streamBoundary(width: string): { inNode: NodeDef; outNode: NodeDef } {
  return boundary({ x: `... ${width}` }, { y: `... ${width}` });
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

const gqaAttention: CompositeDef = {
  kind: "composite",
  type: "gqa_attention",
  category: "attention",
  params: {
    d_model: { type: "int", min: 1, doc: "Residual stream width" },
    heads: { type: "int", min: 1, doc: "Query heads" },
    kv_heads: { type: "int", min: 1, doc: "Key/value heads. Equal to heads gives MHA, 1 gives MQA" },
    head_dim: { type: "int", min: 1 },
    bias: { type: "bool", default: false, doc: "Bias on the q/k/v/o projections" },
    o_bias: { type: "bool", default: null, doc: "Override the output-projection bias (Qwen2.5 has qkv bias only)" },
    causal: { type: "bool", default: true },
    window: { type: "int", default: 0, doc: "Sliding-window width; 0 means full attention" },
    qk_norm: { type: "bool", default: false, doc: "RMSNorm on the query and key heads (Qwen3, Gemma 3)" },
    rope: ROPE_SPEC,
    flash: { type: "bool", default: true },
  },
  ports: { in: { x: "... d_model" }, out: { y: "... d_model" } },
  expand: (raw, r) => {
    const D = ex(raw.d_model);
    const H = ex(raw.heads);
    const KV = ex(raw.kv_heads);
    const dh = ex(raw.head_dim);
    const bias = r.p.bias === true;
    const oBias = r.p.o_bias === null || r.p.o_bias === undefined ? bias : r.p.o_bias === true;
    const rope = r.p.rope as { theta?: number; scaling?: ParamValue } | null;
    const qkNorm = r.p.qk_norm === true;

    const { inNode, outNode } = streamBoundary(D);
    const nodes: NodeDef[] = [
      inNode,
      { id: "q_proj", type: "linear", params: { in_features: D, out_features: `${H}*${dh}`, bias } },
      { id: "k_proj", type: "linear", params: { in_features: D, out_features: `${KV}*${dh}`, bias } },
      { id: "v_proj", type: "linear", params: { in_features: D, out_features: `${KV}*${dh}`, bias } },
      { id: "q_heads", type: "rearrange", params: { from: `B T (${H} ${dh})`, to: `B ${H} T ${dh}` } },
      { id: "k_heads", type: "rearrange", params: { from: `B T (${KV} ${dh})`, to: `B ${KV} T ${dh}` } },
      { id: "v_heads", type: "rearrange", params: { from: `B T (${KV} ${dh})`, to: `B ${KV} T ${dh}` } },
    ];
    const edges: [string, string][] = [
      ["_in:x", "q_proj:x"],
      ["_in:x", "k_proj:x"],
      ["_in:x", "v_proj:x"],
      ["q_proj:y", "q_heads:x"],
      ["k_proj:y", "k_heads:x"],
      ["v_proj:y", "v_heads:x"],
    ];

    let qTail = "q_heads:y";
    let kTail = "k_heads:y";

    if (qkNorm) {
      nodes.push({ id: "q_norm", type: "rmsnorm", params: { dim: dh } });
      nodes.push({ id: "k_norm", type: "rmsnorm", params: { dim: dh } });
      edges.push([qTail, "q_norm:x"], [kTail, "k_norm:x"]);
      qTail = "q_norm:y";
      kTail = "k_norm:y";
    }

    if (rope) {
      const theta = rope.theta ?? 10000;
      nodes.push({
        id: "rope_q",
        type: "rope",
        params: { heads: H, head_dim: dh, theta, scaling: rope.scaling ?? null },
      });
      nodes.push({
        id: "rope_k",
        type: "rope",
        params: { heads: KV, head_dim: dh, theta, scaling: rope.scaling ?? null },
      });
      edges.push([qTail, "rope_q:x"], [kTail, "rope_k:x"]);
      qTail = "rope_q:y";
      kTail = "rope_k:y";
    }

    nodes.push({
      id: "attn",
      type: "sdpa",
      params: {
        heads: H,
        kv_heads: KV,
        head_dim: dh,
        causal: r.p.causal === true,
        window: ex(raw.window, "0"),
        flash: r.p.flash !== false,
      },
    });
    nodes.push({ id: "o_merge", type: "rearrange", params: { from: `B ${H} T ${dh}`, to: `B T (${H} ${dh})` } });
    nodes.push({
      id: "o_proj",
      type: "linear",
      params: { in_features: `${H}*${dh}`, out_features: D, bias: oBias },
    });
    nodes.push(outNode);

    edges.push(
      [qTail, "attn:q"],
      [kTail, "attn:k"],
      ["v_heads:y", "attn:v"],
      ["attn:y", "o_merge:x"],
      ["o_merge:y", "o_proj:x"],
      ["o_proj:y", "_out:y"],
    );

    return { nodes, edges };
  },
  constraints: (r) => {
    const out: BlockFinding[] = [];
    if (r.p.heads % r.p.kv_heads !== 0) {
      out.push({
        id: "ATTN-01",
        severity: "error",
        param: "kv_heads",
        message: `heads (${r.p.heads}) must be divisible by kv_heads (${r.p.kv_heads})`,
        hint: "Grouped-query attention shares one key/value head across a whole group of query heads, so the groups have to come out even.",
      });
    }
    return out;
  },
  docs: {
    summary:
      "Grouped-query attention. kv_heads = heads gives multi-head attention, kv_heads = 1 gives multi-query.",
    formula: "params = d_model*heads*head_dim + 2*d_model*kv_heads*head_dim + heads*head_dim*d_model",
    refs: ["https://arxiv.org/abs/2305.13245"],
  },
};

// ---------------------------------------------------------------------------
// Feed-forward
// ---------------------------------------------------------------------------

const gatedMlp: CompositeDef = {
  kind: "composite",
  type: "gated_mlp",
  category: "mlp",
  params: {
    d_model: { type: "int", min: 1 },
    hidden: { type: "int", min: 1, doc: "Intermediate width" },
    act: { type: "enum", values: ["silu", "gelu", "gelu_tanh", "relu", "relu2"], default: "silu" },
    bias: { type: "bool", default: false },
  },
  ports: { in: { x: "... d_model" }, out: { y: "... d_model" } },
  expand: (raw, r) => {
    const D = ex(raw.d_model);
    const F = ex(raw.hidden);
    const bias = r.p.bias === true;
    const { inNode, outNode } = streamBoundary(D);
    return {
      nodes: [
        inNode,
        { id: "gate", type: "linear", params: { in_features: D, out_features: F, bias } },
        { id: "up", type: "linear", params: { in_features: D, out_features: F, bias } },
        { id: "act", type: "activation", params: { kind: r.p.act, dim: F } },
        { id: "gated", type: "mul", params: { dim: F } },
        { id: "down", type: "linear", params: { in_features: F, out_features: D, bias } },
        outNode,
      ],
      edges: [
        ["_in:x", "gate:x"],
        ["_in:x", "up:x"],
        ["gate:y", "act:x"],
        ["act:y", "gated:a"],
        ["up:y", "gated:b"],
        ["gated:y", "down:x"],
        ["down:y", "_out:y"],
      ],
    };
  },
  docs: {
    summary: "Gated feed-forward network (SwiGLU when act is silu, GeGLU when gelu).",
    formula: "params = 3 * d_model * hidden",
    refs: ["https://arxiv.org/abs/2002.05202"],
  },
};

const denseMlp: CompositeDef = {
  kind: "composite",
  type: "dense_mlp",
  category: "mlp",
  params: {
    d_model: { type: "int", min: 1 },
    hidden: { type: "int", min: 1 },
    act: { type: "enum", values: ["gelu", "gelu_tanh", "relu", "relu2", "silu"], default: "gelu" },
    bias: { type: "bool", default: true },
  },
  ports: { in: { x: "... d_model" }, out: { y: "... d_model" } },
  expand: (raw, r) => {
    const D = ex(raw.d_model);
    const F = ex(raw.hidden);
    const bias = r.p.bias === true;
    const { inNode, outNode } = streamBoundary(D);
    return {
      nodes: [
        inNode,
        { id: "up", type: "linear", params: { in_features: D, out_features: F, bias } },
        { id: "act", type: "activation", params: { kind: r.p.act, dim: F } },
        { id: "down", type: "linear", params: { in_features: F, out_features: D, bias } },
        outNode,
      ],
      edges: [
        ["_in:x", "up:x"],
        ["up:y", "act:x"],
        ["act:y", "down:x"],
        ["down:y", "_out:y"],
      ],
    };
  },
  docs: {
    summary: "Classic two-matrix feed-forward network (GPT-2, Nemotron-H).",
    formula: "params = 2 * d_model * hidden (+ hidden + d_model with bias)",
  },
};

// ---------------------------------------------------------------------------
// Latent attention
// ---------------------------------------------------------------------------

const mlaAttention: CompositeDef = {
  kind: "composite",
  type: "mla_attention",
  category: "attention",
  params: {
    d_model: { type: "int", min: 1 },
    heads: { type: "int", min: 1 },
    q_lora: { type: "int", min: 1, doc: "Width of the compressed query (DeepSeek's q_lora_rank)" },
    kv_lora: { type: "int", min: 1, doc: "Width of the compressed key/value latent, which is what gets cached" },
    nope_dim: { type: "int", min: 1, doc: "Per-head width that carries no position information" },
    rope_dim: { type: "int", min: 2, doc: "Per-head width that carries the rotary embedding" },
    v_dim: { type: "int", min: 1, doc: "Per-head value width" },
    causal: { type: "bool", default: true },
    rope: ROPE_SPEC,
    bias: { type: "bool", default: false },
  },
  ports: { in: { x: "... d_model" }, out: { y: "... d_model" } },
  expand: (raw, r) => {
    const D = ex(raw.d_model);
    const H = ex(raw.heads);
    const QL = ex(raw.q_lora);
    const KL = ex(raw.kv_lora);
    const NOPE = ex(raw.nope_dim);
    const ROPE = ex(raw.rope_dim);
    const VD = ex(raw.v_dim);
    const QK = `${NOPE}+${ROPE}`;
    const LATENT = `${KL}+${ROPE}`;
    const bias = r.p.bias === true;
    const rope = r.p.rope as { theta?: number } | null;
    const theta = rope?.theta ?? 10000;

    const { inNode, outNode } = streamBoundary(D);

    const nodes: NodeDef[] = [
      inNode,
      // Query path: compress, normalize, expand back into heads.
      { id: "q_down", type: "linear", params: { in_features: D, out_features: QL, bias } },
      { id: "q_norm", type: "rmsnorm", params: { dim: QL } },
      { id: "q_up", type: "linear", params: { in_features: QL, out_features: `${H}*(${QK})`, bias } },
      { id: "q_heads", type: "rearrange", params: { from: `B T (${H} (${QK}))`, to: `B ${H} T (${QK})` } },

      // Key/value path: one compressed vector per token is all that is cached.
      { id: "kv_down", type: "linear", params: { in_features: D, out_features: `${LATENT}`, bias } },
      { id: "latent", type: "kv_latent_cache", params: { dim: `${LATENT}` } },
      { id: "kv_split", type: "split", params: { from: `B T (${LATENT})`, sizes: [KL, ROPE] } },
      { id: "kv_norm", type: "rmsnorm", params: { dim: KL } },
      { id: "k_up", type: "linear", params: { in_features: KL, out_features: `${H}*(${NOPE})`, bias } },
      { id: "k_nope_heads", type: "rearrange", params: { from: `B T (${H} (${NOPE}))`, to: `B ${H} T (${NOPE})` } },
      { id: "v_up", type: "linear", params: { in_features: KL, out_features: `${H}*(${VD})`, bias } },
      { id: "v_heads", type: "rearrange", params: { from: `B T (${H} (${VD}))`, to: `B ${H} T (${VD})` } },

      // The rotary part of the key is shared by every head.
      { id: "k_rope_shared", type: "expand_heads", params: { heads: H, dim: ROPE } },
      { id: "k_rope", type: "rope", params: { heads: H, head_dim: ROPE, theta } },
      { id: "k_cat", type: "concat", params: { to: `B ${H} T (${QK})`, sizes: [NOPE, ROPE] } },

      {
        id: "attn",
        type: "sdpa",
        params: {
          heads: H,
          kv_heads: H,
          head_dim: `${QK}`,
          v_head_dim: VD,
          causal: r.p.causal === true,
          // The latent node above owns the cache; counting it here too would
          // double-count it, and at the uncompressed size.
          cache: false,
        },
      },
      { id: "o_merge", type: "rearrange", params: { from: `B ${H} T (${VD})`, to: `B T (${H} (${VD}))` } },
      { id: "o_proj", type: "linear", params: { in_features: `${H}*(${VD})`, out_features: D, bias } },
      outNode,
    ];

    const edges: [string, string][] = [
      ["_in:x", "q_down:x"],
      ["q_down:y", "q_norm:x"],
      ["q_norm:y", "q_up:x"],
      ["q_up:y", "q_heads:x"],
      ["_in:x", "kv_down:x"],
      ["kv_down:y", "latent:x"],
      ["latent:y", "kv_split:x"],
      ["kv_split:y0", "kv_norm:x"],
      ["kv_norm:y", "k_up:x"],
      ["k_up:y", "k_nope_heads:x"],
      ["kv_norm:y", "v_up:x"],
      ["v_up:y", "v_heads:x"],
      ["kv_split:y1", "k_rope_shared:x"],
      ["k_rope_shared:y", "k_rope:x"],
      ["k_nope_heads:y", "k_cat:y0"],
      ["k_rope:y", "k_cat:y1"],
      ["q_heads:y", "attn:q"],
      ["k_cat:y", "attn:k"],
      ["v_heads:y", "attn:v"],
      ["attn:y", "o_merge:x"],
      ["o_merge:y", "o_proj:x"],
      ["o_proj:y", "_out:y"],
    ];

    return { nodes, edges };
  },
  docs: {
    summary:
      "Multi-head latent attention. Keys and values are compressed to one small vector per token, and only that vector is cached.",
    formula:
      "params = d_model*q_lora + q_lora + q_lora*heads*(nope+rope) + d_model*(kv_lora+rope) + kv_lora + " +
      "kv_lora*heads*(nope+v_dim) + heads*v_dim*d_model; cache = layers*(kv_lora+rope)*bytes per token",
    refs: ["https://arxiv.org/abs/2405.04434", "https://arxiv.org/abs/2412.19437"],
  },
};

// ---------------------------------------------------------------------------
// Mixture of experts
// ---------------------------------------------------------------------------

const moeExperts: ContainerDef = {
  kind: "container",
  type: "moe_experts",
  category: "moe",
  params: {
    experts: { type: "int", min: 1, doc: "How many expert copies exist" },
    top_k: { type: "int", min: 1, doc: "How many a single token passes through" },
  },
  // This is the whole of what makes a model sparse: every expert holds weights,
  // but a token only pays for top_k of them.
  multipliers: (r) => ({ total: r.p.experts, active: r.p.top_k }),
  docs: {
    summary: "A bank of experts. Its subgraph describes one expert.",
    formula: "total params scale with experts; FLOPs and activations scale with top_k",
  },
};

const moeLayer: CompositeDef = {
  kind: "composite",
  type: "moe_layer",
  category: "moe",
  params: {
    d_model: { type: "int", min: 1 },
    experts: { type: "int", min: 1, doc: "Routed experts" },
    top_k: { type: "int", min: 1 },
    expert_hidden: { type: "int", min: 1, doc: "Hidden width of one expert" },
    shared_experts: { type: "int", default: 0, min: 0, doc: "Experts every token always passes through" },
    act: { type: "enum", values: ["silu", "gelu", "gelu_tanh", "relu", "relu2"], default: "silu" },
    bias: { type: "bool", default: false },
    router_bias: { type: "bool", default: false },
    normalize: { type: "bool", default: true },
  },
  ports: { in: { x: "... d_model" }, out: { y: "... d_model" } },
  expand: (raw, r) => {
    const D = ex(raw.d_model);
    const E = ex(raw.experts);
    const K = ex(raw.top_k);
    const Fe = ex(raw.expert_hidden);
    const shared = typeof r.p.shared_experts === "number" ? r.p.shared_experts : 0;
    const { inNode, outNode } = streamBoundary(D);

    const nodes: NodeDef[] = [
      inNode,
      {
        id: "router",
        type: "topk_router",
        params: {
          d_model: D,
          experts: E,
          top_k: K,
          bias: r.p.router_bias === true,
          normalize: r.p.normalize !== false,
        },
      },
      {
        id: "experts",
        type: "moe_experts",
        params: { experts: E, top_k: K },
        graph: {
          nodes: [
            { id: "_in", type: "boundary_in", params: { ports: { x: `... ${D}` } } },
            {
              id: "expert",
              type: "gated_mlp",
              params: { d_model: D, hidden: Fe, act: r.p.act, bias: r.p.bias === true },
            },
            { id: "_out", type: "boundary_out", params: { ports: { y: `... ${D}` } } },
          ],
          edges: [
            ["_in:x", "expert:x"],
            ["expert:y", "_out:y"],
          ],
        },
      },
      { id: "combine", type: "weighted_sum", params: { dim: D, n: K } },
    ];
    const edges: [string, string][] = [
      ["_in:x", "router:x"],
      ["_in:x", "experts:x"],
      ["experts:y", "combine:x"],
      ["router:weights", "combine:weights"],
    ];

    if (shared > 0) {
      // Several shared experts are one wider feed-forward network, which is how
      // DeepSeek implements them and gives the same parameter count.
      nodes.push({
        id: "shared",
        type: "gated_mlp",
        params: { d_model: D, hidden: `${shared}*${Fe}`, act: r.p.act, bias: r.p.bias === true },
      });
      nodes.push({ id: "merge", type: "add", params: { dim: D } });
      nodes.push(outNode);
      edges.push(
        ["_in:x", "shared:x"],
        ["combine:y", "merge:a"],
        ["shared:y", "merge:b"],
        ["merge:y", "_out:y"],
      );
    } else {
      nodes.push(outNode);
      edges.push(["combine:y", "_out:y"]);
    }

    return { nodes, edges };
  },
  constraints: (r) => {
    const out: BlockFinding[] = [];
    if (r.p.top_k > r.p.experts) {
      out.push({
        id: "MOE-01",
        severity: "error",
        param: "top_k",
        message: `top_k (${r.p.top_k}) cannot exceed the number of experts (${r.p.experts})`,
      });
    }
    return out;
  },
  docs: {
    summary: "Sparse feed-forward layer: a router picks top_k of the experts for each token.",
    formula:
      "total = router + experts*3*d_model*expert_hidden + shared*3*d_model*expert_hidden; " +
      "active swaps experts for top_k",
    refs: ["https://arxiv.org/abs/2401.06066", "https://arxiv.org/abs/2412.19437"],
  },
};

// ---------------------------------------------------------------------------
// State-space blocks
// ---------------------------------------------------------------------------

const mamba2Block: CompositeDef = {
  kind: "composite",
  type: "mamba2_block",
  category: "ssm",
  params: {
    d_model: { type: "int", min: 1 },
    expand: { type: "int", default: 2, min: 1, doc: "Inner width as a multiple of d_model" },
    head_dim: { type: "int", default: 64, min: 1, doc: "Width per state-space head" },
    state: { type: "int", default: 128, min: 1, doc: "Recurrent state width per head" },
    groups: { type: "int", default: 1, min: 1, doc: "Heads sharing one B/C projection" },
    conv_kernel: { type: "int", default: 4, min: 1 },
    conv_bias: { type: "bool", default: true },
    bias: { type: "bool", default: false, doc: "Bias on the input and output projections" },
  },
  ports: { in: { x: "... d_model" }, out: { y: "... d_model" } },
  expand: (raw, r) => {
    const D = ex(raw.d_model);
    const E = ex(raw.expand);
    const P = ex(raw.head_dim);
    const N = ex(raw.state);
    const G = ex(raw.groups);
    const bias = r.p.bias === true;

    const inner = `${E}*${D}`;
    const heads = `(${inner})/(${P})`;
    const bc = `2*(${G})*(${N})`;
    const xbc = `(${inner})+(${bc})`;
    const inProj = `2*(${inner})+(${bc})+(${heads})`;

    const { inNode, outNode } = streamBoundary(D);
    return {
      nodes: [
        inNode,
        { id: "in_proj", type: "linear", params: { in_features: D, out_features: inProj, bias } },
        {
          id: "split",
          type: "split",
          params: { from: `B T (${inProj})`, sizes: [inner, xbc, heads] },
        },
        { id: "gate_act", type: "activation", params: { kind: "silu", dim: inner } },
        {
          id: "conv",
          type: "conv1d",
          params: { channels: xbc, kernel: ex(raw.conv_kernel), bias: r.p.conv_bias !== false },
        },
        { id: "conv_act", type: "activation", params: { kind: "silu", dim: xbc } },
        {
          id: "scan",
          type: "ssd_scan",
          params: {
            d_inner: inner,
            heads,
            head_dim: P,
            state: N,
            groups: G,
            xbc_width: xbc,
          },
        },
        { id: "norm", type: "rmsnorm", params: { dim: inner } },
        { id: "gate", type: "mul", params: { dim: inner } },
        { id: "out_proj", type: "linear", params: { in_features: inner, out_features: D, bias } },
        outNode,
      ],
      edges: [
        ["_in:x", "in_proj:x"],
        ["in_proj:y", "split:x"],
        ["split:y0", "gate_act:x"],
        ["split:y1", "conv:x"],
        ["conv:y", "conv_act:x"],
        ["conv_act:y", "scan:xbc"],
        ["split:y2", "scan:dt"],
        ["scan:y", "norm:x"],
        ["norm:y", "gate:a"],
        ["gate_act:y", "gate:b"],
        ["gate:y", "out_proj:x"],
        ["out_proj:y", "_out:y"],
      ],
    };
  },
  docs: {
    summary:
      "Mamba-2 block. Its cost is linear in sequence length and it keeps a fixed state per sequence instead of a growing cache.",
    formula:
      "params = d_model*(2*d_inner + 2*groups*state + heads) + conv + 3*heads + d_inner + d_inner*d_model",
    refs: ["https://arxiv.org/abs/2405.21060"],
  },
};

// ---------------------------------------------------------------------------
// Transformer block
// ---------------------------------------------------------------------------

const transformerBlock: CompositeDef = {
  kind: "composite",
  type: "transformer_block",
  category: "block",
  params: {
    d_model: { type: "int", min: 1 },
    heads: { type: "int", min: 1 },
    kv_heads: { type: "int", min: 1 },
    head_dim: { type: "int", min: 1 },
    ffn_hidden: { type: "int", min: 1 },
    attention: { type: "enum", values: ["gqa", "mla"], default: "gqa" },
    q_lora: { type: "int", default: 0, min: 0, doc: "Latent attention: compressed query width" },
    kv_lora: { type: "int", default: 0, min: 0, doc: "Latent attention: cached latent width" },
    nope_dim: { type: "int", default: 0, min: 0 },
    rope_dim: { type: "int", default: 0, min: 0 },
    v_dim: { type: "int", default: 0, min: 0 },
    norm: { type: "enum", values: ["rmsnorm", "layernorm"], default: "rmsnorm" },
    norm_bias: { type: "bool", default: true, doc: "Bias on layernorm; ignored for rmsnorm" },
    post_norm: {
      type: "bool",
      default: false,
      doc: "Also normalize each sublayer's output before the residual add (Gemma 2/3)",
    },
    mlp: { type: "enum", values: ["gated", "dense", "moe"], default: "gated" },
    experts: { type: "int", default: 0, min: 0, doc: "Routed experts, when mlp is moe" },
    top_k: { type: "int", default: 1, min: 1 },
    expert_hidden: { type: "int", default: 0, min: 0, doc: "Hidden width of one expert" },
    shared_experts: { type: "int", default: 0, min: 0 },
    router_bias: { type: "bool", default: false },
    act: { type: "enum", values: ["silu", "gelu", "gelu_tanh", "relu", "relu2"], default: "silu" },
    attn_bias: { type: "bool", default: false },
    attn_o_bias: { type: "bool", default: null },
    mlp_bias: { type: "bool", default: false },
    qk_norm: { type: "bool", default: false },
    causal: { type: "bool", default: true },
    window: { type: "int", default: 0 },
    rope: ROPE_SPEC,
  },
  ports: { in: { x: "... d_model" }, out: { y: "... d_model" } },
  expand: (raw, r) => {
    const D = ex(raw.d_model);
    const normType = String(r.p.norm);
    const normParams = (): Record<string, ParamValue> =>
      normType === "layernorm" ? { dim: D, bias: r.p.norm_bias === true } : { dim: D };

    const { inNode, outNode } = streamBoundary(D);
    let mlpNode: NodeDef;
    if (r.p.mlp === "moe") {
      mlpNode = {
        id: "mlp",
        type: "moe_layer",
        params: {
          d_model: D,
          experts: ex(raw.experts),
          top_k: ex(raw.top_k),
          expert_hidden: ex(raw.expert_hidden),
          shared_experts: ex(raw.shared_experts, "0"),
          act: r.p.act,
          bias: r.p.mlp_bias === true,
          router_bias: r.p.router_bias === true,
        },
      };
    } else {
      mlpNode = {
        id: "mlp",
        type: r.p.mlp === "dense" ? "dense_mlp" : "gated_mlp",
        params: { d_model: D, hidden: ex(raw.ffn_hidden), act: r.p.act, bias: r.p.mlp_bias === true },
      };
    }

    const postNorm = r.p.post_norm === true;
    const nodes: NodeDef[] = [
      inNode,
      { id: "norm1", type: normType, params: normParams() },
      r.p.attention === "mla"
        ? {
            id: "attn",
            type: "mla_attention",
            params: {
              d_model: D,
              heads: ex(raw.heads),
              q_lora: ex(raw.q_lora),
              kv_lora: ex(raw.kv_lora),
              nope_dim: ex(raw.nope_dim),
              rope_dim: ex(raw.rope_dim),
              v_dim: ex(raw.v_dim),
              causal: r.p.causal === true,
              rope: r.p.rope,
              bias: r.p.attn_bias === true,
            },
          }
        : {
            id: "attn",
            type: "gqa_attention",
            params: {
              d_model: D,
              heads: ex(raw.heads),
              kv_heads: ex(raw.kv_heads),
              head_dim: ex(raw.head_dim),
              bias: r.p.attn_bias === true,
              o_bias: r.p.attn_o_bias,
              causal: r.p.causal === true,
              window: ex(raw.window, "0"),
              qk_norm: r.p.qk_norm === true,
              rope: r.p.rope,
            },
          },
      { id: "resid1", type: "add", params: { dim: D } },
      { id: "norm2", type: normType, params: normParams() },
      mlpNode,
      { id: "resid2", type: "add", params: { dim: D } },
      outNode,
    ];
    const edges: [string, string][] = [
      ["_in:x", "norm1:x"],
      ["norm1:y", "attn:x"],
      ["_in:x", "resid1:b"],
      ["resid1:y", "norm2:x"],
      ["norm2:y", "mlp:x"],
      ["resid1:y", "resid2:b"],
      ["resid2:y", "_out:y"],
    ];

    if (postNorm) {
      nodes.push({ id: "post_attn_norm", type: normType, params: normParams() });
      nodes.push({ id: "post_mlp_norm", type: normType, params: normParams() });
      edges.push(
        ["attn:y", "post_attn_norm:x"],
        ["post_attn_norm:y", "resid1:a"],
        ["mlp:y", "post_mlp_norm:x"],
        ["post_mlp_norm:y", "resid2:a"],
      );
    } else {
      edges.push(["attn:y", "resid1:a"], ["mlp:y", "resid2:a"]);
    }

    return { nodes, edges };
  },
  docs: {
    summary: "Pre-norm transformer block: norm, attention, residual, norm, feed-forward, residual.",
    formula: "params = attention + mlp + 2 norms",
  },
};

// ---------------------------------------------------------------------------
// Containers
// ---------------------------------------------------------------------------

const repeat: ContainerDef = {
  kind: "container",
  type: "repeat",
  category: "container",
  params: {
    count: { type: "int", min: 1, doc: "How many times the subgraph is stacked" },
    pattern: {
      type: "str",
      default: null,
      doc: "Optional variant pattern for hybrid stacks, e.g. \"MMMA\" repeated count times",
    },
  },
  multipliers: (r) => ({ total: r.p.count, active: r.p.count }),
  docs: {
    summary: "Stacks its subgraph count times. The subgraph's input and output shapes must match.",
  },
};

export const COMPOSITES: CompositeDef[] = [
  gqaAttention,
  mlaAttention,
  gatedMlp,
  denseMlp,
  moeLayer,
  mamba2Block,
  transformerBlock,
];
export const CONTAINERS: ContainerDef[] = [repeat, moeExperts];
