/**
 * Builders for the reference presets.
 *
 * Every preset is also a regression test: `meta.published` carries the figure
 * reported by the model's authors, and the test suite asserts our analysis
 * reproduces it.
 */

import type { Doc, Graph, NodeDef, ParamValue, SymbolDef } from "../ir/types.js";
import { DOC_VERSION } from "../ir/types.js";

export interface DecoderSpec {
  name: string;
  family?: string;
  notes?: string;

  layers: number;
  dModel: number;
  heads: number;
  kvHeads?: number;
  headDim?: number | string;
  /** Feed-forward width. May be an expression over the other symbols. */
  ffnHidden: number | string;
  vocab: number;

  norm?: "rmsnorm" | "layernorm";
  normBias?: boolean;
  postNorm?: boolean;
  mlp?: "gated" | "dense";
  act?: "silu" | "gelu" | "gelu_tanh" | "relu" | "relu2";

  /** RoPE settings, or null for learned absolute positions. */
  rope?: { theta: number } | null;
  /** Maximum position count when using learned absolute positions. */
  maxSeq?: number;

  tied?: boolean;
  attnBias?: boolean;
  attnOBias?: boolean | null;
  mlpBias?: boolean;
  qkNorm?: boolean;
  window?: number;

  /**
   * An irregular stack, written as one character per layer:
   * `M` state-space, `A` attention, `F` feed-forward. Used by hybrid models
   * whose layer pattern does not repeat on a fixed period.
   */
  hybrid?: {
    pattern: string;
    mamba: { expand?: number; headDim?: number; state?: number; groups?: number; convKernel?: number };
  };

  /** Latent attention. When present, kvHeads and headDim are unused. */
  mla?: {
    qLora: number;
    kvLora: number;
    nopeDim: number;
    ropeDim: number;
    vDim: number;
  };

  /** Sparse feed-forward settings. Omit for a dense model. */
  moe?: {
    experts: number;
    topK: number;
    expertHidden: number;
    sharedExperts?: number;
    routerBias?: boolean;
    /** Leading layers that keep a dense feed-forward (DeepSeek, Qwen). */
    denseLayers?: number;
  };

  defaultSeq?: number;

  published?: {
    params?: number;
    activeParams?: number;
    kvBytesPerToken?: number;
    source?: string;
    tolerance?: number;
  };
}

export function decoderOnly(spec: DecoderSpec): Doc {
  const kvHeads = spec.kvHeads ?? spec.heads;
  const headDim = spec.headDim ?? spec.dModel / spec.heads;
  const norm = spec.norm ?? "rmsnorm";
  const mlp = spec.mlp ?? "gated";
  const act = spec.act ?? (mlp === "gated" ? "silu" : "gelu");
  const rope = spec.rope === undefined ? { theta: 10000 } : spec.rope;
  const tied = spec.tied ?? false;

  const symbols: Record<string, SymbolDef> = {
    B: { kind: "runtime", default: 1, doc: "Batch size" },
    T: { kind: "runtime", default: spec.defaultSeq ?? 4096, doc: "Sequence length in tokens" },
    L: { kind: "design", value: spec.layers, doc: "Number of transformer layers" },
    D: { kind: "design", value: spec.dModel, doc: "Residual stream width (d_model)" },
    H: { kind: "design", value: spec.heads, doc: "Query heads" },
    Hkv: { kind: "design", value: kvHeads, doc: "Key/value heads" },
    dh: { kind: "design", value: headDim, doc: "Head dimension" },
    F: { kind: "design", value: spec.ffnHidden, doc: "Feed-forward hidden width" },
    V: { kind: "design", value: spec.vocab, doc: "Vocabulary size" },
  };
  if (spec.maxSeq) {
    symbols.Tmax = { kind: "design", value: spec.maxSeq, doc: "Maximum position index" };
  }
  if (spec.window) {
    symbols.W = { kind: "design", value: spec.window, doc: "Sliding-window width" };
  }
  if (spec.mla) {
    symbols.Ql = { kind: "design", value: spec.mla.qLora, doc: "Compressed query width" };
    symbols.Kl = { kind: "design", value: spec.mla.kvLora, doc: "Cached latent width" };
    symbols.dnope = { kind: "design", value: spec.mla.nopeDim, doc: "Per-head width without position" };
    symbols.drope = { kind: "design", value: spec.mla.ropeDim, doc: "Per-head rotary width" };
    symbols.dv = { kind: "design", value: spec.mla.vDim, doc: "Per-head value width" };
  }
  if (spec.moe) {
    symbols.E = { kind: "design", value: spec.moe.experts, doc: "Routed experts per layer" };
    symbols.K = { kind: "design", value: spec.moe.topK, doc: "Experts each token is routed to" };
    symbols.Fe = { kind: "design", value: spec.moe.expertHidden, doc: "Hidden width of one expert" };
    if (spec.moe.sharedExperts) {
      symbols.Ns = { kind: "design", value: spec.moe.sharedExperts, doc: "Always-on shared experts" };
    }
    if (spec.moe.denseLayers) {
      symbols.Ld = { kind: "design", value: spec.moe.denseLayers, doc: "Leading dense layers" };
      symbols.Lm = { kind: "design", value: "L - Ld", doc: "Sparse layers" };
    }
  }

  const blockParams: Record<string, ParamValue> = {
    d_model: "D",
    heads: "H",
    kv_heads: "Hkv",
    head_dim: "dh",
    ffn_hidden: "F",
    norm,
    norm_bias: spec.normBias ?? true,
    post_norm: spec.postNorm ?? false,
    mlp,
    act,
    attn_bias: spec.attnBias ?? false,
    attn_o_bias: spec.attnOBias ?? null,
    mlp_bias: spec.mlpBias ?? false,
    qk_norm: spec.qkNorm ?? false,
    causal: true,
    window: spec.window ? "W" : 0,
    rope: rope ? { theta: rope.theta } : null,
    ...(spec.mla
      ? {
          attention: "mla",
          q_lora: "Ql",
          kv_lora: "Kl",
          nope_dim: "dnope",
          rope_dim: "drope",
          v_dim: "dv",
        }
      : {}),
  };

  const makeStack = (id: string, count: string, params: Record<string, ParamValue>, label: string): NodeDef => ({
    id,
    type: "repeat",
    label,
    params: { count },
    graph: {
      nodes: [
        { id: "_in", type: "boundary_in", params: { ports: { x: "B T D" } } },
        { id: "block", type: "transformer_block", params },
        { id: "_out", type: "boundary_out", params: { ports: { x: "B T D" } } },
      ],
      edges: [
        ["_in:x", "block:x"],
        ["block:y", "_out:x"],
      ],
    },
  });

  const moeParams: Record<string, ParamValue> = spec.moe
    ? {
        ...blockParams,
        mlp: "moe",
        experts: "E",
        top_k: "K",
        expert_hidden: "Fe",
        shared_experts: spec.moe.sharedExperts ? "Ns" : 0,
        router_bias: spec.moe.routerBias ?? false,
      }
    : blockParams;

  const stacks: NodeDef[] = [];
  if (spec.hybrid) {
    const m = spec.hybrid.mamba;
    const nodes: NodeDef[] = [{ id: "_in", type: "boundary_in", params: { ports: { x: "B T D" } } }];
    const edges: Graph["edges"] = [];
    let tailPort = "_in:x";

    [...spec.hybrid.pattern].forEach((kind, i) => {
      const normId = `norm${i}`;
      const blockId = `blk${i}`;
      const addId = `add${i}`;
      nodes.push({ id: normId, type: "rmsnorm", params: { dim: "D" } });

      if (kind === "M") {
        nodes.push({
          id: blockId,
          type: "mamba2_block",
          label: `Mamba-2 ${i}`,
          params: {
            d_model: "D",
            expand: m.expand ?? 2,
            head_dim: m.headDim ?? 64,
            state: m.state ?? 128,
            groups: m.groups ?? 1,
            conv_kernel: m.convKernel ?? 4,
          },
        });
      } else if (kind === "A") {
        nodes.push({
          id: blockId,
          type: "gqa_attention",
          label: `Attention ${i}`,
          params: {
            d_model: "D",
            heads: "H",
            kv_heads: "Hkv",
            head_dim: "dh",
            causal: true,
            rope: rope ? { theta: rope.theta } : null,
            bias: spec.attnBias ?? false,
          },
        });
      } else {
        nodes.push({
          id: blockId,
          type: "dense_mlp",
          label: `Feed-forward ${i}`,
          params: { d_model: "D", hidden: "F", act, bias: spec.mlpBias ?? false },
        });
      }

      nodes.push({ id: addId, type: "add", params: { dim: "D" } });
      edges.push([tailPort, `${normId}:x`], [`${normId}:y`, `${blockId}:x`]);
      edges.push([`${blockId}:y`, `${addId}:a`], [tailPort, `${addId}:b`]);
      tailPort = `${addId}:y`;
    });

    nodes.push({ id: "_out", type: "boundary_out", params: { ports: { x: "B T D" } } });
    edges.push([tailPort, "_out:x"]);
    stacks.push({
      id: "layers",
      type: "repeat",
      label: `Hybrid stack, ${spec.hybrid.pattern.length} layers`,
      params: { count: 1 },
      graph: { nodes, edges },
    });
  } else if (spec.moe?.denseLayers) {
    stacks.push(makeStack("dense_layers", "Ld", blockParams, `Dense block x${spec.moe.denseLayers}`));
    stacks.push(makeStack("layers", "Lm", moeParams, `Sparse block x${spec.layers - spec.moe.denseLayers}`));
  } else {
    stacks.push(makeStack("layers", "L", moeParams, `Transformer block x${spec.layers}`));
  }

  const nodes: NodeDef[] = [
    { id: "tokens", type: "input", params: { shape: "B T", dtype: "int64" } },
    { id: "embed", type: "embedding", params: { vocab: "V", dim: "D" } },
  ];
  const edges: Graph["edges"] = [["tokens:x", "embed:ids"]];
  let tail = "embed:y";

  if (spec.maxSeq) {
    nodes.push({ id: "pos", type: "pos_embedding", params: { max_seq: "Tmax", dim: "D" } });
    edges.push([tail, "pos:x"]);
    tail = "pos:y";
  }

  for (const stack of stacks) {
    nodes.push(stack);
    edges.push([tail, `${stack.id}:x`]);
    tail = `${stack.id}:x`;
  }

  nodes.push({
    id: "final_norm",
    type: norm,
    params: norm === "layernorm" ? { dim: "D", bias: spec.normBias ?? true } : { dim: "D" },
  });
  nodes.push({ id: "head", type: "lm_head", params: { vocab: "V", dim: "D", tied } });
  nodes.push({ id: "logits", type: "output" });

  edges.push([tail, "final_norm:x"], ["final_norm:y", "head:x"], ["head:y", "logits:x"]);

  return {
    version: DOC_VERSION,
    meta: {
      name: spec.name,
      family: spec.family,
      notes: spec.notes,
      published: spec.published,
    },
    symbols,
    graph: { nodes, edges },
    ui: {},
  };
}
