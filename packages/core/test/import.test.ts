/**
 * Import tests.
 *
 * Each config below is the architecture-relevant subset of the model's real
 * `config.json`. The test asserts that importing it lands on the same parameter
 * count as the hand-written preset, which is what makes the importer trustworthy.
 */

import { describe, expect, it } from "bun:test";
import { importHfConfig, type HfConfig } from "../src/import/hf.js";
import { countParams } from "../src/analysis/params.js";
import { resolveSymbols } from "../src/ir/symbols.js";
import { getPreset } from "../src/presets/index.js";
import { validate } from "../src/rules/index.js";

function paramsOf(doc: ReturnType<typeof getPreset>): number {
  return countParams(doc, resolveSymbols(doc)).total;
}

const CONFIGS: Record<string, HfConfig> = {
  "llama-3-8b": {
    model_type: "llama",
    num_hidden_layers: 32,
    hidden_size: 4096,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    head_dim: 128,
    intermediate_size: 14336,
    vocab_size: 128256,
    rope_theta: 500000,
    tie_word_embeddings: false,
    max_position_embeddings: 8192,
  },
  "mistral-7b": {
    model_type: "mistral",
    num_hidden_layers: 32,
    hidden_size: 4096,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    head_dim: 128,
    intermediate_size: 14336,
    vocab_size: 32000,
    rope_theta: 10000,
    sliding_window: 4096,
    max_position_embeddings: 32768,
  },
  "mixtral-8x7b": {
    model_type: "mixtral",
    num_hidden_layers: 32,
    hidden_size: 4096,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    head_dim: 128,
    intermediate_size: 14336,
    vocab_size: 32000,
    rope_theta: 1000000,
    num_local_experts: 8,
    num_experts_per_tok: 2,
    max_position_embeddings: 32768,
  },
  "qwen2.5-7b": {
    model_type: "qwen2",
    num_hidden_layers: 28,
    hidden_size: 3584,
    num_attention_heads: 28,
    num_key_value_heads: 4,
    intermediate_size: 18944,
    vocab_size: 152064,
    rope_theta: 1000000,
    use_sliding_window: false,
    max_position_embeddings: 32768,
  },
  "qwen3-8b": {
    model_type: "qwen3",
    num_hidden_layers: 36,
    hidden_size: 4096,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    head_dim: 128,
    intermediate_size: 12288,
    vocab_size: 151936,
    rope_theta: 1000000,
    max_position_embeddings: 32768,
  },
  "gemma-2-9b": {
    model_type: "gemma2",
    num_hidden_layers: 42,
    hidden_size: 3584,
    num_attention_heads: 16,
    num_key_value_heads: 8,
    head_dim: 256,
    intermediate_size: 14336,
    vocab_size: 256000,
    rope_theta: 10000,
    tie_word_embeddings: true,
    max_position_embeddings: 8192,
  },
  "gpt2-small": {
    model_type: "gpt2",
    n_layer: 12,
    n_embd: 768,
    n_head: 12,
    n_positions: 1024,
    vocab_size: 50257,
  },
  "deepseek-v3": {
    model_type: "deepseek_v3",
    num_hidden_layers: 61,
    hidden_size: 7168,
    num_attention_heads: 128,
    num_key_value_heads: 128,
    intermediate_size: 18432,
    vocab_size: 129280,
    rope_theta: 10000,
    q_lora_rank: 1536,
    kv_lora_rank: 512,
    qk_nope_head_dim: 128,
    qk_rope_head_dim: 64,
    v_head_dim: 128,
    n_routed_experts: 256,
    num_experts_per_tok: 8,
    moe_intermediate_size: 2048,
    n_shared_experts: 1,
    first_k_dense_replace: 3,
    max_position_embeddings: 4096,
  },
};

describe("Hugging Face config import", () => {
  for (const [preset, config] of Object.entries(CONFIGS)) {
    it(`reproduces ${preset}`, () => {
      const { doc, warnings } = importHfConfig(config, preset);
      expect(warnings).toEqual([]);
      expect(paramsOf(doc)).toBe(paramsOf(getPreset(preset)));
    });
  }

  it("produces a design that passes the design rules", () => {
    const { doc } = importHfConfig(CONFIGS["llama-3-8b"], "imported");
    const r = validate(doc, { T: 8192 });
    expect(r.findings.filter((f) => f.severity === "error")).toEqual([]);
  });

  it("carries the sliding window through", () => {
    const { doc } = importHfConfig(CONFIGS["mistral-7b"], "m");
    expect(doc.symbols.W).toEqual({ kind: "design", value: 4096, doc: "Sliding-window width" });
  });

  it("carries the leading dense layers through", () => {
    const { doc } = importHfConfig(CONFIGS["deepseek-v3"], "ds");
    expect(doc.graph.nodes.some((node) => node.id === "dense_layers")).toBe(true);
  });

  it("refuses a family it does not know rather than guessing", () => {
    expect(() => importHfConfig({ model_type: "some_new_thing", num_hidden_layers: 1 })).toThrow(
      /Unsupported model_type/,
    );
  });

  it("says what is missing when a field is absent", () => {
    expect(() => importHfConfig({ model_type: "llama" })).toThrow(/num_hidden_layers/);
  });

  it("warns instead of silently approximating a partly sparse stack", () => {
    const { warnings } = importHfConfig(
      { ...CONFIGS["mixtral-8x7b"], decoder_sparse_step: 2 },
      "partly-sparse",
    );
    expect(warnings.join(" ")).toMatch(/only every 2th layer is sparse/);
  });
});
