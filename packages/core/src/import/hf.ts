/**
 * Import a Hugging Face `config.json`.
 *
 * `config.json` is the de facto interchange format for open model
 * architectures, so reading it is the fastest way to get a real design onto the
 * canvas. Each family is mapped explicitly rather than guessed: an unknown
 * `model_type` is an error, not a silent approximation.
 */

import type { Doc } from "../ir/types.js";
import { decoderOnly, type DecoderSpec } from "../presets/builders.js";

export interface HfConfig {
  model_type?: string;
  architectures?: string[];
  [key: string]: unknown;
}

export interface ImportResult {
  doc: Doc;
  /** Things the import could not represent faithfully. */
  warnings: string[];
}

const n = (c: HfConfig, ...keys: string[]): number | undefined => {
  for (const k of keys) {
    const v = c[k];
    if (typeof v === "number" && Number.isFinite(v)) return v;
  }
  return undefined;
};

const b = (c: HfConfig, ...keys: string[]): boolean | undefined => {
  for (const k of keys) {
    const v = c[k];
    if (typeof v === "boolean") return v;
  }
  return undefined;
};

function required(value: number | undefined, field: string): number {
  if (value === undefined) throw new Error(`config.json is missing "${field}"`);
  return value;
}

/** Families whose layout this importer knows. */
export const SUPPORTED_MODEL_TYPES = [
  "gpt2",
  "llama",
  "mistral",
  "mixtral",
  "qwen2",
  "qwen3",
  "qwen3_moe",
  "gemma",
  "gemma2",
  "deepseek_v3",
] as const;

export function importHfConfig(config: HfConfig, name?: string): ImportResult {
  const warnings: string[] = [];
  const type = String(config.model_type ?? "").toLowerCase();
  if (!type) throw new Error('config.json has no "model_type"');
  if (!(SUPPORTED_MODEL_TYPES as readonly string[]).includes(type)) {
    throw new Error(
      `Unsupported model_type "${type}". This importer knows: ${SUPPORTED_MODEL_TYPES.join(", ")}.`,
    );
  }

  const modelName = name ?? String(config._name_or_path ?? type);

  // --- GPT-2 uses its own field names -------------------------------------
  if (type === "gpt2") {
    const spec: DecoderSpec = {
      name: modelName,
      family: "gpt2",
      layers: required(n(config, "n_layer", "num_hidden_layers"), "n_layer"),
      dModel: required(n(config, "n_embd", "hidden_size"), "n_embd"),
      heads: required(n(config, "n_head", "num_attention_heads"), "n_head"),
      ffnHidden: n(config, "n_inner") ?? "4*D",
      vocab: required(n(config, "vocab_size"), "vocab_size"),
      maxSeq: n(config, "n_positions", "n_ctx") ?? 1024,
      norm: "layernorm",
      mlp: "dense",
      act: "gelu",
      rope: null,
      tied: b(config, "tie_word_embeddings") ?? true,
      attnBias: true,
      mlpBias: true,
      defaultSeq: n(config, "n_positions") ?? 1024,
    };
    return { doc: decoderOnly(spec), warnings };
  }

  // --- Everything else follows the Llama field names ----------------------
  const layers = required(n(config, "num_hidden_layers"), "num_hidden_layers");
  const dModel = required(n(config, "hidden_size"), "hidden_size");
  const heads = required(n(config, "num_attention_heads"), "num_attention_heads");
  const kvHeads = n(config, "num_key_value_heads") ?? heads;
  const headDim = n(config, "head_dim") ?? dModel / heads;
  const vocab = required(n(config, "vocab_size"), "vocab_size");

  const spec: DecoderSpec = {
    name: modelName,
    family: type,
    layers,
    dModel,
    heads,
    kvHeads,
    headDim,
    ffnHidden: required(n(config, "intermediate_size"), "intermediate_size"),
    vocab,
    rope: { theta: n(config, "rope_theta") ?? 10000 },
    tied: b(config, "tie_word_embeddings") ?? false,
    attnBias: b(config, "attention_bias") ?? false,
    mlpBias: b(config, "mlp_bias") ?? false,
    defaultSeq: Math.min(n(config, "max_position_embeddings") ?? 4096, 32768),
  };

  // Qwen 2.5 puts a bias on q/k/v but not on the output projection.
  if (type === "qwen2" && (config.use_sliding_window !== undefined || config.attention_dropout !== undefined)) {
    spec.attnBias = true;
    spec.attnOBias = false;
  }
  if (type === "qwen2" && b(config, "attention_bias") === undefined) {
    spec.attnBias = true;
    spec.attnOBias = false;
  }

  // Qwen 3 normalizes each attention head's queries and keys.
  if (type === "qwen3" || type === "qwen3_moe") {
    spec.qkNorm = true;
  }

  // Gemma normalizes both the input and the output of every sublayer.
  if (type === "gemma2" || type === "gemma") {
    spec.postNorm = type === "gemma2";
    spec.act = "gelu_tanh";
    spec.tied = b(config, "tie_word_embeddings") ?? true;
    if (n(config, "sliding_window") && type === "gemma2") {
      warnings.push(
        "Gemma 2 alternates sliding-window and full-attention layers. This import makes every layer full-attention, " +
          "which is right for the parameter count but understates how much the cache is reduced.",
      );
    }
  }

  const window = n(config, "sliding_window");
  if (window && b(config, "use_sliding_window") !== false && type === "mistral") {
    spec.window = window;
  }

  // --- Sparse feed-forward -------------------------------------------------
  const experts = n(config, "num_local_experts", "num_experts", "n_routed_experts");
  const topK = n(config, "num_experts_per_tok");
  if (experts && topK) {
    const expertHidden = n(config, "moe_intermediate_size") ?? spec.ffnHidden;
    if (typeof expertHidden !== "number") {
      throw new Error("Could not determine the expert width (moe_intermediate_size)");
    }
    const sharedIntermediate = n(config, "shared_expert_intermediate_size");
    let sharedExperts = n(config, "n_shared_experts") ?? 0;
    if (!sharedExperts && sharedIntermediate) {
      sharedExperts = Math.round(sharedIntermediate / expertHidden);
    }
    spec.moe = {
      experts,
      topK,
      expertHidden,
      sharedExperts: sharedExperts || undefined,
      routerBias: type === "deepseek_v3",
      denseLayers: n(config, "first_k_dense_replace") ?? undefined,
    };
    const step = n(config, "decoder_sparse_step");
    if (step !== undefined && step !== 1) {
      warnings.push(
        `decoder_sparse_step is ${step}, so only every ${step}th layer is sparse. This import makes them all sparse.`,
      );
    }
    if (Array.isArray(config.mlp_only_layers) && config.mlp_only_layers.length > 0) {
      warnings.push(
        `mlp_only_layers lists ${config.mlp_only_layers.length} dense layer(s) that this import does not reproduce.`,
      );
    }
  }

  // --- Latent attention ----------------------------------------------------
  const kvLora = n(config, "kv_lora_rank");
  if (kvLora) {
    const qLora = n(config, "q_lora_rank");
    if (!qLora) {
      warnings.push(
        "This model compresses keys and values but not queries. The import uses a full-width query path, which changes the parameter count.",
      );
    }
    spec.mla = {
      qLora: qLora ?? dModel,
      kvLora,
      nopeDim: required(n(config, "qk_nope_head_dim"), "qk_nope_head_dim"),
      ropeDim: required(n(config, "qk_rope_head_dim"), "qk_rope_head_dim"),
      vDim: n(config, "v_head_dim") ?? headDim,
    };
    spec.headDim = spec.mla.nopeDim + spec.mla.ropeDim;
  }

  if (n(config, "num_nextn_predict_layers")) {
    warnings.push(
      "This model has a multi-token-prediction module, which the import leaves out. Its published total may include those weights.",
    );
  }

  return { doc: decoderOnly(spec), warnings };
}

/** Convenience wrapper for a JSON string. */
export function importHfConfigJson(text: string, name?: string): ImportResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    throw new Error(`Could not parse config.json: ${(e as Error).message}`);
  }
  if (!parsed || typeof parsed !== "object") throw new Error("config.json is not an object");
  return importHfConfig(parsed as HfConfig, name);
}
