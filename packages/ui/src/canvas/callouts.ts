import type { NodeDef, Resolved, SymbolTable } from "@tensor-cad/engine";
/**
 * Callouts: the annotations a published architecture figure puts around the
 * drawing, each on a leader line pointing at the part it describes.
 *
 * "Supported context length of 1,024 tokens", "Embedding dimension of 1,600",
 * "25 heads", "Intermediate projection size: 4 x 1600 = 6,400". These are the
 * numbers a reader actually wants, and a figure puts them outside the part
 * rather than cramming them into the box.
 *
 * They are derived from the design, so they are never stale.
 */


export interface Callout {
  /** Path of the part this annotates. */
  path: string;
  /** Which side of the part the note sits on. */
  side: "left" | "right";
  /** Lines of the note. The first is emphasised. */
  lines: string[];
}

const fmt = (n: unknown): string =>
  typeof n === "number" && Number.isFinite(n) ? n.toLocaleString("en-US") : String(n ?? "?");

/**
 * What is worth saying about one part.
 *
 * Deliberately sparse: a figure with a note on every box is as unreadable as
 * one with none. Only the parts that carry a defining number get a callout.
 */
export function calloutFor(
  node: NodeDef,
  resolved: Resolved | undefined,
  symbols: SymbolTable,
  path: string,
): Callout | null {
  const p = resolved?.p ?? {};

  switch (node.type) {
    case "input":
      return {
        path,
        side: "left",
        lines: [
          `Supported context length of ${fmt(symbols.values.T)} tokens`,
          "batch and sequence stay symbolic until a run",
        ],
      };

    case "embedding":
      return {
        path,
        side: "left",
        lines: [
          `Embedding dimension of ${fmt(p.dim)}`,
          `Vocabulary size of ${fmt(p.vocab)}`,
        ],
      };

    case "pos_embedding":
      return { path, side: "left", lines: [`Learned positions up to ${fmt(p.max_seq)}`] };

    case "lm_head":
      return {
        path,
        side: "right",
        lines: [
          `Vocabulary size of ${fmt(p.vocab)}`,
          p.tied ? "weights shared with the embedding" : "own output weights",
        ],
      };

    case "repeat":
      return {
        path,
        side: "right",
        lines: [`${fmt(p.count)} stacked layers`],
      };

    case "transformer_block": {
      const heads = p.heads;
      const kv = p.kv_heads;
      const lines = [
        kv === heads
          ? `${fmt(heads)} attention heads`
          : `${fmt(heads)} query heads, ${fmt(kv)} key/value heads`,
        `head dimension ${fmt(p.head_dim)}`,
      ];
      if (p.mlp === "moe") {
        lines.push(`${fmt(p.experts)} experts, ${fmt(p.top_k)} active per token`);
      } else if (typeof p.ffn_hidden === "number" && typeof p.d_model === "number") {
        const ratio = p.ffn_hidden / p.d_model;
        lines.push(
          `Intermediate size: ${ratio % 1 === 0 ? `${ratio} x ` : ""}${fmt(p.d_model)} = ${fmt(p.ffn_hidden)}`,
        );
      }
      if (p.window) lines.push(`sliding window of ${fmt(p.window)} tokens`);
      return { path, side: "right", lines };
    }

    case "gqa_attention": {
      const heads = p.heads;
      const kv = p.kv_heads;
      return {
        path,
        side: "right",
        lines: [
          kv === heads ? `${fmt(heads)} heads` : `${fmt(heads)} query heads, ${fmt(kv)} key/value heads`,
          `head dimension ${fmt(p.head_dim)}`,
        ],
      };
    }

    case "mla_attention":
      return {
        path,
        side: "right",
        lines: [
          `${fmt(p.heads)} heads, latent width ${fmt(p.kv_lora)}`,
          "only the latent is cached",
        ],
      };

    case "moe_layer":
      return {
        path,
        side: "right",
        lines: [
          `${fmt(p.experts)} experts, ${fmt(p.top_k)} active per token`,
          `expert width ${fmt(p.expert_hidden)}`,
        ],
      };

    case "mamba2_block":
      return {
        path,
        side: "right",
        lines: [`state width ${fmt(p.state)}`, "state is fixed per sequence, not per token"],
      };

    case "gated_mlp":
    case "dense_mlp": {
      if (typeof p.hidden !== "number" || typeof p.d_model !== "number") return null;
      const ratio = p.hidden / p.d_model;
      return {
        path,
        side: "right",
        lines: [
          `Intermediate projection size:`,
          `${ratio % 1 === 0 ? `${ratio} x ` : ""}${fmt(p.d_model)} = ${fmt(p.hidden)}`,
        ],
      };
    }

    default:
      return null;
  }
}
