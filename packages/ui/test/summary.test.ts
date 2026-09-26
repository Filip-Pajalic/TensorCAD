/**
 * What a part says on its face.
 *
 * The line under a block's name used to be its parameters' code names —
 * `D 48 · dh 16 · ffn 192` — and ran into an ellipsis on anything larger than
 * a toy. It is a phrase now, the way a figure annotates a block, and it drops
 * its least important parts rather than being cut off in the middle of one.
 */

import { beforeAll, describe, expect, test } from "bun:test";
import type { Resolved } from "@tensor-cad/engine";
import { loadEngine } from "../src/engine.js";

let blockDef: typeof import("../src/engine.js").blockDef;
let paramSummary: typeof import("../src/canvas/blocks.js").paramSummary;
let SUMMARY_ROOM: typeof import("../src/canvas/blocks.js").SUMMARY_ROOM;

beforeAll(async () => {
  await loadEngine();
  ({ blockDef } = await import("../src/engine.js"));
  ({ paramSummary, SUMMARY_ROOM } = await import("../src/canvas/blocks.js"));
});

/** A block of a type with these values, as the resolver would hand it over. */
function said(type: string, p: Record<string, unknown>, room?: number): string {
  return paramSummary(blockDef(type), { p } as unknown as Resolved, "symbolic", undefined, room);
}

describe("a part's summary", () => {
  test("is a phrase a figure would write", () => {
    expect(said("embedding", { vocab: 128256, dim: 4096 })).toBe("128,256 × 4,096");
    expect(said("lm_head", { dim: 4096, vocab: 128256, tied: true })).toBe("4,096 → 128,256 · tied");
    expect(said("gqa_attention", { heads: 32, kv_heads: 8, head_dim: 128 })).toBe("32 heads × 128 · 8 kv");
    expect(said("gqa_attention", { heads: 12, kv_heads: 12, head_dim: 64 })).toBe("12 heads × 64");
    expect(said("gated_mlp", { hidden: 14336, act: "silu" })).toBe("14,336 wide · SiLU");
    expect(said("moe_layer", { experts: 256, top_k: 8 })).toBe("256 experts · top 8");
    expect(said("rmsnorm", { dim: 4096 })).toBe("width 4,096");
  });

  test("says an enum in the words the catalog gives it", () => {
    expect(said("activation", { kind: "gelu_tanh" })).toBe("GELU, tanh approximation");
  });

  test("drops what does not fit rather than trailing off", () => {
    const full = said("sdpa", { heads: 32, kv_heads: 8, head_dim: 128, causal: true }, 40);
    expect(full).toBe("32 heads × 128 · 8 kv · causal");
    // Beside a parameter count there is room for twenty-one characters.
    const beside = said("sdpa", { heads: 32, kv_heads: 8, head_dim: 128, causal: true }, SUMMARY_ROOM.beside);
    expect(beside).toBe("32 heads × 128 · 8 kv");
    expect(beside.length).toBeLessThanOrEqual(SUMMARY_ROOM.beside);
  });

  test("a block with no phrase of its own is described by its parameters' labels", () => {
    // conv1d has none: its channels, by what the catalog calls them.
    expect(said("conv1d", { channels: 8192, kernel: 4, bias: true })).toBe("channels 8,192");
  });

  test("never by a code name", () => {
    for (const [type, p] of [
      ["transformer_block", { d_model: 4096, heads: 32, kv_heads: 8, head_dim: 128, ffn_hidden: 14336, mlp: "gated" }],
      ["gqa_attention", { d_model: 4096, heads: 32, kv_heads: 8, head_dim: 128 }],
      ["rmsnorm", { dim: 4096 }],
    ] as const) {
      expect(said(type, p)).not.toMatch(/\bdh\b|\bd_model\b|\bD \d|\bkv_heads\b/);
    }
  });
});
