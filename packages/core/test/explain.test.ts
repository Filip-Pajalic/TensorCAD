import { describe, expect, it } from "bun:test";
import { explain, explainAll } from "../src/explain.js";
import { getPreset } from "../src/presets/index.js";

describe("explaining a block", () => {
  it("shows both the expression somebody wrote and the value it became", () => {
    const e = explain(getPreset("llama-3-8b"), "layers/block");
    expect(e.params.d_model.expression).toBe("D");
    expect(e.params.d_model.value).toBe(4096);
    // The feed-forward width is a symbol, which is itself an expression.
    expect(e.params.ffn_hidden.expression).toBe("F");
    expect(e.params.ffn_hidden.value).toBe(14336);
  });

  it("reports how many copies exist and how many a token passes through", () => {
    const dense = explain(getPreset("llama-3-8b"), "layers/block");
    expect(dense.copies).toEqual({ total: 32, active: 32 });

    // Eight experts per layer across 32 layers, two of which a token visits.
    const sparse = explain(getPreset("mixtral-8x7b"), "layers/block/mlp/experts/expert");
    expect(sparse.copies.total).toBe(32 * 8);
    expect(sparse.copies.active).toBe(32 * 2);
  });

  it("attributes a share of the model to the block", () => {
    const e = explain(getPreset("llama-3-8b"), "layers/block/mlp");
    // Three matrices of 4096 by 14336, across 32 layers.
    expect(e.contributes.params).toBe(32 * 3 * 4096 * 14336);
    expect(e.contributes.shareOfParams).toBeGreaterThan(0.6);
    expect(e.contributes.shareOfParams).toBeLessThan(0.75);
  });

  it("carries the documentation and its sources", () => {
    const e = explain(getPreset("llama-3-8b"), "layers/block/attn");
    expect(e.docs.summary).toContain("Grouped-query");
    expect(e.docs.formula).toContain("d_model*heads*head_dim");
    expect(e.docs.refs?.[0]).toContain("arxiv");
  });

  it("shows the inferred port shapes", () => {
    const e = explain(getPreset("llama-3-8b"), "layers/block/attn");
    expect(Object.keys(e.shapes.in)).toEqual(["x"]);
    expect(Object.keys(e.shapes.out)).toEqual(["y"]);
  });

  it("breaks a block down into the primitives that carry the weights", () => {
    const e = explain(getPreset("llama-3-8b"), "layers/block/attn");
    const types = e.breakdown.map((b) => b.type);
    expect(types).toContain("linear");
    // The largest single matrix in grouped-query attention is the query or
    // output projection, not a key or value projection.
    expect(e.breakdown[0].path).toMatch(/q_proj|o_proj/);
  });

  it("reports the cache a block is responsible for", () => {
    const e = explain(getPreset("deepseek-v3"), "layers/block/attn");
    // 58 sparse layers, each caching 576 bytes' worth of latent per token.
    expect(e.contributes.cacheBytesPerToken).toBe(58 * 576 * 2);
  });

  it("says so plainly when the path names nothing", () => {
    const e = explain(getPreset("llama-3-8b"), "nope/not/here");
    expect(e.notFound).toBe(true);
  });

  it("ranks every block by what it costs", () => {
    const all = explainAll(getPreset("llama-3-8b"));
    expect(all.length).toBeGreaterThan(3);
    for (let i = 1; i < all.length; i++) {
      expect(all[i - 1].contributes.params).toBeGreaterThanOrEqual(all[i].contributes.params);
    }
  });
});
