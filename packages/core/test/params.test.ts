/**
 * Regression suite: the parameter count of every reference preset must match
 * the figure its authors published.
 *
 * Sources for each number are in docs/research/02-analysis-math.md.
 */

import { describe, expect, it } from "bun:test";
import { countParams } from "../src/analysis/params.js";
import { flatten } from "../src/analysis/flatten.js";
import { resolveSymbols } from "../src/ir/symbols.js";
import { allPresets, getPreset, PRESET_NAMES } from "../src/presets/index.js";

/** Models whose exact count we can reproduce to the parameter. */
const EXACT = new Set([
  "gpt2-small",
  "gpt2-medium",
  "gpt2-large",
  "gpt2-xl",
  "llama-2-7b",
  "mistral-7b",
  "llama-3-8b",
  "llama-3-70b",
  "gemma-2-9b",
  "mixtral-8x7b",
]);

describe("parameter regression against published counts", () => {
  for (const name of PRESET_NAMES) {
    it(name, () => {
      const doc = getPreset(name);
      const symbols = resolveSymbols(doc);
      expect(symbols.errors).toEqual([]);

      const result = countParams(doc, symbols);
      expect(result.errors).toEqual([]);

      const published = doc.meta.published?.params;
      expect(published).toBeDefined();

      const tolerance = doc.meta.published?.tolerance ?? 0.005;
      if (EXACT.has(name)) {
        expect(result.total).toBe(published!);
      } else {
        const rel = Math.abs(result.total - published!) / published!;
        expect({ name, rel: rel < tolerance }).toEqual({ name, rel: true });
      }

      const activePublished = doc.meta.published?.activeParams;
      if (activePublished) {
        if (EXACT.has(name)) {
          expect(result.active).toBe(activePublished);
        } else {
          const rel = Math.abs(result.active - activePublished) / activePublished;
          expect({ name, rel: rel < tolerance }).toEqual({ name, rel: true });
        }
      } else {
        // A dense design activates every weight it holds.
        expect(result.active).toBe(result.total);
      }
    });
  }
});

describe("parameter breakdown", () => {
  it("splits embeddings from the rest", () => {
    const doc = getPreset("llama-3-8b");
    const symbols = resolveSymbols(doc);
    const r = countParams(doc, symbols);

    // 128256 * 4096, counted once for the token embedding.
    expect(r.embedding).toBe(128256 * 4096);
    // Untied head of the same size.
    expect(r.head).toBe(128256 * 4096);
    expect(r.nonEmbedding).toBe(r.total - r.embedding);
  });

  it("counts a tied head as zero parameters", () => {
    const doc = getPreset("gpt2-small");
    const symbols = resolveSymbols(doc);
    const r = countParams(doc, symbols);
    expect(r.head).toBe(0);
  });

  it("attributes weights to the expected categories", () => {
    const doc = getPreset("llama-3-8b");
    const symbols = resolveSymbols(doc);
    const r = countParams(doc, symbols);

    const L = 32;
    const D = 4096;
    const H = 32;
    const Hkv = 8;
    const dh = 128;
    const F = 14336;

    const attn = D * H * dh + 2 * D * Hkv * dh + H * dh * D;
    const mlp = 3 * D * F;
    // q/k/v/o and gate/up/down are all linear; the head is its own category.
    expect(r.byCategory.linear).toBe(L * (attn + mlp));
    expect(r.byCategory.norm).toBe(L * 2 * D + D);
    expect(r.byCategory.embedding).toBe(128256 * D);
    expect(r.byCategory.head).toBe(128256 * D);
  });

  it("scales with the repeat count", () => {
    const doc = getPreset("llama-3-8b");
    const symbols = resolveSymbols(doc);
    const base = countParams(doc, symbols).total;

    const doubled = getPreset("llama-3-8b");
    doubled.symbols.L = { kind: "design", value: 64 };
    const sym2 = resolveSymbols(doubled);
    const grown = countParams(doubled, sym2).total;

    const perLayer = (base - 2 * 128256 * 4096 - 4096) / 32;
    expect(grown - base).toBe(perLayer * 32);
  });
});

describe("flattening", () => {
  it("keeps the graph small by not unrolling repeats", () => {
    const doc = getPreset("llama-3.1-405b");
    const symbols = resolveSymbols(doc);
    const f = flatten(doc, symbols);
    expect(f.errors).toEqual([]);
    // A 126-layer model still flattens to a few dozen primitive nodes.
    expect(f.nodes.length).toBeLessThan(60);
    expect(f.repeats).toEqual([{ path: "layers", type: "repeat", count: 126, active: 126 }]);
  });

  it("expands composites down to primitives only", () => {
    const doc = getPreset("llama-3-8b");
    const symbols = resolveSymbols(doc);
    const f = flatten(doc, symbols);
    for (const n of f.nodes) {
      expect(n.def.kind).toBe("primitive");
    }
  });

  it("every preset flattens without errors", () => {
    for (const doc of allPresets()) {
      const symbols = resolveSymbols(doc);
      const f = flatten(doc, symbols);
      expect({ name: doc.meta.name, errors: f.errors }).toEqual({ name: doc.meta.name, errors: [] });
    }
  });
});
