/**
 * Import tests.
 *
 * Each config below is the architecture-relevant subset of the model's real
 * `config.json`. The test asserts that importing it lands on the same parameter
 * count as the hand-written preset, which is what makes the importer trustworthy.
 */

import { describe, expect, it } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { importHfConfig, type HfConfig } from "../src/import/hf.js";
import { countParams } from "../src/analysis/params.js";
import { resolveSymbols } from "../src/ir/symbols.js";
import { getPreset } from "../src/presets/index.js";
import { validate } from "../src/rules/index.js";

function paramsOf(doc: ReturnType<typeof getPreset>): number {
  return countParams(doc, resolveSymbols(doc)).total;
}

/**
 * The fixtures live beside the Go engine's other test data, because both
 * engines import them and neither should be checked against its own copy.
 */
const CONFIGS: Record<string, HfConfig> = JSON.parse(
  readFileSync(join(import.meta.dir, "..", "..", "core-go", "testdata", "hf-configs.json"), "utf8"),
);

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
