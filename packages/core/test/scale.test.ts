/**
 * Scaling tests.
 *
 * Shrinking a design is how the local bench works: take a real architecture,
 * reduce it to something that trains in minutes, and keep its proportions so
 * the comparison still means something.
 */

import { describe, expect, it } from "bun:test";
import { scaleDesign } from "../src/scale.js";
import { countParams } from "../src/analysis/params.js";
import { resolveSymbols } from "../src/ir/symbols.js";
import { validate } from "../src/rules/index.js";
import { generateTorch } from "../src/codegen/torch.js";
import { getPreset } from "../src/presets/index.js";

function paramsOf(doc: Parameters<typeof resolveSymbols>[0]): number {
  return countParams(doc, resolveSymbols(doc)).total;
}

describe("scaling a design", () => {
  it("hits a small target closely", () => {
    const r = scaleDesign(getPreset("llama-3-8b"), { targetParams: 30e6, vocab: 50304 });
    const relative = Math.abs(r.achieved - 30e6) / 30e6;
    expect(relative).toBeLessThan(0.1);
    expect(paramsOf(r.doc)).toBe(r.achieved);
  });

  it("keeps the width a whole number of heads", () => {
    const base = getPreset("llama-3-8b");
    const r = scaleDesign(base, { targetParams: 40e6, vocab: 50304 });
    const s = resolveSymbols(r.doc);
    expect(s.values.D).toBe(s.values.H * s.values.dh);
    expect(s.values.H % s.values.Hkv).toBe(0);
  });

  it("keeps the depth-to-width aspect close to the original", () => {
    const base = getPreset("llama-3-8b");
    const before = resolveSymbols(base).values;
    const r = scaleDesign(base, { targetParams: 100e6, vocab: 50304 });
    const after = resolveSymbols(r.doc).values;
    const beforeAspect = before.D / before.L;
    const afterAspect = after.D / after.L;
    expect(afterAspect).toBeGreaterThan(beforeAspect * 0.5);
    expect(afterAspect).toBeLessThan(beforeAspect * 2);
  });

  it("lets the feed-forward width follow when it is an expression", () => {
    // Llama 3 writes F as ceil_mult(1.3 * 8/3 * D, 1024), so it should track D
    // without being scaled directly.
    const r = scaleDesign(getPreset("llama-3-8b"), { targetParams: 50e6, vocab: 50304 });
    const s = resolveSymbols(r.doc);
    expect(s.values.F).toBe(Math.ceil((1.3 * 8 * s.values.D) / 3 / 1024) * 1024);
  });

  it("produces a design that still passes the design rules and generates code", () => {
    const r = scaleDesign(getPreset("llama-3-8b"), { targetParams: 25e6, vocab: 50304 });
    const report = validate(r.doc, { T: 1024, B: 8, hardware: "rtx5080" });
    expect(report.findings.filter((f) => f.severity === "error")).toEqual([]);
    const code = generateTorch(r.doc);
    expect(code.warnings).toEqual([]);
    expect(code.files[0].contents).toContain("class ");
  });

  it("fits a scaled-down model on a 16 GB consumer GPU", () => {
    const r = scaleDesign(getPreset("llama-3-8b"), { targetParams: 30e6, vocab: 50304 });
    const report = validate(r.doc, { T: 1024, B: 16, hardware: "rtx5080" });
    expect(report.findings.some((f) => f.rule === "training-fits")).toBe(false);
  });

  it("scales a sparse design without breaking its routing", () => {
    const r = scaleDesign(getPreset("mixtral-8x7b"), { targetParams: 200e6, vocab: 50304 });
    const report = validate(r.doc);
    expect(report.findings.filter((f) => f.severity === "error")).toEqual([]);
    const p = countParams(r.doc, resolveSymbols(r.doc));
    expect(p.active).toBeLessThan(p.total);
  });

  it("can hold the depth fixed and move only the width", () => {
    const base = getPreset("llama-3-8b");
    const r = scaleDesign(base, { targetParams: 60e6, vocab: 50304, keepDepth: true });
    expect(resolveSymbols(r.doc).values.L).toBe(resolveSymbols(base).values.L);
  });

  it("narrows the head rather than leaving a two-head model", () => {
    const r = scaleDesign(getPreset("llama-3-8b"), {
      targetParams: 30e6,
      vocab: 50304,
      targetBasis: "non-embedding",
    });
    const s = resolveSymbols(r.doc);
    expect(s.values.H).toBeGreaterThanOrEqual(4);
    expect(s.values.dh).toBe(64);
    expect(r.notes.join(" ")).toMatch(/Narrowed the head dimension/);
  });

  it("records what it changed and drops the original's published figure", () => {
    const r = scaleDesign(getPreset("llama-3-8b"), { targetParams: 30e6 });
    expect(Object.keys(r.changes)).toContain("D");
    expect(Object.keys(r.changes)).toContain("L");
    expect(r.doc.meta.published).toBeUndefined();
    expect(r.doc.meta.name).toContain("llama-3-8b");
  });
});
