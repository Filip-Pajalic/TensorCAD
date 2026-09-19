import { describe, expect, it } from "bun:test";
import { resolveSymbols } from "../src/ir/symbols.js";
import { inferShapes } from "../src/shapes/infer.js";
import { shapeToString } from "../src/shapes/pattern.js";
import { allPresets, getPreset } from "../src/presets/index.js";
import type { Doc } from "../src/ir/types.js";

function infer(doc: Doc, expandComposites = false) {
  const symbols = resolveSymbols(doc);
  return { symbols, result: inferShapes(doc, symbols, { expandComposites }) };
}

function errorsOf(doc: Doc, expandComposites = false): string[] {
  return infer(doc, expandComposites)
    .result.issues.filter((i) => i.severity === "error")
    .map((i) => `${i.path}${i.port ? `:${i.port}` : ""} ${i.message}`);
}

describe("shape inference", () => {
  it("infers every preset cleanly", () => {
    for (const doc of allPresets()) {
      expect({ name: doc.meta.name, errors: errorsOf(doc) }).toEqual({ name: doc.meta.name, errors: [] });
    }
  });

  it("infers every preset cleanly down to the primitives", () => {
    for (const doc of allPresets()) {
      expect({ name: doc.meta.name, errors: errorsOf(doc, true) }).toEqual({
        name: doc.meta.name,
        errors: [],
      });
    }
  });

  it("labels edges with symbol names rather than numbers", () => {
    const { result } = infer(getPreset("llama-3-8b"));
    expect(shapeToString(result.outputs.get("embed:y")!)).toBe("B T D");
    expect(shapeToString(result.outputs.get("layers:x")!)).toBe("B T D");
    expect(shapeToString(result.outputs.get("head:y")!)).toBe("B T V");
  });

  it("carries symbol names into the attention internals", () => {
    const { result } = infer(getPreset("llama-3-8b"), true);
    const p = "layers/block/attn";
    expect(shapeToString(result.outputs.get(`${p}/q_proj:y`)!)).toBe("B T H*dh");
    expect(shapeToString(result.outputs.get(`${p}/k_proj:y`)!)).toBe("B T Hkv*dh");
    expect(shapeToString(result.outputs.get(`${p}/q_heads:y`)!)).toBe("B H T dh");
    expect(shapeToString(result.outputs.get(`${p}/attn:y`)!)).toBe("B H T dh");
    expect(shapeToString(result.outputs.get(`${p}/o_proj:y`)!)).toBe("B T D");
  });

  it("flags a block whose width does not match the residual stream", () => {
    const doc = getPreset("llama-3-8b");
    const stack = doc.graph.nodes.find((n) => n.id === "layers")!;
    const block = stack.graph!.nodes.find((n) => n.id === "block")!;
    block.params!.d_model = "2*D";
    const errors = errorsOf(doc);
    expect(errors.length).toBeGreaterThan(0);
    expect(errors.join("\n")).toMatch(/expects d_model/);
  });

  it("flags heads that are not divisible by kv heads", () => {
    const doc = getPreset("llama-3-8b");
    doc.symbols.Hkv = { kind: "design", value: 7 };
    const errors = errorsOf(doc, true);
    expect(errors.join("\n")).toMatch(/divisible/);
  });

  it("flags an unconnected input port", () => {
    const doc = getPreset("llama-3-8b");
    doc.graph.edges = doc.graph.edges.filter(([, to]) => to !== "head:x");
    const errors = errorsOf(doc);
    expect(errors.join("\n")).toMatch(/"x" is not connected/);
  });

  it("flags a port with two incoming edges", () => {
    const doc = getPreset("llama-3-8b");
    doc.graph.edges.push(["embed:y", "head:x"]);
    const errors = errorsOf(doc);
    expect(errors.join("\n")).toMatch(/more than one incoming edge/);
  });

  it("flags a repeat container that changes its shape", () => {
    const doc = getPreset("llama-3-8b");
    const stack = doc.graph.nodes.find((n) => n.id === "layers")!;
    // Send the block output through a projection that widens the stream.
    stack.graph!.nodes.push({
      id: "widen",
      type: "linear",
      params: { in_features: "D", out_features: "2*D", bias: false },
    });
    stack.graph!.edges = [
      ["_in:x", "block:x"],
      ["block:y", "widen:x"],
      ["widen:y", "_out:x"],
    ];
    const errors = errorsOf(doc);
    expect(errors.join("\n")).toMatch(/must preserve its shape/);
  });

  it("flags a cycle", () => {
    const doc = getPreset("llama-3-8b");
    doc.graph.edges.push(["head:y", "embed:ids"]);
    const errors = errorsOf(doc);
    expect(errors.join("\n")).toMatch(/Cycle detected|more than one incoming/);
  });

  it("flags an odd head dimension under RoPE", () => {
    const doc = getPreset("llama-3-8b");
    doc.symbols.dh = { kind: "design", value: 127 };
    doc.symbols.D = { kind: "design", value: 127 * 32 };
    const errors = errorsOf(doc, true);
    expect(errors.join("\n")).toMatch(/even head_dim/);
  });
});

describe("symbol table", () => {
  it("evaluates dependent symbols in order", () => {
    const { symbols } = infer(getPreset("llama-3-8b"));
    expect(symbols.errors).toEqual([]);
    // F is written as ceil_mult(1.3 * 8/3 * D, 1024).
    expect(symbols.values.F).toBe(14336);
  });

  it("recomputes dependants when a symbol changes", () => {
    const doc = getPreset("llama-3-8b");
    doc.symbols.D = { kind: "design", value: 8192 };
    const symbols = resolveSymbols(doc);
    expect(symbols.values.F).toBe(28672);
  });

  it("detects a symbol cycle", () => {
    const doc = getPreset("llama-3-8b");
    doc.symbols.D = "F";
    doc.symbols.F = "D";
    const symbols = resolveSymbols(doc);
    expect(symbols.errors.join("\n")).toMatch(/cycle/i);
  });

  it("keeps runtime symbols out of the substitution environment", () => {
    const { symbols } = infer(getPreset("llama-3-8b"));
    expect(symbols.runtime.has("B")).toBe(true);
    expect(symbols.runtime.has("T")).toBe(true);
    expect("B" in symbols.designValues).toBe(false);
    expect(symbols.designValues.D).toBe(4096);
  });
});
