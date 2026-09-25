/**
 * The walkthrough, over every design in the library.
 *
 * The claim it makes is that it narrates *this* design rather than a recorded
 * one, so the test is the same claim: build it for every preset in the library and
 * check that nothing it says is made up. That means three things —
 *
 *   - every block a step names is a block that exists, or the canvas would be
 *     asked to light nothing and the step would point at empty sheet;
 *   - no step prints `NaN`, `undefined` or an unresolved `${...}`, which is how
 *     a number that was not there leaks into prose;
 *   - a design with no attention gets no attention step, because a step exists
 *     only when the design has the block it is about.
 */

import { describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();

const { getPreset, PRESET_NAMES } = await import("../src/engine.js");
const { derive } = await import("../src/state/derive.js");
const { DEFAULT_OPERATING, toAnalysisOptions } = await import("../src/state/operating.js");
const { buildWalkthrough } = await import("../src/state/walkthrough.js");
const { useEditor } = await import("../src/state/store.js");

const state = () => useEditor.getState();

import type { Doc } from "@tensor-cad/engine";

const stepsFor = (doc: Doc) => buildWalkthrough(doc, derive(doc, DEFAULT_OPERATING));

/** Every path in the design, the way the walkthrough finds them. */
function pathsOf(doc: Doc, derived: ReturnType<typeof derive>): Set<string> {
  const out = new Set<string>();
  const walk = (nodes: Doc["graph"]["nodes"], prefix: string): void => {
    for (const node of nodes) {
      const path = prefix ? `${prefix}/${node.id}` : node.id;
      out.add(path);
      const interior = node.graph ?? derived.infer.expansions.get(path);
      if (interior) walk(interior.nodes, path);
    }
  };
  walk(doc.graph.nodes, "");
  return out;
}

describe("every preset gets a walkthrough", () => {
  test("with steps, naming blocks that exist, saying nothing empty", () => {
    const bad: string[] = [];

    for (const name of PRESET_NAMES) {
      const doc = getPreset(name);
      const derived = derive(doc, DEFAULT_OPERATING);
      const steps = buildWalkthrough(doc, derived);
      const known = pathsOf(doc, derived);

      if (steps.length < 4) bad.push(`${name}: only ${steps.length} steps`);

      for (const step of steps) {
        for (const path of step.paths) {
          if (!known.has(path)) bad.push(`${name}/${step.id}: no block at ${path}`);
        }
        for (const paragraph of step.body) {
          if (/NaN|undefined|Infinity|\$\{/.test(paragraph)) {
            bad.push(`${name}/${step.id}: ${paragraph.slice(0, 90)}`);
          }
          if (paragraph.trim().length < 20) {
            bad.push(`${name}/${step.id}: a paragraph with nothing in it`);
          }
        }
        if (step.detail < 0 || step.detail > 5) {
          bad.push(`${name}/${step.id}: detail ${step.detail} is not a level`);
        }
      }
    }

    expect(bad).toEqual([]);
  });

  test("and every step is reachable in order", () => {
    for (const name of PRESET_NAMES) {
      const steps = stepsFor(getPreset(name));
      const ids = steps.map((s) => s.id);
      // One step per id: two steps that both call themselves "attention" would
      // make the dots ambiguous and the React keys collide.
      expect({ name, unique: new Set(ids).size }).toEqual({ name, unique: ids.length });
      expect({ name, first: ids[0] }).toEqual({ name, first: "whole" });
      expect({ name, last: ids.at(-1) }).toEqual({ name, last: "cost" });
    }
  });
});

describe("a step exists because the design has the block", () => {
  test("a convnet is not told about attention", () => {
    const ids = stepsFor(getPreset("alexnet")).map((s) => s.id);
    expect(ids).not.toContain("attention");
    expect(ids).not.toContain("positions");
    expect(ids).toContain("conv");
  });

  test("a dense transformer is not told about experts", () => {
    const mlp = stepsFor(getPreset("llama-3-8b")).find((s) => s.id === "mlp");
    expect(mlp?.title).toBe("Then each token thinks on its own");
    expect(mlp?.body.join(" ")).not.toContain("router");
  });

  test("a mixture of experts is", () => {
    const steps = stepsFor(getPreset("mixtral-8x7b"));
    const mlp = steps.find((s) => s.id === "mlp");
    expect(mlp?.body.join(" ")).toContain("router");
    // And the opening step says the active count differs from the total.
    expect(steps[0]?.body.join(" ")).toContain("were not routed to".slice(4));
  });

  test("a hybrid is told about both", () => {
    const ids = stepsFor(getPreset("nemotron-h-8b")).map((s) => s.id);
    expect(ids).toContain("ssm");
  });

  test("a differential design is told what the second map is for", () => {
    const doc = structuredClone(getPreset("gpt2-small"));
    const block = doc.graph.nodes.find((n) => n.id === "layers")!.graph!.nodes.find((n) => n.id === "block")!;
    block.params = { ...block.params, attention: "diff" };
    const attention = stepsFor(doc).find((s) => s.id === "attention");
    expect(attention?.body.join(" ")).toContain("two attention maps over the same values");
  });

  test("a design that writes its attention out is told what that keeps", () => {
    const doc = structuredClone(getPreset("gpt2-small"));
    const block = doc.graph.nodes.find((n) => n.id === "layers")!.graph!.nodes.find((n) => n.id === "block")!;
    block.params = { ...block.params, talking_heads: true };
    const attention = stepsFor(doc).find((s) => s.id === "attention");
    expect(attention?.body.join(" ")).toContain("written out rather than fused");
    expect(attention?.body.join(" ")).toContain("talking heads");
  });

  test("a design with sinks is told what they are for", () => {
    const attention = stepsFor(getPreset("gpt-oss-20b")).find((s) => s.id === "attention");
    expect(attention?.body.join(" ")).toContain("Each head also has a sink");
    const plain = stepsFor(getPreset("llama-3-8b")).find((s) => s.id === "attention");
    expect(plain?.body.join(" ")).not.toContain("sink");
  });

  test("a design whose sense of order is in its scores is told where", () => {
    // BLOOM has no position embedding and no rotation: ALiBi is a score
    // expression on the attention, and that is the step's block.
    const positions = stepsFor(getPreset("bloom-7b1")).find((s) => s.id === "positions");
    expect(positions?.paths).toEqual(["layers/block/attn/attn"]);
    expect(positions?.body.join(" ")).toContain("score - 2 ** (-8 * (h + 1) / heads) * (q - kv)");
  });

  test("an encoder-decoder is told it has two stacks, and where they meet", () => {
    const steps = new Map(stepsFor(getPreset("t5-small")).map((s) => [s.id, s]));
    expect(steps.get("whole")?.body[0]).toBe(
      "60.5M parameters, in an encoder of 6 layers and a decoder of 6, of width 512.",
    );
    expect(steps.get("input")?.paths).toEqual(["source", "target"]);
    expect(steps.get("embed")?.paths).toEqual(["embed", "dec_embed"]);
    const stack = steps.get("stack")!;
    expect(stack.paths).toEqual(["encoder", "decoder"]);
    expect(stack.body[0]).toContain("every token seeing every other");
    expect(stack.body[0]).toContain("each token seeing only those before it");
    // The encoder's attention is not causal, and says so.
    expect(steps.get("attention")?.body[0]).toContain("every other token offers an answer");
    // The source's keys and values: 512 positions, six layers, keys and
    // values of 512 at two bytes.
    const cross = steps.get("cross")!;
    expect(cross.paths).toEqual(["decoder/block/cross"]);
    expect(cross.body.join(" ")).toContain("6.00 MiB for 512 source tokens");
    // A learned table is stored, which ALiBi's sentence would deny.
    const positions = steps.get("positions")!;
    expect(positions.paths).toEqual(["enc_bias", "dec_bias"]);
    expect(positions.body.join(" ")).toContain("512 parameters in all");
    expect(positions.body.join(" ")).not.toContain("Nothing is stored");
    // A decoder-only design gets none of it.
    expect(stepsFor(getPreset("llama-3-8b")).find((s) => s.id === "cross")).toBeUndefined();
  });

  test("a design that keeps packed documents apart is told what that is for, and what it costs", () => {
    const doc = structuredClone(getPreset("llama-3-8b")) as Doc;
    doc.graph.nodes.push({ id: "docs", type: "input", params: { shape: "B T", dtype: "int64", role: "documents" } });
    doc.graph.edges.push(["docs:x", "layers:doc"]);
    const stack = doc.graph.nodes.find((n) => n.id === "layers")!;
    for (const n of stack.graph!.nodes) {
      if (n.id === "_in") (n.params!.ports as Record<string, string>).doc = "B T";
      if (n.id === "block") n.params!.mask = "doc(b, q) == doc(b, kv)";
    }
    stack.graph!.edges.push(["_in:doc", "block:doc"]);

    const one = buildWalkthrough(doc, derive(doc, { ...DEFAULT_OPERATING, T: 8192 }));
    const said = one.find((s) => s.id === "attention")!.body.join(" ");
    expect(said).toContain("keeps packed documents apart");
    expect(said).not.toContain("Packed in documents");

    const packing = { mean: 1024, spread: 0 };
    expect(toAnalysisOptions({ ...DEFAULT_OPERATING, packing }).packing).toEqual(packing);
    expect("packing" in toAnalysisOptions(DEFAULT_OPERATING)).toBe(false);
    const packed = buildWalkthrough(doc, derive(doc, { ...DEFAULT_OPERATING, T: 8192, packing }));
    expect(packed.find((s) => s.id === "attention")!.body.join(" ")).toMatch(
      /Packed in documents of 1,024 tokens, that is .* of attention a token rather than 2\.15 GFLOP, and a block-sparse kernel computes 1\.3\d times that/,
    );
    // A design that attends across documents is told nothing about them.
    expect(stepsFor(getPreset("llama-3-8b")).find((s) => s.id === "attention")!.body.join(" ")).not.toContain("documents");
  });

  test("a context's cache is every one of its tokens' share", () => {
    // It used to quote the part held per sequence regardless, which is
    // nothing at all for a design with no window: 0 B for a full context.
    const cost = stepsFor(getPreset("gpt2-small")).find((s) => s.id === "cost");
    expect(cost?.body[1]).toContain("a full 1,024-token conversation holds 36.00 MiB");
  });
});

describe("it narrates the design in front of you", () => {
  test("a change to the design changes what it says", () => {
    const doc = getPreset("gpt2-small");
    const before = stepsFor(doc).find((s) => s.id === "embed")?.body.join(" ");

    const wider: Doc = {
      ...doc,
      symbols: { ...doc.symbols, D: { kind: "design", value: 1536, doc: "Residual stream width" } },
    };
    const after = stepsFor(wider).find((s) => s.id === "embed")?.body.join(" ");

    expect(before).toContain("768");
    expect(after).toContain("1,536");
    expect(after).not.toBe(before);
  });
});

describe("opening one", () => {

  test("goes back to the top level, because it narrates the whole design", () => {
    // The steps name blocks of the whole design. Starting one while drilled
    // into a container left every step with nothing to light, so the sheet
    // greyed out entirely and no part of it lit — the drawing said the step
    // was about nothing.
    state().setDoc(getPreset("nano-sort"), "test");
    state().setPath(["layers"]);
    expect(state().path).toEqual(["layers"]);

    state().startWalkthrough();
    expect(state().path).toEqual([]);
    expect(state().walkthrough).toBe(0);

    state().endWalkthrough();
    expect(state().walkthrough).toBeNull();
  });

  test("and takes the selection out of the way", () => {
    state().setDoc(getPreset("nano-sort"), "test");
    state().select("embed");
    state().startWalkthrough();
    // A selection left over from before would be a second thing claiming the
    // eye, against a step that is already lighting what it is about.
    expect(state().selection).toBeNull();
    state().endWalkthrough();
  });
});
