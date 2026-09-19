/**
 * The compiled engine, against the one it replaces.
 *
 * The golden files prove the Go source agrees with the TypeScript source. This
 * proves the thing that actually ships agrees too: the same code after a
 * compiler, a linker and a WebAssembly runtime have all had a turn at it, asked
 * the same questions through the boundary the editor will use.
 *
 * Needs the module built first: `bun run scripts/build-wasm.ts`.
 */

import { beforeAll, describe, expect, it } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { analyze, generateTorch, getPreset, PRESET_NAMES, scaleDesign, validate } from "@tensorcad/core";
import { createEngine, EngineError, type Doc as EngineDoc, type Engine } from "../src/index.js";
import "../vendor/wasm_exec.js";

const wasmPath = join(import.meta.dir, "..", "wasm", "tensorcad.wasm");

/**
 * The same document, described by two type worlds.
 *
 * `@tensorcad/core` and `@tensorcad/engine` each declare the IR, which is the
 * duplication this whole migration exists to remove. Until the TypeScript core
 * goes, comparing the two engines means handing one's document to the other,
 * and the cast is where that happens — once, here, rather than at every call.
 */
function asDoc(doc: ReturnType<typeof getPreset>): EngineDoc {
  return doc as unknown as EngineDoc;
}

let engine: Engine;

beforeAll(async () => {
  if (!existsSync(wasmPath)) {
    throw new Error("The engine is not built. Run: bun run scripts/build-wasm.ts");
  }
  engine = await createEngine({ wasm: readFileSync(wasmPath) });
});

describe("the compiled engine", () => {
  it("says what it is", () => {
    expect(engine.version()).toEqual({ engine: "go", target: "wasm" });
  });

  it("ships the same design library", () => {
    expect(engine.presets().sort()).toEqual([...PRESET_NAMES].sort());
  });

  it("hands back the same documents", () => {
    for (const name of PRESET_NAMES) {
      const fromWasm = engine.preset(name);
      const fromCore = getPreset(name);
      expect(fromWasm.meta.name).toBe(fromCore.meta.name);
      expect(fromWasm.graph.nodes.length).toBe(fromCore.graph.nodes.length);
      expect(fromWasm.graph.edges).toEqual(fromCore.graph.edges);
    }
  });

  // The headline numbers, for every preset. A parameter count is a whole
  // number of weights, so it is exact or it is wrong.
  it("counts the same parameters", () => {
    for (const name of PRESET_NAMES) {
      const doc = getPreset(name);
      expect({ name, ...pick(engine.analyze(asDoc(doc))) }).toEqual({ name, ...pick(analyze(doc)) });
    }
  });

  it("agrees at an operating point nobody defaults to", () => {
    const options = {
      T: 8192,
      B: 4,
      dtype: "fp8",
      inferenceDtype: "fp8",
      recompute: "full",
      optimizer: "adamw8bit",
      hardware: "a100-80",
      gpus: 64,
      concurrency: 32,
      tokens: 15e12,
      mfu: 0.4,
      flash: false,
      parallel: { dp: 4, tp: 8, pp: 2, zero: 3, sequenceParallel: true },
    } as const;
    for (const name of ["llama-3-8b", "mixtral-8x7b", "deepseek-v3", "nemotron-h-8b"]) {
      const doc = getPreset(name);
      const fromWasm = engine.analyze(asDoc(doc), options);
      const fromCore = analyze(doc, options);
      expect({ name, ...pick(fromWasm) }).toEqual({ name, ...pick(fromCore) });
      expect(fromWasm.memory.train.perGpu.total).toBeCloseTo(fromCore.memory.train.perGpu.total, 6);
      expect(fromWasm.memory.notes).toEqual(fromCore.memory.notes);
      expect(fromWasm.throughput.notes).toEqual(fromCore.throughput.notes);
    }
  });

  it("finds the same things wrong", () => {
    for (const name of PRESET_NAMES) {
      const doc = getPreset(name);
      const fromWasm = engine.validate(asDoc(doc));
      const fromCore = validate(doc);
      expect({ name, ok: fromWasm.ok, counts: fromWasm.counts }).toEqual({
        name,
        ok: fromCore.ok,
        counts: fromCore.counts,
      });
      expect(fromWasm.findings.map(line)).toEqual(fromCore.findings.map(line));
    }
  });

  // The strongest one: a model.py is the whole engine's output as a single
  // artifact, and a file that differs by one character was generated
  // differently.
  it("generates the same PyTorch, byte for byte", () => {
    for (const name of PRESET_NAMES) {
      const doc = getPreset(name);
      for (const options of [{}, { moeDispatch: "dense" as const }]) {
        const fromWasm = engine.generateTorch(asDoc(doc), options);
        const fromCore = generateTorch(doc, options);
        expect(fromWasm.warnings).toEqual(fromCore.warnings);
        expect({ name, model: modelOf(fromWasm) }).toEqual({ name, model: modelOf(fromCore) });
      }
    }
  });

  it("shrinks a design to the same widths", () => {
    const doc = getPreset("llama-3-8b");
    const options = { targetParams: 30e6, targetBasis: "non-embedding" as const, vocab: 8192, tieHead: true };
    const fromWasm = engine.scale(asDoc(doc), options);
    const fromCore = scaleDesign(doc, options);
    expect(fromWasm.achieved).toBe(fromCore.achieved);
    expect(fromWasm.changes).toEqual(fromCore.changes);
    expect(fromWasm.notes).toEqual(fromCore.notes);
    expect(fromWasm.doc.meta.name).toBe(fromCore.doc.meta.name);
  });

  it("explains a block the same way", () => {
    const doc = getPreset("gpt2-small");
    const e = engine.explain(asDoc(doc), "layers/block/attn");
    expect(e.type).toBe("gqa_attention");
    expect(e.docs.summary).toBeTruthy();
    expect(e.contributes.params).toBeGreaterThan(0);
    // The parameters come back in the order the block declares them, which is
    // the order the inspector lays its fields out in.
    expect(e.paramOrder[0]).toBe("d_model");
    expect(e.params.d_model.expression).toBe("D");
    expect(e.params.d_model.value).toBe(768);
  });

  it("carries the whole catalog", () => {
    const blocks = engine.catalog();
    expect(blocks.length).toBeGreaterThanOrEqual(30);
    const attention = blocks.find((b) => b.type === "gqa_attention");
    expect(attention?.kind).toBe("composite");
    expect(attention?.params[0]?.name).toBe("d_model");
    expect(attention?.summary).toBeTruthy();
    const linear = blocks.find((b) => b.type === "linear");
    expect(linear?.ports.in.x).toBe("... in_features");
  });

  it("imports a config the same way", () => {
    const configs = JSON.parse(
      readFileSync(join(import.meta.dir, "..", "..", "core-go", "testdata", "hf-configs.json"), "utf8"),
    ) as Record<string, unknown>;
    for (const [name, config] of Object.entries(configs)) {
      const { doc, warnings } = engine.importHuggingFace(JSON.stringify(config), name);
      expect({ name, warnings }).toEqual({ name, warnings: [] });
      expect(analyze(doc as never).params.total).toBe(analyze(getPreset(name)).params.total);
    }
  });

  it("lists the hardware the analysis knows", () => {
    const profiles = engine.hardware();
    expect(profiles.map((h) => h.id)).toContain("h100-sxm");
    expect(profiles.every((h) => h.memory > 0 && h.bandwidth > 0)).toBe(true);
  });
});

describe("refusals", () => {
  it("names what is wrong rather than returning a default", () => {
    expect(() => engine.analyze({ version: 0 } as never)).toThrow(EngineError);
    expect(() => engine.preset("no-such-model")).toThrow(/unknown preset/);
    expect(() => engine.analyze(asDoc(getPreset("gpt2-small")), { hardware: "made-up" })).toThrow(
      /hardware profile/,
    );
    expect(() => engine.scale(asDoc(getPreset("gpt2-small")), { targetParams: 0 })).toThrow(/positive/);
  });

  it("survives a design that is nonsense", () => {
    // The engine must come back with a message rather than take the runtime
    // with it: a crashed WebAssembly instance cannot be restarted without
    // reloading the window.
    const broken = {
      version: 1,
      meta: { name: "nonsense" },
      symbols: { D: { kind: "design", value: "D + 1" } },
      graph: { nodes: [{ id: "a", type: "no_such_block" }], edges: [] },
    };
    const report = engine.validate(broken as never);
    expect(report.ok).toBe(false);
    expect(report.findings.length).toBeGreaterThan(0);
    // And it still works afterwards.
    expect(engine.analyze(asDoc(getPreset("gpt2-small"))).params.total).toBe(124439808);
  });
});

/** The numbers worth comparing exactly, for a whole preset. */
function pick(a: { params: { total: number; active: number }; flops: { fwdTotal: number }; kv: { bytesPerToken: number } }) {
  return {
    params: a.params.total,
    active: a.params.active,
    flops: a.flops.fwdTotal,
    kv: a.kv.bytesPerToken,
  };
}

function line(f: { severity: string; rule: string; path?: string; message: string; hint?: string }): string {
  return `${f.severity} ${f.rule} ${f.path ?? ""}: ${f.message}${f.hint ? ` — ${f.hint}` : ""}`;
}

function modelOf(g: { files: { path: string; contents: string }[] }): string {
  return g.files.find((file) => file.path === "model.py")?.contents ?? "";
}
