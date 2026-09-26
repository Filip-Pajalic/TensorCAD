/**
 * The compiled engine, against the answers the goldens record.
 *
 * The Go tests prove the source. This proves the thing that actually ships:
 * the same code after a compiler, a linker and a WebAssembly runtime have each
 * had a turn at it, asked the same questions through the boundary the editor
 * uses. The two can disagree, and have — a nil slice arrives as `null`, a NaN
 * cannot be encoded at all, and Go and JavaScript print the same double
 * differently. None of that is visible from inside Go.
 *
 * The comparison is against `packages/core-go/testdata` rather than a second
 * implementation, so every expectation here is a file a person can read.
 *
 * Needs the module built first: `bun run build:wasm`.
 */

import { beforeAll, describe, expect, it } from "bun:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import {
  createEngine,
  EngineError,
  type AnalysisOptions,
  type ClusterRequest,
  type Doc,
  type Engine,
  type MupOptions,
  type MupRung,
  type Severity,
  type ScaleOptions,
  type TorchOptions,
} from "../src/index.js";
import "../vendor/wasm_exec.js";

const wasmPath = join(import.meta.dir, "..", "wasm", "tensorcad.wasm");
const testdata = join(import.meta.dir, "..", "..", "core-go", "testdata");

function golden<T>(...parts: string[]): T {
  return JSON.parse(readFileSync(join(testdata, ...parts), "utf8")) as T;
}

/** A rung with its numbers printed the way the golden holds them. */
function shapeOfRung(r: MupRung) {
  return {
    width: r.width,
    multiplier: String(r.multiplier),
    heads: r.heads,
    params: r.params,
    base: r.base,
    notes: r.notes,
    scaling: r.scaling.map((s) => [s.class, String(s.initStd), String(s.adamLr), s.paths]),
  };
}

/** The presets, taken from the files rather than from the engine under test. */
const PRESET_NAMES = readdirSync(join(testdata, "golden"))
  .filter((f) => f.endsWith(".json"))
  .map((f) => f.slice(0, -".json".length))
  .sort();

interface Cases<T> {
  preset: string;
  cases: T[];
}

let engine: Engine;

beforeAll(async () => {
  if (!existsSync(wasmPath)) {
    throw new Error("The engine is not built. Run: bun run build:wasm");
  }
  engine = await createEngine({ wasm: readFileSync(wasmPath) });
});

describe("the compiled engine", () => {
  it("says what it is", () => {
    expect(engine.version()).toEqual({ engine: "go", target: "wasm" });
  });

  it("ships the whole design library", () => {
    expect(engine.presets().sort()).toEqual(PRESET_NAMES);
    expect(PRESET_NAMES.length).toBe(28);
  });

  it("hands back documents that still carry their published figures", () => {
    for (const name of PRESET_NAMES) {
      const doc = engine.preset(name);
      expect({ name, has: doc.meta.name }).toEqual({ name, has: name });
      expect(doc.graph.nodes.length).toBeGreaterThan(0);
    }
  });

  // The strongest of the numeric checks: not a handful of headline figures but
  // every number the analysis produces, for every preset, at three operating
  // points that between them turn on sharding, recomputation, fp8, a dtype for
  // the cache that differs from the weights, and eager attention.
  it("reproduces every number in the analysis goldens", () => {
    for (const name of PRESET_NAMES) {
      const file = golden<Cases<{ label: string; options: AnalysisOptions }>>("analysis", `${name}.json`);
      for (const one of file.cases) {
        // `flat` is the flattening the analysis walked, recorded beside the
        // answer rather than part of it; the Go tests check it on its own.
        const { label, options, flat: _flat, ...want } = one as Record<string, unknown> & {
          label: string;
          options: AnalysisOptions;
        };
        const got = engine.analyze(engine.preset(name), options);
        expect({ name, label, ...(asGolden(got, want) as object) }).toEqual({ name, label, ...want });
      }
    }
  });

  it("finds the same things wrong, in the same order", () => {
    for (const name of PRESET_NAMES) {
      const file = golden<Cases<{ label: string; ok: boolean; counts: Record<Severity, number>; findings: Finding[] }>>(
        "rules",
        `${name}.json`,
      );
      const options = golden<Cases<{ label: string; options: AnalysisOptions }>>("analysis", `${name}.json`);
      for (const one of file.cases) {
        const at = options.cases.find((c) => c.label === one.label);
        const report = engine.validate(engine.preset(name), at?.options);
        expect({ name, label: one.label, ok: report.ok, counts: report.counts }).toEqual({
          name,
          label: one.label,
          ok: one.ok,
          counts: one.counts,
        });
        expect({ name, label: one.label, findings: report.findings.map(line) }).toEqual({
          name,
          label: one.label,
          findings: one.findings.map(line),
        });
      }
    }
  });

  // A model.py is the whole engine's output as a single artifact, and a file
  // that differs by one character was generated differently.
  it("generates the same PyTorch, byte for byte", () => {
    for (const name of PRESET_NAMES) {
      const file = golden<Cases<{ label: string; options: TorchOptions; warnings: string[]; model: string }>>(
        "codegen",
        `${name}.json`,
      );
      for (const one of file.cases) {
        const got = engine.generateTorch(engine.preset(name), one.options);
        expect({ name, label: one.label, warnings: got.warnings }).toEqual({
          name,
          label: one.label,
          warnings: one.warnings,
        });
        const model = got.files.find((f) => f.path === "model.py")?.contents ?? "";
        expect({ name, label: one.label, model }).toEqual({ name, label: one.label, model: one.model });
      }
    }
  });

  it("shrinks a design to the same widths", () => {
    const file = golden<{
      cases: {
        label: string;
        preset: string;
        options: ScaleOptions;
        achieved: number;
        changes: [string, number, number][];
        notes: string[];
        name: string;
      }[];
    }>("scale.json");
    expect(file.cases.length).toBe(6);
    for (const one of file.cases) {
      const got = engine.scale(engine.preset(one.preset), one.options);
      expect({ label: one.label, achieved: got.achieved, notes: got.notes, name: got.doc.meta.name }).toEqual({
        label: one.label,
        achieved: one.achieved,
        notes: one.notes,
        name: one.name,
      });
      const changes = Object.entries(got.changes)
        .map(([k, c]) => [k, c.from, c.to])
        .sort();
      expect({ label: one.label, changes }).toEqual({
        label: one.label,
        changes: one.changes.map((c) => [...c]).sort(),
      });
    }
  });

  // The ladder's multipliers are irrational as often as not — 1/sqrt(2) is the
  // usual one — so this is where Go's %g and ECMAScript's Number::toString have
  // to agree digit for digit. The golden holds them as the strings the engine
  // printed, and a double that crossed differently shows up immediately.
  it("builds the same width ladder", () => {
    const file = golden<{
      cases: {
        label: string;
        preset: string;
        options: MupOptions;
        widthSymbol: string;
        baseWidth: number;
        headDim: number;
        rungs: {
          width: number;
          multiplier: string;
          heads: number;
          params: number;
          base: boolean;
          scaling: { class: string; initStd: string; adamLr: string; paths: string[] }[];
          notes: string[];
        }[];
        notes: string[];
      }[];
    }>("mup.json");
    expect(file.cases.length).toBe(6);
    for (const one of file.cases) {
      const got = engine.mup(engine.preset(one.preset), one.options);
      expect({
        label: one.label,
        widthSymbol: got.widthSymbol,
        baseWidth: got.baseWidth,
        headDim: got.headDim,
        notes: got.notes,
      }).toEqual({
        label: one.label,
        widthSymbol: one.widthSymbol,
        baseWidth: one.baseWidth,
        headDim: one.headDim,
        notes: one.notes,
      });
      expect({ label: one.label, rungs: got.rungs.length }).toEqual({
        label: one.label,
        rungs: one.rungs.length,
      });
      for (const [i, want] of one.rungs.entries()) {
        const rung = got.rungs[i]!;
        expect({ label: one.label, i, rung: shapeOfRung(rung) }).toEqual({
          label: one.label,
          i,
          rung: {
            width: want.width,
            multiplier: want.multiplier,
            heads: want.heads,
            params: want.params,
            base: want.base,
            notes: want.notes,
            scaling: want.scaling.map((s) => [s.class, s.initStd, s.adamLr, s.paths]),
          },
        });
        // A rung is a design, not a report about one: it has to come back as
        // something the rest of the engine will accept.
        expect(engine.analyze(rung.doc).params.total).toBe(want.params);
        // And a class with nothing in it is an empty list, not a null.
        for (const s of rung.scaling) expect(Array.isArray(s.paths)).toBe(true);
      }
    }
  });

  // Memory is arithmetic, so the planner's numbers have to survive the
  // boundary; the ranking is advice, so the order has to as well.
  it("plans a cluster the same way", () => {
    const file = golden<{
      cases: {
        label: string;
        preset: string;
        seq: number;
        cluster: ClusterRequest;
        budget: number;
        considered: number;
        fits: { summary: string; used: number; perGpu: { total: number }; notes: string[] }[];
        closest?: { summary: string };
        notes: string[];
      }[];
    }>("plans.json");
    expect(file.cases.length).toBeGreaterThan(0);
    for (const one of file.cases) {
      const got = engine.plan(engine.preset(one.preset), { T: one.seq, hardware: "h100-sxm" }, one.cluster);
      expect({ label: one.label, considered: got.considered, budget: got.budget }).toEqual({
        label: one.label,
        considered: one.considered,
        budget: one.budget,
      });
      expect({ label: one.label, order: got.fits.map((f) => f.summary) }).toEqual({
        label: one.label,
        order: one.fits.map((f) => f.summary),
      });
      expect({ label: one.label, held: got.fits.map((f) => f.perGpu.total) }).toEqual({
        label: one.label,
        held: one.fits.map((f) => f.perGpu.total),
      });
      expect({ label: one.label, closest: got.closest?.summary ?? null }).toEqual({
        label: one.label,
        closest: one.closest?.summary ?? null,
      });
      expect({ label: one.label, notes: got.notes }).toEqual({ label: one.label, notes: one.notes });
      // Nothing empty came across as null: a plan with no notes is a plan with
      // an empty list of them.
      for (const f of got.fits) expect(Array.isArray(f.notes)).toBe(true);
    }
  });

  // The editor's own call: one walk, both answers. The shapes come back as
  // text in both forms, because only the engine holds the polynomial and the
  // canvas needs to be able to write either one on a wire.
  it("derives the findings and the shapes together", () => {
    for (const name of PRESET_NAMES) {
      const shapes = golden<{ inferExpanded: { outputs: [string, string][] } }>("golden", `${name}.json`);
      const rules = golden<Cases<{ label: string; ok: boolean; findings: Finding[] }>>("rules", `${name}.json`);
      const want = rules.cases.find((c) => c.label === "default")!;

      const derived = engine.derive(engine.preset(name));
      expect({ name, ok: derived.report.ok }).toEqual({ name, ok: want.ok });
      expect({ name, findings: derived.report.findings.map(line) }).toEqual({
        name,
        findings: want.findings.map(line),
      });

      const got = Object.entries(derived.infer.outputs)
        .map(([k, v]) => [k, v.symbolic])
        .sort();
      expect({ name, shapes: got }).toEqual({
        name,
        shapes: shapes.inferExpanded.outputs.map((o) => [...o]).sort(),
      });
    }
  });

  it("writes a shape in both the forms the canvas uses", () => {
    const derived = engine.derive(engine.preset("gpt2-small"));
    const stream = derived.infer.outputs["embed:y"];
    // The runtime symbols stay symbols in both; only the design ones resolve.
    expect(stream.symbolic).toBe("B T D");
    expect(stream.numeric).toBe("B T 768");
  });

  it("infers shapes on their own, for the wire the pointer is over", () => {
    for (const name of PRESET_NAMES) {
      const want = golden<{ infer: { outputs: [string, string][]; issues: unknown[] } }>(
        "golden",
        `${name}.json`,
      );
      const flat = engine.infer(engine.preset(name));
      const got = Object.entries(flat.outputs)
        .map(([k, v]) => [k, v.symbolic])
        .sort();
      expect({ name, shapes: got }).toEqual({ name, shapes: want.infer.outputs.map((o) => [...o]).sort() });
      expect({ name, issues: flat.issues.length }).toEqual({ name, issues: want.infer.issues.length });
    }
  });

  it("carries a parameter as the symbol it was written as, not just its value", () => {
    const flat = engine.infer(engine.preset("gpt2-small"));
    expect(flat.resolved["head"].p.vocab).toBe(50257);
    expect(flat.resolved["head"].s.vocab).toBe("V");
    expect(flat.ports["embed"].out.y.shape).toBe("... dim");
  });

  it("measures a second sequence through the boundary, and says nothing of one a design lacks", () => {
    // An encoder over a source and a decoder over a target, nothing between
    // them yet: M11's first phase.
    const doc = {
      version: 1,
      meta: { name: "two-streams" },
      symbols: {
        B: { kind: "runtime", default: 1 },
        T: { kind: "runtime", default: 64 },
        S: { kind: "runtime", default: 256 },
        D: { kind: "design", value: 64 },
        H: { kind: "design", value: 4 },
        dh: { kind: "design", value: 16 },
        V: { kind: "design", value: 100 },
      },
      graph: {
        nodes: [
          { id: "src", type: "input", params: { shape: "B S", dtype: "int64" } },
          { id: "src_embed", type: "embedding", params: { vocab: "V", dim: "D" } },
          { id: "encoder", type: "transformer_block", params: { d_model: "D", heads: "H", kv_heads: "H", head_dim: "dh", ffn_hidden: "4*D", causal: false } },
          { id: "encoded", type: "output" },
          { id: "tgt", type: "input", params: { shape: "B T", dtype: "int64" } },
          { id: "tgt_embed", type: "embedding", params: { vocab: "V", dim: "D" } },
          { id: "decoder", type: "transformer_block", params: { d_model: "D", heads: "H", kv_heads: "H", head_dim: "dh", ffn_hidden: "4*D" } },
          { id: "head", type: "lm_head", params: { vocab: "V", dim: "D", tied: false } },
          { id: "logits", type: "output" },
        ],
        edges: [
          ["src:x", "src_embed:ids"], ["src_embed:y", "encoder:x"], ["encoder:y", "encoded:x"],
          ["tgt:x", "tgt_embed:ids"], ["tgt_embed:y", "decoder:x"], ["decoder:y", "head:x"], ["head:y", "logits:x"],
        ],
      },
    } as unknown as Doc;
    const a = engine.analyze(doc, { S: 512 });
    expect(a.errors).toEqual([]);
    expect(a.options.S).toBe(512);
    expect(a.flops.perStream!.map((s) => [s.symbol, s.length])).toEqual([["S", 512], ["T", 64]]);
    const [src, tgt] = a.flops.perStream!;
    expect(a.flops.fwdPerExample).toBe(src.fwd * 512 + tgt.fwd * 64);
    // The encoder's layers ran along S, so nothing inside it was told otherwise.
    expect(engine.derive(doc, {}).report.findings.filter((f) => f.severity === "error")).toEqual([]);

    // One sequence: no S, no streams, not even a null.
    const one = engine.analyze(engine.preset("nano-sort"), { S: 512 });
    expect("S" in one.options).toBe(false);
    expect("perStream" in one.flops).toBe(false);
    expect("fwdPerExample" in one.flops).toBe(false);
  });

  it("takes a packing through the boundary, and moves only training with it", () => {
    // llama-3-8b with every layer keeping documents apart, as Llama 3 trained.
    const doc = structuredClone(engine.preset("llama-3-8b")) as Doc;
    doc.graph.nodes.push({ id: "docs", type: "input", params: { shape: "B T", dtype: "int64", role: "documents" } });
    doc.graph.edges.push(["docs:x", "layers:doc"]);
    const stack = doc.graph.nodes.find((n) => n.id === "layers")!;
    for (const n of stack.graph!.nodes) {
      if (n.id === "_in") (n.params!.ports as Record<string, string>).doc = "B T";
      if (n.id === "block") n.params!.mask = "doc(b, q) == doc(b, kv)";
    }
    stack.graph!.edges.push(["_in:doc", "block:doc"]);

    const one = engine.analyze(doc, { T: 8192 });
    expect(one.errors).toEqual([]);
    expect("packed" in one.flops).toBe(false);
    expect("packing" in one.options).toBe(false);

    const packed = engine.analyze(doc, { T: 8192, packing: { mean: 1024 } });
    expect(packed.options.packing).toEqual({ mean: 1024, spread: 0 });
    // 491 keys a query rather than 4,096, and serving untouched.
    expect((4096 * packed.flops.packed!.fwdAttention) / one.flops.fwdAttention).toBeCloseTo(491.1, -0.5);
    expect(packed.flops.fwdTotal).toBe(one.flops.fwdTotal);
    // And what the kernel computes, in whole blocks.
    expect(packed.flops.packed!.fwdAttentionBlocks / packed.flops.packed!.fwdAttention).toBeCloseTo(1.37, 1);
    expect(packed.cost.gpuHours / one.cost.gpuHours).toBeCloseTo(
      packed.flops.packed!.trainPerToken / one.flops.trainPerToken,
      12,
    );
    // Nothing is wrong with the wiring, and the same wire from the tokens is.
    expect(engine.validate(doc, {}).findings.filter((f) => f.rule === "documents")).toEqual([]);

    // A design that attends across documents is not moved by a packing.
    const plain = engine.analyze(engine.preset("llama-3-8b"), { T: 8192, packing: { mean: 1024 } });
    expect("packed" in plain.flops).toBe(false);
    // And a packing that means nothing is refused, not ignored.
    expect(() => engine.analyze(doc, { packing: { mean: 0 } })).toThrow(/packing/);
  });

  it("draws an attention's mask through the boundary", () => {
    const doc = engine.preset("nano-sort");
    // A block with attention inside it answers through that attention, at the
    // design's own length: eleven positions, one a block.
    const causal = engine.attentionMask(doc, "layers/block");
    expect({
      attention: causal.attention,
      found: causal.found,
      cells: causal.cells,
      kept: causal.kept.length,
      density: causal.density,
      mask: causal.mask,
      score: causal.score,
    }).toEqual({
      attention: "layers/block/attn/attn",
      found: true,
      cells: 11,
      kept: 121,
      density: 0.5,
      mask: "kv <= q",
      score: "",
    });
    // The diagonal is kept, the block above it is not.
    expect([causal.kept[0], causal.kept[1], causal.kept[11]]).toEqual([1, 0, 1]);
    // At the operating point's length rather than the design's.
    expect(engine.attentionMask(doc, "layers/block", { T: 4096 }).cells).toBe(32);

    // A block with no attention says so with an empty grid, not a null one.
    const none = engine.attentionMask(doc, "embed");
    expect({ found: none.found, kept: none.kept, attention: none.attention }).toEqual({
      found: false,
      kept: [],
      attention: "",
    });
  });

  it("explains a block the same way", () => {
    const file = golden<{ cases: { preset: string; blocks: ExplainedBlock[] }[] }>("explain.json");
    for (const one of file.cases) {
      const doc = engine.preset(one.preset);
      for (const want of one.blocks) {
        const got = engine.explain(doc, want.path);
        const where = { preset: one.preset, path: want.path };
        expect({ ...where, type: got.type, kind: got.kind }).toEqual({
          ...where,
          type: want.type,
          kind: want.kind,
        });
        expect({ ...where, contributes: got.contributes.params }).toEqual({
          ...where,
          contributes: want.contributes.params,
        });
        // The parameters travel in the order the block declares them, which is
        // the order the inspector lays its fields out in, and each arrives
        // twice: as written, and as evaluated.
        expect({ ...where, order: got.paramOrder }).toEqual({
          ...where,
          order: want.params.map((p) => p[0]),
        });
        for (const [key, expression, value] of want.params) {
          expect({ ...where, key, e: got.params[key].expression ?? null, v: got.params[key].value ?? null }).toEqual(
            { ...where, key, e: expression, v: value },
          );
        }
      }
    }
  });

  it("carries the whole catalog, prose and all", () => {
    const want = golden<{ blocks: CatalogProse[] }>("catalog-docs.json");
    const got = engine.catalog();
    expect(got.length).toBe(want.blocks.length);
    const byType = new Map(got.map((b) => [b.type, b]));
    for (const block of want.blocks) {
      const mine = byType.get(block.type);
      expect({ type: block.type, found: mine !== undefined }).toEqual({ type: block.type, found: true });
      expect({
        type: block.type,
        kind: mine!.kind,
        category: mine!.category,
        summary: mine!.docs.summary ?? "",
        formula: mine!.docs.formula ?? "",
        refs: mine!.docs.refs ?? [],
      }).toEqual({
        type: block.type,
        kind: block.kind,
        category: block.category,
        summary: block.summary,
        formula: block.formula,
        refs: block.refs,
      });
      // A Go map has no order, so the order arrives beside the parameters.
      expect({ type: block.type, order: mine!.paramOrder }).toEqual({
        type: block.type,
        order: block.params.map((p) => p[0]),
      });
      expect(Object.keys(mine!.params).length).toBe(mine!.paramOrder.length);
    }
  });

  it("carries what each parameter is called, and which are rare", () => {
    // What the inspector leads with, across the boundary: a label on every
    // built-in parameter, `advanced` only where it is true, and an enum's
    // words keyed by its values. Go proves the table; this proves it arrives.
    const unlabelled = engine
      .catalog()
      .flatMap((b) => b.paramOrder.filter((name) => !b.params[name].label).map((name) => `${b.type}.${name}`));
    expect(unlabelled).toEqual([]);

    const block = engine.catalog().find((b) => b.type === "transformer_block")!;
    expect(block.params.ffn_hidden.label).toBe("Feed-forward width");
    expect(block.params.ffn_hidden.advanced).toBeUndefined();
    expect(block.params.sinks.advanced).toBe(true);
    expect(block.params.attention.valueLabels).toEqual({
      gqa: "grouped-query",
      mla: "latent (MLA)",
      diff: "differential",
    });
    const stack = engine.catalog().find((b) => b.type === "repeat")!;
    expect(stack.params.count.label).toBe("Repeats");

    // And explain, so an agent reading a block gets the same words.
    const explained = engine.explain(engine.preset("gpt2-small"), "layers");
    expect(explained.params.count.label).toBe("Repeats");
  });

  it("imports a config into the design it came from", () => {
    const configs = golden<Record<string, unknown>>("hf-configs.json");
    for (const [name, config] of Object.entries(configs)) {
      const { doc, warnings } = engine.importHuggingFace(JSON.stringify(config), name);
      expect({ name, warnings }).toEqual({ name, warnings: [] });
      const want = golden<Cases<{ label: string; params: { total: number } }>>("analysis", `${name}.json`);
      const at = want.cases.find((c) => c.label === "default")!;
      expect({ name, params: engine.analyze(doc).params.total }).toEqual({
        name,
        params: at.params.total,
      });
    }
  });

  it("lists the hardware the analysis knows", () => {
    const profiles = engine.hardware();
    expect(profiles.map((h) => h.id)).toContain("h100-sxm");
    expect(profiles.every((h) => h.memory > 0 && h.bandwidth > 0)).toBe(true);
  });
});

describe("designs that are wrong", () => {
  // The findings on documents built to break: an unknown block, a dangling
  // edge, a shape that cannot match, a symbol that refers to itself.
  it("reports what the goldens say it reports", () => {
    const file = golden<{
      cases: { name: string; doc: Doc; ok: boolean; counts: Record<Severity, number>; findings: Finding[] }[];
    }>("broken.json");
    expect(file.cases.length).toBeGreaterThan(0);
    for (const one of file.cases) {
      const report = engine.validate(one.doc);
      expect({ name: one.name, ok: report.ok, counts: report.counts }).toEqual({
        name: one.name,
        ok: one.ok,
        counts: one.counts,
      });
      expect({ name: one.name, findings: report.findings.map(line) }).toEqual({
        name: one.name,
        findings: one.findings.map(line),
      });
    }
  });
});

describe("refusals", () => {
  it("names what is wrong rather than returning a default", () => {
    expect(() => engine.analyze({ version: 0 } as never)).toThrow(EngineError);
    expect(() => engine.preset("no-such-model")).toThrow(/unknown preset/);
    expect(() => engine.analyze(engine.preset("gpt2-small"), { hardware: "made-up" })).toThrow(
      /hardware profile/,
    );
    expect(() => engine.scale(engine.preset("gpt2-small"), { targetParams: 0 })).toThrow(/positive/);
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
    expect(engine.analyze(engine.preset("gpt2-small")).params.total).toBe(124439808);
  });
});

/**
 * The engine's answer, shaped the way the goldens write it down.
 *
 * A golden is a rendering of a result, not a copy of one: a map arrives as
 * sorted [key, value] pairs, because the generator these files descend from
 * held `byPath` and its neighbours as JavaScript Maps, which JSON cannot
 * encode. The shape to convert to is read from the golden itself rather than
 * listed here, so a new map-valued field needs no change.
 *
 * The comparison is one-sided on purpose. Every field the golden records must
 * be there and must match, so an engine that drops one or moves a number
 * fails; a field the engine has gained since is ignored, because the boundary
 * emits an empty `errors` beside several sections that the generator hoists to
 * the top of the file instead. A golden that quietly loses a field is the
 * direction this cannot see, and that one is covered by the Go tests reading
 * the same files and by the generator only ever running deliberately.
 */
function asGolden(got: unknown, want: unknown): unknown {
  if (Array.isArray(want) && want.every(isPair)) {
    if (got && typeof got === "object" && !Array.isArray(got)) {
      return Object.entries(got).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    }
    return got;
  }
  if (want && typeof want === "object" && !Array.isArray(want)) {
    if (!got || typeof got !== "object") return got;
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(want)) {
      out[key] = asGolden((got as Record<string, unknown>)[key], (want as Record<string, unknown>)[key]);
    }
    return out;
  }
  return got;
}

function isPair(v: unknown): boolean {
  return Array.isArray(v) && v.length === 2 && typeof v[0] === "string";
}

interface Finding {
  severity: string;
  rule: string;
  path?: string;
  message: string;
  hint?: string;
}

/** One block as `explain.json` records it: the parameters as written, then as
 *  evaluated, then what the block contributes. */
interface ExplainedBlock {
  path: string;
  type: string;
  kind: BlockKind;
  params: [string, string | null, number | null, string][];
  contributes: { params: number };
}

/** One block's prose, as `catalog-docs.json` records it. */
interface CatalogProse {
  type: string;
  kind: BlockKind;
  category: string;
  summary: string;
  formula: string;
  refs: string[];
  params: [string, string, string, string[]?][];
}

/** The catalog's three kinds, as both the engine and the goldens spell them. */
type BlockKind = "primitive" | "composite" | "container";

function line(f: Finding): string {
  return `${f.severity} ${f.rule} ${f.path ?? ""}: ${f.message}${f.hint ? ` — ${f.hint}` : ""}`;
}
