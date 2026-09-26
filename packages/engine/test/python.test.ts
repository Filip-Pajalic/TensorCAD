/**
 * Cross-check the analysis against PyTorch itself.
 *
 * These tests shell out to `tensorcad-runtime verify`, which imports the generated
 * module, instantiates it on the meta device and counts parameters. That closes
 * the loop the rest of the suite cannot: everywhere else we compare our own
 * numbers to our own numbers.
 *
 * The Python runtime and PyTorch are optional, so the whole block skips (it does
 * not fail) when either is missing. See python/README.md for install steps.
 */

import { describe, expect, it } from "bun:test";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { analyze, generateTorch, getPreset, loadEngine, PRESET_NAMES, scaleDesign, validate } from "../src/node.js";

await loadEngine();

const PRESET = "gpt2-small";
const TIMEOUT_MS = 180_000;

/** Ways to reach the runtime, in order of preference. */
const INVOCATIONS: ReadonlyArray<readonly [string, readonly string[]]> = [
  ["tensorcad-runtime", []],
  ["python", ["-m", "tensorcad_runtime"]],
  ["python3", ["-m", "tensorcad_runtime"]],
];

type Verify = {
  ok: boolean;
  params: number;
  expected: number | null;
  matches: boolean;
  class_name: string;
  params_by_module: Record<string, number>;
  forward: string;
  shapes: { input: number[]; logits: number[] } | null;
  flops: number | null;
  export_ok: boolean | null;
  export_shapes: { input?: string[]; output?: string[] } | null;
  warnings: string[];
  error_kind?: string;
  error?: string;
};

/** Write a freshly generated preset to a scratch directory. */
function writePreset(name: string): { dir: string; model: string } {
  const dir = mkdtempSync(join(tmpdir(), "tensorcad-py-"));
  const out = generateTorch(getPreset(name));
  for (const file of out.files) writeFileSync(join(dir, file.path), file.contents);
  return { dir, model: join(dir, "model.py") };
}

/** Run `verify` and parse the JSON object the runtime writes to stdout. */
function runVerify(
  invocation: readonly [string, readonly string[]],
  model: string,
  extra: string[] = [],
): Verify | { spawnFailed: string } {
  const [cmd, base] = invocation;
  const result = spawnSync(cmd, [...base, "verify", model, ...extra], {
    encoding: "utf8",
    timeout: TIMEOUT_MS,
    // Windows installs the entry point as tensorcad-runtime.exe; the shell
    // resolves that (and PATHEXT) for us.
    shell: process.platform === "win32",
  });
  if (result.error || result.stdout === null) {
    return { spawnFailed: result.error ? String(result.error.message) : "no stdout" };
  }
  const lines = result.stdout.trim().split(/\r?\n/).filter(Boolean);
  const last = lines.at(-1);
  if (!last) return { spawnFailed: `no JSON on stdout (exit ${result.status})` };
  try {
    return JSON.parse(last) as Verify;
  } catch {
    return { spawnFailed: `unparseable stdout: ${last.slice(0, 200)}` };
  }
}

/**
 * Probe once at load time: find a working invocation and confirm torch is
 * there. The probe skips the forward pass, FLOP count and export so it costs
 * little more than a torch import.
 */
function probe(): { invocation: (typeof INVOCATIONS)[number]; report: Verify } | { reason: string } {
  const { dir, model } = writePreset(PRESET);
  try {
    let reason = "no Python runtime found on PATH";
    for (const invocation of INVOCATIONS) {
      const report = runVerify(invocation, model, [
        "--no-export",
        "--no-flops",
        "--max-forward-bytes",
        "0",
      ]);
      if ("spawnFailed" in report) {
        reason = `${invocation[0]}: ${report.spawnFailed}`;
        continue;
      }
      if (report.error_kind === "torch_missing") {
        return { reason: "PyTorch is not installed" };
      }
      if (!report.ok) {
        return { reason: `tensorcad-runtime verify failed: ${report.error ?? "unknown error"}` };
      }
      return { invocation, report };
    }
    return { reason };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

const probed = probe();
const available = "invocation" in probed;

if (!available) {
  console.warn(`[python.test] skipping PyTorch cross-check: ${probed.reason}`);
}

describe.skipIf(!available)("tensorcad-runtime verify", () => {
  const { invocation, report } = probed as Exclude<typeof probed, { reason: string }>;

  it("agrees with the design's own parameter count for gpt2-small", () => {
    expect(report.matches).toBe(true);
    expect(report.params).toBe(124_439_808);
    expect(report.expected).toBe(report.params);
  });

  it("finds the model class and attributes parameters to top-level modules", () => {
    expect(report.class_name).toBe("Gpt2Small");
    const table = report.params_by_module;
    // Order follows the emitter's module declarations, so compare as a set.
    expect(Object.keys(table).sort()).toEqual(["embed", "final_norm", "head", "layers", "pos"]);
    // Tied weights are counted once, so the table adds up to the total.
    const sum = Object.values(table).reduce((a, b) => a + b, 0);
    expect(sum).toBe(report.params);
    // The bulk of GPT-2 is the twelve blocks.
    expect(table.layers).toBe(85_054_464);
  });

  it(
    "runs a forward pass, counts FLOPs and exports with dynamic B and T",
    () => {
      const { dir, model } = writePreset(PRESET);
      try {
        const full = runVerify(invocation, model);
        expect(full).not.toHaveProperty("spawnFailed");
        const r = full as Verify;

        expect(r.ok).toBe(true);
        expect(r.matches).toBe(true);

        // A real forward pass at the default batch 2, sequence 128.
        expect(r.forward).toBe("ok");
        expect(r.shapes?.logits).toEqual([2, 128, 50257]);

        // The FLOP count, against what PyTorch's profiler measures — not
        // roughly, exactly. The two conventions differ over the causal mask
        // and nothing else: a profiler counts the attention operator as if
        // nothing were masked, because the operator's shape does not depend on
        // the mask, and `fwdTotalUnmasked` is that convention. `fwdTotal` is
        // what a fused causal kernel actually does, and is the smaller number.
        const flops = analyze(getPreset(PRESET), { T: 128, B: 2 }).flops;
        expect(r.flops! / (2 * 128)).toBe(flops.fwdTotalUnmasked);
        expect(flops.fwdTotal).toBeLessThan(flops.fwdTotalUnmasked);

        // Both leading dimensions stay symbolic; the vocab does not.
        expect(r.export_ok).toBe(true);
        const output = r.export_shapes?.output;
        expect(output).toHaveLength(3);
        expect(output?.[2]).toBe("50257");
        expect(output?.[0]).not.toBe("2");
        expect(output?.[1]).not.toBe("128");
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

/**
 * A design that alternates, windows and softcaps, small enough to run.
 *
 * GPT-2 exercises none of those. This one is Gemma 2 shrunk to twenty million
 * parameters, which keeps every structural feature — local and global layers in
 * one repeat, a tanh on the attention scores, and a tanh on the logits — and
 * drops only the widths. On a CPU, where this runs, the generated attention
 * takes its unfused path, which is the one the profiler and the export see;
 * if that path were wrong, this is where it would show.
 */
describe.skipIf(!available)("tensorcad-runtime trace", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;

  it(
    "trains nano-sort to sort and records a forward pass it checked",
    () => {
      const { dir, model } = writePreset("nano-sort");
      try {
        const out = join(dir, "trace.json");
        const [cmd, base] = invocation;
        const result = spawnSync(cmd, [...base, "trace", model, "--out", out], {
          encoding: "utf8",
          timeout: TIMEOUT_MS,
          shell: process.platform === "win32",
        });
        const summary = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
        expect({ ok: summary.ok, error: summary.error }).toEqual({ ok: true, error: undefined });
        expect(summary.training.held_out_accuracy).toBe(1);
        expect(summary.answer).toEqual([...summary.input].sort((a: number, b: number) => a - b));
        // The attention matrix is recomputed from the captured query and key,
        // then multiplied by the captured values and held against what the
        // fused kernel produced. A trace that disagreed would leave it out.
        expect(summary.checks.attention_note).toBeNull();
        expect(summary.checks.attention_max_abs_error).toBeLessThan(1e-4);
        expect(summary.params).toBe(analyze(getPreset("nano-sort"), {}).params.total);

        const trace = JSON.parse(readFileSync(out, "utf8"));
        // Keyed by the design's own paths, with the stack index as a layer.
        const q = trace.activations["layers.2.block.attn.q_proj:out"];
        expect({ path: q.path, layer: q.layer, shape: q.shape }).toEqual({
          path: "layers/block/attn/q_proj",
          layer: 2,
          shape: [11, 48],
        });
        // An input that is another module's output is stored once.
        expect(trace.activations["layers.0.block.attn.k_proj:in"].same_as).toBe("layers.0.block.norm1:out");
        expect(trace.attention).toHaveLength(3);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

describe.skipIf(!available)("tensorcad-runtime trace, on a design unlike nano-sort", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;

  it(
    "records an untrained run, recomputes rotary grouped-query attention, and fingerprints the model the engine generates",
    () => {
      // Rotary positions, four query heads sharing two key-value heads, a gated
      // feed-forward and RMSNorm: every way a design can differ from nano-sort
      // that the trace has to read off the model rather than assume.
      const scaled = scaleDesign(getPreset("llama-3-8b"), { targetParams: 200e3, vocab: 256 });
      const doc = scaled.doc;
      for (const [k, v] of [["L", 2], ["H", 4], ["Hkv", 2], ["dh", 16]] as const) {
        (doc.symbols[k] as { value: number }).value = v;
      }
      const dir = mkdtempSync(join(tmpdir(), "tensorcad-trace-"));
      try {
        const files = generateTorch(doc).files;
        for (const file of files) writeFileSync(join(dir, file.path), file.contents);
        const out = join(dir, "trace.json");
        const [cmd, base] = invocation;
        const result = spawnSync(cmd, [...base, "trace", join(dir, "model.py"), "--out", out], {
          encoding: "utf8",
          timeout: TIMEOUT_MS,
          shell: process.platform === "win32",
        });
        const summary = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
        expect({ ok: summary.ok, error: summary.error }).toEqual({ ok: true, error: undefined });
        // A vocabulary of 256 is too many symbols to sort, so it is run as built.
        expect(summary.task.name).toBe("untrained");
        expect(summary.training.steps).toBe(0);
        expect(summary.checks.attention_note).toBeNull();
        expect(summary.checks.attention_max_abs_error).toBeLessThan(1e-4);

        const trace = JSON.parse(readFileSync(out, "utf8"));
        expect(trace.attention.map((a: { layer: number }) => a.layer)).toEqual([0, 1]);
        // [heads, positions, positions], after the key-value heads were repeated.
        expect(trace.attention[0].weights.shape).toEqual([4, 32, 32]);

        // The fingerprint the editor matches a trace to a design by, computed
        // on both sides of the language boundary from the same bytes.
        const source = files.find((f) => f.path === "model.py")!.contents;
        expect(trace.model_sha256).toBe(createHash("sha256").update(source, "utf8").digest("hex"));
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

describe.skipIf(!available)("the fused attention the generated code reaches for", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;

  it(
    "gives what the fallback gives, when FlashAttention is there to call",
    () => {
      // flash-attn is not installable everywhere this runs, so the kernel is a
      // stand-in written from FlashAttention's documented contract, not from
      // the helper: what it checks is that the helper hands the real kernel the
      // right layout, window, cap, scale and heads. On a GPU only, because the
      // helper only takes the fused path on one.
      const scaled = scaleDesign(getPreset("gemma-2-9b"), { targetParams: 20e6, vocab: 256 });
      const doc = scaled.doc;
      // A window shorter than the sequence the check runs, or it never bites.
      (doc.symbols.W as { value: number }).value = 16;
      const dir = mkdtempSync(join(tmpdir(), "tensorcad-fused-"));
      try {
        const files = generateTorch(doc).files;
        for (const file of files) writeFileSync(join(dir, file.path), file.contents);
        const cls = /^class (\w+)\(nn\.Module\)/gm;
        const model = files.find((f) => f.path === "model.py")!.contents;
        const name = [...model.matchAll(cls)].at(-1)![1]!;
        // The probe is a file of its own beside this one: Python full of quotes
        // and backslashes does not survive being a string in TypeScript intact.
        const probe = join(import.meta.dir, "fused_attention_probe.py");
        const [cmd] = invocation;
        const python = cmd === "tensorcad-runtime" ? "python" : cmd;
        const result = spawnSync(python, [probe, join(dir, "model.py"), name], {
          encoding: "utf8",
          timeout: TIMEOUT_MS,
          shell: process.platform === "win32",
        });
        const out = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
        if (!out.cuda) {
          console.warn("[python.test] no CUDA device; the fused attention path was not exercised");
          return;
        }
        // bf16 against the same attention in float32, relative to its size:
        // rounding is a few thousandths; one key too many in a window of 16, or
        // the cap applied before the scale, is several hundredths.
        for (const [key, err] of Object.entries(out.worst as Record<string, number>)) {
          expect({ key, close: err < 1e-2 }).toEqual({ key, close: true });
        }
        const layers = Number((doc.symbols.L as { value: number }).value);
        expect(out.model_calls).toBe(layers);
        // Half the layers windowed, half global: back W - 1, forward nothing.
        expect(out.model_windows).toEqual([[-1, -1], [15, 0]]);
        expect(out.finite).toBe(true);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

describe.skipIf(!available)("a windowed, softcapped design runs", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;

  it(
    "agrees with PyTorch on its parameters and its FLOPs",
    () => {
      const scaled = scaleDesign(getPreset("gemma-2-9b"), { targetParams: 20e6, vocab: 4096 });
      const dir = mkdtempSync(join(tmpdir(), "tensorcad-gemma-"));
      try {
        const out = generateTorch(scaled.doc);
        expect(out.warnings).toEqual([]);
        for (const file of out.files) writeFileSync(join(dir, file.path), file.contents);

        const full = runVerify(invocation, join(dir, "model.py"));
        expect(full).not.toHaveProperty("spawnFailed");
        const r = full as Verify;
        expect({ ok: r.ok, matches: r.matches, forward: r.forward }).toEqual({
          ok: true,
          matches: true,
          forward: "ok",
        });
        expect(r.params).toBe(scaled.achieved);

        const flops = analyze(scaled.doc, { T: 128, B: 2 }).flops;
        expect(r.flops! / (2 * 128)).toBe(flops.fwdTotalUnmasked);
        // The tanh on the scores and on the logits is arithmetic the profiler
        // does not count as a matmul, and neither do we.
        expect(flops.elementwise).toBeGreaterThan(0);
        expect(r.export_ok).toBe(true);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

/**
 * A design with its own mask and score, as the inspector would write them.
 *
 * Llama 3 shrunk to a few hundred thousand parameters, so the grouped heads
 * are still grouped, with a mask that reads the head and the design's symbols
 * and ALiBi's distance penalty on the scores. What it checks is that the two
 * functions the engine prints mean in PyTorch what the engine counted, and
 * that the model they are part of is still one the runtime can verify.
 */
function withExpressions(): ReturnType<typeof scaleDesign>["doc"] {
  const doc = scaleDesign(getPreset("llama-3-8b"), { targetParams: 200e3, vocab: 256 }).doc;
  for (const [k, v] of [["L", 2], ["H", 4], ["Hkv", 2], ["dh", 16]] as const) {
    (doc.symbols[k] as { value: number }).value = v;
  }
  const block = doc.graph.nodes.find((n) => n.id === "layers")!.graph!.nodes.find((n) => n.id === "block")!;
  block.params = {
    ...block.params,
    causal: false,
    mask: "kv <= q and (q - kv < 2 * dh or h == 0) or kv < 4",
    score: "score - 2 ** (-8 * (h + 1) / heads) * abs(q - kv)",
  };
  return doc;
}

describe.skipIf(!available)("attention written as expressions", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;

  it(
    "means in PyTorch what FlexAttention takes it to mean",
    () => {
      const dir = mkdtempSync(join(tmpdir(), "tensorcad-expr-"));
      try {
        const files = generateTorch(withExpressions()).files;
        for (const file of files) writeFileSync(join(dir, file.path), file.contents);
        const probe = join(import.meta.dir, "expression_attention_probe.py");
        const [cmd] = invocation;
        const python = cmd === "tensorcad-runtime" ? "python" : cmd;
        const result = spawnSync(python, [probe, join(dir, "model.py")], {
          encoding: "utf8",
          timeout: TIMEOUT_MS,
          shell: process.platform === "win32",
        });
        const out = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
        // One mask and one score for both layers: they are the same layer.
        expect({ masks: out.masks, scores: out.scores }).toEqual({
          masks: ["mask_mod_1"],
          scores: ["score_mod_1"],
        });
        // Every pairing, with and without grouped heads, against FlexAttention.
        expect(Object.keys(out.cases)).toHaveLength(6);
        for (const [key, err] of Object.entries(out.cases as Record<string, number>)) {
          expect({ key, close: err < 1e-5 }).toEqual({ key, close: true });
        }
        expect(out.empty_row).toEqual({ ours: 0, theirs: 0 });
        expect(out.finite).toBe(true);
        if (!out.cuda) {
          console.warn("[python.test] no CUDA device; the compiled FlexAttention path was not exercised");
          return;
        }
        // Compiled where Triton is, the fallback where it is not, and saying
        // so when it is the fallback. Either gives the same attention.
        expect(out.cuda_vs_cpu).toBeLessThan(1e-4);
        expect(out.said_unfused).toBe(!out.flex_compiled);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );

  it(
    "verifies: the parameters, the FLOPs the profiler counts, the export",
    () => {
      const doc = withExpressions();
      const dir = mkdtempSync(join(tmpdir(), "tensorcad-expr-"));
      try {
        const out = generateTorch(doc);
        expect(out.warnings).toEqual([]);
        for (const file of out.files) writeFileSync(join(dir, file.path), file.contents);
        const full = runVerify(invocation, join(dir, "model.py"));
        expect(full).not.toHaveProperty("spawnFailed");
        const r = full as Verify;
        expect({ ok: r.ok, matches: r.matches, forward: r.forward, export: r.export_ok }).toEqual({
          ok: true,
          matches: true,
          forward: "ok",
          export: true,
        });
        const analysis = analyze(doc, { T: 128, B: 2 });
        expect(r.params).toBe(analysis.params.total);
        // The eager form computes every score and then masks, which is what
        // the profiler counts; the analysis counts what the mask keeps.
        expect(r.flops! / (2 * 128)).toBe(analysis.flops.fwdTotalUnmasked);
        expect(analysis.flops.fwdAttention).toBeLessThan(analysis.flops.fwdAttentionUnmasked / 2);
        expect(analysis.flops.elementwise).toBeGreaterThan(0);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

/**
 * BLOOM, the first preset whose attention is an expression.
 *
 * Its ALiBi is written as a score, not a switch, so what the engine prints is
 * all there is to say that the bias is BLOOM's. It is held against a
 * transcription of Hugging Face's own `build_alibi_tensor`, and the model it is
 * part of against PyTorch's count of its parameters and its FLOPs.
 */
describe.skipIf(!available)("bloom-7b1", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;

  it(
    "has the parameters and the FLOPs it says, and exports",
    () => {
      const { dir, model } = writePreset("bloom-7b1");
      try {
        const full = runVerify(invocation, model);
        expect(full).not.toHaveProperty("spawnFailed");
        const r = full as Verify;
        // Seven billion parameters do not fit a forward pass in the runner's
        // memory; the count, the FLOPs and the export are taken on the meta
        // device, which is where the claim is.
        expect({ ok: r.ok, matches: r.matches, params: r.params, export: r.export_ok }).toEqual({
          ok: true,
          matches: true,
          params: 7_069_016_064,
          export: true,
        });
        const flops = analyze(getPreset("bloom-7b1"), { T: 128, B: 2 }).flops;
        expect(r.flops! / (2 * 128)).toBe(flops.fwdTotalUnmasked);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );

  it(
    "biases its scores the way BLOOM does",
    () => {
      const { dir, model } = writePreset("bloom-7b1");
      try {
        const [cmd] = invocation;
        const python = cmd === "tensorcad-runtime" ? "python" : cmd;
        const result = spawnSync(python, [join(import.meta.dir, "alibi_probe.py"), model, "32"], {
          encoding: "utf8",
          timeout: TIMEOUT_MS,
          shell: process.platform === "win32",
        });
        const out = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
        // BLOOM adds slope × key position, the design subtracts slope ×
        // distance: the same bias up to a constant along each row, which is
        // float32 rounding over values up to fifty, and the same weights.
        expect(out.row_spread).toBeLessThan(1e-4);
        expect(out.weights).toBeLessThan(1e-5);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

/**
 * gpt-oss, the first preset with attention sinks.
 *
 * Twenty-one billion parameters are counted on the meta device. The sinks are
 * held against two other statements of them: FlexAttention's, the log-sum-exp
 * rescale the generated code uses on CUDA, and a transcription of Hugging
 * Face's own gpt-oss attention. And a copy small enough to run goes through a
 * forward pass and an export, with the dense expert dispatch that is there to
 * make a mixture of experts traceable.
 */
describe.skipIf(!available)("gpt-oss-20b", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;

  it(
    "has the parameters it says, sinks included",
    () => {
      const { dir, model } = writePreset("gpt-oss-20b");
      try {
        const full = runVerify(invocation, model, ["--no-export", "--no-flops"]);
        expect(full).not.toHaveProperty("spawnFailed");
        const r = full as Verify;
        expect({ ok: r.ok, matches: r.matches, params: r.params }).toEqual({
          ok: true,
          matches: true,
          params: 20_914_757_184,
        });
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );

  it(
    "puts its sinks where FlexAttention and Hugging Face do",
    () => {
      const { dir, model } = writePreset("gpt-oss-20b");
      try {
        const [cmd] = invocation;
        const python = cmd === "tensorcad-runtime" ? "python" : cmd;
        const result = spawnSync(python, [join(import.meta.dir, "sinks_probe.py"), model], {
          encoding: "utf8",
          timeout: TIMEOUT_MS,
          shell: process.platform === "win32",
        });
        const out = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
        // One mask for the band, one for the full layers.
        expect(Object.keys(out.cases)).toEqual(["mask_mod_1", "mask_mod_2"]);
        for (const [mask, err] of Object.entries(out.cases as Record<string, { fused: number; hugging_face: number }>)) {
          expect({ mask, fused: err.fused < 1e-5, hugging_face: err.hugging_face < 1e-5 }).toEqual({
            mask,
            fused: true,
            hugging_face: true,
          });
        }
        // A query with nothing but its sink puts all of its attention there.
        expect(out.empty_row).toEqual({ ours: 0, fused: 0 });
        expect(out.finite).toBe(true);
        // A sink far below every score takes nothing.
        expect(out.silent_sink).toBeLessThan(1e-6);
        if (out.cuda) expect(out.cuda_vs_cpu).toBeLessThan(1e-4);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );

  it(
    "shrunk to run, goes forward and exports",
    () => {
      const scaled = scaleDesign(getPreset("gpt-oss-20b"), { targetParams: 2e6, vocab: 256 });
      // A pair of layers is the unit, so it cannot shrink to one.
      expect((scaled.doc.symbols.L as { value: number }).value).toBe(2);
      const dir = mkdtempSync(join(tmpdir(), "tensorcad-gptoss-"));
      try {
        const out = generateTorch(scaled.doc, { moeDispatch: "dense" });
        expect(out.warnings).toEqual([]);
        for (const file of out.files) writeFileSync(join(dir, file.path), file.contents);
        const full = runVerify(invocation, join(dir, "model.py"), ["--no-flops"]);
        expect(full).not.toHaveProperty("spawnFailed");
        const r = full as Verify;
        expect({ ok: r.ok, matches: r.matches, forward: r.forward, export: r.export_ok }).toEqual({
          ok: true,
          matches: true,
          forward: "ok",
          export: true,
        });
        expect(r.params).toBe(scaled.achieved);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

/**
 * Differential attention, against the paper's own code and the profiler.
 *
 * nano-sort with its attention made differential: two heads, each a pair of
 * twelve-wide maps over values twice that. No released model to regress
 * against, so the generated module is held against a transcription of
 * Microsoft's reference, given the same weights, and the whole model against
 * PyTorch's count of its parameters and FLOPs.
 */
function differential(extra: Record<string, unknown>): ReturnType<typeof getPreset> {
  const doc = structuredClone(getPreset("nano-sort"));
  for (const [k, v] of [["H", 2], ["Hkv", 2], ["dh", 12]] as const) {
    (doc.symbols[k] as { value: number }).value = v;
  }
  const block = doc.graph.nodes.find((n) => n.id === "layers")!.graph!.nodes.find((n) => n.id === "block")!;
  block.params = { ...block.params, attention: "diff", ...extra };
  return doc;
}

describe.skipIf(!available)("differential attention", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;

  it(
    "computes what the reference implementation computes",
    () => {
      // The reference has no biases and leaves rotary to its caller, and its
      // lambda_init is scheduled by depth: this is its layer one.
      const doc = differential({ rope: null, attn_bias: false, lambda_init: 0.8 - 0.6 * Math.exp(-0.3) });
      const dir = mkdtempSync(join(tmpdir(), "tensorcad-diff-"));
      try {
        for (const file of generateTorch(doc).files) writeFileSync(join(dir, file.path), file.contents);
        const [cmd] = invocation;
        const python = cmd === "tensorcad-runtime" ? "python" : cmd;
        const result = spawnSync(python, [join(import.meta.dir, "diff_attention_probe.py"), join(dir, "model.py"), "1"], {
          encoding: "utf8",
          timeout: TIMEOUT_MS,
          shell: process.platform === "win32",
        });
        const out = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
        expect({ heads: out.heads, head_dim: out.head_dim }).toEqual({ heads: 2, head_dim: 12 });
        expect(out.params_ours).toBe(out.params_reference);
        // Two fused attentions subtracted against one subtraction of two
        // score matrices: the same numbers, to float32 rounding.
        expect(out.max_abs / out.scale).toBeLessThan(1e-5);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );

  it(
    "has the parameters and the FLOPs it says, and exports",
    () => {
      const doc = differential({ rope: { theta: 10000 } });
      const dir = mkdtempSync(join(tmpdir(), "tensorcad-diff-"));
      try {
        const out = generateTorch(doc);
        expect(out.warnings).toEqual([]);
        for (const file of out.files) writeFileSync(join(dir, file.path), file.contents);
        // nano-sort's positions stop at eleven.
        const full = runVerify(invocation, join(dir, "model.py"), ["--seq", "11"]);
        expect(full).not.toHaveProperty("spawnFailed");
        const r = full as Verify;
        expect({ ok: r.ok, matches: r.matches, forward: r.forward, export: r.export_ok }).toEqual({
          ok: true,
          matches: true,
          forward: "ok",
          export: true,
        });
        const analysis = analyze(doc, { T: 11, B: 2 });
        expect(r.params).toBe(analysis.params.total);
        // Values twice as wide as keys: the profiler counts the value
        // product at the value width, and so does the analysis.
        expect(r.flops! / (2 * 11)).toBe(analysis.flops.fwdTotalUnmasked);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

/**
 * Attention written out, with talking heads.
 *
 * The one place a design keeps its score matrix on purpose. At the identity
 * the generated block is plain attention and has to agree with PyTorch's fused
 * kernel; with the mixes learned it has to agree with the paper's own einsum
 * form; and the model it is part of has to have the parameters and FLOPs the
 * analysis says, every score counted since an eager matmul skips none.
 */
describe.skipIf(!available)("talking heads, written out", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;

  const talking = (): ReturnType<typeof getPreset> => {
    const doc = structuredClone(getPreset("nano-sort"));
    const block = doc.graph.nodes.find((n) => n.id === "layers")!.graph!.nodes.find((n) => n.id === "block")!;
    block.params = { ...block.params, talking_heads: true };
    return doc;
  };

  it(
    "is plain attention at the identity and the paper's with the mixes learned",
    () => {
      const dir = mkdtempSync(join(tmpdir(), "tensorcad-talk-"));
      try {
        const out = generateTorch(talking());
        expect(out.warnings).toEqual([]);
        for (const file of out.files) writeFileSync(join(dir, file.path), file.contents);
        const [cmd] = invocation;
        const python = cmd === "tensorcad-runtime" ? "python" : cmd;
        const result = spawnSync(
          python,
          [join(import.meta.dir, "talking_heads_probe.py"), join(dir, "model.py"), "16"],
          { encoding: "utf8", timeout: TIMEOUT_MS, shell: process.platform === "win32" },
        );
        const probe = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
        expect(probe.heads).toBe(3);
        expect(probe.identity_vs_sdpa).toBeLessThan(1e-5);
        expect(probe.mixed_vs_paper / probe.scale).toBeLessThan(1e-5);
        // And the mixing changes the answer, or the second check proved nothing.
        expect(probe.mixing_matters).toBeGreaterThan(1e-2);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );

  it(
    "has the parameters and the FLOPs it says, and exports",
    () => {
      const doc = talking();
      const dir = mkdtempSync(join(tmpdir(), "tensorcad-talk-"));
      try {
        for (const file of generateTorch(doc).files) writeFileSync(join(dir, file.path), file.contents);
        const full = runVerify(invocation, join(dir, "model.py"), ["--seq", "11"]);
        expect(full).not.toHaveProperty("spawnFailed");
        const r = full as Verify;
        expect({ ok: r.ok, matches: r.matches, forward: r.forward, export: r.export_ok }).toEqual({
          ok: true,
          matches: true,
          forward: "ok",
          export: true,
        });
        const analysis = analyze(doc, { T: 11, B: 2 });
        expect(r.params).toBe(analysis.params.total);
        // Written out, the attention is counted as the profiler counts it:
        // every score, masked or not.
        expect(r.flops! / (2 * 11)).toBe(analysis.flops.fwdTotalUnmasked);
        expect(analysis.flops.fwdTotal).toBe(analysis.flops.fwdTotalUnmasked);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

/**
 * An encoder-decoder, M11's second phase: a source and a target, an encoder
 * stack over the one and a decoder stack over the other whose every layer
 * attends to all of the encoder's output. The runtime builds both inputs, the
 * profiler's count per target token has to be the analysis's — the encoder's
 * share spread over the target, cross-attention at every source position —
 * and export has to let the batch, the source and the target all move.
 */
describe.skipIf(!available)("an encoder-decoder", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;
  const seq2seq = JSON.parse(
    readFileSync(join(import.meta.dir, "../../core-go/codegen/testdata/seq2seq.json"), "utf8"),
  ) as ReturnType<typeof getPreset>;

  it(
    "has the parameters and the FLOPs it says, runs on both inputs, and exports",
    () => {
      const dir = mkdtempSync(join(tmpdir(), "tensorcad-seq2seq-"));
      try {
        const out = generateTorch(seq2seq);
        expect(out.warnings).toEqual([]);
        for (const file of out.files) writeFileSync(join(dir, file.path), file.contents);
        const full = runVerify(invocation, join(dir, "model.py"), ["--seq", "16", "--source", "24"]);
        expect(full).not.toHaveProperty("spawnFailed");
        const r = full as Verify & { inputs?: Record<string, number[]> };
        expect({ ok: r.ok, matches: r.matches, forward: r.forward, export: r.export_ok }).toEqual({
          ok: true,
          matches: true,
          forward: "ok",
          export: true,
        });
        expect(r.inputs).toEqual({ src: [2, 24], tgt: [2, 16] });
        const analysis = analyze(seq2seq, { T: 16, B: 2, S: 24 });
        expect(r.params).toBe(analysis.params.total);
        // Per target token, the encoder's work spread over the target's tokens.
        expect(r.flops! / (2 * 16)).toBe(analysis.flops.fwdTotalUnmasked);
        // One example is every target token's share, the source's included.
        expect(analysis.flops.fwdPerExample).toBe(analysis.flops.fwdTotal * 16);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

/**
 * T5's relative-position bias, M11's third phase: a learned table per stack,
 * outside it, that every layer's score expression reads. The generated score
 * has to add exactly Hugging Face's bias, two-sided in the encoder and
 * one-sided in the decoder; FlexAttention has to agree with the eager
 * fallback; and a gradient has to reach the table through both, which is what
 * lets the table be a parameter rather than a constant.
 */
describe.skipIf(!available)("a relative-position bias", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;
  const t5ish = JSON.parse(
    readFileSync(join(import.meta.dir, "../../core-go/codegen/testdata/t5ish.json"), "utf8"),
  ) as ReturnType<typeof getPreset>;
  const write = (): string => {
    const dir = mkdtempSync(join(tmpdir(), "tensorcad-t5-"));
    const out = generateTorch(t5ish);
    expect(out.warnings).toEqual([]);
    for (const file of out.files) writeFileSync(join(dir, file.path), file.contents);
    return dir;
  };

  it(
    "is T5's, in both directions, and learns through FlexAttention",
    () => {
      const dir = write();
      try {
        const [cmd] = invocation;
        const python = cmd === "tensorcad-runtime" ? "python" : cmd;
        const result = spawnSync(python, [join(import.meta.dir, "t5_bias_probe.py"), join(dir, "model.py"), "4"], {
          encoding: "utf8",
          timeout: TIMEOUT_MS,
          shell: process.platform === "win32",
        });
        const out = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
        expect(out).toEqual({
          score_mod_1: 0,
          score_mod_2: 0,
          flex_vs_eager: expect.any(Number),
          eager_grad: true,
          flex_grad: true,
        });
        expect(out.flex_vs_eager).toBeLessThan(1e-5);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );

  it(
    "has the parameters and the FLOPs it says, runs, and exports",
    () => {
      const dir = write();
      try {
        const full = runVerify(invocation, join(dir, "model.py"), ["--seq", "16", "--source", "24"]);
        expect(full).not.toHaveProperty("spawnFailed");
        const r = full as Verify;
        expect({ ok: r.ok, matches: r.matches, forward: r.forward, export: r.export_ok }).toEqual({
          ok: true,
          matches: true,
          forward: "ok",
          export: true,
        });
        const analysis = analyze(t5ish, { T: 16, B: 2, S: 24 });
        expect(r.params).toBe(analysis.params.total);
        expect(r.flops! / (2 * 16)).toBe(analysis.flops.fwdTotalUnmasked);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );
});

/**
 * T5, the first encoder-decoder presets: M11's fourth phase, and the end of
 * M10's relative bias. Each is held three ways. Its parameters, profiled FLOPs,
 * forward pass and export, as every design is. And weight for weight against a
 * transcription of Hugging Face's own forward pass, with the generated model's
 * parameters renamed into Hugging Face's layout: the logits have to agree, and
 * the renaming has to use every weight exactly once, which is what makes the
 * two the same parameters rather than the same count.
 */
describe.skipIf(!available)("T5", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;
  const cases = [
    { name: "t5-small", params: 60_506_624, layers: 6, heads: 8, gated: false, tied: true },
    { name: "flan-t5-base", params: 247_577_856, layers: 12, heads: 12, gated: true, tied: false },
  ] as const;

  for (const c of cases) {
    it(
      `${c.name} has the parameters and the FLOPs it says, runs, and exports`,
      () => {
        const { dir, model } = writePreset(c.name);
        try {
          const full = runVerify(invocation, model, ["--seq", "16", "--source", "24"]);
          expect(full).not.toHaveProperty("spawnFailed");
          const r = full as Verify & { inputs?: Record<string, number[]> };
          expect({ ok: r.ok, matches: r.matches, forward: r.forward, export: r.export_ok }).toEqual({
            ok: true,
            matches: true,
            forward: "ok",
            export: true,
          });
          expect(r.params).toBe(c.params);
          expect(r.inputs).toEqual({ source: [2, 24], target: [2, 16] });
          const analysis = analyze(getPreset(c.name), { T: 16, B: 2, S: 24 });
          expect(r.flops! / (2 * 16)).toBe(analysis.flops.fwdTotalUnmasked);
        } finally {
          rmSync(dir, { recursive: true, force: true });
        }
      },
      TIMEOUT_MS,
    );

    it(
      `${c.name} computes what Hugging Face's T5 computes`,
      () => {
        const { dir, model } = writePreset(c.name);
        try {
          const [cmd] = invocation;
          const python = cmd === "tensorcad-runtime" ? "python" : cmd;
          const result = spawnSync(python, [join(import.meta.dir, "t5_probe.py"), model], {
            encoding: "utf8",
            timeout: TIMEOUT_MS,
            shell: process.platform === "win32",
          });
          const out = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
          expect(out).toMatchObject({
            layers: c.layers,
            heads: c.heads,
            gated: c.gated,
            tied: c.tied,
            every_weight_once: true,
            hf_params: c.params,
          });
          // float32 rounding over a few dozen layers, and nothing else.
          expect(out.logits_diff / out.logits_scale).toBeLessThan(1e-5);
        } finally {
          rmSync(dir, { recursive: true, force: true });
        }
      },
      TIMEOUT_MS,
    );
  }
});

/**
 * Packed documents kept apart, M12's second phase: llama-3-8b scaled to two
 * million parameters, with every layer's attention reading each position's
 * document. The runtime makes up a packing for the documents input, as a
 * pretraining pipeline would, and the model verifies on it. The probe holds
 * the rest against PyTorch's own code: one document's tokens move no other
 * document's outputs, flex_attention given the model's mask computes what the
 * eager form does, the block mask is built once a batch, and over many rows of
 * the runtime's packing the scores kept and the blocks create_block_mask says
 * a kernel computes are the engine's two figures.
 */
describe.skipIf(!available)("packed documents", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;
  const T = 512;
  const design = (): ReturnType<typeof getPreset> => {
    const doc = structuredClone(scaleDesign(getPreset("llama-3-8b"), { targetParams: 2e6, vocab: 256 }).doc);
    doc.graph.nodes.push({ id: "docs", type: "input", params: { shape: "B T", dtype: "int64", role: "documents" } });
    doc.graph.edges.push(["docs:x", "layers:doc"]);
    const stack = doc.graph.nodes.find((n) => n.id === "layers")!;
    for (const n of stack.graph!.nodes) {
      if (n.id === "_in") (n.params!.ports as Record<string, string>).doc = "B T";
      if (n.id === "block") n.params!.mask = "doc(b, q) == doc(b, kv)";
    }
    stack.graph!.edges.push(["_in:doc", "block:doc"]);
    return doc;
  };
  const write = (): { dir: string; model: string } => {
    const dir = mkdtempSync(join(tmpdir(), "tensorcad-packed-"));
    const out = generateTorch(design());
    expect(out.warnings).toEqual([]);
    for (const file of out.files) writeFileSync(join(dir, file.path), file.contents);
    return { dir, model: join(dir, "model.py") };
  };

  it(
    "has the parameters and the FLOPs it says, runs on a made-up packing, and exports",
    () => {
      const { dir, model } = write();
      try {
        const full = runVerify(invocation, model, ["--seq", "256"]);
        expect(full).not.toHaveProperty("spawnFailed");
        const r = full as Verify & { inputs?: Record<string, number[]> };
        expect({ ok: r.ok, matches: r.matches, forward: r.forward, export: r.export_ok }).toEqual({
          ok: true,
          matches: true,
          forward: "ok",
          export: true,
        });
        expect(r.inputs).toEqual({ tokens: [2, 256], docs: [2, 256] });
        const analysis = analyze(design(), { T: 256, B: 2 });
        expect(r.params).toBe(analysis.params.total);
        expect(r.flops! / (2 * 256)).toBe(analysis.flops.fwdTotalUnmasked);
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    },
    TIMEOUT_MS,
  );

  for (const [mean, spread, keptWithin] of [
    [96, 0, 0.02],
    [96, 1, 0.04],
  ] as const) {
    it(
      `keeps documents apart, builds a block mask once a batch, and costs what the engine says (mean ${mean}, spread ${spread})`,
      () => {
        const { dir, model } = write();
        try {
          const one = analyze(design(), { T });
          const packed = analyze(design(), { T, packing: { mean, spread } }).flops.packed!;
          // The engine's figures as keys a query: causal attention is T/2.
          const perKey = one.flops.fwdAttention / (T / 2);
          const kept = packed.fwdAttention / perKey;
          const blocks = packed.fwdAttentionBlocks / perKey;
          const [cmd] = invocation;
          const python = cmd === "tensorcad-runtime" ? "python" : cmd;
          const result = spawnSync(
            python,
            [join(import.meta.dir, "documents_probe.py"), model, `${T}`, `${mean}`, `${spread}`, `${kept}`, `${blocks}`],
            { encoding: "utf8", timeout: TIMEOUT_MS, shell: process.platform === "win32" },
          );
          const out = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
          expect(out).toMatchObject({
            other_documents_moved: 0,
            other_rows_moved: 0,
            flex_vs_eager: 0,
            layers_share_one: true,
            next_batch_rebuilds: true,
            edited_documents_rebuild: true,
            one_kept_per_mask: true,
            block_size: 128,
          });
          expect(out.own_document_moved).toBeGreaterThan(0);
          expect(Math.abs(out.kept_vs_engine)).toBeLessThan(keptWithin);
          expect(Math.abs(out.blocks_vs_engine)).toBeLessThan(0.02);
        } finally {
          rmSync(dir, { recursive: true, force: true });
        }
      },
      TIMEOUT_MS,
    );
  }
});

/**
 * Positions that restart at every document, M12's third phase. A row is packed
 * the way Hugging Face's DataCollatorWithFlattening packs one, and with its
 * documents kept apart the row has to compute what each document computes
 * alone. For rotary positions that holds whether the positions restart or not,
 * because attention sees only the distance between two tokens of a document;
 * for learned ones it holds only when they restart, because a position vector
 * is looked up by the position itself. The runtime's positions are the
 * collator's, and verify runs on its made-up documents and positions together.
 */
describe.skipIf(!available)("positions that restart", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;
  const packed = (preset: string, learned: boolean): ReturnType<typeof getPreset> => {
    const doc = structuredClone(scaleDesign(getPreset(preset), { targetParams: 2e6, vocab: 256 }).doc);
    doc.graph.nodes.push({ id: "docs", type: "input", params: { shape: "B T", dtype: "int64", role: "documents" } });
    doc.graph.nodes.push({ id: "positions", type: "input", params: { shape: "B T", dtype: "int64", role: "positions" } });
    doc.graph.edges.push(["docs:x", "layers:doc"]);
    const stack = doc.graph.nodes.find((n) => n.id === "layers")!;
    for (const n of stack.graph!.nodes) {
      const ports = n.params?.ports as Record<string, string> | undefined;
      if (n.id === "_in" && ports) {
        ports.doc = "B T";
        if (!learned) ports.pos = "B T";
      }
      if (n.id === "block") {
        n.params!.mask = "doc(b, q) == doc(b, kv)";
        if (!learned) n.params!.positions = true;
      }
    }
    stack.graph!.edges.push(["_in:doc", "block:doc"]);
    if (learned) {
      const table = doc.graph.nodes.find((n) => n.type === "pos_embedding")!;
      table.params!.positions = true;
      doc.graph.edges.push(["positions:x", `${table.id}:pos`]);
    } else {
      stack.graph!.edges.push(["_in:pos", "block:pos"]);
      doc.graph.edges.push(["positions:x", "layers:pos"]);
    }
    return doc;
  };

  for (const [preset, learned] of [
    ["llama-3-8b", false],
    ["gpt2-small", true],
  ] as const) {
    it(
      `${learned ? "learned" : "rotary"} positions: a packed row computes what its documents compute alone, and verifies`,
      () => {
        const dir = mkdtempSync(join(tmpdir(), "tensorcad-positions-"));
        try {
          const doc = packed(preset, learned);
          const out = generateTorch(doc);
          expect(out.warnings).toEqual([]);
          expect(validate(doc, {}).findings.filter((f) => f.severity === "error")).toEqual([]);
          for (const file of out.files) writeFileSync(join(dir, file.path), file.contents);
          const model = join(dir, "model.py");

          const [cmd] = invocation;
          const python = cmd === "tensorcad-runtime" ? "python" : cmd;
          const result = spawnSync(python, [join(import.meta.dir, "positions_probe.py"), model, "256"], {
            encoding: "utf8",
            timeout: TIMEOUT_MS,
            shell: process.platform === "win32",
          });
          const probe = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!);
          expect(probe.matches_collator).toBe(true);
          expect(probe.restarted_vs_alone / probe.scale).toBeLessThan(1e-5);
          if (learned) expect(probe.continued_vs_alone / probe.scale).toBeGreaterThan(0.1);
          else expect(probe.continued_vs_alone / probe.scale).toBeLessThan(1e-5);

          const full = runVerify(invocation, model, ["--seq", "128"]);
          expect(full).not.toHaveProperty("spawnFailed");
          const r = full as Verify & { inputs?: Record<string, number[]> };
          expect({ ok: r.ok, matches: r.matches, forward: r.forward, export: r.export_ok }).toEqual({
            ok: true,
            matches: true,
            forward: "ok",
            export: true,
          });
          expect(Object.keys(r.inputs!).sort()).toEqual(["docs", "positions", "tokens"]);
        } finally {
          rmSync(dir, { recursive: true, force: true });
        }
      },
      TIMEOUT_MS,
    );
  }
});

/**
 * Every preset, through `ast.parse`.
 *
 * Cheaper than the block above and answering a different question: not "does
 * this model have the weights we said" but "is this a Python file at all".
 * Nothing in the IR stops a block being called `global` or `class`, and
 * `self.global = ...` is a syntax error — a file that imports nowhere, in a
 * design whose every number is right. Needs an interpreter, not PyTorch.
 */
const PYTHONS = ["python", "python3"] as const;

function anyPython(): string | null {
  for (const exe of PYTHONS) {
    const probe = spawnSync(exe, ["-c", "import ast"], { encoding: "utf8" });
    if (probe.status === 0) return exe;
  }
  return null;
}

const python = anyPython();
if (!python) console.warn("[python.test] skipping the parse check: no interpreter");

describe.skipIf(!python)("every generated model is valid Python", () => {
  it(
    "parses, for all twenty presets and both mixture-of-experts dispatches",
    () => {
      for (const name of PRESET_NAMES) {
        const doc = getPreset(name);
        for (const options of [{}, { moeDispatch: "dense" as const }]) {
          const model = generateTorch(doc, options).files.find((f) => f.path === "model.py");
          expect({ name, has: model !== undefined }).toEqual({ name, has: true });
          const check = spawnSync(
            python!,
            ["-c", "import ast,sys;ast.parse(sys.stdin.read())"],
            { encoding: "utf8", input: model!.contents },
          );
          expect({ name, dispatch: options.moeDispatch ?? "sparse", error: check.stderr.trim() }).toEqual({
            name,
            dispatch: options.moeDispatch ?? "sparse",
            error: "",
          });
        }
      }
    },
    TIMEOUT_MS,
  );
});

/**
 * Training memory, measured on the GPU and held against the analysis.
 *
 * `tensorcad-runtime measure` runs real AdamW steps and reads PyTorch's own
 * allocator. Two of its numbers are what the analysis claims under the same
 * conditions, and are held to it here:
 *
 * - At rest, between steps, weights, gradients and optimizer state. The
 *   analysis's bf16 accounting is sixteen bytes a parameter — bf16 weights and
 *   gradients over an fp32 master copy and two fp32 moments — and PyTorch's
 *   mixed precision comes to the same sixteen by another route: fp32 weights,
 *   fp32 gradients, two fp32 moments.
 * - The activations a forward pass saves for the backward one, with the whole
 *   model in bf16, which is the activation dtype the analysis assumes.
 *
 * What autocast saves is not held yet, only reported: it keeps the residual
 * stream and the norms in fp32 and a bf16 copy of every weight for the whole
 * step, which is the next thing the analysis has to learn.
 *
 * Needs a CUDA device; skips without one, with a warning, as CI does.
 */
describe.skipIf(!available)("training memory, measured", () => {
  const { invocation } = probed as Exclude<typeof probed, { reason: string }>;
  const MiB = 2 ** 20;

  type Measured = {
    ok: boolean;
    error_kind?: string;
    params: number;
    weights_bytes: number;
    grads_bytes: number;
    optimizer_bytes: number;
    saved_bytes: number;
    peak_bytes: number;
  };

  function measure(model: string, recipe: string, batch: number, seq: number): Measured {
    const [cmd, base] = invocation;
    const result = spawnSync(
      cmd,
      [...base, "measure", model, "--recipe", recipe, "--batch", String(batch), "--seq", String(seq)],
      { encoding: "utf8", timeout: TIMEOUT_MS, shell: process.platform === "win32" },
    );
    return JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1)!) as Measured;
  }

  const designs: [string, () => ReturnType<typeof getPreset>][] = [
    ["gpt2-small", () => getPreset("gpt2-small")],
    // A Llama at a size a desk GPU trains: grouped-query attention, a gated
    // feed-forward and RMS norm, where GPT-2 has none of the three.
    ["llama-style 150M", () => scaleDesign(getPreset("llama-3-8b"), { targetParams: 150e6 }).doc],
  ];

  for (const [name, make] of designs) {
    it(
      `${name}: the resting state and the saved activations are what the analysis says`,
      () => {
        const doc = make();
        const dir = mkdtempSync(join(tmpdir(), "tensorcad-mem-"));
        try {
          for (const file of generateTorch(doc).files) writeFileSync(join(dir, file.path), file.contents);
          const model = join(dir, "model.py");
          const [B, T] = [2, 512];
          const amp = measure(model, "amp", B, T);
          if (!amp.ok && amp.error_kind === "no_cuda") {
            console.warn("[python.test] no CUDA device; training memory was not measured");
            return;
          }
          expect(amp.ok).toBe(true);
          const bf16 = measure(model, "bf16", B, T);
          expect(bf16.ok).toBe(true);

          const a = analyze(doc, { B, T, dtype: "bf16", optimizer: "adamw", recompute: "none", gpus: 1, flash: true });
          const train = a.memory.train;
          expect(amp.params).toBe(a.params.total);

          const resting = amp.weights_bytes + amp.grads_bytes + amp.optimizer_bytes;
          const claimed = train.weights + train.grads + train.optimizer;
          expect(Math.abs(resting / claimed - 1)).toBeLessThan(0.02);

          const saved = bf16.saved_bytes / train.activations;
          console.info(
            `[python.test] ${name} at B=${B} T=${T}: activations ${(train.activations / MiB).toFixed(0)} MiB analysed, ` +
              `${(bf16.saved_bytes / MiB).toFixed(0)} measured in bf16 (${saved.toFixed(2)}x), ` +
              `${(amp.saved_bytes / MiB).toFixed(0)} under autocast (${(amp.saved_bytes / train.activations).toFixed(2)}x)`,
          );
          expect(Math.abs(saved - 1)).toBeLessThan(0.15);
        } finally {
          rmSync(dir, { recursive: true, force: true });
        }
      },
      TIMEOUT_MS,
    );
  }
});
