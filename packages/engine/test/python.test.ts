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
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { analyze, generateTorch, getPreset, loadEngine, PRESET_NAMES, scaleDesign } from "../src/node.js";

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
 * one repeat, a tanh on the attention scores that rules out the fused kernel,
 * and a tanh on the logits — and drops only the widths. If the emitted eager
 * attention were wrong, this is where it would show.
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
