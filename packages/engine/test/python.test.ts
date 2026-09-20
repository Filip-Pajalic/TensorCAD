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
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { generateTorch, getPreset, loadEngine, PRESET_NAMES } from "../src/node.js";

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

        // Roughly 2 * params per token, plus attention.
        expect(r.flops).toBeGreaterThan(2 * r.params * 2 * 128);

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
