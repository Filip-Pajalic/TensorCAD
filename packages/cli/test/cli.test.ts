/**
 * CLI contract tests.
 *
 * These spawn the real entry point so the exit codes are covered too: a broken
 * design must fail CI, a good one must not.
 */

import { describe, expect, test, afterAll } from "bun:test";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const ENTRY = resolve(import.meta.dir, "../src/index.ts");
const ROOT = resolve(import.meta.dir, "../../..");
const BROKEN = resolve(import.meta.dir, "fixtures/broken.tensorcad.json");

const temps: string[] = [];
function tempDir(): string {
  const d = mkdtempSync(join(tmpdir(), "tensorcad-cli-"));
  temps.push(d);
  return d;
}
afterAll(() => {
  for (const d of temps) rmSync(d, { recursive: true, force: true });
});

function cli(...args: string[]): { code: number; stdout: string; stderr: string } {
  const r = spawnSync(process.execPath, ["run", ENTRY, ...args], {
    cwd: ROOT,
    encoding: "utf8",
    env: { ...process.env, NO_COLOR: "1", FORCE_COLOR: "0" },
  });
  return { code: r.status ?? -1, stdout: r.stdout ?? "", stderr: r.stderr ?? "" };
}

describe("help", () => {
  test("no arguments prints usage and exits 0", () => {
    const r = cli();
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("Commands");
    expect(r.stdout).toContain("analyze");
    expect(r.stdout).toContain("llama-3-8b");
  });

  test("an unknown command exits non-zero", () => {
    const r = cli("frobnicate");
    expect(r.code).toBe(1);
    expect(r.stderr).toContain("Unknown command");
  });
});

describe("list", () => {
  test("names every preset and hardware profile", () => {
    const r = cli("list");
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("llama-3-8b");
    expect(r.stdout).toContain("gpt2-small");
    expect(r.stdout).toContain("h100-sxm");
    expect(r.stdout).toContain("rtx5080");
  });

  test("--json is machine readable", () => {
    const r = cli("list", "--json");
    expect(r.code).toBe(0);
    const parsed = JSON.parse(r.stdout);
    expect(parsed.presets.length).toBeGreaterThan(10);
    expect(parsed.hardware.map((h: { id: string }) => h.id)).toContain("h100-sxm");
    const llama = parsed.presets.find((p: { name: string }) => p.name === "llama-3-8b");
    expect(llama.published_params).toBeGreaterThan(8e9);
  });
});

describe("validate", () => {
  test("a preset passes and exits 0", () => {
    const r = cli("validate", "llama-3-8b");
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("llama-3-8b");
  });

  test("a broken design exits 1 and names the bad port", () => {
    const r = cli("validate", BROKEN);
    expect(r.code).toBe(1);
    expect(r.stdout).toContain("error");
    expect(r.stdout).toContain("final_norm");
  });

  test("--json reports the counts", () => {
    const r = cli("validate", BROKEN, "--json");
    expect(r.code).toBe(1);
    const parsed = JSON.parse(r.stdout);
    expect(parsed.ok).toBe(false);
    expect(parsed.counts.error).toBeGreaterThan(0);
    expect(parsed.findings.some((f: { rule: string }) => f.rule === "shape")).toBe(true);
  });

  test("an unreadable path is a usage error, not a crash", () => {
    const r = cli("validate", "does-not-exist.tensorcad.json");
    expect(r.code).toBe(2);
    expect(r.stderr).toContain("neither a preset nor a readable file");
  });
});

describe("analyze", () => {
  test("the text report covers every section", () => {
    const r = cli("analyze", "llama-3-8b", "--T", "8192");
    expect(r.code).toBe(0);
    for (const section of [
      "Configuration",
      "Parameters",
      "FLOPs per token",
      "KV cache",
      "Training memory",
      "Serving memory",
      "Throughput",
      "Training cost",
      "Chinchilla",
    ]) {
      expect(r.stdout).toContain(section);
    }
    expect(r.stdout).toContain("8.03B");
  });

  test("--json agrees with the published parameter count", () => {
    const r = cli("analyze", "llama-3-8b", "--T", "8192", "--json");
    expect(r.code).toBe(0);
    const a = JSON.parse(r.stdout);
    expect(a.params.total).toBe(8_030_261_248);
    expect(a.options.T).toBe(8192);
    expect(a.kv.bytes_per_token).toBe(128 * 1024);
    expect(a.flops.train_per_token).toBeGreaterThan(0);
    expect(a.memory.train.per_gpu.total).toBeGreaterThan(0);
    expect(a.cost.dollars).toBeGreaterThan(0);
  });

  test("flags change the numbers they claim to change", () => {
    const base = JSON.parse(cli("analyze", "llama-3-8b", "--T", "4096", "--json").stdout);
    const longer = JSON.parse(cli("analyze", "llama-3-8b", "--T", "16384", "--json").stdout);
    expect(longer.flops.fwd_attention).toBeGreaterThan(base.flops.fwd_attention);
    expect(longer.memory.infer.kv).toBeGreaterThan(base.memory.infer.kv);

    const sharded = JSON.parse(
      cli("analyze", "llama-3-8b", "--T", "4096", "--zero", "3", "--dp", "8", "--json").stdout,
    );
    expect(sharded.options.gpus).toBe(8);
    expect(sharded.memory.train.per_gpu.optimizer).toBeLessThan(base.memory.train.per_gpu.optimizer);
  });

  test("an unknown hardware id is a usage error", () => {
    const r = cli("analyze", "llama-3-8b", "--hardware", "gtx260");
    expect(r.code).toBe(2);
    expect(r.stderr).toContain("Unknown hardware");
  });
});

describe("show", () => {
  test("prints the tree with shapes, nested into the repeat", () => {
    const r = cli("show", "llama-3-8b");
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("repeat x32");
    expect(r.stdout).toContain("transformer_block");
    expect(r.stdout).toContain("B T D");
    expect(r.stdout).toContain("B T V");
    // The block inside the repeat is indented further than the repeat itself.
    const lines = r.stdout.split("\n");
    const repeat = lines.find((l) => l.includes("repeat x32"))!;
    const block = lines.find((l) => l.includes("transformer_block"))!;
    const indent = (s: string) => s.length - s.trimStart().length;
    expect(indent(block)).toBeGreaterThan(indent(repeat));
  });
});

describe("diff", () => {
  test("reports symbol, block and numeric changes", () => {
    const r = cli("diff", "llama-2-7b", "llama-3-8b");
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("Symbols");
    expect(r.stdout).toContain("Hkv");
    expect(r.stdout).toContain("Numbers");
    expect(r.stdout).toContain("parameters");
  });

  test("--json carries the deltas", () => {
    const r = cli("diff", "llama-2-7b", "llama-3-8b", "--json");
    expect(r.code).toBe(0);
    const d = JSON.parse(r.stdout);
    expect(d.identical).toBe(false);
    expect(d.symbols.changed.map((s: { name: string }) => s.name)).toContain("Hkv");
    const params = d.metrics.find((m: { metric: string }) => m.metric === "parameters");
    expect(params.a).toBeLessThan(params.b);
    expect(params.delta).toBe(params.b - params.a);
    const kv = d.metrics.find((m: { metric: string }) => m.metric === "KV bytes/token");
    // Llama 3 swapped MHA for 8-way GQA: a quarter of the cache.
    expect(kv.b).toBeLessThan(kv.a);
    // Both sides are measured at one sequence length.
    expect(d.at.T).toBe(8192);
  });

  test("a design against itself is identical", () => {
    const d = JSON.parse(cli("diff", "gpt2-small", "gpt2-small", "--json").stdout);
    expect(d.identical).toBe(true);
    expect(d.metrics.every((m: { delta: number }) => m.delta === 0)).toBe(true);
  });

  test("one argument is a usage error", () => {
    const r = cli("diff", "gpt2-small");
    expect(r.code).toBe(2);
    expect(r.stderr).toContain("two designs");
  });
});

describe("codegen", () => {
  test("writes the files it reports", () => {
    const out = tempDir();
    const r = cli("codegen", "gpt2-small", "--out", out);
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("model.py");
    const model = join(out, "model.py");
    expect(existsSync(model)).toBe(true);
    const contents = readFileSync(model, "utf8");
    expect(contents).toContain("import torch");
    expect(contents).toContain("class ");
  });

  test("--json returns the files without touching the disk", () => {
    const r = cli("codegen", "gpt2-small", "--json");
    expect(r.code).toBe(0);
    const parsed = JSON.parse(r.stdout);
    expect(parsed.files.some((f: { path: string }) => f.path === "model.py")).toBe(true);
    expect(Array.isArray(parsed.warnings)).toBe(true);
  });
});

describe("plan", () => {
  test("lists ways to split the work, least demanding first", () => {
    const r = cli("plan", "llama-3-70b", "--gpus", "64", "--T", "8192", "--hardware", "h100-sxm");
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("llama-3-70b on 64 x H100 SXM");
    expect(r.stdout).toContain("per device after headroom");
    expect(r.stdout).toContain("plans priced");
    // Every plan names its split and what it costs to hold.
    expect(r.stdout).toMatch(/DP \d+/);
    expect(r.stdout).toMatch(/\d+% of budget/);
  });

  test("--json carries the plans and the budget", () => {
    const r = cli("plan", "mixtral-8x7b", "--gpus", "64", "--T", "4096", "--hardware", "h100-sxm", "--json");
    expect(r.code).toBe(0);
    const out = JSON.parse(r.stdout) as {
      fits: { parallel: { dp: number; tp: number; pp: number; ep: number }; used: number; summary: string }[];
      budget: number;
      considered: number;
    };
    expect(out.fits.length).toBeGreaterThan(0);
    expect(out.considered).toBeGreaterThan(0);
    for (const f of out.fits) {
      const { dp, tp, pp, ep } = f.parallel;
      expect({ summary: f.summary, devices: dp * tp * pp * ep }).toEqual({
        summary: f.summary,
        devices: 64,
      });
      expect(f.used).toBeLessThanOrEqual(1);
    }
  });

  test("exits 1 when nothing fits, and says what came closest", () => {
    const r = cli("plan", "llama-3.1-405b", "--gpus", "8", "--T", "8192", "--hardware", "h100-sxm");
    expect(r.code).toBe(1);
    expect(r.stdout).toContain("Nothing fits");
    expect(r.stdout).toContain("closest");
  });

  test("needs to be told how big the cluster is", () => {
    const r = cli("plan", "gpt2-small");
    expect(r.code).toBe(1);
    expect(r.stdout).toContain("--gpus");
  });
});

describe("import", () => {
  const configs = JSON.parse(
    readFileSync(resolve(import.meta.dir, "../../core-go/testdata/hf-configs.json"), "utf8"),
  ) as Record<string, Record<string, unknown>>;

  /** A config on disk, in a directory named after the model the way a download is. */
  function config(name: string, body: unknown = configs[name]): string {
    const dir = join(tempDir(), name);
    mkdirSync(dir, { recursive: true });
    const path = join(dir, "config.json");
    writeFileSync(path, JSON.stringify(body, null, 2));
    return path;
  }

  test("lands on the same parameter count as the preset", () => {
    const r = cli("import", config("llama-3-8b"));
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("8.03B parameters");
    // The name comes from the directory, which is what a model download is
    // called; "config" would not be.
    expect(r.stdout).toContain("llama-3-8b");
  });

  test("--json gives the document, and the warnings stay on stderr", () => {
    const r = cli("import", config("mixtral-8x7b"), "--json");
    expect(r.code).toBe(0);
    const out = JSON.parse(r.stdout) as { doc: { meta: { name: string } }; params: number };
    expect(out.doc.meta.name).toBe("mixtral-8x7b");
    expect(out.params).toBe(46_702_792_704);
  });

  test("--out writes a design that can be read back", () => {
    const dir = tempDir();
    const r = cli("import", config("gpt2-small"), "--out", dir, "--name", "from-config");
    expect(r.code).toBe(0);
    const path = join(dir, "from-config.tensorcad.json");
    expect(existsSync(path)).toBe(true);
    const back = cli("analyze", path, "--json");
    expect(back.code).toBe(0);
    expect((JSON.parse(back.stdout) as { params: { total: number } }).params.total).toBe(124_439_808);
  });

  test("a family it does not know is refused by name", () => {
    const r = cli("import", config("mystery", { model_type: "not-a-model", num_hidden_layers: 4 }));
    expect(r.code).not.toBe(0);
    expect(`${r.stdout}${r.stderr}`).toContain("not-a-model");
  });

  test("a file that is not JSON is refused readably", () => {
    const path = join(tempDir(), "broken.json");
    writeFileSync(path, "{ oops");
    const r = cli("import", path);
    expect(r.code).not.toBe(0);
    expect(`${r.stdout}${r.stderr}`.toLowerCase()).toContain("parse");
  });

  test("a file that is not there says so", () => {
    const r = cli("import", join(tempDir(), "absent.json"));
    expect(r.code).not.toBe(0);
    expect(`${r.stdout}${r.stderr}`).toContain("Could not read");
  });
});
