/**
 * CLI contract tests.
 *
 * These spawn the real entry point so the exit codes are covered too: a broken
 * design must fail CI, a good one must not.
 */

import { describe, expect, test, afterAll } from "bun:test";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, existsSync, writeFileSync } from "node:fs";
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

describe("import", () => {
  // The same corpus the importer's own test uses, so a config that changes
  // there cannot leave a stale copy asserted against here.
  const CONFIGS = JSON.parse(
    readFileSync(resolve(import.meta.dir, "../../core-go/testdata/hf-configs.json"), "utf8"),
  ) as Record<string, Record<string, unknown>>;

  function configFile(dir: string, config: unknown, name = "config.json"): string {
    const path = join(dir, name);
    writeFileSync(path, JSON.stringify(config, null, 2), "utf8");
    return path;
  }

  test("writes a document whose count matches the preset the config describes", () => {
    const dir = tempDir();
    const out = join(dir, "gpt2.tensorcad.json");
    const r = cli("import", configFile(dir, CONFIGS["gpt2-small"]), "--out", out);
    expect(r.code).toBe(0);
    // The count is printed both ways: rounded to read, exact to compare.
    expect(r.stdout).toContain("124.4M");
    expect(r.stdout).toContain("124,439,808");

    expect(existsSync(out)).toBe(true);
    const doc = JSON.parse(readFileSync(out, "utf8"));
    expect(doc.version).toBe(1);
    expect(doc.graph.nodes.length).toBeGreaterThan(0);

    // And it is a design the rest of the command line accepts.
    const analyzed = JSON.parse(cli("analyze", out, "--json").stdout);
    expect(analyzed.params.total).toBe(124_439_808);
  });

  test("--name names the design; without it the config does", () => {
    const dir = tempDir();
    const config = configFile(dir, { ...CONFIGS["gpt2-small"], _name_or_path: "openai-community/gpt2" });

    const named = cli("import", config, "--name", "my-gpt", "--out", join(dir, "named.json"));
    expect(named.code).toBe(0);
    expect(JSON.parse(readFileSync(join(dir, "named.json"), "utf8")).meta.name).toBe("my-gpt");

    const unnamed = cli("import", config, "--out", join(dir, "unnamed.json"));
    expect(unnamed.code).toBe(0);
    expect(JSON.parse(readFileSync(join(dir, "unnamed.json"), "utf8")).meta.name).toBe(
      "openai-community/gpt2",
    );
  });

  test("reports what the import could not represent rather than swallowing it", () => {
    const dir = tempDir();
    // DeepSeek-V3's own config carries a multi-token-prediction module, which
    // the importer leaves out and says so.
    const config = configFile(dir, { ...CONFIGS["deepseek-v3"], num_nextn_predict_layers: 1 });
    const r = cli("import", config, "--out", join(dir, "ds.json"));
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("warning");
    expect(r.stdout).toContain("multi-token-prediction");
    // The document is still written, and still the design the preset is.
    expect(r.stdout).toContain("671,026,419,200");
  });

  test("--json carries the document without touching the disk", () => {
    const dir = tempDir();
    const out = join(dir, "not-written.json");
    const r = cli("import", configFile(dir, CONFIGS["llama-3-8b"]), "--out", out, "--json");
    expect(r.code).toBe(0);
    expect(existsSync(out)).toBe(false);
    const parsed = JSON.parse(r.stdout);
    expect(parsed.params).toBe(8_030_261_248);
    expect(parsed.warnings).toEqual([]);
    expect(parsed.doc.meta.name).toBe(parsed.name);
  });

  test("an unsupported model_type names the ones it knows", () => {
    const dir = tempDir();
    const r = cli("import", configFile(dir, { model_type: "mamba", num_hidden_layers: 4 }));
    expect(r.code).toBe(2);
    expect(r.stderr).toContain("unsupported model_type");
    expect(r.stderr).toContain("llama");
  });

  test("a missing config is a usage error, not a crash", () => {
    const r = cli("import", "no-such-config.json");
    expect(r.code).toBe(2);
    expect(r.stderr).toContain("Could not read");
  });
});
