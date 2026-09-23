/**
 * MCP contract tests.
 *
 * These drive the real server over a real stdio pipe with the SDK's own client,
 * and every tool result is validated against the `outputSchema` the server
 * advertised, so the declared contract and the actual answer cannot drift.
 */

import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { AjvJsonSchemaValidator } from "@modelcontextprotocol/client/validators/ajv";
import type { CallToolResult, Tool } from "@modelcontextprotocol/client";

const ENTRY = resolve(import.meta.dir, "../src/stdio.ts");

let client: Client;
let tools: Tool[];
let workDir: string;
const validator = new AjvJsonSchemaValidator();

beforeAll(async () => {
  workDir = mkdtempSync(join(tmpdir(), "tensorcad-mcp-"));
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: ["run", ENTRY],
    cwd: workDir,
    env: { ...process.env, TENSORCAD_ROOT: workDir } as Record<string, string>,
    stderr: "pipe",
  });
  client = new Client({ name: "tensorcad-test", version: "0.0.1" });
  await client.connect(transport);
  tools = (await client.listTools()).tools;
}, 30_000);

afterAll(async () => {
  await client?.close();
  if (workDir) rmSync(workDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------

/** Call a tool and assert its `structuredContent` matches the declared `outputSchema`. */
async function call(name: string, args: Record<string, unknown> = {}): Promise<Record<string, any>> {
  return (await callFull(name, args)).data;
}

/** As `call`, but also hands back the text mirror that lands in the model's context. */
async function callFull(
  name: string,
  args: Record<string, unknown> = {},
): Promise<{ data: Record<string, any>; text: string }> {
  const result = (await client.callTool({ name, arguments: args })) as CallToolResult;
  const text = (result.content ?? []).map((c: any) => (c.type === "text" ? c.text : "")).join("\n");
  if (result.isError) throw new Error(`${name} returned isError: ${text}`);

  const tool = tools.find((t) => t.name === name);
  expect(tool, `tool ${name} is not advertised`).toBeDefined();
  expect(tool!.outputSchema, `tool ${name} declares no outputSchema`).toBeDefined();
  expect(result.structuredContent, `tool ${name} returned no structuredContent`).toBeDefined();

  const check = validator.getValidator(tool!.outputSchema as Record<string, unknown>);
  const verdict = check(result.structuredContent);
  expect(verdict.valid ? "" : `${name}: ${verdict.errorMessage}`).toBe("");

  // Every tool also mirrors its answer as readable text.
  expect(text.length).toBeGreaterThan(0);
  return { data: result.structuredContent as Record<string, any>, text };
}

/** The text of a resource's first content block. */
function resourceText(result: { contents: unknown[] }): string {
  const first = result.contents[0] as { text?: string };
  expect(first.text, "resource returned no text").toBeDefined();
  return first.text!;
}

/** Call a tool expecting a model-correctable failure rather than a protocol error. */
async function callExpectingError(name: string, args: Record<string, unknown>): Promise<string> {
  const result = (await client.callTool({ name, arguments: args })) as CallToolResult;
  const text = (result.content ?? []).map((c: any) => (c.type === "text" ? c.text : "")).join("\n");
  expect(result.isError, `${name} was expected to fail but returned: ${text}`).toBe(true);
  return text;
}

async function newLlama(): Promise<{ id: string; revision: number; params: number }> {
  const r = await call("tensorcad_new_design", { preset: "llama-3-8b" });
  return { id: r.design_id, revision: r.revision, params: r.params_total };
}

// ---------------------------------------------------------------------------

describe("handshake", () => {
  test("connects and reports itself", () => {
    expect(client.getServerVersion()?.name).toBe("tensorcad");
    expect(client.getInstructions()).toContain("design_id");
  });

  // The cap is against sprawl, not a budget to spend: every tool costs an agent
  // context on every turn whether or not it is used, so one that overlaps
  // another should be merged rather than added. Twenty is the room for the
  // engine's distinct capabilities and no more; if it is ever reached, the
  // question to ask is which two of them are the same tool.
  test("advertises at most twenty tools, every one namespaced and schema'd", () => {
    expect(tools.length).toBeGreaterThan(0);
    expect(tools.length).toBeLessThanOrEqual(20);
    for (const tool of tools) {
      expect(tool.name).toStartWith("tensorcad_");
      expect(tool.description).toBeTruthy();
      expect(tool.inputSchema).toBeDefined();
      expect(tool.outputSchema, `${tool.name} has no outputSchema`).toBeDefined();
      expect(tool.annotations, `${tool.name} has no annotations`).toBeDefined();
    }
  });

  test("reads are marked read-only and idempotent, deletes destructive", () => {
    const byName = new Map(tools.map((t) => [t.name, t.annotations!]));
    for (const name of [
      "tensorcad_list_designs",
      "tensorcad_get_design",
      "tensorcad_get_block",
      "tensorcad_search_catalog",
      "tensorcad_validate",
      "tensorcad_analyze",
    ]) {
      expect(byName.get(name)?.readOnlyHint, name).toBe(true);
      expect(byName.get(name)?.idempotentHint, name).toBe(true);
    }
    for (const name of ["tensorcad_apply_ops", "tensorcad_new_design", "tensorcad_open_design"]) {
      expect(byName.get(name)?.readOnlyHint, name).not.toBe(true);
    }
    for (const name of ["tensorcad_save_design", "tensorcad_restore"]) {
      expect(byName.get(name)?.destructiveHint, name).toBe(true);
    }
  });

  test("the tool order is stable across listings", async () => {
    const again = (await client.listTools()).tools.map((t) => t.name);
    expect(again).toEqual(tools.map((t) => t.name));
  });
});

describe("annotations", () => {
  // Read off the protocol, as a client sees them, rather than out of the source.
  const hints = (name: string) => tools.find((t) => t.name === name)?.annotations ?? {};

  test("every tool states all four hints", () => {
    // `destructiveHint` defaults to true for anything that is not read-only, so
    // leaving it out is not neutral: it tells a client the tool may destroy
    // something. Seventeen of nineteen did, including "New design".
    const missing: string[] = [];
    for (const tool of tools) {
      const a = tool.annotations ?? {};
      for (const key of ["readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"] as const) {
        if (typeof a[key] !== "boolean") missing.push(`${tool.name}.${key}`);
      }
    }
    expect(missing).toEqual([]);
  });

  test("nothing read-only claims to destroy anything", () => {
    for (const tool of tools) {
      const a = tool.annotations ?? {};
      if (a.readOnlyHint) expect({ tool: tool.name, destructive: a.destructiveHint }).toEqual({
        tool: tool.name,
        destructive: false,
      });
    }
  });

  test("the hints that were wrong say the true thing", () => {
    // Every call adopts a new design under a new id, so none of these is
    // idempotent. They all said they were.
    for (const name of ["tensorcad_scale", "tensorcad_mup", "tensorcad_import_hf"]) {
      expect({ name, idempotent: hints(name).idempotentHint }).toEqual({ name, idempotent: false });
      expect({ name, destructive: hints(name).destructiveHint }).toEqual({ name, destructive: false });
    }
    // Writes into out_dir with an unconditional writeFile: a hand-edited
    // model.py there is gone. The same arguments write the same bytes.
    expect(hints("tensorcad_generate_code").destructiveHint).toBe(true);
    expect(hints("tensorcad_generate_code").idempotentHint).toBe(true);
    // The same path twice is the same handle, as the description promises.
    expect(hints("tensorcad_open_design").idempotentHint).toBe(true);
    // Creating a design destroys nothing.
    expect(hints("tensorcad_new_design").destructiveHint).toBe(false);
  });

  test("and the behaviour agrees with the hint", async () => {
    // Not idempotent means a second identical call has an effect. For scale
    // the effect is a second design, which is what this looks for.
    const llama = await newLlama();
    const args = {
      design_id: llama.id,
      target_params: 30e6,
      target_basis: "non-embedding",
      vocab: 8192,
      tie_head: true,
    };
    const first = await call("tensorcad_scale", args);
    const second = await call("tensorcad_scale", args);
    expect(second.design_id).not.toBe(first.design_id);
  });
});

describe("designs", () => {
  test("creating from a preset gives a handle at revision 1", async () => {
    const r = await call("tensorcad_new_design", { preset: "llama-3-8b" });
    expect(r.design_id).toMatch(/^dsn_/);
    expect(r.revision).toBe(1);
    expect(r.source).toBe("preset");
    expect(r.params_total).toBe(8_030_261_248);
    expect(r.outline.name).toBe("llama-3-8b");
  });

  test("an unknown preset is a readable error, not a crash", async () => {
    const text = await callExpectingError("tensorcad_new_design", { preset: "llama-9-900b" });
    expect(text).toContain("Unknown preset");
    expect(text).toContain("llama-3-8b");
  });

  test("list_designs sees the open designs and the presets", async () => {
    const created = await newLlama();
    const r = await call("tensorcad_list_designs", {});
    expect(r.designs.map((d: any) => d.design_id)).toContain(created.id);
    expect(r.presets.map((p: any) => p.name)).toContain("llama-3-8b");
    expect(r.files).toEqual([]);
  });

  test("the outline carries the structure, the symbols and the edge shapes", async () => {
    const { id } = await newLlama();
    const r = await call("tensorcad_get_design", { design_id: id, format: "outline" });
    expect(r.format).toBe("outline");

    const o = r.outline;
    expect(o.symbols.find((s: any) => s.name === "D").resolved).toBe(4096);
    expect(o.symbols.find((s: any) => s.name === "T").kind).toBe("runtime");

    const paths = o.blocks.map((b: any) => b.path);
    expect(paths).toContain("embed");
    expect(paths).toContain("layers");
    expect(paths).toContain("layers/block");

    const repeat = o.blocks.find((b: any) => b.path === "layers");
    expect(repeat.kind).toBe("container");
    expect(repeat.repeat).toBe(32);
    expect(o.blocks.find((b: any) => b.path === "layers/block").depth).toBeGreaterThan(repeat.depth);

    const edge = o.edges.find((e: any) => e.from === "embed:y");
    expect(edge.shape).toBe("B T D");
    expect(o.issues).toBe(0);

    // The outline really is the cheaper read: what lands in context is the text
    // mirror, and the outline's is a fraction of the document's.
    const outlineCall = await callFull("tensorcad_get_design", { design_id: id, format: "outline" });
    const fullCall = await callFull("tensorcad_get_design", { design_id: id, format: "full" });
    expect(fullCall.data.document.meta.name).toBe("llama-3-8b");
    expect(outlineCall.text.length).toBeLessThan(fullCall.text.length / 2);
  });

  test("an unknown design_id names the handles that do exist", async () => {
    const text = await callExpectingError("tensorcad_get_design", { design_id: "dsn_nope" });
    expect(text).toContain("Unknown design_id");
    expect(text).toContain("tensorcad_new_design");
  });

  test("get_block reports params, shapes and wiring", async () => {
    const { id } = await newLlama();
    const r = await call("tensorcad_get_block", { design_id: id, path: "layers/block" });
    expect(r.type).toBe("transformer_block");
    expect(r.kind).toBe("composite");
    expect(r.instances).toBe(32);
    expect(r.resolved_params.d_model).toBe(4096);
    expect(r.resolved_params.kv_heads).toBe(8);
    expect(r.params_count).toBeGreaterThan(6e9);
    expect(r.inputs.find((p: any) => p.name === "x").shape).toBe("B T D");
    expect(r.outputs.find((p: any) => p.name === "y").connected_to).toContain("layers/_out:x");
  });

  test("get_block on a missing path lists the blocks that are there", async () => {
    const { id } = await newLlama();
    const text = await callExpectingError("tensorcad_get_block", { design_id: id, path: "layers/nope" });
    expect(text).toContain("No block");
    expect(text).toContain("block");
  });

  test("open_design reads a file written by save_design", async () => {
    const { id } = await newLlama();
    const target = join(workDir, "saved.tensorcad.json");
    const saved = await call("tensorcad_save_design", { design_id: id, path: target });
    expect(saved.path).toBe(target);
    expect(saved.bytes).toBeGreaterThan(100);
    expect(existsSync(target)).toBe(true);
    expect(JSON.parse(readFileSync(target, "utf8")).meta.name).toBe("llama-3-8b");

    const opened = await call("tensorcad_open_design", { path: target });
    expect(opened.name).toBe("llama-3-8b");
    expect(opened.params_total).toBe(8_030_261_248);
    expect(opened.validation.ok).toBe(true);

    const listed = await call("tensorcad_list_designs", { include_files: true });
    expect(listed.files).toContain(target);
  });

  test("opening something that is not a design says so", async () => {
    const bad = join(workDir, "not-a-design.tensorcad.json");
    writeFileSync(bad, JSON.stringify({ hello: "world" }), "utf8");
    const text = await callExpectingError("tensorcad_open_design", { path: bad });
    expect(text).toContain("does not look like a design document");
  });
});

describe("catalog", () => {
  test("search returns parameter schemas and port patterns", async () => {
    const r = await call("tensorcad_search_catalog", { query: "attention" });
    expect(r.total).toBeGreaterThan(0);
    const gqa = r.blocks.find((b: any) => b.type === "gqa_attention");
    expect(gqa.kind).toBe("composite");
    expect(gqa.params.map((p: any) => p.name)).toContain("kv_heads");
    expect(gqa.summary.length).toBeGreaterThan(10);
  });

  test("filters compose and the limit is honoured", async () => {
    const all = await call("tensorcad_search_catalog", { limit: 100 });
    const norms = await call("tensorcad_search_catalog", { category: "norm", limit: 100 });
    expect(norms.total).toBeGreaterThan(0);
    expect(norms.total).toBeLessThan(all.total);
    expect(norms.blocks.every((b: any) => b.category === "norm")).toBe(true);

    const one = await call("tensorcad_search_catalog", { limit: 1 });
    expect(one.blocks.length).toBe(1);
    expect(one.total).toBeGreaterThan(1);
    expect(one.categories.length).toBeGreaterThan(1);
  });
});

describe("editing", () => {
  test("a set_symbol batch moves the parameter count and the revision", async () => {
    const { id, revision, params } = await newLlama();

    const r = await call("tensorcad_apply_ops", {
      design_id: id,
      expected_revision: revision,
      ops: [
        { op: "set_symbol", name: "L", value: 40 },
        { op: "set_symbol", name: "D", value: 5120 },
      ],
    });

    expect(r.previous_revision).toBe(revision);
    expect(r.revision).toBe(revision + 1);
    expect(r.applied.length).toBe(2);
    expect(r.applied[0]).toContain("L");
    expect(r.params_total).toBeGreaterThan(params);
    expect(r.params_delta).toBe(r.params_total - params);
    expect(r.validation.counts.error).toBe(0);

    // The read path agrees with what the write path reported.
    const after = await call("tensorcad_get_design", { design_id: id });
    expect(after.revision).toBe(r.revision);
    expect(after.params_total).toBe(r.params_total);
    expect(after.outline.symbols.find((s: any) => s.name === "D").resolved).toBe(5120);
    expect(after.dirty).toBe(true);
  });

  test("a stale expected_revision is rejected and nothing changes", async () => {
    const { id, revision, params } = await newLlama();
    await call("tensorcad_apply_ops", {
      design_id: id,
      expected_revision: revision,
      ops: [{ op: "set_symbol", name: "L", value: 40 }],
    });

    const text = await callExpectingError("tensorcad_apply_ops", {
      design_id: id,
      expected_revision: revision, // now stale
      ops: [{ op: "set_symbol", name: "L", value: 48 }],
    });
    expect(text).toContain(`revision ${revision + 1}`);
    expect(text).toContain("tensorcad_get_design");

    const after = await call("tensorcad_get_design", { design_id: id });
    expect(after.revision).toBe(revision + 1);
    expect(after.outline.symbols.find((s: any) => s.name === "L").resolved).toBe(40);
    expect(after.params_total).toBeGreaterThan(params);
  });

  test("omitting expected_revision still works", async () => {
    const { id } = await newLlama();
    const r = await call("tensorcad_apply_ops", {
      design_id: id,
      ops: [{ op: "set_symbol", name: "V", value: 32000 }],
    });
    expect(r.revision).toBe(2);
    expect(r.params_delta).toBeLessThan(0);
  });

  test("a batch is all-or-nothing", async () => {
    const { id, revision, params } = await newLlama();
    const text = await callExpectingError("tensorcad_apply_ops", {
      design_id: id,
      ops: [
        { op: "set_symbol", name: "L", value: 64 },
        { op: "remove_node", path: "no_such_block" },
      ],
    });
    expect(text).toContain("op 1");
    expect(text).toContain("no_such_block");

    const after = await call("tensorcad_get_design", { design_id: id });
    expect(after.revision).toBe(revision);
    expect(after.params_total).toBe(params);
  });

  test("rename rewrites edges, and label and set_param land", async () => {
    const { id } = await newLlama();
    const r = await call("tensorcad_apply_ops", {
      design_id: id,
      ops: [
        { op: "rename", path: "final_norm", id: "out_norm" },
        { op: "set_label", path: "out_norm", label: "final RMSNorm" },
        { op: "set_param", path: "layers/block", key: "kv_heads", value: 4 },
      ],
    });
    expect(r.applied.length).toBe(3);

    const after = await call("tensorcad_get_design", { design_id: id });
    const paths = after.outline.blocks.map((b: any) => b.path);
    expect(paths).toContain("out_norm");
    expect(paths).not.toContain("final_norm");
    expect(after.outline.blocks.find((b: any) => b.path === "out_norm").label).toBe("final RMSNorm");
    // Renaming rewrote the edges rather than orphaning them.
    expect(after.outline.edges.some((e: any) => e.to === "out_norm:x")).toBe(true);
    expect(after.outline.edges.some((e: any) => e.from === "out_norm:y")).toBe(true);

    const block = await call("tensorcad_get_block", { design_id: id, path: "layers/block" });
    expect(block.resolved_params.kv_heads).toBe(4);
  });

  test("adding and connecting a block into a container", async () => {
    const r = await call("tensorcad_new_design", { name: "scratch" });
    const id = r.design_id;
    await call("tensorcad_apply_ops", {
      design_id: id,
      ops: [
        { op: "set_symbol", name: "D", value: 256 },
        { op: "set_symbol", name: "V", value: 1024 },
        { op: "add_node", id: "tokens", type: "input", params: { shape: "B T", dtype: "int64" } },
        { op: "add_node", id: "embed", type: "embedding", params: { vocab: "V", dim: "D" } },
        { op: "add_node", id: "out", type: "output" },
        { op: "connect", from: "tokens:x", to: "embed:ids" },
        { op: "connect", from: "embed:y", to: "out:x" },
      ],
    });

    const after = await call("tensorcad_get_design", { design_id: id });
    expect(after.outline.blocks.map((b: any) => b.path)).toEqual(["tokens", "embed", "out"]);
    expect(after.params_total).toBe(256 * 1024);
    expect(after.outline.edges.find((e: any) => e.from === "embed:y").shape).toBe("B T D");
  });

  test("an unknown block type and a double connection are both refused", async () => {
    const r = await call("tensorcad_new_design", { name: "scratch2" });
    const unknown = await callExpectingError("tensorcad_apply_ops", {
      design_id: r.design_id,
      ops: [{ op: "add_node", id: "x", type: "quantum_attention" }],
    });
    expect(unknown).toContain("unknown block type");

    const { id } = await newLlama();
    const twice = await callExpectingError("tensorcad_apply_ops", {
      design_id: id,
      ops: [{ op: "connect", from: "embed:y", to: "layers:x" }],
    });
    expect(twice).toContain("already");
  });
});

describe("checking and costing", () => {
  test("validate reports a clean preset and the rules that fired", async () => {
    const { id } = await newLlama();
    const r = await call("tensorcad_validate", { design_id: id });
    expect(r.ok).toBe(true);
    expect(r.counts.error).toBe(0);
    expect(r.params_total).toBe(8_030_261_248);
    for (const f of r.findings) expect(["error", "warning", "info"]).toContain(f.severity);
  });

  test("validate finds a shape error the moment one is introduced", async () => {
    const { id } = await newLlama();
    await call("tensorcad_apply_ops", {
      design_id: id,
      ops: [{ op: "set_param", path: "final_norm", key: "dim", value: 999 }],
    });
    const r = await call("tensorcad_validate", { design_id: id });
    expect(r.ok).toBe(false);
    expect(r.counts.error).toBeGreaterThan(0);
    const shape = r.findings.find((f: any) => f.rule === "shape");
    expect(shape.severity).toBe("error");
    expect(shape.path).toBe("final_norm");
  });

  test("validate honours the hardware and severity filters", async () => {
    const { id } = await newLlama();
    const small = await call("tensorcad_validate", { design_id: id, hardware: "rtx5080" });
    expect(small.counts.warning + small.counts.error).toBeGreaterThan(0);

    const errorsOnly = await call("tensorcad_validate", { design_id: id, hardware: "rtx5080", severity: "error" });
    expect(errorsOnly.findings.every((f: any) => f.severity === "error")).toBe(true);

    const bad = await callExpectingError("tensorcad_validate", { design_id: id, hardware: "gtx260" });
    expect(bad).toContain("Unknown hardware");
  });

  test("analyze produces every number, and the knobs move them", async () => {
    const { id } = await newLlama();
    const r = await call("tensorcad_analyze", { design_id: id, T: 8192 });

    expect(r.options.T).toBe(8192);
    expect(r.options.hardware).toBe("h100-sxm");
    expect(r.params.total).toBe(8_030_261_248);
    expect(r.kv.bytes_per_token).toBe(128 * 1024);
    expect(r.kv.bytes_per_sequence).toBe(128 * 1024 * 8192);
    expect(r.flops.train_per_token).toBeGreaterThan(r.flops.fwd_total!);
    expect(r.memory.train_per_gpu).toBeGreaterThan(r.memory.infer_total!);
    expect(r.throughput.decode_tokens_per_second).toBeGreaterThan(0);
    expect(r.cost.dollars).toBeGreaterThan(0);
    expect(r.chinchilla.tokens_per_param).toBeGreaterThan(0);
    expect(r.errors).toEqual([]);

    const longer = await call("tensorcad_analyze", { design_id: id, T: 32768 });
    expect(longer.flops.fwd_attention).toBeGreaterThan(r.flops.fwd_attention!);

    const sharded = await call("tensorcad_analyze", { design_id: id, T: 8192, zero: 3, dp: 8 });
    expect(sharded.options.gpus).toBe(8);
    expect(sharded.memory.train_optimizer).toBeLessThan(r.memory.train_optimizer!);
  });

  test("generate_code returns runnable files, and writes them when asked", async () => {
    const r = await call("tensorcad_new_design", { preset: "gpt2-small" });
    const inline = await call("tensorcad_generate_code", { design_id: r.design_id });
    expect(inline.wrote).toBe(false);
    const model = inline.files.find((f: any) => f.path === "model.py");
    expect(model.contents).toContain("import torch");
    expect(model.lines).toBeGreaterThan(20);

    const outDir = join(workDir, "generated");
    const written = await call("tensorcad_generate_code", {
      design_id: r.design_id,
      out_dir: outDir,
      class_name: "TinyGPT",
    });
    expect(written.wrote).toBe(true);
    expect(written.out_dir).toBe(outDir);
    for (const f of written.files) {
      expect(f.contents).toBeUndefined();
      expect(existsSync(f.written_to)).toBe(true);
    }
    expect(readFileSync(join(outDir, "model.py"), "utf8")).toContain("TinyGPT");
  });
});

describe("every reference architecture", () => {
  test("outlines, validates, analyses and generates within the declared schemas", async () => {
    const { presets } = await call("tensorcad_list_designs", {});
    expect(presets.length).toBeGreaterThan(5);

    for (const preset of presets as { name: string; published_params?: number }[]) {
      const created = await call("tensorcad_new_design", { preset: preset.name });
      const id = created.design_id;

      // Sparse models report fewer active parameters than total; dense ones equal.
      expect(created.outline.params_active).toBeLessThanOrEqual(created.outline.params_total);
      if (preset.published_params) {
        const drift = Math.abs(created.params_total - preset.published_params) / preset.published_params;
        expect(drift, `${preset.name} drifted from its published count`).toBeLessThan(0.05);
      }

      await call("tensorcad_get_design", { design_id: id });
      const report = await call("tensorcad_validate", { design_id: id });
      expect(report.counts.error, `${preset.name} has validation errors`).toBe(0);

      // The numbers must stay JSON-representable: no Infinity, no NaN.
      const a = await call("tensorcad_analyze", { design_id: id, T: 4096 });
      expect(a.errors, preset.name).toEqual([]);
      expect(a.params.total).toBeGreaterThan(0);
      expect(a.cost.dollars).toBeGreaterThan(0);

      await call("tensorcad_generate_code", { design_id: id });
    }
  }, 60_000);
});

describe("checkpoints", () => {
  test("checkpoint then restore puts the design back", async () => {
    const { id, params } = await newLlama();
    const cp = await call("tensorcad_checkpoint", { design_id: id, label: "pristine llama" });
    expect(cp.checkpoint_id).toMatch(/^ckpt_/);
    expect(cp.label).toBe("pristine llama");
    expect(cp.checkpoints.length).toBeGreaterThan(0);

    await call("tensorcad_apply_ops", {
      design_id: id,
      ops: [
        { op: "set_symbol", name: "L", value: 80 },
        { op: "set_symbol", name: "D", value: 8192 },
      ],
    });
    const grown = await call("tensorcad_get_design", { design_id: id });
    expect(grown.params_total).toBeGreaterThan(params * 2);

    const restored = await call("tensorcad_restore", { design_id: id, checkpoint_id: cp.checkpoint_id });
    expect(restored.params_total).toBe(params);
    expect(restored.restored_from).toContain(cp.checkpoint_id);
    // Restoring moves the revision forward, so a stale expected_revision still fails.
    expect(restored.revision).toBeGreaterThan(grown.revision);
  });

  test("restore without a checkpoint undoes the last batch only", async () => {
    const { id, params } = await newLlama();
    await call("tensorcad_apply_ops", { design_id: id, ops: [{ op: "set_symbol", name: "L", value: 40 }] });
    const middle = (await call("tensorcad_get_design", { design_id: id })).params_total;
    await call("tensorcad_apply_ops", { design_id: id, ops: [{ op: "set_symbol", name: "D", value: 6144 }] });

    const undone = await call("tensorcad_restore", { design_id: id });
    expect(undone.params_total).toBe(middle);
    expect(undone.params_total).not.toBe(params);
    expect(undone.restored_from).toContain("undo");

    const again = await call("tensorcad_restore", { design_id: id });
    expect(again.params_total).toBe(params);

    const empty = await callExpectingError("tensorcad_restore", { design_id: id });
    expect(empty).toContain("no edits to undo");
  });

  test("an unknown checkpoint lists the ones that exist", async () => {
    const { id } = await newLlama();
    const cp = await call("tensorcad_checkpoint", { design_id: id });
    const text = await callExpectingError("tensorcad_restore", { design_id: id, checkpoint_id: "ckpt_999" });
    expect(text).toContain("Unknown checkpoint");
    expect(text).toContain(cp.checkpoint_id);
  });
});

describe("resources", () => {
  test("the catalog and the document schema are readable", async () => {
    const { resources } = await client.listResources();
    const uris = resources.map((r) => r.uri);
    expect(uris).toContain("tensorcad://catalog");
    expect(uris).toContain("tensorcad://schema/design");

    const parsed = JSON.parse(resourceText(await client.readResource({ uri: "tensorcad://catalog" })));
    expect(parsed.count).toBeGreaterThan(10);
    expect(parsed.blocks.some((b: any) => b.type === "rmsnorm")).toBe(true);

    const doc = JSON.parse(resourceText(await client.readResource({ uri: "tensorcad://schema/design" })));
    expect(doc.$id).toBe("tensorcad://schema/design");
    expect(doc.required).toContain("graph");
  });

  test("templates cover the design, its validation and its analysis", async () => {
    const { resourceTemplates } = await client.listResourceTemplates();
    const templates = resourceTemplates.map((t) => t.uriTemplate);
    expect(templates).toContain("tensorcad://designs/{id}");
    expect(templates).toContain("tensorcad://designs/{id}/validation");
    expect(templates).toContain("tensorcad://designs/{id}/analysis");
    expect(templates).toContain("tensorcad://catalog/{type}");

    const { id } = await newLlama();
    const design = JSON.parse(resourceText(await client.readResource({ uri: `tensorcad://designs/${id}` })));
    expect(design.document.meta.name).toBe("llama-3-8b");
    expect(design.outline.params_total).toBe(8_030_261_248);

    const validation = JSON.parse(
      resourceText(await client.readResource({ uri: `tensorcad://designs/${id}/validation` })),
    );
    expect(validation.ok).toBe(true);
    expect(Array.isArray(validation.findings)).toBe(true);

    const analysis = JSON.parse(
      resourceText(await client.readResource({ uri: `tensorcad://designs/${id}/analysis` })),
    );
    expect(analysis.params.total).toBe(8_030_261_248);

    const block = JSON.parse(resourceText(await client.readResource({ uri: "tensorcad://catalog/rmsnorm" })));
    expect(block.type).toBe("rmsnorm");
    expect(block.params.some((p: any) => p.name === "dim")).toBe(true);
  });

  test("open designs are listed as resources", async () => {
    const created = await newLlama();
    const { resources } = await client.listResources();
    expect(resources.map((r) => r.uri)).toContain(`tensorcad://designs/${created.id}`);
  });
});

describe("prompts", () => {
  test("all four workflows are offered with arguments", async () => {
    const { prompts } = await client.listPrompts();
    const names = prompts.map((p) => p.name);
    for (const n of ["design_model", "review_design", "scale_design", "explain_costs"]) {
      expect(names).toContain(n);
    }
    const design = prompts.find((p) => p.name === "design_model")!;
    expect(design.arguments!.map((a) => a.name)).toContain("target_params");
  });

  test("design_model renders the argument into the instructions", async () => {
    const r = await client.getPrompt({
      name: "design_model",
      arguments: { target_params: "3B", context_len: "8192", family: "llama-3-8b" },
    });
    const text = (r.messages[0].content as any).text as string;
    expect(text).toContain("3B");
    expect(text).toContain("8192");
    expect(text).toContain("tensorcad_validate");
  });

  test("explain_costs and review_design name the design", async () => {
    for (const name of ["explain_costs", "review_design"]) {
      const r = await client.getPrompt({ name, arguments: { design_id: "dsn_1" } });
      expect((r.messages[0].content as any).text).toContain("dsn_1");
    }
  });
});

describe("explain", () => {
  test("says what one block contributes and what its parameters resolved to", async () => {
    const llama = await newLlama();
    const { data, text } = await callFull("tensorcad_explain", {
      design_id: llama.id,
      path: "layers/block/attn",
    });
    expect(data.type).toBe("gqa_attention");
    expect(data.kind).toBe("composite");
    expect(data.params).toBeGreaterThan(0);
    expect(data.share_of_params).toBeGreaterThan(0);
    expect(data.share_of_params).toBeLessThan(1);
    // The parameters come back in the order the block declares them, as
    // written and as evaluated.
    expect(data.parameters[0].name).toBe("d_model");
    expect(data.parameters[0].expression).toBe("D");
    expect(data.parameters[0].value).toBe(4096);
    expect(text).toContain("gqa_attention");
    expect(text).toContain("% of the model");
  });

  test("a path that names nothing is a readable failure", async () => {
    const llama = await newLlama();
    const text = await callExpectingError("tensorcad_explain", {
      design_id: llama.id,
      path: "no/such/block",
    });
    expect(text.length).toBeGreaterThan(0);
  });
});

describe("scale", () => {
  test("a design it made can still be listed", async () => {
    // The output schema knew three sources and the store four, so the first
    // design made by scale, mup or import broke tensorcad_list_designs for the
    // rest of the session. It was only ever seen because a new test happened
    // to run before the one that lists.
    const llama = await newLlama();
    const made = await call("tensorcad_scale", {
      design_id: llama.id,
      target_params: 30e6,
      target_basis: "non-embedding",
      vocab: 8192,
      tie_head: true,
    });
    const listed = await call("tensorcad_list_designs");
    const row = listed.designs.find((d: { design_id: string }) => d.design_id === made.design_id);
    expect(row?.source).toBe("derived");
  });

  test("shrinks a design to a budget and saves the result", async () => {
    const llama = await newLlama();
    const data = await call("tensorcad_scale", {
      design_id: llama.id,
      target_params: 30e6,
      target_basis: "non-embedding",
      vocab: 8192,
      tie_head: true,
    });
    expect(data.design_id).not.toBe(llama.id);
    expect(data.from).toBe(llama.id);
    expect(data.achieved).toBeGreaterThan(0);
    expect(data.achieved).toBeLessThan(llama.params);
    expect(data.changes.length).toBeGreaterThan(0);
    // And the result is a design in this session, analysable like any other.
    const check = await call("tensorcad_analyze", { design_id: data.design_id });
    expect(check.params.total).toBeGreaterThan(0);
    expect(check.params.total).toBeLessThan(llama.params);
  });
});

describe("mup", () => {
  test("saves every rung as a design of its own", async () => {
    const llama = await newLlama();
    const { data, text } = await callFull("tensorcad_mup", {
      design_id: llama.id,
      widths: [512, 1024, 2048],
      base_width: 512,
    });
    expect(data.rungs.length).toBe(3);
    expect(data.head_dim).toBeGreaterThan(0);
    expect(data.rungs.filter((r: { base: boolean }) => r.base).length).toBe(1);

    let last = 0;
    for (const rung of data.rungs) {
      // The rung is a design in this session, analysable like any other, and
      // its parameter count is the one the ladder reported.
      const check = await call("tensorcad_analyze", { design_id: rung.design_id });
      expect(check.params.total, String(rung.width)).toBe(rung.params);
      expect(rung.params, String(rung.width)).toBeGreaterThan(last);
      last = rung.params;
      // Every head is the same width at every rung: that is the premise.
      expect(rung.heads * data.head_dim).toBe(rung.width);
      // And the multipliers are the table's, against this rung's m.
      const by = Object.fromEntries(
        rung.scaling.map((s: { class: string; init_std: number; adam_lr: number }) => [s.class, s]),
      );
      expect(by.input.init_std).toBe(1);
      expect(by.input.adam_lr).toBe(1);
      expect(by.hidden.init_std).toBeCloseTo(1 / Math.sqrt(rung.multiplier), 12);
      expect(by.hidden.adam_lr).toBeCloseTo(1 / rung.multiplier, 12);
      expect(by.output.init_std).toBeCloseTo(1 / rung.multiplier, 12);
    }
    expect(text).toContain("laddered by D");
  });

  test("refuses a design with no width to move", async () => {
    const conv = await call("tensorcad_new_design", { preset: "alexnet" });
    const text = await callExpectingError("tensorcad_mup", { design_id: conv.design_id });
    expect(text).toContain("D");
  });
});

describe("plan", () => {
  test("returns splits that fit, each using the whole cluster", async () => {
    const llama = await newLlama();
    const { data, text } = await callFull("tensorcad_plan", {
      design_id: llama.id,
      gpus: 8,
      T: 8192,
      hardware: "h100-sxm",
    });
    expect(data.fits.length).toBeGreaterThan(0);
    expect(data.considered).toBeGreaterThan(data.fits.length);
    for (const p of data.fits) {
      expect(p.dp * p.tp * p.pp * p.ep, p.summary).toBe(8);
      expect(p.used, p.summary).toBeLessThanOrEqual(1);
      expect(p.per_gpu_bytes).toBeLessThanOrEqual(data.budget_bytes);
    }
    expect(text).toContain("of budget");
  });

  test("says plainly when nothing fits, and what came closest", async () => {
    const big = await call("tensorcad_new_design", { preset: "llama-3.1-405b" });
    const { data, text } = await callFull("tensorcad_plan", {
      design_id: big.design_id,
      gpus: 8,
      T: 8192,
      hardware: "h100-sxm",
    });
    expect(data.fits.length).toBe(0);
    expect(data.closest).toBeDefined();
    expect(data.closest.per_gpu_bytes).toBeGreaterThan(data.budget_bytes);
    expect(text).toContain("nothing fits");
  });
});

describe("diff", () => {
  test("a design against itself is identical", async () => {
    const llama = await newLlama();
    const data = await call("tensorcad_diff", { a: llama.id, b: llama.id });
    expect(data.identical).toBe(true);
    for (const m of data.metrics) expect(`${m.metric} ${m.delta}`).toBe(`${m.metric} 0`);
  });

  test("reports what an edit moved, structurally and numerically", async () => {
    const before = await newLlama();
    const after = await newLlama();
    await call("tensorcad_apply_ops", {
      design_id: after.id,
      ops: [{ op: "set_symbol", name: "L", value: 16 }],
    });
    const { data, text } = await callFull("tensorcad_diff", { a: before.id, b: after.id });
    expect(data.identical).toBe(false);
    expect(data.symbols.changed.map((s: any) => s.name)).toContain("L");
    const params = data.metrics.find((m: any) => m.metric === "parameters");
    expect(params.delta).toBeLessThan(0);
    expect(text).toContain("L");
  });
});

describe("import_hf", () => {
  test("reads a config into a design that lands on the preset's count", async () => {
    const configs = JSON.parse(
      readFileSync(resolve(import.meta.dir, "../../core-go/testdata/hf-configs.json"), "utf8"),
    ) as Record<string, unknown>;
    const data = await call("tensorcad_import_hf", {
      config: JSON.stringify(configs["llama-3-8b"]),
      name: "llama-3-8b",
    });
    expect(data.warnings).toEqual([]);
    const preset = await newLlama();
    expect(data.params_total).toBe(preset.params);
    // It is a design in this session, so it can be diffed against the preset.
    const d = await call("tensorcad_diff", { a: preset.id, b: data.design_id });
    const params = d.metrics.find((m: any) => m.metric === "parameters");
    expect(params.delta).toBe(0);
  });

  test("a config it cannot read is a readable failure", async () => {
    const text = await callExpectingError("tensorcad_import_hf", { config: "{ not json" });
    expect(text.length).toBeGreaterThan(0);
  });
});

describe("registry manifest", () => {
  const manifest = JSON.parse(
    readFileSync(resolve(import.meta.dir, "../server.json"), "utf8"),
  ) as {
    $schema: string;
    name: string;
    description: string;
    version: string;
    packages: { identifier: string; version: string; transport: { type: string } }[];
  };
  const pkg = JSON.parse(readFileSync(resolve(import.meta.dir, "../package.json"), "utf8")) as {
    name: string;
    version: string;
    mcpName: string;
    description: string;
  };

  // The schema is fetched and checked when the file is written; what drifts
  // afterwards is this file against package.json, silently, until a publish
  // is rejected or — worse — accepted under the wrong version.
  test("agrees with package.json about what is being published", () => {
    expect(manifest.name).toBe(pkg.mcpName);
    expect(manifest.version).toBe(pkg.version);
    expect(manifest.packages).toHaveLength(1);
    expect(manifest.packages[0].identifier).toBe(pkg.name);
    expect(manifest.packages[0].version).toBe(pkg.version);
    expect(manifest.packages[0].transport.type).toBe("stdio");
  });

  // The registry grants a GitHub login `io.github.<owner>/*` in the owner's
  // own casing and compares case-sensitively, so a namespace that differs from
  // the account only in case is a 403 at publish time — after npm has taken
  // the package. registry#689.
  test("names its namespace exactly as GitHub spells the owner", () => {
    const repo = (manifest as unknown as { repository: { url: string } }).repository.url;
    const owner = new URL(repo).pathname.split("/")[1]!;
    expect(manifest.name.split("/")[0]).toBe(`io.github.${owner}`);
  });

  test("says the same version the server reports over the protocol", () => {
    expect(client.getServerVersion()?.version).toBe(manifest.version);
  });

  // The registry caps it, and the failure is a rejected publish rather than
  // anything visible here.
  test("the description fits what the registry allows", () => {
    expect(manifest.description.length).toBeGreaterThan(0);
    expect(manifest.description.length).toBeLessThanOrEqual(100);
    expect(manifest.$schema).toStartWith("https://static.modelcontextprotocol.io/schemas/");
  });
});
