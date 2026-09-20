/**
 * The tool surface: thirteen tools, namespaced `tensorcad_`.
 *
 * Deliberately few and coarse. Some clients cap how many tools they keep
 * active, so graph editing is one batched `tensorcad_apply_ops` rather than a
 * setter per property, and reads default to the compact outline.
 */

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import * as z from "zod";
import type { McpServer } from "@modelcontextprotocol/server";
import {
  AnalysisOutput,
  analysisOptionsShape,
  BlockPort,
  Counts,
  DESIGN_ID,
  DesignSummary,
  Finding,
  Op,
  Outline,
  ValidationSummary,
} from "./schemas.js";
import {
  allCatalogEntries,
  analysisJson,
  analysisText,
  blockDetail,
  blockText,
  catalogText,
  findingsJson,
  findingsText,
  outlineOf,
  outlineText,
  validationSummary,
} from "./summarize.js";
import type { DocumentStore } from "./store/types.js";
import type { Op as OpType } from "./ops.js";
import type { AnalysisOptions } from "@tensorcad/engine";
import { formatBytes, formatCount } from "@tensorcad/engine";
import {
  HARDWARE,
  PRESET_NAMES,
  analyze,
  diffDesigns,
  explain,
  generateTorch,
  getPreset,
  importHfConfig,
  planCluster,
  scaleDesign,
  validate,
} from "@tensorcad/engine/node";

type ToolResult = {
  content: { type: "text"; text: string }[];
  structuredContent?: Record<string, unknown>;
  isError?: boolean;
};

const ok = (text: string, structured: Record<string, unknown>): ToolResult => ({
  content: [{ type: "text", text }],
  structuredContent: structured,
});

/**
 * Turn a thrown error into a result the model can read and retry from, rather
 * than a protocol error that ends the turn.
 */
async function guard(run: () => ToolResult | Promise<ToolResult>): Promise<ToolResult> {
  try {
    return await run();
  } catch (e) {
    return { content: [{ type: "text", text: `Error: ${(e as Error).message}` }], isError: true };
  }
}

// ---------------------------------------------------------------------------

type AnalysisInput = {
  T?: number;
  B?: number;
  dtype?: "fp32" | "bf16" | "fp16" | "fp8";
  hardware?: string;
  gpus?: number;
  tokens?: number;
  optimizer?: "adamw" | "adamw8bit" | "muon" | "sgd_momentum" | "sgd" | "bf16_adam";
  recompute?: "none" | "selective" | "full";
  zero?: number;
  tp?: number;
  dp?: number;
  pp?: number;
  ep?: number;
  concurrency?: number;
  mfu?: number;
};

function toAnalysisOptions(input: AnalysisInput): AnalysisOptions {
  const out: AnalysisOptions = {};
  if (input.T !== undefined) out.T = input.T;
  if (input.B !== undefined) out.B = input.B;
  if (input.dtype) out.dtype = input.dtype;
  if (input.tokens !== undefined) out.tokens = input.tokens;
  if (input.optimizer) out.optimizer = input.optimizer;
  if (input.recompute) out.recompute = input.recompute;
  if (input.concurrency !== undefined) out.concurrency = input.concurrency;
  if (input.mfu !== undefined) out.mfu = input.mfu;

  if (input.hardware) {
    if (!HARDWARE.some((h) => h.id === input.hardware)) {
      throw new Error(`Unknown hardware "${input.hardware}". Known ids: ${HARDWARE.map((h) => h.id).join(", ")}.`);
    }
    out.hardware = input.hardware;
  }

  const parallel: NonNullable<AnalysisOptions["parallel"]> = {};
  if (input.zero !== undefined) parallel.zero = input.zero as 0 | 1 | 2 | 3;
  if (input.tp !== undefined) parallel.tp = input.tp;
  if (input.dp !== undefined) parallel.dp = input.dp;
  if (input.pp !== undefined) parallel.pp = input.pp;
  if (input.ep !== undefined) parallel.ep = input.ep;
  if (Object.keys(parallel).length > 0) out.parallel = parallel;

  if (input.gpus !== undefined) out.gpus = input.gpus;
  else {
    const implied = (parallel.dp ?? 1) * (parallel.tp ?? 1) * (parallel.pp ?? 1);
    if (implied > 1) out.gpus = implied;
  }
  return out;
}

const READ = { readOnlyHint: true, idempotentHint: true, openWorldHint: false } as const;
const WRITE = { readOnlyHint: false, idempotentHint: false, openWorldHint: false } as const;
const DESTRUCTIVE = { readOnlyHint: false, idempotentHint: false, destructiveHint: true, openWorldHint: false } as const;

// ---------------------------------------------------------------------------

export function registerTools(server: McpServer, store: DocumentStore): void {
  // -- 1. list_designs -----------------------------------------------------
  server.registerTool(
    "tensorcad_list_designs",
    {
      title: "List designs",
      description:
        "List the designs this server has open, the built-in reference architectures you can start from, " +
        "and the .tensorcad.json files it can see on disk. Start here when you do not already hold a design_id.",
      inputSchema: z.object({
        include_files: z.boolean().optional().describe("Also scan the working directory for .tensorcad.json files."),
      }),
      outputSchema: z.object({
        designs: z.array(DesignSummary),
        presets: z.array(
          z.object({
            name: z.string(),
            family: z.string().optional(),
            published_params: z.number().optional(),
            notes: z.string().optional(),
          }),
        ),
        files: z.array(z.string()),
      }),
      annotations: { ...READ, title: "List designs" },
    },
    async ({ include_files }) =>
      guard(async () => {
        const designs = store.list();
        const presets = PRESET_NAMES.map((name) => {
          const doc = getPreset(name);
          const p: { name: string; family?: string; published_params?: number; notes?: string } = { name };
          if (doc.meta.family) p.family = doc.meta.family;
          if (doc.meta.published?.params) p.published_params = doc.meta.published.params;
          if (doc.meta.notes) p.notes = doc.meta.notes;
          return p;
        });
        const files = include_files ? await store.listFiles() : [];

        const text = [
          designs.length > 0
            ? `open designs:\n${designs
                .map((d) => `  ${d.design_id}  ${d.name}  rev ${d.revision}${d.dirty ? " (unsaved)" : ""}`)
                .join("\n")}`
            : "open designs: none. Use tensorcad_new_design or tensorcad_open_design.",
          "",
          `presets (${presets.length}):`,
          ...presets.map(
            (p) => `  ${p.name}${p.published_params ? `  ${formatCount(p.published_params)}` : ""}`,
          ),
          ...(include_files ? ["", `files (${files.length}):`, ...files.map((f) => `  ${f}`)] : []),
        ].join("\n");

        return ok(text, { designs, presets, files });
      }),
  );

  // -- 2. new_design -------------------------------------------------------
  server.registerTool(
    "tensorcad_new_design",
    {
      title: "New design",
      description:
        "Create a design from a reference architecture, or an empty one with just the B and T runtime symbols. " +
        "Returns the design_id every other tool needs.",
      inputSchema: z.object({
        preset: z.string().optional().describe(`One of: ${PRESET_NAMES.join(", ")}. Omit for an empty design.`),
        name: z.string().optional().describe("Name for the new design. Defaults to the preset's name."),
      }),
      outputSchema: z.object({
        design_id: z.string(),
        revision: z.number().int(),
        name: z.string(),
        source: z.enum(["preset", "file", "empty"]),
        params_total: z.number(),
        outline: Outline,
      }),
      annotations: { ...WRITE, title: "New design" },
    },
    async (args) =>
      guard(() => {
        const record = store.create(args);
        const outline = outlineOf(record.doc);
        return ok(
          `${record.design_id} (revision ${record.revision})\n\n${outlineText(outline)}`,
          {
            design_id: record.design_id,
            revision: record.revision,
            name: record.name,
            source: record.source,
            params_total: outline.params_total,
            outline,
          },
        );
      }),
  );

  // -- 3. open_design ------------------------------------------------------
  server.registerTool(
    "tensorcad_open_design",
    {
      title: "Open design",
      description:
        "Load a .tensorcad.json document from disk and return a design_id for it. " +
        "Opening the same path twice returns the same handle.",
      inputSchema: z.object({
        path: z.string().describe("Path to a .tensorcad.json file, absolute or relative to the server's directory."),
      }),
      outputSchema: z.object({
        design_id: z.string(),
        revision: z.number().int(),
        name: z.string(),
        path: z.string(),
        params_total: z.number(),
        validation: ValidationSummary,
      }),
      annotations: { ...WRITE, title: "Open design" },
    },
    async ({ path }) =>
      guard(async () => {
        const record = await store.open(path);
        const report = validate(record.doc);
        return ok(
          `${record.design_id}  ${record.name}  revision ${record.revision}\n` +
            `${record.path}\n${formatCount(report.analysis.params.total)} parameters, ` +
            `${report.counts.error} error(s), ${report.counts.warning} warning(s)`,
          {
            design_id: record.design_id,
            revision: record.revision,
            name: record.name,
            path: record.path ?? path,
            params_total: report.analysis.params.total,
            validation: validationSummary(report),
          },
        );
      }),
  );

  // -- 4. save_design ------------------------------------------------------
  server.registerTool(
    "tensorcad_save_design",
    {
      title: "Save design",
      description:
        "Write a design to disk as .tensorcad.json. Overwrites the file it was opened from unless a path is given.",
      inputSchema: z.object({
        design_id: DESIGN_ID,
        path: z.string().optional().describe("Where to write. Defaults to the path it was opened from."),
      }),
      outputSchema: z.object({
        design_id: z.string(),
        revision: z.number().int(),
        path: z.string(),
        bytes: z.number().int(),
      }),
      annotations: { ...DESTRUCTIVE, title: "Save design" },
    },
    async ({ design_id, path }) =>
      guard(async () => {
        const { record, path: written, bytes } = await store.save(design_id, path);
        return ok(`wrote ${written} (${bytes} bytes, revision ${record.revision})`, {
          design_id: record.design_id,
          revision: record.revision,
          path: written,
          bytes,
        });
      }),
  );

  // -- 5. get_design -------------------------------------------------------
  server.registerTool(
    "tensorcad_get_design",
    {
      title: "Get design",
      description:
        'Read a design. Use format "outline" (the default) first: it is the whole structure, symbol table and ' +
        'edge shapes in a fraction of the tokens. Use format "full" only when you need the literal JSON document.',
      inputSchema: z.object({
        design_id: DESIGN_ID,
        format: z
          .enum(["full", "outline"])
          .optional()
          .describe('"outline" is a compact block/edge summary; "full" is the whole document.'),
      }),
      outputSchema: z.object({
        design_id: z.string(),
        revision: z.number().int(),
        name: z.string(),
        format: z.enum(["full", "outline"]),
        dirty: z.boolean(),
        path: z.string().optional(),
        params_total: z.number(),
        params_active: z.number(),
        outline: Outline.optional(),
        document: z.record(z.string(), z.unknown()).optional().describe("The literal design document."),
      }),
      annotations: { ...READ, title: "Get design" },
    },
    async ({ design_id, format }) =>
      guard(() => {
        const record = store.get(design_id);
        const outline = outlineOf(record.doc);
        const mode = format ?? "outline";

        const base = {
          design_id: record.design_id,
          revision: record.revision,
          name: record.name,
          format: mode,
          dirty: record.dirty,
          params_total: outline.params_total,
          params_active: outline.params_active,
          ...(record.path ? { path: record.path } : {}),
        };

        if (mode === "full") {
          return ok(JSON.stringify(record.doc, null, 2), {
            ...base,
            document: record.doc as unknown as Record<string, unknown>,
          });
        }
        return ok(`${record.design_id} revision ${record.revision}\n\n${outlineText(outline)}`, {
          ...base,
          outline,
        });
      }),
  );

  // -- 6. get_block --------------------------------------------------------
  server.registerTool(
    "tensorcad_get_block",
    {
      title: "Get block",
      description:
        "One block of a design: its parameters as written and as resolved, the inferred shape on every port, " +
        "what each port is wired to, and how many trainable parameters it contributes.",
      inputSchema: z.object({
        design_id: DESIGN_ID,
        path: z.string().describe('Block path from the outline, e.g. "layers/block" or "embed".'),
      }),
      outputSchema: z.object({
        design_id: z.string(),
        revision: z.number().int(),
        path: z.string(),
        id: z.string(),
        type: z.string(),
        kind: z.string(),
        category: z.string(),
        label: z.string().optional(),
        summary: z.string(),
        formula: z.string().optional(),
        params: z.record(z.string(), z.unknown()),
        resolved_params: z.record(z.string(), z.unknown()),
        param_errors: z.array(z.string()),
        inputs: z.array(BlockPort),
        outputs: z.array(BlockPort),
        params_count: z.number(),
        instances: z.number(),
        children: z.array(z.string()),
      }),
      annotations: { ...READ, title: "Get block" },
    },
    async ({ design_id, path }) =>
      guard(() => {
        const record = store.get(design_id);
        const detail = blockDetail(record.doc, path);
        return ok(blockText(detail), {
          design_id: record.design_id,
          revision: record.revision,
          ...detail,
        });
      }),
  );

  // -- 7. search_catalog ---------------------------------------------------
  server.registerTool(
    "tensorcad_search_catalog",
    {
      title: "Search catalog",
      description:
        "Search the block catalog. Returns each block's parameter schema, port patterns and documentation, " +
        "which is what you need before adding a block with tensorcad_apply_ops.",
      inputSchema: z.object({
        query: z.string().optional().describe("Substring matched against type, category, summary and formula."),
        category: z.string().optional().describe("attention, mlp, norm, embedding, container, io, ..."),
        kind: z.enum(["primitive", "composite", "container"]).optional(),
        limit: z.number().int().positive().max(100).optional().describe("Default 20."),
      }),
      outputSchema: z.object({
        total: z.number().int().describe("Matches before the limit was applied."),
        categories: z.array(z.string()),
        blocks: z.array(
          z.object({
            type: z.string(),
            kind: z.string(),
            category: z.string(),
            summary: z.string(),
            formula: z.string().optional(),
            refs: z.array(z.string()),
            params: z.array(
              z.object({
                name: z.string(),
                type: z.string(),
                default: z.string().optional(),
                doc: z.string().optional(),
                values: z.array(z.string()).optional(),
              }),
            ),
            inputs: z.array(z.string()),
            outputs: z.array(z.string()),
            dynamic_ports: z.boolean(),
          }),
        ),
      }),
      annotations: { ...READ, title: "Search catalog" },
    },
    async ({ query, category, kind, limit }) =>
      guard(() => {
        const all = allCatalogEntries();
        const q = query?.toLowerCase();
        const matched = all.filter((e) => {
          if (category && e.category !== category) return false;
          if (kind && e.kind !== kind) return false;
          if (!q) return true;
          const hay = `${e.type} ${e.category} ${e.summary} ${e.formula ?? ""}`.toLowerCase();
          return hay.includes(q);
        });
        const blocks = matched.slice(0, limit ?? 20);
        const categories = [...new Set(all.map((e) => e.category))].sort();
        const header = `${matched.length} match(es)${matched.length > blocks.length ? `, showing ${blocks.length}` : ""}`;
        return ok(`${header}\n\n${catalogText(blocks)}`, {
          total: matched.length,
          categories,
          blocks,
        });
      }),
  );

  // -- 8. apply_ops --------------------------------------------------------
  server.registerTool(
    "tensorcad_apply_ops",
    {
      title: "Apply edits",
      description:
        "Apply a batch of edits to a design. The batch is all-or-nothing: the first rejected operation aborts it " +
        "and the design is left untouched. Pass expected_revision to be told about a concurrent edit instead of " +
        "silently overwriting it. Returns the new revision, what changed, and a fresh validation summary.",
      inputSchema: z.object({
        design_id: DESIGN_ID,
        expected_revision: z
          .number()
          .int()
          .optional()
          .describe("Revision you last read. The call is rejected if the design has moved on."),
        ops: z.array(Op).min(1).describe("Edits applied in order."),
      }),
      outputSchema: z.object({
        design_id: z.string(),
        revision: z.number().int(),
        previous_revision: z.number().int(),
        name: z.string(),
        applied: z.array(z.string()).describe("One line per operation, in order."),
        params_total: z.number(),
        params_active: z.number(),
        params_delta: z.number().describe("Change in total parameters caused by this batch."),
        validation: ValidationSummary,
      }),
      annotations: { ...WRITE, title: "Apply edits" },
    },
    async ({ design_id, expected_revision, ops }) =>
      guard(() => {
        const before = store.get(design_id);
        const paramsBefore = outlineOf(before.doc).params_total;

        const outcome = store.apply(design_id, ops as OpType[], expected_revision);
        const report = validate(outcome.record.doc);
        const total = report.analysis.params.total;
        const delta = total - paramsBefore;

        const text = [
          `${outcome.record.design_id} revision ${outcome.previousRevision} -> ${outcome.record.revision}`,
          ...outcome.applied.map((a) => `  ${a}`),
          "",
          `parameters ${formatCount(total)}` +
            (delta === 0 ? " (unchanged)" : ` (${delta > 0 ? "+" : ""}${formatCount(delta)})`),
          `${report.counts.error} error(s), ${report.counts.warning} warning(s)`,
          ...(report.findings.length > 0 ? ["", findingsText(findingsJson(report).slice(0, 5))] : []),
        ].join("\n");

        return ok(text, {
          design_id: outcome.record.design_id,
          revision: outcome.record.revision,
          previous_revision: outcome.previousRevision,
          name: outcome.record.name,
          applied: outcome.applied,
          params_total: total,
          params_active: report.analysis.params.active,
          params_delta: delta,
          validation: validationSummary(report),
        });
      }),
  );

  // -- 9. validate ---------------------------------------------------------
  server.registerTool(
    "tensorcad_validate",
    {
      title: "Validate design",
      description:
        "Run every design rule: shape and symbol errors, kernel-friendly head dimensions, tensor-core multiples, " +
        "whether training and serving fit the chosen device, Chinchilla sanity and drift from published numbers. " +
        "Each finding carries a fix hint.",
      inputSchema: z.object({
        design_id: DESIGN_ID,
        severity: z.enum(["error", "warning", "info"]).optional().describe("Only return findings at least this bad."),
        ...analysisOptionsShape,
      }),
      outputSchema: z.object({
        design_id: z.string(),
        revision: z.number().int(),
        name: z.string(),
        ok: z.boolean(),
        counts: Counts,
        findings: z.array(Finding),
        params_total: z.number(),
      }),
      annotations: { ...READ, title: "Validate design" },
    },
    async ({ design_id, severity, ...rest }) =>
      guard(() => {
        const record = store.get(design_id);
        const report = validate(record.doc, toAnalysisOptions(rest));
        const rank = { error: 0, warning: 1, info: 2 };
        const findings = findingsJson(report).filter(
          (f) => severity === undefined || rank[f.severity] <= rank[severity],
        );

        const text = [
          `${record.name} (${record.design_id} revision ${record.revision})`,
          `${report.ok ? "ok" : "FAILED"}: ${report.counts.error} error(s), ` +
            `${report.counts.warning} warning(s), ${report.counts.info} info`,
          "",
          findingsText(findings),
        ].join("\n");

        return ok(text, {
          design_id: record.design_id,
          revision: record.revision,
          name: record.name,
          ok: report.ok,
          counts: report.counts,
          findings,
          params_total: report.analysis.params.total,
        });
      }),
  );

  // -- 10. analyze ---------------------------------------------------------
  server.registerTool(
    "tensorcad_analyze",
    {
      title: "Analyze design",
      description:
        "Parameters, FLOPs per token, KV cache, training and serving memory, decode throughput, training cost and " +
        "Chinchilla position, for a given sequence length, batch, dtype, device, GPU count and parallel plan.",
      inputSchema: z.object({ design_id: DESIGN_ID, ...analysisOptionsShape }),
      outputSchema: AnalysisOutput,
      annotations: { ...READ, title: "Analyze design" },
    },
    async ({ design_id, ...rest }) =>
      guard(() => {
        const record = store.get(design_id);
        const result = analyze(record.doc, toAnalysisOptions(rest));
        return ok(analysisText(result), {
          design_id: record.design_id,
          revision: record.revision,
          ...analysisJson(result),
        });
      }),
  );

  // -- 11. generate_code ---------------------------------------------------
  server.registerTool(
    "tensorcad_generate_code",
    {
      title: "Generate code",
      description:
        "Emit a runnable PyTorch module plus the design document. Without out_dir the file contents come back in " +
        "the result; with out_dir they are written to disk and only a manifest comes back.",
      inputSchema: z.object({
        design_id: DESIGN_ID,
        class_name: z.string().optional().describe("Class name for the top-level module."),
        include_smoke_test: z.boolean().optional().describe("Emit a __main__ block that checks the size."),
        out_dir: z.string().optional().describe("Directory to write into. Omit to get the contents inline."),
      }),
      outputSchema: z.object({
        design_id: z.string(),
        revision: z.number().int(),
        wrote: z.boolean(),
        out_dir: z.string().optional(),
        files: z.array(
          z.object({
            path: z.string(),
            bytes: z.number().int(),
            lines: z.number().int(),
            contents: z.string().optional().describe("Present only when out_dir was not given."),
            written_to: z.string().optional(),
          }),
        ),
        warnings: z.array(z.string()),
      }),
      annotations: { readOnlyHint: false, idempotentHint: true, openWorldHint: false, title: "Generate code" },
    },
    async ({ design_id, class_name, include_smoke_test, out_dir }) =>
      guard(async () => {
        const record = store.get(design_id);
        const generated = generateTorch(record.doc, {
          ...(class_name ? { className: class_name } : {}),
          includeSmokeTest: include_smoke_test ?? false,
        });

        const root = out_dir ? (isAbsolute(out_dir) ? out_dir : resolve(process.cwd(), out_dir)) : undefined;
        const files = [];
        for (const file of generated.files) {
          const bytes = Buffer.byteLength(file.contents, "utf8");
          const lines = file.contents.split("\n").length;
          if (root) {
            const target = join(root, file.path);
            await mkdir(dirname(target), { recursive: true });
            await writeFile(target, file.contents, "utf8");
            files.push({ path: file.path, bytes, lines, written_to: target });
          } else {
            files.push({ path: file.path, bytes, lines, contents: file.contents });
          }
        }

        const text = root
          ? [`wrote ${files.length} file(s) to ${root}`, ...files.map((f) => `  ${f.path}  ${f.bytes} bytes`)].join("\n")
          : generated.files.map((f) => `# ${f.path}\n${f.contents}`).join("\n\n");

        return ok(
          generated.warnings.length > 0 ? `${text}\n\nwarnings:\n${generated.warnings.map((w) => `  ${w}`).join("\n")}` : text,
          {
            design_id: record.design_id,
            revision: record.revision,
            wrote: Boolean(root),
            ...(root ? { out_dir: root } : {}),
            files,
            warnings: generated.warnings,
          },
        );
      }),
  );

  // -- 12. checkpoint ------------------------------------------------------
  server.registerTool(
    "tensorcad_checkpoint",
    {
      title: "Checkpoint design",
      description:
        "Snapshot a design under a name you can come back to. Take one before an experiment so tensorcad_restore " +
        "can put it back exactly.",
      inputSchema: z.object({
        design_id: DESIGN_ID,
        label: z.string().optional().describe("What this snapshot is, e.g. \"before widening the FFN\"."),
      }),
      outputSchema: z.object({
        design_id: z.string(),
        checkpoint_id: z.string(),
        label: z.string(),
        revision: z.number().int(),
        created_at: z.string(),
        checkpoints: z.array(
          z.object({
            checkpoint_id: z.string(),
            label: z.string(),
            revision: z.number().int(),
            created_at: z.string(),
          }),
        ),
      }),
      annotations: { ...WRITE, title: "Checkpoint design" },
    },
    async ({ design_id, label }) =>
      guard(() => {
        const info = store.checkpoint(design_id, label);
        return ok(`${info.checkpoint_id} at revision ${info.revision}: ${info.label}`, {
          design_id,
          ...info,
          checkpoints: store.checkpoints(design_id),
        });
      }),
  );

  // -- 13. restore ---------------------------------------------------------
  server.registerTool(
    "tensorcad_restore",
    {
      title: "Restore design",
      description:
        "Put a design back. With a checkpoint_id it restores that snapshot; without one it undoes the most recent " +
        "tensorcad_apply_ops batch. Either way the revision moves forward, so a stale expected_revision still fails.",
      inputSchema: z.object({
        design_id: DESIGN_ID,
        checkpoint_id: z.string().optional().describe("Omit to undo the last batch of edits."),
      }),
      outputSchema: z.object({
        design_id: z.string(),
        revision: z.number().int(),
        name: z.string(),
        restored_from: z.string(),
        params_total: z.number(),
        validation: ValidationSummary,
      }),
      annotations: { ...DESTRUCTIVE, title: "Restore design" },
    },
    async ({ design_id, checkpoint_id }) =>
      guard(() => {
        const { record, restoredFrom } = store.restore(design_id, checkpoint_id);
        const report = validate(record.doc);
        return ok(
          `${record.design_id} restored from ${restoredFrom}; now revision ${record.revision}, ` +
            `${formatCount(report.analysis.params.total)} parameters`,
          {
            design_id: record.design_id,
            revision: record.revision,
            name: record.name,
            restored_from: restoredFrom,
            params_total: report.analysis.params.total,
            validation: validationSummary(report),
          },
        );
      }),
  );

  // -- 14. explain ---------------------------------------------------------
  server.registerTool(
    "tensorcad_explain",
    {
      title: "Explain a block",
      description:
        "What one block is and what it contributes: its parameters as written and as evaluated, the shape on " +
        "every port, its share of the model's weights and compute, and its documentation. Use it to answer " +
        "\"why is this block this size\" without reading the whole design.",
      inputSchema: z.object({
        design_id: DESIGN_ID,
        path: z.string().describe('Full path of the block, e.g. "layers/block/attn".'),
        ...analysisOptionsShape,
      }),
      outputSchema: z.object({
        design_id: z.string(),
        revision: z.number().int(),
        path: z.string(),
        type: z.string(),
        kind: z.string(),
        summary: z.string().optional(),
        params: z.number(),
        share_of_params: z.number(),
        flops_per_token: z.number(),
        share_of_flops: z.number(),
        copies: z.object({ total: z.number(), active: z.number() }),
        parameters: z.array(
          z.object({
            name: z.string(),
            expression: z.string().optional(),
            value: z.number().optional(),
            doc: z.string().optional(),
          }),
        ),
        ports: z.object({
          in: z.record(z.string(), z.string()),
          out: z.record(z.string(), z.string()),
        }),
      }),
      annotations: { ...READ, title: "Explain a block" },
    },
    async ({ design_id, path, ...rest }) =>
      guard(() => {
        const record = store.get(design_id);
        const e = explain(record.doc, path, toAnalysisOptions(rest));
        const lines = [
          `${path}  ${e.type} (${e.kind})`,
          e.docs.summary ?? "",
          `parameters ${formatCount(e.contributes.params)}  ${(e.contributes.shareOfParams * 100).toFixed(1)}% of the model`,
          `FLOPs/token ${formatCount(e.contributes.flopsPerToken)}  ${(e.contributes.shareOfFlops * 100).toFixed(1)}%`,
          `copies ${e.copies.total} total, ${e.copies.active} active per token`,
          "",
          ...e.paramOrder.map((name) => {
            const p = e.params[name];
            const written = p.expression !== undefined && String(p.expression) !== String(p.value);
            return `  ${name} = ${p.value ?? "\u2014"}${written ? `  (${p.expression})` : ""}`;
          }),
        ];
        return ok(lines.filter((l) => l !== "").join("\n"), {
          design_id: record.design_id,
          revision: record.revision,
          path,
          type: e.type,
          kind: e.kind,
          ...(e.docs.summary ? { summary: e.docs.summary } : {}),
          params: e.contributes.params,
          share_of_params: e.contributes.shareOfParams,
          flops_per_token: e.contributes.flopsPerToken,
          share_of_flops: e.contributes.shareOfFlops,
          copies: e.copies,
          parameters: e.paramOrder.map((name) => ({
            name,
            ...(e.params[name].expression !== undefined
              ? { expression: String(e.params[name].expression) }
              : {}),
            ...(typeof e.params[name].value === "number" ? { value: e.params[name].value } : {}),
            ...(e.params[name].doc ? { doc: e.params[name].doc } : {}),
          })),
          ports: {
            in: Object.fromEntries(Object.entries(e.shapes.in).map(([k, v]) => [k, String(v)])),
            out: Object.fromEntries(Object.entries(e.shapes.out).map(([k, v]) => [k, String(v)])),
          },
        });
      }),
  );

  // -- 15. scale -----------------------------------------------------------
  server.registerTool(
    "tensorcad_scale",
    {
      title: "Scale a design",
      description:
        "Shrink a design towards a parameter budget while keeping its proportions, and save the result as a new " +
        "design. Use it to get a bench-sized proxy of a large architecture: the widths and depth move together, " +
        "the head dimension stays sane, and the result is reported with how close it landed.",
      inputSchema: z.object({
        design_id: DESIGN_ID,
        target_params: z.number().positive().describe("The parameter count to aim for."),
        target_basis: z
          .enum(["total", "non-embedding"])
          .optional()
          .describe('Whether target_params counts the embedding tables. At bench sizes "non-embedding" is usually meant.'),
        vocab: z.number().int().positive().optional().describe("Replace the vocabulary, for a smaller tokenizer."),
        tie_head: z.boolean().optional().describe("Share the output projection with the embedding."),
        keep_depth: z.boolean().optional().describe("Hold the layer count fixed and move only the width."),
      }),
      outputSchema: z.object({
        design_id: z.string().describe("The new design, saved in this session."),
        from: z.string(),
        name: z.string(),
        achieved: z.number(),
        target: z.number(),
        changes: z.array(z.object({ symbol: z.string(), from: z.number(), to: z.number() })),
        notes: z.array(z.string()),
      }),
      annotations: { readOnlyHint: false, idempotentHint: true, openWorldHint: false, title: "Scale a design" },
    },
    async ({ design_id, target_params, target_basis, vocab, tie_head, keep_depth }) =>
      guard(() => {
        const record = store.get(design_id);
        const result = scaleDesign(record.doc, {
          targetParams: target_params,
          ...(target_basis ? { targetBasis: target_basis } : {}),
          ...(vocab !== undefined ? { vocab } : {}),
          ...(tie_head !== undefined ? { tieHead: tie_head } : {}),
          ...(keep_depth !== undefined ? { keepDepth: keep_depth } : {}),
        });
        const saved = store.adopt(result.doc);
        const changes = Object.entries(result.changes).map(([symbol, c]) => ({
          symbol,
          from: c.from,
          to: c.to,
        }));
        const text = [
          `${result.doc.meta.name}: ${formatCount(result.achieved)} against a target of ${formatCount(result.target)}`,
          ...changes.map((c) => `  ${c.symbol}  ${c.from} -> ${c.to}`),
          ...result.notes.map((n) => `  note: ${n}`),
        ].join("\n");
        return ok(text, {
          design_id: saved.design_id,
          from: record.design_id,
          name: result.doc.meta.name,
          achieved: result.achieved,
          target: result.target,
          changes,
          notes: result.notes,
        });
      }),
  );

  // -- 16. plan ------------------------------------------------------------
  server.registerTool(
    "tensorcad_plan",
    {
      title: "Plan a cluster",
      description:
        "Every way of splitting the training across a cluster that fits, least demanding first. Prices data, " +
        "tensor, pipeline and expert parallelism, the four ZeRO stages, sequence parallelism and the three " +
        "recompute settings. Memory is the claim and it is arithmetic; which plan is fastest is not claimed, so " +
        "each one carries a note about what it costs to run.",
      inputSchema: z.object({
        design_id: DESIGN_ID,
        ...analysisOptionsShape,
        // After the spread, because the shared shape has an optional `gpus` and
        // the planner needs one: this is the cluster being searched, not a
        // condition being asserted.
        gpus: z.number().int().positive().describe("How many devices there are."),
        gpus_per_node: z.number().int().positive().optional().describe("Bounds the tensor-parallel degree. Default 8."),
        headroom: z.number().positive().max(0.9).optional().describe("Fraction of device memory left free. Default 0.1."),
        limit: z.number().int().positive().optional().describe("How many plans to return. Default 8."),
      }),
      outputSchema: z.object({
        design_id: z.string(),
        revision: z.number().int(),
        hardware: z.string(),
        budget_bytes: z.number(),
        considered: z.number().int(),
        fits: z.array(
          z.object({
            summary: z.string(),
            dp: z.number(),
            tp: z.number(),
            pp: z.number(),
            ep: z.number(),
            zero: z.number().int(),
            sequence_parallel: z.boolean(),
            recompute: z.string(),
            per_gpu_bytes: z.number(),
            used: z.number(),
            notes: z.array(z.string()),
          }),
        ),
        closest: z.object({ summary: z.string(), per_gpu_bytes: z.number() }).optional(),
        notes: z.array(z.string()),
      }),
      annotations: { ...READ, title: "Plan a cluster" },
    },
    async ({ design_id, gpus, gpus_per_node, headroom, limit, ...rest }) =>
      guard(() => {
        const record = store.get(design_id);
        const result = planCluster(record.doc, toAnalysisOptions(rest), {
          gpus,
          ...(gpus_per_node !== undefined ? { gpusPerNode: gpus_per_node } : {}),
          ...(headroom !== undefined ? { headroom } : {}),
          ...(limit !== undefined ? { limit } : {}),
        });
        const text =
          result.fits.length === 0
            ? [
                `nothing fits on ${gpus} x ${result.hardware}`,
                ...(result.closest
                  ? [`  closest: ${result.closest.summary} at ${formatBytes(result.closest.perGpu.total)}`]
                  : []),
                ...result.notes.map((n) => `  ${n}`),
              ].join("\n")
            : [
                `${gpus} x ${result.hardware}, ${formatBytes(result.budget)} usable each, ${result.considered} plans priced`,
                ...result.fits.map(
                  (p) =>
                    `  ${p.summary}  ${formatBytes(p.perGpu.total)}  ${Math.round(p.used * 100)}% of budget`,
                ),
              ].join("\n");
        return ok(text, {
          design_id: record.design_id,
          revision: record.revision,
          hardware: result.hardware,
          budget_bytes: result.budget,
          considered: result.considered,
          fits: result.fits.map((p) => ({
            summary: p.summary,
            dp: p.parallel.dp,
            tp: p.parallel.tp,
            pp: p.parallel.pp,
            ep: p.parallel.ep,
            zero: p.parallel.zero,
            sequence_parallel: p.parallel.sequenceParallel,
            recompute: p.recompute,
            per_gpu_bytes: p.perGpu.total,
            used: p.used,
            notes: p.notes,
          })),
          ...(result.closest
            ? { closest: { summary: result.closest.summary, per_gpu_bytes: result.closest.perGpu.total } }
            : {}),
          notes: result.notes,
        });
      }),
  );

  // -- 17. diff ------------------------------------------------------------
  server.registerTool(
    "tensorcad_diff",
    {
      title: "Compare two designs",
      description:
        "What changed between two designs and what it cost: the symbols, blocks and wires that moved, then the " +
        "parameters, FLOPs, cache and memory. Both sides are measured at one operating point, so the attention " +
        "terms are comparable. Use it after an edit, or against a preset, to check the change did what was meant.",
      inputSchema: z.object({
        a: DESIGN_ID.describe("The design to compare from."),
        b: DESIGN_ID.describe("The design to compare to."),
        ...analysisOptionsShape,
      }),
      outputSchema: z.object({
        a: z.string(),
        b: z.string(),
        identical: z.boolean().describe("True when nothing structural moved; the numbers may still differ."),
        at: z.object({ T: z.number(), B: z.number(), hardware: z.string() }),
        symbols: z.object({
          added: z.array(z.string()),
          removed: z.array(z.string()),
          changed: z.array(z.object({ name: z.string(), from: z.string(), to: z.string() })),
        }),
        blocks: z.object({
          added: z.array(z.string()),
          removed: z.array(z.string()),
          changed: z.array(
            z.object({
              path: z.string(),
              params: z.array(z.object({ key: z.string(), from: z.string(), to: z.string() })),
            }),
          ),
        }),
        edges: z.object({ added: z.number().int(), removed: z.number().int() }),
        metrics: z.array(
          z.object({
            metric: z.string(),
            a: z.number(),
            b: z.number(),
            delta: z.number(),
            ratio: z.number().nullable(),
          }),
        ),
      }),
      annotations: { ...READ, title: "Compare two designs" },
    },
    async ({ a, b, ...rest }) =>
      guard(() => {
        const left = store.get(a);
        const right = store.get(b);
        const d = diffDesigns(left.doc, right.doc, toAnalysisOptions(rest));
        const brief = (v: unknown): string => {
          if (v === undefined || v === null) return "\u2014";
          if (typeof v === "object") {
            const o = v as Record<string, unknown>;
            const n = o.value ?? o.expr ?? o.default;
            if (n !== undefined) return String(n);
          }
          return String(v);
        };
        const text = [
          `${d.a} -> ${d.b} at T=${d.at.T}, B=${d.at.B}`,
          ...(d.identical ? ["structurally identical"] : []),
          ...d.symbols.changed.map((s) => `  ~ ${s.name}  ${brief(s.from)} -> ${brief(s.to)}`),
          ...d.symbols.added.map((s) => `  + ${s.name} = ${brief(s.to)}`),
          ...d.symbols.removed.map((s) => `  - ${s.name}`),
          ...d.blocks.added.map((x) => `  + ${x.path}  ${x.type}`),
          ...d.blocks.removed.map((x) => `  - ${x.path}  ${x.type}`),
          ...d.blocks.changed.map((c) => `  ~ ${c.path}  ${c.params.map((p) => p.key).join(", ")}`),
          "",
          ...d.metrics
            .filter((m) => m.delta !== 0)
            .map((m) => `  ${m.metric}  ${formatCount(m.a)} -> ${formatCount(m.b)}`),
        ]
          .filter((l) => l !== "")
          .join("\n");
        return ok(text, {
          a: d.a,
          b: d.b,
          identical: d.identical,
          at: d.at,
          symbols: {
            added: d.symbols.added.map((x) => x.name),
            removed: d.symbols.removed.map((x) => x.name),
            changed: d.symbols.changed.map((x) => ({
              name: x.name,
              from: brief(x.from),
              to: brief(x.to),
            })),
          },
          blocks: {
            added: d.blocks.added.map((x) => x.path),
            removed: d.blocks.removed.map((x) => x.path),
            changed: d.blocks.changed.map((c) => ({
              path: c.path,
              params: c.params.map((p) => ({ key: p.key, from: brief(p.from), to: brief(p.to) })),
            })),
          },
          edges: { added: d.edges.added.length, removed: d.edges.removed.length },
          metrics: d.metrics,
        });
      }),
  );

  // -- 18. import_hf -------------------------------------------------------
  server.registerTool(
    "tensorcad_import_hf",
    {
      title: "Import a Hugging Face config",
      description:
        "Read a Hugging Face `config.json` into a design and save it in this session. Covers the Llama, Mistral, " +
        "Qwen, Gemma, Mixtral, DeepSeek and GPT-2 families. Anything the importer cannot model faithfully comes " +
        "back as a warning rather than being approximated silently.",
      inputSchema: z.object({
        config: z.string().describe("The contents of config.json."),
        name: z.string().optional().describe("A name for the design; the config's own is used otherwise."),
      }),
      outputSchema: z.object({
        design_id: z.string(),
        name: z.string(),
        params_total: z.number(),
        warnings: z.array(z.string()),
      }),
      annotations: { readOnlyHint: false, idempotentHint: true, openWorldHint: false, title: "Import a config" },
    },
    async ({ config, name }) =>
      guard(() => {
        const result = importHfConfig(config, name);
        const record = store.adopt(result.doc);
        const total = analyze(result.doc).params.total;
        const text = [
          `${result.doc.meta.name}: ${formatCount(total)} parameters`,
          ...result.warnings.map((w) => `  warning: ${w}`),
        ].join("\n");
        return ok(text, {
          design_id: record.design_id,
          name: result.doc.meta.name,
          params_total: total,
          warnings: result.warnings,
        });
      }),
  );
}

/** The tools this server registers, in the order it registers them. */
export const TOOL_NAMES = [
  "tensorcad_list_designs",
  "tensorcad_new_design",
  "tensorcad_open_design",
  "tensorcad_save_design",
  "tensorcad_get_design",
  "tensorcad_get_block",
  "tensorcad_search_catalog",
  "tensorcad_apply_ops",
  "tensorcad_validate",
  "tensorcad_analyze",
  "tensorcad_generate_code",
  "tensorcad_checkpoint",
  "tensorcad_restore",
  "tensorcad_explain",
  "tensorcad_scale",
  "tensorcad_plan",
  "tensorcad_diff",
  "tensorcad_import_hf",
] as const;
