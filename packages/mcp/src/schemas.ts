/**
 * The wire contract.
 *
 * Every tool declares both an `inputSchema` and an `outputSchema`, so a client
 * can validate what it gets back and a model can see the shape of an answer
 * before it asks for one.
 */

import * as z from "zod";

// ---------------------------------------------------------------------------
// Shared fragments
// ---------------------------------------------------------------------------

export const DESIGN_ID = z
  .string()
  .describe("Handle returned by tensorcad_new_design or tensorcad_open_design, e.g. \"dsn_1\".");

export const Severity = z.enum(["error", "warning", "info"]);

export const Finding = z.object({
  rule: z.string().describe("Stable rule id, e.g. \"flash-head-dim\"."),
  severity: Severity,
  path: z.string().optional().describe("Block path the finding is about."),
  port: z.string().optional(),
  message: z.string(),
  hint: z.string().optional().describe("What to change to clear the finding."),
});

export const Counts = z.object({
  error: z.number().int(),
  warning: z.number().int(),
  info: z.number().int(),
});

export const ValidationSummary = z.object({
  ok: z.boolean().describe("True when nothing blocks building this design."),
  counts: Counts,
  top_findings: z.array(Finding).describe("The worst few findings. Call tensorcad_validate for all of them."),
});

export const DesignSummary = z.object({
  design_id: z.string(),
  name: z.string(),
  revision: z.number().int(),
  path: z.string().optional(),
  source: z.enum(["preset", "file", "empty"]),
  dirty: z.boolean(),
  created_at: z.string(),
  updated_at: z.string(),
});

export const OutlineSymbol = z.object({
  name: z.string(),
  kind: z.enum(["design", "runtime"]),
  value: z.string().describe("Literal or expression as written in the document."),
  resolved: z.number().optional(),
  doc: z.string().optional(),
});

export const OutlineBlock = z.object({
  path: z.string().describe("Slash-separated path, e.g. \"layers/block\"."),
  type: z.string(),
  kind: z.string().describe("primitive, composite, container or unknown."),
  depth: z.number().int(),
  label: z.string().optional(),
  params: z.number().describe("Trainable parameters under this path, repeats included."),
  repeat: z.number().optional().describe("Instance count, for repeat containers."),
});

export const OutlineEdge = z.object({
  graph: z.string().describe("Container path holding the edge; \"\" is the root graph."),
  from: z.string(),
  to: z.string(),
  shape: z.string().optional().describe("Inferred shape on the wire, e.g. \"B T D\"."),
});

export const Outline = z.object({
  name: z.string(),
  family: z.string().optional(),
  notes: z.string().optional(),
  symbols: z.array(OutlineSymbol),
  blocks: z.array(OutlineBlock),
  edges: z.array(OutlineEdge),
  params_total: z.number(),
  params_active: z.number(),
  issues: z.number().int(),
});

export const BlockPort = z.object({
  name: z.string(),
  pattern: z.string().describe("Declared shape pattern, e.g. \"B T D\"."),
  shape: z.string().optional().describe("Shape actually inferred for this port."),
  dtype: z
    .string()
    .optional()
    .describe("What the tensor carries, when the port declares it rather than inheriting it."),
  optional: z.boolean().optional().describe("True when this port may legitimately dangle."),
  connected_to: z.array(z.string()).optional(),
});

/** The knobs `tensorcad_analyze` and `tensorcad_validate` share. */
export const analysisOptionsShape = {
  T: z.number().int().positive().optional().describe("Sequence length. Defaults to the document's own T."),
  B: z.number().int().positive().optional().describe("Micro-batch size."),
  dtype: z.enum(["fp32", "bf16", "fp16", "fp8"]).optional().describe("Training dtype. Default bf16."),
  hardware: z
    .string()
    .optional()
    .describe("Hardware id: h100-sxm, h200-sxm, b200, a100-80, rtx5080 or rtx4090. Default h100-sxm."),
  gpus: z.number().int().positive().optional(),
  tokens: z.number().positive().optional().describe("Training token budget. Defaults to Chinchilla-optimal."),
  optimizer: z.enum(["adamw", "adamw8bit", "muon", "sgd_momentum", "sgd", "bf16_adam"]).optional(),
  recompute: z.enum(["none", "selective", "full"]).optional(),
  zero: z.number().int().min(0).max(3).optional().describe("ZeRO/FSDP sharding stage."),
  tp: z.number().int().positive().optional().describe("Tensor parallel degree."),
  dp: z.number().int().positive().optional().describe("Data parallel degree."),
  pp: z.number().int().positive().optional().describe("Pipeline parallel degree."),
  ep: z.number().int().positive().optional().describe("Expert parallel degree."),
  concurrency: z.number().int().positive().optional().describe("Concurrent sequences when serving."),
  mfu: z.number().positive().max(1).optional().describe("Model FLOPs utilization, 0..1."),
};

// ---------------------------------------------------------------------------
// Edit operations
// ---------------------------------------------------------------------------

const ParamValue = z
  .any()
  .describe("A number, an expression string over the design symbols, a boolean, null, or an object.");

export const Op = z.discriminatedUnion("op", [
  z.object({
    op: z.literal("add_node"),
    parent: z.string().optional().describe("Container path to add into. Omit for the root graph."),
    id: z.string().describe("New block id, unique within its graph."),
    type: z.string().describe("Catalog block type; see tensorcad_search_catalog."),
    params: z.record(z.string(), ParamValue).optional(),
    label: z.string().optional(),
  }),
  z.object({ op: z.literal("remove_node"), path: z.string().describe("Block path. Its edges go with it.") }),
  z.object({
    op: z.literal("set_param"),
    path: z.string(),
    key: z.string(),
    value: ParamValue,
  }),
  z.object({
    op: z.literal("connect"),
    graph: z.string().optional().describe("Container path holding the edge. Omit for the root graph."),
    from: z.string().describe("Producer endpoint, \"blockId:port\", local to that graph."),
    to: z.string().describe("Consumer endpoint, \"blockId:port\", local to that graph."),
  }),
  z.object({
    op: z.literal("disconnect"),
    graph: z.string().optional(),
    from: z.string(),
    to: z.string(),
  }),
  z.object({
    op: z.literal("set_symbol"),
    name: z.string(),
    value: z
      .union([z.number(), z.string(), z.null()])
      .describe("Number, expression over earlier symbols, or null to delete the symbol."),
    doc: z.string().optional(),
    runtime: z.boolean().optional().describe("Keep the symbol indeterminate (B, T). Defaults to the current kind."),
  }),
  z.object({ op: z.literal("rename"), path: z.string(), id: z.string().describe("New id; edges are rewritten.") }),
  z.object({
    op: z.literal("set_label"),
    path: z.string(),
    label: z.string().nullable().optional().describe("Null or empty clears the label."),
  }),
]);

// ---------------------------------------------------------------------------
// Analysis output
// ---------------------------------------------------------------------------

/** Non-finite numbers are reported as null rather than being dropped. */
const num = () => z.number().nullable();

export const AnalysisOutput = z.object({
  design_id: z.string(),
  revision: z.number().int(),
  name: z.string(),
  options: z.object({
    T: z.number(),
    B: z.number(),
    dtype: z.string(),
    hardware: z.string(),
    gpus: z.number(),
    parallel: z.object({
      dp: z.number(),
      tp: z.number(),
      pp: z.number(),
      ep: z.number(),
      zero: z.number(),
      sequenceParallel: z.boolean(),
    }),
    optimizer: z.string(),
    recompute: z.string(),
    tokens: z.number(),
    tokens_were_defaulted: z.boolean(),
    mfu: z.number(),
    concurrency: z.number(),
  }),
  params: z.object({
    total: z.number(),
    active: z.number(),
    embedding: z.number(),
    head: z.number(),
    non_embedding: z.number(),
    non_embedding_active: z.number(),
    by_category: z.record(z.string(), z.number()),
    by_type: z.record(z.string(), z.number()),
  }),
  flops: z.object({
    fwd_dense: num(),
    fwd_attention: num(),
    fwd_total: num(),
    elementwise: num(),
    train_per_token: num(),
    attention_share: num(),
  }),
  kv: z.object({ bytes_per_token: num(), bytes_per_sequence: num() }),
  memory: z.object({
    optimizer_label: z.string(),
    train_weights: num(),
    train_grads: num(),
    train_optimizer: num(),
    train_activations: num(),
    train_per_gpu: num(),
    train_total: num(),
    infer_weights: num(),
    infer_kv: num(),
    infer_total: num(),
    device_memory: num(),
    notes: z.array(z.string()),
  }),
  throughput: z.object({
    decode_tokens_per_second: num(),
    decode_weight_bytes: num().describe("What a step at this batch reads: for a mixture of experts, the union of what its tokens routed to."),
    resident_weight_bytes: num().describe("Every weight the device holds, read or not."),
    prefill_seconds: num(),
    memory_bound: z.boolean(),
    notes: z.array(z.string()),
  }),
  cost: z.object({
    total_flops: num(),
    gpu_hours: num(),
    wall_clock_hours: num(),
    dollars: num(),
    tokens: z.number(),
  }),
  chinchilla: z.object({
    optimal_tokens: num(),
    tokens_per_param: num(),
    over_training_ratio: num(),
    verdict: z.string(),
  }),
  errors: z.array(z.string()),
});

export type OpInput = z.infer<typeof Op>;
