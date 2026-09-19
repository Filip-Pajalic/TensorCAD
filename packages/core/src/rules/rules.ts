/**
 * Design rules: the equivalent of a CAD design-rule check.
 *
 * Each rule is independent and returns findings with a severity and, where
 * possible, a concrete suggestion. Errors mean the design cannot be built as
 * drawn; warnings mean it will build but something is likely wrong or wasteful;
 * info is advisory.
 */

import type { Rule, Finding } from "./types.js";
import { CATALOG, isComposite, isPrimitive, validateUserBlock, type UserBlockDef } from "../catalog/index.js";
import { formatBytes } from "../analysis/kvcache.js";
import { formatCount } from "../analysis/params.js";

/** Head dimensions that fused attention kernels actually support. */
const FLASH_HEAD_DIMS = [32, 64, 96, 128, 160, 192, 256];

const shapeIssues: Rule = {
  id: "shape",
  title: "Tensor interfaces",
  description: "Every edge must carry the shape the receiving port declares.",
  run: ({ infer }) =>
    infer.issues.map<Finding>((i) => ({
      rule: "shape",
      severity: i.severity,
      path: i.path,
      port: i.port,
      message: i.message,
    })),
};

const symbolErrors: Rule = {
  id: "symbols",
  title: "Symbol table",
  description: "Design symbols must evaluate to numbers and must not depend on each other in a cycle.",
  run: ({ symbols }) =>
    symbols.errors.map<Finding>((message) => ({ rule: "symbols", severity: "error", message })),
};

const blockConstraints: Rule = {
  id: "block-constraints",
  title: "Block constraints",
  description: "Per-block checks that hold regardless of which level of the design is on screen.",
  run: ({ flat }) => {
    const out: Finding[] = [];
    for (const b of flat.blocks) {
      const def = b.def;
      const check = isPrimitive(def) || isComposite(def) ? def.constraints : undefined;
      if (!check) continue;
      try {
        for (const f of check(b.resolved)) {
          // The block's own id is kept as the rule, so a finding can be
          // excluded or documented individually rather than as a class.
          out.push({
            rule: f.id,
            severity: f.severity,
            path: b.path,
            param: f.param,
            port: f.port,
            message: f.message,
            hint: f.hint,
          });
        }
      } catch {
        // A block whose parameters failed to resolve is already reported.
      }
    }
    return out;
  },
};

const flashHeadDim: Rule = {
  id: "flash-head-dim",
  title: "Kernel-supported head dimension",
  description: "Fused attention kernels only implement a fixed set of head dimensions.",
  run: ({ flat }) => {
    const out: Finding[] = [];
    const seen = new Set<number>();
    for (const n of flat.nodes) {
      if (n.type !== "sdpa") continue;
      const dh = n.resolved.p.head_dim;
      if (typeof dh !== "number" || seen.has(dh)) continue;
      seen.add(dh);
      if (!FLASH_HEAD_DIMS.includes(dh)) {
        const nearest = FLASH_HEAD_DIMS.reduce((a, b) => (Math.abs(b - dh) < Math.abs(a - dh) ? b : a));
        out.push({
          rule: "flash-head-dim",
          severity: "warning",
          path: n.path,
          message: `Head dimension ${dh} is not one a fused attention kernel supports.`,
          hint: `Use ${nearest} instead, or expect to fall back to an unfused attention path.`,
        });
      }
    }
    return out;
  },
};

const tensorCoreShapes: Rule = {
  id: "tensor-core-multiples",
  title: "Tensor-core friendly widths",
  description: "Matrix dimensions that are not multiples of 64 leave tensor-core throughput on the table.",
  run: ({ flat }) => {
    const out: Finding[] = [];
    const reported = new Set<string>();
    for (const n of flat.nodes) {
      if (n.type !== "linear") continue;
      for (const key of ["in_features", "out_features"]) {
        const v = n.resolved.p[key];
        if (typeof v !== "number" || v % 64 === 0) continue;
        const tag = `${key}:${v}`;
        if (reported.has(tag)) continue;
        reported.add(tag);
        out.push({
          rule: "tensor-core-multiples",
          severity: "info",
          path: n.path,
          message: `Projection width ${v} is not a multiple of 64.`,
          hint: `Rounding up to ${Math.ceil(v / 64) * 64} usually costs little memory and speeds up the matmul.`,
        });
      }
    }
    return out;
  },
};

const vocabPadding: Rule = {
  id: "vocab-padding",
  title: "Vocabulary padding",
  description: "An unpadded vocabulary makes the largest matmul in the model slower than it needs to be.",
  run: ({ flat }) => {
    const out: Finding[] = [];
    for (const n of flat.nodes) {
      if (n.type !== "lm_head") continue;
      const v = n.resolved.p.vocab;
      if (typeof v !== "number" || v % 128 === 0) continue;
      const padded = Math.ceil(v / 128) * 128;
      out.push({
        rule: "vocab-padding",
        severity: "info",
        path: n.path,
        message: `Vocabulary ${v} is not a multiple of 128.`,
        hint: `Padding to ${padded} adds ${formatCount((padded - v) * (n.resolved.p.dim ?? 0))} parameters and speeds up the output projection.`,
      });
    }
    return out;
  },
};

const windowVsContext: Rule = {
  id: "window-vs-context",
  title: "Sliding window against context",
  description: "A sliding window wider than the context does nothing.",
  run: ({ flat, analysis }) => {
    const out: Finding[] = [];
    for (const n of flat.nodes) {
      if (n.type !== "sdpa") continue;
      const w = n.resolved.p.window;
      if (typeof w !== "number" || w <= 0) continue;
      if (w >= analysis.options.T) {
        out.push({
          rule: "window-vs-context",
          severity: "info",
          path: n.path,
          message: `The ${w}-token sliding window is at least as wide as the ${analysis.options.T}-token context, so it has no effect here.`,
          hint: "Analyse at a longer context to see what the window buys.",
        });
      }
    }
    return out;
  },
};

const inferenceFits: Rule = {
  id: "inference-fits",
  title: "Serving footprint",
  description: "Weights plus cache must fit in device memory at the target context and concurrency.",
  run: ({ analysis }) => {
    const { memory, options } = analysis;
    const cap = options.hardware.memory;
    if (memory.infer.total <= cap) return [];
    return [
      {
        rule: "inference-fits",
        severity: "warning",
        message:
          `Serving needs about ${formatBytes(memory.infer.total)} but ${options.hardware.name} has ` +
          `${formatBytes(cap)}. Weights are ${formatBytes(memory.infer.weights)} and the cache is ` +
          `${formatBytes(memory.infer.kv)} at ${options.T} tokens across ${options.concurrency} sequence(s).`,
        hint:
          memory.infer.kv > memory.infer.weights
            ? "The cache dominates. Reduce KV heads, shorten the context, or move some layers to a sliding window."
            : `Quantize the weights, or split the model across ${Math.ceil(memory.infer.total / cap)} devices.`,
      },
    ];
  },
};

const trainingFits: Rule = {
  id: "training-fits",
  title: "Training footprint",
  description: "Weights, gradients, optimizer state and activations must fit on each GPU.",
  run: ({ analysis }) => {
    const { memory, options } = analysis;
    const cap = options.hardware.memory;
    const per = memory.train.perGpu;
    if (per.total <= cap) return [];

    const dominant = Math.max(per.weights, per.grads, per.optimizer, per.activations);
    let hint: string;
    if (dominant === per.activations) {
      hint =
        options.recompute === "full"
          ? "Activations still dominate. Reduce the micro-batch or the context length."
          : "Activations dominate. Turn on activation recomputation, or reduce the micro-batch.";
    } else if (options.parallel.zero < 3) {
      hint = `Model state dominates. Raise the ZeRO stage from ${options.parallel.zero} to 3, or increase tensor parallelism.`;
    } else {
      hint = "Model state dominates even when fully sharded. Add more data-parallel replicas or use a smaller optimizer state.";
    }

    return [
      {
        rule: "training-fits",
        severity: "warning",
        message:
          `Training needs about ${formatBytes(per.total)} per GPU but ${options.hardware.name} has ${formatBytes(cap)}. ` +
          `Weights ${formatBytes(per.weights)}, gradients ${formatBytes(per.grads)}, optimizer ${formatBytes(per.optimizer)}, ` +
          `activations ${formatBytes(per.activations)}.`,
        hint,
      },
    ];
  },
};

const logitsMemory: Rule = {
  id: "logits-memory",
  title: "Logits buffer",
  description: "At a large vocabulary the logits tensor can rival the rest of the activations.",
  run: ({ analysis }) => {
    const { memory } = analysis;
    if (memory.train.activations <= 0) return [];
    const share = memory.train.logits / memory.train.activations;
    if (share < 0.25) return [];
    return [
      {
        rule: "logits-memory",
        severity: "info",
        message: `The logits buffer is ${formatBytes(memory.train.logits)}, which is ${(share * 100).toFixed(0)}% of all activation memory.`,
        hint: "Compute the cross-entropy in chunks over the sequence so the full logits tensor is never materialized.",
      },
    ];
  },
};

const recomputeHint: Rule = {
  id: "recompute-hint",
  title: "Activation recomputation",
  description: "Flags when recomputation would free a large share of memory.",
  run: ({ analysis }) => {
    const { memory, options } = analysis;
    if (options.recompute !== "none") return [];
    const per = memory.train.perGpu;
    if (per.total <= 0) return [];
    const share = per.activations / per.total;
    if (share < 0.5) return [];
    return [
      {
        rule: "recompute-hint",
        severity: "info",
        message: `Activations are ${(share * 100).toFixed(0)}% of the training footprint at batch ${options.B} and ${options.T} tokens.`,
        hint: "Full recomputation trades about 33% more compute for most of that memory.",
      },
    ];
  },
};

const attentionShare: Rule = {
  id: "attention-share",
  title: "Attention share of compute",
  description: "Warns when the sequence-dependent term makes the 6N rule misleading.",
  run: ({ analysis }) => {
    const share = analysis.flops.attentionShare;
    if (share < 0.2) return [];
    return [
      {
        rule: "attention-share",
        severity: "info",
        message: `At ${analysis.options.T} tokens, attention is ${(share * 100).toFixed(0)}% of forward FLOPs.`,
        hint: "The 2N and 6N rules of thumb quietly stop working here. Compare against the reported attention term rather than the parameter count.",
      },
    ];
  },
};

const chinchillaRatio: Rule = {
  id: "chinchilla",
  title: "Token budget",
  description: "Compares the training budget against the compute-optimal ratio.",
  run: ({ analysis }) => {
    const c = analysis.chinchilla;
    if (analysis.options.tokensWereDefaulted) return [];
    if (c.overTrainingRatio >= 0.5 && c.overTrainingRatio <= 20) return [];
    return [
      {
        rule: "chinchilla",
        severity: "info",
        message: `${(analysis.options.tokens / 1e9).toFixed(0)}B tokens is ${c.overTrainingRatio.toFixed(1)}x the compute-optimal budget of ${(c.optimalTokens / 1e9).toFixed(0)}B. ${c.verdict}`,
        hint:
          c.overTrainingRatio < 0.5
            ? "Either train longer or shrink the model; as drawn, most of the parameters will not be paid for."
            : "This is fine when inference cost matters more than training cost, which is usually why it is done.",
      },
    ];
  },
};

const unusedSymbols: Rule = {
  id: "unused-symbol",
  title: "Unused symbols",
  description: "A symbol nothing refers to is usually a leftover from an edit.",
  run: ({ doc, symbols, flat }) => {
    const used = new Set<string>();
    for (const b of flat.blocks) {
      for (const sym of Object.values(b.resolved.s)) {
        for (const name of sym.symbols()) used.add(name);
      }
    }
    // Shape patterns reference runtime symbols directly.
    for (const name of symbols.runtime) used.add(name);
    // A symbol used by another symbol's expression counts as used.
    for (const name of Object.keys(doc.symbols)) {
      const def = doc.symbols[name];
      const expr =
        typeof def === "string" ? def : def && typeof def === "object" && def.kind === "design" && typeof def.value === "string" ? def.value : null;
      if (!expr) continue;
      for (const dep of symbols.order) if (expr.includes(dep)) used.add(dep);
    }

    const out: Finding[] = [];
    for (const name of Object.keys(doc.symbols)) {
      if (used.has(name)) continue;
      out.push({
        rule: "unused-symbol",
        severity: "info",
        message: `Symbol "${name}" is not referenced by any block.`,
        hint: "Remove it, or wire it into the parameter that should follow it.",
      });
    }
    return out;
  },
};

const danglingOutputs: Rule = {
  id: "dangling-output",
  title: "Unused outputs",
  description: "A computed tensor nobody consumes is dead weight.",
  run: ({ doc, infer }) => {
    const out: Finding[] = [];
    // An output is used when it appears as the *source* of an edge.
    const consumed = new Set(doc.graph.edges.map(([from]) => from));
    for (const node of doc.graph.nodes) {
      if (node.type === "output" || node.type === "boundary_out") continue;
      const ports = infer.ports.get(node.id);
      if (!ports) continue;
      for (const portName of Object.keys(ports.out)) {
        if (consumed.has(`${node.id}:${portName}`)) continue;
        out.push({
          rule: "dangling-output",
          severity: "warning",
          path: node.id,
          port: portName,
          message: `Output "${portName}" is not connected to anything.`,
          hint: "Connect it downstream or delete the block.",
        });
      }
    }
    return out;
  },
};

const publishedDrift: Rule = {
  id: "published-drift",
  title: "Drift from the published model",
  description: "A preset whose parameter count no longer matches its source has drifted.",
  run: ({ doc, analysis }) => {
    const published = doc.meta.published?.params;
    if (!published) return [];
    const tolerance = doc.meta.published?.tolerance ?? 0.005;
    const delta = Math.abs(analysis.params.total - published) / published;
    if (delta < tolerance) return [];
    return [
      {
        rule: "published-drift",
        severity: "warning",
        message:
          `This design computes ${formatCount(analysis.params.total)} parameters but is recorded as ` +
          `${formatCount(published)} (${(delta * 100).toFixed(2)}% off, tolerance ${(tolerance * 100).toFixed(1)}%).`,
        hint: doc.meta.published?.source ? `Compare against ${doc.meta.published.source}` : undefined,
      },
    ];
  },
};

const activeParamsDrift: Rule = {
  id: "active-params-drift",
  title: "Active parameter drift",
  description: "A sparse design's active parameter count should match what its authors report.",
  run: ({ doc, analysis }) => {
    const published = doc.meta.published?.activeParams;
    if (!published) return [];
    const tolerance = doc.meta.published?.tolerance ?? 0.005;
    const delta = Math.abs(analysis.params.active - published) / published;
    if (delta < tolerance) return [];
    return [
      {
        rule: "active-params-drift",
        severity: "warning",
        message:
          `This design activates ${formatCount(analysis.params.active)} parameters per token but is ` +
          `recorded as ${formatCount(published)} (${(delta * 100).toFixed(2)}% off).`,
      },
    ];
  },
};

/**
 * Blocks the design defines for itself.
 *
 * A definition that fails to compile is dropped from the catalog rather than
 * thrown, so without this rule the only symptom would be "unknown block type"
 * against every instance of it, which points at the wrong thing.
 */
const userBlocks: Rule = {
  id: "user-blocks",
  title: "Block definitions",
  description: "Blocks a design defines for itself must name real parameters and real ports.",
  run: ({ doc }) => {
    const defs = doc.defs as Record<string, UserBlockDef> | undefined;
    if (!defs) return [];
    const builtIn = new Set(Object.keys(CATALOG));
    const out: Finding[] = [];
    for (const [type, def] of Object.entries(defs)) {
      for (const message of validateUserBlock({ ...def, type }, builtIn)) {
        out.push({
          rule: "user-blocks",
          severity: "error",
          message: `Block "${type}": ${message}`,
          hint: "Edit the definition, or delete it and rebuild the block.",
        });
      }
    }
    return out;
  },
};

export const RULES: Rule[] = [
  userBlocks,
  shapeIssues,
  symbolErrors,
  blockConstraints,
  flashHeadDim,
  tensorCoreShapes,
  vocabPadding,
  windowVsContext,
  inferenceFits,
  trainingFits,
  logitsMemory,
  recomputeHint,
  attentionShare,
  chinchillaRatio,
  unusedSymbols,
  danglingOutputs,
  publishedDrift,
  activeParamsDrift,
];
