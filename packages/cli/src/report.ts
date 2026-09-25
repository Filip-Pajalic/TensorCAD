/**
 * The analysis report, in both of its shapes.
 *
 * `analysisJson` is the machine-readable projection (the `AnalysisResult`
 * carries Maps, which do not survive `JSON.stringify`), and `analysisText` is
 * what a person reads.
 */

import { bold, dim, finite, heading, percent, rows } from "./format.js";
import type { AnalysisResult } from "@tensor-cad/engine";
import { formatBytes, formatCount, formatDollars, formatFlops, formatHours } from "@tensor-cad/engine";

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

/** JSON does not have Infinity or NaN; report those as null. */
function n(v: number): number | null {
  return Number.isFinite(v) ? v : null;
}

export function analysisJson(a: AnalysisResult): Record<string, unknown> {
  const o = a.options;
  return {
    name: a.name,
    options: {
      T: o.T,
      B: o.B,
      dtype: o.dtype,
      inference_dtype: o.inferenceDtype,
      kv_dtype: o.kvDtype,
      hardware: o.hardware.id,
      gpus: o.gpus,
      parallel: o.parallel,
      optimizer: o.optimizer,
      recompute: o.recompute,
      flash: o.flash,
      tokens: o.tokens,
      tokens_were_defaulted: o.tokensWereDefaulted,
      mfu: o.mfu,
      concurrency: o.concurrency,
      ...(o.packing ? { packing: { ...o.packing } } : {}),
    },
    params: {
      total: a.params.total,
      active: a.params.active,
      embedding: a.params.embedding,
      head: a.params.head,
      non_embedding: a.params.nonEmbedding,
      non_embedding_active: a.params.nonEmbeddingActive,
      by_category: a.params.byCategory,
      by_type: a.params.byType,
    },
    flops: {
      fwd_dense: n(a.flops.fwdDense),
      fwd_attention: n(a.flops.fwdAttention),
      fwd_total: n(a.flops.fwdTotal),
      elementwise: n(a.flops.elementwise),
      train_per_token: n(a.flops.trainPerToken),
      attention_share: n(a.flops.attentionShare),
      by_category: a.flops.byCategory,
      ...(a.flops.packed
        ? {
            packed: {
              fwd_attention: n(a.flops.packed.fwdAttention),
              fwd_total: n(a.flops.packed.fwdTotal),
              train_per_token: n(a.flops.packed.trainPerToken),
              attention_share: n(a.flops.packed.attentionShare),
              fwd_attention_blocks: n(a.flops.packed.fwdAttentionBlocks),
            },
          }
        : {}),
    },
    kv: {
      bytes_per_token: n(a.kv.bytesPerToken),
      bytes_per_sequence_fixed: n(a.kv.bytesPerSequenceFixed),
      bytes_per_sequence: n(a.kv.bytesPerToken * a.options.T + a.kv.bytesPerSequenceFixed),
      bytes_per_token_decompressed: n(a.kv.bytesPerTokenDecompressed),
    },
    memory: {
      weights_bytes: n(a.memory.weightsBytes),
      optimizer_label: a.memory.optimizerLabel,
      train: {
        weights: n(a.memory.train.weights),
        grads: n(a.memory.train.grads),
        optimizer: n(a.memory.train.optimizer),
        activations: n(a.memory.train.activations),
        logits: n(a.memory.train.logits),
        total: n(a.memory.train.total),
        per_gpu: {
          weights: n(a.memory.train.perGpu.weights),
          grads: n(a.memory.train.perGpu.grads),
          optimizer: n(a.memory.train.perGpu.optimizer),
          activations: n(a.memory.train.perGpu.activations),
          total: n(a.memory.train.perGpu.total),
        },
      },
      infer: {
        weights: n(a.memory.infer.weights),
        kv: n(a.memory.infer.kv),
        overhead: n(a.memory.infer.overhead),
        total: n(a.memory.infer.total),
      },
      notes: a.memory.notes,
    },
    throughput: {
      decode_tokens_per_second: n(a.throughput.decodeTokensPerSecond),
      decode_seconds_per_step: n(a.throughput.decodeSecondsPerStep),
      decode_bytes_per_step: n(a.throughput.decodeBytesPerStep),
      decode_flops_per_step: n(a.throughput.decodeFlopsPerStep),
      memory_bound: a.throughput.memoryBound,
      ridge_point: n(a.throughput.ridgePoint),
      prefill_seconds: n(a.throughput.prefillSeconds),
      notes: a.throughput.notes,
    },
    cost: {
      total_flops: n(a.cost.totalFlops),
      gpu_hours: n(a.cost.gpuHours),
      wall_clock_hours: n(a.cost.wallClockHours),
      dollars: n(a.cost.dollars),
      tokens: a.cost.tokens,
      mfu: a.cost.mfu,
    },
    chinchilla: {
      optimal_tokens: n(a.chinchilla.optimalTokens),
      tokens_per_param: n(a.chinchilla.tokensPerParam),
      tokens_per_active_param: n(a.chinchilla.tokensPerActiveParam),
      over_training_ratio: n(a.chinchilla.overTrainingRatio),
      predicted_loss: a.chinchilla.predictedLoss,
      verdict: a.chinchilla.verdict,
    },
    errors: a.errors,
  };
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

function topShare(by: Record<string, number>, total: number, limit = 6): [string, string, string?][] {
  return Object.entries(by)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([k, v]) => [k, formatCount(v), total > 0 ? percent(v / total) : undefined]);
}

export function analysisText(a: AnalysisResult, opts: { title?: string } = {}): string {
  const o = a.options;
  const out: string[] = [];
  const T = o.T;

  out.push(bold(opts.title ?? a.name));

  out.push(heading("Configuration"));
  out.push(
    rows([
      ["sequence T", String(T), "tokens"],
      ["micro-batch B", String(o.B)],
      ["dtype", o.dtype, o.inferenceDtype === o.dtype ? undefined : `serving ${o.inferenceDtype}`],
      ["hardware", `${o.hardware.name}`, `${o.hardware.id} x${o.gpus}`],
      ["optimizer", o.optimizer, a.memory.optimizerLabel],
      ["recompute", o.recompute],
      [
        "parallel",
        `dp${o.parallel.dp} tp${o.parallel.tp} pp${o.parallel.pp} ep${o.parallel.ep}`,
        `zero${o.parallel.zero}${o.parallel.sequenceParallel ? " +sp" : ""}`,
      ],
    ]),
  );

  out.push(heading("Parameters"));
  out.push(
    rows([
      ["total", formatCount(a.params.total), a.params.total.toLocaleString("en-US")],
      ["active", formatCount(a.params.active), a.params.active === a.params.total ? "dense" : "per token"],
      ["non-embedding", formatCount(a.params.nonEmbedding)],
      ["embedding", formatCount(a.params.embedding)],
      ["lm head", formatCount(a.params.head)],
    ]),
  );
  const byCategory = topShare(a.params.byCategory, a.params.total);
  if (byCategory.length > 0) {
    out.push(dim("  by category"));
    out.push(rows(byCategory, "    "));
  }
  out.push(heading("FLOPs per token"));
  out.push(
    rows([
      ["forward dense", finite(a.flops.fwdDense, formatFlops)],
      ["forward attention", finite(a.flops.fwdAttention, formatFlops), `at T=${T}`],
      ["forward total", finite(a.flops.fwdTotal, formatFlops)],
      ["elementwise", finite(a.flops.elementwise, formatFlops), "not in the 6N total"],
      ["training fwd+bwd", finite(a.flops.trainPerToken, formatFlops)],
      ["attention share", percent(a.flops.attentionShare)],
    ]),
  );
  const packed = a.flops.packed;
  if (packed && o.packing) {
    // Training rows packed with documents the mask keeps apart. The figures
    // above are one document a row, which is what serving is.
    out.push(dim(`  packed training, documents of ${o.packing.mean} tokens (spread ${o.packing.spread})`));
    out.push(
      rows(
        [
          ["forward attention", finite(packed.fwdAttention, formatFlops)],
          [
            "  as the kernel runs it",
            finite(packed.fwdAttentionBlocks, formatFlops),
            `${(packed.fwdAttentionBlocks / packed.fwdAttention).toFixed(2)}x: whole 128-token blocks`,
          ],
          ["training fwd+bwd", finite(packed.trainPerToken, formatFlops), "what the cost is counted from"],
          ["attention share", percent(packed.attentionShare)],
        ],
        "    ",
      ),
    );
  }

  out.push(heading("KV cache"));
  out.push(
    rows([
      ["per token", finite(a.kv.bytesPerToken, formatBytes), `${o.kvDtype} cache`],
      ["per sequence", finite(a.kv.bytesPerToken * T + a.kv.bytesPerSequenceFixed, formatBytes), `T=${T}`],
      ["fixed per sequence", finite(a.kv.bytesPerSequenceFixed, formatBytes)],
      // Only latent attention has a second answer; everything else caches the
      // same bytes however the kernel is written.
      ...(a.kv.bytesPerTokenDecompressed > a.kv.bytesPerToken
        ? ([
            [
              "if decompressed",
              finite(a.kv.bytesPerTokenDecompressed, formatBytes),
              `${(a.kv.bytesPerTokenDecompressed / a.kv.bytesPerToken).toFixed(0)}x, per token`,
            ],
          ] as [string, string, string][])
        : []),
    ]),
  );

  out.push(heading(`Training memory${o.gpus > 1 ? ` (${o.gpus} GPUs)` : ""}`));
  const perGpu = a.memory.train.perGpu;
  out.push(
    rows([
      ["weights", finite(perGpu.weights, formatBytes)],
      ["gradients", finite(perGpu.grads, formatBytes)],
      ["optimizer", finite(perGpu.optimizer, formatBytes)],
      [
        "activations",
        finite(perGpu.activations, formatBytes),
        `B=${o.B} T=${T}, logits ${finite(a.memory.train.logits, formatBytes)}`,
      ],
      ["total per GPU", finite(perGpu.total, formatBytes), `fits ${finite(o.hardware.memory, formatBytes)}? ${perGpu.total <= o.hardware.memory ? "yes" : "no"}`],
      ["total all GPUs", finite(a.memory.train.total, formatBytes)],
    ]),
  );
  for (const note of a.memory.notes) out.push(dim(`  ${note}`));

  out.push(heading("Serving memory"));
  out.push(
    rows([
      ["weights", finite(a.memory.infer.weights, formatBytes), o.inferenceDtype],
      ["kv cache", finite(a.memory.infer.kv, formatBytes), `${o.concurrency} concurrent x T=${T}`],
      ["overhead", finite(a.memory.infer.overhead, formatBytes)],
      ["total", finite(a.memory.infer.total, formatBytes)],
    ]),
  );

  out.push(heading("Throughput"));
  out.push(
    rows([
      [
        "decode",
        `${finite(a.throughput.decodeTokensPerSecond, (v) => v.toFixed(1))} tok/s`,
        a.throughput.memoryBound ? "memory bound" : "compute bound",
      ],
      ["per step", `${finite(a.throughput.decodeSecondsPerStep * 1000, (v) => v.toFixed(2))} ms`],
      ["bytes per step", finite(a.throughput.decodeBytesPerStep, formatBytes)],
      ["prefill", `${finite(a.throughput.prefillSeconds, (v) => v.toFixed(2))} s`, `T=${T}`],
      ["ridge point", finite(a.throughput.ridgePoint, (v) => `${v.toFixed(0)} FLOP/B`)],
    ]),
  );
  for (const note of a.throughput.notes) out.push(dim(`  ${note}`));

  out.push(heading("Training cost"));
  out.push(
    rows([
      [
        "tokens",
        formatCount(a.cost.tokens),
        o.tokensWereDefaulted ? "Chinchilla default (20x non-embedding)" : "given",
      ],
      ["total FLOPs", finite(a.cost.totalFlops, formatFlops)],
      ["gpu-hours", finite(a.cost.gpuHours, formatHours)],
      ["wall clock", finite(a.cost.wallClockHours, formatHours), `${o.gpus} x ${o.hardware.id} at MFU ${percent(a.cost.mfu, 0)}`],
      ["cost", finite(a.cost.dollars, formatDollars), `$${o.hardware.pricePerHour.toFixed(2)}/GPU-h`],
    ]),
  );

  out.push(heading("Chinchilla"));
  out.push(
    rows([
      ["optimal tokens", finite(a.chinchilla.optimalTokens, formatCount)],
      ["tokens / param", finite(a.chinchilla.tokensPerParam, (v) => v.toFixed(1))],
      ["tokens / active param", finite(a.chinchilla.tokensPerActiveParam, (v) => v.toFixed(1))],
      ["over-training", finite(a.chinchilla.overTrainingRatio, (v) => `${v.toFixed(2)}x`)],
      ...Object.entries(a.chinchilla.predictedLoss).map(
        ([k, v]) => [`predicted loss (${k})`, finite(v, (x) => x.toFixed(3))] as [string, string],
      ),
    ]),
  );
  out.push(dim(`  ${a.chinchilla.verdict}`));

  if (a.errors.length > 0) {
    out.push(heading("Errors"));
    for (const e of a.errors) out.push(`  ${e}`);
  }

  return out.join("\n");
}
