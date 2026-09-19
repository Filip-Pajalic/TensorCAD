/**
 * Turning command-line flags into `AnalysisOptions`.
 *
 * Shared by `validate` and `analyze` so both commands take the same knobs.
 */

import { HARDWARE, type AnalysisOptions } from "@tensorcad/core";
import { num, str, UsageError, type Args } from "./args.js";

const DTYPES = ["fp32", "bf16", "fp16", "fp8"] as const;
const OPTIMIZERS = ["adamw", "adamw8bit", "muon", "sgd_momentum", "sgd", "bf16_adam"] as const;
const RECOMPUTE = ["none", "selective", "full"] as const;

function oneOf<T extends string>(args: Args, name: string, values: readonly T[]): T | undefined {
  const v = str(args, name);
  if (v === undefined) return undefined;
  if (!(values as readonly string[]).includes(v)) {
    throw new UsageError(`--${name} expects one of ${values.join(", ")}, got "${v}"`);
  }
  return v as T;
}

export function analysisOptions(args: Args): AnalysisOptions {
  const out: AnalysisOptions = {};

  const T = num(args, "T");
  if (T !== undefined) out.T = T;
  const B = num(args, "B");
  if (B !== undefined) out.B = B;

  const hardware = str(args, "hardware");
  if (hardware !== undefined) {
    if (!HARDWARE.some((h) => h.id === hardware)) {
      throw new UsageError(
        `Unknown hardware "${hardware}". Known: ${HARDWARE.map((h) => h.id).join(", ")}`,
      );
    }
    out.hardware = hardware;
  }

  const gpus = num(args, "gpus");
  if (gpus !== undefined) out.gpus = gpus;
  const tokens = num(args, "tokens");
  if (tokens !== undefined) out.tokens = tokens;
  const mfu = num(args, "mfu");
  if (mfu !== undefined) out.mfu = mfu;
  const concurrency = num(args, "concurrency");
  if (concurrency !== undefined) out.concurrency = concurrency;

  const dtype = oneOf(args, "dtype", DTYPES);
  if (dtype) out.dtype = dtype;
  const optimizer = oneOf(args, "optimizer", OPTIMIZERS);
  if (optimizer) out.optimizer = optimizer;
  const recompute = oneOf(args, "recompute", RECOMPUTE);
  if (recompute) out.recompute = recompute;

  const parallel: NonNullable<AnalysisOptions["parallel"]> = {};
  const zero = num(args, "zero");
  if (zero !== undefined) {
    if (![0, 1, 2, 3].includes(zero)) throw new UsageError(`--zero expects 0, 1, 2 or 3, got ${zero}`);
    parallel.zero = zero as 0 | 1 | 2 | 3;
  }
  for (const key of ["tp", "dp", "pp", "ep"] as const) {
    const v = num(args, key);
    if (v !== undefined) parallel[key] = v;
  }
  if (Object.keys(parallel).length > 0) out.parallel = parallel;

  return out;
}

/**
 * `--gpus` is the honest default when a parallel plan is given but the GPU
 * count is not: dp x tp x pp.
 */
export function impliedGpus(options: AnalysisOptions): AnalysisOptions {
  if (options.gpus !== undefined || !options.parallel) return options;
  const { dp = 1, tp = 1, pp = 1 } = options.parallel;
  const gpus = dp * tp * pp;
  return gpus > 1 ? { ...options, gpus } : options;
}
