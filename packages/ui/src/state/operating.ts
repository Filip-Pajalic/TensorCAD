import type { AnalysisOptions, Dtype, OptimizerKind } from "@tensorcad/engine";
import { DEFAULT_HARDWARE, DEFAULT_PARALLEL } from "@tensorcad/engine";
/**
 * The operating point.
 *
 * Every number past the parameter count depends on conditions the design does
 * not contain: how long the sequence is, what precision the weights are in,
 * which device runs it, how many of them there are. CAD calls this a study;
 * a circuit simulator calls it an operating point. Whatever the name, the
 * numbers on screen are meaningless without it, so it is a visible, editable
 * part of the editor rather than a set of hidden defaults.
 *
 * It is editor state, not document state: the same design analysed on two
 * different machines is one design, not two.
 */


export type Recompute = "none" | "selective" | "full";

export interface OperatingPoint {
  /** Micro-batch. */
  B: number;
  /** Sequence length. Null means whatever the document declares. */
  T: number | null;
  /** Training precision for weights and activations. */
  dtype: Dtype;
  /** Serving precision, which is usually smaller. */
  inferenceDtype: Dtype;
  hardware: string;
  gpus: number;
  optimizer: OptimizerKind;
  recompute: Recompute;
  /** Assume a memory-efficient attention kernel. */
  flash: boolean;
  /** ZeRO/FSDP stage. */
  zero: 0 | 1 | 2 | 3;
  /** Tensor-parallel degree. */
  tp: number;
  /** Pipeline stages. */
  pp: number;
  /** Expert-parallel degree. Meaningless on a design with no experts. */
  ep: number;
  /**
   * Shard the activations along the sequence across the tensor-parallel group.
   *
   * Only means anything when tp > 1, and then it is most of what tensor
   * parallelism is worth: without it the norms and the dropouts stay
   * replicated on every rank.
   */
  sequenceParallel: boolean;
  /** Concurrent sequences when serving. */
  concurrency: number;
  /** Training token budget. Null means the Chinchilla-optimal budget. */
  tokens: number | null;
}

export const DEFAULT_OPERATING: OperatingPoint = {
  B: 1,
  T: null,
  dtype: "bf16",
  inferenceDtype: "bf16",
  hardware: DEFAULT_HARDWARE,
  gpus: 8,
  optimizer: "adamw",
  recompute: "selective",
  flash: true,
  zero: 1,
  tp: 1,
  pp: 1,
  ep: 1,
  sequenceParallel: false,
  concurrency: 1,
  tokens: null,
};

/**
 * Translate to what `analyze` and `validate` take. Data-parallel degree is
 * derived rather than asked for: it is whatever is left after tensor and
 * pipeline parallelism have claimed their share of the GPUs.
 */
export function toAnalysisOptions(o: OperatingPoint): AnalysisOptions {
  const dp = Math.max(1, Math.floor(o.gpus / Math.max(1, o.tp * o.pp * o.ep)));
  return {
    B: o.B,
    ...(o.T !== null ? { T: o.T } : {}),
    dtype: o.dtype,
    inferenceDtype: o.inferenceDtype,
    hardware: o.hardware,
    gpus: o.gpus,
    optimizer: o.optimizer,
    recompute: o.recompute,
    flash: o.flash,
    concurrency: o.concurrency,
    ...(o.tokens !== null ? { tokens: o.tokens } : {}),
    parallel: {
      ...DEFAULT_PARALLEL,
      dp,
      tp: o.tp,
      pp: o.pp,
      ep: o.ep,
      zero: o.zero,
      // Meaningless without a group to shard across, and the analysis would
      // otherwise take it at its word.
      sequenceParallel: o.tp > 1 && o.sequenceParallel,
    },
  };
}

const STORAGE_KEY = "tensorcad.operating";

export function loadOperating(): OperatingPoint {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_OPERATING;
    // Merge rather than replace, so a stored point from an older build gains
    // any field added since without failing to load.
    return {
      ...DEFAULT_OPERATING,
      ...(JSON.parse(raw) as Partial<OperatingPoint>),
    };
  } catch {
    return DEFAULT_OPERATING;
  }
}

export function saveOperating(o: OperatingPoint): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(o));
  } catch {
    // Not being able to remember the operating point is not a reason to stop.
  }
}
