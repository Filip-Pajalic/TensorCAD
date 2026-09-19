/**
 * Hardware profiles.
 *
 * Peak numbers are dense (no structured sparsity) tensor-core throughput from
 * vendor spec sheets, and prices are rough market rates. Both are approximate
 * and meant to be overridden: treat them as a starting point, not a
 * measurement. `mfuHint` is a realistic model-FLOPs-utilization band for
 * training on that part, drawn from published runs where one exists.
 */

export interface HardwareProfile {
  id: string;
  name: string;
  /** Dense BF16/FP16 tensor throughput, FLOP/s. */
  peakBf16: number;
  /** Dense FP8 tensor throughput, FLOP/s. Zero when unsupported. */
  peakFp8: number;
  /** Device memory, bytes. */
  memory: number;
  /** Device memory bandwidth, bytes/s. */
  bandwidth: number;
  /** Indicative rental price, US dollars per GPU-hour. */
  pricePerHour: number;
  mfuHint: [low: number, high: number];
  notes?: string;
}

const T = 1e12;
const GB = 1024 ** 3;
const GBs = 1e9;

export const HARDWARE: HardwareProfile[] = [
  {
    id: "rtx5080",
    name: "GeForce RTX 5080 (16 GB)",
    peakBf16: 225 * T,
    peakFp8: 450 * T,
    memory: 16 * GB,
    bandwidth: 960 * GBs,
    pricePerHour: 0.2,
    mfuHint: [0.2, 0.35],
    notes: "Blackwell consumer part. Approximate spec-sheet values; no NVLink, so multi-GPU scaling is poor.",
  },
  {
    id: "rtx4090",
    name: "GeForce RTX 4090 (24 GB)",
    peakBf16: 165 * T,
    peakFp8: 330 * T,
    memory: 24 * GB,
    bandwidth: 1008 * GBs,
    pricePerHour: 0.35,
    mfuHint: [0.2, 0.35],
  },
  {
    id: "a100-80",
    name: "A100 SXM (80 GB)",
    peakBf16: 312 * T,
    peakFp8: 0,
    memory: 80 * GB,
    bandwidth: 2039 * GBs,
    pricePerHour: 1.6,
    mfuHint: [0.3, 0.45],
  },
  {
    id: "h100-sxm",
    name: "H100 SXM (80 GB)",
    peakBf16: 989 * T,
    peakFp8: 1979 * T,
    memory: 80 * GB,
    bandwidth: 3350 * GBs,
    pricePerHour: 2.5,
    mfuHint: [0.35, 0.45],
    notes: "Llama 3 405B reported 38-43% BF16 MFU on H100 clusters.",
  },
  {
    id: "h200-sxm",
    name: "H200 SXM (141 GB)",
    peakBf16: 989 * T,
    peakFp8: 1979 * T,
    memory: 141 * GB,
    bandwidth: 4800 * GBs,
    pricePerHour: 3.2,
    mfuHint: [0.35, 0.45],
  },
  {
    id: "b200",
    name: "B200 SXM (192 GB)",
    peakBf16: 2250 * T,
    peakFp8: 4500 * T,
    memory: 192 * GB,
    bandwidth: 8000 * GBs,
    pricePerHour: 5.5,
    mfuHint: [0.3, 0.45],
    notes: "Approximate; dense Blackwell datacenter throughput.",
  },
];

export const HARDWARE_BY_ID: Record<string, HardwareProfile> = Object.fromEntries(
  HARDWARE.map((h) => [h.id, h]),
);

export const DEFAULT_HARDWARE = "h100-sxm";

export function resolveHardware(h: string | HardwareProfile | undefined): HardwareProfile {
  if (!h) return HARDWARE_BY_ID[DEFAULT_HARDWARE];
  if (typeof h === "string") {
    const hit = HARDWARE_BY_ID[h];
    if (!hit) throw new Error(`Unknown hardware profile "${h}". Known: ${HARDWARE.map((x) => x.id).join(", ")}`);
    return hit;
  }
  return h;
}

export type Dtype = "fp32" | "bf16" | "fp16" | "fp8";

export const DTYPE_BYTES: Record<Dtype, number> = {
  fp32: 4,
  bf16: 2,
  fp16: 2,
  fp8: 1,
};

/** Peak throughput for a dtype, falling back to BF16 when FP8 is unsupported. */
export function peakFlops(hw: HardwareProfile, dtype: Dtype): number {
  if (dtype === "fp8" && hw.peakFp8 > 0) return hw.peakFp8;
  return hw.peakBf16;
}

export type OptimizerKind = "adamw" | "adamw8bit" | "muon" | "sgd_momentum" | "sgd" | "bf16_adam";

/**
 * Bytes per parameter held by the training state, split so the analysis can
 * shard each part independently. Mixed-precision AdamW is the classic
 * 2 + 2 + 4 + 4 + 4 = 16 bytes per parameter.
 */
export const OPTIMIZER_BYTES: Record<OptimizerKind, { weights: number; grads: number; optimizer: number; label: string }> = {
  adamw: { weights: 2, grads: 2, optimizer: 12, label: "AdamW, mixed precision (16 B/param)" },
  adamw8bit: { weights: 2, grads: 2, optimizer: 6, label: "8-bit AdamW (10 B/param)" },
  muon: { weights: 2, grads: 2, optimizer: 8, label: "Muon with fp32 master weights (12 B/param)" },
  sgd_momentum: { weights: 2, grads: 2, optimizer: 8, label: "SGD with momentum (12 B/param)" },
  sgd: { weights: 2, grads: 2, optimizer: 4, label: "SGD (8 B/param)" },
  bf16_adam: { weights: 2, grads: 2, optimizer: 4, label: "Pure bf16 Adam, no master weights (8 B/param)" },
};
