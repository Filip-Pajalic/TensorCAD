/**
 * Block catalog types.
 *
 * Two kinds of block:
 *  - `primitive`: carries the actual formulas (params, FLOPs, activation bytes,
 *    cache state). There are about twenty of these.
 *  - `composite`: a named subgraph of primitives with its own parameters, e.g.
 *    `gqa_attention` or `gated_mlp`. Composites carry no formulas; analysis
 *    expands them. Adding a new architecture block therefore needs no new math.
 *
 * `container` is the third kind and covers `repeat` only.
 */

import type { Graph, ParamValue, Resolved } from "../ir/types.js";
import type { Sym } from "../shapes/symexpr.js";

export type ParamSpec =
  | { type: "int"; default?: number | string; min?: number; max?: number; doc?: string }
  | { type: "num"; default?: number | string; min?: number; max?: number; doc?: string }
  // `null` marks a tri-state flag: unset means "inherit from a sibling".
  | { type: "bool"; default?: boolean | null; doc?: string }
  | { type: "enum"; values: readonly string[]; default?: string | null; doc?: string }
  | { type: "str"; default?: string | null; doc?: string }
  | { type: "pattern"; default?: string | null; doc?: string }
  | { type: "obj"; default?: ParamValue; doc?: string };

/**
 * What a pin is.
 *
 * A port used to be one string — its shape pattern — with everything else about
 * it inferred somewhere downstream. Which side a wire left from was a lookup
 * table in the renderer keyed on `"type:port"`; whether a tensor carried
 * integers was decided by checking whether a dtype name started with "int".
 * Both of those are facts about the port, so they are declared on the port.
 *
 * A bare string is still a legal port and means `{ shape }`, so a block that
 * has nothing more to say does not have to say it.
 */
/**
 * Something a block says about itself.
 *
 * This used to be a bare string, which meant a finding could not be filtered by
 * severity, could not be excluded, and could not say which field caused it. All
 * three of those are things a design-rule system has to do, so a constraint now
 * returns the same shape as every other finding in the system.
 */
export interface BlockFinding {
  /** Stable id, e.g. `"ATTN-01"`. Stable so it can be excluded and documented. */
  id: string;
  severity: "error" | "warning" | "info";
  message: string;
  /** The parameter that caused it, so the inspector can highlight the field. */
  param?: string;
  /** The port that caused it, so the canvas can point at the pin. */
  port?: string;
  hint?: string;
}

export interface PortSpec {
  /** Shape pattern, as before: `"... d_model"`. */
  shape: string;
  /**
   * What the tensor carries. `inherit` (the default) takes it from the producer.
   *
   * Declaring it is what lets an integer tensor arriving at a float matmul be
   * refused: shape inference cannot catch that, because the shapes agree.
   */
  dtype?: "float" | "half" | "fp8" | "int" | "bool" | "inherit";
  /** An unconnected required port is a warning; an optional one is not. */
  optional?: boolean;
  /** What a consumer should assume when nothing is wired here. */
  whenUnconnected?: "zero" | "identity" | "causal" | { tensor: string };
  /**
   * Which side of the symbol the wire leaves by.
   *
   * `flow` follows the reading direction — down the sheet, or across if the
   * blocks sit side by side. `side` always leaves sideways, whatever the
   * geometry, because some pins mean something by it: a residual bypass
   * entering from above would read as the main path rather than the one that
   * skips it.
   */
  anchor?: "flow" | "side";
  /** Draw the port's name beside the pin. */
  showName?: boolean;
  doc?: string;
}

export interface Ports {
  in: Record<string, string | PortSpec>;
  out: Record<string, string | PortSpec>;
}

/** Ports with every default filled in, which is what consumers work with. */
export interface ResolvedPort extends PortSpec {
  shape: string;
  dtype: NonNullable<PortSpec["dtype"]>;
  optional: boolean;
  anchor: NonNullable<PortSpec["anchor"]>;
}

export interface ResolvedPorts {
  in: Record<string, ResolvedPort>;
  out: Record<string, ResolvedPort>;
}

/** A bare string is a port with nothing more to say than its shape. */
export function normalisePort(v: string | PortSpec): ResolvedPort {
  const spec: PortSpec = typeof v === "string" ? { shape: v } : v;
  return {
    ...spec,
    shape: spec.shape,
    dtype: spec.dtype ?? "inherit",
    optional: spec.optional ?? false,
    anchor: spec.anchor ?? "flow",
  };
}

export function normalisePorts(ports: Ports): ResolvedPorts {
  const map = (side: Record<string, string | PortSpec>): Record<string, ResolvedPort> => {
    const out: Record<string, ResolvedPort> = {};
    for (const [name, v] of Object.entries(side)) out[name] = normalisePort(v);
    return out;
  };
  return { in: map(ports.in), out: map(ports.out) };
}

export type PortsSpec = Ports | ((r: Resolved) => Ports);

/** Context handed to formula callbacks. */
export interface AnalysisCtx {
  /** Sequence length in tokens. */
  T: number;
  /** Batch size. */
  B: number;
  /** Bytes per element of the activation/parameter dtype. */
  bytes: number;
  /** Whether a memory-efficient attention kernel is assumed. */
  flash: boolean;
}

export interface StateBytes {
  /** Cache that grows with each generated token (KV cache). */
  perToken: number;
  /** Fixed per-sequence state (SSM recurrent state, sliding-window cache). */
  perSeq: number;
}

export interface FlopsPerToken {
  /** Forward matmul FLOPs per token that do not scale with sequence length. */
  fwd: number;
  /**
   * Forward FLOPs that scale with sequence length (attention scores and the
   * value product). Reported separately because the 6N rule excludes them.
   */
  fwdSeq?: number;
  /**
   * The sequence-dependent term counted as if nothing were masked.
   *
   * A profiler such as `torch.utils.flop_counter` reports this, because the
   * operator's shape is the same whether or not a mask is applied. A fused
   * causal kernel skips the masked blocks and does about half the work, which
   * is what `fwdSeq` reports. Both are correct answers to different questions,
   * and reporting both is what makes a cross-check against a profiler legible.
   */
  fwdSeqUnmasked?: number;
  /**
   * Elementwise work (norms, activations, RoPE, residual adds). Excluded from
   * the matmul total because the 6N convention excludes it and because these
   * ops are memory-bound rather than compute-bound.
   */
  elementwise?: number;
}

export interface BlockDocs {
  summary: string;
  formula?: string;
  refs?: string[];
}

export interface PrimitiveDef {
  kind: "primitive";
  type: string;
  category: string;
  params: Record<string, ParamSpec>;
  ports: PortsSpec;
  /** Trainable parameter count. */
  paramCount?: (r: Resolved) => number;
  /** Forward FLOPs per token. Backward is derived as 2x by the analysis. */
  flops?: (r: Resolved, ctx: AnalysisCtx) => FlopsPerToken;
  /**
   * Input ports whose incoming tensor must stay alive for the backward pass.
   *
   * Memory is attributed to the *tensor*, not the consumer, so a tensor read by
   * several blocks (the residual stream feeding q, k and v) is counted once.
   */
  retains?: (r: Resolved) => string[];
  /**
   * Bytes per token this block keeps beyond the tensors on its edges: a fused
   * attention kernel's log-sum-exp statistics, or the logits buffer.
   */
  extraActivationBytes?: (r: Resolved, ctx: AnalysisCtx) => number;
  /** Inference cache footprint. */
  stateBytes?: (r: Resolved, ctx: AnalysisCtx) => StateBytes;
  /** Extra checks beyond shape inference. Returns human-readable problems. */
  constraints?: (r: Resolved) => BlockFinding[];
  docs: BlockDocs;
}

export interface CompositeDef {
  kind: "composite";
  type: string;
  category: string;
  params: Record<string, ParamSpec>;
  ports: PortsSpec;
  /**
   * Build the inner graph. Receives the *raw* parameter values so the expansion
   * can pass expressions through (`"D"` stays `"D"` rather than becoming 4096),
   * which keeps symbol names visible on inner shapes.
   */
  expand: (raw: Record<string, ParamValue>, r: Resolved) => Graph;
  constraints?: (r: Resolved) => BlockFinding[];
  docs: BlockDocs;
}

export interface ContainerMultipliers {
  /** How many copies of the subgraph exist. Drives the parameter count. */
  total: number;
  /** How many copies a single token passes through. Drives FLOPs and memory. */
  active: number;
}

export interface ContainerDef {
  kind: "container";
  type: string;
  category: string;
  params: Record<string, ParamSpec>;
  /**
   * How the subgraph is instantiated. A stack of layers has total = active;
   * a bank of experts has total = experts but active = top_k, which is exactly
   * the distinction between a sparse model's total and active parameters.
   */
  multipliers: (r: Resolved) => ContainerMultipliers;
  docs: BlockDocs;
}

export type BlockDef = PrimitiveDef | CompositeDef | ContainerDef;

/** True when the whole string is already one parenthesised group. */
function wrapped(text: string): boolean {
  if (!text.startsWith("(") || !text.endsWith(")")) return false;
  let depth = 0;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "(") depth++;
    else if (text[i] === ")") {
      depth--;
      // Closed before the end, so the outer parens are not one group:
      // "(a)*(b)" must still be wrapped.
      if (depth === 0) return i === text.length - 1;
    }
  }
  return false;
}

const ATOM = /^[A-Za-z_][A-Za-z0-9_]*$/;
const NUMBER = /^\d+(?:\.\d+)?$/;

/**
 * A parameter as an expression, safe to interpolate into a larger one.
 *
 * Parentheses are added only when they are needed. A composite that expands
 * into another composite passes its parameters down, so re-wrapping something
 * already atomic compounds: three levels of nesting turned `H` into `(((H)))`,
 * which parses the same and reads like line noise wherever it is shown.
 */
export function ex(v: ParamValue | undefined, fallback = "0"): string {
  if (v === undefined || v === null) return fallback;
  if (typeof v === "number") return String(v);
  if (typeof v === "boolean") return v ? "1" : "0";
  if (typeof v !== "string") {
    throw new Error(`Cannot use ${JSON.stringify(v)} as an expression`);
  }
  const text = v.trim();
  if (text === "") return fallback;
  if (ATOM.test(text) || NUMBER.test(text) || wrapped(text)) return text;
  return `(${text})`;
}

/** Helper for composites: read a resolved symbolic parameter. */
export function symOf(r: Resolved, key: string): Sym | undefined {
  return r.s[key];
}
