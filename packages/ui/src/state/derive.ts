/**
 * Derived analysis.
 *
 * One call into the engine produces every number the editor shows, every
 * design-rule finding and every shape, and it is fast enough to redo on each
 * edit: the heaviest preset takes a few milliseconds and the rest take one. A
 * cache keyed on the document object and the operating point means the
 * components that ask for it repeatedly during a render pass share a single
 * computation.
 */

import { toAnalysisOptions, type OperatingPoint } from "./operating.js";
import type {
  AnalysisResult,
  Doc,
  Finding,
  Graph,
  Inference,
  ParamsResult,
  Resolved,
  ResolvedPorts,
  Shape,
  SymbolTable,
} from "@tensorcad/engine";
import { engine } from "../engine.js";

/**
 * Shape inference as the editor reads it.
 *
 * The engine answers in plain objects, because that is what JSON is. The panels
 * ask by key many times per repaint, so the keyed parts become Maps once here
 * rather than being indexed as objects everywhere.
 */
export interface InferResult {
  /** `"path:port"` to the shape an output port produces. */
  outputs: Map<string, Shape>;
  /** `"path:port"` to the shape an input port actually received. */
  inputs: Map<string, Shape>;
  /** Consumer `"path:port"` to producer `"path:port"`. */
  producerOf: Map<string, string>;
  ports: Map<string, ResolvedPorts>;
  resolved: Map<string, Resolved>;
  /** The subgraph each composite stood for, so interiors can be drawn. */
  expansions: Map<string, Graph>;
}

function asMaps(infer: Inference): InferResult {
  return {
    outputs: new Map(Object.entries(infer.outputs)),
    inputs: new Map(Object.entries(infer.inputs)),
    producerOf: new Map(Object.entries(infer.producerOf)),
    ports: new Map(Object.entries(infer.ports)),
    resolved: new Map(Object.entries(infer.resolved)),
    expansions: new Map(Object.entries(infer.expansions)),
  };
}

export type Severity = "error" | "warning" | "info";

export interface UiIssue {
  key: string;
  /** Node path the issue belongs to, when it could be attributed to one. */
  path: string | null;
  port?: string;
  message: string;
  /** What to do about it, when the rule knows. */
  hint?: string;
  severity: Severity;
  /** The rule that raised it, e.g. `"flash-head-dim"`. */
  rule: string;
  /** The parameter that caused it, so the inspector can highlight the field. */
  param?: string;
}

export interface Derived {
  analysis: AnalysisResult;
  symbols: SymbolTable;
  /** Shape inference with composites expanded, so interiors can be browsed. */
  infer: InferResult;
  params: ParamsResult;
  /** Parameter count per node path, summed over everything beneath it. */
  paramsByPath: Map<string, number>;
  /** FLOPs per token per node path, summed over everything beneath it. */
  flopsByPath: Map<string, number>;
  issues: UiIssue[];
  counts: Record<Severity, number>;
  /** True when nothing blocks building this design. */
  ok: boolean;
  /** Worst severity at a path or anywhere beneath it. */
  severityByPath: Map<string, Severity>;
  /**
   * Findings attributed to exactly this path, not rolled up.
   *
   * `severityByPath` answers "is anything wrong in here", which is what a tree
   * needs. This answers "what is wrong with *this*", which is what a marker on
   * the drawing needs — a marker that fired because of something three levels
   * down would be pointing at the wrong thing.
   */
  findingsByPath: Map<string, UiIssue[]>;
  /** Milliseconds the whole analysis took, shown in the status bar. */
  elapsedMs: number;
}

interface CacheEntry {
  key: string;
  value: Derived;
}

const cache = new WeakMap<Doc, CacheEntry>();

/** Sum every primitive's contribution into each of its ancestor paths. */
function rollUp(byPath: Record<string, number>): Map<string, number> {
  const out = new Map<string, number>();
  for (const [path, n] of Object.entries(byPath)) {
    const segs = path.split("/");
    for (let i = segs.length; i > 0; i--) {
      const key = segs.slice(0, i).join("/");
      out.set(key, (out.get(key) ?? 0) + n);
    }
  }
  return out;
}

const RANK: Record<Severity, number> = { error: 0, warning: 1, info: 2 };

function bumpSeverity(map: Map<string, Severity>, path: string, severity: Severity): void {
  const segs = path.split("/");
  for (let i = segs.length; i > 0; i--) {
    const key = segs.slice(0, i).join("/");
    const held = map.get(key);
    if (held === undefined || RANK[severity] < RANK[held]) map.set(key, severity);
  }
}

/**
 * Some findings name their node inside the message because the rule builds the
 * sentence around it. Strip the prefix when it duplicates the path we already
 * show beside it.
 */
function tidy(f: Finding): string {
  if (f.path && f.message.startsWith(`${f.path}: `)) return f.message.slice(f.path.length + 2);
  return f.message;
}

export function derive(doc: Doc, operating: OperatingPoint): Derived {
  const key = JSON.stringify(operating);
  const hit = cache.get(doc);
  if (hit && hit.key === key) return hit.value;

  const started = performance.now();
  const derived = engine().derive(doc, toAnalysisOptions(operating));
  const report = derived.report;
  const analysis = report.analysis;

  const issues: UiIssue[] = report.findings.map((f: Finding, i: number) => ({
    key: `${f.rule}:${i}`,
    path: f.path ?? null,
    port: f.port,
    param: f.param,
    message: tidy(f),
    hint: f.hint,
    severity: f.severity,
    rule: f.rule,
  }));

  const severityByPath = new Map<string, Severity>();
  const findingsByPath = new Map<string, UiIssue[]>();
  for (const i of issues) {
    if (!i.path) continue;
    bumpSeverity(severityByPath, i.path, i.severity);
    const held = findingsByPath.get(i.path);
    if (held) held.push(i);
    else findingsByPath.set(i.path, [i]);
  }

  const value: Derived = {
    analysis,
    symbols: analysis.symbols,
    // Composites expanded, so a block's interior can be opened and inspected.
    infer: asMaps(derived.infer),
    params: analysis.params,
    paramsByPath: rollUp(analysis.params.byPath),
    flopsByPath: rollUp(analysis.flops.byPath),
    issues,
    counts: report.counts,
    ok: report.ok,
    severityByPath,
    findingsByPath,
    elapsedMs: performance.now() - started,
  };
  cache.set(doc, { key, value });
  return value;
}
