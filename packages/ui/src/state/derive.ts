/**
 * Derived analysis.
 *
 * One call to `validate` produces every number the editor shows and every
 * design-rule finding, and the core is fast enough to redo it on each edit:
 * the heaviest preset takes ten milliseconds, the rest take one. A cache keyed
 * on the document object and the operating point means the components that ask
 * for it repeatedly during a render pass share a single computation.
 */

import type {
  AnalysisResult,
  Doc,
  Finding,
  InferResult,
  ParamsResult,
  SymbolTable,
} from "@tensorcad/core";
import { validate } from "@tensorcad/core";
import { toAnalysisOptions, type OperatingPoint } from "./operating.js";

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
  const report = validate(doc, toAnalysisOptions(operating));
  const analysis = report.analysis;

  const issues: UiIssue[] = report.findings.map((f, i) => ({
    key: `${f.rule}:${i}`,
    path: f.path ?? null,
    port: f.port,
    message: tidy(f),
    hint: f.hint,
    severity: f.severity,
    rule: f.rule,
  }));

  const severityByPath = new Map<string, Severity>();
  for (const i of issues) if (i.path) bumpSeverity(severityByPath, i.path, i.severity);

  const value: Derived = {
    analysis,
    symbols: analysis.symbols,
    // `validate` infers shapes with composites collapsed; the editor needs the
    // expanded pass so a composite's interior can be opened and inspected.
    infer: analysis.expanded,
    params: analysis.params,
    paramsByPath: rollUp(analysis.params.byPath),
    flopsByPath: rollUp(analysis.flops.byPath),
    issues,
    counts: report.counts,
    ok: report.ok,
    severityByPath,
    elapsedMs: performance.now() - started,
  };
  cache.set(doc, { key, value });
  return value;
}
