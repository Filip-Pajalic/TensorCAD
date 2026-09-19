/**
 * The design-rule check.
 *
 * `validate` runs every rule over a design and returns findings sorted by
 * severity. It also returns the analysis it computed, so a caller that wants
 * both numbers and findings pays for the work once.
 */

import type { Doc } from "../ir/types.js";
import { resolveSymbols } from "../ir/symbols.js";
import { inferShapes } from "../shapes/infer.js";
import { flatten } from "../analysis/flatten.js";
import { analyze, type AnalysisOptions, type AnalysisResult } from "../analysis/index.js";
import { RULES } from "./rules.js";
import type { Finding, RuleCtx, Severity } from "./types.js";

export interface ValidationReport {
  name: string;
  findings: Finding[];
  counts: Record<Severity, number>;
  /** True when nothing blocks building this design. */
  ok: boolean;
  analysis: AnalysisResult;
}

const SEVERITY_ORDER: Record<Severity, number> = { error: 0, warning: 1, info: 2 };

export function validate(doc: Doc, options: AnalysisOptions = {}): ValidationReport {
  const symbols = resolveSymbols(doc);
  const infer = inferShapes(doc, symbols);
  const flat = flatten(doc, symbols);
  const analysis = analyze(doc, options, { symbols, infer, flat });

  const ctx: RuleCtx = { doc, symbols, infer, flat, analysis };
  const findings: Finding[] = [];

  for (const rule of RULES) {
    try {
      findings.push(...rule.run(ctx));
    } catch (e) {
      findings.push({
        rule: rule.id,
        severity: "error",
        message: `Rule "${rule.id}" failed: ${(e as Error).message}`,
      });
    }
  }

  for (const message of flat.errors) {
    findings.push({ rule: "graph", severity: "error", message });
  }

  findings.sort((a, b) => {
    const s = SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity];
    if (s !== 0) return s;
    return (a.path ?? "").localeCompare(b.path ?? "");
  });

  const counts: Record<Severity, number> = { error: 0, warning: 0, info: 0 };
  for (const f of findings) counts[f.severity]++;

  return { name: doc.meta.name, findings, counts, ok: counts.error === 0, analysis };
}

export { RULES } from "./rules.js";
export type { Finding, Rule, RuleCtx, Severity } from "./types.js";
