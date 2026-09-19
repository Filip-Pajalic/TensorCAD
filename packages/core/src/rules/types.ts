import type { Doc, SymbolTable } from "../ir/types.js";
import type { InferResult } from "../shapes/infer.js";
import type { FlatResult } from "../analysis/flatten.js";
import type { AnalysisResult } from "../analysis/index.js";

export type Severity = "error" | "warning" | "info";

export interface Finding {
  /** Stable rule identifier, e.g. `"flash-head-dim"`. */
  rule: string;
  severity: Severity;
  /** Node path the finding is about, when it is about one node. */
  path?: string;
  port?: string;
  /** The parameter that caused it, so the inspector can highlight the field. */
  param?: string;
  message: string;
  /** What to do about it. */
  hint?: string;
}

export interface RuleCtx {
  doc: Doc;
  symbols: SymbolTable;
  infer: InferResult;
  flat: FlatResult;
  analysis: AnalysisResult;
}

export interface Rule {
  id: string;
  title: string;
  /** One line describing what the rule protects against, for the docs panel. */
  description: string;
  run(ctx: RuleCtx): Finding[];
}
