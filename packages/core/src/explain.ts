/**
 * Explain one block: what it is, what it was given, and what it costs.
 *
 * This is the learning surface. Every number the tool reports should be
 * traceable to a formula and a source, and every parameter should show both the
 * expression somebody wrote and the value it evaluated to.
 */

import type { Doc, ParamValue, SymbolTable } from "./ir/types.js";
import { joinPath } from "./ir/types.js";
import { resolveSymbols } from "./ir/symbols.js";
import { inferShapes, type InferResult } from "./shapes/infer.js";
import { CATALOG, catalogOf } from "./catalog/index.js";
import type { BlockDocs } from "./catalog/types.js";
import { flatten, type FlatResult } from "./analysis/flatten.js";
import { analyze, type AnalysisOptions, type AnalysisResult } from "./analysis/index.js";

export interface ExplainedParam {
  /** The expression as written in the document, when it was an expression. */
  expression: string | null;
  /** The value it evaluated to. */
  value: unknown;
  doc?: string;
}

export interface Explanation {
  path: string;
  type: string;
  kind: "primitive" | "composite" | "container";
  label?: string;
  docs: BlockDocs;
  /** How many copies exist, and how many a single token passes through. */
  copies: { total: number; active: number };
  params: Record<string, ExplainedParam>;
  shapes: { in: Record<string, string>; out: Record<string, string> };
  /** What this block and everything inside it contribute to the whole model. */
  contributes: {
    params: number;
    activeParams: number;
    shareOfParams: number;
    flopsPerToken: number;
    shareOfFlops: number;
    activationBytes: number;
    cacheBytesPerToken: number;
    cacheBytesPerSequence: number;
  };
  /** The primitives this block expands into, largest first. */
  breakdown: { path: string; type: string; params: number }[];
  notFound?: boolean;
}

function findNode(doc: Doc, path: string): { type: string; label?: string; params?: Record<string, ParamValue> } | null {
  const parts = path.split("/").filter(Boolean);
  let graph = doc.graph;
  let node: { type: string; label?: string; params?: Record<string, ParamValue>; graph?: typeof graph } | null = null;
  for (const part of parts) {
    const hit = graph?.nodes.find((n) => n.id === part);
    if (!hit) return null;
    node = hit;
    graph = hit.graph ?? { nodes: [], edges: [] };
  }
  return node;
}

/** Subtree of the flattened graph rooted at `path`. */
function subtree(flat: FlatResult, path: string) {
  const prefix = `${path}/`;
  return flat.nodes.filter((n) => n.path === path || n.path.startsWith(prefix));
}

export function explain(
  doc: Doc,
  path: string,
  options: AnalysisOptions = {},
  pre?: { symbols?: SymbolTable; infer?: InferResult; flat?: FlatResult; analysis?: AnalysisResult },
): Explanation {
  const symbols = pre?.symbols ?? resolveSymbols(doc);
  const infer = pre?.infer ?? inferShapes(doc, symbols, { expandComposites: true });
  const flat = pre?.flat ?? flatten(doc, symbols);
  const analysis = pre?.analysis ?? analyze(doc, options, { symbols, flat, expanded: infer });

  const empty: Explanation = {
    path,
    type: "unknown",
    kind: "primitive",
    docs: { summary: `No block at "${path}".` },
    copies: { total: 0, active: 0 },
    params: {},
    shapes: { in: {}, out: {} },
    contributes: {
      params: 0,
      activeParams: 0,
      shareOfParams: 0,
      flopsPerToken: 0,
      shareOfFlops: 0,
      activationBytes: 0,
      cacheBytesPerToken: 0,
      cacheBytesPerSequence: 0,
    },
    breakdown: [],
    notFound: true,
  };

  const block = flat.blocks.find((b) => b.path === path);
  const def = block ? block.def : null;
  if (!block || !def) {
    // The path may name a node inside a composite expansion, which `blocks`
    // records only for document nodes. Fall back to the inference result.
    const resolved = infer.resolved.get(path);
    const ports = infer.ports.get(path);
    if (!resolved || !ports) return empty;
    const fallbackDef = catalogOf(doc)[resolved.type];
    if (!fallbackDef) return empty;
    return buildExplanation({
      path,
      def: fallbackDef,
      resolvedParams: resolved,
      ports,
      copies: { total: 1, active: 1 },
      nodes: subtree(flat, path),
      analysis,
      docNode: null,
    });
  }

  const ports = infer.ports.get(path) ?? { in: {}, out: {} };
  const docNode = findNode(doc, path);
  return buildExplanation({
    path,
    def,
    resolvedParams: block.resolved,
    ports,
    copies: { total: block.multiplier, active: block.activeMultiplier },
    nodes: subtree(flat, path),
    analysis,
    docNode,
  });
}

function buildExplanation(input: {
  path: string;
  def: (typeof CATALOG)[string];
  resolvedParams: { p: Record<string, unknown>; raw: Record<string, ParamValue>; rawFull: Record<string, ParamValue> };
  ports: { in: Record<string, string>; out: Record<string, string> };
  copies: { total: number; active: number };
  nodes: FlatResult["nodes"];
  analysis: AnalysisResult;
  docNode: { label?: string } | null;
}): Explanation {
  const { path, def, resolvedParams, ports, copies, nodes, analysis } = input;

  const params: Record<string, ExplainedParam> = {};
  const specs = (def.params ?? {}) as Record<string, { doc?: string }>;
  for (const [key, value] of Object.entries(resolvedParams.p)) {
    const raw = resolvedParams.rawFull[key];
    params[key] = {
      expression: typeof raw === "string" && raw !== value ? raw : null,
      value,
      doc: specs[key]?.doc,
    };
  }

  let paramSum = 0;
  let activeSum = 0;
  let flopSum = 0;
  let activationSum = 0;
  let cacheToken = 0;
  let cacheSeq = 0;
  const breakdown: { path: string; type: string; params: number }[] = [];

  for (const node of nodes) {
    const p = analysis.params.byPath[node.path] ?? 0;
    paramSum += p;
    activeSum += node.multiplier > 0 ? (p / node.multiplier) * node.activeMultiplier : 0;
    flopSum += analysis.flops.byPath[node.path] ?? 0;
    activationSum += analysis.memory.train.activationsByPath[node.path] ?? 0;
    if (node.def.stateBytes) {
      try {
        const s = node.def.stateBytes(node.resolved, {
          T: analysis.options.T,
          B: analysis.options.B,
          bytes: 2,
          flash: analysis.options.flash,
        });
        cacheToken += s.perToken * node.multiplier;
        cacheSeq += s.perSeq * node.multiplier;
      } catch {
        // A block whose parameters failed to resolve is reported elsewhere.
      }
    }
    if (p > 0) breakdown.push({ path: node.path, type: node.type, params: p });
  }

  breakdown.sort((a, b) => b.params - a.params);

  return {
    path,
    type: def.type,
    kind: def.kind,
    label: input.docNode?.label,
    docs: def.docs,
    copies,
    params,
    shapes: { in: { ...ports.in }, out: { ...ports.out } },
    contributes: {
      params: paramSum,
      activeParams: Math.round(activeSum),
      shareOfParams: analysis.params.total > 0 ? paramSum / analysis.params.total : 0,
      flopsPerToken: flopSum,
      shareOfFlops: analysis.flops.fwdTotal > 0 ? flopSum / analysis.flops.fwdTotal : 0,
      activationBytes: activationSum,
      cacheBytesPerToken: cacheToken,
      cacheBytesPerSequence: cacheSeq,
    },
    breakdown: breakdown.slice(0, 12),
  };
}

/** Every block in the document, largest contribution first. */
export function explainAll(doc: Doc, options: AnalysisOptions = {}): Explanation[] {
  const symbols = resolveSymbols(doc);
  const infer = inferShapes(doc, symbols, { expandComposites: true });
  const flat = flatten(doc, symbols);
  const analysis = analyze(doc, options, { symbols, flat, expanded: infer });
  return flat.blocks
    .map((b) => explain(doc, b.path, options, { symbols, infer, flat, analysis }))
    .sort((a, b) => b.contributes.params - a.contributes.params);
}
