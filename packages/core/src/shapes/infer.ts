/**
 * Shape inference over a design document.
 *
 * Walks the graph topologically, checks every incoming edge against the
 * consumer's declared port pattern, and instantiates the producer's output
 * patterns. Design symbols are substituted before comparison; runtime symbols
 * (`B`, `T`) stay indeterminate, so a mismatch is a genuine polynomial
 * difference rather than a coincidence of numbers.
 */

import type { Doc, Graph, NodeDef, Resolved, SymbolTable } from "../ir/types.js";
import { joinPath, splitEndpoint } from "../ir/types.js";
import { catalogOf, isContainer, isComposite, isPrimitive } from "../catalog/index.js";
import { portsOf, resolveNodeParams } from "../catalog/resolve.js";
import type { Ports } from "../catalog/types.js";
import type { EvalCtx } from "./expr.js";
import { instantiate, matchPattern, parsePattern, shapeToString, type Shape } from "./pattern.js";

export interface InferIssue {
  path: string;
  port?: string;
  message: string;
  severity: "error" | "warning";
}

export interface InferResult {
  /** `"path:port"` -> shape, for output ports. */
  outputs: Map<string, Shape>;
  /** `"path:port"` -> shape, for input ports as actually received. */
  inputs: Map<string, Shape>;
  /**
   * Consumer `"path:port"` -> producer `"path:port"`. Lets the memory model
   * count a tensor once even when several blocks read it.
   */
  producerOf: Map<string, string>;
  /** Resolved parameters per node path, reused by the rules engine. */
  resolved: Map<string, Resolved>;
  /** Ports per node path, for the editor. */
  ports: Map<string, Ports>;
  issues: InferIssue[];
}

export interface InferOptions {
  /** Recurse into composite expansions instead of using their declared ports. */
  expandComposites?: boolean;
}

export function evalCtxFor(resolved: Resolved, symbols: SymbolTable): EvalCtx {
  const values: Record<string, number> = { ...symbols.values };
  for (const [k, v] of Object.entries(resolved.p)) {
    if (typeof v === "number" && Number.isFinite(v)) values[k] = v;
  }
  return { values, known: new Set(Object.keys(values)), substitutions: resolved.s };
}

/** Ports of a `repeat` container, derived from its boundary nodes. */
function containerPorts(node: NodeDef): Ports {
  const g = node.graph ?? { nodes: [], edges: [] };
  const bIn = g.nodes.find((n) => n.type === "boundary_in");
  const bOut = g.nodes.find((n) => n.type === "boundary_out");
  return {
    in: { ...((bIn?.params?.ports as Record<string, string>) ?? {}) },
    out: { ...((bOut?.params?.ports as Record<string, string>) ?? {}) },
  };
}

function topoOrder(graph: Graph, issues: InferIssue[], prefix: string): NodeDef[] {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const deps = new Map<string, Set<string>>();
  for (const n of graph.nodes) deps.set(n.id, new Set());

  for (const [from, to] of graph.edges) {
    const f = splitEndpoint(from);
    const t = splitEndpoint(to);
    if (!byId.has(f.node)) {
      issues.push({ path: joinPath(prefix, f.node), message: `Edge source "${from}" refers to a missing node`, severity: "error" });
      continue;
    }
    if (!byId.has(t.node)) {
      issues.push({ path: joinPath(prefix, t.node), message: `Edge target "${to}" refers to a missing node`, severity: "error" });
      continue;
    }
    deps.get(t.node)!.add(f.node);
  }

  const order: NodeDef[] = [];
  const state = new Map<string, "pending" | "done">();
  const visit = (id: string, stack: string[]): void => {
    const st = state.get(id);
    if (st === "done") return;
    if (st === "pending") {
      issues.push({
        path: joinPath(prefix, id),
        message: `Cycle detected: ${[...stack, id].join(" -> ")}. Use a repeat container for recurrence.`,
        severity: "error",
      });
      return;
    }
    state.set(id, "pending");
    for (const d of deps.get(id) ?? []) visit(d, [...stack, id]);
    state.set(id, "done");
    const n = byId.get(id);
    if (n) order.push(n);
  };
  for (const n of graph.nodes) visit(n.id, []);
  return order;
}

function producerMap(graph: Graph, issues: InferIssue[], prefix: string): Map<string, string> {
  const map = new Map<string, string>();
  for (const [from, to] of graph.edges) {
    if (map.has(to)) {
      const t = splitEndpoint(to);
      issues.push({
        path: joinPath(prefix, t.node),
        port: t.port,
        message: `Input port "${t.port}" has more than one incoming edge`,
        severity: "error",
      });
      continue;
    }
    map.set(to, from);
  }
  return map;
}

interface Walker {
  doc: Doc;
  symbols: SymbolTable;
  opts: InferOptions;
  result: InferResult;
}

function inferGraph(w: Walker, graph: Graph, prefix: string, seeds: Map<string, Shape>): void {
  const { issues, outputs, inputs, resolved: resolvedMap, ports: portsMap } = w.result;
  const order = topoOrder(graph, issues, prefix);
  const producers = producerMap(graph, issues, prefix);

  for (const node of order) {
    const path = joinPath(prefix, node.id);
    const def = catalogOf(w.doc)[node.type];
    if (!def) {
      issues.push({ path, message: `Unknown block type "${node.type}"`, severity: "error" });
      continue;
    }

    const resolved = resolveNodeParams(def, node.params, w.symbols);
    resolvedMap.set(path, resolved);
    for (const e of resolved.errors) issues.push({ path, message: e, severity: "error" });

    let nodePorts: Ports;
    try {
      nodePorts = isContainer(def) ? containerPorts(node) : portsOf(def.ports, resolved);
    } catch (e) {
      issues.push({ path, message: `Could not determine ports: ${(e as Error).message}`, severity: "error" });
      continue;
    }
    portsMap.set(path, nodePorts);

    if (isPrimitive(def) && def.constraints) {
      for (const c of def.constraints(resolved)) issues.push({ path, message: c, severity: "error" });
    }
    if (isComposite(def) && def.constraints) {
      for (const c of def.constraints(resolved)) issues.push({ path, message: c, severity: "error" });
    }

    const ctx = evalCtxFor(resolved, w.symbols);

    // --- inputs ----------------------------------------------------------
    let batch: Shape | null = null;

    for (const [portName, patternSrc] of Object.entries(nodePorts.in)) {
      const key = `${node.id}:${portName}`;
      const from = producers.get(key);
      if (!from) {
        issues.push({
          path,
          port: portName,
          message: `Input port "${portName}" is not connected`,
          severity: "error",
        });
        continue;
      }
      const fromParsed = splitEndpoint(from);
      const producerEndpoint = `${joinPath(prefix, fromParsed.node)}:${fromParsed.port}`;
      w.result.producerOf.set(`${path}:${portName}`, producerEndpoint);
      const actual = outputs.get(producerEndpoint);
      if (!actual) {
        issues.push({
          path,
          port: portName,
          message: `Upstream shape for "${from}" is unavailable`,
          severity: "warning",
        });
        continue;
      }
      inputs.set(`${path}:${portName}`, actual);

      let pattern;
      try {
        pattern = parsePattern(patternSrc);
      } catch (e) {
        issues.push({ path, port: portName, message: (e as Error).message, severity: "error" });
        continue;
      }
      const m = matchPattern(actual, pattern, ctx, w.symbols.designValues);
      for (const err of m.errors) {
        issues.push({
          path,
          port: portName,
          message: `Port "${portName}" received ${shapeToString(actual)}: ${err}`,
          severity: "error",
        });
      }
      if (m.ok) {
        if (batch === null) {
          batch = m.batch;
        } else if (
          batch.length !== m.batch.length ||
          !batch.every((d, i) => d.equalsUnder(m.batch[i], w.symbols.designValues))
        ) {
          issues.push({
            path,
            port: portName,
            message: `Batch dimensions disagree between ports: ${shapeToString(batch)} vs ${shapeToString(m.batch)}`,
            severity: "error",
          });
        }
      }
    }

    // --- recurse ---------------------------------------------------------
    if (isContainer(def) && node.graph) {
      const innerSeeds = new Map<string, Shape>();
      const bIn = node.graph.nodes.find((n) => n.type === "boundary_in");
      if (bIn) {
        for (const portName of Object.keys(nodePorts.in)) {
          const s = inputs.get(`${path}:${portName}`);
          if (s) innerSeeds.set(`${bIn.id}:${portName}`, s);
        }
      }
      inferGraph(w, node.graph, path, innerSeeds);

      // The stack applies its subgraph repeatedly, so the residual stream must
      // come out the same shape it went in.
      const bOut = node.graph.nodes.find((n) => n.type === "boundary_out");
      if (bIn && bOut) {
        for (const portName of Object.keys(nodePorts.out)) {
          const outShape = inputs.get(`${joinPath(path, bOut.id)}:${portName}`);
          const inShape = inputs.get(`${path}:${portName}`);
          if (outShape) outputs.set(`${path}:${portName}`, outShape);
          if (outShape && inShape) {
            const same =
              inShape.length === outShape.length &&
              inShape.every((d, i) => d.equalsUnder(outShape[i], w.symbols.designValues));
            if (!same) {
              issues.push({
                path,
                port: portName,
                message:
                  `A repeat container must preserve its shape, but port "${portName}" goes in as ` +
                  `${shapeToString(inShape)} and comes out as ${shapeToString(outShape)}`,
                severity: "error",
              });
            }
          }
        }
      }
      continue;
    }

    if (isComposite(def) && w.opts.expandComposites) {
      let inner: Graph;
      try {
        inner = def.expand(resolved.rawFull, resolved);
      } catch (e) {
        issues.push({ path, message: `Expansion failed: ${(e as Error).message}`, severity: "error" });
        continue;
      }
      const innerSeeds = new Map<string, Shape>();
      const bIn = inner.nodes.find((n) => n.type === "boundary_in");
      if (bIn) {
        for (const portName of Object.keys(nodePorts.in)) {
          const s = inputs.get(`${path}:${portName}`);
          if (s) innerSeeds.set(`${bIn.id}:${portName}`, s);
        }
      }
      inferGraph(w, inner, path, innerSeeds);
      const bOut = inner.nodes.find((n) => n.type === "boundary_out");
      if (bOut) {
        for (const portName of Object.keys(nodePorts.out)) {
          const s = inputs.get(`${joinPath(path, bOut.id)}:${portName}`);
          if (s) outputs.set(`${path}:${portName}`, s);
        }
      }
      continue;
    }

    // --- outputs ---------------------------------------------------------
    for (const [portName, patternSrc] of Object.entries(nodePorts.out)) {
      let pattern;
      try {
        pattern = parsePattern(patternSrc);
      } catch (e) {
        issues.push({ path, port: portName, message: (e as Error).message, severity: "error" });
        continue;
      }
      const seed = seeds.get(`${node.id}:${portName}`);
      if (seed) {
        outputs.set(`${path}:${portName}`, seed);
        continue;
      }
      const inst = instantiate(pattern, ctx, batch ?? []);
      for (const err of inst.errors) issues.push({ path, port: portName, message: err, severity: "error" });
      if (inst.shape) outputs.set(`${path}:${portName}`, inst.shape);
    }
  }
}

export function inferShapes(doc: Doc, symbols: SymbolTable, opts: InferOptions = {}): InferResult {
  const result: InferResult = {
    outputs: new Map(),
    inputs: new Map(),
    producerOf: new Map(),
    resolved: new Map(),
    ports: new Map(),
    issues: [],
  };
  inferGraph({ doc, symbols, opts, result }, doc.graph, "", new Map());
  return result;
}
