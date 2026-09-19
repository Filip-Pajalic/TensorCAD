/**
 * `tensorcad show <file|preset>` — the block tree with the shape on every edge.
 */

import { bool, type Args } from "../args.js";
import { loadDesign } from "../load.js";
import { bold, cyan, dim, magenta, pad, writeOut, yellow } from "../format.js";
import type { Doc, Graph, NodeDef } from "@tensorcad/engine";
import { formatCount, joinPath, splitEndpoint } from "@tensorcad/engine";
import { countParams, getBlock, inferShapes, resolveSymbols } from "@tensorcad/engine/node";

interface TreeNode {
  path: string;
  id: string;
  type: string;
  label?: string;
  kind: string;
  params: number;
  /** `port -> { shape, to[] }` for every output port of this node. */
  outputs: { port: string; shape: string; to: string[] }[];
  children: TreeNode[];
  /** For `repeat` containers: the instance count shown next to the type. */
  repeat?: string;
}

export function cmdShow(args: Args): number {
  const { doc } = loadDesign(args._[0]);
  const tree = buildTree(doc);

  if (bool(args, "json")) {
    writeOut(JSON.stringify({ name: doc.meta.name, tree }, null, 2));
    return 0;
  }

  const symbols = resolveSymbols(doc);
  const params = countParams(doc);
  const lines: string[] = [
    `${bold(doc.meta.name)}  ${dim(`${formatCount(params.total)} parameters`)}`,
  ];
  // The design symbols only: B and T stay indeterminate until a run, so
  // printing them beside D and L would read as though they were settings.
  const symbolLine = Object.entries(symbols.designValues)
    .map(([k, v]) => `${k}=${v}`)
    .join("  ");
  if (symbolLine) lines.push(dim(`  ${symbolLine}`));
  lines.push("");

  render(tree, lines, 1);
  writeOut(lines.join("\n"));
  return 0;
}

function buildTree(doc: Doc): TreeNode[] {
  const infer = inferShapes(doc);
  const params = countParams(doc);

  // Invert `consumer -> producer` so a producer port can list its consumers.
  const consumers = new Map<string, string[]>();
  for (const [consumer, producer] of Object.entries(infer.producerOf)) {
    const list = consumers.get(producer);
    if (list) list.push(consumer);
    else consumers.set(producer, [consumer]);
  }

  const paramsAt = (path: string): number => {
    let sum = 0;
    for (const [p, v] of Object.entries(params.byPath)) {
      if (p === path || p.startsWith(`${path}/`)) sum += v;
    }
    return sum;
  };

  const walk = (graph: Graph, prefix: string): TreeNode[] =>
    graph.nodes.map((node: NodeDef) => {
      const path = joinPath(prefix, node.id);
      const def = getBlock(node.type);
      const ports = infer.ports[path];
      const outputs = Object.keys(ports?.out ?? {}).map((port) => {
        const key = `${path}:${port}`;
        const shape = infer.outputs[key];
        return {
          port,
          shape: shape ? shape.symbolic : "?",
          to: (consumers.get(key) ?? []).map((c) => {
            const { node: n, port: p } = splitEndpoint(c);
            return `${n.slice(prefix ? prefix.length + 1 : 0)}:${p}`;
          }),
        };
      });

      const child: TreeNode = {
        path,
        id: node.id,
        type: node.type,
        kind: def?.kind ?? "unknown",
        params: paramsAt(path),
        outputs,
        children: node.graph ? walk(node.graph, path) : [],
      };
      if (node.label) child.label = node.label;
      if (node.graph) {
        const resolved = infer.resolved[path];
        const count = resolved?.p?.count;
        if (typeof count === "number") child.repeat = `x${count}`;
      }
      return child;
    });

  return walk(doc.graph, "");
}

function render(nodes: TreeNode[], lines: string[], depth: number): void {
  const indent = "  ".repeat(depth);
  const idWidth = Math.max(0, ...nodes.map((n) => n.id.length));

  for (const node of nodes) {
    const type = node.repeat ? `${node.type} ${node.repeat}` : node.type;
    const paint = node.kind === "container" ? magenta : node.kind === "composite" ? cyan : (s: string) => s;
    const size = node.params > 0 ? dim(`  ${formatCount(node.params)}`) : "";
    const label = node.label ? dim(`  "${node.label}"`) : "";
    lines.push(`${indent}${pad(node.id, idWidth)}  ${paint(type)}${size}${label}`);

    for (const out of node.outputs) {
      const targets = out.to.length > 0 ? out.to.join(", ") : dim("(unconnected)");
      lines.push(`${indent}  ${dim(`${out.port} ->`)} ${pad(targets, 24)} ${yellow(out.shape)}`);
    }

    if (node.children.length > 0) render(node.children, lines, depth + 1);
  }
}
