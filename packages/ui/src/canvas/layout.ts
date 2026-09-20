/**
 * Auto-layout with elkjs.
 *
 * The layered algorithm running downwards matches how these designs read:
 * tokens at the top, logits at the bottom. ELK runs in a Web Worker when the
 * bundler can produce one, and falls back to the bundled main-thread build.
 */

import { createLayoutWorker } from "./elk-worker.js";

export interface LayoutNode {
  id: string;
  width: number;
  height: number;
}

export interface LayoutEdge {
  id: string;
  source: string;
  target: string;
}

interface ElkNode {
  id: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  children?: ElkNode[];
}

type ElkLike = {
  layout: (graph: unknown) => Promise<ElkNode>;
};

let elkPromise: Promise<ElkLike> | null = null;

/**
 * How long the probe waits before deciding the worker is not coming.
 *
 * It answers an empty graph in single-digit milliseconds when it is alive, so
 * this is not a performance budget — it is the longest anybody waits for a
 * worker that is never going to reply. The `error` event usually gets there
 * first; this is for the cases where nothing is raised at all.
 */
const PROBE_TIMEOUT_MS = 2000;

async function getElk(): Promise<ElkLike> {
  elkPromise ??= (async (): Promise<ElkLike> => {
    try {
      // Null in the packaged build, which ships no worker file. Thrown rather
      // than branched, so there is one path out of here and it is the fallback.
      //
      // Held in a local first because the narrowing has to survive into the
      // factory closure below, and TypeScript will not carry it there through
      // an imported binding.
      const makeWorker = createLayoutWorker;
      if (!makeWorker) throw new Error("this build ships no layout worker");

      const api = await import("elkjs/lib/elk-api.js");
      const ELK = api.default;

      // Set by the probe below, so a worker that dies on load rejects it
      // rather than leaving it outstanding.
      let abandon: ((reason: Error) => void) | null = null;

      const elk: ElkLike = new ELK({
        workerFactory: () => {
          const worker = makeWorker();
          worker.addEventListener("error", (event) => {
            abandon?.(new Error(`the layout worker did not start: ${event.message}`));
          });
          return worker;
        },
      });

      // Smoke test, because a worker that failed to start only shows up on
      // first use — and, worse, shows up as *nothing*. When the worker script
      // is not where the bundler said it would be, a dev server answers the
      // request with its single-page fallback: the worker is handed `index.html`,
      // throws "Unexpected token '<'" inside itself, and never posts a message
      // back. A bare `await elk.layout(...)` then waits forever, this promise
      // never settles, no layout ever runs, and every node stays at the origin
      // stacked on top of the next. The fallback below was already right; it
      // simply never got the chance to run.
      await new Promise<void>((resolve, reject) => {
        abandon = reject;
        const timer = setTimeout(
          () => reject(new Error("the layout worker did not answer")),
          PROBE_TIMEOUT_MS,
        );
        const done = (): void => clearTimeout(timer);
        elk.layout({ id: "probe", layoutOptions: {}, children: [], edges: [] }).then(
          () => {
            done();
            resolve();
          },
          (error: unknown) => {
            done();
            reject(error instanceof Error ? error : new Error(String(error)));
          },
        );
      });

      abandon = null;
      return elk;
    } catch {
      // The same algorithm on this thread. Slower on a large graph, and a
      // drawing that arrives late beats one that never arrives.
      const bundled = await import("elkjs/lib/elk.bundled.js");
      const ELK = bundled.default;
      return new ELK() as ElkLike;
    }
  })();
  return elkPromise;
}

/**
 * Signal runs downwards, which is how Netron and every other model-graph viewer
 * draws a network, and it fills a landscape canvas far better than one very
 * long row. The pins are drawn on the top and bottom edges to match, so a wire
 * leaves the bottom of one block and enters the top of the next without ever
 * doubling back.
 */
export const LAYOUT_DIRECTION = "DOWN";

const OPTIONS = {
  "elk.algorithm": "layered",
  "elk.direction": LAYOUT_DIRECTION,
  "elk.layered.spacing.nodeNodeBetweenLayers": "72",
  "elk.spacing.nodeNode": "64",
  "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
  "elk.layered.crossingMinimization.semiInteractive": "true",
  "elk.edgeRouting": "ORTHOGONAL",
};

/**
 * Options minus the one that cannot survive nesting.
 *
 * `considerModelOrder` keeps a hand-authored order when the layered algorithm
 * has a free choice, which is worth having on a flat graph. Combined with
 * `hierarchyHandling: INCLUDE_CHILDREN` it throws inside ELK, and the failure
 * arrives as a bare "cannot read properties of undefined", so it is worth
 * saying plainly why it is dropped rather than rediscovering it.
 */
const NESTED_OPTIONS = Object.fromEntries(
  Object.entries(OPTIONS).filter(([key]) => !key.includes("considerModelOrder")),
);

export async function layoutGraph(
  nodes: LayoutNode[],
  edges: LayoutEdge[],
): Promise<Record<string, [number, number]>> {
  const placed = await layoutTree(
    nodes.map((n) => ({ id: n.id, width: n.width, height: n.height, parent: null })),
    edges,
  );
  const out: Record<string, [number, number]> = {};
  for (const [id, box] of Object.entries(placed)) out[id] = [box.x, box.y];
  return out;
}

export interface LayoutTreeNode extends LayoutNode {
  /** The frame that encloses this node, or null at the top of the drawing. */
  parent: string | null;
}

export interface LayoutBox {
  /** Position relative to the enclosing frame, which is what React Flow wants. */
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Room for a frame's own caption above its children. */
const FRAME_PADDING = "[top=42,left=22,bottom=22,right=22]";

/**
 * Lay out a tree of frames and leaves in one pass.
 *
 * ELK sizes a parent around its children, which is the whole reason the frames
 * can be drawn at all: nothing here has to guess how big a block that contains
 * six other blocks should be. Edges are declared at the root and
 * `hierarchyHandling: INCLUDE_CHILDREN` lets them cross a frame's wall, which
 * is what a residual bypass and a container's input and output both do.
 */
export async function layoutTree(
  nodes: LayoutTreeNode[],
  edges: LayoutEdge[],
): Promise<Record<string, LayoutBox>> {
  if (nodes.length === 0) return {};
  const elk = await getElk();

  const children = new Map<string | null, LayoutTreeNode[]>();
  for (const n of nodes) {
    const held = children.get(n.parent);
    if (held) held.push(n);
    else children.set(n.parent, [n]);
  }

  const nested = nodes.some((n) => n.parent !== null);
  const options = nested ? NESTED_OPTIONS : OPTIONS;

  const build = (node: LayoutTreeNode): Record<string, unknown> => {
    const kids = children.get(node.id);
    if (!kids || kids.length === 0) {
      return { id: node.id, width: node.width, height: node.height };
    }
    return {
      id: node.id,
      layoutOptions: { ...options, "elk.padding": FRAME_PADDING },
      children: kids.map(build),
    };
  };

  const result = await elk.layout({
    id: "root",
    layoutOptions: nested
      ? { ...options, "elk.hierarchyHandling": "INCLUDE_CHILDREN" }
      : options,
    children: (children.get(null) ?? []).map(build),
    edges: edges.map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] })),
  });

  // Snap to the grid so a laid-out drawing and a hand-placed one share a pitch.
  const snap = (v: number | undefined): number => Math.round((v ?? 0) / 8) * 8;
  const out: Record<string, LayoutBox> = {};
  const collect = (node: ElkNode): void => {
    out[node.id] = {
      x: snap(node.x),
      y: snap(node.y),
      width: Math.round(node.width ?? 0),
      height: Math.round(node.height ?? 0),
    };
    for (const kid of node.children ?? []) collect(kid);
  };
  for (const kid of result.children ?? []) collect(kid);
  return out;
}
