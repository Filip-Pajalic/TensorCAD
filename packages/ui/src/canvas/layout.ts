/**
 * Auto-layout with elkjs.
 *
 * The layered algorithm running downwards matches how these designs read:
 * tokens at the top, logits at the bottom. ELK runs in a Web Worker when the
 * bundler can produce one, and falls back to the bundled main-thread build.
 */

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

async function getElk(): Promise<ElkLike> {
  elkPromise ??= (async (): Promise<ElkLike> => {
    try {
      const api = await import("elkjs/lib/elk-api.js");
      const ELK = api.default;
      const elk: ElkLike = new ELK({
        workerFactory: () =>
          new Worker(new URL("elkjs/lib/elk-worker.min.js", import.meta.url), { type: "classic" }),
      });
      // Smoke test: a worker that failed to start only shows up on first use.
      await elk.layout({ id: "probe", layoutOptions: {}, children: [], edges: [] });
      return elk;
    } catch {
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
