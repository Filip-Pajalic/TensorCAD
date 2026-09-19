/**
 * Unfolding: draw a container's interior inside the container.
 *
 * Every published architecture figure draws the whole model at once. The
 * repeated block is a rounded rectangle with `32×` on its edge, and inside it
 * are the norms, the attention, the feed-forward and the two residual sums,
 * with each bypass drawn as a line up the side. None of them make you open a
 * block to find out what is in it.
 *
 * This turns one flat level into that: a tree of frames and leaves, plus edges
 * between the leaves with every container boundary short-circuited away. A
 * `repeat` holding a `transformer_block` stops being two opaque boxes and
 * becomes a frame inside a frame around the six parts that do the work.
 *
 * `depth` is how many levels to open. Zero is the old behaviour: one flat
 * graph, editable and hand-placed. Above zero the drawing is generated, so it
 * is arranged by the layout engine and its blocks are inspected, not dragged.
 */

import type { BlockDef, Doc, Graph, NodeDef } from "@tensorcad/core";
import { catalogOf, isComposite, isContainer, joinPath, splitEndpoint } from "@tensorcad/core";
import type { Derived } from "./derive.js";
import type { Level } from "./level.js";

export interface UnfoldedNode {
  /** Full path from the document root. */
  path: string;
  node: NodeDef;
  def: BlockDef | undefined;
  /** The frame that encloses this node, or null at the top of the view. */
  parent: string | null;
  /** How many frames deep, for ordering and for tinting. */
  depth: number;
  /** True when this node is drawn as a frame around its children. */
  frame: boolean;
  /** Stack count written on a frame's edge, as a figure writes "32x". */
  multiplier: number | null;
  /**
   * When a frame held nothing but one other frame, the two were drawn as one.
   * This is what the inner one was, which is the identity worth showing: a
   * `repeat` around a single `transformer_block` is one rectangle in every
   * published figure, not two.
   */
  mergedType?: string;
  mergedCategory?: string;
}

export interface UnfoldedEdge {
  id: string;
  source: string;
  sourcePort: string;
  target: string;
  targetPort: string;
  /**
   * True when the wire arrives at the second input of a sum: the residual line
   * every transformer figure draws running up the side of the block.
   */
  bypass: boolean;
}

export interface Unfolded {
  nodes: UnfoldedNode[];
  edges: UnfoldedEdge[];
  /** Paths drawn as frames, so the canvas knows which node type to use. */
  frames: Set<string>;
  /** True when anything was opened, so the view is generated rather than placed. */
  generated: boolean;
  /** True when something could still be opened, for the depth control. */
  moreAvailable: boolean;
}

interface Endpoint {
  path: string;
  port: string;
}

const key = (e: Endpoint): string => `${e.path}:${e.port}`;

interface Entry {
  node: NodeDef;
  def: BlockDef | undefined;
  parent: string | null;
}

/** The subgraph a node would show if it were opened, or null if it has none. */
function interiorOf(
  node: NodeDef,
  def: BlockDef | undefined,
  path: string,
  derived: Derived,
): Graph | null {
  if (!def) return null;
  if (isContainer(def)) return node.graph && node.graph.nodes.length > 0 ? node.graph : null;
  if (!isComposite(def)) return null;
  const resolved = derived.infer.resolved.get(path);
  if (!resolved) return null;
  try {
    return def.expand(resolved.rawFull, resolved);
  } catch {
    // A block whose parameters did not resolve is already reported as an issue.
    // Here it simply stays closed.
    return null;
  }
}

const isBoundaryIn = (def: BlockDef | undefined): boolean => def?.type === "boundary_in";
const isBoundaryOut = (def: BlockDef | undefined): boolean => def?.type === "boundary_out";

function push<K, V>(map: Map<K, V[]>, k: K, v: V): void {
  const held = map.get(k);
  if (held) held.push(v);
  else map.set(k, [v]);
}


/**
 * Collapse a frame that contains nothing but one other frame.
 *
 * `repeat` around a single `transformer_block` is how the document models a
 * stack, but drawing it as two nested rectangles says there are two things
 * there when there is one. Every published figure draws a single rounded
 * rectangle with the count on its edge, so the outer frame keeps its count and
 * takes the inner one's identity, and the inner one goes away.
 */
function mergeTrivialFrames(nodes: UnfoldedNode[], frames: Set<string>): UnfoldedNode[] {
  let live = nodes;
  for (;;) {
    const kids = new Map<string, UnfoldedNode[]>();
    for (const n of live) {
      if (n.parent === null) continue;
      const held = kids.get(n.parent);
      if (held) held.push(n);
      else kids.set(n.parent, [n]);
    }

    const victim = live.find((f) => {
      if (!f.frame) return false;
      const mine = kids.get(f.path);
      return mine?.length === 1 && mine[0].frame;
    });
    if (!victim) return live;

    const inner = kids.get(victim.path)![0];
    victim.mergedType = inner.mergedType ?? inner.node.type;
    victim.mergedCategory = inner.mergedCategory ?? inner.def?.category;
    frames.delete(inner.path);
    live = live
      .filter((n) => n.path !== inner.path)
      .map((n) => (n.parent === inner.path ? { ...n, parent: victim.path } : n));
  }
}

export function unfold(doc: Doc, level: Level, derived: Derived, depth: number): Unfolded {
  const cat = catalogOf(doc);
  const nodes: UnfoldedNode[] = [];
  /** Every node reached, including the boundaries that get short-circuited. */
  const byPath = new Map<string, Entry>();
  const frames = new Set<string>();
  const rawEdges: { from: Endpoint; to: Endpoint }[] = [];
  const wanted = Math.max(0, depth);
  let moreAvailable = false;

  const walk = (graph: Graph, prefix: string, parent: string | null, remaining: number): void => {
    for (const node of graph.nodes) {
      const path = joinPath(prefix, node.id);
      const def = cat[node.type];
      byPath.set(path, { node, def, parent });

      const interior = interiorOf(node, def, path, derived);
      const open = interior !== null && remaining > 0;
      if (interior !== null && !open) moreAvailable = true;

      // A boundary inside an opened frame is that frame's own port, not a part
      // of the drawing. It stays in `byPath` so edges can be traced through it
      // and out of `nodes` so nothing draws it. At the top of the view — when
      // the user has drilled into a container — it is a real node again.
      const hidden = parent !== null && (isBoundaryIn(def) || isBoundaryOut(def));
      if (!hidden) {
        const count = derived.infer.resolved.get(path)?.p.count;
        nodes.push({
          path,
          node,
          def,
          parent,
          depth: wanted - remaining,
          frame: open,
          multiplier: open && typeof count === "number" && count > 1 ? count : null,
        });
      }
      if (open) {
        frames.add(path);
        walk(interior, path, path, remaining - 1);
      }
    }

    for (const [from, to] of graph.edges) {
      try {
        const f = splitEndpoint(from);
        const t = splitEndpoint(to);
        rawEdges.push({
          from: { path: joinPath(prefix, f.node), port: f.port },
          to: { path: joinPath(prefix, t.node), port: t.port },
        });
      } catch {
        // A malformed endpoint is reported by the analysis, not drawn.
      }
    }
  };

  walk(level.graph, level.prefix, null, wanted);

  // --- short-circuit the boundaries ----------------------------------------
  // A frame's port and its boundary node's port carry the same name — both
  // `streamBoundary` and every authored container subgraph honour that — so the
  // two match without knowing which kind of block the frame is.

  const bySource = new Map<string, Endpoint[]>();
  const byTarget = new Map<string, Endpoint[]>();
  for (const e of rawEdges) {
    push(bySource, key(e.from), e.to);
    push(byTarget, key(e.to), e.from);
  }

  const boundaries = new Map<string, { in: string | null; out: string | null }>();
  for (const [path, entry] of byPath) {
    if (entry.parent === null) continue;
    const which = isBoundaryIn(entry.def) ? "in" : isBoundaryOut(entry.def) ? "out" : null;
    if (!which) continue;
    const held = boundaries.get(entry.parent) ?? { in: null, out: null };
    held[which] = path;
    boundaries.set(entry.parent, held);
  }

  /**
   * Follow a signal forwards until it reaches nodes the drawing shows. Entering
   * an opened frame means entering whatever its input boundary feeds; reaching
   * a frame's output boundary means leaving the frame.
   */
  const realTargets = (ep: Endpoint, seen: Set<string>): Endpoint[] => {
    if (seen.has(key(ep))) return [];
    seen.add(key(ep));
    const entry = byPath.get(ep.path);
    if (!entry) return [];

    if (frames.has(ep.path)) {
      const inb = boundaries.get(ep.path)?.in;
      if (!inb) return [ep];
      return (bySource.get(`${inb}:${ep.port}`) ?? []).flatMap((n) => realTargets(n, seen));
    }
    if (entry.parent !== null && isBoundaryOut(entry.def)) {
      return (bySource.get(`${entry.parent}:${ep.port}`) ?? []).flatMap((n) =>
        realTargets(n, seen),
      );
    }
    return [ep];
  };

  const drawn = new Set(nodes.map((n) => n.path));
  const edges: UnfoldedEdge[] = [];
  const emitted = new Set<string>();

  for (const raw of rawEdges) {
    // Trace only from a real producer. An edge starting at a hidden boundary or
    // at an opened frame is reached by following one of these instead, so
    // skipping it here is what keeps each wire from being drawn twice.
    const src = raw.from;
    if (!drawn.has(src.path) || frames.has(src.path)) continue;

    for (const dst of realTargets(raw.to, new Set())) {
      if (!drawn.has(dst.path)) continue;
      const id = `${key(src)}->${key(dst)}`;
      if (emitted.has(id)) continue;
      emitted.add(id);
      edges.push({
        id,
        source: src.path,
        sourcePort: src.port,
        target: dst.path,
        targetPort: dst.port,
        bypass: dst.port === "b" && byPath.get(dst.path)?.def?.type === "add",
      });
    }
  }

  // Merging happens last. `frames` is what `realTargets` walks through to find
  // the parts on either side of a container wall, so folding two frames into
  // one before the edges are resolved would cut every wire that crosses it.
  const merged = mergeTrivialFrames(nodes, frames);
  return { nodes: merged, edges, frames, generated: frames.size > 0, moreAvailable };
}
