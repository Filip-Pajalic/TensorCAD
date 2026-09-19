/**
 * The design as a volume.
 *
 * This is a port of the layout in Brendan Bycroft's LLM visualisation
 * (https://github.com/bbycroft/llm-viz — MIT, Copyright (c) 2023-2026 Brendan
 * Bycroft), specifically `src/llm/GptModelLayout.ts`. The conventions are his,
 * kept deliberately close so the two read the same:
 *
 *   - x runs left and right, y is **positive downward** and the stack advances
 *     down from the origin, z comes out of the page;
 *   - a block is a grid of cells: `cx` cells wide, `cy` **tall**, `cz` deep,
 *     each `cell` units across, so `dx = cx * cell`, `dy = cy * cell`,
 *     `dz = cz * cell`;
 *   - the residual pathway is `cx: T, cy: C, cz: B` — a tall standing plate, not
 *     a flat tile — and it runs down the centre at x = 0;
 *   - weights and off-residual values are placed to its left, each positioned by
 *     exactly one of a left, right or middle edge in x, and one in z;
 *   - attention heads fan out along z.
 *
 * Two departures, both forced by this being a design tool rather than a fixed
 * demo. Their `cell` is a constant 1.5, which is right for a 48-channel model
 * and impossible for a 128k vocabulary, so `cell` here is chosen to give the
 * residual column a fixed height. And the number of heads drawn is capped, since
 * a 32-layer, 32-head design is a thousand head groups.
 */

import type { Derived } from "../state/derive.js";
import type { Doc, NodeDef } from "@tensorcad/engine";
import { formatCount, joinPath } from "@tensorcad/engine";
import { CATALOG } from "../engine.js";

/** Weights, intermediate values, or an aggregate (layer norm and softmax). */
export type BlkKind = "w" | "i" | "a";

export interface Blk {
  name: string;
  kind: BlkKind;
  /** Node path this came from, when one can be named, for cross-probing. */
  path: string | null;
  /** Minimum corner, in the reference's frame: y positive downward. */
  x: number;
  y: number;
  z: number;
  dx: number;
  dy: number;
  dz: number;
  /** Cell counts, which is what the grid on each face is drawn from. */
  cx: number;
  cy: number;
  cz: number;
  /** Axis labels, e.g. `T` by `C`. */
  dimX: string;
  dimY: string;
  /** Not worth drawing on a large model. */
  small: boolean;
  /**
   * Which label group this belongs to.
   *
   * The reference groups its cubes — the Q weights, the Q bias and the Q
   * vectors share one label — and lights the whole group when any one of them
   * is hovered. Same idea here: hovering the attention matrix names the softmax
   * beside it, because they are one step rather than three boxes.
   */
  group: string;
  /** Which block of the stack, or -1 outside it. */
  layer: number;
}

/** Which way a dogleg turns, for the rounded inside of the corner. */
export type ArrowCorner = "none" | "left" | "right";

/**
 * One run of ribbon, as the reference's `drawArrow` takes it.
 *
 * This is deliberately not a polyline. The reference builds a frame from the
 * run's direction, flattens the endpoints into it, and then — whenever the two
 * ends are at different depths, or the ribbon has to arrive side-on — sweeps a
 * cubic bezier between them, so the pathway genuinely curves through z. That is
 * what carries six attention heads back into one output, and a straight segment
 * list cannot express it. A dogleg is therefore two runs, the second carrying
 * the corner and the head.
 */
export interface Arrow {
  start: [number, number, number];
  end: [number, number, number];
  width: number;
  kind: BlkKind;
  head: boolean;
  corner: ArrowCorner;
  /**
   * The direction the ribbon arrives from, when it lands on a face other than
   * the top — which is how Q and K reach the front of the attention matrix.
   * Null means straight down, the reference's default of `(0, 1, 0)`.
   */
  endDir: [number, number, number] | null;
}

export interface Model3D {
  name: string;
  blocks: Blk[];
  arrows: Arrow[];
  cell: number;
  shape: Shape;
  blocksDrawn: number;
  headsDrawn: number;
  columns: number;
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
    centre: [number, number, number];
    radius: number;
  };
  totalParams: number;
  notes: string[];
}

export interface Shape {
  B: number;
  T: number;
  C: number;
  /** Head dimension, `A` in the reference. */
  A: number;
  nHeads: number;
  nKvHeads: number;
  ffn: number;
  vocab: number;
  nBlocks: number;
  hasPosEmbed: boolean;
  tiedHead: boolean;
  experts: number;
  /** Path of the repeat container, for cross-probing a layer. */
  stackPath: string;
  /** Path of a representative block inside it. */
  blockPath: string;
  embedPath: string | null;
  headPath: string | null;
}

const num = (v: unknown, fallback = 0): number =>
  typeof v === "number" && Number.isFinite(v) ? v : fallback;

/** Read the design's shape, which is all the layout needs. */
function shapeOf(doc: Doc, derived: Derived): Shape {
  const s: Shape = {
    B: derived.analysis.options.B,
    T: derived.analysis.options.T,
    C: 768,
    A: 64,
    nHeads: 12,
    nKvHeads: 12,
    ffn: 3072,
    vocab: 50257,
    nBlocks: 1,
    hasPosEmbed: false,
    tiedHead: false,
    experts: 0,
    stackPath: "",
    blockPath: "",
    embedPath: null,
    headPath: null,
  };

  const visit = (nodes: NodeDef[], prefix: string): void => {
    for (const node of nodes) {
      const path = joinPath(prefix, node.id);
      const r = derived.infer.resolved.get(path);
      if (!r) continue;
      switch (node.type) {
        case "embedding":
          s.vocab = num(r.p.vocab, s.vocab);
          s.C = num(r.p.dim, s.C);
          s.embedPath = path;
          break;
        case "pos_embedding":
          s.hasPosEmbed = true;
          break;
        case "lm_head":
          s.tiedHead = r.p.tied === true;
          s.headPath = path;
          break;
        case "transformer_block":
          s.C = num(r.p.d_model, s.C);
          s.nHeads = num(r.p.heads, s.nHeads);
          s.nKvHeads = num(r.p.kv_heads, s.nHeads);
          s.A = num(r.p.head_dim, Math.max(1, Math.floor(s.C / s.nHeads)));
          s.ffn = num(r.p.ffn_hidden, s.C * 4);
          if (r.p.mlp === "moe") {
            s.experts = num(r.p.experts, 0);
            s.ffn = num(r.p.expert_hidden, s.ffn);
          }
          s.blockPath = path;
          break;
        case "repeat":
          s.nBlocks = Math.max(s.nBlocks, num(r.p.count, 1));
          s.stackPath = path;
          break;
        default:
          break;
      }
      if (node.graph) visit(node.graph.nodes, path);
    }
  };
  visit(doc.graph.nodes, "");
  return s;
}

/** How tall the residual column is, in world units, whatever the model. */
const COLUMN_HEIGHT = 30;
/** Past this the reference merges Q, K and V and drops the inter-head margin. */
const LARGE_MODEL_BLOCKS = 12;
const MAX_BLOCKS_DRAWN = 32;
const MAX_HEADS_DRAWN = 6;

interface MkArgs {
  kind: BlkKind;
  cx: number;
  cy: number;
  cz: number;
  y: number;
  xL?: number;
  xR?: number;
  xM?: number;
  zF?: number;
  zB?: number;
  zM?: number;
  name: string;
  dimX: string;
  dimY: string;
  small?: boolean;
  path?: string | null;
  layer?: number;
}

export function buildModel3D(doc: Doc, derived: Derived): Model3D {
  const notes: string[] = [];
  const shape = shapeOf(doc, derived);
  const { B, T, C, A, nHeads, ffn, vocab, hasPosEmbed, tiedHead } = shape;

  const cell = COLUMN_HEIGHT / Math.max(1, C);
  const margin = (C * cell) / 6;
  const isLargeModel = shape.nBlocks > LARGE_MODEL_BLOCKS;

  /**
   * The thinnest a block is drawn.
   *
   * The reference's `cell` is a constant 1.5 against a 48-channel model, so a
   * one-cell-deep weight plate is two per cent of its own height and plainly
   * visible. Scaling `cell` to the residual width — which is what makes a real
   * model fit — shrinks that to two hundredths of a per cent, and every plate
   * renders as a flat sheet with no sides at all.
   */
  const MIN_THICKNESS = COLUMN_HEIGHT * 0.035;

  /**
   * How far a ribbon starts off the face it leaves.
   *
   * The reference's `pad` is 2.0 against a 12-unit margin, so it is a sixth of
   * the gap between two stacked blocks. Taking the ratio rather than the number
   * keeps the arrows off the plates whatever `cell` works out to here.
   */
  const ARROW_PAD = margin / 6;

  /**
   * How wide a ribbon between two blocks is.
   *
   * The reference sizes an arrow to the tensor it carries: 6 units against a
   * residual plate 9 wide, and an explicit 2 where it feeds a one-cell-wide
   * aggregate. So the width is two thirds of the narrower of the two faces,
   * which reproduces both cases.
   *
   * The ceiling is the part that has to be reasoned about rather than copied.
   * The reference's 6 sits inside a 12-unit margin, and a ribbon wider than the
   * gap it runs through turns its own dog-leg inside out — the corner ends up
   * above the edge it left. Its demo has six tokens, so its plates are narrower
   * than its margins; a real design's are six times wider, and two thirds of a
   * face would be far past that. So the width is capped at the margin: the
   * widest ribbon that still fits between two stacked blocks.
   */
  const arrowWidth = (src: Blk, dst: Blk): number =>
    Math.min(Math.max((Math.min(src.dx, dst.dx) * 2) / 3, margin / 3), margin);

  /**
   * The width of the ribbon running down the residual pathway.
   *
   * A branch off it starts at that ribbon's own edge rather than the block's,
   * which is the reference's `residWidth`, so it has to be known up front.
   */
  const RESID_WIDTH = Math.max(T * cell, MIN_THICKNESS);
  const RESID_ARROW = Math.min(Math.max((RESID_WIDTH * 2) / 3, margin / 3), margin);
  /** The reference passes an explicit 2 against its 6 wherever an arrow feeds an aggregate. */
  const AGG_ARROW = RESID_ARROW / 3;

  const blocksDrawn = Math.min(shape.nBlocks, MAX_BLOCKS_DRAWN);
  if (shape.nBlocks > blocksDrawn) {
    notes.push(`Showing ${blocksDrawn} of ${shape.nBlocks} blocks.`);
  }
  const headsDrawn = Math.min(nHeads, MAX_HEADS_DRAWN);
  if (nHeads > headsDrawn) {
    notes.push(`Showing ${headsDrawn} of ${nHeads} attention heads per block.`);
  }

  const blocks: Blk[] = [];
  /** Deferred per-block wiring, run once the arrow helpers below exist. */
  const wiring: (() => void)[] = [];
  let y = 0;
  let layer = -1;
  /** The label group blocks are currently being added to. */
  let group = "";

  function mk(args: MkArgs): Blk {
    const dx = Math.max(args.cx * cell, MIN_THICKNESS);
    const dy = Math.max(args.cy * cell, MIN_THICKNESS);
    const dz = Math.max(args.cz * cell, MIN_THICKNESS);
    const x = args.xL !== undefined ? args.xL : args.xR !== undefined ? args.xR - dx : args.xM! - dx / 2;
    const z = args.zB !== undefined ? args.zB : args.zF !== undefined ? args.zF - dz : args.zM! - dz / 2;
    const blk: Blk = {
      name: args.name,
      kind: args.kind,
      path: args.path ?? null,
      x,
      y: args.y,
      z,
      dx,
      dy,
      dz,
      cx: args.cx,
      cy: args.cy,
      cz: args.cz,
      dimX: args.dimX,
      dimY: args.dimY,
      small: args.small ?? false,
      group: group || args.name,
      layer: args.layer ?? layer,
    };
    blocks.push(blk);
    return blk;
  }

  // --- input ---------------------------------------------------------------

  group = "";
  mk({
    kind: "i",
    cx: T,
    cy: 1,
    cz: B,
    y,
    xM: 0,
    zM: 0,
    name: "Tokens",
    dimX: "T",
    dimY: "",
    path: shape.embedPath,
  });

  let leftX = (-T * cell) / 2 - margin;
  let rightX = (T * cell) / 2 + margin;
  y += cell + margin;

  mk({
    kind: "w",
    cx: vocab,
    cy: C,
    cz: 1,
    y,
    xR: leftX,
    zM: 0,
    name: "Token Embed",
    dimX: "n_vocab",
    dimY: "C",
    path: shape.embedPath,
  });
  if (hasPosEmbed) {
    mk({
      kind: "w",
      cx: T,
      cy: C,
      cz: 1,
      y,
      xL: rightX,
      zM: 0,
      name: "Position Embed",
      dimX: "T",
      dimY: "C",
    });
  }
  const inputEmbed = mk({
    kind: "i",
    cx: T,
    cy: C,
    cz: B,
    y,
    xM: 0,
    zM: 0,
    name: "Input Embed",
    dimX: "T",
    dimY: "C",
  });

  y += C * cell + margin;

  // --- one layer norm ------------------------------------------------------

  let lnLeftX = leftX - (T + 2) * cell - 3 * margin;

  interface Ln {
    agg: Blk;
    gamma: Blk;
    resid: Blk;
  }

  function createLn(src: string | null, name: string): Ln {
    const resLeftX = lnLeftX - T * cell - margin;
    group = `ln:${name}:${layer}:${Math.round(y)}`;
    const agg = mk({ kind: "a", cx: T, cy: 1, cz: B, y, xR: lnLeftX, zM: 0, name: "LN Agg: μ, σ", dimX: "T", dimY: "", small: true });
    mk({ kind: "a", cx: T, cy: 1, cz: B, y: y + cell, xR: lnLeftX, zM: 0, name: "", dimX: "T", dimY: "", small: true });
    y += 2 * cell + margin;
    const gamma = mk({ kind: "w", cx: 1, cy: C, cz: 1, y, xR: resLeftX, zM: 0, name: "γ", dimX: "", dimY: "C", small: true });
    const resid = mk({ kind: "i", cx: T, cy: C, cz: B, y, xR: lnLeftX, zM: 0, name, dimX: "T", dimY: "C", path: src });
    return { agg, gamma, resid };
  }

  // --- one transformer block ----------------------------------------------

  /** The blocks of one layer, kept so the arrows can be wired between them. */
  interface Head {
    qkvWeight?: Blk;
    qkv?: Blk;
    mtx?: Blk;
    smx?: Blk;
    vOut?: Blk;
  }

  function createLayer(prevResid: Blk | undefined): Blk | undefined {
    const blockPath = shape.blockPath || null;
    const ln1 = createLn(blockPath, "Layer Norm");
    const heads: Head[] = [];

    const interHeadMargin = 3 * margin + (C * cell) / 16;
    const qkvMargin = 1 * margin + (C * cell) / 16;
    const headWidth = 3 * B * cell + qkvMargin * 2 + (isLargeModel ? 0 : interHeadMargin);

    const attn1Y = y + A * cell + margin + (isLargeModel ? 2 * A * cell : 0);
    const vOutY = attn1Y + T * cell + margin;

    const attnLeftX = lnLeftX;
    const qkvValLeftX = attnLeftX - T * cell - margin;
    const qkvBiasLeftX = qkvValLeftX - C * cell - margin;
    const attnPath = blockPath ? `${blockPath}/attn` : null;

    const wire: (() => void)[] = [];
    for (let i = 0; i < headsDrawn; i++) {
      const head: Head = {};
      const headZMid = headWidth * i - ((headsDrawn - 1) * headWidth) / 2;
      const qMid = headZMid + B * cell + qkvMargin;
      const kMid = headZMid;
      const vMid = headZMid - B * cell - qkvMargin;

      // Q, K and V are always three blocks at three depths, whatever the size
      // of the model. The reference keeps them apart too, and only drops the
      // margin between heads once there are a lot of them — and it has to,
      // because the depth between them is what every arrow into the attention
      // matrix curves through. Merging them, which this used to do, threw that
      // away and left the head reading as one flat sandwich.
      const vectors: Partial<Record<"Q" | "K" | "V", Blk>> = {};
      for (const [name, zM] of [
        ["Q", qMid],
        ["K", kMid],
        ["V", vMid],
      ] as const) {
        group = `${name}:${layer}:${i}`;
        const w = mk({ kind: "w", cx: C, cy: A, cz: 1, y, xR: qkvValLeftX, zM, name: `${name} Weights`, dimX: "C", dimY: "A", path: attnPath });
        const bias = mk({ kind: "w", cx: 1, cy: A, cz: 1, y, xR: qkvBiasLeftX, zM, name: `${name} Bias`, dimX: "", dimY: "A", small: true });
        const v = mk({ kind: "i", cx: T, cy: A, cz: B, y, xR: attnLeftX, zM, name: `${name} vectors`, dimX: "T", dimY: "A", path: attnPath });
        vectors[name] = v;
        if (name === "K") {
          head.qkvWeight = w;
          head.qkv = v;
        }
        wire.push(() => {
          horiz(bias, w);
          horiz(w, v);
          arrow(ln1.resid, "left", v, "right");
        });
      }

      const attn2LeftX = attnLeftX - (T + 2) * cell - 2 * margin;
      group = `mtx:${layer}:${i}`;
      head.mtx = mk({ kind: "i", cx: T, cy: T, cz: B, y: attn1Y, xR: attnLeftX, zM: headZMid, name: "Attention Matrix", dimX: "T", dimY: "T", path: attnPath });
      const agg = mk({ kind: "a", cx: 1, cy: T, cz: B, y: attn1Y, xR: attnLeftX - T * cell - margin, zM: headZMid, name: "", dimX: "", dimY: "T", small: true });
      head.smx = mk({ kind: "i", cx: T, cy: T, cz: B, y: attn1Y, xR: attn2LeftX, zM: headZMid, name: "Attn Matrix Softmax", dimX: "T", dimY: "T", path: attnPath });
      group = `vout:${layer}:${i}`;
      head.vOut = mk({ kind: "i", cx: T, cy: A, cz: B, y: vOutY, xR: attnLeftX, zM: headZMid, name: "V Output", dimX: "T", dimY: "A", path: attnPath });

      heads.push(head);
      wire.push(() => {
        // Q and K leave the bottom of their vectors and arrive on the face of
        // the attention matrix, which is at a different depth; V does the same
        // into its output. Those four are the arrows that curve.
        botToSide(vectors.Q ?? head.qkv, head.mtx);
        botToSide(vectors.K ?? head.qkv, head.mtx);
        botToSide(vectors.V ?? head.qkv, head.vOut);
        arrow(head.mtx, "left", agg, "right");
        arrow(agg, "left", head.smx, "right");
        arrow(head.smx, "bot", head.vOut, "left");
      });
    }

    group = `proj:${layer}`;
    const vFinalY = Math.max(vOutY + A * cell + 2 * margin, y + C * cell + margin);

    const projWeight = mk({ kind: "w", cx: C, cy: C, cz: 1, y: vFinalY, xR: qkvValLeftX, zM: 0, name: "Projection Weights", dimX: "C", dimY: "C", path: attnPath });
    const attnOut = mk({ kind: "i", cx: T, cy: C, cz: B, y: vFinalY, xR: attnLeftX, zM: 0, name: "Attention Output", dimX: "T", dimY: "C", path: attnPath });
    const attnResid = mk({ kind: "i", cx: T, cy: C, cz: B, y: vFinalY, xM: 0, zM: 0, name: "Attention Residual", dimX: "T", dimY: "C" });

    y = vFinalY + C * cell + margin;

    const ln2 = createLn(blockPath, "Layer Norm");

    const mlpPath = blockPath ? `${blockPath}/mlp` : null;
    group = `mlp:${layer}`;
    const mlpName = shape.experts > 0 ? `Expert Weights ×${shape.experts}` : "MLP Weights";

    const mlpW = mk({ kind: "w", cx: ffn, cy: C, cz: 1, y, xR: attnLeftX, zM: 0, name: mlpName, dimX: "F", dimY: "C", path: mlpPath });
    y += C * cell + margin;
    const mlpFc = mk({ kind: "i", cx: ffn, cy: T, cz: B, y, xR: attnLeftX, zM: 0, name: "MLP", dimX: "F", dimY: "T", path: mlpPath });
    y += T * cell + margin;
    const mlpAct = mk({ kind: "i", cx: ffn, cy: T, cz: B, y, xR: attnLeftX, zM: 0, name: "MLP Activation", dimX: "F", dimY: "T", path: mlpPath });
    y += T * cell + margin;
    const mlpProjW = mk({ kind: "w", cx: ffn, cy: C, cz: 1, y, xR: attnLeftX, zM: 0, name: "MLP Projection Weights", dimX: "F", dimY: "C", path: mlpPath });
    const mlpResult = mk({ kind: "i", cx: T, cy: C, cz: B, y, xL: attnLeftX + margin, zM: 0, name: "MLP Result", dimX: "T", dimY: "C", path: mlpPath });
    const mlpResid = mk({ kind: "i", cx: T, cy: C, cz: B, y, xM: 0, zM: 0, name: "MLP Residual", dimX: "T", dimY: "C" });

    y += C * cell - margin;

    // The whole block's wiring, deferred until the arrow helpers exist.
    wiring.push(() => {
      // The residual runs straight down and is tapped twice on the way.
      vert(prevResid, attnResid, RESID_ARROW);
      residSplit(prevResid, ln1.resid);
      residSplit(prevResid, ln1.agg, AGG_ARROW);
      vert(ln1.agg, ln1.resid, AGG_ARROW);
      horiz(ln1.gamma, ln1.resid);

      for (const step of wire) step();
      for (const h of heads) vert(h.vOut, attnOut);

      vert(attnResid, mlpResid, RESID_ARROW);
      horiz(projWeight, attnOut);
      horiz(attnOut, attnResid);

      residSplit(attnResid, ln2.resid);
      residSplit(attnResid, ln2.agg, AGG_ARROW);
      vert(ln2.agg, ln2.resid, AGG_ARROW);
      horiz(ln2.gamma, ln2.resid);

      arrow(ln2.resid, "bot", mlpFc, "right");
      vert(mlpW, mlpFc);
      vert(mlpFc, mlpAct);
      horiz(mlpProjW, mlpResult);
      arrow(mlpAct, "right", mlpResult, "top");
      horiz(mlpResult, mlpResid);
    });

    return mlpResid;
  }

  // The stack wraps into columns rather than running to a spike: past twelve
  // blocks the reference shifts every x reference right by a column width and
  // starts again from the top. Without this a 32-block design is a sliver
  // twenty-six times taller than it is wide.
  const blockHalfMargin = 2 * margin;
  const columnWidth = C * 14 * cell + margin * 2;
  const blocksPerColumn = 12;
  let blockIdxInColumn = 0;
  let columns = 1;

  y += blockHalfMargin;
  const blockYTop = y;
  let resid: Blk | undefined = inputEmbed;


  // --- arrows --------------------------------------------------------------
  // Ported from the reference's `components/Arrow.ts`. A ribbon leaves the
  // middle of one block's edge and arrives at the middle of another's, padded
  // off the face; where the two edges do not face each other it dog-legs, and
  // where they sit at different depths, or it has to land side-on, the renderer
  // sweeps it as a curve. Only the first few blocks get them — the reference
  // hard-codes three — because past that they are a green haze rather than a
  // path.

  const arrows: Arrow[] = [];
  type Side = "left" | "right" | "top" | "bot";
  type V3 = [number, number, number];

  const edgeOf = (b: Blk, side: Side): V3 => {
    const z = b.z + b.dz / 2;
    switch (side) {
      case "left":
        return [b.x - ARROW_PAD, b.y + b.dy / 2, z];
      case "right":
        return [b.x + b.dx + ARROW_PAD, b.y + b.dy / 2, z];
      case "top":
        return [b.x + b.dx / 2, b.y - ARROW_PAD, z];
      default:
        return [b.x + b.dx / 2, b.y + b.dy + ARROW_PAD, z];
    }
  };

  function run(
    start: V3,
    end: V3,
    width: number,
    kind: BlkKind,
    head: boolean,
    corner: ArrowCorner = "none",
    endDir: V3 | null = null,
  ): void {
    arrows.push({ start, end, width, kind, head, corner, endDir });
  }

  /** `drawArrowBetween`: edge to edge, dog-legged where they do not face. */
  function arrow(
    src: Blk | undefined,
    srcSide: Side,
    dst: Blk | undefined,
    dstSide: Side = "left",
    width?: number,
  ): void {
    if (!src || !dst) return;
    const start = edgeOf(src, srcSide);
    const end = edgeOf(dst, dstSide);
    const w = width ?? arrowWidth(src, dst);

    // Side to side reads as one straight run, so the start is levelled first.
    if ((srcSide === "left" && dstSide === "right") || (srcSide === "right" && dstSide === "left")) {
      start[1] = end[1];
    }

    if (srcSide === "right" && dstSide === "top") {
      // dogleg right then down
      run(start, [end[0] - w / 2, start[1], start[2]], w, src.kind, false);
      run([end[0], start[1] + w / 2, end[2]], end, w, src.kind, true, "left");
    } else if (srcSide === "bot" && dstSide === "right") {
      // dogleg down then right
      run(start, [start[0], end[1] - w / 2, end[2]], w, src.kind, false);
      run([start[0] - w / 2, end[1], end[2]], end, w, src.kind, true, "left");
    } else if (srcSide === "bot" && dstSide === "left") {
      // dogleg down then left
      run(start, [start[0], end[1] - w / 2, end[2]], w, src.kind, false, "none", [0, 1, 0]);
      run([start[0] + w / 2, end[1], end[2]], end, w, src.kind, true, "right");
    } else {
      run(start, end, w, src.kind, true);
    }
  }

  const vert = (src: Blk | undefined, dst: Blk | undefined, width?: number): void =>
    arrow(src, "bot", dst, "top", width);
  const horiz = (src: Blk | undefined, dst: Blk | undefined, width?: number): void =>
    arrow(src, "right", dst, "left", width);

  /**
   * `drawArrowResidSplit`: the branch off the residual pathway.
   *
   * It does not dogleg. It starts on the edge of the vertical ribbon already
   * running down the residual and goes straight across, which is what makes the
   * split read as a tap rather than a second route.
   */
  function residSplit(src: Blk | undefined, dst: Blk | undefined, width?: number): void {
    if (!src || !dst) return;
    const start = edgeOf(src, "bot");
    const end = edgeOf(dst, "right");
    const w = width ?? arrowWidth(src, dst);
    run([start[0] - RESID_ARROW / 2, end[1], start[2]], end, w, src.kind, true);
  }

  /**
   * `drawArrowBotToSide`: down out of one block and into the face of another.
   *
   * Q and K leave the bottom of their vectors and arrive on the front or back
   * of the attention matrix, which sits at a different depth because the heads
   * fan along z. Handing the renderer an `endDir` is what makes it sweep the
   * ribbon round rather than draw a flat L.
   */
  function botToSide(
    src: Blk | undefined,
    dst: Blk | undefined,
    offset = 0,
    width?: number,
    forceOffset = false,
  ): void {
    if (!src || !dst) return;
    const start = edgeOf(src, "bot");
    const dstMidZ = dst.z + dst.dz / 2;
    const left = start[2] > dstMidZ;
    let end: V3 = [
      dst.x + dst.dx / 2,
      dst.y + cell * (offset + 0.5),
      left ? dstMidZ + ARROW_PAD : dst.z - ARROW_PAD,
    ];
    let endDir: V3 | null = [0, 0, left ? -1 : 1];

    // Close enough in depth that a curve would be a wobble: land on the top.
    if (Math.abs(start[2] - dstMidZ) < MIN_THICKNESS && !forceOffset) {
      endDir = null;
      end = edgeOf(dst, "top");
    }
    run(start, end, width ?? arrowWidth(src, dst), src.kind, true, "none", endDir);
  }

  for (let i = 0; i < blocksDrawn; i++) {
    if (blockIdxInColumn >= blocksPerColumn) {
      blockIdxInColumn = 0;
      columns++;
      y = blockYTop;
      lnLeftX += columnWidth;
      leftX += columnWidth;
      rightX += columnWidth;
    }
    layer = i;
    y += blockHalfMargin;
    resid = createLayer(resid);
    y += blockHalfMargin;
    blockIdxInColumn++;
  }
  layer = -1;

  // Every block that is drawn is wired.
  //
  // The reference stops at three — `for (let i = 0; i < 3; i++)`, with the real
  // bound commented out beside it — and this followed it. On a model with
  // twelve blocks that is most of the stack; on one with thirty-two it means
  // the drawing simply stops being a drawing a tenth of the way down, which
  // reads as a bug rather than as a decision. Whatever the reference's reason
  // was, it cost about eight thousand triangles in four draw calls to disagree
  // with it.
  for (const run of wiring) run();

  // --- output --------------------------------------------------------------

  y += blockHalfMargin;
  createLn(null, "Final Layer Norm");

  group = "head";
  y += C * cell + margin;
  const leftX2 = leftX - T * cell - margin;

  mk({
    kind: "w",
    cx: C,
    cy: vocab,
    cz: 1,
    y,
    xR: leftX2,
    zM: 0,
    name: tiedHead ? "LM Head (tied)" : "LM Head Weights",
    dimX: "C",
    dimY: "n_vocab",
    path: shape.headPath,
  });
  mk({ kind: "i", cx: T, cy: vocab, cz: B, y, xR: leftX, zM: 0, name: "Logits", dimX: "T", dimY: "n_vocab", path: shape.headPath });

  y += vocab * cell + margin;
  mk({ kind: "a", cx: T, cy: 1, cz: B, y, xR: leftX, zM: 0, name: "SM Agg", dimX: "T", dimY: "", small: true });
  y += 2 * cell + margin;
  mk({ kind: "i", cx: T, cy: vocab, cz: B, y, xR: leftX, zM: 0, name: "Logits Softmax", dimX: "T", dimY: "n_vocab" });

  // --- bounds --------------------------------------------------------------

  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const b of blocks) {
    min[0] = Math.min(min[0], b.x);
    max[0] = Math.max(max[0], b.x + b.dx);
    min[1] = Math.min(min[1], -(b.y + b.dy));
    max[1] = Math.max(max[1], -b.y);
    min[2] = Math.min(min[2], b.z);
    max[2] = Math.max(max[2], b.z + b.dz);
  }
  if (!Number.isFinite(min[0])) {
    min[0] = min[1] = min[2] = -1;
    max[0] = max[1] = max[2] = 1;
  }
  const centre: [number, number, number] = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];

  return {
    name: doc.meta.name || "design",
    blocks,
    arrows,
    cell,
    shape,
    blocksDrawn,
    headsDrawn,
    columns,
    bounds: {
      min,
      max,
      centre,
      radius: Math.max(
        4,
        0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]),
      ),
    },
    totalParams: derived.params.total,
    notes: [...new Set(notes)],
  };
}

/** A one-line description, for the hover readout. */
export function describeBlk(b: Blk): string {
  const dims = [b.cx, b.cy, b.cz].filter((n) => n > 1);
  const shape = dims.map((n) => n.toLocaleString("en-US")).join(" × ");
  const kind = b.kind === "w" ? "weights" : b.kind === "a" ? "aggregate" : "activations";
  return `${b.name || kind} · ${shape}`;
}

export { formatCount };
