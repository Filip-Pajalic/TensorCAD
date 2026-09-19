/**
 * Joint-embedding predictive architectures.
 *
 * Three towers rather than one, which is the whole idea. A context encoder sees
 * part of the image; a narrow predictor is handed those embeddings plus one
 * learned mask token for every patch it has to guess; and a target encoder —
 * the same network, kept as an exponential moving average and never
 * back-propagated through — produces what it is guessing at. No decoder, and no
 * pixel-space loss anywhere.
 *
 * Two things a JEPA has to say that a language preset never does:
 *
 *  - **Attention is not causal.** A patch may look at every other patch. The
 *    catalog has carried `causal` as a parameter all along; this is the first
 *    design to turn it off.
 *  - **Positions are free.** I-JEPA fixes its position table to 2-D sincos with
 *    `requires_grad=False` and uses no class token, so neither costs a
 *    parameter. A ViT-H is usually quoted at 632M precisely because it carries
 *    a learned table and a classifier head. This one carries neither, and comes
 *    out at 630.4M.
 */

import type { Doc, Graph, NodeDef, ParamValue, SymbolDef } from "../ir/types.js";
import { DOC_VERSION } from "../ir/types.js";
import type { DecoderSpec } from "./builders.js";

export interface JepaSpec {
  name: string;
  family?: string;
  notes?: string;

  /** Context and target encoder. */
  layers: number;
  dModel: number;
  heads: number;
  ffnHidden: number | string;

  /** Predictor. Narrow and shallow beside the encoder. */
  predLayers: number;
  predDim: number;
  predHeads: number;
  predFfnHidden?: number | string;

  /** Patchifier. A stride-p, kernel-p convolution is a linear over flat patches. */
  patch: number;
  imageSize: number;
  channels?: number;
  /** Video models cut a time dimension into tubelets as well. */
  frames?: number;
  tubelet?: number;

  /**
   * Draw the exponential-moving-average target encoder as its own tower.
   *
   * It is a real copy of the weights and it really is resident while you train,
   * so counting it is honest — but it is not the number anyone quotes, and
   * `meta.published` has to be explicit about which one it is.
   */
  emaTarget?: boolean;

  published?: DecoderSpec["published"];
}

export function visionJepa(spec: JepaSpec): Doc {
  const channels = spec.channels ?? 3;
  const tubelet = spec.tubelet ?? 1;
  const frames = spec.frames ?? tubelet;
  const side = Math.floor(spec.imageSize / spec.patch);
  const patches = side * side * Math.floor(frames / tubelet);
  const patchPixels = channels * spec.patch * spec.patch * tubelet;

  const headDim = spec.dModel / spec.heads;
  const predHeadDim = spec.predDim / spec.predHeads;
  const predFfn = spec.predFfnHidden ?? spec.predDim * 4;

  const grid = spec.frames ? `${side}x${side}x${frames / tubelet}` : `${side}x${side}`;

  const symbols: Record<string, SymbolDef> = {
    B: { kind: "runtime", default: 1, doc: "Batch size" },
    T: { kind: "runtime", default: patches, doc: `Patches in the whole grid (${grid})` },
    L: { kind: "design", value: spec.layers, doc: "Encoder layers" },
    D: { kind: "design", value: spec.dModel, doc: "Encoder width" },
    H: { kind: "design", value: spec.heads, doc: "Encoder heads" },
    dh: { kind: "design", value: headDim, doc: "Encoder head dimension" },
    F: { kind: "design", value: spec.ffnHidden, doc: "Encoder feed-forward width" },
    Lp: { kind: "design", value: spec.predLayers, doc: "Predictor layers" },
    Dp: { kind: "design", value: spec.predDim, doc: "Predictor width" },
    Hp: { kind: "design", value: spec.predHeads, doc: "Predictor heads" },
    dhp: { kind: "design", value: predHeadDim, doc: "Predictor head dimension" },
    Fp: { kind: "design", value: predFfn, doc: "Predictor feed-forward width" },
    P: {
      kind: "design",
      value: patchPixels,
      doc: `Values in one patch (${channels} x ${spec.patch} x ${spec.patch}${tubelet > 1 ? ` x ${tubelet}` : ""})`,
    },
  };

  /** A ViT block: pre-norm, bidirectional attention, dense GELU feed-forward. */
  const vitBlock = (d: string, h: string, hd: string, f: string): Record<string, ParamValue> => ({
    d_model: d,
    heads: h,
    kv_heads: h,
    head_dim: hd,
    ffn_hidden: f,
    norm: "layernorm",
    norm_bias: true,
    mlp: "dense",
    act: "gelu",
    attn_bias: true,
    mlp_bias: true,
    // Every patch may attend to every other. This one line is what makes it a
    // vision transformer rather than a language model.
    causal: false,
    rope: null,
  });

  const stack = (
    id: string,
    count: string,
    label: string,
    params: Record<string, ParamValue>,
    width: string,
    seq: string,
  ): NodeDef => ({
    id,
    type: "repeat",
    label,
    params: { count },
    graph: {
      nodes: [
        { id: "_in", type: "boundary_in", params: { ports: { x: `B ${seq} ${width}` } } },
        { id: "block", type: "transformer_block", params },
        { id: "_out", type: "boundary_out", params: { ports: { x: `B ${seq} ${width}` } } },
      ],
      edges: [
        ["_in:x", "block:x"],
        ["block:y", "_out:x"],
      ],
    },
  });

  const nodes: NodeDef[] = [
    { id: "image", type: "input", label: "Image patches", params: { shape: "B T P", dtype: "fp32" } },
    {
      id: "patchify",
      type: "linear",
      label: "Patch embed",
      params: { in_features: "P", out_features: "D", bias: true },
    },
    stack("context", "L", "Context encoder x" + spec.layers, vitBlock("D", "H", "dh", "F"), "D", "T"),
    { id: "context_norm", type: "layernorm", label: "Encoder norm", params: { dim: "D", bias: true, eps: 1e-6 } },

    {
      id: "to_pred",
      type: "linear",
      label: "Into predictor",
      params: { in_features: "D", out_features: "Dp", bias: true },
    },
    {
      id: "mask_token",
      type: "learned_tokens",
      label: "Mask token",
      // One vector learned, stood in every position the encoder was not shown.
      params: { count: 1, dim: "Dp", tokens: "T" },
    },
    {
      id: "with_masks",
      type: "add",
      label: "Context + masks",
      /*
       * An approximation, and the only one in this design.
       *
       * I-JEPA masks per step: the encoder is shown one block covering 85-100%
       * of the grid and the predictor is handed those embeddings concatenated
       * with a mask token for each position left out, resampled every
       * iteration. Drawing that needs two towers running at two different
       * sequence lengths, and `T` is a reserved symbol every composite is
       * written against, so both towers here run over the whole grid and the
       * choice between an embedding and the mask token is drawn as a combine.
       *
       * Nothing this tool computes is affected: parameters are identical, and
       * FLOPs and activations over the full grid are the worst case, which is
       * the number worth designing against.
       */
      params: { dim: "Dp" },
    },
    stack("predictor", "Lp", "Predictor x" + spec.predLayers, vitBlock("Dp", "Hp", "dhp", "Fp"), "Dp", "T"),
    { id: "pred_norm", type: "layernorm", label: "Predictor norm", params: { dim: "Dp", bias: true, eps: 1e-6 } },
    {
      id: "to_embed",
      type: "linear",
      label: "Back to encoder width",
      params: { in_features: "Dp", out_features: "D", bias: true },
    },
    { id: "prediction", type: "output", label: "Predicted embeddings" },
  ];

  const edges: Graph["edges"] = [
    ["image:x", "patchify:x"],
    ["patchify:y", "context:x"],
    ["context:x", "context_norm:x"],
    ["context_norm:y", "to_pred:x"],
    ["to_pred:y", "with_masks:a"],
    ["mask_token:y", "with_masks:b"],
    ["with_masks:y", "predictor:x"],
    ["predictor:x", "pred_norm:x"],
    ["pred_norm:y", "to_embed:x"],
    ["to_embed:y", "prediction:x"],
  ];

  if (spec.emaTarget) {
    nodes.push(
      {
        id: "target_patchify",
        type: "linear",
        label: "Target patch embed",
        params: { in_features: "P", out_features: "D", bias: true },
      },
      stack("target", "L", "Target encoder x" + spec.layers + " (EMA)", vitBlock("D", "H", "dh", "F"), "D", "T"),
      { id: "target_norm", type: "layernorm", label: "Target norm", params: { dim: "D", bias: true, eps: 1e-6 } },
      { id: "targets", type: "output", label: "Target embeddings" },
    );
    edges.push(
      ["image:x", "target_patchify:x"],
      ["target_patchify:y", "target:x"],
      ["target:x", "target_norm:x"],
      ["target_norm:y", "targets:x"],
    );
  }

  return {
    version: DOC_VERSION,
    meta: { name: spec.name, family: spec.family, notes: spec.notes, published: spec.published },
    symbols,
    graph: { nodes, edges },
    ui: {},
  };
}
