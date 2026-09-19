/**
 * The blocks a joint-embedding predictive architecture needed that a language
 * model never did, and the preset built out of them.
 *
 * Every number here was worked out from I-JEPA's own source rather than from a
 * headline figure, because the headline figure for a ViT-H is 632M and this is
 * not that: I-JEPA fixes its positions to 2-D sincos with `requires_grad=False`
 * and carries no class token, so neither costs a parameter.
 */

import { describe, expect, it } from "bun:test";
import { analyze, getPreset, validate } from "../src/index.js";

describe("learned tokens", () => {
  it("costs one vector however far it is broadcast", () => {
    const doc = getPreset("ijepa-vit-h14");
    const mask = doc.graph.nodes.find((n) => n.id === "mask_token");
    expect(mask?.type).toBe("learned_tokens");
    // One vector of 384, standing in at every position of the grid. What it
    // costs and how far it stretches are different numbers.
    expect(mask?.params).toMatchObject({ count: 1, dim: "Dp", tokens: "T" });

    const a = analyze(doc, {});
    expect(a.params.byPath["mask_token"]).toBe(384);
  });
});

describe("I-JEPA ViT-H/14", () => {
  const doc = getPreset("ijepa-vit-h14");
  const a = analyze(doc, {});
  const byPath = a.params.byPath;
  /** Containers are not leaves, so a tower is the sum of what is inside it. */
  const tower = (prefix: string): number =>
    Object.entries(byPath)
      .filter(([path]) => path === prefix || path.startsWith(prefix + "/"))
      .reduce((n, [, v]) => n + v, 0);

  it("counts the three towers the way PyTorch does", () => {
    // Checked against `python -m tensorcad_runtime verify`, module for module.
    const patchify = 3 * 14 * 14 * 1280 + 1280;
    expect(byPath["patchify"]).toBe(patchify);
    expect(patchify).toBe(753_920);

    // 32 blocks of 1280 wide: two layer norms, qkv, the output projection and a
    // 4x feed-forward, all with bias.
    expect(tower("context")).toBe(629_678_080);
    expect(tower("target")).toBe(629_678_080);
    expect(tower("predictor")).toBe(21_293_568);

    const encoder = patchify + 629_678_080 + 2_560;
    expect(encoder).toBe(630_434_560);

    const predictor = 491_904 + 384 + 21_293_568 + 768 + 492_800;
    expect(predictor).toBe(22_279_424);

    // What trains, and what is resident while it trains.
    expect(encoder + predictor).toBe(652_713_984);
    expect(a.params.total).toBe(encoder + predictor + encoder);
    expect(a.params.total).toBe(1_283_148_544);
  });

  it("reproduces its published figure", () => {
    expect(a.params.total).toBe(doc.meta.published!.params!);
  });

  it("attends bidirectionally", () => {
    // The one line that makes it a vision transformer rather than a language
    // model. Every other preset in the suite leaves this true.
    for (const id of ["context", "predictor", "target"]) {
      const stack = doc.graph.nodes.find((n) => n.id === id);
      const block = stack?.graph?.nodes.find((n) => n.id === "block");
      expect(block?.params?.causal).toBe(false);
    }
  });

  it("pays nothing for its positions", () => {
    // No pos_embedding anywhere, and no class token: I-JEPA's table is sincos
    // and frozen. This is the whole of the difference from a 632M ViT-H.
    const types = new Set<string>();
    const walk = (nodes: typeof doc.graph.nodes): void => {
      for (const n of nodes) {
        types.add(n.type);
        if (n.graph) walk(n.graph.nodes);
      }
    };
    walk(doc.graph.nodes);
    expect(types.has("pos_embedding")).toBe(false);
    expect(types.has("embedding")).toBe(false);
    expect(types.has("lm_head")).toBe(false);
  });

  it("passes the design rules", () => {
    const { findings } = validate(doc, {});
    expect(findings.filter((f) => f.severity === "error")).toEqual([]);
  });
});

describe("AlexNet", () => {
  const doc = getPreset("alexnet");
  const a = analyze(doc, { B: 2, T: 1 });
  const byPath = a.params.byPath;

  it("counts what torchvision counts", () => {
    // Every one of these was checked module for module against
    // `python -m tensorcad_runtime verify`.
    expect(byPath["conv1"]).toBe(3 * 64 * 11 * 11 + 64);
    expect(byPath["conv2"]).toBe(64 * 192 * 5 * 5 + 192);
    expect(byPath["conv3"]).toBe(192 * 384 * 3 * 3 + 384);
    expect(byPath["conv4"]).toBe(384 * 256 * 3 * 3 + 256);
    expect(byPath["conv5"]).toBe(256 * 256 * 3 * 3 + 256);
    expect(byPath["fc1"]).toBe(9216 * 4096 + 4096);
    expect(byPath["fc2"]).toBe(4096 * 4096 + 4096);
    expect(byPath["fc3"]).toBe(4096 * 1000 + 1000);
    expect(a.params.total).toBe(61_100_840);
    expect(a.params.total).toBe(doc.meta.published!.params!);
  });

  it("puts almost all of its weights in the classifier", () => {
    // The thing worth seeing on the drawing: the convolutions do the
    // arithmetic and hold almost none of the memory.
    const conv = ["conv1", "conv2", "conv3", "conv4", "conv5"].reduce((n, k) => n + byPath[k], 0);
    const fc = ["fc1", "fc2", "fc3"].reduce((n, k) => n + byPath[k], 0);
    expect(conv).toBe(2_469_696);
    expect(fc).toBe(58_631_144);
    expect(fc / (conv + fc)).toBeCloseTo(0.96, 2);
  });

  it("agrees with a profiler to the FLOP", () => {
    // `torch.utils.flop_counter` measures 2,856,753,920 at batch 2. Unlike a
    // language model there is no causal mask to disagree about, so this is
    // exact rather than close: 1.428 GFLOP is the familiar 0.714 GMac doubled.
    expect(a.flops.fwdTotal).toBe(1_428_376_960);
  });

  it("shrinks the way the convolutions say it does", () => {
    // 224 -> 55 -> 27 -> 27 -> 13 -> 13 -> 6, computed by the blocks rather
    // than written down: floor((H + 2p - k)/s) + 1 at every step.
    const flat = doc.graph.nodes.find((n) => n.id === "flatten");
    expect(flat?.params).toMatchObject({ channels: 256, in_h: 6, in_w: 6 });
    expect(256 * 6 * 6).toBe(9216);
  });

  it("passes the design rules", () => {
    const { findings } = validate(doc, { B: 2, T: 1 });
    expect(findings.filter((f) => f.severity === "error")).toEqual([]);
  });
});
