/**
 * The committed trace, and the view's reading of it.
 *
 * A trace is numbers somebody else computed, drawn on a layout that decides
 * for itself which axis of which tensor runs across which box. Both halves can
 * be wrong without anything looking wrong — a transposed weight is still a
 * plausible texture — so the tests do not look at the picture. They recompute
 * things from the pieces the view draws and require the pieces to agree:
 *
 *   - the trace belongs to the design the editor opens, and to no edit of it;
 *   - every box the layout names a tensor for gets cells;
 *   - the input embedding the view draws is the token row plus the position
 *     row it draws beside it, which only holds if all three are read the right
 *     way round;
 *   - each head's output is its own attention row times its own values, which
 *     only holds if the head slices line up across four different boxes.
 */

import { describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();

const { getPreset } = await import("../src/engine.js");
const { derive } = await import("../src/state/derive.js");
const { DEFAULT_OPERATING } = await import("../src/state/operating.js");
const { buildModel3D } = await import("../src/three/model3d.js");
const { cellsFor, modelSource, traceFor } = await import("../src/three/trace.js");
const { buildWalkthrough } = await import("../src/state/walkthrough.js");

import type { Doc } from "@tensor-cad/engine";
import type { Blk } from "../src/three/model3d.js";

const doc = getPreset("nano-sort") as Doc;
const model = buildModel3D(doc, derive(doc, DEFAULT_OPERATING));
const trace = await traceFor(doc);

const named = (name: string, layer = -1, nth = 0): Blk =>
  model.blocks.filter((b) => b.name === name && b.layer === layer)[nth]!;
const cells = (b: Blk): Float32Array => cellsFor(trace!, b.source!, b.cx, b.cy)!;

describe("which design a trace describes", () => {
  test("the one nano-sort generates", async () => {
    // When this fails the code generator has changed what nano-sort compiles
    // to, and the committed numbers are for a model the editor no longer
    // builds. Regenerate them: `bun run trace`.
    expect(trace).not.toBeNull();
  });

  test("and no edit of it that changes what it computes", async () => {
    const wider = structuredClone(doc);
    wider.symbols.D = { ...wider.symbols.D!, value: 64 } as (typeof wider.symbols)[string];
    expect(await traceFor(wider)).toBeNull();

    // Same shapes, same parameter count, different function. A fingerprint
    // made of symbols or counts would let this through.
    const relu = structuredClone(doc);
    const block = relu.graph.nodes.find((n) => n.id === "layers")!.graph!.nodes.find((n) => n.id === "block")!;
    block.params = { ...block.params, act: "relu" };
    expect(modelSource(relu)).not.toBe(modelSource(doc));
    expect(await traceFor(relu)).toBeNull();
  });

  test("but a note is not an edit to the model", async () => {
    const noted = structuredClone(doc);
    noted.meta = { ...noted.meta, notes: `${noted.meta.notes ?? ""} Annotated.` };
    expect(await traceFor(noted)).not.toBeNull();
  });

  test("the model it records sorted its input, and the attention was checked", () => {
    const f = trace!.file;
    expect(f.checks.sorted_correctly).toBe(true);
    expect(f.checks.attention_note).toBeNull();
    expect(f.checks.attention_max_abs_error!).toBeLessThan(1e-4);
    expect(f.answer).toEqual([...f.input].sort((a, b) => a - b));
    expect(f.params).toBe(model.totalParams);
  });
});

describe("the view's reading of it", () => {
  test("every box that names a tensor gets its cells", () => {
    const missing = model.blocks
      .filter((b) => b.source && !cellsFor(trace!, b.source, b.cx, b.cy))
      .map((b) => `${b.name} (layer ${b.layer})`);
    expect(missing).toEqual([]);
    // And almost everything names one. What is left is the softmax aggregate
    // strip and the like, which the trace has no tensor for.
    const unnamed = model.blocks.filter((b) => !b.source).map((b) => b.name);
    expect(unnamed.length).toBeLessThan(model.blocks.length / 10);
  });

  test("the input embedding is the token's row plus the position's", () => {
    const [embed, pos, input, tokens] = [
      named("Token Embed"),
      named("Position Embed"),
      named("Input Embed"),
      named("Tokens"),
    ];
    const [e, p, x, t] = [cells(embed), cells(pos), cells(input), cells(tokens)];
    const T = input.cx;
    for (let pos_ = 0; pos_ < T; pos_++) {
      const token = t[pos_]!;
      for (let c = 0; c < input.cy; c++) {
        // Every one of these is a block's own (x, y): x across, y down.
        const want = e[c * embed.cx + token]! + p[c * pos.cx + pos_]!;
        expect(x[c * T + pos_]!).toBeCloseTo(want, 5);
      }
    }
  });

  test("each head's output is its attention times its values", () => {
    for (let layer = 0; layer < model.blocksDrawn; layer++) {
      for (let h = 0; h < model.headsDrawn; h++) {
        const probs = named("Attn Matrix Softmax", layer, h);
        const values = named("V vectors", layer, h);
        const out = named("V Output", layer, h);
        const [P, V, O] = [cells(probs), cells(values), cells(out)];
        const T = out.cx;
        for (let q = 0; q < T; q++) {
          // Row q of the attention is what position q looked at, and it sums to one.
          let total = 0;
          for (let k = 0; k < T; k++) total += P[q * T + k]!;
          expect(total).toBeCloseTo(1, 5);
          for (let a = 0; a < out.cy; a++) {
            let want = 0;
            for (let k = 0; k <= q; k++) want += P[q * T + k]! * V[a * T + k]!;
            expect(O[a * T + q]!).toBeCloseTo(want, 4);
          }
        }
      }
    }
  });

  test("a masked score is not a zero", () => {
    const scores = cells(named("Attention Matrix", 0, 0));
    const T = named("Attention Matrix", 0, 0).cx;
    // Above the diagonal a query would be looking at the future.
    expect(Number.isNaN(scores[0 * T + 1]!)).toBe(true);
    expect(Number.isFinite(scores[1 * T + 0]!)).toBe(true);
    expect(cells(named("Attn Matrix Softmax", 0, 0))[0 * T + 1]).toBe(0);
  });

  test("a shorter drawing shows the start of the run, and a longer one shows nothing", () => {
    // Causal, so the first T positions of the traced run are what a run of T
    // would have computed. Past the trace there is nothing to show.
    const input = named("Input Embed");
    const full = cells(input);
    const short = cellsFor(trace!, input.source!, 4, input.cy)!;
    for (let c = 0; c < input.cy; c++) {
      for (let t = 0; t < 4; t++) expect(short[c * 4 + t]).toBe(full[c * input.cx + t]!);
    }
    expect(cellsFor(trace!, input.source!, trace!.positions + 1, input.cy)).toBeNull();
  });
});

describe("the walkthrough, with the run", () => {
  const derived = derive(doc, DEFAULT_OPERATING);
  const plain = buildWalkthrough(doc, derived);
  const quoted = buildWalkthrough(doc, derived, trace);

  test("has the same steps, lighting the same blocks", () => {
    // The canvas lights a step by its index. A file finishing loading must
    // not move which blocks a step lights.
    expect(quoted.map((s) => [s.id, s.paths])).toEqual(plain.map((s) => [s.id, s.paths]));
  });

  test("and quotes what the run did, without inventing any of it", () => {
    const said = (id: string) => quoted.find((s) => s.id === id)!.body.join(" ");
    const letters = trace!.file.input.map((t) => trace!.file.task.symbols[t]).join(" ");
    expect(said("input")).toContain(letters);
    expect(said("output")).toContain([...letters.split(" ")].sort().join(" "));
    expect(said("attention")).toMatch(/head \d puts \d+% of its attention on position \d+, the [ABC]\./);
    // The first embedding row, as the table holds it.
    const first = trace!.tensor("embed", -1, "weight")!.data[trace!.file.sequence[0]! * 48]!;
    expect(said("embed")).toContain(first.toPrecision(3).replace(/^-/, "−"));
    for (const step of quoted) {
      for (const p of step.body) expect(p).not.toMatch(/NaN|undefined|\$\{/);
    }
  });
});
