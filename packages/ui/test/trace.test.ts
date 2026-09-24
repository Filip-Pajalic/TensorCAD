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
const { addTrace, cellsFor, loadTrace, modelSource, parseTrace, traceFor } = await import("../src/three/trace.js");
const { buildWalkthrough } = await import("../src/state/walkthrough.js");
const { KEEP, memoryShelf, setTraceShelf, traceShelf } = await import("../src/three/trace-shelf.js");
/** The committed trace's contents, as they came out of the file. */
const committedFile = () => (trace as NonNullable<typeof trace>).file;

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

describe("a trace added while the editor runs", () => {
  const committed = trace!.file;
  const relu = structuredClone(doc);
  const block = relu.graph.nodes.find((n) => n.id === "layers")!.graph!.nodes.find((n) => n.id === "block")!;
  block.params = { ...block.params, act: "relu" };
  relu.meta = { ...relu.meta, name: "nano-sort-relu" };

  const hashOf = async (d: Doc): Promise<string> => {
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(modelSource(d)!));
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  };

  test("is refused when it is not a trace, and says why", () => {
    expect(() => parseTrace("{ nope")).toThrow(/not JSON/);
    expect(() => parseTrace(JSON.stringify({ version: 2 }))).toThrow(/version 2/);
    expect(() => parseTrace(JSON.stringify({ ...committed, model_sha256: "x" }))).toThrow(/fingerprint/);
  });

  test("shows on the design whose model it was made from, and no other", async () => {
    // The committed numbers relabelled as a run of the relu variant: the same
    // shapes, so every box resolves, and a fingerprint only that design has.
    const file = { ...structuredClone(committed), design: "nano-sort-relu", model_sha256: await hashOf(relu) };
    expect(await traceFor(relu)).toBeNull();

    const result = await loadTrace(JSON.stringify(file), relu);
    expect(result.ok).toBe(true);
    expect(result.message).toContain("Open the volume view");
    expect((await traceFor(relu))?.file.design).toBe("nano-sort-relu");

    // The committed design keeps its own.
    expect((await traceFor(doc))?.file.design).toBe("nano-sort");
  });

  test("of another design is kept, and the message says it is not this one", async () => {
    const other = { ...structuredClone(committed), design: "elsewhere", model_sha256: "0".repeat(64) };
    const result = await loadTrace(JSON.stringify(other), doc);
    expect(result.ok).toBe(true);
    expect(result.message).toContain("does not generate the model it was made from");
    expect((await traceFor(doc))?.file.design).toBe("nano-sort");
  });

  test("untrained, says so everywhere it is quoted, and invents nothing", async () => {
    const untrained: typeof committed = {
      ...structuredClone(committed),
      design: "nano-sort-untrained",
      task: { name: "untrained", length: null, vocab: 3, symbols: [] },
      training: { seed: 1337, steps: 0, final_loss: null, held_out_accuracy: null },
      answer: null,
      checks: { ...committed.checks, sorted_correctly: null },
    };
    const t = addTrace(untrained);
    expect(t.untrained).toBe(true);
    expect(t.letters.slice(0, 3)).toEqual(["2", "1", "0"]);
    expect(t.summary).toStartWith("untrained, as initialised, on 11 token ids");

    // Only what the trace added: the design's own notes mention sorting, and
    // that is the design talking, not the run.
    const derived = derive(doc, DEFAULT_OPERATING);
    const before = new Set(buildWalkthrough(doc, derived).flatMap((s) => s.body));
    const said = buildWalkthrough(doc, derived, t)
      .flatMap((s) => s.body)
      .filter((p) => !before.has(p))
      .join(" ");
    expect(said).toContain("never been trained");
    expect(said).toMatch(/against \d+% if it were exactly even/);
    expect(said).toMatch(/against \d+% for an even spread over all 3/);
    expect(said).not.toMatch(/sort|NaN|undefined|null/);
  });
});

describe("the shelf a trace is kept on between visits", () => {
  const hashOf = async (d: Doc): Promise<string> => {
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(modelSource(d)!));
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  };
  const variant = (act: string): Doc => {
    const d = structuredClone(doc);
    const block = d.graph.nodes.find((n) => n.id === "layers")!.graph!.nodes.find((n) => n.id === "block")!;
    block.params = { ...block.params, act };
    d.meta = { ...d.meta, name: `nano-sort-${act}` };
    return d;
  };

  test("keeps the most recent few, and taking one counts as using it", async () => {
    const shelf = memoryShelf();
    for (let i = 0; i < KEEP; i++) await shelf.put(`h${i}`, `t${i}`);
    expect(await shelf.get("h0")).toBe("t0");
    await shelf.put("new", "t");
    expect(await shelf.count()).toBe(KEEP);
    // h1 was the least recently used once h0 had been taken.
    expect(await shelf.get("h1")).toBeNull();
    expect(await shelf.get("h0")).toBe("t0");
  });

  test("gives a design back a trace that is only on the shelf, as after a reload", async () => {
    setTraceShelf(memoryShelf());
    const silu = variant("silu");
    expect(await traceFor(silu)).toBeNull();
    const text = JSON.stringify({ ...structuredClone(committedFile()), design: "nano-sort-silu", model_sha256: await hashOf(silu) });
    // Put there directly, as an earlier visit would have: nothing in memory knows of it.
    await traceShelf().put(await hashOf(silu), text);
    expect((await traceFor(silu))?.file.design).toBe("nano-sort-silu");
  });

  test("treats something unreadable on the shelf as nothing there", async () => {
    setTraceShelf(memoryShelf());
    const tanh = variant("tanh");
    await traceShelf().put(await hashOf(tanh), "{ not a trace");
    expect(await traceFor(tanh)).toBeNull();
  });

  test("never holds the committed trace, which every copy of the editor has", async () => {
    // The committed one as the file first found it, before any test here added
    // another under nano-sort's own fingerprint.
    expect(trace!.committed).toBe(true);
    const shelf = memoryShelf();
    setTraceShelf(shelf);
    await traceFor(doc);
    expect(await shelf.count()).toBe(0);
  });
});
