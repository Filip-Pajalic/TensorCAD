/**
 * Reading a training run, and knowing when two of them are not comparable.
 *
 * The files here are the two shapes the trainer actually writes: the `.jsonl`
 * it appends a line to per logged step, and the `.json` record of the whole
 * run. Both are read, because the one that matters most is usually the run
 * that stopped early — and that one has only the `.jsonl`.
 */

import { describe, expect, test } from "bun:test";
import { incomparable, parseRun, type RunRecord } from "../src/state/runs.js";

const step = (n: number, loss: number, extra: Record<string, unknown> = {}) =>
  JSON.stringify({
    step: n,
    loss,
    lr: 3e-4,
    tokens: n * 2048,
    tokens_per_second: 151213,
    peak_memory_bytes: 6_500_000_000,
    seconds: n * 0.05,
    batch: 8,
    seq: 256,
    ...extra,
  });

const jsonl = [step(1, 10.85), step(2, 8.1), step(3, 5.4), step(4, 3.23)].join("\n");

const record = (patch: Record<string, unknown> = {}) =>
  JSON.stringify({
    ok: true,
    model_path: "out/tiny-gpt2/model.py",
    params: 30_100_000,
    steps: 4,
    final_loss: 3.23,
    tokens_per_second: 151213,
    device: "cuda",
    dtype: "bfloat16",
    batch: 8,
    seq: 256,
    lr: 3e-4,
    seed: 1337,
    data_source: "fineweb-edu",
    log: JSON.parse(`[${[step(1, 10.85), step(2, 8.1), step(3, 5.4), step(4, 3.23)].join(",")}]`),
    ...patch,
  });

describe("reading a run", () => {
  test("a step log is enough on its own", () => {
    const run = parseRun("tiny-gpt2-500.jsonl", jsonl);
    expect(run.steps).toBe(4);
    expect(run.initial_loss).toBe(10.85);
    expect(run.final_loss).toBe(3.23);
    expect(run.best_loss).toBe(3.23);
    // Taken off the last step, which is the only place a step log has them.
    expect(run.tokens).toBe(8192);
    expect(run.seq).toBe(256);
  });

  test("a record carries what the log cannot", () => {
    const run = parseRun("tiny-gpt2.json", record());
    expect(run.params).toBe(30_100_000);
    expect(run.seed).toBe(1337);
    expect(run.data_source).toBe("fineweb-edu");
    expect(run.log).toHaveLength(4);
  });

  test("the label says which design it was, not only which file", () => {
    expect(parseRun("20260918-160225.json", record()).label).toBe("tiny-gpt2 · 20260918-160225");
    // Unless the file name already said so, in which case saying it twice is
    // just a longer label.
    expect(parseRun("tiny-gpt2-500.json", record()).label).toBe("tiny-gpt2-500");
  });

  test("the same file twice is the same run, not two", () => {
    expect(parseRun("a.jsonl", jsonl).id).toBe(parseRun("a.jsonl", jsonl).id);
    expect(parseRun("a.jsonl", jsonl).id).not.toBe(parseRun("a.jsonl", jsonl.replace("3.23", "3.24")).id);
  });

  test("a failed run says so rather than drawing an empty chart", () => {
    expect(() => parseRun("bad.json", JSON.stringify({ ok: false, error: "CUDA out of memory" }))).toThrow(
      /CUDA out of memory/,
    );
  });

  test("something that is not a run is refused by name", () => {
    expect(() => parseRun("design.json", JSON.stringify({ meta: { name: "x" } }))).toThrow(/no "log"/);
    expect(() => parseRun("empty.jsonl", "\n\n")).toThrow(/no steps/);
  });
});

describe("whether two runs can be compared", () => {
  const base = parseRun("a.json", record());
  const twin = parseRun("b.json", record());

  test("one run is not a comparison, so there is nothing to warn about", () => {
    expect(incomparable([base])).toEqual([]);
  });

  test("two runs under the same conditions are comparable", () => {
    expect(incomparable([base, twin])).toEqual([]);
  });

  test("a different sequence length is not a fair fight", () => {
    const longer = parseRun("c.json", record({ seq: 1024 }));
    expect(incomparable([base, longer]).join("; ")).toContain("sequence length differs");
  });

  test("so is a different corpus, seed, batch, learning rate or dtype", () => {
    const check = (patch: Record<string, unknown>, expected: string): void => {
      const other = parseRun(`${expected}.json`, record(patch));
      expect(incomparable([base, other]).join("; ")).toContain(expected);
    };
    check({ data_source: "tinystories" }, "corpus differs");
    check({ seed: 7 }, "seed differs");
    check({ batch: 16 }, "batch size differs");
    check({ lr: 1e-3 }, "learning rate differs");
    check({ dtype: "float32" }, "dtype differs");
  });

  test("every difference is named, not just the first", () => {
    const other = parseRun("d.json", record({ seq: 1024, seed: 7 }));
    expect(incomparable([base, other])).toHaveLength(2);
  });

  test("a field only one run reports is not a difference", () => {
    // A step log carries no seed; the record beside it does. One value and one
    // silence is not two values, and "seed differs: 1337" would be a warning
    // that names a difference nobody can see — which is how people learn to
    // stop reading warnings.
    const bare: RunRecord = parseRun("e.jsonl", jsonl);
    expect(bare.seed).toBeUndefined();
    expect(incomparable([base, bare])).toEqual([]);
  });
});
