/**
 * Shapes and reshapes written in English.
 *
 * The rendering is built in the editor rather than handed over by the engine,
 * because it needs what each letter *counts* and that is a fact about the
 * design: `T` is tokens in a language model, patches in a vision transformer
 * and one image in a convnet, and each design says so in its own symbol table.
 *
 * So the test that matters is not a handful of hand-written cases — it is every
 * shape of every preset, checked for the two ways this can go wrong: printing
 * `NaN` or `undefined` where a symbol did not resolve, and quietly leaving a
 * symbol name in the output where the whole point was to take them out.
 */

import { describe, expect, test } from "bun:test";
import { loadEngine } from "../src/engine.js";

await loadEngine();

const { getPreset, PRESET_NAMES, engine } = await import("../src/engine.js");
const { toEnglish, reshapeInEnglish, formatShape } = await import("../src/canvas/shapes.js");

import type { SymbolTable } from "@tensor-cad/engine";

function symbolsOf(name: string): SymbolTable {
  return engine().analyze(getPreset(name)).symbols;
}

describe("a shape in English", () => {
  test("names what a language model's axes count", () => {
    const s = symbolsOf("llama-3-8b");
    expect(toEnglish("B T D", s)).toBe("1 batch × 8,192 tokens × 4,096");
    expect(toEnglish("B H T dh", s)).toBe("1 batch × 32 heads × 8,192 tokens × 128");
    // A product is one axis holding several things, so the side it measures
    // leads and the factors say what it is made of.
    expect(toEnglish("B T H*dh", s)).toBe("1 batch × 8,192 tokens × 4,096 (32 heads × 128)");
  });

  test("follows the design rather than the letter", () => {
    // The same `T`, in three designs that mean three different things by it.
    expect(toEnglish("B T D", symbolsOf("ijepa-vit-h14"))).toContain("patches");
    expect(toEnglish("B T D", symbolsOf("llama-3-8b"))).toContain("tokens");
    expect(toEnglish("B T", symbolsOf("alexnet"))).toBe("1 image × 1");
  });

  test("prints a literal as itself", () => {
    // A convnet writes its feature maps out: `B 64 55 55`. There is nothing to
    // look up and nothing to name.
    // AlexNet's batch axis is documented as "Images per batch", so it says so.
    expect(toEnglish("B 64 55 55", symbolsOf("alexnet"))).toBe("1 image × 64 × 55 × 55");
  });

  test("degrades to the symbolic form with no table to translate against", () => {
    const shape = { symbolic: "B T D", numeric: "B T 4096" };
    expect(formatShape(shape, "english")).toBe("B T D");
    expect(formatShape(shape, "numeric")).toBe("B T 4096");
    expect(formatShape(shape, "symbolic")).toBe("B T D");
  });
});

describe("a reshape in English", () => {
  const s = symbolsOf("llama-3-8b");

  test("says which way the heads are going", () => {
    expect(reshapeInEnglish("B T (H dh)", "B H T dh", s)).toBe("split 4,096 into 32 heads of 128");
    expect(reshapeInEnglish("B H T dh", "B T (H dh)", s)).toBe(
      "fold 32 heads of 128 back into 4,096",
    );
  });

  test("refuses anything it cannot name", () => {
    // A permutation groups nothing, so there is no split to describe and the
    // einops pattern is the better thing to print.
    expect(reshapeInEnglish("B H T dh", "B T H dh", s)).toBeNull();
    // Both sides grouped: whatever this is, it is not the head split.
    expect(reshapeInEnglish("B (H dh) T", "B T (H dh)", s)).toBeNull();
    // A symbol with no value cannot be counted.
    expect(reshapeInEnglish("B T (Zz dh)", "B Zz T dh", s)).toBeNull();
  });
});

describe("every shape of every preset", () => {
  test("renders without a NaN, an undefined or a leftover symbol", () => {
    const bad: string[] = [];
    for (const name of PRESET_NAMES) {
      const doc = getPreset(name);
      const symbols = symbolsOf(name);
      const known = new Set(Object.keys(symbols.values));
      const inference = engine().infer(doc, "expanded") as unknown as Record<string, unknown>;

      const walk = (value: unknown): void => {
        if (!value || typeof value !== "object") return;
        const shape = value as { symbolic?: unknown };
        if (typeof shape.symbolic === "string") {
          const said = toEnglish(shape.symbolic, symbols);
          if (/NaN|undefined|Infinity/.test(said)) bad.push(`${name}: ${shape.symbolic} → ${said}`);
          // A symbol that resolved must not survive into the English form. One
          // that did not resolve is printed as written, deliberately, so only
          // the known ones are an error.
          for (const word of said.split(/[^A-Za-z_]+/)) {
            if (word && known.has(word)) bad.push(`${name}: ${shape.symbolic} → ${said}`);
          }
          return;
        }
        for (const v of Object.values(value)) walk(v);
      };
      walk(inference);
    }
    expect(bad).toEqual([]);
  });
});
