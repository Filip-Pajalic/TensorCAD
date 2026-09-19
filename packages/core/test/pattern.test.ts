import { describe, expect, it } from "bun:test";
import { Sym } from "../src/shapes/symexpr.js";
import { instantiate, matchPattern, parsePattern, shapeToString } from "../src/shapes/pattern.js";

const ctx = (values: Record<string, number>) => ({ values, known: new Set(Object.keys(values)) });
const env = { D: 4096, H: 32, dh: 128, Hkv: 8, B: 1, T: 8192 };

describe("patterns", () => {
  it("parses plain dims", () => {
    expect(parsePattern("B T D").atoms).toHaveLength(3);
  });

  it("treats whitespace inside parentheses as multiplication", () => {
    const p = parsePattern("B T (H dh)");
    const inst = instantiate(p, ctx(env), null);
    expect(inst.errors).toEqual([]);
    expect(shapeToString(inst.shape!)).toBe("B T H*dh");
  });

  it("rejects more than one ellipsis", () => {
    expect(() => parsePattern("... D ...")).toThrow(/more than one/);
  });

  it("binds leading dims to the ellipsis", () => {
    const actual = [Sym.v("B"), Sym.v("T"), Sym.v("D")];
    const m = matchPattern(actual, parsePattern("... D"), ctx({ ...env }), env);
    expect(m.ok).toBe(true);
    expect(shapeToString(m.batch)).toBe("B T");
  });

  it("accepts a reshape that preserves the product", () => {
    const actual = [Sym.v("B"), Sym.v("T"), Sym.v("D")];
    const m = matchPattern(actual, parsePattern("B T (H dh)"), ctx({ ...env }), env);
    expect(m.ok).toBe(true);
  });

  it("rejects a reshape that does not preserve the product", () => {
    const actual = [Sym.v("B"), Sym.v("T"), Sym.v("D")];
    const bad = { ...env, dh: 64 };
    const m = matchPattern(actual, parsePattern("B T (H dh)"), ctx(bad), bad);
    expect(m.ok).toBe(false);
    expect(m.errors[0]).toMatch(/dim 2/);
  });

  it("reports a rank mismatch", () => {
    const actual = [Sym.v("B"), Sym.v("D")];
    const m = matchPattern(actual, parsePattern("B H T dh"), ctx({ ...env }), env);
    expect(m.ok).toBe(false);
    expect(m.errors[0]).toMatch(/rank/);
  });
});
