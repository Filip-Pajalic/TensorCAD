import { describe, expect, it } from "bun:test";
import { Sym } from "../src/shapes/symexpr.js";
import { evalExpr, exprSymbols } from "../src/shapes/expr.js";

const ctx = (values: Record<string, number>) => ({ values, known: new Set(Object.keys(values)) });

describe("Sym", () => {
  it("adds and cancels", () => {
    const D = Sym.v("D");
    expect(D.add(D).toString()).toBe("2*D");
    expect(D.sub(D).isZero()).toBe(true);
  });

  it("multiplies into monomials", () => {
    const e = Sym.v("H").mul(Sym.v("dh"));
    expect(e.toString()).toBe("H*dh");
    expect(e.mul(Sym.v("H")).toString()).toBe("H^2*dh");
  });

  it("keeps rational coefficients exact", () => {
    const e = Sym.v("D").mul(Sym.con(8)).divExact(Sym.con(3));
    expect(e).not.toBeNull();
    expect(e!.toString()).toBe("8/3*D");
    expect(e!.toNumber({ D: 3 })).toBe(8);
  });

  it("divides monomials exactly and refuses when it cannot", () => {
    const num = Sym.v("H").mul(Sym.v("dh"));
    expect(num.divExact(Sym.v("H"))!.toString()).toBe("dh");
    expect(Sym.v("D").divExact(Sym.v("H"))).toBeNull();
  });

  it("substitutes design symbols and keeps runtime symbols indeterminate", () => {
    const e = Sym.v("B").mul(Sym.v("T")).mul(Sym.v("D"));
    const s = e.subst({ D: 4096 });
    expect(s.toString()).toBe("4096*B*T");
    expect(s.toNumber({ B: 2, T: 8 })).toBe(4096 * 16);
  });

  it("compares under a partial environment", () => {
    const lhs = Sym.v("D");
    const rhs = Sym.v("H").mul(Sym.v("dh"));
    expect(lhs.equalsUnder(rhs, { D: 4096, H: 32, dh: 128 })).toBe(true);
    expect(lhs.equalsUnder(rhs, { D: 4096, H: 32, dh: 64 })).toBe(false);
  });

  it("detects the difference between distinct runtime symbols", () => {
    expect(Sym.v("B").equalsUnder(Sym.v("T"), {})).toBe(false);
  });

  it("round-trips through toString for a mixed polynomial", () => {
    const e = Sym.v("D").pow(2).mul(Sym.con(12)).add(Sym.v("D").mul(Sym.con(13)));
    expect(e.toString()).toBe("12*D^2 + 13*D");
    expect(e.toNumber({ D: 768 })).toBe(12 * 768 * 768 + 13 * 768);
  });

  it("recognizes a bare atom", () => {
    expect(Sym.v("D").asAtom()).toBe("D");
    expect(Sym.v("D").mul(Sym.con(2)).asAtom()).toBeNull();
  });
});

describe("expression parser", () => {
  it("parses arithmetic with precedence", () => {
    const e = evalExpr("2*D + H*dh", ctx({ D: 10, H: 3, dh: 4 }));
    expect(e.toNumber({ D: 10, H: 3, dh: 4 })).toBe(32);
  });

  it("folds function calls eagerly", () => {
    // Llama 3's feed-forward rule.
    const e = evalExpr("ceil_mult(1.3*8/3*D, 1024)", ctx({ D: 4096 }));
    expect(e.asConst()).toBe(14336);
  });

  it("supports exponents", () => {
    expect(evalExpr("D^2", ctx({ D: 5 })).toNumber({ D: 5 })).toBe(25);
  });

  it("rejects unknown symbols and functions", () => {
    expect(() => evalExpr("Q", ctx({ D: 1 }))).toThrow(/Unknown symbol/);
    expect(() => evalExpr("nope(D)", ctx({ D: 1 }))).toThrow(/Unknown function/);
  });

  it("reports the symbols an expression uses", () => {
    expect(exprSymbols("ceil_mult(1.3*8/3*D, 1024) + H").sort()).toEqual(["D", "H"]);
  });

  it("substitutes named expressions so shapes show symbol names", () => {
    const e = evalExpr("in_features", {
      values: { in_features: 4096, D: 4096 },
      known: new Set(["in_features", "D"]),
      substitutions: { in_features: Sym.v("D") },
    });
    expect(e.toString()).toBe("D");
  });
});
