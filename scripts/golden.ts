/**
 * Golden files for the Go port.
 *
 * The Go engine is replacing this one, and "it looks right" is not a migration
 * plan. So the TypeScript writes down, for every preset, the document it builds
 * and the answers it gets; the Go tests read the same documents and have to
 * produce the same answers. A stage of the port is done when its golden file
 * matches, and not before.
 *
 *   bun run scripts/golden.ts
 *
 * Regenerate deliberately: a diff here is either a bug being fixed or a
 * behaviour being changed, and both want to be seen in review.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  PRESET_NAMES,
  Sym,
  evalExpr,
  getPreset,
  instantiate,
  matchPattern,
  parsePattern,
  resolveSymbols,
  shapeToString,
} from "@tensorcad/core";

const root = join(import.meta.dir, "..", "packages", "core-go", "testdata");
const docsDir = join(root, "presets");
const goldenDir = join(root, "golden");
mkdirSync(docsDir, { recursive: true });
mkdirSync(goldenDir, { recursive: true });

/** JSON has no NaN, and a NaN here would mean the preset itself is broken. */
function finite(label: string, values: Record<string, number>): Record<string, number> {
  for (const [k, v] of Object.entries(values)) {
    if (!Number.isFinite(v)) throw new Error(`${label}: symbol ${k} is ${v}`);
  }
  return values;
}

const index: string[] = [];

for (const name of PRESET_NAMES) {
  const doc = getPreset(name);
  writeFileSync(join(docsDir, `${name}.json`), JSON.stringify(doc, null, 2) + "\n");

  const table = resolveSymbols(doc);
  const golden = {
    preset: name,
    symbols: {
      order: table.order,
      values: finite(name, table.values),
      designValues: finite(name, table.designValues),
      runtime: [...table.runtime].sort(),
      docs: table.docs,
      errors: table.errors,
    },
  };
  writeFileSync(join(goldenDir, `${name}.json`), JSON.stringify(golden, null, 2) + "\n");
  index.push(name);
}

writeFileSync(join(root, "presets.json"), JSON.stringify(index, null, 2) + "\n");
/**
 * The expression algebra, pinned separately.
 *
 * A symbol table only exercises the paths its presets happen to take. These are
 * the corners: rational coefficients that do not divide, exact monomial
 * division, eager folding of functions, and the printed form of a polynomial.
 * The printed form is what a shape label on the canvas says, so a divergence
 * there is visible rather than theoretical.
 */
const ENV: Record<string, number> = {
  B: 4,
  T: 2048,
  D: 4096,
  H: 32,
  Hkv: 8,
  dh: 128,
  F: 14336,
  V: 128256,
  L: 32,
};
const KNOWN = new Set(Object.keys(ENV));

const CASES = [
  "D",
  "-D",
  "D + 1",
  "4*D",
  "D*4",
  "H*dh",
  "B*T*D",
  "D/H",
  "D/4",
  "D/3",
  "1/3*D",
  "1.3*8/3*D",
  "ceil_mult(1.3*8/3*D, 1024)",
  "(H + 2*Hkv)*dh",
  "H*dh - D",
  "D^2",
  "2^10",
  "(D + D)/2",
  "D*dh/dh",
  "B*T*(H*dh)/H",
  "floor(D/H)",
  "ceil(D/3)",
  "round(D/3)",
  "min(D, F)",
  "max(D, F)",
  "floor_mult(F, 1024)",
  "round_mult(D*2/3, 128)",
  "log2(D)",
  "sqrt(D)",
  "abs(0 - D)",
  "1_000 + D",
  "1e3 + D",
  "0.5*D",
  "D - D",
  "3*D - 2*D",
];

const ERRORS = ["D +", "D $ 1", "nope", "floor(D, 2)", "notafunction(D)", "D^(0 - 1)", "D/0", "B/T"];

const expressions = {
  env: ENV,
  ok: CASES.map((src) => {
    const sym = evalExpr(src, { values: ENV, known: KNOWN });
    return { src, text: sym.toString(), value: sym.toNumber(ENV), symbols: sym.symbols() };
  }),
  errors: ERRORS.map((src) => {
    try {
      const sym = evalExpr(src, { values: ENV, known: KNOWN });
      return { src, message: "", text: sym.toString() };
    } catch (e) {
      return { src, message: (e as Error).message, text: "" };
    }
  }),
};
writeFileSync(join(root, "expressions.json"), JSON.stringify(expressions, null, 2) + "\n");

/**
 * Shape patterns, pinned the same way.
 *
 * A pattern is where a shape stops being a string and becomes something the
 * engine can disagree with. The cases below are the corners: groups, ellipses
 * that bind nothing, ellipses that bind three dimensions, rank mismatches, and
 * the message a mismatch produces — which is what a person reads off the canvas
 * when a design is wrong, so it has to be identical in both engines.
 */
const PATTERNS = [
  "B T D",
  "B T (H dh)",
  "... D",
  "...",
  "B H T dh",
  "... 2*D",
  "B (H dh) T",
  "B 3 224 224",
  "B T (H dh) D",
];

const PATTERN_ERRORS = ["B T (H dh", "B () D", "... T ... D"];

/** actual shape (as patterns, instantiated), checked against a pattern. */
const MATCHES: [string, string][] = [
  ["B T D", "B T D"],
  ["B T D", "... D"],
  ["B T D", "B T (H dh)"],
  ["B H T dh", "... dh"],
  ["B T D", "B T D D"],
  ["B T D", "B H T dh"],
  ["D", "... D"],
  ["B T F", "... D"],
];

const patternCtx = { values: ENV, known: KNOWN };

const patterns = {
  parse: PATTERNS.map((src) => {
    const p = parsePattern(src);
    const shape = instantiate(p, patternCtx, [Sym.v("B"), Sym.v("T")]);
    return {
      src,
      atoms: p.atoms.map((a) => (a.kind === "ellipsis" ? "..." : a.parts.join(" "))),
      instantiated: shape.shape ? shapeToString(shape.shape) : null,
      errors: shape.errors,
    };
  }),
  parseErrors: PATTERN_ERRORS.map((src) => {
    try {
      parsePattern(src);
      return { src, message: "" };
    } catch (e) {
      return { src, message: (e as Error).message };
    }
  }),
  matches: MATCHES.map(([actualSrc, patternSrc]) => {
    const actual = instantiate(parsePattern(actualSrc), patternCtx, []).shape ?? [];
    const r = matchPattern(actual, parsePattern(patternSrc), patternCtx, ENV);
    return {
      actual: actualSrc,
      pattern: patternSrc,
      ok: r.ok,
      batch: shapeToString(r.batch),
      errors: r.errors,
    };
  }),
};
writeFileSync(join(root, "patterns.json"), JSON.stringify(patterns, null, 2) + "\n");

console.log(
  `wrote ${index.length} presets, their golden symbol tables, ` +
    `${CASES.length + ERRORS.length} expression cases, and ` +
    `${PATTERNS.length + PATTERN_ERRORS.length + MATCHES.length} pattern cases`,
);
