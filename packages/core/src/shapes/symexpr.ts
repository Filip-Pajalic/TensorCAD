/**
 * Exact symbolic arithmetic for tensor dimensions.
 *
 * A `Sym` is a multivariate polynomial with rational coefficients over named
 * symbols (`B`, `T`, `D`, `H`, `dh`, ...). This is exactly the expressive power
 * tensor shapes need: products for reshapes (`(H dh)`), sums for concatenation,
 * and exact divisibility obligations for splits.
 *
 * Design symbols (D, H, F, ...) always have a concrete numeric value; runtime
 * symbols (B, T) stay indeterminate. Two shapes are compatible when their
 * difference is the zero polynomial after substituting the concrete values.
 */

// ---------------------------------------------------------------------------
// Rational numbers
// ---------------------------------------------------------------------------

export interface Rat {
  readonly n: number;
  readonly d: number;
}

function igcd(a: number, b: number): number {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a || 1;
}

/** Build a normalized rational. Accepts non-integer inputs (e.g. 1.3) exactly. */
export function rat(n: number, d = 1): Rat {
  if (d === 0) throw new Error("SymExpr: division by zero");
  if (!Number.isFinite(n) || !Number.isFinite(d)) throw new Error("SymExpr: non-finite value");
  if (!Number.isInteger(n) || !Number.isInteger(d)) {
    let scale = 1;
    while ((!Number.isInteger(n * scale) || !Number.isInteger(d * scale)) && scale < 1e12) scale *= 10;
    n = Math.round(n * scale);
    d = Math.round(d * scale);
  }
  if (d < 0) {
    n = -n;
    d = -d;
  }
  const g = igcd(n, d);
  return { n: n / g, d: d / g };
}

const RAT_ZERO: Rat = { n: 0, d: 1 };
const RAT_ONE: Rat = { n: 1, d: 1 };

const ratAdd = (a: Rat, b: Rat): Rat => rat(a.n * b.d + b.n * a.d, a.d * b.d);
const ratMul = (a: Rat, b: Rat): Rat => rat(a.n * b.n, a.d * b.d);
const ratDiv = (a: Rat, b: Rat): Rat => {
  if (b.n === 0) throw new Error("SymExpr: division by zero");
  return rat(a.n * b.d, a.d * b.n);
};
const ratNeg = (a: Rat): Rat => ({ n: -a.n, d: a.d });
const ratIsZero = (a: Rat): boolean => a.n === 0;
const ratIsOne = (a: Rat): boolean => a.n === 1 && a.d === 1;
const ratNum = (a: Rat): number => a.n / a.d;

function ratToString(a: Rat): string {
  return a.d === 1 ? String(a.n) : `${a.n}/${a.d}`;
}

// ---------------------------------------------------------------------------
// Monomials
// ---------------------------------------------------------------------------

/** Exponent per symbol. Absent symbol means exponent 0. */
export type Mono = Readonly<Record<string, number>>;

function monoKey(m: Mono): string {
  const names = Object.keys(m)
    .filter((k) => m[k] !== 0)
    .sort();
  if (names.length === 0) return "";
  return names.map((k) => (m[k] === 1 ? k : `${k}^${m[k]}`)).join("*");
}

function monoMul(a: Mono, b: Mono): Mono {
  const out: Record<string, number> = { ...a };
  for (const k of Object.keys(b)) {
    const e = (out[k] ?? 0) + b[k];
    if (e === 0) delete out[k];
    else out[k] = e;
  }
  return out;
}

function monoDegree(m: Mono): number {
  let d = 0;
  for (const k of Object.keys(m)) d += m[k];
  return d;
}

interface Term {
  readonly c: Rat;
  readonly m: Mono;
}

// ---------------------------------------------------------------------------
// Sym
// ---------------------------------------------------------------------------

export class Sym {
  /** Canonical monomial key -> term. Never contains zero coefficients. */
  private readonly terms: ReadonlyMap<string, Term>;

  private constructor(terms: Map<string, Term>) {
    this.terms = terms;
  }

  // --- construction -------------------------------------------------------

  static readonly ZERO = new Sym(new Map());

  static con(v: number | Rat): Sym {
    const r = typeof v === "number" ? rat(v) : v;
    if (ratIsZero(r)) return Sym.ZERO;
    return new Sym(new Map([["", { c: r, m: {} }]]));
  }

  static v(name: string): Sym {
    return new Sym(new Map([[name, { c: RAT_ONE, m: { [name]: 1 } }]]));
  }

  static sum(parts: Sym[]): Sym {
    return parts.reduce<Sym>((a, b) => a.add(b), Sym.ZERO);
  }

  static product(parts: Sym[]): Sym {
    return parts.reduce<Sym>((a, b) => a.mul(b), Sym.con(1));
  }

  // --- arithmetic ---------------------------------------------------------

  add(other: Sym): Sym {
    const out = new Map<string, Term>(this.terms);
    for (const [k, t] of other.terms) {
      const cur = out.get(k);
      if (!cur) {
        out.set(k, t);
      } else {
        const c = ratAdd(cur.c, t.c);
        if (ratIsZero(c)) out.delete(k);
        else out.set(k, { c, m: cur.m });
      }
    }
    return new Sym(out);
  }

  neg(): Sym {
    const out = new Map<string, Term>();
    for (const [k, t] of this.terms) out.set(k, { c: ratNeg(t.c), m: t.m });
    return new Sym(out);
  }

  sub(other: Sym): Sym {
    return this.add(other.neg());
  }

  mul(other: Sym): Sym {
    const out = new Map<string, Term>();
    for (const a of this.terms.values()) {
      for (const b of other.terms.values()) {
        const m = monoMul(a.m, b.m);
        const k = monoKey(m);
        const c = ratMul(a.c, b.c);
        const cur = out.get(k);
        if (!cur) {
          out.set(k, { c, m });
        } else {
          const nc = ratAdd(cur.c, c);
          if (ratIsZero(nc)) out.delete(k);
          else out.set(k, { c: nc, m });
        }
      }
    }
    return new Sym(out);
  }

  pow(k: number): Sym {
    if (!Number.isInteger(k) || k < 0) throw new Error(`SymExpr: unsupported exponent ${k}`);
    let acc = Sym.con(1);
    for (let i = 0; i < k; i++) acc = acc.mul(this);
    return acc;
  }

  /**
   * Exact division. Succeeds when `other` is a single term (constant or
   * monomial) that divides every term of this expression, or when the result is
   * otherwise provably exact. Returns null when divisibility cannot be proven,
   * which callers turn into a design-rule obligation.
   */
  divExact(other: Sym): Sym | null {
    if (other.isZero()) return null;
    if (other.terms.size !== 1) {
      // Only handle the case where this is an exact multiple by trying the
      // constant-quotient shortcut.
      const oc = other.asConst();
      if (oc === null) return null;
      return this.divExact(Sym.con(oc));
    }
    const [b] = [...other.terms.values()];
    const out = new Map<string, Term>();
    for (const a of this.terms.values()) {
      const m: Record<string, number> = { ...a.m };
      for (const k of Object.keys(b.m)) {
        const e = (m[k] ?? 0) - b.m[k];
        if (e < 0) return null;
        if (e === 0) delete m[k];
        else m[k] = e;
      }
      const c = ratDiv(a.c, b.c);
      out.set(monoKey(m), { c, m });
    }
    return new Sym(out);
  }

  // --- inspection ---------------------------------------------------------

  isZero(): boolean {
    return this.terms.size === 0;
  }

  /** Numeric value when this is a constant, else null. */
  asConst(): number | null {
    if (this.terms.size === 0) return 0;
    if (this.terms.size === 1) {
      const t = this.terms.get("");
      if (t) return ratNum(t.c);
    }
    return null;
  }

  symbols(): string[] {
    const s = new Set<string>();
    for (const t of this.terms.values()) for (const k of Object.keys(t.m)) s.add(k);
    return [...s].sort();
  }

  /** True when this is exactly a single named symbol with coefficient 1. */
  asAtom(): string | null {
    if (this.terms.size !== 1) return null;
    const [t] = [...this.terms.values()];
    if (!ratIsOne(t.c)) return null;
    const names = Object.keys(t.m);
    if (names.length !== 1 || t.m[names[0]] !== 1) return null;
    return names[0];
  }

  /** Substitute a subset of symbols with numbers, leaving the rest symbolic. */
  subst(values: Readonly<Record<string, number>>): Sym {
    let acc = Sym.ZERO;
    for (const t of this.terms.values()) {
      let term = Sym.con(t.c);
      for (const k of Object.keys(t.m)) {
        const e = t.m[k];
        const base = k in values ? Sym.con(values[k]) : Sym.v(k);
        term = term.mul(base.pow(e));
      }
      acc = acc.add(term);
    }
    return acc;
  }

  /** Fully evaluate; returns null when some symbol has no value. */
  toNumber(values: Readonly<Record<string, number>>): number | null {
    return this.subst(values).asConst();
  }

  /** Structural equality of the canonical form (no substitution). */
  equals(other: Sym): boolean {
    return this.sub(other).isZero();
  }

  /**
   * Equality under a partial environment. Used for shape compatibility: design
   * symbols are substituted, runtime symbols stay indeterminate, and the
   * difference must be the zero polynomial.
   */
  equalsUnder(other: Sym, values: Readonly<Record<string, number>>): boolean {
    return this.sub(other).subst(values).isZero();
  }

  toString(): string {
    if (this.terms.size === 0) return "0";
    const list = [...this.terms.values()].sort((a, b) => {
      const da = monoDegree(a.m);
      const db = monoDegree(b.m);
      if (da !== db) return db - da;
      return monoKey(a.m).localeCompare(monoKey(b.m));
    });
    let out = "";
    for (let i = 0; i < list.length; i++) {
      const t = list[i];
      const negative = t.c.n < 0;
      const abs: Rat = negative ? ratNeg(t.c) : t.c;
      if (i === 0) out += negative ? "-" : "";
      else out += negative ? " - " : " + ";
      const key = monoKey(t.m);
      if (key === "") {
        out += ratToString(abs);
      } else if (ratIsOne(abs)) {
        out += key;
      } else {
        out += `${ratToString(abs)}*${key}`;
      }
    }
    return out;
  }
}

/** Convenience: a constant-zero check that tolerates null. */
export function symEq(a: Sym | null, b: Sym | null, values: Readonly<Record<string, number>>): boolean {
  if (!a || !b) return false;
  return a.equalsUnder(b, values);
}
