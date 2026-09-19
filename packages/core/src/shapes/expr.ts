/**
 * Expression parser for parameter values and shape-pattern atoms.
 *
 * Grammar:
 *   expr    := term (('+' | '-') term)*
 *   term    := unary (('*' | '/') unary)*
 *   unary   := '-' unary | power
 *   power   := primary ('^' unary)?
 *   primary := NUMBER | IDENT | IDENT '(' args ')' | '(' expr ')'
 *
 * Symbol references evaluate to symbolic atoms so that a design keeps showing
 * `B T D` rather than `B T 4096`. Function calls are folded eagerly: their
 * arguments must be numerically determined, which is always true for design
 * parameters.
 */

import { Sym, rat } from "./symexpr.js";

type Tok =
  | { k: "num"; v: number; pos: number }
  | { k: "ident"; v: string; pos: number }
  | { k: "op"; v: string; pos: number }
  | { k: "eof"; pos: number };

function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === " " || c === "\t" || c === "\n") {
      i++;
      continue;
    }
    if (/[0-9]/.test(c) || (c === "." && /[0-9]/.test(src[i + 1] ?? ""))) {
      let j = i;
      while (j < src.length && /[0-9._]/.test(src[j])) j++;
      if (src[j] === "e" || src[j] === "E") {
        let k = j + 1;
        if (src[k] === "+" || src[k] === "-") k++;
        if (/[0-9]/.test(src[k] ?? "")) {
          j = k;
          while (j < src.length && /[0-9]/.test(src[j])) j++;
        }
      }
      const text = src.slice(i, j).replace(/_/g, "");
      const v = Number(text);
      if (!Number.isFinite(v)) throw new Error(`Bad number "${text}" at ${i}`);
      out.push({ k: "num", v, pos: i });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < src.length && /[A-Za-z0-9_]/.test(src[j])) j++;
      out.push({ k: "ident", v: src.slice(i, j), pos: i });
      i = j;
      continue;
    }
    if ("+-*/^(),".includes(c)) {
      out.push({ k: "op", v: c, pos: i });
      i++;
      continue;
    }
    throw new Error(`Unexpected character "${c}" at ${i} in "${src}"`);
  }
  out.push({ k: "eof", pos: src.length });
  return out;
}

export type Ast =
  | { k: "num"; v: number }
  | { k: "sym"; name: string }
  | { k: "call"; name: string; args: Ast[] }
  | { k: "bin"; op: "+" | "-" | "*" | "/" | "^"; l: Ast; r: Ast }
  | { k: "neg"; e: Ast };

class Parser {
  private i = 0;
  constructor(
    private readonly toks: Tok[],
    private readonly src: string,
  ) {}

  private peek(): Tok {
    return this.toks[this.i];
  }
  private eat(op: string): boolean {
    const t = this.peek();
    if (t.k === "op" && t.v === op) {
      this.i++;
      return true;
    }
    return false;
  }
  private expect(op: string): void {
    if (!this.eat(op)) throw new Error(`Expected "${op}" at ${this.peek().pos} in "${this.src}"`);
  }

  parse(): Ast {
    const e = this.expr();
    if (this.peek().k !== "eof") {
      throw new Error(`Trailing input at ${this.peek().pos} in "${this.src}"`);
    }
    return e;
  }

  private expr(): Ast {
    let l = this.term();
    for (;;) {
      if (this.eat("+")) l = { k: "bin", op: "+", l, r: this.term() };
      else if (this.eat("-")) l = { k: "bin", op: "-", l, r: this.term() };
      else return l;
    }
  }

  private term(): Ast {
    let l = this.unary();
    for (;;) {
      if (this.eat("*")) l = { k: "bin", op: "*", l, r: this.unary() };
      else if (this.eat("/")) l = { k: "bin", op: "/", l, r: this.unary() };
      else return l;
    }
  }

  private unary(): Ast {
    if (this.eat("-")) return { k: "neg", e: this.unary() };
    return this.power();
  }

  private power(): Ast {
    const base = this.primary();
    if (this.eat("^")) return { k: "bin", op: "^", l: base, r: this.unary() };
    return base;
  }

  private primary(): Ast {
    const t = this.peek();
    if (t.k === "num") {
      this.i++;
      return { k: "num", v: t.v };
    }
    if (t.k === "ident") {
      this.i++;
      if (this.eat("(")) {
        const args: Ast[] = [];
        if (!this.eat(")")) {
          for (;;) {
            args.push(this.expr());
            if (this.eat(",")) continue;
            this.expect(")");
            break;
          }
        }
        return { k: "call", name: t.v, args };
      }
      return { k: "sym", name: t.v };
    }
    if (t.k === "op" && t.v === "(") {
      this.i++;
      const e = this.expr();
      this.expect(")");
      return e;
    }
    throw new Error(`Unexpected token at ${t.pos} in "${this.src}"`);
  }
}

const astCache = new Map<string, Ast>();

export function parseExpr(src: string): Ast {
  const hit = astCache.get(src);
  if (hit) return hit;
  const ast = new Parser(tokenize(src), src).parse();
  astCache.set(src, ast);
  return ast;
}

/** Numeric functions available inside parameter expressions. */
export const EXPR_FUNCTIONS: Record<string, { arity: number; fn: (...a: number[]) => number; doc: string }> = {
  floor: { arity: 1, fn: (x) => Math.floor(x), doc: "floor(x)" },
  ceil: { arity: 1, fn: (x) => Math.ceil(x), doc: "ceil(x)" },
  round: { arity: 1, fn: (x) => Math.round(x), doc: "round(x) to nearest integer" },
  abs: { arity: 1, fn: (x) => Math.abs(x), doc: "abs(x)" },
  sqrt: { arity: 1, fn: (x) => Math.sqrt(x), doc: "sqrt(x)" },
  log2: { arity: 1, fn: (x) => Math.log2(x), doc: "log2(x)" },
  min: { arity: 2, fn: (a, b) => Math.min(a, b), doc: "min(a, b)" },
  max: { arity: 2, fn: (a, b) => Math.max(a, b), doc: "max(a, b)" },
  ceil_mult: { arity: 2, fn: (x, k) => Math.ceil(x / k) * k, doc: "ceil_mult(x, k): smallest multiple of k >= x" },
  floor_mult: { arity: 2, fn: (x, k) => Math.floor(x / k) * k, doc: "floor_mult(x, k): largest multiple of k <= x" },
  round_mult: { arity: 2, fn: (x, k) => Math.round(x / k) * k, doc: "round_mult(x, k): nearest multiple of k" },
};

export interface EvalCtx {
  /** Every known symbol with its numeric value (runtime symbols use defaults). */
  values: Readonly<Record<string, number>>;
  /** Symbols allowed to appear. Unknown symbols raise. */
  known: ReadonlySet<string>;
  /**
   * Names that expand to an expression rather than to an atom. Used so a port
   * pattern written over parameter names (`"... in_features"`) displays the
   * symbol the parameter was bound to (`D`) instead of the parameter name.
   */
  substitutions?: Readonly<Record<string, Sym>>;
}

export function evalAst(ast: Ast, ctx: EvalCtx): Sym {
  switch (ast.k) {
    case "num":
      return Sym.con(ast.v);
    case "sym": {
      const sub = ctx.substitutions?.[ast.name];
      if (sub) return sub;
      if (!ctx.known.has(ast.name)) {
        throw new Error(`Unknown symbol "${ast.name}"`);
      }
      return Sym.v(ast.name);
    }
    case "neg":
      return evalAst(ast.e, ctx).neg();
    case "call": {
      const f = EXPR_FUNCTIONS[ast.name];
      if (!f) throw new Error(`Unknown function "${ast.name}"`);
      if (ast.args.length !== f.arity) {
        throw new Error(`Function "${ast.name}" expects ${f.arity} argument(s), got ${ast.args.length}`);
      }
      const nums = ast.args.map((a) => {
        const s = evalAst(a, ctx);
        const n = s.toNumber(ctx.values);
        if (n === null) throw new Error(`Argument of "${ast.name}" is not numeric: ${s.toString()}`);
        return n;
      });
      return Sym.con(f.fn(...nums));
    }
    case "bin": {
      const l = evalAst(ast.l, ctx);
      const r = evalAst(ast.r, ctx);
      switch (ast.op) {
        case "+":
          return l.add(r);
        case "-":
          return l.sub(r);
        case "*":
          return l.mul(r);
        case "/": {
          const rc = r.asConst();
          if (rc !== null) {
            if (rc === 0) throw new Error("Division by zero");
            return l.mul(Sym.con(rat(1, rc)));
          }
          const q = l.divExact(r);
          if (q) return q;
          // Fall back to numeric division when both sides are determined.
          const ln = l.toNumber(ctx.values);
          const rn = r.toNumber(ctx.values);
          if (ln !== null && rn !== null && rn !== 0) return Sym.con(ln / rn);
          throw new Error(`Cannot divide ${l.toString()} by ${r.toString()} exactly`);
        }
        case "^": {
          const e = r.asConst();
          if (e === null || !Number.isInteger(e) || e < 0) {
            throw new Error(`Exponent must be a non-negative integer, got ${r.toString()}`);
          }
          return l.pow(e);
        }
      }
    }
  }
}

export function evalExpr(src: string, ctx: EvalCtx): Sym {
  return evalAst(parseExpr(src), ctx);
}

/** Symbols referenced by an expression (before evaluation). */
export function exprSymbols(src: string): string[] {
  const out = new Set<string>();
  const walk = (a: Ast): void => {
    switch (a.k) {
      case "sym":
        out.add(a.name);
        break;
      case "call":
        a.args.forEach(walk);
        break;
      case "bin":
        walk(a.l);
        walk(a.r);
        break;
      case "neg":
        walk(a.e);
        break;
      case "num":
        break;
    }
  };
  walk(parseExpr(src));
  return [...out];
}
