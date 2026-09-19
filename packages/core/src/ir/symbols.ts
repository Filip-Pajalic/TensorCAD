/**
 * Symbol table resolution.
 *
 * Symbols may depend on each other (`F = ceil_mult(1.3*8/3*D, 1024)`), so they
 * are evaluated in dependency order with cycle detection.
 */

import type { Doc, SymbolDef, SymbolTable } from "./types.js";
import { RUNTIME_SYMBOLS } from "./types.js";
import { evalExpr, exprSymbols } from "../shapes/expr.js";

interface Normalized {
  name: string;
  runtime: boolean;
  literal: number | null;
  expr: string | null;
  doc: string;
}

function normalize(name: string, def: SymbolDef): Normalized {
  if (typeof def === "number") return { name, runtime: false, literal: def, expr: null, doc: "" };
  if (typeof def === "string") return { name, runtime: false, literal: null, expr: def, doc: "" };
  if (def && typeof def === "object" && def.kind === "runtime") {
    return { name, runtime: true, literal: def.default, expr: null, doc: def.doc ?? "" };
  }
  if (def && typeof def === "object" && def.kind === "design") {
    const v = def.value;
    return {
      name,
      runtime: false,
      literal: typeof v === "number" ? v : null,
      expr: typeof v === "string" ? v : null,
      doc: def.doc ?? "",
    };
  }
  throw new Error(`Symbol "${name}" has an unsupported definition`);
}

export function resolveSymbols(doc: Doc): SymbolTable {
  const table: SymbolTable = {
    order: [],
    values: {},
    designValues: {},
    runtime: new Set(),
    docs: {},
    errors: [],
  };

  const defs = new Map<string, Normalized>();
  for (const [name, def] of Object.entries(doc.symbols ?? {})) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
      table.errors.push(`Symbol name "${name}" is not a valid identifier`);
      continue;
    }
    try {
      defs.set(name, normalize(name, def));
    } catch (e) {
      table.errors.push((e as Error).message);
    }
  }

  // Runtime symbols are always available, even if the document forgot them.
  for (const name of RUNTIME_SYMBOLS) {
    if (!defs.has(name)) {
      defs.set(name, { name, runtime: true, literal: name === "B" ? 1 : 2048, expr: null, doc: "" });
    }
  }

  const known = new Set(defs.keys());
  const state = new Map<string, "pending" | "done">();

  const visit = (name: string, stack: string[]): void => {
    const st = state.get(name);
    if (st === "done") return;
    if (st === "pending") {
      table.errors.push(`Symbol cycle: ${[...stack, name].join(" -> ")}`);
      table.values[name] = Number.NaN;
      state.set(name, "done");
      return;
    }
    const def = defs.get(name);
    if (!def) {
      table.errors.push(`Unknown symbol "${name}"`);
      return;
    }
    state.set(name, "pending");

    if (def.runtime) {
      table.runtime.add(name);
      table.values[name] = def.literal ?? 1;
    } else if (def.literal !== null) {
      table.values[name] = def.literal;
      table.designValues[name] = def.literal;
    } else if (def.expr !== null) {
      let deps: string[] = [];
      try {
        deps = exprSymbols(def.expr);
      } catch (e) {
        table.errors.push(`Symbol "${name}": ${(e as Error).message}`);
      }
      for (const d of deps) {
        if (!known.has(d)) {
          table.errors.push(`Symbol "${name}" references unknown symbol "${d}"`);
          continue;
        }
        visit(d, [...stack, name]);
      }
      try {
        const sym = evalExpr(def.expr, { values: table.values, known });
        const v = sym.toNumber(table.values);
        if (v === null) {
          table.errors.push(`Symbol "${name}" does not evaluate to a number (got ${sym.toString()})`);
          table.values[name] = Number.NaN;
        } else {
          table.values[name] = v;
          table.designValues[name] = v;
        }
      } catch (e) {
        table.errors.push(`Symbol "${name}": ${(e as Error).message}`);
        table.values[name] = Number.NaN;
      }
    }

    table.docs[name] = def.doc;
    table.order.push(name);
    state.set(name, "done");
  };

  for (const name of defs.keys()) visit(name, []);
  return table;
}

/** Evaluation context for parameter expressions in a document. */
export function symbolCtx(table: SymbolTable): { values: Record<string, number>; known: Set<string> } {
  return { values: table.values, known: new Set(Object.keys(table.values)) };
}
