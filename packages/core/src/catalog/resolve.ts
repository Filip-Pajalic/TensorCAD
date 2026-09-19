/**
 * Parameter resolution: raw document values -> numbers, booleans and symbolic
 * forms, driven by the block's declared parameter specs.
 *
 * Type direction matters: without the spec we could not tell the enum `"silu"`
 * from the expression `"D"`.
 */

import type { ParamValue, Resolved, SymbolTable } from "../ir/types.js";
import type { BlockDef, ParamSpec, Ports, PortsSpec } from "./types.js";
import { Sym } from "../shapes/symexpr.js";
import { evalExpr } from "../shapes/expr.js";
import { symbolCtx } from "../ir/symbols.js";

function specOf(def: BlockDef, key: string): ParamSpec | undefined {
  return (def.params as Record<string, ParamSpec>)[key];
}

export function resolveNodeParams(
  def: BlockDef,
  raw: Record<string, ParamValue> | undefined,
  symbols: SymbolTable,
): Resolved {
  const ctx = symbolCtx(symbols);
  const out: Resolved = { type: def.type, p: {}, s: {}, raw: raw ?? {}, rawFull: {}, errors: [] };

  const keys = new Set<string>([...Object.keys(def.params ?? {}), ...Object.keys(raw ?? {})]);

  for (const key of keys) {
    const spec = specOf(def, key);
    const given = raw?.[key];

    if (!spec) {
      out.errors.push(`Unknown parameter "${key}" on block "${def.type}"`);
      out.p[key] = given;
      continue;
    }

    const value = given === undefined ? (spec as { default?: ParamValue }).default : given;
    if (value !== undefined) out.rawFull[key] = value;

    if (value === undefined) {
      if (spec.type === "bool") out.p[key] = false;
      else if (spec.type === "obj") out.p[key] = null;
      else out.errors.push(`Missing required parameter "${key}" on block "${def.type}"`);
      continue;
    }

    switch (spec.type) {
      case "int":
      case "num": {
        if (typeof value === "number") {
          out.p[key] = value;
          out.s[key] = Sym.con(value);
        } else if (typeof value === "string") {
          try {
            const sym = evalExpr(value, ctx);
            const n = sym.toNumber(ctx.values);
            if (n === null) {
              out.errors.push(`Parameter "${key}" does not evaluate to a number: ${sym.toString()}`);
            } else {
              out.p[key] = n;
              out.s[key] = sym;
            }
          } catch (e) {
            out.errors.push(`Parameter "${key}": ${(e as Error).message}`);
          }
        } else if (value === null) {
          out.p[key] = null;
        } else {
          out.errors.push(`Parameter "${key}" should be a number or expression`);
        }
        if (spec.type === "int" && typeof out.p[key] === "number" && !Number.isInteger(out.p[key])) {
          out.errors.push(
            `Parameter "${key}" must be an integer but evaluates to ${out.p[key]}` +
              (typeof value === "string" ? ` (from "${value}")` : ""),
          );
        }
        if (typeof out.p[key] === "number") {
          const s = spec as { min?: number; max?: number };
          if (s.min !== undefined && out.p[key] < s.min) {
            out.errors.push(`Parameter "${key}" must be >= ${s.min}, got ${out.p[key]}`);
          }
          if (s.max !== undefined && out.p[key] > s.max) {
            out.errors.push(`Parameter "${key}" must be <= ${s.max}, got ${out.p[key]}`);
          }
        }
        break;
      }
      case "bool": {
        // `null` is kept rather than coerced: some booleans are tri-state, such
        // as an output-projection bias that defaults to whatever the other
        // projections use. Consumers test with `=== true`.
        if (typeof value === "boolean") out.p[key] = value;
        else if (value === null) out.p[key] = null;
        else out.errors.push(`Parameter "${key}" should be a boolean`);
        break;
      }
      case "enum": {
        if (typeof value === "string" && spec.values.includes(value)) out.p[key] = value;
        else if (value === null) out.p[key] = null;
        else out.errors.push(`Parameter "${key}" should be one of ${spec.values.join(", ")}, got ${String(value)}`);
        break;
      }
      case "str":
      case "pattern": {
        if (typeof value === "string") out.p[key] = value;
        else if (value === null) out.p[key] = null;
        else out.errors.push(`Parameter "${key}" should be a string`);
        break;
      }
      case "obj": {
        out.p[key] = value;
        break;
      }
    }
  }

  return out;
}

export function portsOf(spec: PortsSpec, r: Resolved): Ports {
  return typeof spec === "function" ? spec(r) : spec;
}
