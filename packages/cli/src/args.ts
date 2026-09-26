/**
 * A hand-rolled argument parser.
 *
 * Small enough that a dependency would cost more than it saves: we only need
 * `--flag`, `--flag value`, `--flag=value` and positionals.
 */

export interface Args {
  /** Positional arguments in order. */
  _: string[];
  flags: Record<string, string | boolean>;
}

/** Flags that always take a value, so `--out dir` does not swallow `dir` as a positional. */
const VALUE_FLAGS = new Set([
  "T",
  "B",
  "S",
  "hardware",
  "gpus",
  "tokens",
  "optimizer",
  "recompute",
  "precision",
  "zero",
  "tp",
  "dp",
  "pp",
  "ep",
  "dtype",
  "mfu",
  "concurrency",
  "pack",
  "pack-spread",
  "out",
  "class-name",
  "name",
]);

export function parseArgs(argv: string[]): Args {
  const out: Args = { _: [], flags: {} };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--") {
      out._.push(...argv.slice(i + 1));
      break;
    }
    if (!a.startsWith("-")) {
      out._.push(a);
      continue;
    }

    const name = a.replace(/^--?/, "");
    const eq = name.indexOf("=");
    if (eq >= 0) {
      out.flags[name.slice(0, eq)] = name.slice(eq + 1);
      continue;
    }

    const next = argv[i + 1];
    const wantsValue = VALUE_FLAGS.has(name) || (next !== undefined && !next.startsWith("-") && !isBoolish(name));
    if (wantsValue && next !== undefined && !next.startsWith("--")) {
      out.flags[name] = next;
      i++;
    } else {
      out.flags[name] = true;
    }
  }

  return out;
}

const BOOL_FLAGS = new Set(["json", "help", "h", "version", "no-color", "quiet", "smoke-test"]);

function isBoolish(name: string): boolean {
  return BOOL_FLAGS.has(name);
}

export function num(args: Args, name: string): number | undefined {
  const v = args.flags[name];
  if (v === undefined || v === true) return undefined;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new UsageError(`--${name} expects a number, got "${v}"`);
  return n;
}

export function str(args: Args, name: string): string | undefined {
  const v = args.flags[name];
  return typeof v === "string" ? v : undefined;
}

export function bool(args: Args, name: string): boolean {
  return args.flags[name] === true || args.flags[name] === "true";
}

/** A problem with what the user typed, as opposed to a problem with the design. */
export class UsageError extends Error {}
