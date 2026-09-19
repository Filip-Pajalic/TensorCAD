/**
 * Terminal output helpers: colour that turns itself off when nobody is looking,
 * and a couple of layout primitives the reports share.
 */

const FORCED = process.env.FORCE_COLOR;
const enabled =
  FORCED !== undefined && FORCED !== "0"
    ? true
    : process.env.NO_COLOR !== undefined || FORCED === "0"
      ? false
      : Boolean(process.stdout.isTTY);

function wrap(open: number, close: number) {
  return (s: string): string => (enabled ? `[${open}m${s}[${close}m` : s);
}

export const colorEnabled = enabled;

export const bold = wrap(1, 22);
export const dim = wrap(2, 22);
export const red = wrap(31, 39);
export const green = wrap(32, 39);
export const yellow = wrap(33, 39);
export const blue = wrap(34, 39);
export const magenta = wrap(35, 39);
export const cyan = wrap(36, 39);

/** Visible width, ignoring any escape sequences we may have added. */
export function width(s: string): number {
  // eslint-disable-next-line no-control-regex
  return s.replace(/\[[0-9;]*m/g, "").length;
}

export function pad(s: string, to: number): string {
  const w = width(s);
  return w >= to ? s : s + " ".repeat(to - w);
}

export function padStart(s: string, to: number): string {
  const w = width(s);
  return w >= to ? s : " ".repeat(to - w) + s;
}

export function heading(title: string): string {
  return `\n${bold(title)}`;
}

/** A two-column `label  value` block, right-aligning the values. */
export function rows(pairs: [string, string, string?][], indent = "  "): string {
  const labelWidth = Math.max(0, ...pairs.map((p) => width(p[0])));
  const valueWidth = Math.max(0, ...pairs.map((p) => width(p[1])));
  return pairs
    .map(([label, value, note]) => {
      const line = `${indent}${pad(label, labelWidth)}  ${padStart(value, valueWidth)}`;
      return note ? `${line}  ${dim(note)}` : line;
    })
    .join("\n");
}

export function percent(n: number, digits = 1): string {
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(digits)}%`;
}

/** `formatX` helpers in core assume finite input; guard the edges. */
export function finite(n: number, fmt: (n: number) => string): string {
  return Number.isFinite(n) ? fmt(n) : "-";
}

export function writeOut(s: string): void {
  process.stdout.write(s.endsWith("\n") ? s : `${s}\n`);
}

export function writeErr(s: string): void {
  process.stderr.write(s.endsWith("\n") ? s : `${s}\n`);
}
