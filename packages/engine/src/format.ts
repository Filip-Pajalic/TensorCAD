/**
 * Display helpers.
 *
 * TypeScript rather than a call into the engine: these run on every repaint,
 * beside a number the panel already has, and a boundary crossing to turn 4096
 * into "4.1K" would be the slowest thing in the frame. The Go engine has the
 * same set for the command line, and a test pins the two against each other.
 */

/** A parameter count as a person reads it: 8.03B, 124.4M, 12.9K. */
export function formatCount(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

/** A FLOP count: 312.00 TFLOP, 1.20 PFLOP. */
export function formatFlops(n: number): string {
  const units: [number, string][] = [
    [1e18, "EFLOP"],
    [1e15, "PFLOP"],
    [1e12, "TFLOP"],
    [1e9, "GFLOP"],
    [1e6, "MFLOP"],
    [1e3, "kFLOP"],
  ];
  for (const [scale, unit] of units) {
    if (Math.abs(n) >= scale) return `${(n / scale).toFixed(2)} ${unit}`;
  }
  return `${n.toFixed(0)} FLOP`;
}

/** A byte count, in the binary units the hardware is sold in. */
export function formatBytes(n: number): string {
  const units: [number, string][] = [
    [1024 ** 5, "PiB"],
    [1024 ** 4, "TiB"],
    [1024 ** 3, "GiB"],
    [1024 ** 2, "MiB"],
    [1024, "KiB"],
  ];
  for (const [scale, unit] of units) {
    if (Math.abs(n) >= scale) return `${(n / scale).toFixed(2)} ${unit}`;
  }
  return `${Math.round(n)} B`;
}

/** A duration: 45.0 min, 12.5 h, 3.2 days. */
export function formatHours(h: number): string {
  if (h < 1) return `${(h * 60).toFixed(1)} min`;
  if (h < 48) return `${h.toFixed(1)} h`;
  return `${(h / 24).toFixed(1)} days`;
}

/** A price: $1.20M, $4.5k, $12.34. */
export function formatDollars(d: number): string {
  if (d >= 1e6) return `$${(d / 1e6).toFixed(2)}M`;
  if (d >= 1e3) return `$${(d / 1e3).toFixed(1)}k`;
  return `$${d.toFixed(2)}`;
}
