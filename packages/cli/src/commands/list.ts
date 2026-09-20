/**
 * `tensorcad list` — what you can pass to the other commands.
 */

import { bold, dim, heading, pad, writeOut } from "../format.js";
import { bool, type Args } from "../args.js";
import { formatBytes, formatCount, formatFlops } from "@tensor-cad/engine";
import { HARDWARE, PRESET_NAMES, getPreset, peakFlops } from "@tensor-cad/engine/node";

export function cmdList(args: Args): number {
  if (bool(args, "json")) {
    writeOut(
      JSON.stringify(
        {
          presets: PRESET_NAMES.map((name) => {
            const doc = getPreset(name);
            return {
              name,
              family: doc.meta.family ?? null,
              published_params: doc.meta.published?.params ?? null,
              notes: doc.meta.notes ?? null,
            };
          }),
          hardware: HARDWARE.map((h) => ({
            id: h.id,
            name: h.name,
            memory_bytes: h.memory,
            bandwidth_bytes_per_second: h.bandwidth,
            price_per_hour: h.pricePerHour,
            mfu_hint: h.mfuHint,
          })),
        },
        null,
        2,
      ),
    );
    return 0;
  }

  const out: string[] = [];

  out.push(bold(`Presets (${PRESET_NAMES.length})`));
  const nameWidth = Math.max(...PRESET_NAMES.map((n) => n.length));
  for (const name of PRESET_NAMES) {
    const doc = getPreset(name);
    const published = doc.meta.published?.params;
    const size = published ? formatCount(published) : "";
    out.push(`  ${pad(name, nameWidth)}  ${pad(size, 8)}${dim(doc.meta.family ?? "")}`);
  }

  out.push(heading(`Hardware (${HARDWARE.length})`));
  const idWidth = Math.max(...HARDWARE.map((h) => h.id.length));
  const hwNameWidth = Math.max(...HARDWARE.map((h) => h.name.length));
  for (const h of HARDWARE) {
    out.push(
      `  ${pad(h.id, idWidth)}  ${pad(h.name, hwNameWidth)}  ${pad(formatBytes(h.memory), 10)}` +
        `  ${pad(`${formatBytes(h.bandwidth)}/s`, 13)}  ${pad(`${formatFlops(peakFlops(h, "bf16"))}/s`, 14)}` +
        `  ${pad(`$${h.pricePerHour.toFixed(2)}/h`, 9)}` +
        `  ${dim(`MFU ${(h.mfuHint[0] * 100).toFixed(0)}-${(h.mfuHint[1] * 100).toFixed(0)}%`)}`,
    );
  }

  writeOut(out.join("\n"));
  return 0;
}
