/**
 * Resolving a `<file|preset>` argument.
 *
 * A bare name that matches a preset wins; anything else is a path to a
 * `.tensorcad.json` document.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { UsageError } from "./args.js";
import type { Doc } from "@tensorcad/engine";
import { PRESET_NAMES, getPreset } from "@tensorcad/engine/node";

export interface LoadedDoc {
  doc: Doc;
  /** How the document was found, for error messages and report headers. */
  source: "preset" | "file";
  /** Preset name or absolute file path. */
  ref: string;
}

export function loadDesign(ref: string | undefined): LoadedDoc {
  if (!ref) {
    throw new UsageError(`Missing <file|preset>. Presets: ${PRESET_NAMES.join(", ")}`);
  }
  if (PRESET_NAMES.includes(ref)) {
    return { doc: getPreset(ref), source: "preset", ref };
  }

  const path = resolve(process.cwd(), ref);
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch {
    throw new UsageError(
      `"${ref}" is neither a preset nor a readable file.\nPresets: ${PRESET_NAMES.join(", ")}`,
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    throw new UsageError(`${path} is not valid JSON: ${(e as Error).message}`);
  }

  const doc = parsed as Doc;
  if (!doc || typeof doc !== "object" || !doc.graph || !doc.meta) {
    throw new UsageError(`${path} does not look like a design document (no meta/graph).`);
  }
  return { doc, source: "file", ref: path };
}
