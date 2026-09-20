#!/usr/bin/env bun
/**
 * `tensorcad` — a thin command line over the analysis engine.
 *
 * Every command takes a `<file|preset>`: a preset name if it matches one, a
 * path to a `.tensorcad.json` document otherwise.
 */

import { parseArgs, UsageError, type Args } from "./args.js";
import { bold, dim, red, writeErr, writeOut } from "./format.js";
import { cmdList } from "./commands/list.js";
import { cmdValidate } from "./commands/validate.js";
import { cmdAnalyze } from "./commands/analyze.js";
import { cmdCodegen } from "./commands/codegen.js";
import { cmdDiff } from "./commands/diff.js";
import { cmdPlan } from "./commands/plan.js";
import { cmdMup } from "./commands/mup.js";
import { cmdShow } from "./commands/show.js";
import { cmdImport } from "./commands/import.js";
import { PRESET_NAMES, loadEngine } from "@tensor-cad/engine/node";

const COMMANDS: Record<string, { run: (args: Args) => number; usage: string; blurb: string }> = {
  list: { run: cmdList, usage: "list [--json]", blurb: "presets and hardware profiles" },
  validate: {
    run: cmdValidate,
    usage: "validate <file|preset> [--T n] [--hardware id] [--json]",
    blurb: "run the design rules; exits 1 on any error",
  },
  analyze: {
    run: cmdAnalyze,
    usage:
      "analyze <file|preset> [--T n] [--B n] [--hardware id] [--gpus n] [--tokens n]\n" +
      "                       [--optimizer k] [--recompute none|selective|full] [--zero 0..3]\n" +
      "                       [--tp n] [--dp n] [--dtype bf16] [--concurrency n] [--json]",
    blurb: "parameters, FLOPs, KV cache, memory, throughput, cost, Chinchilla",
  },
  codegen: {
    run: cmdCodegen,
    usage: "codegen <file|preset> [--out dir] [--class-name Name] [--smoke-test] [--json]",
    blurb: "write the PyTorch module and config",
  },
  plan: {
    run: cmdPlan,
    usage:
      "plan <file|preset> --gpus n [--gpus-per-node n] [--headroom f] [--T n]\n" +
      "                    [--micro-batch 1,2,4] [--recompute none|selective|full]\n" +
      "                    [--hardware id] [--limit n] [--json]",
    blurb: "ways to split training across a cluster; exits 1 if none fit",
  },
  mup: {
    run: cmdMup,
    usage: "mup <file|preset> [--widths 256,512,1024] [--base-width n] [--json]",
    blurb: "the same design at several widths, and what to scale by at each",
  },
  diff: { run: cmdDiff, usage: "diff <a> <b> [--json]", blurb: "structural and numeric difference" },
  show: { run: cmdShow, usage: "show <file|preset> [--json]", blurb: "block tree with inferred shapes" },
  import: {
    run: cmdImport,
    usage: "import <config.json> [--name name] [--out file] [--json]",
    blurb: "turn a Hugging Face config.json into a design",
  },
};

function help(): string {
  const names = Object.keys(COMMANDS);
  const width = Math.max(...names.map((n) => n.length));
  const lines = [
    bold("tensorcad"),
    dim("  Node-based CAD for LLM architectures."),
    "",
    bold("Usage"),
    "  bun run packages/cli/src/index.ts <command> [args]",
    "",
    bold("Commands"),
    ...names.map((n) => `  ${n.padEnd(width)}  ${dim(COMMANDS[n].blurb)}`),
    "",
    bold("Details"),
    ...names.map((n) => `  ${COMMANDS[n].usage}`),
    "",
    bold("Presets"),
    `  ${PRESET_NAMES.join(", ")}`,
  ];
  return lines.join("\n");
}

export function run(argv: string[]): number {
  const args = parseArgs(argv);
  const name = args._.shift();

  if (!name || name === "help" || args.flags.help === true || args.flags.h === true) {
    writeOut(help());
    return name && name !== "help" ? 1 : 0;
  }

  const command = COMMANDS[name];
  if (!command) {
    writeErr(`${red("Unknown command")} "${name}".\n`);
    writeErr(help());
    return 1;
  }

  try {
    return command.run(args);
  } catch (e) {
    if (e instanceof UsageError) {
      writeErr(`${red("error")} ${e.message}\n`);
      writeErr(dim(`usage: ${command.usage}`));
      return 2;
    }
    writeErr(`${red("error")} ${(e as Error).message}`);
    if (process.env.TENSORCAD_DEBUG) writeErr(String((e as Error).stack));
    return 2;
  }
}

// `import.meta.main` is true under bun when this file is the entry point.
if (import.meta.main) {
  // The engine is WebAssembly and loading it is the one asynchronous thing in
  // the whole command line. Everything past here is a function of a document.
  try {
    await loadEngine();
    process.exitCode = run(process.argv.slice(2));
  } catch (e) {
    process.stderr.write(`${(e as Error).message}
`);
    process.exitCode = 2;
  }
}
