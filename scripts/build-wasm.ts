/**
 * Compile the engine for the browser.
 *
 *   bun run scripts/build-wasm.ts
 *
 * The Go toolchain has to be on the path. The output is a build artifact and is
 * not checked in: it is seven megabytes, it changes with every engine change,
 * and it is reproducible from source in a few seconds.
 *
 * `wasm_exec.js` is vendored rather than copied here on every build, because it
 * belongs to whichever Go release built the module and the two have to match.
 * When the toolchain moves, this says so.
 */

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { join } from "node:path";

const repo = join(import.meta.dir, "..");
const engineDir = join(repo, "packages", "core-go");
const outDir = join(repo, "packages", "engine", "wasm");
const vendored = join(repo, "packages", "engine", "vendor", "wasm_exec.js");

function run(command: string, args: string[], cwd: string, env?: Record<string, string>): string {
  const result = spawnSync(command, args, { cwd, encoding: "utf8", env: { ...process.env, ...env } });
  if (result.error) throw new Error(`${command}: ${result.error.message}`);
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed:\n${result.stderr || result.stdout}`);
  }
  return result.stdout.trim();
}

const goroot = run("go", ["env", "GOROOT"], repo);
const version = run("go", ["version"], repo);

mkdirSync(outDir, { recursive: true });
const target = join(outDir, "tensorcad.wasm");

// -s -w drop the symbol table and DWARF. They are worth about 100 KB here, and
// a stack trace out of the engine is a Go panic that the wrapper turns into a
// message anyway.
run(
  "go",
  ["build", "-ldflags=-s -w", "-trimpath", "-o", target, "./cmd/wasm"],
  engineDir,
  { GOOS: "js", GOARCH: "wasm" },
);

// The loader and the module come from the same toolchain or neither works. Go
// changes the interface between them without ceremony, because it considers
// them one thing.
const fresh = readFileSync(join(goroot, "lib", "wasm", "wasm_exec.js"), "utf8");
if (!existsSync(vendored) || readFileSync(vendored, "utf8") !== fresh) {
  writeFileSync(vendored, fresh);
  console.log(`updated vendor/wasm_exec.js from ${version}`);
}

const bytes = statSync(target).size;
const compressed = gzipSync(readFileSync(target), { level: 9 }).length;
console.log(
  `wrote packages/engine/wasm/tensorcad.wasm — ${(bytes / 1e6).toFixed(1)} MB, ` +
    `${(compressed / 1e6).toFixed(2)} MB gzipped (${version})`,
);
