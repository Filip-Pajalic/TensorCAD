/**
 * Build the MCPB bundle: the MCP server as one double-clickable file.
 *
 *   bun run scripts/build-mcpb.ts
 *
 * An MCPB is a zip with a `manifest.json` and a server inside it, and Claude
 * Desktop installs one in a click. The catch is that it runs the server with
 * *Node*, which ships with the desktop app, where this repository runs it with
 * Bun and TypeScript. So the bundle is a build rather than a copy: one bundled
 * `index.js`, the WebAssembly module beside it, and nothing else.
 *
 * The module is not embedded in the JavaScript. It is two megabytes that would
 * become three as base64, and `node.ts` already knows how to read it from disk
 * next to itself — which is the same path it takes in the repository.
 */

import { $ } from "bun";
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");
const OUT = join(ROOT, "dist", "mcpb");
const BUNDLE = join(ROOT, "dist", "tensorcad.mcpb");

interface Pkg {
  name: string;
  version: string;
  description: string;
  mcpName: string;
}
const pkg = JSON.parse(readFileSync(join(ROOT, "packages/mcp/package.json"), "utf8")) as Pkg;

// The engine first: the bundle carries whatever is in `wasm/`, and a stale one
// is the one way to ship an answer the source does not give.
await $`bun run ${join(ROOT, "scripts/build-wasm.ts")}`.quiet();

rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, "server"), { recursive: true });

/**
 * One file, for Node, with the shebang dropped.
 *
 * `--target=node` is what makes `import.meta.url` resolve to the output file
 * rather than to the source, which is how the engine finds its module.
 */
const built = await Bun.build({
    // `stdio.ts`, not `index.ts`: the program rather than the library.
  entrypoints: [join(ROOT, "packages/mcp/src/stdio.ts")],
  target: "node",
  format: "esm",
  minify: false,
  external: [],
});
if (!built.success) {
  for (const log of built.logs) console.error(log);
  throw new Error("the server did not build");
}
const [artifact] = built.outputs;

// Go's own loader, prepended rather than imported.
//
// `node.ts` imports it for its side effect — it assigns `globalThis.Go` and
// exports nothing — and a bundler with no reason to keep a module that exports
// nothing drops it. The result started, reached `createEngine`, and threw
// "the Go WebAssembly runtime is missing" from inside a file that had imported
// it. Concatenating is blunt and cannot be tree-shaken.
const loader = readFileSync(join(ROOT, "packages/engine/vendor/wasm_exec.js"), "utf8");
// `stdio.ts` opens with a shebang, which is legal on the first line of a file
// and a syntax error on the five-hundredth. Node is the thing running this, and
// it is being told to by the manifest, so the line has nothing left to do.
const server = (await artifact.text()).replace(/^#![^\n]*\n/, "");
writeFileSync(join(OUT, "server", "index.js"), `${loader}
${server}`, "utf8");

// Node reads a bare `.js` as CommonJS unless something nearby says otherwise,
// and this is an ES module. Naming it `.mjs` would work too; a package.json is
// what the manifest's `entry_point` and every other tool expect to still be
// called `index.js`.
writeFileSync(
  join(OUT, "server", "package.json"),
  `${JSON.stringify({ type: "module", private: true }, null, 2)}
`,
  "utf8",
);

// `node.ts` looks for the module at `../wasm/tensorcad.wasm` relative to
// itself, and itself is now `server/index.js`, so the module goes at the
// bundle root's `wasm/`.
mkdirSync(join(OUT, "wasm"), { recursive: true });
cpSync(join(ROOT, "packages/engine/wasm/tensorcad.wasm"), join(OUT, "wasm", "tensorcad.wasm"));
cpSync(join(ROOT, "LICENSE.md"), join(OUT, "LICENSE.md"));
cpSync(join(ROOT, "packages/mcp/README.md"), join(OUT, "README.md"));

const manifest = {
  manifest_version: "0.3",
  name: "tensorcad",
  display_name: "TensorCAD",
  version: pkg.version,
  description: pkg.description,
  author: { name: "TensorCAD", url: "https://github.com/Filip-Pajalic/TensorCAD" },
  homepage: "https://github.com/Filip-Pajalic/TensorCAD",
  documentation: "https://github.com/Filip-Pajalic/TensorCAD/tree/main/packages/mcp",
  repository: { type: "git", url: "https://github.com/Filip-Pajalic/TensorCAD" },
  license: "MIT",
  keywords: ["llm", "architecture", "pytorch", "analysis", "cad"],
  server: {
    type: "node",
    entry_point: "server/index.js",
    mcp_config: {
      command: "node",
      args: ["${__dirname}/server/index.js"],
      env: { TENSORCAD_ROOT: "${user_config.root}" },
    },
  },
  // Where the server looks for `.tensorcad.json` files and resolves relative
  // paths. Asked for rather than assumed, because the working directory of a
  // desktop app is not anywhere a person keeps their designs.
  user_config: {
    root: {
      type: "directory",
      title: "Designs folder",
      description: "Where TensorCAD looks for .tensorcad.json files and writes new ones.",
      required: false,
      default: "${HOME}",
    },
  },
  compatibility: { runtimes: { node: ">=20.0.0" } },
};
writeFileSync(join(OUT, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

// Zip it. `mcpb pack` would do this and validate the manifest, but it is
// another global install for what is a zip with a known layout; the check that
// matters is that the thing inside it starts, which `--check` does.
rmSync(BUNDLE, { force: true });
await $`bun x @anthropic-ai/mcpb pack ${OUT} ${BUNDLE}`.quiet().catch(async () => {
  // No mcpb CLI: fall back to the platform's zip. The format is a zip.
  const files = ["manifest.json", "server", "wasm", "LICENSE.md", "README.md"];
  await $`cd ${OUT} && tar -a -c -f ${BUNDLE} ${files}`.quiet();
});

const size = existsSync(BUNDLE) ? Bun.file(BUNDLE).size : 0;
console.log(
  `wrote ${BUNDLE} — ${(size / 1024 / 1024).toFixed(2)} MB\n` +
    `  manifest ${manifest.name} ${manifest.version}, entry ${manifest.server.entry_point}`,
);

/** Collect a stream until it contains `needle`, or the clock runs out. */
async function readUntil(
  stream: ReadableStream<Uint8Array>,
  needle: string,
  ms: number,
): Promise<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  const until = Date.now() + ms;
  let seen = "";
  while (Date.now() < until) {
    const { done, value } = await reader.read();
    if (done) break;
    seen += decoder.decode(value, { stream: true });
    if (seen.includes(needle)) break;
  }
  reader.releaseLock();
  return seen;
}

if (process.argv.includes("--check")) {
  // The only check worth making: does it start under Node, which is what
  // Claude Desktop will run it with, and does it answer the handshake.
  const proc = Bun.spawn(["node", join(OUT, "server", "index.js")], {
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, TENSORCAD_ROOT: OUT },
  });
  const hello = {
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-06-18",
      capabilities: {},
      clientInfo: { name: "mcpb-check", version: "0" },
    },
  };
  proc.stdin.write(`${JSON.stringify(hello)}\n`);
  await proc.stdin.flush();
  // Read until the answer arrives, not until the stream closes: a server that
  // has answered is a server that is still running, so waiting for the end of
  // its stdout waits forever and then reports that it said nothing.
  const reply = await Promise.race([readUntil(proc.stdout, '"serverInfo"', 30_000), Bun.sleep(30_000).then(() => "")]);
  proc.kill();
  if (!reply.includes('"serverInfo"')) {
    throw new Error(`the bundled server did not answer initialize:\n${reply.slice(0, 400)}`);
  }
  console.log("  starts under node and answers initialize");
}
