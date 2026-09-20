#!/usr/bin/env bun
/**
 * Build the publishable form of `@tensorcad/engine` and `@tensorcad/mcp`.
 *
 * Inside this repository both packages are consumed as TypeScript, because Bun
 * runs it. Neither would work for anybody else: `main` points at a `.ts` file
 * that Node cannot import, the MCP server's `bin` starts with a Bun shebang,
 * and its dependency on the engine is written `workspace:*`, which npm does not
 * understand. Publishing them as they sit would produce two packages that
 * install and then fail on the first import.
 *
 * So this emits what a consumer actually needs — JavaScript, type declarations
 * and a manifest with real version ranges — into each package's `dist`, and
 * leaves the source alone. `package.json` keeps pointing at `src` for this
 * repository's own use; the published manifest is written here and points at
 * `dist`, so there is one place where the difference lives.
 *
 *   bun run scripts/build-dist.ts            # build
 *   bun run scripts/build-dist.ts --check    # build, then import it under Node
 */

import { mkdir, rm, writeFile, cp } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { $ } from "bun";

const ROOT = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const check = process.argv.includes("--check");

/** The version every published package and the registry manifest carries. */
const version = (await Bun.file(join(ROOT, "package.json")).json()).version as string;

type Pkg = {
  dir: string;
  entries: Record<string, string>;
  /** Extra files copied verbatim, relative to the package. */
  assets?: string[];
  /** Turned into `dependencies`, with `workspace:*` resolved to this version. */
  bin?: Record<string, string>;
};

const PACKAGES: Pkg[] = [
  {
    dir: "packages/engine",
    entries: { "index.js": "src/index.ts", "node.js": "src/node.ts" },
    // The loader and the module travel with it: the engine is the WebAssembly
    // file, and a client that cannot find it is a client that does nothing.
    assets: ["vendor/wasm_exec.js", "wasm/tensorcad.wasm", "README.md"],
  },
  {
    dir: "packages/mcp",
    entries: { "index.js": "src/index.ts", "server.js": "src/serve.ts", "stdio.js": "src/stdio.ts" },
    assets: ["README.md", "mcp.example.json", "server.json"],
    bin: { "tensorcad-mcp": "./dist/stdio.js" },
  },
];

async function build(pkg: Pkg): Promise<void> {
  const dir = join(ROOT, pkg.dir);
  const dist = join(dir, "dist");
  await rm(dist, { recursive: true, force: true });
  await mkdir(dist, { recursive: true });

  const manifest = await Bun.file(join(dir, "package.json")).json();

  // Bundled rather than transpiled file by file, because the entry points are
  // small and the alternative is publishing a directory of relative imports
  // whose extensions have to be rewritten. Dependencies stay external: a
  // published package that inlined `zod` would ship it twice.
  const result = await Bun.build({
    entrypoints: Object.values(pkg.entries).map((p) => join(dir, p)),
    outdir: dist,
    target: "node",
    format: "esm",
    external: Object.keys(manifest.dependencies ?? {}),
    naming: "[name].js",
  });
  if (!result.success) {
    for (const log of result.logs) console.error(log);
    throw new Error(`${pkg.dir}: bundle failed`);
  }

  // Types, from the same source. `tsc` is the only thing that can write these,
  // and a package without them is one every TypeScript consumer has to `any`.
  //
  // Through a config of its own rather than the package's: that one includes
  // the tests, which are not part of what is published and which `rootDir`
  // would reject for sitting outside `src`.
  const typesConfig = join(dir, "tsconfig.dist.json");
  await writeFile(
    typesConfig,
    JSON.stringify(
      {
        extends: "./tsconfig.json",
        compilerOptions: {
          noEmit: false,
          declaration: true,
          emitDeclarationOnly: true,
          outDir: "dist",
          rootDir: "src",
        },
        include: ["src/**/*.ts"],
      },
      null,
      2,
    ) + "\n",
  );
  try {
    await $`bun x tsc -p ${typesConfig}`.quiet();
  } finally {
    await rm(typesConfig, { force: true });
  }

  for (const asset of pkg.assets ?? []) {
    const to = join(dist, asset.includes("/") ? asset.slice(asset.lastIndexOf("/") + 1) : asset);
    await cp(join(dir, asset), to).catch(() => {
      throw new Error(`${pkg.dir}: missing ${asset}. Run \`bun run build:wasm\` first.`);
    });
  }

  // The published manifest: the same package, pointed at what was just built.
  const exports: Record<string, unknown> = {};
  for (const [out, src] of Object.entries(pkg.entries)) {
    const name = out === "index.js" ? "." : `./${out.replace(/\.js$/, "")}`;
    exports[name] = { types: `./${out.replace(/\.js$/, ".d.ts")}`, import: `./${out}` };
    void src;
  }
  if (pkg.dir.endsWith("engine")) {
    exports["./wasm_exec"] = "./wasm_exec.js";
    exports["./tensorcad.wasm"] = "./tensorcad.wasm";
  }

  const deps: Record<string, string> = {};
  for (const [name, range] of Object.entries(manifest.dependencies ?? {})) {
    // `workspace:*` is a Bun and pnpm spelling that npm rejects. Inside one
    // release every package is the same version, so that is what it becomes.
    deps[name] = String(range).startsWith("workspace:") ? version : String(range);
  }

  const published = {
    name: manifest.name,
    version,
    description: manifest.description,
    ...(manifest.mcpName ? { mcpName: manifest.mcpName } : {}),
    type: "module",
    main: "./index.js",
    types: "./index.d.ts",
    exports,
    ...(pkg.bin ? { bin: Object.fromEntries(Object.entries(pkg.bin).map(([k, v]) => [k, v.replace("./dist/", "./")])) } : {}),
    ...(Object.keys(deps).length > 0 ? { dependencies: deps } : {}),
    license: manifest.license,
    repository: { type: "git", url: "git+https://github.com/Filip-Pajalic/TensorCAD.git" },
    ...(manifest.sideEffects ? { sideEffects: ["./wasm_exec.js"] } : {}),
  };
  await writeFile(join(dist, "package.json"), JSON.stringify(published, null, 2) + "\n");

  // `node.ts` imports the Go runtime for its side effect, and a bundler drops
  // a side-effect import that exports nothing — the same tree-shake that broke
  // the desktop bundle. Concatenated rather than imported, so there is nothing
  // left to shake.
  const runtime = join(dist, "wasm_exec.js");
  if (await Bun.file(runtime).exists()) {
    for (const entry of ["node.js"]) {
      const file = join(dist, entry);
      if (!(await Bun.file(file).exists())) continue;
      await writeFile(file, `${await Bun.file(runtime).text()}
${await Bun.file(file).text()}`);
    }
  }

  // The MCP server is a program, and a program that npm installed has to start
  // under whatever node the user has.
  if (pkg.bin) {
    const stdio = join(dist, "stdio.js");
    const text = await Bun.file(stdio).text();
    await writeFile(stdio, `#!/usr/bin/env node\n${text.replace(/^#![^\n]*\n/, "")}`);
  }

  const bytes = (
    await Promise.all(
      [...Object.keys(pkg.entries), ...(pkg.assets ?? []).map((a) => a.split("/").pop()!)].map(
        async (f) => Bun.file(join(dist, f)).size,
      ),
    )
  ).reduce((a, b) => a + b, 0);
  console.log(`${manifest.name}@${version} -> ${pkg.dir}/dist (${(bytes / 1e6).toFixed(2)} MB)`);
}

for (const pkg of PACKAGES) await build(pkg);

// The registry manifest carries the version too, and a mismatch is what the
// registry rejects the submission for.
const serverPath = join(ROOT, "packages/mcp/server.json");
const server = await Bun.file(serverPath).json();
if (server.version !== version) {
  server.version = version;
  for (const p of server.packages ?? []) if (p.version !== undefined) p.version = version;
  await writeFile(serverPath, JSON.stringify(server, null, 2) + "\n");
  console.log(`server.json -> ${version}`);
}

if (!check) process.exit(0);

// Packed and installed, not imported in place.
//
// In this repository `@tensorcad/engine` resolves through the workspace to the
// TypeScript source, so importing the built server here would test the source
// rather than what was built. `npm pack` produces the exact tarball a publish
// would upload, and installing it into an empty directory is the only way to
// find out whether the thing being published works.
const tmp = join(ROOT, "dist", "verify");
await rm(tmp, { recursive: true, force: true });
await mkdir(tmp, { recursive: true });

console.log("\npacking:");
const tarballs: string[] = [];
for (const pkg of PACKAGES) {
  const out = await $`npm pack --pack-destination ${tmp}`.cwd(join(ROOT, pkg.dir, "dist")).text();
  const name = out.trim().split("\n").pop()!.trim();
  tarballs.push(join(tmp, name));
  console.log(`  ${name}`);
}

console.log("\ninstalling into an empty directory:");
await writeFile(
  join(tmp, "package.json"),
  JSON.stringify({ name: "verify", private: true, type: "module" }, null, 2),
);
await $`npm install --no-audit --no-fund ${tarballs}`.cwd(tmp).quiet();

// Started under Node, not Bun. The difference between the two has broken the
// desktop bundle twice, and it is the whole reason this build exists: a package
// that only works under Bun is one nobody who installs it can use.
console.log("\nimporting under node:");
await writeFile(
  join(tmp, "check.mjs"),
  [
    `import { loadEngine, analyze, getPreset, mupLadder } from "@tensorcad/engine/node";`,
    `import * as mcp from "@tensorcad/mcp";`,
    ``,
    `await loadEngine();`,
    `const a = analyze(getPreset("gpt2-small"));`,
    `if (a.params.total !== 124439808) throw new Error("wrong parameter count: " + a.params.total);`,
    `console.log("  engine: gpt2-small is", a.params.total.toLocaleString(), "parameters");`,
    ``,
    `const l = mupLadder(getPreset("llama-3-8b"));`,
    `console.log("  engine: its ladder is", l.rungs.map((r) => r.width).join(", "));`,
    ``,
    `if (Object.keys(mcp).length === 0) throw new Error("the MCP library exports nothing");`,
    `console.log("  mcp: exports", Object.keys(mcp).join(", "));`,
  ].join("\n"),
);
const ran = Bun.spawnSync(["node", "check.mjs"], { cwd: tmp, stdout: "inherit", stderr: "inherit" });
if (ran.exitCode !== 0) throw new Error("the published packages do not work under node");

// And the command the server registers has to start. A library that imports is
// not the same as a program that runs, and the program is what a user installs
// this for.
const bin = join(
  tmp,
  "node_modules",
  ".bin",
  process.platform === "win32" ? "tensorcad-mcp.cmd" : "tensorcad-mcp",
);
if (await Bun.file(bin).exists()) {
  const server = Bun.spawn([bin], { cwd: tmp, stdin: "pipe", stdout: "pipe", stderr: "pipe" });
  const hello = JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-06-18",
      capabilities: {},
      clientInfo: { name: "verify", version: "0" },
    },
  });
  server.stdin.write(hello + "\n");
  // Read until it says who it is, rather than to the end of the stream: a
  // server that answered correctly never closes its output.
  const answered = await Promise.race([
    (async () => {
      let text = "";
      for await (const chunk of server.stdout as ReadableStream<Uint8Array>) {
        text += new TextDecoder().decode(chunk);
        if (text.includes('"serverInfo"')) return true;
      }
      return false;
    })(),
    new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 30_000)),
  ]);
  server.kill();
  if (!answered) throw new Error("tensorcad-mcp did not answer initialize");
  console.log("  mcp: tensorcad-mcp answered initialize");
}

console.log("\nBoth packages install and run under node. Nothing was published.");
