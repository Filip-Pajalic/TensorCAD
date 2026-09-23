#!/usr/bin/env bun
/**
 * Cut a release, in the two halves a release actually has.
 *
 *   bun run release bump 0.1.5    # on a branch: move every version, prove it builds
 *   bun run release tag           # on main, after that has merged: tag and push
 *
 * Two commands because the version change goes through a pull request like any
 * other change, and the tag is cut from what `main` actually holds afterwards —
 * never from a branch, and never before the bump has landed.
 *
 * Both halves exist because of a release that went wrong twice, silently.
 *
 * The version lives in five places, and they disagree in different ways at
 * different points. `build:dist` writes the root version into the published
 * packages and `server.json`; the release workflow's gate compares those against
 * the tag; and a test compares `server.json` against `packages/mcp/package.json`.
 * Bump the root alone and the build passes and the test fails; bump the root and
 * the manifest and nothing fails until a package goes out under the wrong number.
 * `bump` moves all five, then runs the same build the gate does.
 *
 * And `git push --follow-tags` pushes only *annotated* tags. This repository's
 * are lightweight, so that command succeeds, pushes nothing and says nothing —
 * the release never starts and nothing tells you. `tag` pushes the ref by its
 * full name, which works whatever kind of tag it is, and then asks the remote
 * whether it arrived rather than trusting the push.
 */

import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { $ } from "bun";

const ROOT = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");

/** Every file that carries the release version, and how to read and set it there. */
const MANIFESTS = [
  "package.json",
  "packages/engine/package.json",
  "packages/mcp/package.json",
  "packages/ui/package.json",
] as const;
const REGISTRY = "packages/mcp/server.json";

const SEMVER = /^\d+\.\d+\.\d+$/;

function fail(message: string): never {
  console.error(`\n  ${message}\n`);
  process.exit(1);
}

async function readJson(rel: string): Promise<Record<string, any>> {
  return JSON.parse(await readFile(join(ROOT, rel), "utf8"));
}

/** Every version the repository carries, by where it was read from. */
async function versions(): Promise<Map<string, string>> {
  const out = new Map<string, string>();
  for (const rel of MANIFESTS) out.set(rel, String((await readJson(rel)).version));
  const server = await readJson(REGISTRY);
  out.set(REGISTRY, String(server.version));
  (server.packages ?? []).forEach((p: { version?: string }, i: number) => {
    if (p.version !== undefined) out.set(`${REGISTRY} packages[${i}]`, String(p.version));
  });
  return out;
}

/**
 * Replace the version in place rather than re-serialising the file.
 *
 * A JSON round trip would reorder nothing here, but it would re-indent and
 * re-wrap whatever the file's author chose, and a version bump that rewrites
 * forty lines of a manifest is a diff nobody can review.
 */
async function setVersion(rel: string, from: string, to: string): Promise<number> {
  const path = join(ROOT, rel);
  const text = await readFile(path, "utf8");
  const pattern = new RegExp(`("version"\\s*:\\s*")${from.replace(/\./g, "\\.")}(")`, "g");
  let count = 0;
  const next = text.replace(pattern, (_, a: string, b: string) => {
    count++;
    return `${a}${to}${b}`;
  });
  if (count === 0) fail(`${rel} does not carry version ${from}; refusing to guess what to change.`);
  await writeFile(path, next);
  return count;
}

async function bump(to: string | undefined): Promise<void> {
  if (!to || !SEMVER.test(to)) fail("Usage: bun run release bump <major.minor.patch>, e.g. 0.1.5");

  const now = await versions();
  const distinct = new Set(now.values());
  if (distinct.size !== 1) {
    // Moving five numbers that already disagree would hide the disagreement
    // inside the bump. Say what is out of step and let it be fixed first.
    const lines = [...now].map(([where, v]) => `    ${v.padEnd(10)} ${where}`).join("\n");
    fail(`The versions already disagree, so there is no single one to move from:\n\n${lines}`);
  }
  const from = [...distinct][0]!;
  if (from === to) fail(`Already at ${to}.`);

  const tags = (await $`git tag --list v${to}`.cwd(ROOT).text()).trim();
  if (tags) fail(`v${to} is already a tag here. Pick the next version.`);

  for (const rel of [...MANIFESTS, REGISTRY]) {
    const n = await setVersion(rel, from, to);
    console.log(`  ${from} -> ${to}  ${rel}${n > 1 ? `  (${n} fields)` : ""}`);
  }

  // The same build the release workflow's gate runs before it compares the
  // built manifests against the tag: pack all three, install them into an
  // empty directory, import them under Node and start the server.
  console.log("\n  building what would be published, and installing it under Node ...");
  await $`bun run build:wasm`.cwd(ROOT).quiet();
  await $`bun run build:dist`.cwd(ROOT);

  const after = await versions();
  const wrong = [...after].filter(([, v]) => v !== to);
  if (wrong.length > 0) fail(`Still not ${to}: ${wrong.map(([w]) => w).join(", ")}`);

  console.log(`
  Every version reads ${to}, and the packages build and install.

  Next: commit, open a pull request, merge it, then on main:

    bun run release tag
`);
}

async function tag(): Promise<void> {
  const branch = (await $`git branch --show-current`.cwd(ROOT).text()).trim();
  if (branch !== "main") fail(`Tags are cut from main, and this is ${branch || "a detached HEAD"}.`);

  const dirty = (await $`git status --porcelain`.cwd(ROOT).text()).trim();
  if (dirty) fail("The working tree has uncommitted changes. A tag names a commit, not a working tree.");

  await $`git fetch --quiet origin main --tags`.cwd(ROOT);
  const head = (await $`git rev-parse HEAD`.cwd(ROOT).text()).trim();
  const remote = (await $`git rev-parse origin/main`.cwd(ROOT).text()).trim();
  if (head !== remote) {
    fail("Local main is not origin/main. Pull first, so the tag names what everybody else has.");
  }

  const now = await versions();
  const distinct = new Set(now.values());
  if (distinct.size !== 1) {
    const lines = [...now].map(([where, v]) => `    ${v.padEnd(10)} ${where}`).join("\n");
    fail(`The versions disagree, and the release gate would refuse this tag:\n\n${lines}`);
  }
  const version = [...distinct][0]!;
  const name = `v${version}`;

  const onRemote = (await $`git ls-remote --tags origin refs/tags/${name}`.cwd(ROOT).text()).trim();
  if (onRemote) fail(`${name} is already on origin. Bump the version first: bun run release bump <next>`);

  const local = (await $`git tag --list ${name}`.cwd(ROOT).text()).trim();
  if (local) {
    const at = (await $`git rev-parse ${`refs/tags/${name}`}`.cwd(ROOT).text()).trim();
    if (at !== head) fail(`${name} exists locally at ${at.slice(0, 7)}, not at main (${head.slice(0, 7)}).`);
  } else {
    await $`git tag ${name}`.cwd(ROOT);
  }

  // By its full name. `--follow-tags` would skip a lightweight tag without a
  // word, and a bare `v0.1.5` is ambiguous the moment a branch shares the name.
  await $`git push origin ${`refs/tags/${name}:refs/tags/${name}`}`.cwd(ROOT);

  // A push that returned zero is not the same as a tag on the remote. Ask.
  const landed = (await $`git ls-remote --tags origin refs/tags/${name}`.cwd(ROOT).text()).trim();
  if (!landed) fail(`The push returned, but origin has no ${name}. Nothing was released.`);

  console.log(`
  ${name} is on origin at ${head.slice(0, 7)}. The release workflow starts from here:
  desktop builds, the engine, and the npm packages.

  npm's read path has lagged the publish job by minutes before. Before anything
  depends on the new version:

    npm view @tensor-cad/ui version
`);
}

const [command, arg] = process.argv.slice(2);
if (command === "bump") await bump(arg);
else if (command === "tag") await tag();
else fail("Usage: bun run release bump <version>   |   bun run release tag");
