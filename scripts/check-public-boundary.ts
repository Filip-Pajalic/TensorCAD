/**
 * This repository is the open one. Keep it that way.
 *
 *   bun run scripts/check-public-boundary.ts
 *
 * Accounts, sessions, billing and the hosted service live in a separate
 * private repository. The seam between them is `packages/ui/src/state/storage.ts`,
 * which names no vendor and no host: a deployment registers a provider, a plain
 * checkout registers none and behaves exactly as it always has.
 *
 * That boundary is easy to state and easy to cross by accident — a debugging
 * line, a copied snippet, a `wrangler.jsonc` gaining a binding "just to try
 * it". Once a credential or a vendor's schema is in the history of a public
 * repository it is in it for good, and rewriting history is not a fix anybody
 * enjoys discovering they need.
 *
 * So this is a test rather than a convention. It reads every tracked file and
 * fails on anything that could only be here because the boundary moved.
 */

import { join } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");

/**
 * What must not appear, and the sentence to print when it does.
 *
 * Each is a *word*, matched case-insensitively, not a pattern to be clever
 * with — a rule nobody can predict the behaviour of is a rule people route
 * around.
 */
const FORBIDDEN: { pattern: RegExp; why: string }[] = [
  {
    pattern: /\b(supabase|clerk|workos|logto|auth0|better-auth|betterauth)\b/i,
    why: "an identity provider. Authentication lives in the private repository; this one has a storage interface that names nobody.",
  },
  {
    pattern: /\b(SUPABASE|CLERK|WORKOS|LOGTO|AUTH0|BETTER_AUTH)_[A-Z_]+\b/,
    why: "a vendor's environment variable. The public repository's only secrets are the CLOUDFLARE_* pair that deploys the documentation.",
  },
  {
    pattern: /"(d1_databases|r2_buckets|kv_namespaces|durable_objects|hyperdrive|vectorize)"/,
    why: "a stateful Cloudflare binding. The public repository deploys static assets and nothing else; a Worker with state belongs in the private one.",
  },
  {
    pattern: /\bTensorCAD-cloud\b/i,
    why: "the private repository. Referring to it by name from here is how a fork learns about something it cannot clone.",
  },
  {
    pattern: /\bservice_role\b|\bSERVICE_ROLE\b/,
    why: "a privileged key's name. Nothing in a public repository should be within reach of one.",
  },
];

/** Where a match is expected, because this file is the one describing them. */
const EXEMPT = new Set(["scripts/check-public-boundary.ts"]);

const SKIP_DIRS = new Set([
  "node_modules",
  ".git",
  "dist",
  "site",
  "out",
  "target",
  "runs",
  ".wrangler",
  ".turbo",
  "wasm",
]);

/** Binary and generated files, which are neither readable nor ours. */
const SKIP_EXT = new Set([
  ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg",
  ".wasm", ".lock", ".lockb", ".woff", ".woff2", ".ttf", ".pdf",
]);

interface Finding {
  file: string;
  line: number;
  text: string;
  why: string;
}

async function walk(dir: string, rel = "", out: string[] = []): Promise<string[]> {
  for (const entry of await Array.fromAsync(new Bun.Glob("*").scan({ cwd: dir, onlyFiles: false, dot: true }))) {
    const path = join(dir, entry);
    const relative = rel ? `${rel}/${entry}` : entry;
    const file = Bun.file(path);
    const stat = await file.stat().catch(() => null);
    if (stat?.isDirectory()) {
      if (!SKIP_DIRS.has(entry)) await walk(path, relative, out);
    } else if (!SKIP_EXT.has(entry.slice(entry.lastIndexOf(".")))) {
      out.push(relative);
    }
  }
  return out;
}

async function main(): Promise<void> {
  const files = await walk(ROOT);
  const findings: Finding[] = [];

  for (const relative of files) {
    if (EXEMPT.has(relative)) continue;
    let text: string;
    try {
      text = await Bun.file(join(ROOT, relative)).text();
    } catch {
      continue; // Not text after all.
    }
    const lines = text.split("\n");
    for (const { pattern, why } of FORBIDDEN) {
      for (let i = 0; i < lines.length; i++) {
        if (pattern.test(lines[i]!)) {
          findings.push({ file: relative, line: i + 1, text: lines[i]!.trim().slice(0, 100), why });
        }
      }
    }
  }

  if (findings.length === 0) {
    console.log(`  the boundary holds — ${files.length} files, nothing from the private side`);
    return;
  }

  console.error("\nThis is the public repository, and these lines do not belong in it:\n");
  for (const f of findings) {
    console.error(`  ${f.file}:${f.line}`);
    console.error(`    ${f.text}`);
    console.error(`    ${f.why}\n`);
  }
  console.error(
    "If one of these is a false positive, add it to EXEMPT in scripts/check-public-boundary.ts\n" +
      "with a comment saying why — deliberately, and in a commit somebody can review.\n",
  );
  process.exit(1);
}

await main();
