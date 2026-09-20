/**
 * Starting the server, as a function rather than as a side effect.
 *
 * It used to be a bare `if (import.meta.main)` at the bottom of `index.ts`,
 * which made that module two things: a library other code imports and a program
 * that starts a server. That works under Bun and does not survive a bundler —
 * `import.meta.main` compiles to a CommonJS check that is not defined in an ESM
 * output, so the bundled server crashed before its first line ran.
 */

import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { loadEngine } from "@tensorcad/engine/node";
import { createServer } from "./server.js";

/** Serve on stdio until the transport closes. */
export async function serve(root = process.env.TENSORCAD_ROOT ?? process.cwd()): Promise<void> {
  // Before the transport opens: a tool call that arrived while the engine was
  // still loading would fail for a reason that has nothing to do with it.
  await loadEngine();
  serveStdio(() => createServer({ root }));
  // Nothing but newline-delimited JSON-RPC may reach stdout.
  process.stderr.write(`tensorcad mcp server on stdio, root ${root}\n`);
}
