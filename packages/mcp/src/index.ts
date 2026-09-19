#!/usr/bin/env bun
/**
 * stdio entry point.
 *
 * Nothing but newline-delimited JSON-RPC may reach stdout; anything we want to
 * say goes to stderr.
 */

import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { loadEngine } from "@tensorcad/engine/node";
import { createServer } from "./server.js";

export { createServer, SERVER_NAME, SERVER_VERSION } from "./server.js";
export { FileStore } from "./store/file-store.js";
export type { DocumentStore, DesignRecord, DesignSummary } from "./store/types.js";
export { TOOL_NAMES } from "./tools.js";
export { PROMPT_NAMES } from "./prompts.js";
export { applyOps, type Op } from "./ops.js";

if (import.meta.main) {
  const root = process.env.TENSORCAD_ROOT ?? process.cwd();
  // Before the transport opens: a tool call that arrived while the engine was
  // still loading would fail for a reason that has nothing to do with it.
  await loadEngine();
  serveStdio(() => createServer({ root }));
  process.stderr.write(`tensorcad mcp server on stdio, root ${root}\n`);
}
