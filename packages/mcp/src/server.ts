/**
 * The TensorCAD MCP server.
 *
 * Headless by design: tools operate on `.tensorcad.json` files and the built-in
 * reference architectures, and designs cross calls as an explicit
 * server-minted `design_id` rather than as connection state, which is what the
 * 2026-07-28 revision asks for.
 */

import { McpServer } from "@modelcontextprotocol/server";
import { FileStore, type FileStoreOptions } from "./store/file-store.js";
import { registerPrompts } from "./prompts.js";
import { registerResources } from "./resources.js";
import { registerTools } from "./tools.js";
import type { DocumentStore } from "./store/types.js";

export const SERVER_NAME = "tensorcad";
export const SERVER_VERSION = "0.0.1";

const INSTRUCTIONS = [
  "TensorCAD designs transformer language models as a graph of typed blocks and reports what they would cost.",
  "",
  "Hold a design_id from tensorcad_new_design or tensorcad_open_design and pass it to everything else.",
  'Read with tensorcad_get_design format "outline" before reaching for the full document: it carries the whole',
  "structure, the symbol table and the shape on every edge for a fraction of the tokens.",
  "",
  "Edit through tensorcad_apply_ops. A batch is all-or-nothing, and passing expected_revision turns a concurrent",
  "edit into a clear error instead of a silent overwrite. Take an tensorcad_checkpoint before an experiment;",
  "tensorcad_restore puts it back, or undoes the last batch when you name no checkpoint.",
  "",
  "Most designs are parameterised by symbols (L layers, D width, H heads, Hkv key/value heads, dh head dim,",
  "F feed-forward width, V vocabulary), so a set_symbol operation is usually the right edit rather than",
  "touching individual blocks.",
].join("\n");

export interface ServerOptions extends FileStoreOptions {
  /** Defaults to a `FileStore` rooted at the working directory. */
  store?: DocumentStore;
}

export function createServer(options: ServerOptions = {}): McpServer {
  const { store: given, ...storeOptions } = options;
  const store = given ?? new FileStore(storeOptions);

  const server = new McpServer(
    { name: SERVER_NAME, version: SERVER_VERSION },
    {
      capabilities: { tools: {}, resources: {}, prompts: {}, completions: {} },
      instructions: INSTRUCTIONS,
    },
  );

  registerTools(server, store);
  registerResources(server, store);
  registerPrompts(server);

  return server;
}
