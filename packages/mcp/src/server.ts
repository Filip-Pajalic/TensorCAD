/**
 * The TensorCAD MCP server.
 *
 * Headless by design: tools operate on `.tensorcad.json` files and the built-in
 * reference architectures, and designs cross calls as an explicit
 * server-minted `design_id` rather than as connection state, which is what the
 * 2026-07-28 revision asks for.
 */

import { McpServer } from "@modelcontextprotocol/server";
import manifest from "../package.json" with { type: "json" };
import { FileStore, type FileStoreOptions } from "./store/file-store.js";
import { DiskSink } from "./store/disk-artifacts.js";
import { registerPrompts } from "./prompts.js";
import { registerResources } from "./resources.js";
import { registerTools } from "./tools.js";
import type { DocumentStore } from "./store/types.js";
import type { ArtifactSink } from "./artifacts.js";
import type { BridgeServer } from "./bridge/server.js";

export const SERVER_NAME = "tensorcad";
/**
 * What the server tells a client it is, read from the package rather than
 * written here. A second copy is a copy that goes stale, and this one is
 * compared against `server.json` and against what the registry was told — so
 * the failure would be a published version claiming to be another.
 */
export const SERVER_VERSION: string = manifest.version;

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
  "",
  "After an edit, tensorcad_diff against the design you started from says what moved and what it cost;",
  "tensorcad_explain answers why one block is the size it is without reading the whole document.",
  "tensorcad_plan answers whether the thing would train on a given number of GPUs and how it would have to",
  "be split. tensorcad_scale shrinks a design to a bench budget, and tensorcad_import_hf reads a Hugging Face",
  "config.json into one. Both save their result as a new design, analysable and diffable like any other.",
].join("\n");

export interface ServerOptions extends FileStoreOptions {
  /** Defaults to a `FileStore` rooted at the working directory. */
  store?: DocumentStore;
  /**
   * Where `tensorcad_generate_code` puts what it emits.
   *
   * Defaults to writing into a directory. A hosted server has no filesystem
   * and supplies object storage instead, which is the difference between an
   * agent being handed a model it can run and one it can only read.
   */
  artifacts?: ArtifactSink;
  /**
   * The live editor bridge, when one is running. Given one, the server tells
   * its client that a design's resources changed whenever the *human* changed
   * them — which is the half of the bridge an agent can act on.
   */
  bridge?: BridgeServer;
}

export function createServer(options: ServerOptions = {}): McpServer {
  const { store: given, bridge, artifacts: givenArtifacts, ...storeOptions } = options;
  const store = given ?? new FileStore(storeOptions);
  const artifacts = givenArtifacts ?? new DiskSink(storeOptions.root ?? process.cwd());

  const server = new McpServer(
    { name: SERVER_NAME, version: SERVER_VERSION },
    {
      capabilities: { tools: {}, resources: {}, prompts: {}, completions: {} },
      instructions: INSTRUCTIONS,
    },
  );

  registerTools(server, store, artifacts);
  registerResources(server, store);
  registerPrompts(server);
  if (bridge) followEditor(server, bridge);

  return server;
}

/**
 * Tell the client that a design changed under it, when a human was the one who
 * changed it.
 *
 * Three URIs rather than one, because a client that subscribed to a design's
 * *analysis* cares that the numbers moved and never asked about the document.
 * The notification carries no content: whoever wants the new value reads the
 * resource, which is what makes this cheap enough to send on every keystroke's
 * worth of edit.
 *
 * `serveStdio` builds one of these per connection — and one more for a
 * `server/discover` probe it throws away — so the watcher comes off again when
 * the connection closes. Otherwise the probe's dead instance would keep
 * answering for the life of the process.
 */
function followEditor(server: McpServer, bridge: BridgeServer): void {
  const stop = bridge.watch((change, from) => {
    if (from !== "editor") return;
    const base = `tensorcad://designs/${change.record.design_id}`;
    for (const uri of [base, `${base}/analysis`, `${base}/validation`]) {
      // Nothing to do about a client that has gone away mid-notification.
      void server.server.sendResourceUpdated({ uri }).catch(() => {});
    }
  });

  const closed = server.server.onclose;
  server.server.onclose = () => {
    stop();
    closed?.();
  };
}
