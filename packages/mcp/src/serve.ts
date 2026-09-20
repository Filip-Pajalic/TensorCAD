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
import { BridgeServer, DEFAULT_BRIDGE_PORT } from "./bridge/server.js";
import { clearSessionSync } from "./bridge/session.js";
import { createServer, SERVER_NAME, SERVER_VERSION } from "./server.js";
import { FileStore } from "./store/file-store.js";

/** Serve on stdio until the transport closes. */
export async function serve(root = process.env.TENSORCAD_ROOT ?? process.cwd()): Promise<void> {
  // Before the transport opens: a tool call that arrived while the engine was
  // still loading would fail for a reason that has nothing to do with it.
  await loadEngine();

  // One store for the process rather than one per connection. It has to be:
  // the bridge mirrors *a* store, and a store the tools do not write to would
  // mirror nothing. It is also what the era probe should not get a second copy
  // of — `serveStdio` builds an instance for `server/discover` and discards it.
  const store = new FileStore({ root });
  const bridge = await startBridge(store, root);

  serveStdio(() => createServer({ root, store, bridge }));
  // Nothing but newline-delimited JSON-RPC may reach stdout.
  process.stderr.write(`tensorcad mcp server on stdio, root ${root}\n`);
}

/**
 * The live editor bridge, when it was asked for.
 *
 * Opt-in, because opening a port is not something a program should do on the
 * strength of having been started: every CI job that runs this server would
 * then be listening on one. `TENSORCAD_BRIDGE=1` asks for it.
 *
 * A bridge that will not start is reported and shrugged off. The server's own
 * job — answering tool calls over stdio — does not depend on it, and taking
 * the whole thing down because a port was busy would trade a feature for the
 * product.
 */
async function startBridge(store: FileStore, root: string): Promise<BridgeServer | undefined> {
  if (process.env.TENSORCAD_BRIDGE !== "1") return undefined;

  const bridge = new BridgeServer({
    store,
    root,
    name: SERVER_NAME,
    version: SERVER_VERSION,
    port: Number(process.env.TENSORCAD_BRIDGE_PORT) || DEFAULT_BRIDGE_PORT,
  });

  try {
    await bridge.start();
  } catch (e) {
    process.stderr.write(`tensorcad: the editor bridge did not start: ${(e as Error).message}\n`);
    return undefined;
  }

  // `stop` is async and `exit` cannot wait, so the session file is removed
  // synchronously here. A file naming a port that answers nothing is how an
  // editor spends ten seconds failing to connect to a server that is gone.
  process.once("exit", () => clearSessionSync());
  for (const signal of ["SIGINT", "SIGTERM"] as const) {
    process.once(signal, () => {
      clearSessionSync();
      process.exit(0);
    });
  }
  return bridge;
}
