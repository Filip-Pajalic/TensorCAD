/**
 * The live editor bridge: a loopback socket the running editor attaches to, so
 * an agent's edits appear on the canvas as it makes them and the human's edits
 * reach the agent.
 *
 * It is *not* a second document store. It watches the one store the tools
 * already write to and mirrors every change outward; an editor's own edit goes
 * through the same `apply` a tool call does, revision check and all. There is
 * one document, one revision counter and one undo log, which is what makes
 * "who is right" a question that never has to be answered.
 *
 * Off unless asked for. An MCP server that opened a port nobody requested
 * would be a surprise in every CI job that runs one, so `TENSORCAD_BRIDGE=1`
 * turns it on and the server is otherwise exactly as headless as it was.
 *
 * ## Who is allowed to connect
 *
 * Three things, in order of how much they actually do:
 *
 * 1. **The bind.** 127.0.0.1 only, so nothing off this machine can reach it.
 * 2. **The origin.** A page on the internet can open a WebSocket to your
 *    loopback address — the same-origin policy does not stop it — so the
 *    upgrade is refused unless the `Origin` header is absent (a native client)
 *    or names a localhost origin. This is the check that matters for a browser.
 * 3. **The token.** A random 32 bytes, in `~/.tensorcad/session.json` with
 *    owner-only permissions, and handed to loopback callers of `/session`.
 *    It guards against another program on this machine that guessed the port.
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { randomBytes } from "node:crypto";
import { WebSocketServer, type WebSocket } from "ws";
import type { Doc } from "@tensor-cad/engine";
import { Op as OpSchema } from "../schemas.js";
import type { Op } from "../ops.js";
import type { DesignRecord, DocumentStore, StoreChange } from "../store/types.js";
import { RevisionConflictError } from "../store/types.js";
import {
  BRIDGE_PROTOCOL,
  type ChangeReason,
  type ClientMessage,
  type DesignMessage,
  type ServerMessage,
} from "./protocol.js";
import { clearSession, writeSession } from "./session.js";

/**
 * Where the editor looks first. Fixed rather than ephemeral because a browser
 * cannot read the session file to find out — it probes this and the three
 * ports above it, which is also the range this server falls back through when
 * one is taken.
 */
export const DEFAULT_BRIDGE_PORT = 7357;
export const PORT_ATTEMPTS = 4;

export interface BridgeOptions {
  store: DocumentStore;
  /** What `hello` reports as the agent's working directory. */
  root: string;
  name: string;
  version: string;
  port?: number;
  /** Supplied by tests; otherwise 32 random bytes. */
  token?: string;
  /** Where to write the session file. Omitted means the real one. */
  sessionFile?: string;
  /** Somewhere to say what happened. Defaults to stderr, which is where a stdio server's diagnostics go. */
  log?: (line: string) => void;
}

/**
 * A design changed; something outside the bridge may want to know.
 *
 * `from` is the half the store cannot say. The MCP server needs it: telling an
 * agent that a resource changed because the agent changed it is noise, and
 * telling it when the *human* changed it is the whole point.
 */
export type BridgeWatcher = (change: StoreChange, from: "agent" | "editor") => void;

export class BridgeServer {
  readonly token: string;
  private readonly http: Server;
  private readonly wss: WebSocketServer;
  private readonly options: BridgeOptions;
  private readonly log: (line: string) => void;
  private unsubscribe?: () => void;
  private readonly watchers = new Set<BridgeWatcher>();
  private port_ = 0;

  /**
   * The connection whose message is being handled right now.
   *
   * The store calls its listeners synchronously, inside `apply`, so this is
   * set for exactly the duration of that call and is how a change is
   * attributed to the socket that caused it. An editor that has already drawn
   * its own edit does not want it sent back.
   */
  private acting?: WebSocket;

  constructor(options: BridgeOptions) {
    this.options = options;
    this.token = options.token ?? randomBytes(32).toString("hex");
    this.log = options.log ?? ((line) => process.stderr.write(`${line}\n`));

    this.http = createServer((req, res) => this.serveHttp(req, res));
    this.wss = new WebSocketServer({ noServer: true });
    this.http.on("upgrade", (req, socket, head) => {
      const refusal = this.refuse(req, req.url ?? "/");
      if (refusal) {
        // `end` rather than `write` then `destroy`, because destroying a
        // socket throws away whatever has not been flushed and the caller is
        // then left waiting for a refusal that was written and never sent.
        //
        // The timer is for Bun, which delivers *nothing* written to a socket
        // taken off an `upgrade` event — not on `write`, not on `end`, not
        // with a callback. Under Node the status line arrives and `end` has
        // already closed the socket by the time this fires; under Bun the
        // caller at least sees the connection refused rather than hanging on
        // a handshake that will never complete.
        socket.end(`HTTP/1.1 ${refusal}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n`);
        setTimeout(() => socket.destroy(), 50).unref();
        return;
      }
      this.wss.handleUpgrade(req, socket, head, (ws) => this.attach(ws));
    });
  }

  get port(): number {
    return this.port_;
  }

  /** `ws://127.0.0.1:<port>/bridge?token=…`, which is what the editor opens. */
  get url(): string {
    return `ws://127.0.0.1:${this.port_}/bridge?token=${this.token}`;
  }

  get connections(): number {
    return this.wss.clients.size;
  }

  /** Called for every store change, after the mirror has gone out. */
  watch(watcher: BridgeWatcher): () => void {
    this.watchers.add(watcher);
    return () => this.watchers.delete(watcher);
  }

  async start(): Promise<void> {
    const first = this.options.port ?? DEFAULT_BRIDGE_PORT;
    this.port_ = await listenSomewhere(this.http, first, PORT_ATTEMPTS);

    this.unsubscribe = this.options.store.subscribe((change) => this.mirror(change));

    await writeSession(
      {
        port: this.port_,
        token: this.token,
        pid: process.pid,
        root: this.options.root,
        started_at: new Date().toISOString(),
      },
      this.options.sessionFile,
    );
    this.log(`tensorcad bridge on 127.0.0.1:${this.port_}`);
  }

  async stop(): Promise<void> {
    this.unsubscribe?.();
    this.unsubscribe = undefined;
    for (const client of this.wss.clients) client.close(1001, "server stopping");
    await new Promise<void>((done) => this.wss.close(() => done()));
    // `close` alone waits for every open connection to end on its own, so an
    // editor that has stopped reading would keep the agent's process alive.
    this.http.closeAllConnections();
    await new Promise<void>((done) => this.http.close(() => done()));
    await clearSession(this.options.sessionFile);
  }

  // -- http ----------------------------------------------------------------

  /**
   * One endpoint, and it exists for one reason: a browser cannot read the
   * session file. Everything else is a 404, because this is not a web server.
   */
  private serveHttp(req: IncomingMessage, res: ServerResponse): void {
    const path = (req.url ?? "/").split("?")[0];
    if (path !== "/session") {
      res.writeHead(404, { "content-type": "text/plain" });
      res.end("tensorcad bridge\n");
      return;
    }
    const refusal = this.refuse(req, req.url ?? "/", { token: false });
    if (refusal) {
      res.writeHead(Number(refusal.split(" ")[0]), { "content-type": "text/plain" });
      res.end(`${refusal}\n`);
      return;
    }
    res.writeHead(200, {
      "content-type": "application/json",
      // The editor fetches this from its own origin, which is not this one.
      "access-control-allow-origin": req.headers.origin ?? "*",
    });
    res.end(JSON.stringify({ protocol: BRIDGE_PROTOCOL, port: this.port_, token: this.token }));
  }

  /** The reason to refuse, as an HTTP status line, or undefined to allow. */
  private refuse(req: IncomingMessage, url: string, checks = { token: true }): string | undefined {
    const remote = req.socket.remoteAddress ?? "";
    if (!isLoopback(remote)) return "403 Forbidden";

    const origin = req.headers.origin;
    if (origin !== undefined && !isLocalOrigin(origin)) return "403 Forbidden";

    if (checks.token) {
      const supplied = new URL(url, "http://127.0.0.1").searchParams.get("token");
      if (supplied !== this.token) return "401 Unauthorized";
    }
    return undefined;
  }

  // -- connections ---------------------------------------------------------

  private attach(ws: WebSocket): void {
    this.send(ws, {
      type: "hello",
      protocol: BRIDGE_PROTOCOL,
      server: this.options.name,
      version: this.options.version,
      root: this.options.root,
      designs: this.options.store.list(),
    });

    ws.on("message", (raw) => {
      let message: ClientMessage;
      try {
        message = JSON.parse(String(raw)) as ClientMessage;
      } catch (e) {
        this.send(ws, { type: "error", message: `not JSON: ${(e as Error).message}` });
        return;
      }
      try {
        this.handle(ws, message);
      } catch (e) {
        this.send(ws, { type: "error", message: (e as Error).message, about: message?.type });
      }
    });
  }

  private handle(ws: WebSocket, message: ClientMessage): void {
    switch (message?.type) {
      case "publish": {
        const doc = asDocument(message.doc);
        // `acting` so the editor is not sent back the document it just sent.
        // It still needs the id, which is what the reply carries.
        const record = this.during(ws, () => this.options.store.adopt(doc));
        this.send(ws, designMessage(record, "published"));
        return;
      }

      case "attach": {
        const record = this.options.store.get(message.design_id);
        this.send(ws, designMessage(record, "requested"));
        return;
      }

      case "ops":
      case "replace": {
        const write =
          message.type === "ops"
            ? () => this.options.store.apply(message.design_id, parseOps(message.ops), message.revision)
            : () => this.options.store.replace(message.design_id, asDocument(message.doc), message.revision);
        try {
          const { record } = this.during(ws, write);
          // Not a mirror of its own edit — an acknowledgement that it landed,
          // carrying the revision the editor must quote next.
          this.send(ws, designMessage(record, message.type === "ops" ? "applied" : "replaced"));
        } catch (e) {
          if (e instanceof RevisionConflictError) {
            this.send(ws, { type: "error", message: e.message, about: message.type });
            // And the truth, so the editor can rebuild rather than guess.
            this.send(ws, designMessage(this.options.store.get(message.design_id), "requested"));
            return;
          }
          throw e;
        }
        return;
      }

      default:
        this.send(ws, {
          type: "error",
          message: `unknown message type ${JSON.stringify((message as { type?: unknown })?.type)}`,
        });
    }
  }

  // -- mirroring -----------------------------------------------------------

  private mirror(change: StoreChange): void {
    const from = this.acting ? "editor" : "agent";
    const message = designMessage(change.record, change.kind, change.ops);
    for (const client of this.wss.clients) {
      if (client === this.acting) continue;
      this.send(client, message);
    }
    for (const watcher of this.watchers) {
      try {
        watcher(change, from);
      } catch (e) {
        this.log(`tensorcad bridge: watcher failed: ${(e as Error).message}`);
      }
    }
  }

  private during<T>(ws: WebSocket, work: () => T): T {
    this.acting = ws;
    try {
      return work();
    } finally {
      this.acting = undefined;
    }
  }

  private send(ws: WebSocket, message: ServerMessage): void {
    if (ws.readyState !== ws.OPEN) return;
    ws.send(JSON.stringify(message));
  }
}

// -- helpers ---------------------------------------------------------------

function designMessage(record: DesignRecord, reason: ChangeReason, ops?: Op[]): DesignMessage {
  const { doc, ...design } = record;
  const message: DesignMessage = { type: "design", reason, design, doc };
  if (ops) message.ops = ops;
  return message;
}

/**
 * Bind to `first`, or the next few ports when it is taken.
 *
 * Another editor session, or a stale process, is the usual reason — and
 * failing outright would take the whole MCP server down with it over a
 * convenience feature.
 */
async function listenSomewhere(server: Server, first: number, attempts: number): Promise<number> {
  let lastError: Error | undefined;
  for (let port = first; port < first + attempts; port++) {
    try {
      await new Promise<void>((resolve, reject) => {
        const onError = (e: Error) => {
          server.removeListener("listening", onListening);
          reject(e);
        };
        const onListening = () => {
          server.removeListener("error", onError);
          resolve();
        };
        server.once("error", onError);
        server.once("listening", onListening);
        server.listen(port, "127.0.0.1");
      });
      return port;
    } catch (e) {
      lastError = e as Error;
      if ((e as NodeJS.ErrnoException).code !== "EADDRINUSE") break;
    }
  }
  throw new Error(
    `tensorcad bridge could not listen on 127.0.0.1:${first}..${first + attempts - 1}: ${lastError?.message}`,
  );
}

function isLoopback(address: string): boolean {
  const bare = address.startsWith("::ffff:") ? address.slice("::ffff:".length) : address;
  return bare === "127.0.0.1" || bare === "::1" || bare.startsWith("127.");
}

/**
 * A localhost page, the desktop shell, or a file:// document.
 *
 * `*.localhost` is here for Wails, which serves the desktop build from
 * `http://wails.localhost` on Windows.
 */
export function isLocalOrigin(origin: string): boolean {
  if (origin === "null" || origin === "file://") return true;
  let url: URL;
  try {
    url = new URL(origin);
  } catch {
    return false;
  }
  if (url.protocol === "wails:" || url.protocol === "file:") return true;
  if (url.protocol !== "http:" && url.protocol !== "https:") return false;
  const host = url.hostname;
  return host === "localhost" || host.endsWith(".localhost") || host === "127.0.0.1" || host === "[::1]" || host === "::1";
}

function asDocument(value: unknown): Doc {
  const doc = value as Doc;
  if (!doc || typeof doc !== "object" || !doc.graph || !doc.meta) {
    throw new Error(`published value is not a design document: it has no "meta" and "graph".`);
  }
  return doc;
}

/**
 * The editor's operations go through the same schema a tool call does.
 *
 * It is the one place untrusted-shaped input reaches the store, and "the
 * editor sent it" is not a reason to trust it — a bad op there would corrupt
 * the document the agent is reading.
 */
function parseOps(value: unknown): Op[] {
  if (!Array.isArray(value)) throw new Error("ops must be an array");
  const parsed = OpSchema.array().safeParse(value);
  if (!parsed.success) {
    throw new Error(`ops rejected: ${parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ")}`);
  }
  return parsed.data as Op[];
}
