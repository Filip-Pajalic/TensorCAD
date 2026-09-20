/**
 * The live editor bridge, over a real socket.
 *
 * Every test here opens an actual WebSocket to an actual listening server and
 * reads actual frames. A test that called the handler directly would prove the
 * handler works and nothing about the upgrade, the origin check or the token —
 * which is the half of this that decides whether a page you visited can drive
 * your editor.
 */

import { afterEach, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { connect } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import WebSocket from "ws";
import { getPreset, loadEngine } from "@tensorcad/engine/node";
import { BridgeServer } from "../src/bridge/server.js";
import type { ServerMessage } from "../src/bridge/protocol.js";
import { BRIDGE_PROTOCOL } from "../src/bridge/protocol.js";
import { FileStore } from "../src/store/file-store.js";

/** A socket that remembers everything it was sent, so a test can wait for one message. */
class Client {
  readonly seen: ServerMessage[] = [];
  private constructor(readonly socket: WebSocket) {}

  static async open(url: string, options: WebSocket.ClientOptions = {}): Promise<Client> {
    const socket = new WebSocket(url, options);
    const client = new Client(socket);
    socket.on("message", (raw) => client.seen.push(JSON.parse(String(raw)) as ServerMessage));
    await new Promise<void>((resolve, reject) => {
      socket.once("open", resolve);
      socket.once("error", reject);
    });
    return client;
  }

  send(message: unknown): void {
    this.socket.send(JSON.stringify(message));
  }

  /** The next message of this type, or a failure that says what did arrive. */
  async next<T extends ServerMessage["type"]>(
    type: T,
    after = 0,
  ): Promise<Extract<ServerMessage, { type: T }>> {
    const deadline = Date.now() + 2000;
    for (;;) {
      const found = this.seen.slice(after).find((m) => m.type === type);
      if (found) return found as Extract<ServerMessage, { type: T }>;
      if (Date.now() > deadline) {
        throw new Error(`no ${type} message; saw ${JSON.stringify(this.seen.map((m) => m.type))}`);
      }
      await new Promise((r) => setTimeout(r, 5));
    }
  }

  /** Nothing more arrived — which is the assertion when an echo would be wrong. */
  async quiet(from: number, ms = 150): Promise<ServerMessage[]> {
    await new Promise((r) => setTimeout(r, ms));
    return this.seen.slice(from);
  }

  close(): void {
    this.socket.close();
  }
}

/**
 * Whether a handshake was turned away, written by hand over a plain socket.
 *
 * Three things about this, each of them a Bun quirk that cost an afternoon:
 *
 * The `ws` client cannot answer the question. Bun's shim does not implement
 * `unexpected-response`, so a refused upgrade is a socket that simply never
 * opens: the test hangs rather than failing, and says nothing about what the
 * server did.
 *
 * The question is "was it turned away", not "what status came back", because
 * Bun delivers *nothing* written to a socket taken off an `upgrade` event. The
 * 401 the server sends arrives under Node and is swallowed under Bun. Whether
 * the connection became a WebSocket is the property that matters and the one
 * both runtimes agree on; the status codes themselves are pinned by the
 * `/session` tests below, which go through the ordinary response path.
 *
 * And it is only ever pointed at a refusal. A raw socket that *does* get the
 * upgrade leaves `ws` holding a connection this helper then destroys without a
 * close handshake, and Bun aborts the whole process — no error, no stack, exit
 * 127 — when the server is closed over it. Every other test here opens a real
 * `ws` client, so that a good token upgrades is not a thing this needs to say.
 */
async function refused(port: number, path: string, headers: Record<string, string> = {}): Promise<boolean> {
  const socket = connect(port, "127.0.0.1");
  const lines = [
    `GET ${path} HTTP/1.1`,
    `Host: 127.0.0.1:${port}`,
    "Upgrade: websocket",
    "Connection: Upgrade",
    "Sec-WebSocket-Key: AAAAAAAAAAAAAAAAAAAAAA==",
    "Sec-WebSocket-Version: 13",
    ...Object.entries(headers).map(([k, v]) => `${k}: ${v}`),
    "",
    "",
  ];
  return await new Promise<boolean>((resolve) => {
    const done = (answer: boolean) => {
      socket.destroy();
      resolve(answer);
    };
    socket.on("error", () => done(true));
    socket.on("close", () => resolve(true));
    socket.once("connect", () => socket.write(lines.join("\r\n")));
    socket.once("data", (chunk) => done(!String(chunk).startsWith("HTTP/1.1 101")));
    setTimeout(() => done(true), 1500);
  });
}

describe("the live editor bridge", () => {
  let dir: string;
  let store: FileStore;
  let bridge: BridgeServer;
  let sessionFile: string;

  // The store reaches the presets through the engine, and these tests hold it
  // in this process rather than behind a stdio pipe.
  beforeAll(async () => {
    await loadEngine();
  });

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "tensorcad-bridge-"));
    sessionFile = join(dir, "session.json");
    store = new FileStore({ root: dir });
    bridge = new BridgeServer({
      store,
      root: dir,
      name: "tensorcad",
      version: "0.0.0-test",
      // Away from the default, so a bridge a developer is actually running
      // does not decide whether these pass.
      port: 7411,
      token: "test-token",
      sessionFile,
      log: () => {},
    });
    await bridge.start();
  });

  afterEach(async () => {
    await bridge.stop();
    await rm(dir, { recursive: true, force: true });
  });

  test("greets a new connection with the protocol and what is open", async () => {
    store.create({ preset: "gpt2-small" });

    const client = await Client.open(bridge.url);
    const hello = await client.next("hello");

    expect(hello.protocol).toBe(BRIDGE_PROTOCOL);
    expect(hello.server).toBe("tensorcad");
    expect(hello.root).toBe(dir);
    expect(hello.designs.map((d) => d.name)).toEqual(["gpt2-small"]);
    client.close();
  });

  test("an agent's edit reaches the canvas, with what it did", async () => {
    const record = store.create({ preset: "gpt2-small" });
    const client = await Client.open(bridge.url);
    await client.next("hello");
    const before = client.seen.length;

    store.apply(record.design_id, [{ op: "set_symbol", name: "L", value: 6 }]);

    const message = await client.next("design", before);
    expect(message.reason).toBe("applied");
    expect(message.design.design_id).toBe(record.design_id);
    expect(message.design.revision).toBe(2);
    expect(message.ops).toEqual([{ op: "set_symbol", name: "L", value: 6 }]);
    expect(message.doc.symbols?.L).toMatchObject({ value: 6 });
    client.close();
  });

  test("the editor publishes what it has open and the agent can see it", async () => {
    const client = await Client.open(bridge.url);
    await client.next("hello");
    const before = client.seen.length;

    client.send({ type: "publish", doc: getPreset("llama-3-8b") });

    const message = await client.next("design", before);
    expect(message.reason).toBe("published");
    expect(store.list().map((d) => d.name)).toEqual(["llama-3-8b"]);
    expect(store.get(message.design.design_id).doc.meta.name).toBe("llama-3-8b");
    client.close();
  });

  test("the human's edit goes through the same apply, and is not sent back to them", async () => {
    const record = store.create({ preset: "gpt2-small" });
    const client = await Client.open(bridge.url);
    await client.next("hello");
    const before = client.seen.length;

    client.send({
      type: "ops",
      design_id: record.design_id,
      revision: 1,
      ops: [{ op: "set_symbol", name: "L", value: 4 }],
    });

    const ack = await client.next("design", before);
    expect(ack.design.revision).toBe(2);
    expect(store.get(record.design_id).doc.symbols?.L).toMatchObject({ value: 4 });

    // One message, not two: the acknowledgement, and no mirror of its own edit.
    expect(await client.quiet(before)).toHaveLength(1);
    client.close();
  });

  test("a whole document from the editor lands like an edit, and undoes like one", async () => {
    const record = store.create({ preset: "gpt2-small" });
    const client = await Client.open(bridge.url);
    await client.next("hello");
    const before = client.seen.length;

    const edited = structuredClone(record.doc);
    edited.meta.name = "renamed by a human";

    client.send({ type: "replace", design_id: record.design_id, revision: 1, doc: edited });

    const ack = await client.next("design", before);
    expect(ack.reason).toBe("replaced");
    expect(ack.design.revision).toBe(2);
    expect(store.get(record.design_id).doc.meta.name).toBe("renamed by a human");

    // The undo log took it, which is what makes the agent's `tensorcad_restore`
    // able to put back what a human did.
    store.restore(record.design_id);
    expect(store.get(record.design_id).doc.meta.name).toBe("gpt2-small");
    client.close();
  });

  test("a second editor does see the first one's edit", async () => {
    const record = store.create({ preset: "gpt2-small" });
    const one = await Client.open(bridge.url);
    const two = await Client.open(bridge.url);
    await one.next("hello");
    await two.next("hello");
    const before = two.seen.length;

    one.send({ type: "ops", design_id: record.design_id, ops: [{ op: "set_symbol", name: "L", value: 4 }] });

    const mirrored = await two.next("design", before);
    expect(mirrored.reason).toBe("applied");
    expect(mirrored.design.revision).toBe(2);
    one.close();
    two.close();
  });

  test("a stale revision is refused, and the truth follows so the editor can rebuild", async () => {
    const record = store.create({ preset: "gpt2-small" });
    store.apply(record.design_id, [{ op: "set_symbol", name: "L", value: 6 }]);

    const client = await Client.open(bridge.url);
    await client.next("hello");
    const before = client.seen.length;

    client.send({
      type: "ops",
      design_id: record.design_id,
      revision: 1,
      ops: [{ op: "set_symbol", name: "L", value: 4 }],
    });

    const error = await client.next("error", before);
    expect(error.about).toBe("ops");
    expect(error.message).toContain("revision 2");

    const truth = await client.next("design", before);
    expect(truth.reason).toBe("requested");
    expect(truth.design.revision).toBe(2);
    expect(store.get(record.design_id).doc.symbols?.L).toMatchObject({ value: 6 });
    client.close();
  });

  test("attaching again after a reconnect does not make a second copy", async () => {
    const one = await Client.open(bridge.url);
    await one.next("hello");
    one.send({ type: "publish", doc: getPreset("gpt2-small") });
    const published = await one.next("design");
    one.close();

    const two = await Client.open(bridge.url);
    await two.next("hello");
    const before = two.seen.length;
    two.send({ type: "attach", design_id: published.design.design_id });

    const again = await two.next("design", before);
    expect(again.reason).toBe("requested");
    expect(again.design.design_id).toBe(published.design.design_id);
    expect(store.list()).toHaveLength(1);
    two.close();
  });

  test("an operation the schema does not know is refused rather than applied", async () => {
    const record = store.create({ preset: "gpt2-small" });
    const client = await Client.open(bridge.url);
    await client.next("hello");
    const before = client.seen.length;

    client.send({ type: "ops", design_id: record.design_id, ops: [{ op: "drop_everything" }] });

    const error = await client.next("error", before);
    expect(error.message).toContain("rejected");
    expect(store.get(record.design_id).revision).toBe(1);
    client.close();
  });

  describe("telling the agent the human moved", () => {
    test("an edit the agent made is its own doing", async () => {
      const record = store.create({ preset: "gpt2-small" });
      const seen: string[] = [];
      bridge.watch((_change, from) => seen.push(from));

      store.apply(record.design_id, [{ op: "set_symbol", name: "L", value: 6 }]);

      expect(seen).toEqual(["agent"]);
    });

    test("an edit the editor made is not", async () => {
      const record = store.create({ preset: "gpt2-small" });
      const seen: string[] = [];
      bridge.watch((_change, from) => seen.push(from));

      const client = await Client.open(bridge.url);
      await client.next("hello");
      client.send({ type: "ops", design_id: record.design_id, ops: [{ op: "set_symbol", name: "L", value: 4 }] });
      await client.next("design", 1);

      // The `create` above was the agent's; the edit was the human's. Without
      // the distinction the agent would be told about its own work and not
      // told about the only thing it cannot see.
      expect(seen).toEqual(["editor"]);
      client.close();
    });
  });

  describe("who is allowed in", () => {
    test("the wrong token is refused", async () => {
      expect(await refused(bridge.port, "/bridge?token=wrong")).toBe(true);
    });

    test("no token at all is refused", async () => {
      expect(await refused(bridge.port, "/bridge")).toBe(true);
    });

    test("a page on the internet is refused even with the right token", async () => {
      const got = await refused(bridge.port, "/bridge?token=test-token", {
        Origin: "https://not-your-editor.example",
      });
      expect(got).toBe(true);
    });

    test("a page on localhost is let in", async () => {
      const client = await Client.open(bridge.url, { origin: "http://localhost:5173" });
      expect((await client.next("hello")).protocol).toBe(BRIDGE_PROTOCOL);
      client.close();
    });

    test("the desktop shell's own origin is let in", async () => {
      const client = await Client.open(bridge.url, { origin: "http://wails.localhost" });
      expect((await client.next("hello")).protocol).toBe(BRIDGE_PROTOCOL);
      client.close();
    });
  });

  describe("finding it", () => {
    test("the session file says where it is while it is listening", async () => {
      const session = JSON.parse(await readFile(sessionFile, "utf8"));
      expect(session).toMatchObject({
        protocol: BRIDGE_PROTOCOL,
        port: bridge.port,
        token: "test-token",
        pid: process.pid,
        root: dir,
      });
    });

    test("and is gone when it stops", async () => {
      await bridge.stop();
      expect(readFile(sessionFile, "utf8")).rejects.toThrow();
      // afterEach stops it again, which must not throw either.
    });

    test("a localhost page can ask for the token, because it cannot read the file", async () => {
      const res = await fetch(`http://127.0.0.1:${bridge.port}/session`, {
        headers: { origin: "http://localhost:5173" },
      });
      expect(res.status).toBe(200);
      expect(await res.json()).toMatchObject({ protocol: BRIDGE_PROTOCOL, token: "test-token" });
    });

    test("a page on the internet cannot", async () => {
      const res = await fetch(`http://127.0.0.1:${bridge.port}/session`, {
        headers: { origin: "https://not-your-editor.example" },
      });
      expect(res.status).toBe(403);
    });
  });
});
