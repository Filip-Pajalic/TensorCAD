/**
 * The editor's end of the live agent bridge.
 *
 * An agent running the MCP server with `TENSORCAD_BRIDGE=1` listens on
 * 127.0.0.1. This finds it, publishes whatever is on screen so the agent can
 * work on *that* design rather than a copy of a file, and then keeps the two
 * in step: the agent's edits arrive here and land on the undo stack, and this
 * editor's edits go back the other way.
 *
 * ## Finding it
 *
 * The bridge writes its port and token to `~/.tensorcad/session.json`, which a
 * browser cannot read. So it also answers `GET /session` for a caller on
 * loopback with a localhost origin, and this probes a small range of ports for
 * that. Four fetches at startup, and only where a bridge could possibly be:
 * a page served over https is not on the same machine as the agent in any
 * arrangement worth supporting, and probing from one would be four console
 * errors about mixed content on every load of the hosted editor.
 *
 * ## Not fighting over the document
 *
 * The agent's store is the one that counts. Every edit here is sent as the
 * whole document with the revision it was made against; if the agent has moved
 * on, the bridge refuses it and sends the truth, which is taken. The
 * alternative — merging — needs a model of intent that neither end has.
 *
 * What comes *back* lands through `applyRemote`, which keeps the level you are
 * looking at and pushes the previous document onto the undo stack. An agent's
 * edit is undoable by the person watching it, which is the property that makes
 * watching one bearable.
 */

import { create } from "zustand";
import type { Doc } from "@tensor-cad/engine";
import { useEditor } from "./store.js";

/** The port the agent's bridge tries first, and the three it falls back through. */
const FIRST_PORT = 7357;
const PORTS = 4;

/** Long enough that a drag does not send a document per frame. */
const SEND_AFTER_MS = 250;

/** Remembered, so turning it off stays off. */
const ENABLED_KEY = "tensorcad.bridge.enabled";

export type BridgeStatus = "off" | "looking" | "connected" | "failed";

interface SessionAnswer {
  protocol: number;
  port: number;
  token: string;
}

export interface BridgeState {
  status: BridgeStatus;
  /** What the agent calls itself, from `hello`. */
  agent: string | null;
  /** The design this editor is attached to, in the agent's numbering. */
  designId: string | null;
  /** The agent's working directory, which is where its relative paths resolve. */
  root: string | null;
  /** Why it is not connected, when that is worth saying. */
  detail: string | null;
  /** What the agent last did, for the status bar. */
  lastFromAgent: string | null;
  enabled: boolean;
  setEnabled: (enabled: boolean) => void;
}

function rememberedEnabled(): boolean {
  try {
    return localStorage.getItem(ENABLED_KEY) !== "0";
  } catch {
    return true;
  }
}

export const useBridge = create<BridgeState>((set) => ({
  status: "off",
  agent: null,
  designId: null,
  root: null,
  detail: null,
  lastFromAgent: null,
  enabled: rememberedEnabled(),
  setEnabled: (enabled) => {
    try {
      localStorage.setItem(ENABLED_KEY, enabled ? "1" : "0");
    } catch {
      /* not remembering it is harmless */
    }
    set({ enabled });
    if (enabled) void connect();
    else disconnect();
  },
}));

/**
 * Whether a bridge could be on this machine at all.
 *
 * The desktop shell and a dev server can reach one. An editor served over
 * https cannot: the browser blocks a `ws://` from an https page, and there is
 * no agent on the far side of the internet to talk to anyway.
 */
export function bridgeIsPossible(): boolean {
  if (typeof location === "undefined") return false;
  if (location.protocol === "wails:" || location.protocol === "file:") return true;
  if (location.protocol !== "http:") return false;
  const host = location.hostname;
  return host === "localhost" || host.endsWith(".localhost") || host === "127.0.0.1" || host === "[::1]";
}

// -- the connection ---------------------------------------------------------

let socket: WebSocket | null = null;
/**
 * The document as both ends last agreed on it.
 *
 * Compared by identity, not by value: what it prevents is sending back the
 * very object that just arrived, which would be an edit the agent then mirrors
 * to us, and so on.
 */
let agreed: Doc | null = null;
let agreedRevision = 0;
let pending: ReturnType<typeof setTimeout> | null = null;
let watching: (() => void) | null = null;

async function findSession(): Promise<{ port: number; token: string } | undefined> {
  for (let port = FIRST_PORT; port < FIRST_PORT + PORTS; port++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/session`, { signal: AbortSignal.timeout(400) });
      if (!res.ok) continue;
      const answer = (await res.json()) as SessionAnswer;
      if (typeof answer?.token === "string") return { port, token: answer.token };
    } catch {
      // Nothing listening there, which is the ordinary case.
    }
  }
  return undefined;
}

/** Attach to a running agent, if there is one. Safe to call twice. */
export async function connect(): Promise<void> {
  if (socket || !useBridge.getState().enabled || !bridgeIsPossible()) return;

  useBridge.setState({ status: "looking", detail: null });
  const found = await findSession();
  if (!found) {
    useBridge.setState({ status: "off", detail: "No agent is listening." });
    return;
  }

  const ws = new WebSocket(`ws://127.0.0.1:${found.port}/bridge?token=${found.token}`);
  socket = ws;

  ws.onopen = () => {
    // Publish what is on screen. The agent should be able to work on the
    // design the person is looking at, not on a file that happens to resemble
    // it.
    const doc = useEditor.getState().doc;
    agreed = doc;
    ws.send(JSON.stringify({ type: "publish", doc }));
  };

  ws.onmessage = (event) => receive(String(event.data));

  ws.onerror = () => {
    useBridge.setState({ status: "failed", detail: "The connection failed." });
  };

  ws.onclose = () => {
    if (socket === ws) socket = null;
    stopWatching();
    useBridge.setState({
      status: useBridge.getState().enabled ? "off" : "off",
      agent: null,
      designId: null,
      root: null,
      detail: "The agent disconnected.",
    });
  };
}

export function disconnect(): void {
  stopWatching();
  socket?.close();
  socket = null;
  agreed = null;
  agreedRevision = 0;
  useBridge.setState({ status: "off", agent: null, designId: null, root: null, lastFromAgent: null });
}

function receive(text: string): void {
  let message: Record<string, unknown>;
  try {
    message = JSON.parse(text) as Record<string, unknown>;
  } catch {
    return;
  }

  if (message.type === "hello") {
    useBridge.setState({
      status: "connected",
      agent: `${String(message.server)} ${String(message.version)}`,
      root: String(message.root),
      detail: null,
    });
    return;
  }

  if (message.type === "error") {
    // A refused edit is followed by the truth, so there is nothing to do here
    // but say what happened.
    useBridge.setState({ detail: String(message.message) });
    return;
  }

  if (message.type !== "design") return;

  const design = message.design as { design_id: string; revision: number };
  const doc = message.doc as Doc;
  const reason = String(message.reason);
  const state = useBridge.getState();

  // Only the design this editor attached to. An agent working on a second one
  // is not a reason to replace what is on screen.
  if (state.designId && design.design_id !== state.designId) return;

  agreed = doc;
  agreedRevision = design.revision;
  useBridge.setState({
    designId: design.design_id,
    lastFromAgent: reason === "published" ? null : describe(reason, message.ops as unknown[] | undefined),
  });

  // The answer to our own publish is the document we just sent.
  if (reason === "published" || reason === "replaced") {
    startWatching();
    return;
  }

  useEditor.getState().applyRemote(doc, describe(reason, message.ops as unknown[] | undefined));
  startWatching();
}

function describe(reason: string, ops: unknown[] | undefined): string {
  if (reason === "applied" && ops?.length) {
    const names = ops.map((o) => String((o as { op?: unknown }).op ?? "edit"));
    return names.length === 1 ? `Agent: ${names[0]}` : `Agent: ${names.length} operations`;
  }
  if (reason === "restored") return "Agent: undo";
  if (reason === "saved") return "Agent: saved";
  return `Agent: ${reason}`;
}

// -- sending our own edits back --------------------------------------------

function startWatching(): void {
  if (watching) return;
  watching = useEditor.subscribe((state, previous) => {
    if (state.doc === previous.doc) return;
    if (state.doc === agreed) return;
    schedule();
  });
}

function stopWatching(): void {
  watching?.();
  watching = null;
  if (pending) clearTimeout(pending);
  pending = null;
}

/**
 * Coalesce a burst of edits into one message.
 *
 * Dragging a block changes the document on every frame, and a document a
 * frame would have the agent's revision counter racing ahead of anything it
 * could act on.
 */
function schedule(): void {
  if (pending) clearTimeout(pending);
  pending = setTimeout(() => {
    pending = null;
    send();
  }, SEND_AFTER_MS);
}

function send(): void {
  const { designId } = useBridge.getState();
  const doc = useEditor.getState().doc;
  if (!socket || socket.readyState !== WebSocket.OPEN || !designId || doc === agreed) return;
  socket.send(JSON.stringify({ type: "replace", design_id: designId, revision: agreedRevision, doc }));
}
