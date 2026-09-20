/**
 * `~/.tensorcad/session.json` — how a program on this machine finds the bridge.
 *
 * A file rather than a fixed port alone, because the port is only half the
 * answer: the other half is the token, and a token that lives in a file with
 * owner-only permissions is a token that a page you happened to visit cannot
 * read. The desktop shell reads this file. A browser cannot, and gets the
 * token from the bridge's own `/session` endpoint instead, which answers only
 * a loopback caller — see `server.ts`.
 *
 * It is written when the bridge starts listening and removed when it stops,
 * including on a clean exit. A file left behind by a crash names a port that
 * answers nothing, which is why every reader checks the port rather than
 * trusting the file.
 */

import { rmSync } from "node:fs";
import { chmod, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { BRIDGE_PROTOCOL, type SessionFile } from "./protocol.js";

export const SESSION_VERSION = 1;

/** Overridable so a test does not write into the developer's home directory. */
export function sessionPath(): string {
  return process.env.TENSORCAD_SESSION_FILE ?? join(homedir(), ".tensorcad", "session.json");
}

export async function writeSession(
  session: Omit<SessionFile, "version" | "protocol">,
  path = sessionPath(),
): Promise<void> {
  const full: SessionFile = { version: SESSION_VERSION, protocol: BRIDGE_PROTOCOL, ...session };
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(full, null, 2)}\n`, "utf8");
  // Best effort: on Windows this is close to a no-op, and the loopback bind
  // plus the origin check are what actually keep the bridge private there.
  try {
    await chmod(path, 0o600);
  } catch {
    /* not every filesystem has modes */
  }
}

export async function readSession(path = sessionPath()): Promise<SessionFile | undefined> {
  try {
    const parsed = JSON.parse(await readFile(path, "utf8")) as SessionFile;
    if (parsed.version !== SESSION_VERSION) return undefined;
    return parsed;
  } catch {
    return undefined;
  }
}

export async function clearSession(path = sessionPath()): Promise<void> {
  await rm(path, { force: true });
}

/** The same, for `process.on("exit")`, which cannot wait for a promise. */
export function clearSessionSync(path = sessionPath()): void {
  try {
    rmSync(path, { force: true });
  } catch {
    /* going away anyway */
  }
}
