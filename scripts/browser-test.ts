#!/usr/bin/env bun
/**
 * The editor, driven by a keyboard in a real browser.
 *
 *   bun run test:browser              # builds the site first
 *   bun run test:browser --no-build   # uses packages/ui/dist as it is
 *
 * Everything under `packages/ui/test` runs without a browser, which is what
 * makes it fast and is also what it cannot see: whether the engine arrives
 * before the first frame, whether a key does what the shortcut sheet says once
 * React Flow has had its turn at it, whether focus survives a redraw. Each of
 * those has been wrong at some point with every unit test passing.
 *
 * So this serves the production build, opens it in headless Chrome over the
 * DevTools protocol — the same client `screenshots.ts` uses, no automation
 * dependency — and presses keys as the operating system would. Keys are sent
 * with `Input.dispatchKeyEvent`, which the page cannot tell from a person:
 * an event built with `new KeyboardEvent` is untrusted, and a browser treats
 * it differently in exactly the places this is meant to check.
 *
 * Skips, rather than fails, on a machine with no Chrome — unless it is CI,
 * where a missing browser means the check silently stopped running.
 */

import { rm } from "node:fs/promises";
import { join } from "node:path";
import { $ } from "bun";
import { CHROME, Devtools, wait } from "./screenshots.js";

const ROOT = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const DIST = join(ROOT, "packages/ui/dist");
const PORT = 9335;

// --- the site ----------------------------------------------------------------

if (!process.argv.includes("--no-build")) {
  console.log("  building the site ...");
  await $`bunx vite build`.cwd(join(ROOT, "packages/ui")).quiet();
}

/** A static host, as the deployment is: a file if there is one, else the app. */
const server = Bun.serve({
  port: 0,
  async fetch(req) {
    const path = decodeURIComponent(new URL(req.url).pathname);
    const file = Bun.file(join(DIST, path));
    if (path !== "/" && (await file.exists())) return new Response(file);
    return new Response(Bun.file(join(DIST, "index.html")));
  },
});
const SITE = `http://localhost:${server.port}`;

// --- the browser -------------------------------------------------------------

let chrome = "";
for (const candidate of CHROME) {
  if (await Bun.file(candidate).exists()) {
    chrome = candidate;
    break;
  }
}
if (!chrome) {
  server.stop();
  if (process.env.CI) {
    console.error("  No Chrome on this runner, so nothing was tested.");
    process.exit(1);
  }
  console.warn("  No Chrome found; skipping the browser tests.");
  process.exit(0);
}

const profile = join(ROOT, "dist", "chrome-profile-test");
await rm(profile, { recursive: true, force: true });
const browser = Bun.spawn(
  [
    chrome,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${profile}`,
    "--window-size=1440,900",
    "about:blank",
  ],
  { stdout: "ignore", stderr: "ignore" },
);

/** Keys by the name the page sees, with what the protocol needs to send them. */
const KEYS: Record<string, { code: string; vk: number; text?: string }> = {
  Enter: { code: "Enter", vk: 13, text: "\r" },
  Escape: { code: "Escape", vk: 27 },
  Tab: { code: "Tab", vk: 9 },
  "[": { code: "BracketLeft", vk: 219, text: "[" },
  w: { code: "KeyW", vk: 87, text: "w" },
  V: { code: "KeyV", vk: 86, text: "V" },
};

const failures: string[] = [];
let passed = 0;

async function check(name: string, run: () => Promise<void>): Promise<void> {
  try {
    await run();
    passed++;
    console.log(`  ok    ${name}`);
  } catch (e) {
    failures.push(name);
    console.log(`  FAIL  ${name}\n        ${(e as Error).message}`);
  }
}

function expect(what: string, actual: unknown, want: unknown): void {
  if (JSON.stringify(actual) !== JSON.stringify(want)) {
    throw new Error(`${what}: expected ${JSON.stringify(want)}, got ${JSON.stringify(actual)}`);
  }
}

try {
  let target: { webSocketDebuggerUrl: string } | undefined;
  for (let i = 0; i < 60 && !target; i++) {
    await wait(250);
    target = await fetch(`http://127.0.0.1:${PORT}/json/list`)
      .then((r) => r.json() as Promise<{ type: string; webSocketDebuggerUrl: string }[]>)
      .then((list) => list.find((t) => t.type === "page"))
      .catch(() => undefined);
  }
  if (!target) throw new Error("Chrome never opened its debugging port");

  const dt = await Devtools.connect(target.webSocketDebuggerUrl);
  const errors: string[] = [];
  dt.on("Runtime.exceptionThrown", (p) =>
    errors.push(p.exceptionDetails?.exception?.description ?? p.exceptionDetails?.text ?? "exception"),
  );
  dt.on("Runtime.consoleAPICalled", (p) => {
    if (p.type === "error") errors.push(p.args.map((a: { value?: unknown; description?: string }) => a.value ?? a.description).join(" "));
  });
  await dt.send("Page.enable");
  await dt.send("Runtime.enable");

  const page = <T>(expression: string): Promise<T> => dt.eval<T>(expression);
  /** Wait until an expression in the page is truthy, and say which one never was. */
  const until = async (expression: string, ms = 10_000): Promise<void> => {
    for (let t = 0; t < ms; t += 100) {
      if (await page<boolean>(`Boolean(${expression})`)) return;
      await wait(100);
    }
    throw new Error(`never true: ${expression}`);
  };
  const key = async (name: string, modifiers = 0): Promise<void> => {
    const k = KEYS[name]!;
    const base = { key: name, code: k.code, windowsVirtualKeyCode: k.vk, nativeVirtualKeyCode: k.vk, modifiers };
    await dt.send("Input.dispatchKeyEvent", { type: k.text ? "keyDown" : "rawKeyDown", text: k.text, ...base });
    await dt.send("Input.dispatchKeyEvent", { type: "keyUp", ...base });
    await wait(150);
  };
  const focused = (): Promise<string | null> => page(`document.activeElement?.getAttribute("aria-label") ?? null`);
  const nodes = (): Promise<string[]> =>
    page(`[...document.querySelectorAll(".react-flow__node")].map((n) => n.getAttribute("aria-label"))`);
  const focusNode = (prefix: string): Promise<void> =>
    page(`[...document.querySelectorAll(".react-flow__node")].find((n) => (n.getAttribute("aria-label") || "").startsWith(${JSON.stringify(prefix)})).focus()`);
  const resources = (): Promise<{ name: string; initiator: string }[]> =>
    page(`performance.getEntriesByType("resource").map((e) => ({ name: e.name, initiator: e.initiatorType }))`);

  await dt.send("Page.navigate", { url: SITE });
  // Measured, not merely rendered: React Flow keeps a node invisible until it
  // knows its size, and an invisible node cannot take focus.
  await until(`[...document.querySelectorAll(".react-flow__node")].some((n) => getComputedStyle(n).visibility === "visible")`, 30_000);

  await check("the engine is fetched once, by the preload in the page", async () => {
    const wasm = (await resources()).filter((r) => r.name.endsWith(".wasm"));
    expect("fetches of the engine", wasm.map((r) => r.initiator), ["link"]);
  });

  await check("the volume view is not part of the first load", async () => {
    const view = (await resources()).filter((r) => /View3D/.test(r.name));
    expect("volume view chunks", view.length, 0);
  });

  await check("every block and wire has a name, and none of them says undefined", async () => {
    await key("[");
    await key("[");
    await until(`document.querySelectorAll(".react-flow__edge").length > 0`);
    const labels = await page<(string | null)[]>(
      `[...document.querySelectorAll(".react-flow__node, .react-flow__edge")].map((n) => n.getAttribute("aria-label"))`,
    );
    const bad = labels.filter((l) => !l || /undefined|NaN|null/.test(l));
    expect("unnamed or broken labels", bad, []);
    expect("the sheet itself", await page(`document.querySelector(".react-flow").getAttribute("aria-label")`), "Schematic of nano-sort");
  });

  await check("Enter selects a block, Enter again opens it, and focus goes inside", async () => {
    await focusNode("Transformer block");
    await key("Enter");
    await key("Enter");
    await until(`document.querySelectorAll(".react-flow__node").length === 3`);
    await until(`(document.activeElement?.getAttribute("aria-label") || "").startsWith("_in")`);
  });

  await check("Escape on a selected block deselects it and stays", async () => {
    await key("Enter");
    await until(`document.activeElement?.classList.contains("selected")`);
    await key("Escape");
    expect("blocks on the sheet", (await nodes()).length, 3);
    await until(`!document.querySelector(".react-flow__node.selected")`);
  });

  await check("Escape with nothing selected comes back out onto the block that was left", async () => {
    await key("Escape");
    await until(`document.querySelectorAll(".react-flow__node").length === 7`);
    await until(`(document.activeElement?.getAttribute("aria-label") || "").startsWith("Transformer block")`);
  });

  await check("Tab moves between blocks", async () => {
    const before = await focused();
    await key("Tab");
    const after = await focused();
    if (!after || after === before) throw new Error(`focus went from ${before} to ${after}`);
  });

  await check("Enter on a button presses it and opens nothing", async () => {
    await focusNode("Transformer block");
    await key("Enter"); // select it, so an Enter anywhere could open it
    await page(`document.querySelector("button[aria-label='Fit to window']")?.focus()`);
    await key("Enter");
    expect("blocks on the sheet", (await nodes()).length, 7);
    await key("Escape");
  });

  await check("the volume view loads on demand, with the traced values", async () => {
    await page(`document.activeElement?.blur()`);
    await key("V", 8); // Shift
    await until(`document.querySelector(".app__center canvas")`, 15_000);
    await until(`[...document.querySelectorAll("div")].some((d) => d.textContent?.startsWith("real values"))`, 15_000);
    const view = (await resources()).filter((r) => /View3D/.test(r.name));
    expect("volume view chunks", view.length, 1);
    expect(
      "the picture says what it is",
      await page<boolean>(`(document.querySelector("[role=img]")?.getAttribute("aria-label") || "").includes("C B A B A C")`),
      true,
    );
    await key("V", 8);
    await until(`document.querySelector(".react-flow")`);
  });

  await check("the walkthrough quotes the run", async () => {
    await page(`document.activeElement?.blur()`);
    await key("w");
    await until(`document.querySelector(".wt")`);
    for (let i = 0; i < 1; i++) await page(`[...document.querySelectorAll(".wt button")].find((b) => b.textContent.trim() === "Next").click()`);
    await until(`document.querySelector(".wt")?.textContent.includes("C B A B A C")`);
    await key("Escape");
  });

  await check("nothing threw, and nothing was logged as an error", async () => {
    expect("errors", errors, []);
  });

  dt.close();
} finally {
  browser.kill();
  server.stop();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}

console.log(`\n  ${passed} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
