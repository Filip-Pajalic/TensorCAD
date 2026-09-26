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

import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { $ } from "bun";
import { generateTorch, getPreset, loadEngine } from "@tensor-cad/engine/node";
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

// --- a trace to load ----------------------------------------------------------

/**
 * A design that is not nano-sort, and a trace of it, as files on disk.
 *
 * The committed trace shows on nano-sort by itself, so loading it proves
 * nothing about loading. nano-sort with a ReLU is a different model with the
 * same shapes: the committed numbers relabelled with *its* fingerprint are a
 * trace only that design matches, and every box still has a tensor to draw.
 * Made here rather than committed, so the fingerprint follows the generator.
 */
await loadEngine();
const fixtures = await mkdtemp(join(tmpdir(), "tensorcad-trace-load-"));
const variant = structuredClone(getPreset("nano-sort"));
variant.meta.name = "nano-sort-relu";
const stack = variant.graph.nodes.find((n) => n.id === "layers")!;
const inner = stack.graph!.nodes.find((n) => n.id === "block")!;
inner.params = { ...inner.params, act: "relu" };
const variantModel = generateTorch(variant).files.find((f) => f.path === "model.py")!.contents;
const committedTrace = JSON.parse(await readFile(join(ROOT, "packages/ui/src/three/traces/nano-sort.json"), "utf8"));
const DESIGN_FILE = join(fixtures, "nano-sort-relu.tensorcad.json");
const TRACE_FILE = join(fixtures, "trace.json");
await writeFile(DESIGN_FILE, JSON.stringify(variant, null, 2));
await writeFile(
  TRACE_FILE,
  JSON.stringify({
    ...committedTrace,
    design: "nano-sort-relu",
    model_sha256: createHash("sha256").update(variantModel, "utf8").digest("hex"),
  }),
);

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
    // A CI runner's /dev/shm is small, and Chrome that runs out of it there
    // crashes at start without a word. Temporary files instead.
    "--disable-dev-shm-usage",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${profile}`,
    "--window-size=1440,900",
    "about:blank",
  ],
  // Kept rather than ignored, so a Chrome that never starts can say why.
  { stdout: "ignore", stderr: "pipe" },
);

// Drained as it comes, keeping only the tail: a pipe nobody reads fills, and a
// Chrome blocked writing to it stops answering halfway through the checks.
let chromeSaid = "";
void (async () => {
  const text = new TextDecoder();
  for await (const chunk of browser.stderr) chromeSaid = (chromeSaid + text.decode(chunk)).slice(-4000);
})().catch(() => {});

/** Keys by the name the page sees, with what the protocol needs to send them. */
const KEYS: Record<string, { code: string; vk: number; text?: string }> = {
  Enter: { code: "Enter", vk: 13, text: "\r" },
  Escape: { code: "Escape", vk: 27 },
  Tab: { code: "Tab", vk: 9 },
  "[": { code: "BracketLeft", vk: 219, text: "[" },
  w: { code: "KeyW", vk: 87, text: "w" },
  V: { code: "KeyV", vk: 86, text: "V" },
  f: { code: "KeyF", vk: 70, text: "f" },
  "=": { code: "Equal", vk: 187 },
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
  // A minute, not fifteen seconds: a first start with a fresh profile on a
  // busy runner has taken longer than that, and the cost of waiting is only
  // paid when it is slow.
  for (let i = 0; i < 240 && !target && browser.exitCode === null; i++) {
    await wait(250);
    target = await fetch(`http://127.0.0.1:${PORT}/json/list`)
      .then((r) => r.json() as Promise<{ type: string; webSocketDebuggerUrl: string }[]>)
      .then((list) => list.find((t) => t.type === "page"))
      .catch(() => undefined);
  }
  if (!target) {
    browser.kill();
    const said = chromeSaid.trim().split("\n").slice(-15).join("\n");
    throw new Error(
      `Chrome never opened its debugging port${browser.exitCode !== null ? ` (it exited with ${browser.exitCode})` : ""}` +
        (said ? `. It said:\n${said}` : ""),
    );
  }

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

  await check("a first visit: nothing over the drawing, four fields, four tabs, and Start here", async () => {
    expect(
      "what covers the sheet",
      await page(`({
        key: !!document.querySelector(".key:not(.key--shut)"),
        titleBlock: !!document.querySelector(".title-block"),
        minimap: !!document.querySelector(".react-flow__minimap"),
      })`),
      { key: false, titleBlock: false, minimap: false },
    );
    expect(
      "the operating point",
      await page(`[...document.querySelectorAll(".op__label")].map((l) => l.textContent)`),
      ["batch", "sequence", "device", "GPUs"],
    );
    expect(
      "the tabs",
      await page(`[...document.querySelectorAll(".panel--edit [role=tab]")].map((t) => t.textContent)`),
      ["Inspector", "Symbols", "Training", "History"],
    );
    await page(`document.querySelector("[data-testid=start-here]").click()`);
    await until(`document.querySelector(".wt")`);
    await page(`document.querySelector(".wt__shut").click()`);
    await until(`!document.querySelector(".wt")`);

    // And the toolbar holds on a thirteen-inch laptop: at 1280 it once pushed
    // Share off the right-hand edge.
    await dt.send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 800, deviceScaleFactor: 1, mobile: false });
    await wait(300);
    const share = await page<number>(`document.querySelector("[data-testid=share]").getBoundingClientRect().right`);
    await dt.send("Emulation.clearDeviceMetricsOverride");
    await wait(300);
    if (share > 1280) throw new Error(`Share ends at ${share}, past a 1280-wide window`);
  });

  await check("More keeps how a run is set up, and says what of it has changed", async () => {
    const more = `document.querySelector("[data-testid=operating-more]")`;
    expect("shut, it says what is inside", await page(`${more}.textContent`), "▸Moreprecision, optimizer, parallelism");
    await page(`${more}.click()`);
    const zero = `[...document.querySelectorAll(".op__field")].find((f) => f.querySelector(".op__label")?.textContent === "ZeRO")?.querySelector("select")`;
    await until(zero);
    await page(`(() => {
      const select = ${zero};
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value").set.call(select, "3");
      select.dispatchEvent(new Event("change", { bubbles: true }));
    })()`);
    await page(`${more}.click()`);
    await until(`${more}.textContent.endsWith("ZeRO 3")`);
    await page(`[...document.querySelectorAll(".op__head .linkish, .op__head button")].find((b) => b.textContent === "reset")?.click()`);
    // Reset is on the open header only; open the operating point's own fold if
    // it had been shut, and put the defaults back for the checks after this.
    await until(`!${more}.textContent.includes("ZeRO")`);
  });

  await check("a block drawn inside an unfolded frame is inspected where it is", async () => {
    // Opened until the attention is drawn inside its frames, however deep the
    // sheet starts.
    const drawn = `[...document.querySelectorAll(".react-flow__node")].some((n) => (n.getAttribute("aria-label") || "").startsWith("attn, grouped-query attention"))`;
    for (let i = 0; i < 4 && !(await page<boolean>(drawn)); i++) {
      await page(`document.querySelector("button[aria-label='Open one more level']").click()`);
      await wait(400);
    }
    await until(drawn);
    await focusNode("attn, grouped-query attention");
    await key("Enter");
    await until(`document.querySelector(".inspector__type")?.textContent === "grouped-query attention"`);
    const where = await page<string>(`document.querySelector("[data-testid=inspector-where]")?.textContent ?? ""`);
    if (!where.includes("read-only")) throw new Error(`the inspector said: ${where}`);
    expect(
      "its fields, which a built-in block's interior cannot change",
      await page(`[...document.querySelectorAll(".inspector .param input, .inspector .param select")].every((f) => f.disabled)`),
      true,
    );
    await key("Escape");
    await until(`document.querySelector("[data-testid=inspector-start]")`);
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

  await check("f fits the sheet, with nothing else happening", async () => {
    // A fit request waited for some other render to come along, so on a sheet
    // nobody was changing, pressing f did nothing.
    const scale = () =>
      page<number>(`Number(document.querySelector(".react-flow__viewport").style.transform.split("scale(")[1]?.split(")")[0])`);
    await page(`document.activeElement?.blur()`);
    for (let i = 0; i < 4; i++) await key("=", 2); // Ctrl+=
    await wait(600);
    const zoomed = await scale();
    if (!(zoomed > 1.1)) throw new Error(`zooming in did not take: ${zoomed}`);
    await key("f");
    await wait(800);
    const fitted = await scale();
    if (!(fitted <= 1.1)) throw new Error(`f left the zoom at ${fitted}`);
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

  // Files as a person would give them: through the inputs File > Open and
  // File > Load a trace click.
  const setFile = async (input: string, path: string): Promise<void> => {
    const { root } = await dt.send<{ root: { nodeId: number } }>("DOM.getDocument", { depth: 0 });
    const { nodeId } = await dt.send<{ nodeId: number }>("DOM.querySelector", { nodeId: root.nodeId, selector: `#${input}` });
    await dt.send("DOM.setFileInputFiles", { nodeId, files: [path] });
  };
  const realValues = `[...document.querySelectorAll("div")].some((d) => d.textContent?.startsWith("real values"))`;

  await check("a trace loaded from a file shows on the design it was made from", async () => {
    await setFile("tensorcad-open-input", DESIGN_FILE);
    await until(`document.body.innerText.includes("nano-sort-relu")`);
    await setFile("tensorcad-trace-input", TRACE_FILE);
    // textContent, not innerText: the toolbar hides its status line until a
    // whole sentence fits, and a narrow headless window may not have the room.
    await until(`document.body.textContent.includes("Loaded a trace of nano-sort-relu") && document.body.textContent.includes("Open the volume view")`);
    await page(`document.activeElement?.blur()`);
    await key("V", 8); // Shift
    await until(realValues, 15_000);
    await key("V", 8);
    await until(`document.querySelector(".react-flow")`);
  });

  await check("and is still there after a reload, with only the design opened again", async () => {
    // The trace went onto the shelf when it was loaded. A new page has nothing
    // in memory; opening the design is all it is given.
    await dt.send("Page.navigate", { url: SITE });
    await until(`[...document.querySelectorAll(".react-flow__node")].some((n) => getComputedStyle(n).visibility === "visible")`, 30_000);
    await setFile("tensorcad-open-input", DESIGN_FILE);
    await until(`document.body.innerText.includes("nano-sort-relu")`);
    await page(`document.activeElement?.blur()`);
    await key("V", 8);
    await until(realValues, 15_000);
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

  await check("a mask typed into the inspector changes what attention costs", async () => {
    // The walkthrough sets how far the sheet is unfolded whenever the design
    // changes, and takes the selection with it; it is not what is being
    // checked, so it is closed.
    await page(`document.querySelector(".wt__shut")?.click()`);
    await until(`!document.querySelector(".wt")`);
    await page(`document.activeElement?.blur()`);
    if ((await nodes()).length !== 7) {
      await key("[");
      await key("[");
    }
    await until(`document.querySelectorAll(".react-flow__node").length === 7`);
    await focusNode("Transformer block");
    await key("Enter");
    await key("Enter");
    await until(`(document.activeElement?.getAttribute("aria-label") || "").startsWith("_in")`);
    await focusNode("block,");
    await key("Enter");
    // A field leads with what it is called and keeps its name beside it, and
    // the rare ones, the mask among them, wait under Advanced on a block that
    // has changed none of them.
    const advanced = `document.querySelector("[data-testid=advanced]")`;
    await until(`${advanced}?.getAttribute("aria-expanded") === "false"`);
    expect(
      "a field's label and name",
      await page(`[...document.querySelector(".inspector .param").querySelectorAll(".param__label, .param__name")].map((e) => e.textContent)`),
      ["Model width", "d_model"],
    );
    if (await page(`!!document.querySelector("[data-testid=mask-preview]")`)) {
      throw new Error("the mask was on screen before Advanced was opened");
    }
    await page(`${advanced}.click()`);
    await until(`document.querySelector("[data-testid=mask-preview]")`);

    const attention = `[...document.querySelectorAll("tr")].find((r) => r.cells[0]?.textContent.startsWith("forward, attention"))?.cells[1].textContent`;
    const density = `document.querySelector("[data-testid=mask-density]")?.textContent`;
    // nano-sort is eleven positions long: causal keeps 5.5 keys a query,
    // 3 layers × 4 × 5.5 × 3 heads × 16 wide.
    expect("causal attention", await page(attention), "3.17 kFLOP");
    expect("causal density", await page(density), "50.0%");

    // Written over a design symbol, which the field then shows resolved.
    const field = `[...document.querySelectorAll(".param")].find((p) => p.querySelector(".param__name")?.textContent === "mask")?.querySelector("input")`;
    await page(`(${field}).focus(), (${field}).select()`);
    await dt.send("Input.insertText", { text: "q - kv < L - 1" });
    await key("Enter");
    await until(`document.body.textContent.includes("= q - kv < 2")`);
    // Now a query sees itself and the one before: 21 of the 66 scores causal
    // keeps, so 5.5 × 21 / 66 = 1.75 keys.
    await until(`${attention} === "1.01 kFLOP"`);
    expect("masked density", await page(density), "15.9%");
    expect(
      "the mask as the kernel is given it",
      await page(`[...document.querySelectorAll("[data-testid=mask-preview] .mask__expr")].map((e) => e.textContent)`),
      ["kv <= q and q - kv < 2"],
    );
  });

  await check("Share makes a link that opens the same design in a new page", async () => {
    await page(`document.activeElement?.blur()`);
    // Renamed, so the design that opens is the one in the link and not one
    // that happened to be there.
    await page(`(() => {
      const input = document.querySelector('[aria-label="Design name"]');
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(input, "shared-in-a-link");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    })()`);
    await page(`document.querySelector("[data-testid=share]").click()`);
    await until(`document.querySelector("[data-testid=share-link]")?.value`);
    const link = await page<string>(`document.querySelector("[data-testid=share-link]").value`);
    // Deflated by the browser, on this deployment's own origin.
    if (!link.startsWith(`${SITE}/#design=z`)) throw new Error(`the link was ${link.slice(0, 80)}`);
    await key("Escape");
    // Through a blank page: a navigation that changes only the fragment would
    // not load the page again, and loading is when a link is read.
    await dt.send("Page.navigate", { url: "about:blank" });
    await wait(200);
    await dt.send("Page.navigate", { url: link });
    await until(`document.querySelector('[aria-label="Design name"]')?.value === "shared-in-a-link"`, 30_000);
    expect("the address bar, once it has opened", await page(`location.hash`), "");
  });

  await check("nothing threw, and nothing was logged as an error", async () => {
    expect("errors", errors, []);
  });

  dt.close();
} finally {
  browser.kill();
  server.stop();
  await rm(fixtures, { recursive: true, force: true }).catch(() => {});
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}

console.log(`\n  ${passed} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
