/**
 * The sheet as a vector, from a headless browser.
 *
 *   bun run --cwd packages/ui dev      # in one terminal
 *   bun run scripts/export-svg.ts      # in another
 *
 * The editor has `File > Export the sheet as SVG`, which is where a person
 * reaches for this. This is the same exporter driven without a person, so a
 * figure for a paper can be regenerated the way the screenshots are — and so
 * there is somewhere to check that the wires come out, which is the half of
 * the export that is not obvious.
 *
 * It drives the same Chrome over the same DevTools connection the screenshot
 * script uses, and imports its driver rather than keeping a second copy.
 */

import { mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { CHROME, Devtools, PORT, URL_BASE, wait } from "./screenshots.js";

const ROOT = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const OUT = join(ROOT, "docs", "images");
const WIDTH = 1440;
const HEIGHT = 900;

/** Which preset, and how many levels of container to draw open. */
const PRESET = process.argv[2] ?? "llama-3-8b";
const DETAIL = Number(process.argv[3] ?? 1);

async function main(): Promise<void> {
  const reachable = await fetch(URL_BASE)
    .then((r) => r.ok)
    .catch(() => false);
  if (!reachable) {
    console.error(`Nothing is serving ${URL_BASE}. Start it with:\n  bun run --cwd packages/ui dev`);
    process.exit(1);
  }

  let chrome = "";
  for (const candidate of CHROME) {
    if (await Bun.file(candidate).exists()) {
      chrome = candidate;
      break;
    }
  }
  if (!chrome) {
    console.error("No Chrome found. Set one of:\n  " + CHROME.join("\n  "));
    process.exit(1);
  }

  const profile = join(ROOT, "dist", "chrome-svg-profile");
  await rm(profile, { recursive: true, force: true });
  const browser = Bun.spawn(
    [
      chrome,
      "--headless=new",
      "--disable-gpu",
      "--no-sandbox",
      "--hide-scrollbars",
      `--remote-debugging-port=${PORT + 1}`,
      `--user-data-dir=${profile}`,
      `--window-size=${WIDTH},${HEIGHT}`,
      "about:blank",
    ],
    { stdout: "ignore", stderr: "ignore" },
  );

  try {
    let target: { webSocketDebuggerUrl: string } | undefined;
    for (let i = 0; i < 60 && !target; i++) {
      target = await fetch(`http://127.0.0.1:${PORT + 1}/json/list`)
        .then((r) => r.json() as Promise<{ webSocketDebuggerUrl: string; type: string }[]>)
        .then((list) => list.find((t) => t.type === "page"))
        .catch(() => undefined);
      if (!target) await wait(250);
    }
    if (!target) throw new Error("Chrome never accepted a connection");

    const dev = await Devtools.connect(target.webSocketDebuggerUrl);
    await dev.send("Page.enable");
    await dev.send("Runtime.enable");
    await dev.send("Page.navigate", { url: URL_BASE });
    // The engine loads before the first frame, the layout runs after it.
    await wait(5000);

    await dev.eval<string>(`(async () => {
      const ed = await import("/src/state/store.ts");
      ed.useEditor.getState().loadPreset(${JSON.stringify(PRESET)});
      ed.useEditor.getState().setDetail(${DETAIL});
      return "ok";
    })()`);
    await wait(3500);

    const svg = await dev.eval<string | null>(`(async () => {
      const { sheetToSvg } = await import("/src/canvas/svg.ts");
      const root = document.querySelector(".react-flow");
      return root ? (sheetToSvg(root, ${JSON.stringify(PRESET)}) ?? null) : null;
    })()`);
    if (!svg) throw new Error("the sheet exported nothing; is anything drawn?");

    const wires = (svg.match(/<path /g) ?? []).length;
    const blocks = (svg.match(/<rect /g) ?? []).length - 1;
    if (wires === 0) {
      // Worth failing over: a schematic with no nets in it is not a schematic,
      // and silently writing one would be worse than not writing it.
      throw new Error(`${blocks} blocks and no wires — the edges were not rendered when this ran`);
    }

    await mkdir(OUT, { recursive: true });
    const file = join(OUT, `${PRESET}-sheet.svg`);
    await writeFile(file, svg + "\n", "utf8");
    console.log(
      `  docs/images/${PRESET}-sheet.svg  ${Math.round(svg.length / 1024)} KB  ${blocks} blocks, ${wires} wires`,
    );
    dev.close();
  } finally {
    browser.kill();
  }
}

await main();
