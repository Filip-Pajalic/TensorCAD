/**
 * The bridge between the Go shell and the editor.
 *
 * Go owns the operating system: files, dialogs, menus, the theme, and the
 * Python jobs. The editor owns the document. This module is the only place the
 * two meet, so neither side has to know about the other's internals.
 *
 * Every handler here is written so the browser build keeps working: when the
 * Wails runtime is absent, `available()` is false and the editor falls back to
 * downloads and `prefers-color-scheme`.
 */

import { Events } from "@wailsio/runtime";
import {
  loadTrace,
  parseRun,
  runCommand,
  setSystemTheme,
  setThemePreference,
  useEditor,
  useRuns,
  parseDoc,
  serializeDoc,
} from "@tensor-cad/ui";
import { engine, generateTorch, resolveSymbols } from "@tensor-cad/ui/engine";
import {
  DesignService,
  RuntimeService,
  WorkspaceService,
} from "../bindings/github.com/tensorcad/desktop/services/index.js";

/** True when running inside the desktop shell rather than a plain browser. */
export function available(): boolean {
  return typeof window !== "undefined" && "_wails" in window;
}

/** Where the open design came from, so Save can write back without asking. */
let currentPath = "";

export function currentDesignPath(): string {
  return currentPath;
}

async function openDesign(): Promise<void> {
  const file = await DesignService.Open();
  if (!file) return;
  try {
    useEditor.getState().setDoc(parseDoc(file.contents), `Opened ${file.name}`);
    currentPath = file.path;
  } catch (e) {
    useEditor.getState().setStatus(`Could not read ${file.name}: ${(e as Error).message}`);
  }
}

async function saveDesign(askWhere: boolean): Promise<void> {
  const { doc } = useEditor.getState();
  const path = askWhere ? "" : currentPath;
  try {
    const written = await DesignService.Save(path, serializeDoc(doc), doc.meta.name);
    if (!written) return; // cancelled
    currentPath = written;
    useEditor.getState().setStatus(`Saved to ${written}`);
  } catch (e) {
    useEditor.getState().setStatus(`Could not save: ${(e as Error).message}`);
  }
}

async function generateCode(): Promise<void> {
  const { doc } = useEditor.getState();
  const generated = generateTorch(doc);
  if (generated.warnings.length > 0) {
    useEditor.getState().setStatus(`Generated with ${generated.warnings.length} warning(s)`);
  }
  try {
    const directory = await WorkspaceService.ChooseDirectory("Where should the generated model go?");
    if (!directory) return;
    const result = await WorkspaceService.Write(
      directory,
      doc.meta.name,
      generated.files.map((f) => ({ path: f.path, contents: f.contents })),
    );
    if (!result) return;
    useEditor
      .getState()
      .setStatus(`Wrote ${result.written?.length ?? 0} file(s) to ${result.directory}`);
  } catch (e) {
    useEditor.getState().setStatus(`Could not write the model: ${(e as Error).message}`);
  }
}

// --- design jobs -------------------------------------------------------------
//
// Verify, smoke-train and trace all run the open design through the Python
// runtime, which is the one thing the browser build cannot do. Each generates
// the model here, hands the files to the shell to run in a folder of its own,
// and says what came of it when the job ends — in a sentence, on the Runs
// chart, or in the volume view — rather than as an exit code.

type Finish = (exitCode: number, result: Record<string, unknown> | undefined) => Promise<void>;

/** Jobs in flight, by id, and what to do with each one's result. */
const jobs = new Map<string, Finish>();

const count = (n: unknown): string => (typeof n === "number" ? n.toLocaleString("en-US") : "?");

/** Why a job failed, from the object the runtime prints last, or its exit code. */
function why(exitCode: number, result: Record<string, unknown> | undefined): string {
  return typeof result?.error === "string" ? result.error : `the runtime exited with ${exitCode}`;
}

function generatedFiles(): { name: string; files: { path: string; contents: string }[] } {
  const { doc } = useEditor.getState();
  return {
    name: doc.meta.name,
    files: generateTorch(doc).files.map((f) => ({ path: f.path, contents: f.contents })),
  };
}

async function launch(
  what: string,
  start: () => Promise<{ id: string } | null>,
  finish: Finish,
  running: string,
): Promise<void> {
  const state = useEditor.getState();
  try {
    const started = await start();
    if (!started) return;
    jobs.set(started.id, finish);
    state.setStatus(running);
  } catch (e) {
    state.setStatus(`Could not start ${what}: ${(e as Error).message}`);
  }
}

/** Build the design in PyTorch and hold its parameter count to the design's. */
async function verifyDesign(): Promise<void> {
  const { name, files } = generatedFiles();
  await launch(
    "the check against PyTorch",
    () => RuntimeService.VerifyDesign(name, files),
    async (exitCode, result) => {
      const state = useEditor.getState();
      if (!result || typeof result.params !== "number") {
        state.setStatus(`Could not verify ${name}: ${why(exitCode, result)}`);
        return;
      }
      const counted =
        result.matches === true
          ? `PyTorch counts ${count(result.params)} parameters, the same as the design`
          : `PyTorch counts ${count(result.params)} parameters and the design says ${count(result.expected)}`;
      const ran = typeof result.forward === "string" && result.forward !== "ok" ? `; forward pass ${result.forward}` : "; a forward pass ran";
      const exported = result.export_ok === true ? "; it exports" : result.export_ok === false ? "; it does not export" : "";
      state.setStatus(`Verified ${name}: ${counted}${ran}${exported}.`);
    },
    `Building ${name} in PyTorch to check it against the design`,
  );
}

/**
 * The largest design a smoke run is started for. Past this the point of a
 * smoke run — a loss curve in a minute — is gone, and on a desktop GPU the
 * weights alone stop fitting. The Ladder shrinks a design to a size that does.
 */
const SMOKE_LIMIT = 300e6;

/** Train the design briefly and put its loss curve on the Runs chart. */
async function smokeTrain(): Promise<void> {
  const state = useEditor.getState();
  const { doc } = state;
  const params = engine().analyze(doc, {}).params.total;
  if (params > SMOKE_LIMIT) {
    state.setStatus(
      `${doc.meta.name} has ${count(params)} parameters, too many for a smoke run here. Scale it down with the Ladder first.`,
    );
    return;
  }
  // A few hundred steps at the sequence the readout is measuring, capped where
  // a smoke run stops being quick.
  const designT = Number(resolveSymbols(doc).values.T ?? 256);
  const seq = Math.max(8, Math.min(state.operating.T ?? designT, 256));
  const { name, files } = generatedFiles();
  await launch(
    "the smoke run",
    () => RuntimeService.SmokeTrainDesign(name, files, 200, 8, seq),
    async (exitCode, result) => {
      const now = useEditor.getState();
      const record = typeof result?.record_file === "string" ? result.record_file : "";
      if (exitCode !== 0 || !record) {
        now.setStatus(`Could not train ${name}: ${why(exitCode, result)}`);
        return;
      }
      try {
        const file = record.split(/[\\/]/).pop() ?? `${name}.json`;
        useRuns.getState().add([parseRun(file, await RuntimeService.ReadResult(record))]);
        now.setRightTab("runs");
        const loss = (v: unknown): string => (typeof v === "number" ? v.toFixed(3) : "?");
        const data = result?.data_source === "synthetic" ? "synthetic tokens" : `${String(result?.data_source ?? "a corpus")}`;
        now.setStatus(
          `Trained ${name} for ${count(result?.steps)} steps on ${data}: loss ${loss(result?.initial_loss)} to ${loss(result?.final_loss)}, ` +
            `${count(Math.round(Number(result?.tokens_per_second ?? 0)))} tokens a second on ${String(result?.device ?? "?")}. It is on the Runs chart.`,
        );
      } catch (e) {
        now.setStatus(`The run finished but its record could not be read: ${(e as Error).message}`);
      }
    },
    `Training ${name} for 200 steps at ${seq} tokens`,
  );
}

/**
 * Run the open design and show what it computes.
 *
 * The trace comes back through the same `loadTrace` a file opened with File >
 * Load a trace goes through, so the two cannot disagree about which design it
 * describes.
 */
async function traceDesign(): Promise<void> {
  const { name, files } = generatedFiles();
  let out = "";
  await launch(
    "the trace",
    async () => {
      const started = await RuntimeService.Trace(name, files);
      out = started?.out ?? "";
      return started;
    },
    async (exitCode, result) => {
      const state = useEditor.getState();
      if (exitCode !== 0) {
        // The runtime says why in the object it prints last: too large, no
        // token embedding, PyTorch missing.
        state.setStatus(`Could not trace ${name}: ${why(exitCode, result)}`);
        return;
      }
      try {
        const { message } = await loadTrace(await RuntimeService.ReadResult(out), state.doc);
        state.setStatus(message);
        // A trace is something to look at: go to where it is drawn, at a
        // length it covers.
        const positions = Array.isArray(result?.sequence) ? result.sequence.length : 0;
        if (positions > 0 && (state.operating.T ?? Number.POSITIVE_INFINITY) > positions) {
          state.setOperating({ T: positions });
        }
        state.setViewMode("volume");
      } catch (e) {
        state.setStatus(`The trace ran but could not be read: ${(e as Error).message}`);
      }
    },
    `Tracing ${name}: a small vocabulary is trained to sort first, which takes a few seconds`,
  );
}

/** Commands the native menu can send. The editor decides what each one means. */
const COMMANDS: Record<string, () => void | Promise<void>> = {
  new: () => {
    useEditor.getState().newDoc();
    currentPath = "";
  },
  open: openDesign,
  save: () => saveDesign(false),
  "save-as": () => saveDesign(true),
  generate: generateCode,

  // Everything the editor can already do is defined once, in its command list.
  // The native menu names a command rather than reimplementing it, so a Wails
  // menu item and its keyboard shortcut cannot drift apart.
  undo: () => runCommand("edit.undo"),
  redo: () => runCommand("edit.redo"),
  lock: () => runCommand("edit.lock"),
  delete: () => runCommand("edit.delete"),
  duplicate: () => runCommand("edit.duplicate"),

  fit: () => runCommand("view.fit"),
  arrange: () => runCommand("view.arrange"),
  "toggle-callouts": () => runCommand("view.callouts"),
  "toggle-shape-mode": () => runCommand("view.shapes"),
  "detail-in": () => runCommand("view.detailIn"),
  "detail-out": () => runCommand("view.detailOut"),
  settings: () => runCommand("help.settings"),
  shortcuts: () => runCommand("help.shortcuts"),

  "theme:light": () => setThemePreference("light"),
  "theme:dark": () => setThemePreference("dark"),
  "theme:system": () => setThemePreference("system"),

  validate: () => runCommand("panel.rules"),
  verify: verifyDesign,
  "smoke-train": smokeTrain,
  trace: traceDesign,
};

/**
 * Connect the shell. Safe to call in a browser, where it does nothing beyond
 * leaving the web fallbacks in place.
 */
export function connect(): void {
  if (!available()) return;

  Events.On("menu:command", (event: { data: string }) => {
    const run = COMMANDS[event.data];
    if (run) void run();
  });

  Events.On("theme:changed", (event) => {
    setSystemTheme(event.data === "dark" ? "dark" : "light");
  });

  Events.On("runtime:line", (event: { data: { stream: string; text: string } }) => {
    // Progress lines are noisy; only the summary ones reach the status bar.
    if (event.data.stream === "stdout") return;
    const text = event.data.text.trim();
    if (text) useEditor.getState().setStatus(text.slice(0, 160));
  });

  // Typed by the event Go registers, so a field renamed there is an error here.
  Events.On("runtime:done", (event) => {
    // A design job reports for itself, with what came of it rather than how
    // long it took.
    const finish = jobs.get(event.data.id);
    if (finish) {
      jobs.delete(event.data.id);
      void finish(event.data.exitCode, event.data.result ?? undefined);
      return;
    }
    const { exitCode, seconds } = event.data;
    useEditor
      .getState()
      .setStatus(exitCode === 0 ? `Finished in ${seconds.toFixed(1)}s` : `Job failed with exit code ${exitCode}`);
  });

  // Report what the Python side can do, so the user learns it now rather than
  // when they press a button.
  void RuntimeService.Probe().then((env) => {
    if (env.detail) useEditor.getState().setStatus(env.detail);
  });
}
