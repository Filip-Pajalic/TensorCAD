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
  runCommand,
  setSystemTheme,
  setThemePreference,
  useEditor,
  parseDoc,
  serializeDoc,
  type ResolvedTheme,
} from "@tensor-cad/ui";
import { generateTorch } from "@tensor-cad/ui/engine";
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
    useEditor
      .getState()
      .setStatus(`Wrote ${result.written.length} file(s) to ${result.directory}`);
  } catch (e) {
    useEditor.getState().setStatus(`Could not write the model: ${(e as Error).message}`);
  }
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
  verify: () => useEditor.getState().setStatus("Generate the model first, then verify it."),
  "smoke-train": () => useEditor.getState().setStatus("Generate the model first, then train it."),
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

  Events.On("theme:changed", (event: { data: ResolvedTheme }) => {
    setSystemTheme(event.data === "dark" ? "dark" : "light");
  });

  Events.On("runtime:line", (event: { data: { stream: string; text: string } }) => {
    // Progress lines are noisy; only the summary ones reach the status bar.
    if (event.data.stream === "stdout") return;
    const text = event.data.text.trim();
    if (text) useEditor.getState().setStatus(text.slice(0, 160));
  });

  Events.On("runtime:done", (event: { data: { exitCode: number; seconds: number } }) => {
    const { exitCode, seconds } = event.data;
    useEditor
      .getState()
      .setStatus(
        exitCode === 0
          ? `Finished in ${seconds.toFixed(1)}s`
          : `Job failed with exit code ${exitCode}`,
      );
  });

  // Report what the Python side can do, so the user learns it now rather than
  // when they press a button.
  void RuntimeService.Probe().then((env) => {
    if (env.detail) useEditor.getState().setStatus(env.detail);
  });
}
