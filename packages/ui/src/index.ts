/**
 * The editor as a package.
 *
 * The browser build mounts this through `main.tsx`; the desktop build mounts
 * the same component inside a Wails window. Nothing here knows which one it is
 * running in: the shell tells it, through the theme and command bridges.
 */

export { default as App } from "./app/App.js";
export { useEditor } from "./state/store.js";
export {
  initTheme,
  setThemePreference,
  setSystemTheme,
  onThemeChange,
  resolvedTheme,
  themePreference,
  themeValue,
  type ThemePreference,
  type ResolvedTheme,
} from "./state/theme.js";
export { parseDoc, serializeDoc, fileNameFor } from "./state/serialize.js";
/**
 * The seam a store plugs into.
 *
 * Exported because a deployment that keeps designs somewhere is assembled
 * outside this package: it imports `App`, registers a provider before the
 * first frame, and renders. Without these it would have to fork the editor to
 * add a panel, which is the outcome the interface exists to avoid.
 */
export {
  registerStorage,
  storage,
  subscribeStorage,
  type StorageProvider,
  type StoredDesign,
  type Account,
  type ViewState,
} from "./state/storage.js";
export { captureView, applyView, openFromLocation } from "./state/session.js";
export {
  COMMANDS,
  commandsIn,
  runCommand,
  prettyShortcut,
  type Command,
  type CommandGroup,
} from "./state/commands.js";
export {
  DEFAULT_OPERATING,
  toAnalysisOptions,
  type OperatingPoint,
} from "./state/operating.js";
