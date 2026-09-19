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
export { parseDoc, serializeDoc } from "./state/serialize.js";
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
