/**
 * Theme control.
 *
 * Three settings, which is what every application that does this well offers:
 * light, dark, or follow the system. The choice is stored; what it resolves to
 * is derived. Applying it is one attribute on `<html>`, so no component ever
 * asks which theme is on.
 *
 * In the desktop build the system signal comes from the operating system via
 * the Go side. In a browser it comes from `prefers-color-scheme`. Both land
 * here through `setSystemTheme`.
 */

export type ThemePreference = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "tensorcad.theme";

let preference: ThemePreference = "system";
let systemTheme: ResolvedTheme = "light";
const listeners = new Set<(resolved: ResolvedTheme, pref: ThemePreference) => void>();

/** What the current preference actually resolves to. */
export function resolvedTheme(): ResolvedTheme {
  return preference === "system" ? systemTheme : preference;
}

export function themePreference(): ThemePreference {
  return preference;
}

function apply(): void {
  const resolved = resolvedTheme();
  if (typeof document !== "undefined") {
    document.documentElement.setAttribute("data-theme", resolved);
  }
  for (const fn of listeners) fn(resolved, preference);
}

/** Choose light, dark, or follow the system. Persists across restarts. */
export function setThemePreference(next: ThemePreference): void {
  preference = next;
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // A private window or a locked-down profile: the theme still works, it
    // just will not be remembered.
  }
  apply();
}

/** Report what the operating system or the browser currently prefers. */
export function setSystemTheme(next: ResolvedTheme): void {
  systemTheme = next;
  if (preference === "system") apply();
}

export function onThemeChange(fn: (resolved: ResolvedTheme, pref: ThemePreference) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/**
 * Read a theme colour as a literal, for a library that will not take a CSS
 * variable. React Flow's background and minimap are the reason this exists.
 */
export function themeValue(token: string, fallback = "#888888"): string {
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(token).trim();
  return value || fallback;
}

/**
 * Start theming. Reads the stored preference, picks up the browser's system
 * setting, and keeps following it. The desktop build additionally pushes the
 * operating system's setting through `setSystemTheme`.
 */
export function initTheme(): void {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark" || stored === "system") {
      preference = stored;
    }
  } catch {
    // Fall back to following the system.
  }

  if (typeof window !== "undefined" && window.matchMedia) {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    systemTheme = query.matches ? "dark" : "light";
    query.addEventListener("change", (e) => setSystemTheme(e.matches ? "dark" : "light"));
  }

  apply();
}
