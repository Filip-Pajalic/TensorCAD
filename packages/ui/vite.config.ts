import { defineConfig, normalizePath, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwind from "@tailwindcss/vite";

/**
 * Start fetching the engine with the page, not after the script that asks for it.
 *
 * The editor draws nothing until the engine is loaded, and the engine's URL is
 * only known once `main.tsx` has downloaded, parsed and run — so without this
 * the largest file on the page waits for the second largest. Its name is hashed,
 * so the link is written from the bundle rather than by hand. `crossorigin`
 * matches what `fetch()` sends, or the browser fetches it twice.
 */
function preloadEngine(): Plugin {
  return {
    name: "tensorcad:preload-engine",
    apply: "build",
    transformIndexHtml(_html, ctx) {
      const wasm = Object.keys(ctx.bundle ?? {}).find((f) => /tensorcad-[^/]*\.wasm$/.test(f));
      if (!wasm) return [];
      return [
        {
          tag: "link",
          attrs: { rel: "preload", href: `/${wasm}`, as: "fetch", type: "application/wasm", crossorigin: "anonymous" },
          injectTo: "head",
        },
      ];
    },
  };
}

/**
 * Tailwind, handing Vite the files it scanned in the form Vite keys modules by.
 *
 * On Windows its scanner reports `C:\Git\...` with backslashes, and it passes
 * each one to `addWatchFile` as a dependency of the stylesheet. Vite turns a
 * dependency into a module by checking whether the path starts with the root —
 * which it keeps as `C:/Git/...` — so every component gets a second entry, at
 * `/@fs/C:\Git\...`. Save one and that is the URL the browser is told to import:
 * a fresh copy of the component, of the store it imports, and of `engine.ts`,
 * whose engine was never loaded. "The engine is not loaded yet" on every save.
 *
 * `createFileOnlyEntry` normalises its path and this branch of Vite does not;
 * this does it on the way in. Nothing changes on a path that was already
 * forward-slashed, which is every path on every other platform.
 */
function tailwindWithPosixPaths(): Plugin[] {
  return tailwind().map((plugin) => {
    const t = plugin.transform;
    if (!t) return plugin;
    const handler = typeof t === "function" ? t : t.handler;
    const wrapped = function (this: ThisParameterType<typeof handler>, ...args: Parameters<typeof handler>) {
      const original = this;
      // Everything else reads through to Vite's own context, so what the
      // handler records lands where Vite looks for it afterwards.
      const ctx = Object.create(original, {
        addWatchFile: { value: (id: string) => original.addWatchFile(normalizePath(id)) },
      });
      return handler.apply(ctx, args);
    };
    return { ...plugin, transform: typeof t === "function" ? wrapped : { ...t, handler: wrapped } } as Plugin;
  });
}

export default defineConfig({
  plugins: [react(), tailwindWithPosixPaths(), preloadEngine()],
  // @tensor-cad/engine is a TypeScript workspace package consumed straight from
  // source; keep it out of the dependency pre-bundler so Vite's own resolver
  // handles the `./x.js` -> `./x.ts` convention it uses, and so the `.wasm` it
  // resolves with `import.meta.url` is not rewritten to a path that no longer
  // points at it.
  optimizeDeps: { exclude: ["@tensor-cad/engine"] },
  server: { port: 5173, strictPort: false },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500,
    // Seven megabytes of engine. Inlining it as a data URL would be slower to
    // parse and impossible to cache separately from the app.
    assetsInlineLimit: 0,
  },
});
