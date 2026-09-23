import { defineConfig, type Plugin } from "vite";
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

export default defineConfig({
  plugins: [react(), tailwind(), preloadEngine()],
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
