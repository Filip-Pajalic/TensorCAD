import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwind from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwind()],
  // @tensorcad/engine is a TypeScript workspace package consumed straight from
  // source; keep it out of the dependency pre-bundler so Vite's own resolver
  // handles the `./x.js` -> `./x.ts` convention it uses, and so the `.wasm` it
  // resolves with `import.meta.url` is not rewritten to a path that no longer
  // points at it.
  optimizeDeps: { exclude: ["@tensorcad/engine"] },
  server: { port: 5173, strictPort: false },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500,
    // Seven megabytes of engine. Inlining it as a data URL would be slower to
    // parse and impossible to cache separately from the app.
    assetsInlineLimit: 0,
  },
});
