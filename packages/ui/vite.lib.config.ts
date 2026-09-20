/**
 * The editor as something another application can import.
 *
 * `vite.config.ts` builds the editor as a *site*: one bundle with React and
 * everything else inside it, served at app.tensorcad.dev. That is the wrong
 * shape for a dependency. An application that installs this already has React,
 * and two copies of React in one page is not a larger download — it is hooks
 * throwing `Invalid hook call` on the first render, because the component was
 * built against one copy and mounted by the other.
 *
 * So everything the consumer declares for themselves stays outside the bundle,
 * and what is emitted is this package's own code and nothing else.
 *
 * ## Why the CSS is compiled here and not there
 *
 * `app.css` imports `tailwind.css`, which is Tailwind v4 — so consuming this
 * package as source would mean the consumer also installing Tailwind, adding
 * its Vite plugin, and pointing its content scanner at somebody else's
 * `node_modules`. Getting any of that subtly wrong produces an editor with no
 * styles rather than an error. Running it here emits plain CSS: the consumer
 * imports one file and needs to know nothing.
 *
 *     bun run build:lib
 */

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwind from "@tailwindcss/vite";
import { createRequire } from "node:module";

const manifest = createRequire(import.meta.url)("./package.json");

/**
 * What the consumer provides.
 *
 * Every runtime dependency, taken from the manifest rather than listed again
 * here — a second list is a list that goes stale, and the failure it causes is
 * a duplicated package rather than an error. The regular expression also
 * catches subpaths, so `react/jsx-runtime` and `@tensor-cad/engine/node` are
 * external for the same reason their packages are.
 */
const external = Object.keys(manifest.dependencies ?? {}).map(
  (name) => new RegExp(`^${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(/.*)?$`),
);

export default defineConfig({
  plugins: [react(), tailwind()],
  build: {
    target: "es2022",
    // Beside the site build rather than on top of it: `bun run build` and
    // `bun run build:lib` produce different things and neither should silently
    // become the other.
    outDir: "lib",
    emptyOutDir: true,
    sourcemap: true,
    cssCodeSplit: false,
    lib: {
      entry: {
        index: "src/index.ts",
        engine: "src/engine.ts",
        style: "src/style.ts",
      },
      formats: ["es"],
      // Vite names the stylesheet after the package ("ui.css") otherwise, and
      // the published export is `./style.css`.
      cssFileName: "style",
    },
    rollupOptions: {
      external,
      output: {
        // Stable names, because they are what the published `exports` map
        // points at. A hash here would mean regenerating the manifest from the
        // build output on every release.
        entryFileNames: "[name].js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: "[name][extname]",
      },
    },
  },
});
