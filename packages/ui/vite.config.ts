import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwind from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwind()],
  // @tensorcad/core is a TypeScript workspace package consumed straight from
  // source; keep it out of the dependency pre-bundler so Vite's own resolver
  // handles the `./x.js` -> `./x.ts` convention the core uses.
  optimizeDeps: { exclude: ["@tensorcad/core"] },
  server: { port: 5173, strictPort: false },
  build: { target: "es2022", chunkSizeWarningLimit: 1500 },
});
