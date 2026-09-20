/**
 * The stand-in for `elk-worker.ts` in the packaged build.
 *
 * A worker is a *file*, and a file is the one thing this package cannot hand
 * whoever installs it: the path baked into the module is wrong wherever the
 * package ends up. In development it 404s to a single-page fallback, the worker
 * is handed HTML, and the drawing comes out stacked because no layout ever
 * runs. In a consumer's production build it is worse — their bundler tries to
 * resolve the path as an entry module and the build fails outright.
 *
 * `layout.ts` sees null and takes its main-thread path, which is the same
 * algorithm over a module this package already depends on.
 */

export const createLayoutWorker: (() => Worker) | null = null;
