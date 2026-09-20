/**
 * Making the layout worker — kept in its own module so a build can leave it out.
 *
 * Vite turns `new Worker(new URL(…), …)` into an emitted asset plus a path to
 * it, and it does that while *transforming* this file, before anything has a
 * chance to decide the worker is unwanted. A runtime flag is therefore no use:
 * the asset is emitted either way, and the wrong path goes out with it.
 *
 * So the decision is made by which module is resolved. `vite.lib.config.ts`
 * points this specifier at `elk-worker.none.ts`, and the packaged build then
 * contains no worker, no asset and no path — see there for why a library
 * cannot ship one.
 */

/** Null in a build that ships no worker. */
export const createLayoutWorker: (() => Worker) | null = () =>
  new Worker(new URL("elkjs/lib/elk-worker.min.js", import.meta.url), { type: "classic" });
