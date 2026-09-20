/**
 * The library: everything this package exposes to code that imports it.
 *
 * Starting a server is `stdio.ts`, and the split matters. This used to be both,
 * with an `if (import.meta.main)` at the bottom — which works under Bun and
 * does not survive a bundler, because `import.meta.main` compiles to a
 * CommonJS check that is not defined in an ESM output. The desktop bundle
 * crashed before its first line ran.
 */

export { serve } from "./serve.js";
export { createServer, SERVER_NAME, SERVER_VERSION } from "./server.js";
export { FileStore } from "./store/file-store.js";
export type { DocumentStore, DesignRecord, DesignSummary } from "./store/types.js";
export { TOOL_NAMES } from "./tools.js";
export { PROMPT_NAMES } from "./prompts.js";
export { applyOps, type Op } from "./ops.js";
