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
export { DiskSink } from "./store/disk-artifacts.js";
export type { ArtifactSink, WrittenArtifact } from "./artifacts.js";
/**
 * Everything it takes to *be* a store, not just to hold one.
 *
 * `DocumentStore` alone is not implementable from outside this package: the
 * two errors are part of its contract — a caller distinguishes a revision
 * conflict from an unknown id by catching them — and the option and outcome
 * shapes appear in its signatures. Exporting the interface without them is
 * exporting a shape nobody else can satisfy, which is what the hosted server
 * found on its first attempt.
 */
export {
  RevisionConflictError,
  UnknownDesignError,
} from "./store/types.js";
export type {
  DocumentStore,
  DesignRecord,
  DesignSummary,
  DesignSource,
  ApplyOutcome,
  CheckpointInfo,
  NewDesignOptions,
  StoreChange,
  StoreListener,
} from "./store/types.js";
export { TOOL_NAMES } from "./tools.js";
export { PROMPT_NAMES } from "./prompts.js";
export { applyOps, type Op } from "./ops.js";
