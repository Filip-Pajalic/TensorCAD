/**
 * The compiled engine, against the answers the goldens record.
 *
 * The Go tests prove the source. This proves the thing that actually ships:
 * the same code after a compiler, a linker and a WebAssembly runtime have each
 * had a turn at it, asked the same questions through the boundary the editor
 * uses. The two can disagree, and have — a nil slice arrives as `null`, a NaN
 * cannot be encoded at all, and Go and JavaScript print the same double
 * differently. None of that is visible from inside Go.
 *
 * The comparison is against `packages/core-go/testdata` rather than a second
 * implementation, so every expectation here is a file a person can read.
 *
 * Needs the module built first: `bun run build:wasm`.
 */
import "../vendor/wasm_exec.js";
