// Package core is the TensorCAD analysis engine.
//
// This is the Go port of `packages/core`, which it replaces: the IR, the
// symbolic shape algebra, the block catalog, the design rules, the analysis and
// the code generator. The frontend is a client of it — every number on screen
// comes from here, over Wails bindings, rather than being recomputed in the
// window.
//
// The port is staged, and each stage is proven rather than asserted: the
// TypeScript engine writes golden files for all seventeen presets under
// `testdata/`, and the Go tests read the same documents and require the same
// answers. `bun run scripts/golden.ts` regenerates them.
package core
