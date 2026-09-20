// Package core is the TensorCAD analysis engine.
//
// The IR, the symbolic shape algebra, the block catalog, shape inference, the
// design rules, the analysis and the code generator. Everything that computes a
// number about a design is here, and every client — the editor, the command
// line, the MCP server, the desktop shell — asks this rather than working it
// out again, so an answer cannot depend on where it was asked. The editor
// reaches it as WebAssembly, built by `cmd/wasm`.
//
// What holds it to its answers is `testdata/`: the symbol table and inferred
// shapes for all twenty presets, the full analysis and the design-rule check at
// three operating points, every byte of three generated `model.py` variants,
// and the prose of every block. `go run ./cmd/golden` rewrites those files, and
// nothing else does — a test that rewrote its own expectations would pass
// whatever the engine did. They began as the answers of the TypeScript this was
// ported from, and each one was compared against it before that was deleted.
package core
