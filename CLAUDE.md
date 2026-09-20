# TensorCAD

Node-based CAD for designing LLM architectures at the pretraining level: blocks, tensor wiring, design-rule checks, quantitative analysis, PyTorch generation, and an MCP server.

## Commands

```bash
bun run build:wasm                   # the engine; nothing else runs without it
bun run test:all                     # types, both suites, the Go engine
bun test packages                    # the TypeScript suite, including the compiled engine
go test ./...                        # from packages/core-go: the engine against the goldens
bun run scripts/golden.ts            # regenerate those golden files (deliberately)
bun run scripts/report.ts            # parameter regression table vs published counts
bun run scripts/analyze-demo.ts      # full analysis + design rules for one preset
bun run scripts/codegen-demo.ts <preset>   # writes out/<preset>/model.py
bun run scripts/scale-demo.ts        # shrink a design to a bench budget
```

A stale `.wasm` is the one way to see an answer the source does not give. If you
changed a formula, rebuild it.

## The engine's API

One engine, reached the same way everywhere. `createEngine()` from
`@tensorcad/engine` loads the WebAssembly module in a browser;
`@tensorcad/engine/node` does it from a process and exposes the same calls as
free functions over a per-process singleton.

`analyze(doc, options)` gives every number at once. `validate(doc, options)`
runs the design rules and returns the analysis with them. `derive(doc, options)`
is the editor's call: the findings and every shape from one walk of the graph,
because asking separately would walk it twice per keystroke. `infer(doc, mode)`
is the shapes alone, which is what answers "would this wire type-check" for
every handle the pointer passes over. `generateTorch(doc)` emits PyTorch.
`explain(doc, path)` describes one block: its parameters as written and as
evaluated, its shapes, its share of the model, and its documentation.
`scale(doc, {targetParams})` shrinks a design while keeping its proportions.
`importHuggingFace(text)` reads a `config.json`.

Everything crosses as JSON text. A design *is* JSON and so is every report, so
serialising costs a copy and buys a boundary with nothing clever in it.

## Verified against PyTorch

`python -m tensorcad_runtime verify out/<preset>/model.py` instantiates a generated model on the meta device and reports its real parameter count. Every preset matches the analysis exactly, up to DeepSeek-V3 at 671,026,419,200.

Two cross-checks worth knowing:

- **FLOPs.** For GPT-2 small at batch 2, sequence 128, `torch.utils.flop_counter` measures 251.78 MFLOP per token and the analysis reports 249.42. The whole difference is the causal mask: a profiler counts the attention operator as if nothing were masked, because the operator's shape does not depend on the mask. `flops.fwdTotalUnmasked` reproduces the profiler's number exactly; `flops.fwdTotal` is what a fused causal kernel actually does. A test pins both.
- **Export.** `generateTorch` defaults to a gather-based mixture-of-experts dispatch, which uses `nonzero` and so cannot be traced by `torch.export`. Pass `moeDispatch: "dense"` for a traceable variant that computes the same thing at `experts / top_k` times the cost. Use it to verify a design, not to train one.
- **Initialization.** Generated models carry an `init_weights()` method rather than relying on PyTorch's defaults, because `nn.Embedding` defaults to a unit normal. Measured on GPT-2 small: defaults give a next-token loss of 466, `init_weights()` gives 10.94 against the uniform baseline of `ln(50257) = 10.82`. It is a method rather than part of `__init__` so the model can still be built on the meta device for verification.

## Layout

- **The engine is Go, compiled to WebAssembly.** `packages/core-go` is the whole
  analysis; `packages/engine` is that module plus the TypeScript client that loads it. The
  editor, the command line, the MCP server and the desktop shell are all clients of the same
  module, so an answer cannot depend on where it was asked.
- **`packages/core` no longer ships. It is the oracle.** It is the TypeScript the engine was
  ported from, and `bun run scripts/golden.ts` is it writing down what it says for all twenty
  presets: symbol tables, inferred shapes at two expansion settings, the full analysis and the
  design-rule check at three operating points, and every byte of a generated `model.py`. The
  Go tests reproduce those exactly, and `packages/engine/test` runs the *compiled* module
  against the same answers. **So a block added to one engine and not the other fails a test
  that names it.** Add it to both. Do not "improve" the Go while you are in there: it is
  bug-compatible on purpose, and every deliberate divergence is documented where it is made.
- **The boundary is where answers get quietly lost.** A nil Go slice is `null`, not an empty
  list; `encoding/json` refuses a NaN, which a design with a failed symbol produces; Go
  rounds a half to even where JavaScript rounds it away from zero. `packages/core-go/jsonx`
  and `analysis/format.go` exist for those three, and `report/wire_test.go` walks a whole
  report objecting to every `null` it was not told to expect. Add to that list rather than
  papering over it in a client.
- **A preset is a document, not a builder.** `packages/core-go/presets/data` holds the
  library as JSON, embedded into the binary. `packages/core/src/presets` still builds those
  documents and is what `scripts/golden.ts` runs; when the TypeScript goes, the JSON stays and
  nothing has to be ported.
- `packages/ui` — React + React Flow editor. `state/unfold.ts` turns one flat level into the
  nested drawing the published figures use: containers become frames around their contents and
  container boundaries are short-circuited out of the wiring. Merging frames must happen after
  edges are resolved, because the set of open frames is what the edge tracer walks through.
  `state/derive.ts` makes one `derive()` call per
  document and operating point and every panel reads the result; nothing in the UI computes
  its own numbers. `src/engine.ts` is the editor's handle on the engine: it loads the module
  before the first frame, which is why `main.tsx` imports the app *after* the load rather
  than beside it — the store builds a starting design the moment its module runs. The right column splits: the readout (`Operating` + `Analysis`) is always
  on screen above the tabbed editing pane (Inspector, Symbols, Rules). Colours live only in
  `app/theme.css` as `data-theme` tokens — a literal colour in a stylesheet or a component is
  a bug, because it will not switch themes.
  `three/model3d.ts` and `three/View3D.tsx` are the volume view, a port of Brendan Bycroft's
  LLM visualisation (MIT). Its conventions are his and deliberately so — y positive downward,
  a block as a grid of cells, the residual as a tall standing plate, flow ribbons blue out of
  a weight and green out of a value. **When something there looks wrong, read
  `GptModelLayout.ts` and `components/Arrow.ts`. Do not reason about it from a
  description — that has been wrong every time.** In particular the arrows are
  not polylines: `drawArrow` builds a frame from the run's direction and sweeps
  a cubic bezier whenever the ends differ in depth or the ribbon lands side-on,
  which is what carries the attention heads back into one output. Chrome is shadcn-style components over Base UI in
  `src/ui`, with Tailwind v4 in `app/tailwind.css` mapping the shadcn token names onto this
  project's; every radius is zero. `state/commands.ts` is the single list behind the keyboard,
  the menu, the shortcut sheet and the Wails native menu, so a key can never be documented
  wrong; `panels/menu-tree.tsx` declares how those commands are grouped into submenus and is
  rendered by both the menu bar in the toolbar and the right-click menu. `canvas/wiring.ts` decides which of a port's four sides a wire leaves by and what
  kind of line it is. What is *drawn* at a connection point is eeschema's vocabulary and
  nothing else: a connected pin draws nothing at all, a net that branches gets a filled
  junction dot, an unwired pin gets a hollow circle, and every one of those sizes is a
  multiple of `--wire-w` rather than a number picked by eye. Do not add a marker that is
  on almost everywhere — a diamond for "this shape mentions B or T" was exactly that, and
  it distinguished nothing because nearly every tensor in a transformer has both.
- `packages/cli`, `packages/mcp` — thin adapters over the engine. Both load it once at
  startup, before the first command or the transport opens.
- `python/tensorcad_runtime` — the only Python: instantiates generated models to verify them, and runs small training jobs.
- `docs/` — documentation, organised by Diátaxis (tutorials, how-to, reference, explanation). `reference/analysis-math.md` is the sourced maths behind the analysis engine.

## Invariants

1. **The document is our JSON IR.** Editor state is a view of it, never the source of truth.
   It also carries `defs`: composites the design defines for itself. Resolve blocks through
   `catalogOf(doc)`, never the bare `CATALOG`, or a design's own blocks become "unknown block
   type". A user block is parameters, ports and a template subgraph whose node parameters
   reference the block's own as `$name`; boundary nodes are generated from the declared ports.
2. **`ex()` adds parentheses only when they are needed.** Composites pass parameters into the
   composites they expand into, so unconditional wrapping compounds: three levels turned `H`
   into `(((H)))`. It counts the same and reads like line noise everywhere it is shown.
3. **Formulas live on primitives.** Composites (`gqa_attention`, `gated_mlp`, `moe_layer`, `transformer_block`) are subgraphs of primitives, expanded by the analysis. Adding an architecture block needs no new math.
4. **Containers carry two multipliers.** `total` drives the parameter count, `active` drives FLOPs and activation memory. That single mechanism is what makes mixture-of-experts work.
5. **Activation memory is attributed to tensors, not blocks.** A block lists the input ports it must keep alive (`retains`); each producing tensor is counted once even when several blocks read it.
6. **`B` and `T` are reserved runtime symbols.** They stay indeterminate through shape checking, so a mismatch is a real polynomial difference.
7. **The operating point is editor state, not document state.** Batch, sequence length,
   dtype, device, GPU count and the sharding plan are conditions the analysis is measured
   under, not properties of the design. `packages/ui/src/state/operating.ts` owns them and
   translates them to `AnalysisOptions`; the CLI and MCP pass their own.
8. **Presets are the regression suite.** Every preset carries `meta.published` and the tests assert the analysis reproduces it. Seventeen of twenty match to the parameter; the other three are checked against rounded vendor figures with an explicit `tolerance`. Not all of them are language models. `ijepa-vit-h14` is a vision transformer with bidirectional attention and no vocabulary (`presets/jepa.ts`), and `alexnet` is a convolutional classifier whose tensors are `B C H W` rather than a sequence (`presets/convnet.ts`). A convnet's token is one image, so `T` is 1 and every per-token figure reads as per-image; its FLOPs match `torch.utils.flop_counter` exactly, there being no causal mask to disagree about. Everything downstream treats all three kinds identically.

## When adding a block

A catalog entry needs: parameter specs, port patterns, `docs.summary` and `docs.formula` with a source link, and, for a primitive, the formulas (`paramCount`, `flops`, `retains`, `stateBytes`). Add a preset that uses it with a published figure, or a test that pins the arithmetic. Run `bun run scripts/report.ts` before and after.

## Known gaps

See `ROADMAP.md` M5. The main ones: linear-attention blocks (gated DeltaNet), multi-token prediction, and Gemma's alternating local and global attention layers, which the importer warns about rather than approximating.

Two figures in `docs/reference/analysis-math.md` were wrong in the original research and are corrected by the implementation: the Llama-3-70B cache is 320 KiB per token, not 160, and the per-layer activation estimate of 147 KB assumes a fused gated feed-forward. An unfused one keeps two more intermediate tensors, which is the 176 KB the analysis reports.
