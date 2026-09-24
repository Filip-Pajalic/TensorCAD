# TensorCAD

Node-based CAD for designing LLM architectures at the pretraining level: blocks, tensor wiring, design-rule checks, quantitative analysis, PyTorch generation, and an MCP server.

## Commands

```bash
bun run build:wasm                   # the engine; nothing else runs without it
bun run test:all                     # types, both suites, the browser, the Go engine
bun run test:browser                 # the built editor in headless Chrome, driven by real key events
bun test packages                    # the TypeScript suite: the compiled engine and its clients
go test ./...                        # from packages/core-go: the engine against the goldens
go run ./cmd/golden                  # from packages/core-go: rewrite them (deliberately)
bun run scripts/report.ts            # parameter regression table vs published counts
bun run scripts/analyze-demo.ts      # full analysis + design rules for one preset
bun run scripts/codegen-demo.ts <preset>   # writes out/<preset>/model.py
bun run scripts/scale-demo.ts        # shrink a design to a bench budget
bun run packages/cli/src/index.ts mup <preset>   # the width ladder for a sweep
bun run trace                        # retrain nano-sort and rewrite the committed trace (needs torch)
python -m tensorcad_runtime trace out/<design>/model.py --out trace.json   # any small design, then File > Load a trace
```

A stale `.wasm` is the one way to see an answer the source does not give. If you
changed a formula, rebuild it.

## The engine's API

One engine, reached the same way everywhere. `createEngine()` from
`@tensor-cad/engine` loads the WebAssembly module in a browser;
`@tensor-cad/engine/node` does it from a process and exposes the same calls as
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
`mup(doc, options)` is the same design at several widths, with what to multiply
the initialization and the learning rate by at each: it holds the head dimension
and grows the head count, and it classifies a weight by measuring which of its
sides moved rather than by its block type.
`plan(doc, options, {gpus})` prices every way of splitting the training across a
cluster and returns the ones that fit, least demanding first — memory is the
claim, which is arithmetic; which is *fastest* is not claimed, because that turns
on the interconnect and the kernels, so each plan carries a note about what it
costs to run instead. `diff(a, b, options)` says what changed between two designs, structure and numbers
together, measuring both sides at one operating point. `importHuggingFace(text)` reads a `config.json`.

Everything crosses as JSON text. A design *is* JSON and so is every report, so
serialising costs a copy and buys a boundary with nothing clever in it.

## Verified against PyTorch

`python -m tensorcad_runtime verify out/<preset>/model.py` instantiates a generated model on the meta device and reports its real parameter count. Every preset matches the analysis exactly, up to DeepSeek-V3 at 671,026,419,200.

Two cross-checks worth knowing:

- **FLOPs.** For GPT-2 small at batch 2, sequence 128, `torch.utils.flop_counter` measures 251.78 MFLOP per token and the analysis reports 249.42. The whole difference is the causal mask: a profiler counts the attention operator as if nothing were masked, because the operator's shape does not depend on the mask. `flops.fwdTotalUnmasked` reproduces the profiler's number exactly; `flops.fwdTotal` is what a fused causal kernel actually does. A test pins both.
- **Export.** `generateTorch` defaults to a gather-based mixture-of-experts dispatch, which uses `nonzero` and so cannot be traced by `torch.export`. Pass `moeDispatch: "dense"` for a traceable variant that computes the same thing at `experts / top_k` times the cost. Use it to verify a design, not to train one.
- **Attention expressions.** A design's `mask` and `score` are generated as FlexAttention's `mask_mod` and `score_mod` and called through `expression_attention`, which compiles `flex_attention` on CUDA and otherwise applies the same functions to the whole score matrix. The second form is what verification profiles and exports; `expression_attention_probe.py` holds it against uncompiled `flex_attention` to the bit. The language is `packages/core-go/attnexpr`; `catalog/attention.go` is what it costs, and a mask is counted by *evaluating* it — seeded and stratified, so reproducible — rather than by a formula.
- **Initialization.** Generated models carry an `init_weights()` method rather than relying on PyTorch's defaults, because `nn.Embedding` defaults to a unit normal. Measured on GPT-2 small: defaults give a next-token loss of 466, `init_weights()` gives 10.94 against the uniform baseline of `ln(50257) = 10.82`. It is a method rather than part of `__init__` so the model can still be built on the meta device for verification.

## Layout

- **The engine is Go, compiled to WebAssembly.** `packages/core-go` is the whole
  analysis; `packages/engine` is that module plus the TypeScript client that loads it. The
  editor, the command line, the MCP server and the desktop shell are all clients of the same
  module, so an answer cannot depend on where it was asked.
- **`packages/core-go/testdata` is the specification.** A hundred and seven files saying what
  the engine answers for all twenty-four presets: symbol tables, inferred shapes at two expansion
  settings, the full analysis and the design-rule check at three operating points, every byte
  of a generated `model.py`, the prose of every block. `go run ./cmd/golden` rewrites them and
  nothing else does — never a test, which would pass whatever the engine did. They began as
  what the TypeScript said, and every one of them was compared against it, parsed rather than
  as bytes, on the commit that retired it.
- **Two suites read those files, and they are not the same test.** `go test ./...` checks the
  source. `packages/engine/test` checks the *compiled* module through the JavaScript boundary,
  which is the only place a nil slice, a NaN or a printed double can go wrong. A change that
  passes the first and fails the second is a boundary bug, not an arithmetic one.
- **The boundary is where answers get quietly lost.** A nil Go slice is `null`, not an empty
  list; `encoding/json` refuses a NaN, which a design with a failed symbol produces; Go
  rounds a half to even where JavaScript rounds it away from zero. `packages/core-go/jsonx`
  and `analysis/format.go` exist for those three, and `report/wire_test.go` walks a whole
  report objecting to every `null` it was not told to expect. Add to that list rather than
  papering over it in a client.
- **A preset is a document, not a builder.** `packages/core-go/presets/data` holds the
  library as twenty-four JSON files, embedded into the binary. There is no builder any more and
  nothing generates them: a new preset is a file, and `meta.published` is what the tests hold
  it to.
- `packages/engine` — the compiled module and the TypeScript that loads it.
  `src/types.ts` is the wire contract: every option a client can send and every field it
  gets back, in the names the boundary reads. `wasm/` is build output and is not committed,
  so a fresh clone runs `bun run build:wasm` before anything works.
- `packages/ui` — React + React Flow editor. `state/unfold.ts` turns one flat level into the
  nested drawing the published figures use: containers become frames around their contents and
  container boundaries are short-circuited out of the wiring. Merging frames must happen after
  edges are resolved, because the set of open frames is what the edge tracer walks through.
  `state/storage.ts` is the seam a store plugs into and `state/session.ts` is where you *were*
  as distinct from what you had — the level, the selection, the unfold depth and the operating
  point, composed out of the store's own actions so a restore goes through the same doors an
  edit does. A view is kept beside a document and the two can drift, so a stale part is dropped
  rather than taking the whole restore with it. `/d/<id>` is handled by `openFromLocation`,
  called by whatever assembled the editor; there is still no router, because
  `not_found_handling: "single-page-application"` already returns `index.html` for every path.
  `state/derive.ts` makes one `derive()` call per
  document and operating point and every panel reads the result; nothing in the UI computes
  its own numbers. `src/engine.ts` is the editor's handle on the engine: it loads the module
  before the first frame, which is why `main.tsx` imports the app *after* the load rather
  than beside it — the store builds a starting design the moment its module runs. The right column splits: the readout (`Operating` + `Analysis`) is always
  on screen above the tabbed editing pane (Inspector, Symbols, Cluster, Ladder), and the
  design-rule findings are a dock along the bottom rather than a tab — `panels/FindingsDock.tsx`
  wraps `Rules` in a strip that carries the counts even when collapsed, because the checks are
  about the drawing and belong under it.
  `state/commands.ts` is the single list behind the keyboard, the menu and the shortcut sheet.
  React Flow answers Enter and Escape on a focused block itself — selecting and deselecting it —
  before the window's handler sees the key, so a command that asks "is something selected"
  must ask `selectionAtKeyDown`, recorded by `beforeKey` in the capture phase, or one press does
  two things. `scripts/browser-test.ts` is what caught that: the unit suite has no React Flow in it.
  `commit(edit, label)` in `state/store.ts` is the single door every document edit goes through,
  and it takes a **value** rather than a closure — `{ kind: "setParam", path, key, value }` —
  because that is what makes the history a feature timeline rather than an undo stack. The
  state is `base` plus `steps` plus `at`, and the document is the fold; `state/edits.ts` owns
  the union and `applyEdit`. A step can be suppressed, and everything after it replays on top
  of what is left. Failure is an edit returning the document it was given, so an `ops` function
  that clones unconditionally when it has nothing to do reports success for work it did not
  do — three of them did, and it was invisible until something read the answer. The label is
  both what the toolbar says at the time and what the timeline shows later: one sentence, not
  two descriptions of the same event.
  `Cluster`, `Ladder` and `Runs` are the panels that do not read `derive()`. `Runs` reads
  nothing from the engine at all — it is the one panel about what a design *did* rather than
  what it would cost, reading the records `tensorcad-runtime smoke-train` writes and drawing
  their loss curves against tokens, not steps, because a step is not a fixed amount of work.
  `state/runs.ts` also decides when two runs are not comparable, and the chart says so before
  it is read rather than after. The other two call `plan()` and
  `mup()` themselves, because those are a few hundred analyses and one per rung, and neither
  answer moves between keystrokes that do not change the design. Pressing a cluster plan
  changes the operating point; pressing a rung opens a different *design*, through `setDoc`,
  so it lands on the undo stack. Pressing a
  plan writes it into the operating point, which is why the parallelism has to be expressible
  there — a plan the operating point cannot hold is a plan whose promised memory the readout
  would then contradict. Colours live only in
  `app/theme.css` as `data-theme` tokens — a literal colour in a stylesheet or a component is
  a bug, because it will not switch themes.
  `three/trace.ts` is what a design *computes*, where everything else is what it
  costs: a run recorded by `tensorcad-runtime trace`. One is committed under
  `three/traces`, for `nano-sort`; any other is *added* — File > Load a trace, or
  the desktop shell's Design > Trace this design, both ending in `loadTrace` — and
  kept on the shelf in `three/trace-shelf.ts`: IndexedDB, keyed by the same model
  fingerprint, the most recent eight, so a design opened again finds its trace
  with nothing loaded. The desktop also writes it beside the design file, as
  `<name>.trace.json`, so the pair travels. A trace is either trained to sort or
  `untrained`, and everything that quotes one says which. It is shown only on a
  design that still generates the `model.py` it was made from, matched by that
  file's hash, because anything cheaper — a name, the symbols, the parameter
  count — lets `gelu` swapped for `relu` through. Each box in `model3d.ts` names the tensor it is a picture of as
  a `CellSource` (a path, a layer, a role, which axis runs across); the view fills
  those from the trace and leaves the rest as decoration, and says which is which.
  `trace.test.ts` recomputes the input embedding and each head's output from the
  boxes beside them, which is what catches a transposed read — a transposed
  weight is still a plausible texture.
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
  rendered by both the menu bar in the toolbar and the right-click menu. `canvas/svg.ts` writes the sheet out as a vector by
  reading what was rendered: the wires are taken as the `<path d>` the router already
  produced, and the blocks are found by looking for anything with a background or a visible
  border rather than by naming classes — the fill is on `.part__body`, not on the node, which
  is invisible until a block with dark text on a light fill turns up. `scripts/export-svg.ts`
  drives it headless and refuses to write a file with no wires in it.
  `canvas/wiring.ts` decides which of a port's four sides a wire leaves by and what
  kind of line it is. What is *drawn* at a connection point is eeschema's vocabulary and
  nothing else: a connected pin draws nothing at all, a net that branches gets a filled
  junction dot, an unwired pin gets a hollow circle, and every one of those sizes is a
  multiple of `--wire-w` rather than a number picked by eye. Do not add a marker that is
  on almost everywhere — a diamond for "this shape mentions B or T" was exactly that, and
  it distinguished nothing because nearly every tensor in a transformer has both.
- `packages/cli`, `packages/mcp` — thin adapters over the engine. Both load it once at
  startup, before the first command or the transport opens.
- **The live editor bridge is an observer, not a second store.** `packages/mcp/src/bridge`
  is a loopback WebSocket that mirrors the one `FileStore` the tools write to; an editor's
  edit comes back through the same `apply` a tool call uses, so there is one document, one
  revision counter and one undo log. It is off unless `TENSORCAD_BRIDGE=1` asks for it.
  `packages/ui/src/state/bridge.ts` is the other end: it finds the bridge by probing
  `127.0.0.1:7357..7360` for `GET /session`, publishes what is on screen, and lands what
  arrives through `applyRemote` — which keeps the open level and the selection where
  `setDoc` would throw them away, because somebody is watching. Two Bun quirks are written
  into `packages/mcp/test/bridge.test.ts`: its `ws` shim has no `unexpected-response`, and
  nothing written to a socket taken off an `upgrade` event is ever delivered. A raw
  handshake that succeeds and is then dropped without a close aborts the test process
  outright — exit 127, no message.
- `python/tensorcad_runtime` — the only Python: instantiates generated models to verify them, and runs small training jobs.
- `docs/` — documentation, organised by Diátaxis (tutorials, how-to, reference, explanation). `reference/analysis-math.md` is the sourced maths behind the analysis engine.
- **Both sites are static.** `packages/ui/wrangler.jsonc` serves the Vite bundle at the apex,
  `tensorcad.dev`; `wrangler.docs.jsonc` serves what MkDocs renders at `docs.tensorcad.dev`.
  Neither has a Worker script — `assets` with no `main` is an assets-only deployment, which
  is all this needs, because the engine is WebAssembly and runs in the tab.
  **One hostname each, and adding a route is not a small change.** A custom domain
  belongs to whichever Worker claimed it most recently, so a route added here can take
  a hostname away from whatever is serving it, on the next push to `main`, with nothing
  to warn you. These two deployments claim what this repository builds and nothing else.
  `.github/workflows/deploy.yml` uploads both from `main` when
  `CLOUDFLARE_API_TOKEN` is set and says so in the job summary when it is not, so a fork
  still builds. The editor is loaded with `WebAssembly.instantiate` over bytes rather than
  `instantiateStreaming`, so nothing depends on the `.wasm` arriving with the right content
  type.

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
5. **Activation memory is attributed to tensors, not blocks.** A block lists the input ports it
   must keep alive (`retains`); each producing tensor is counted once even when several blocks
   read it. It is *reported* both ways: `activationsByPath` per block and `activationsByTensor`
   per producing pin. The two agree row for row in a plain transformer and diverge wherever a
   block fans out — Nemotron-H's `split` holds 290 MiB across three pins, 128, 160 and 2 — which
   is why clicking a wire in the editor can answer a question clicking a block cannot.
6. **`B` and `T` are reserved runtime symbols.** They stay indeterminate through shape checking, so a mismatch is a real polynomial difference.
7. **A configuration is a view of the design; the operating point is a view of the measurement.**
   `doc.configurations` are named sets of symbol values and `doc.active` says which is in
   force — four GPT-2 presets are one architecture at four sizes. A configuration overrides
   symbols and nothing else, says only what differs so an expression still follows, and never
   touches the document's own symbols, so switching twice ends where it started. It is document
   state because it changes what the design *is*.
8. **The operating point is editor state, not document state.** Batch, sequence length,
   dtype, device, GPU count and the sharding plan are conditions the analysis is measured
   under, not properties of the design. `packages/ui/src/state/operating.ts` owns them and
   translates them to `AnalysisOptions`; the CLI and MCP pass their own.
9. **This repository is the open one, and keeps nothing private in it.** Accounts, sessions,
   billing and the hosted service live in a separate private repository. The seam is
   `packages/ui/src/state/storage.ts`: a `StorageProvider` that names no vendor, no host and no
   protocol, handed **text** from `serializeDoc` rather than a `Doc`, so it cannot disagree with
   the engine about what a design is. A deployment registers a provider before the first frame;
   a plain checkout registers none, the `Designs` tab is not in the row, and the editor is
   exactly what it was — which is a test, not a hope. `bun run boundary` reads every file and
   fails on a vendor name, a vendor's environment variable, a stateful Cloudflare binding or a
   privileged key's name; it runs in `test:all` and in CI before the tests, because what it
   guards against is a commit rather than a behaviour.
10. **Presets are the regression suite.** Every preset carries `meta.published` and the tests assert the analysis reproduces it. Twenty-one of twenty-four match to the parameter; the other three — `qwen3-30b-a3b`, `qwen3-235b-a22b` and `deepseek-v3` — are checked against rounded vendor figures with an explicit `tolerance`. Not all of them are language models. `ijepa-vit-h14` is a vision transformer with bidirectional attention and no vocabulary (`presets/jepa.ts`), and `alexnet` is a convolutional classifier whose tensors are `B C H W` rather than a sequence (`presets/convnet.ts`). A convnet's token is one image, so `T` is 1 and every per-token figure reads as per-image; its FLOPs match `torch.utils.flop_counter` exactly, there being no causal mask to disagree about. Everything downstream treats all three kinds identically.

## When adding a block

A catalog entry needs: parameter specs, port patterns, `docs.summary` and `docs.formula` with a source link, and, for a primitive, the formulas (`paramCount`, `flops`, `retains`, `stateBytes`). Add a preset that uses it with a published figure, or a test that pins the arithmetic. Run `bun run scripts/report.ts` before and after.

## Known gaps

See `ROADMAP.md` M5. The architecture blocks are all there now; what is left is presets for the models that use the newest of them, whose configs are still moving, and the analysis extensions in that section.

A stack whose layers are not all alike needs no new mechanism, and there is no
`pattern` parameter to reach for — there was one, it was never implemented, and
it has been removed. When the variation has a period, the repeating unit is the
*group*: Gemma 2 is one `repeat` of `L/2` holding a windowed block and a full
one in series, which is what makes half its cache stop growing with the
sequence. When there is no period, the layers are written out, as
Nemotron-H's fifty-two are.

Two figures in `docs/reference/analysis-math.md` were wrong in the original research and are corrected by the implementation: the Llama-3-70B cache is 320 KiB per token, not 160, and the per-layer activation estimate of 147 KB assumes a fused gated feed-forward. An unfused one keeps two more intermediate tensors, which is the 176 KB the analysis reports.
