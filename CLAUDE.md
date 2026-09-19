# TensorCAD

Node-based CAD for designing LLM architectures at the pretraining level: blocks, tensor wiring, design-rule checks, quantitative analysis, PyTorch generation, and an MCP server.

## Commands

```bash
bun test packages/core/test          # core suite
go test ./packages/core-go/...       # the Go port, against the TypeScript's golden files
bun run scripts/golden.ts            # regenerate those golden files (deliberately)
bun x tsc -p packages/core/tsconfig.json --noEmit   # type-check
bun run scripts/report.ts            # parameter regression table vs published counts
bun run scripts/analyze-demo.ts      # full analysis + design rules for one preset
bun run scripts/codegen-demo.ts <preset>   # writes out/<preset>/model.py
bun run scripts/scale-demo.ts        # shrink a design to a bench budget
```

## Core API

`analyze(doc, options)` gives every number at once. `validate(doc, options)` runs the design rules and returns the analysis with them. `generateTorch(doc)` emits PyTorch. `explain(doc, path)` describes one block: its parameters as written and as evaluated, its shapes, its share of the model, and its documentation. `scaleDesign(doc, {targetParams})` shrinks a design while keeping its proportions. `importHfConfig(config)` reads a Hugging Face `config.json`.

## Verified against PyTorch

`python -m tensorcad_runtime verify out/<preset>/model.py` instantiates a generated model on the meta device and reports its real parameter count. Every preset matches the analysis exactly, up to DeepSeek-V3 at 671,026,419,200.

Two cross-checks worth knowing:

- **FLOPs.** For GPT-2 small at batch 2, sequence 128, `torch.utils.flop_counter` measures 251.78 MFLOP per token and the analysis reports 249.42. The whole difference is the causal mask: a profiler counts the attention operator as if nothing were masked, because the operator's shape does not depend on the mask. `flops.fwdTotalUnmasked` reproduces the profiler's number exactly; `flops.fwdTotal` is what a fused causal kernel actually does. A test pins both.
- **Export.** `generateTorch` defaults to a gather-based mixture-of-experts dispatch, which uses `nonzero` and so cannot be traced by `torch.export`. Pass `moeDispatch: "dense"` for a traceable variant that computes the same thing at `experts / top_k` times the cost. Use it to verify a design, not to train one.
- **Initialization.** Generated models carry an `init_weights()` method rather than relying on PyTorch's defaults, because `nn.Embedding` defaults to a unit normal. Measured on GPT-2 small: defaults give a next-token loss of 466, `init_weights()` gives 10.94 against the uniform baseline of `ln(50257) = 10.82`. It is a method rather than part of `__init__` so the model can still be built on the meta device for verification.

## Layout

- **The engine is moving to Go.** `packages/core-go` is replacing `packages/core`; the
  frontend becomes a client of it over Wails bindings rather than running the analysis in the
  window. `docs/index.md` has the stages, what is done, and how each one is proven —
  the TypeScript writes golden files for all seventeen presets and the Go tests have to
  reproduce them exactly, including the printed form of every polynomial. Until a stage
  lands, `packages/core` is still the engine and still the specification. Do not "improve"
  the port as you go: it is bug-compatible on purpose, and the one deliberate divergence is
  documented.
- `packages/core` — pure TypeScript, **zero runtime dependencies**. IR, symbolic shapes, block catalog, design rules, analysis, code generation. Everything else is a client of this.
- `packages/ui` — React + React Flow editor. `state/unfold.ts` turns one flat level into the
  nested drawing the published figures use: containers become frames around their contents and
  container boundaries are short-circuited out of the wiring. Merging frames must happen after
  edges are resolved, because the set of open frames is what the edge tracer walks through.
  `state/derive.ts` calls `validate()` once per
  document and operating point and every panel reads the result; nothing in the UI computes
  its own numbers. The right column splits: the readout (`Operating` + `Analysis`) is always
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
- `packages/cli`, `packages/mcp` — thin adapters over the core.
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
