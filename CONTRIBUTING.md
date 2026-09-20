# Contributing to TensorCAD

Thanks for looking. This file says how the project is built, what it will not
compromise on, and what "done" means — for people and for coding agents, who
are both expected here.

## Getting set up

```bash
bun install                              # workspace deps
bun test packages                        # 273 tests
bun run scripts/report.ts                # the parameter regression table
```

The Python runtime is optional and only needed to verify generated models
against real PyTorch:

```bash
pip install -e python/tensorcad_runtime
```

## The one command that matters

```bash
bun run scripts/report.ts
```

Twenty presets, each carrying the parameter count its authors published. If that
table changes and you did not mean it to, stop and find out why. It is the
project's conscience.

## Before you open a pull request

```bash
bun run build:wasm            # the engine; everything else needs it
bun test                      # the whole suite, both engines
bun run typecheck             # types
go test ./...                 # from packages/core-go
bun run scripts/report.ts     # presets unchanged
```

`bun run test:all` is the three of them in order.

If you touched the desktop app:

```bash
cd desktop && wails3 task build
```

The engine is Go compiled to WebAssembly, and the editor, the command line, the
MCP server and the desktop shell all load the same module. If you change a
formula, rebuild it — a stale `.wasm` is the one way to see an answer the source
does not give.

## The invariants

These are in [`CLAUDE.md`](./CLAUDE.md) in full. They are not style preferences;
the architecture rests on them.

1. **The document is the source of truth.** Editor state is a view of it.
2. **`ex()` parenthesises only when it has to.** Composites nest, so
   unconditional wrapping compounds into `(((H)))`.
3. **Formulas live on primitives.** Composites are subgraphs the analysis
   expands. Adding an architecture block should need no new maths.
4. **Containers carry two multipliers.** `total` drives parameters, `active`
   drives FLOPs. That one mechanism is what makes mixture-of-experts work.
5. **Activation memory belongs to tensors, not blocks.** A block declares the
   inputs it must keep alive; each producing tensor is counted once.
6. **`B` and `T` are reserved runtime symbols.** They stay indeterminate, so a
   shape mismatch is a real polynomial difference.
7. **The operating point is editor state, not document state.** Batch, sequence
   length, dtype, device and sharding are conditions a design is measured under.
8. **Presets are the regression suite.** Every one carries `meta.published`.

## Adding a block

The checklist, in order:

1. A catalog entry with parameter specs and port patterns.
2. `docs.summary` and `docs.formula`, with a source link. Someone will want to
   know where the arithmetic came from, and that someone is usually you in six
   months.
3. For a primitive: `paramCount`, `flops`, `retains`, `stateBytes`.
4. Code generation, if it can be emitted.
5. **Either** a preset that uses it with a published figure, **or** a test that
   pins the arithmetic. Preferably both.
6. `bun run scripts/report.ts` before and after.

A composite needs none of the formulas — it expands into primitives and the
analysis does the rest. If you find yourself writing new maths for a composite,
that is a sign the primitive underneath is missing.

## Adding a preset

A preset is an assertion about a real model, so it has to be checkable:

- `meta.published.params` with a `source` link to the config or paper it came
  from. Exact where the source is exact; a `tolerance` only where the published
  figure is itself rounded ("22B active"), and say so in the notes.
- Verify it end to end:
  ```bash
  bun run scripts/codegen-demo.ts <preset>
  python -m tensorcad_runtime verify out/<preset>/model.py
  ```
  This instantiates the generated model on the meta device and compares its true
  parameter count, module by module.
- If the number disagrees with the one people quote, **say why in the notes**.
  I-JEPA's ViT-H is 630.4M, not the 632M everyone cites, because its positions
  are frozen sincos and it has no class token. That sentence is worth more than
  the number.

## Working on the 3D view

`three/model3d.ts` and `three/View3D.tsx` are a port of
[llm-viz](https://github.com/bbycroft/llm-viz). When something looks wrong,
**read `GptModelLayout.ts` and `components/Arrow.ts`** rather than reasoning
about it from a description. This has been wrong every single time it was
guessed at. The conventions are deliberately kept close to the original: y is
positive downward, a block is a grid of cells, the residual is a tall standing
plate, ribbons are blue out of a weight and green out of a value.

## Style

The code is commented for a reader who is competent but new — what a thing is
for and why it is that way, not what the line does. Where a decision was
contentious, the comment says what the alternative was and why it lost.

Colours live only in `packages/ui/src/app/theme.css` as `data-theme` tokens. A
literal colour in a stylesheet or a component is a bug, because it will not
switch themes.

Prose in comments and docs uses sentences. Numbers are exact or explicitly
approximate, never vague.

## Good first issues

- **Presets.** V-JEPA 2 needs only the existing builder with `frames` and
  `tubelet`. ResNet and VGG need the conv primitives that already exist.
- **The engine.** `packages/core-go/` is the whole analysis, compiled to
  WebAssembly and loaded by everything else. Adding a block means a catalog
  entry, a preset or a test that pins its arithmetic, and
  `go run ./cmd/golden` to write down what it changed.
- **Known gaps** in [`ROADMAP.md`](./ROADMAP.md): linear attention (gated
  DeltaNet), multi-token prediction, Gemma's alternating local/global attention.
- **Instancing in the 3D view.** It is currently one mesh and one material per
  plate, about 3,000 of them.

## What will get a change sent back

- A preset without a published figure to check it against.
- A new number with no source.
- A literal colour outside `theme.css`.
- "Improving" the Go port away from bug-compatibility with the TypeScript. It is
  bug-compatible on purpose; the one deliberate divergence is documented.
- Formulas added to a composite instead of a primitive.

## Documentation

Docs live in [`docs/`](./docs/index.md) and follow [Diátaxis](https://diataxis.fr/).
Put a change where its *reader's need* is, not where its subject is: a tutorial
teaches by doing, a how-to solves one problem for someone who already knows the
tool, a reference is looked things up in, and an explanation argues. A page that
tries to be two of these serves neither.

Filenames are lowercase and hyphenated.

## Reporting a problem

Say what you expected, what happened, and which preset or design shows it. If it
is a number, include what the engine reported and what you believe it should be,
with a source. A design attached as JSON is worth a page of description.
