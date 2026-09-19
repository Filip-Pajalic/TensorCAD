# Moving the engine to Go

The decision: **Go owns the logic, the frontend is a client.** Every number on
screen comes from `packages/core-go` over Wails bindings, rather than being
recomputed in the window. `packages/core` is being replaced, not wrapped.

## Why this is not a straight rewrite

`main.go` currently says the opposite, and it had a reason:

> The analysis engine is TypeScript and runs in the window, because it
> recomputes on every keystroke and a process boundary in that loop would only
> make it slower.

Wails bindings are in-process — WebView2 message passing, not a socket — so the
boundary costs microseconds rather than milliseconds, and `analyze()` on the
largest preset is already tens of milliseconds. The reason still bites in one
place: the editor currently calls `validate()` synchronously on every document
change, and that becomes an async call. `state/derive.ts` is the single place
that happens, so it becomes a debounced request with the last good result held
while one is in flight. Nothing else in the UI computes its own numbers, which
is what makes this a contained change rather than a rewrite of the editor.

## How correctness is kept

The TypeScript engine is the specification until it is gone. `bun run
scripts/golden.ts` writes, for all seventeen presets, the document it builds and
the answers it gets, into `packages/core-go/testdata/`. The Go tests read the
same documents and require the same answers.

A stage is done when its golden file matches, and not before. This is stronger
than the existing regression suite: the presets already pin parameter counts
against published figures, but the golden files pin *everything* — the symbol
evaluation order, the printed form of every polynomial, the text of every error.

Two consequences worth stating:

- **The port is bug-compatible on purpose.** `rat()` scales by ten until both
  sides are whole, so 1/3 becomes 333333333333/1000000000000 rather than 1/3.
  Go does the same. Making it exact in one engine and not the other would have
  them print different shapes for the same design, and a shape label is the
  thing a person reads off the canvas. The imprecision gets fixed in both at
  once, afterwards, or not at all.
- **One thing is deliberately not bug-compatible.** The TypeScript carries
  numerator and denominator as doubles, exact to 2^53 and silently wrong past
  it. A 671-billion-parameter model is comfortably inside that; its FLOP counts
  are not. Go uses `math/big`, and the golden files agree because no preset
  currently crosses the line.

## Stages

| # | Stage | TS lines | Status |
|---|-------|---------:|--------|
| 0 | Module layout, `go.work`, golden harness | — | **done** |
| 1 | `shapes/symexpr`, `shapes/expr` — rational polynomials, parser | 621 | **done** |
| 2 | `ir/types`, `ir/symbols` — document, symbol table | 281 | **done** |
| 3 | `shapes/pattern`, `shapes/infer` — port patterns, shape inference | 526 | |
| 4 | `catalog/*` — types, primitives, composites, resolve, user blocks | 1,689 | |
| 5 | `analysis/*` — flatten, params, flops, memory, kvcache, cost, hardware | 1,433 | |
| 6 | `rules/*` — the seventeen design rules | 566 | |
| 7 | `codegen/torch` — PyTorch emission | 954 | |
| 8 | `presets`, `explain`, `scale`, `import/hf` | 1,058 | |
| 9 | Wails services, generated client, `state/derive.ts` cutover | — | |
| 10 | CLI and MCP server as Go commands; delete `packages/core` | — | |

Stages 3 to 8 are ordered by dependency, not by value: nothing can be proven
until the catalog can be resolved, and the catalog needs shape inference.

## Layout

```
go.work                     the two modules
packages/core-go/           module github.com/tensorcad/core
  shapes/                   rational polynomials, expression parser
  ir/                       the document and its symbol table
  testdata/
    presets/<name>.json     documents, written by the TypeScript
    golden/<name>.json      answers, written by the TypeScript
    expressions.json        the algebra's corners
desktop/                    module github.com/tensorcad/desktop, the Wails app
```

## Commands

```bash
bun run scripts/golden.ts              # regenerate the golden files
go test ./packages/core-go/...         # the port against them
go build ./desktop/... ./packages/core-go/...
```

Regenerate the golden files deliberately. A diff there is either a bug being
fixed or a behaviour being changed, and both want to be seen in review.

## What the frontend keeps

The drawing. React Flow, the canvas, the 3D volume view, layout, theming,
selection, undo — those are the client. What leaves is every call into
`@tensorcad/core`: `analyze`, `validate`, `explain`, `generateTorch`,
`scaleDesign`, `importHfConfig`, the catalog and the presets.

The 3D layout in `three/model3d.ts` is the awkward case. It reads `Derived` and
turns it into boxes, which is presentation, but it also reads the catalog to
find the shape of a design. It stays in TypeScript and takes what it needs from
the analysis result, which means the result has to carry the block shapes it
currently rummages for.
