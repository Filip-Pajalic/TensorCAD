# @tensorcad/engine

The TensorCAD analysis engine: one WebAssembly module with a JSON interface, and
the TypeScript that loads it.

Give it a design and it tells you what the design costs — parameters, FLOPs, KV
cache, memory at an operating point, throughput, dollars — checks it against
eighteen design rules, writes the PyTorch, and searches for a way to train it on
a cluster. The same module answers in a browser, in a Node process and inside
the desktop shell, so an answer cannot depend on where it was asked.

## Using it

In a browser or anything with `fetch`:

```ts
import { createEngine } from "@tensorcad/engine";
import "@tensorcad/engine/wasm_exec";

const engine = await createEngine({ wasm: "/tensorcad.wasm" });
const doc = engine.preset("llama-3-8b");
console.log(engine.analyze(doc).params.total); // 8030261248
```

From a Node process, where the module is read rather than fetched and held as a
singleton:

```ts
import { analyze, getPreset, loadEngine } from "@tensorcad/engine/node";

await loadEngine();
const report = analyze(getPreset("mixtral-8x7b"), { T: 4096 });
console.log(report.params.total, report.params.active);
```

`wasm_exec.js` is Go's own loader, vendored from the toolchain that built the
module. The two travel together or neither works.

## What it answers

| call | what it gives |
| --- | --- |
| `analyze(doc, options)` | every number at once |
| `validate(doc, options)` | the design rules, and the analysis they ran against |
| `derive(doc, options)` | the findings and every shape from one walk of the graph |
| `infer(doc, mode)` | the shapes alone, for the wire the pointer is over |
| `explain(doc, path)` | one block: its parameters as written and as evaluated, its shapes, its share |
| `generateTorch(doc, options)` | a `model.py` and the design that produced it |
| `scale(doc, {targetParams})` | the design shrunk to a budget, proportions kept |
| `mup(doc, options)` | the same design at several widths, and what to scale by at each |
| `plan(doc, options, {gpus})` | every way to split the training across a cluster, and which fit |
| `importHuggingFace(text)` | a `config.json` read into a design |
| `preset(name)`, `presets()` | the twenty designs it ships with |
| `catalog()`, `rules()`, `hardware()` | what it knows about blocks, rules and devices |

Everything crosses as JSON text. A design *is* JSON and so is every report, so
serializing costs a copy and buys a boundary with nothing clever in it. A call
that cannot answer throws an `EngineError` naming what was wrong rather than
returning a default.

## What it is held to

`packages/core-go/testdata` in the repository: the symbol table and inferred
shapes for twenty published architectures, the full analysis and the design-rule
check at three operating points each, and every byte of three generated
`model.py` variants. Seventeen of the twenty reproduce their published parameter
count exactly and the other three are within a stated tolerance, and every one
of them has been instantiated in PyTorch to confirm the count is real — up to
DeepSeek-V3 at 671,026,419,200.

## Building it

```bash
bun run build:wasm
```

from the repository root. The output lands in `wasm/` and is not committed, so a
fresh clone builds it before anything works.

MIT. See LICENSE.md.
