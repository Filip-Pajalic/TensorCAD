# TensorCAD

<p align="center">
  <b>Schematic capture for neural network architectures.</b><br/>
  Draw the model, get the numbers, generate the PyTorch.
</p>

<p align="center">
  <a href="https://github.com/Filip-Pajalic/TensorCAD/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Filip-Pajalic/TensorCAD/ci.yml?branch=main&style=flat-square&label=CI" alt="CI status" /></a>
  <img src="https://img.shields.io/badge/license-MIT-111111?style=flat-square" alt="MIT licensed" />
  <img src="https://img.shields.io/badge/engine-Go%20%E2%86%92%20WebAssembly-2f6fb0?style=flat-square" alt="Go engine compiled to WebAssembly" />
  <img src="https://img.shields.io/badge/presets-20%20verified-1b6834?style=flat-square" alt="20 verified presets" />
  <img src="https://img.shields.io/badge/tests-287%20passing-1b6834?style=flat-square" alt="287 tests passing" />
  <img src="https://img.shields.io/badge/MCP-server%20included-8a5b9c?style=flat-square" alt="MCP server included" />
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-it-does">What it does</a> ·
  <a href="#how-it-is-kept-honest">Correctness</a> ·
  <a href="#for-agents">For agents</a> ·
  <a href="./docs/index.md">Docs</a> ·
  <a href="./CONTRIBUTING.md">Contributing</a> ·
  <a href="./ROADMAP.md">Roadmap</a> ·
  <a href="./LICENSE.md">License</a>
</p>

---

TensorCAD treats a neural network the way an EDA tool treats a circuit. Blocks
are symbols with typed pins. Tensors are nets. Shapes are checked by a real
algebra rather than by running the thing. A design-rule check tells you the
model will not fit on your GPUs *before* you rent them. And the drawing is not
a picture of the model — it **is** the model, and PyTorch falls out of it.

It started as a tool for language models. It now draws vision transformers and
convolutional classifiers too, because the same machinery turned out to work.

## Quick start

Needs [Bun](https://bun.sh) and [Go](https://go.dev) 1.25 or later. The analysis
engine is Go compiled to WebAssembly, which is the one build step:

```bash
git clone https://github.com/Filip-Pajalic/TensorCAD
cd TensorCAD
bun install
bun run build:wasm
```

Every number in the project comes out of one command:

```bash
bun run scripts/report.ts
```

```
preset              calculated     published   delta
------------------------------------------------------------
gpt2-small              124.4M        124.4M   exact
llama-3-8b               8.03B         8.03B   exact
deepseek-v3            671.03B       671.00B   0.0039%
ijepa-vit-h14            1.28B         1.28B   exact
alexnet                  61.1M         61.1M   exact
```

Generate a model and check it against real PyTorch:

```bash
bun run scripts/codegen-demo.ts llama-3-8b      # writes out/llama-3-8b/model.py
python -m tensorcad_runtime verify out/llama-3-8b/model.py
```

Open the editor:

```bash
bun run --cwd packages/ui dev      # browser
cd desktop && wails3 task build    # desktop app (Wails v3 + Go)
```

Both load the same engine. So do the command line and the MCP server, which is
the point: the numbers cannot depend on where you asked for them.

## What it does

**Schematic capture.** Blocks, typed pins, orthogonal wire routing, four-sided
pin anchors, junction dots on branching nets, hollow circles on unconnected
pins. Containers unfold in place so a 32-layer stack reads as one frame with a
`32×` bracket, the way published architecture figures draw it.

**A real shape algebra.** Every tensor dimension is a multivariate polynomial
with exact rational coefficients over named symbols. `B` and `T` stay
indeterminate all the way through, so a mismatch is a genuine polynomial
difference rather than two numbers that happened not to match. Splits carry
divisibility obligations instead of silently rounding.

**Design-rule checks.** Eighteen rules: head divisibility, vocabulary padding,
RoPE dimension parity, interface breakage, whether the design fits the GPUs you
selected under the sharding plan you chose. The DRC panel correctly refuses
Llama-3-8B at 90.16 GiB/GPU against an H100's 80.

**Quantitative analysis.** Parameters, FLOPs, activation memory (Megatron
formulas), KV cache, ZeRO/FSDP/TP/PP sharding, roofline throughput, Chinchilla
budgets. Nothing in the UI computes its own numbers; one `validate()` call per
document and operating point feeds every panel.

**PyTorch generation.** `generateTorch(doc)` emits a runnable model with an
`init_weights()` method — because `nn.Embedding` defaults to a unit normal, and
that is the difference between a next-token loss of 466 and 10.94 against the
`ln(50257) = 10.82` baseline.

**A 3D volume view.** Every tensor as a plate, sized by its real dimensions,
with flow ribbons between them. Ported from Brendan Bycroft's
[LLM visualisation](https://github.com/bbycroft/llm-viz).

**An MCP server.** So an agent can design, validate, analyse and generate
without a human in the loop.

## How it is kept honest

This is the part worth reading, and the reason to trust the numbers.

**Twenty presets are the regression suite.** Each one carries the parameter
count its authors published, and the tests assert the analysis reproduces it.
Seventeen match *to the parameter*; the other three are checked against rounded
vendor figures with an explicit tolerance.

**Every preset is instantiated in real PyTorch.** `python -m tensorcad_runtime
verify` builds the generated model on the meta device and reports its true
parameter count, module by module, up to DeepSeek-V3 at 671,026,419,200.

**FLOPs are checked against a profiler.** For AlexNet the agreement is exact —
1,428,376,960 per image, ratio 1.000000 against `torch.utils.flop_counter`. For
GPT-2 small the profiler says 251.78 MFLOP/token and the analysis says 249.42,
and *the whole difference is the causal mask*: a profiler counts the attention
operator as if nothing were masked. `flops.fwdTotalUnmasked` reproduces the
profiler exactly; `flops.fwdTotal` is what a fused causal kernel actually does.
A test pins both numbers.

**The Go port is proven against the TypeScript it replaces.** The engine is
migrating to Go; the TypeScript writes golden files for all twenty presets and
the Go tests must reproduce them exactly — including the evaluation order of the
symbol table, the printed form of every polynomial, and the text of every error.

## Architectures it draws

| | |
|---|---|
| **Language** | GPT-2 (small→XL), nanoGPT, Llama 2/3/3.1, Mistral, Qwen 2.5/3, Gemma 2, Mixtral, DeepSeek-V3, Nemotron-H |
| **Vision** | I-JEPA ViT-H/14 — bidirectional attention, three towers including the EMA target encoder |
| **Convolutional** | AlexNet — `B C H W` tensors, spatial downsampling, 96% of its weights in the classifier |

Mechanisms covered: GQA/MQA/MHA, multi-head latent attention, SwiGLU/GeGLU,
RMSNorm/LayerNorm, RoPE with scaling, mixture-of-experts with shared experts and
routing bias, Mamba-2 state-space layers, sliding-window attention, QK-norm,
post-norm, tied embeddings, hybrid stacks.

## Repository shape

```
packages/core-go/   the engine — Go, no dependencies outside the standard library
packages/engine/    the engine compiled to WebAssembly, and its TypeScript client
packages/ui/        React + React Flow editor, 2D sheet and 3D volume view
packages/cli/       command line: validate, analyze, show, diff, codegen
packages/mcp/       MCP server
desktop/            Wails v3 + Go desktop application
python/             the only Python: verifies generated models against PyTorch
docs/               tutorials, how-to guides, reference and explanation
```

One engine, everywhere. The editor, the command line, the MCP server and the
desktop shell all load the same WebAssembly module and ask it the same
questions, so an answer cannot depend on where it was asked. What it is held to
is `packages/core-go/testdata`: every preset's symbol table, inferred shapes,
full analysis, design-rule findings and generated PyTorch, byte for byte,
checked both against the Go source and against the compiled module. Those files
began as the answers of the TypeScript this was ported from, which has since
been deleted.

## For agents

This repository is written to be worked on by coding agents as well as people.

- **[`CLAUDE.md`](./CLAUDE.md)** is the entry point: commands, layout, the eight
  invariants, and what to do when adding a block. Read it first.
- **Invariants are load-bearing.** The document is the source of truth; formulas
  live only on primitives; containers carry two multipliers; activation memory
  is attributed to tensors rather than blocks; `B` and `T` are reserved. Break
  one and the tests will tell you, but the design rules will not.
- **Adding a block** needs parameter specs, port patterns, `docs.summary` and
  `docs.formula` with a source link, and — for a primitive — `paramCount`,
  `flops`, `retains` and `stateBytes`. Then a preset that uses it with a
  published figure, or a test pinning the arithmetic. Run
  `bun run scripts/report.ts` before and after.
- **The MCP server** exposes the whole engine as tools. Point your agent at
  `.mcp.json`.

## Built on

TensorCAD borrows from work that deserves naming:

- **[llm-viz](https://github.com/bbycroft/llm-viz)** by Brendan Bycroft (MIT) —
  the 3D volume view is a port of its layout and arrow rendering. The residual
  pathway down the centre, weights to either side, blocks wrapping into columns,
  the ribbon arrows with lines down their edges: the arrangement is his.
- **[KiCad](https://www.kicad.org/)** — the interaction model. Pick apertures,
  net highlighting, junction dots, dangling-pin marks and the selection
  semantics are taken from eeschema's behaviour and documentation.
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** by Andrej Karpathy (MIT) —
  a preset, and the reference for GPT-2-shaped arithmetic.
- **[I-JEPA](https://github.com/facebookresearch/ijepa)** by Meta AI — the
  vision preset is built from its published configuration.
- **[torchvision](https://github.com/pytorch/vision)** (BSD-3) — the AlexNet
  definition and its parameter count.
- **[React Flow](https://reactflow.dev/)**, **[ELK](https://eclipse.dev/elk/)**,
  **[Three.js](https://threejs.org/)**, **[Base UI](https://base-ui.com/)**,
  **[Wails](https://wails.io/)** — the editor stands on these.

Formulas are sourced individually in
[`docs/reference/analysis-math.md`](./docs/reference/analysis-math.md),
including two figures the original research got wrong that the implementation
corrects.

## Status

Working and useful, with rough edges. The Go migration is at stage 2 of 10. See
[`ROADMAP.md`](./ROADMAP.md) for what is known to be missing — linear-attention
blocks, multi-token prediction, and Gemma's alternating local/global attention,
which the importer warns about rather than approximating.

## Documentation

[`docs/`](./docs/index.md), organised by [Diátaxis](https://diataxis.fr/):
[tutorials](./docs/tutorials/first-design.md) to learn from,
[how-to guides](./docs/how-to/add-a-block.md) to work from,
[reference](./docs/reference/ports.md) to look things up in, and
[explanation](./docs/explanation/why-schematic-capture.md) for why any of it is
the way it is.

## License

MIT. See [`LICENSE.md`](./LICENSE.md), which also carries the notices for the
MIT-licensed work this project ports.
