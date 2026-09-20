# TensorCAD — Roadmap

Goal: a node-based CAD tool for designing neural network architectures at the pretraining level, with checks, analysis, code generation, a local test bench, and an MCP server. It began with language models and now covers vision transformers and convolutional classifiers on the same machinery. Primary use: learning, validating, and testing designs. See [docs/index.md](docs/index.md) for the documentation.

Effort estimates assume one developer working with an AI coding assistant, part-time. They are ranges, not commitments.

## Status as of 2026-09-19

| Milestone | State |
|---|---|
| M0 Sketch (core IR, symbolic shapes, catalog, params) | **Done.** 20 presets, exact parameter match on 17 of them. |
| M1 Check (design rules, full analysis) | **Done.** 18 rules, drawn on the canvas where the work happens; FLOPs, KV cache, memory, throughput, cost, Chinchilla. |
| M2 Manufacture (PyTorch codegen, verification) | **Done.** Every generated model's parameter count matches PyTorch exactly, and the FLOPs estimate matches a profiler once the causal mask is accounted for. |
| M3 Agent (MCP server, CLI) | **Done**, minus the live-UI bridge. 13 MCP tools over stdio, 6 CLI commands, 67 tests. |
| M4 Test bench | **Partly done.** `tensorcad-runtime smoke-train` trains a scaled design on the local GPU and logs a loss curve; `scaleDesign` shrinks a design to a budget. No run registry or comparison view in the editor yet. |
| M5 Advanced parts | **Mixture of experts, latent attention and state-space blocks all done.** DeepSeek-V3 and Nemotron-H-8B reproduce exactly. |
| Go engine | **Done.** The whole analysis is Go, compiled to WebAssembly, and the editor, the command line, the MCP server and the desktop shell all load the same module. The TypeScript it was ported from has been deleted; the golden files it wrote are the specification the engine is held to. |
| Editor UI | **Reworked against CAD convention.** Orthogonal wires, a grid with snap, schematic-style blocks, a model tree with locking, typed pins, a status bar, and named refusals. Remaining items in `docs/explanation/interaction-design.md`. |

Two suites, reading the same files. `go test ./...` in `packages/core-go` checks
the Go source; `bun test packages` checks the compiled module through the
JavaScript boundary, along with the command line, the MCP server and the editor.
Between them: every preset's symbol table, inferred shapes at two expansion
settings, the full analysis and the design-rule check at three operating points,
and the generated PyTorch byte for byte. `go run ./cmd/golden` rewrites those
files, deliberately and never as part of a test.

20 presets, 17 matching their published parameter count exactly and 3 within a
stated tolerance, and every one of them confirmed against PyTorch 2.11 by
instantiating the generated model. Not all are language models:
`ijepa-vit-h14` is a vision transformer and `alexnet` a convolutional
classifier, on the same machinery.

## Principles

1. **Own the IR.** The design document is our JSON format; React Flow state is a view. (ComfyUI lesson.)
2. **Pure core.** Schema, validation, analysis, and codegen are pure TypeScript functions with no I/O, shared by UI, CLI, and MCP.
3. **Analyze primitives, present composites.** Param/FLOP/memory formulas live on ~20 primitive ops; composite blocks (GQA attention, SwiGLU MLP, MoE layer) are subgraphs of primitives with exposed parameters. New blocks need no new math.
4. **Verify against PyTorch.** Every estimate can be cross-checked by instantiating the generated model on the meta device.
5. **Regression-test against real models.** The 15-model table in [docs/reference/analysis-math.md](docs/reference/analysis-math.md) is the test suite from day one.
6. **Assistant-native.** The MCP server is a first-class client of the core, not an afterthought.

## Milestones

### M0 — Sketch (proof of concept) · 1–2 weeks

Prove the node-based approach feels right and the math is trustworthy on dense models.

- Monorepo with `core`, `ui`; strict TypeScript; tests.
- IR schema (zod) with symbols, nodes, edges, `repeat` container; save/load.
- Primitives: input/output, embedding, linear, rmsnorm/layernorm, activation, add, mul, rearrange, rope, sdpa, lm_head. Composites: gqa_attention (covers MHA/MQA), gated_mlp, dense_mlp, transformer_block_prenorm.
- Symbolic shape engine and per-edge shape labels; connection validation.
- Parameter count (total, per block).
- Canvas: palette, typed handles, inspector, symbols panel, repeat sub-flow, auto-layout, undo.
- Presets: GPT-2 small, Llama-3-8B, Mistral-7B.

**Done when**: opening the Llama-3-8B preset shows 8.03B params matching the published count; changing `Hkv` from 8 to 32 updates the count live; dropping an edge shows an error badge; the file round-trips through save/load byte-identically.

**Result**: all of it, verified in the browser. The editor shows one graph level at a time with a breadcrumb: double-clicking the stack enters it, double-clicking the block shows the catalog expansion read-only with a banner explaining why. Editing `D` from 4096 to 8192 moves the header from 8.03B to 27.33B and `F` from 14,336 to 28,672, because `F` is stored as the expression Llama 3 actually uses. The Analysis panel shows a green "exact" badge against the published figure, and for DeepSeek-V3 it reports active parameters as 5.6% of the total.

### M1 — Check (design rules and analysis) · 2 weeks

Make it a real engineering tool for dense models.

- Rule engine with severities and fix-its: connectivity, shape, kernel/hardware, capacity, scaling-sanity rules.
- Analysis: FLOPs/token (fwd/bwd/train, attention term), KV cache per token/sequence, weight memory by dtype, training memory (optimizer, activations with/without recompute/FlashAttention, logits, ZeRO/FSDP/TP/PP sharding), inference memory, roofline throughput, training cost/time, Chinchilla ratio.
- Hardware profiles (RTX 5080, 4090, A100, H100, H200, B200, custom).
- Analysis panel with per-block breakdown and formula hover; what-if sliders on symbols.
- Regression tests for all dense models in the reference table; FLOPs spot-checked against EleutherAI cookbook numbers.

**Done when**: `tensorcad validate` and `tensorcad analyze` run headless on every preset in CI; a deliberately broken design (D=4000, H=32) is flagged with a fix-it; Llama-3-8B at T=8K shows KV cache 128 KiB/token and ~38 GB activations/layer-set without recompute.

### M2 — Manufacture (codegen and verification) · 2 weeks

Close the loop with PyTorch.

- PyTorch single-file `nn.Module` codegen from composites (idiomatic) with primitive fallback; golden-file tests.
- Python runtime package (`tensorcad-runtime`): instantiate on meta device, count params, `torch.export` shape check with symbolic B/T, `FlopCounterMode` FLOPs; JSON report over subprocess.
- UI "Verify" button showing the estimate-vs-torch diff table.
- HF `config.json` import for Llama/Mistral/Qwen/Gemma families; litgpt `Config` export.
- Training-harness template (nanoGPT-style `train.py` + data prep) emitted next to the model.

**Done when**: every preset generates code that instantiates and exports without error, and param counts match to the parameter; importing Qwen3-8B's `config.json` reproduces 8.19B.

**Result**: parameter counts match exactly for every preset. Import reproduces nine model families. Two findings came out of the cross-check and are recorded in `CLAUDE.md`: a profiler's FLOPs figure counts attention as though nothing were masked, and the default mixture-of-experts dispatch is not traceable by `torch.export` because it uses `nonzero`, so `generateTorch` now offers a dense dispatch that is.

### M3 — Agent (MCP and CLI) · 1 week

Let Claude Code and friends drive the tool.

- `@tensorcad/mcp` on the TypeScript SDK v2, stdio transport, file-mode `DocumentStore`; 15 tools with `outputSchema` and annotations; resources and prompts; see [docs/how-to/use-the-mcp-server.md](docs/how-to/use-the-mcp-server.md).
- `.mcp.json` for Claude Code; README snippets for Cursor and Claude Desktop.
- CLI (`tensorcad`) sharing the same command set.
- Live bridge: UI writes a session file; MCP server attaches over local WebSocket; ops flow both ways; `resources/updated` on human edits.

**Done when**: from Claude Code, "open the Llama-3-8B preset, make it a 4-expert MoE with top-2, and tell me the new active params" works end to end and the change appears on the canvas.

**Result**: the tool half works, verified over a real stdio pipe. That edit takes Llama-3-8B from 8.03B dense to 24.94B total and 13.67B active. The canvas half waits on the live bridge, which is the one piece of M3 still outstanding: `DocumentStore` is the seam, `FileStore` is the only implementation, and the design for a `LiveStore` over a local WebSocket is recorded at the top of `packages/mcp/src/store/file-store.ts`.

### M4 — Test bench (simulation) · 2–3 weeks

Actually train designs, small.

- Tiny-track scaler: shrink a design to a target param budget while keeping ratios; emits config + `train.py`.
- Runner in `tensorcad-runtime`: fixed dataset (FineWeb-Edu sample or TinyStories), fixed seeds, bf16, `torch.compile`; logs loss/tokens/s/peak memory to `runs/*.jsonl`.
- Run registry and compare view in the UI (loss vs tokens, loss vs wall-clock, memory); baseline designs.
- Verified on the local RTX 5080 (16 GB): a ~20M-param design trains in ~10 minutes.

**Done when**: two designs (GPT-2-style vs Llama-style at equal params) can be trained back to back from the UI and compared on one chart, reproducibly.

**Progress**: the training half works. A 30.1M-parameter GPT-2 shrunk by `scaleDesign` trains 500 steps on the RTX 5080 in 27 seconds at 151,213 tokens per second, loss 10.85 down to 3.23, 6.5 GiB peak. A Llama-shaped design of 48.2M reaches 3.53 in 200 steps. What is missing is the registry and the comparison view, so the runs are reproducible but not yet side by side.

### M5 — Advanced parts (ongoing after M2)

Done:
- `moe_layer` with routed and shared experts, top-k routing and an optional router bias, plus the `moe_experts` container that makes total and active parameter counts fall out of one mechanism.
- Sliding-window attention, QK-norm, post-norm sublayers, asymmetric attention bias.
- Presets: Mixtral-8x7B, Qwen3-30B-A3B, Qwen3-235B-A22B, Gemma-2-9B, Qwen2.5-7B, Qwen3-8B.

- `mla_attention`, DeepSeek's latent attention, with `split`, `concat`, `expand_heads` and a `kv_latent_cache` primitive so the compressed cache is modelled honestly rather than approximated. DeepSeek-V3 reproduces 671.03B total, 37.55B active and 70,272 cache bytes per token.
- `mamba2_block` with `conv1d` and `ssd_scan`, and irregular stacks written as one character per layer. Nemotron-H-8B reproduces exactly, and correctly reports a fixed per-sequence state instead of a cache that grows per token.
- Hugging Face `config.json` import for nine families, asserted against the hand-written presets on both the parameter count and the cache.
- Parallelism planner, in the editor as well: a `Cluster` tab beside Inspector, Symbols and Rules lists the plans that fit with a bar for how much of the device each fills, and pressing one applies it to the operating point so the whole readout follows. `plan(doc, options, {gpus})` and `tensorcad plan --gpus n` price every split the cluster admits — DP, TP, PP, EP, the four ZeRO stages, sequence parallelism and the three recompute settings — and return the ones that fit, least demanding first. Llama-3-70B on 64 H100s is a thousand-plan search that takes 20ms.
- Expert parallelism, which the options carried and nothing read: a sparse model's experts are a separate pool that shards by EP where everything else shards by TP and PP. Mixtral is 45.10B expert weights of 46.70B, so the distinction is most of the model.
- Value embeddings and U-net skips, which turned out to be one primitive. `mix` is `w0*a + w1*b` with both weights learned — not `add`, which has no weights, and not `mul`, which has no parameters. Value embeddings arrive on a port because the table is shared by every layer that reads it; a U-net skip is written out, because layer i feeding layer L-1-i is not a repeating unit.
- Design diff, in the editor as `View > Compare` and on the command line: `diff(a, b)` and `tensorcad diff <a> <b>` report what moved structurally and what it cost, measuring both sides at one operating point so the attention terms are comparable. It was 202 lines in the command line where the editor could not reach it.
- `gated_deltanet_block` and a `gated_delta_scan` primitive: linear attention with DeltaNet's write rule and Mamba-2's decay gate, so a layer of it caches nothing that grows with context — a matrix per head, fixed for the sequence, plus what the depthwise convolution remembers. The generated file carries a readable sequential reference that runs unmodified; swap it for `fla`'s chunked kernel to train.
- `mtp_head` and a `shift` primitive: multi-token prediction as DeepSeek-V3 describes it — normalize the hidden state, normalize the embedding of the token ahead, join, project 2D down to D — then a transformer block and the model's own output head, which is shared and so costs a second pass over the vocabulary and no weights. Depth is stacking rather than a parameter.
- Logit softcapping, on the output logits and on the attention scores. The one on the scores rules out a fused kernel — it never builds the matrix there is anything to cap — so a capped layer is counted as eager attention, which for Gemma-2-9B at 8k is 199 GiB of activations rather than 73.
- The μP width ladder: `mup(doc, options)`, `tensorcad mup` and `tensorcad_mup` build the same design at several widths, with what to multiply the initialization and the learning rate by at each. It holds the head dimension and grows the head count, which is the convention the paper is stated in and the one that leaves every head the shape it had at the base. A weight is classified by measuring which of its sides moved between a rung and the same design twice as wide, rather than by its block type: that is how a router lands in the output row, its fan_out being the expert count, and how an expert's own matrices land in the hidden row. Building it surfaced two designs the scaler had been shrinking wrongly — a "compressed" attention latent eight times wider than the model it compressed, and a predictor wider and deeper than the encoder it predicts for.
- Alternating local and global attention, as Gemma 2 and 3 use it: a repeat of the group rather than of the layer, so half of Gemma-2-9B's cache is bounded by the window. At 128k context that is 21.66 GiB where treating every layer as global said 42.00 GiB. The importer builds it rather than warning about it.

Remaining:
- Presets for the linear-attention hybrids, once their configs settle.
- Presets: Gemma-3, Jamba, and 2026 models as their configs stabilize.
- Analysis extensions: MoE active vs resident, MLA absorbed vs decompressed KV.
- User-defined composites library.

### M6 — Ship

- Docs site with the guided tours; explain tab content for every block.
- **Done.** `server.json` for the MCP registry, checked against the registry's own schema and pinned against `package.json` by a test. An MCPB bundle for Claude Desktop, built for Node with the engine beside it and started under Node before it is attached to a release.
- Publish `@tensorcad/engine` and `@tensorcad/mcp` to npm, and `server.json` to the registry. Both are one command and a decision about version numbers.
- Optional: MCP Apps canvas preview for Claude Desktop/Cursor; hosted read-only viewer for sharing designs.

## Sequencing and dependencies

```
M0 Sketch ──► M1 Check ──► M2 Manufacture ──► M3 Agent ──► M4 Test bench
                               │                 │
                               └──► M5 Advanced parts (starts after M2, runs alongside M3/M4)
                                                                 └──► M6 Ship
```

M3 is deliberately short because the core is pure; the MCP server is a thin adapter. M5 is where most of the long-tail work lives and is driven by which architectures you want to learn next.

## Open questions

- **Bun vs Node for the MCP binary**: develop with Bun, but publish the MCP server so `npx` on Node 22 works (no Bun-only APIs).
- **How far to go op-level**: the plan keeps `sdpa` and `ssd_scan` as primitives rather than decomposing to matmul/softmax. Decomposing further would let users invent new attention cores but makes the FLOP/memory model kernel-unaware. Revisit after M2.
- **Python dependency footprint**: `fla` and `mamba_ssm` need CUDA builds; keep them optional extras so the core runtime installs cleanly on CPU.
- **Tiny-track dataset**: FineWeb-Edu sample (closer to real pretraining) vs TinyStories (faster signal). Start with a 100M-token FineWeb-Edu shard.
