# TensorCAD — Roadmap

Goal: a node-based CAD tool for designing neural network architectures at the pretraining level, with checks, analysis, code generation, a local test bench, and an MCP server. It began with language models and now covers vision transformers and convolutional classifiers on the same machinery. Primary use: learning, validating, and testing designs. See [docs/index.md](docs/index.md) for the documentation.

Effort estimates assume one developer working with an AI coding assistant, part-time. They are ranges, not commitments.

## Status as of 2026-09-20

| Milestone | State |
|---|---|
| M0 Sketch (core IR, symbolic shapes, catalog, params) | **Done.** 23 presets, exact parameter match on 20 of them. |
| M1 Check (design rules, full analysis) | **Done.** 18 rules, drawn on the canvas where the work happens; FLOPs, KV cache, memory, throughput, cost, Chinchilla. |
| M2 Manufacture (PyTorch codegen, verification) | **Done.** Every generated model's parameter count matches PyTorch exactly, and the FLOPs estimate matches a profiler once the causal mask is accounted for. |
| M3 Agent (MCP server, CLI) | **Done.** 13 MCP tools over stdio, 6 CLI commands, and the live editor bridge: an agent's edits land on the canvas as it makes them and the human's come back. |
| M4 Test bench | **Done.** `tensorcad-runtime smoke-train` trains a scaled design on the local GPU and writes a run record; the editor's `Runs` tab opens several and draws their loss curves on one chart, naming anything that makes the comparison unfair. |
| M5 Advanced parts | **Every block the plan named is there.** Mixture of experts, latent attention, state-space and linear-attention blocks; DeepSeek-V3, Nemotron-H-8B, Jamba, Qwen3-Next and Gemma-3 reproduce exactly. What is left is presets for models whose configs are still moving, which is not a thing that finishes. |
| M6 Ship | **Done bar one step.** Documentation at [docs.tensorcad.dev](https://docs.tensorcad.dev/) and the editor at [app.tensorcad.dev](https://app.tensorcad.dev/), both static-asset Workers deployed from `main`; a release workflow that builds the engine, the MCPB bundle and the desktop binaries and cuts a GitHub release on a tag. The npm packages are prepared and verified on every tag and publish only when `NPM_TOKEN` is set. |
| Go engine | **Done.** The whole analysis is Go, compiled to WebAssembly, and the editor, the command line, the MCP server and the desktop shell all load the same module. The TypeScript it was ported from has been deleted; the golden files it wrote are the specification the engine is held to. |
| Editor UI | **Reworked against CAD convention.** Orthogonal wires, a grid with snap, schematic-style blocks, a model tree with locking, typed pins, a status bar, named refusals, a definition editor, a selectable tensor, a feature timeline, an SVG export of the sheet and a volume view whose stages are named. Twenty passes, each with what was wrong and what replaced it, in `docs/explanation/interaction-design.md`. |

Two suites, reading the same files. `go test ./...` in `packages/core-go` checks
the Go source; `bun test packages` checks the compiled module through the
JavaScript boundary, along with the command line, the MCP server and the editor.
Between them: every preset's symbol table, inferred shapes at two expansion
settings, the full analysis and the design-rule check at three operating points,
and the generated PyTorch byte for byte. `go run ./cmd/golden` rewrites those
files, deliberately and never as part of a test.

23 presets, 20 matching their published parameter count exactly and 3 within a
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

- `@tensor-cad/mcp` on the TypeScript SDK v2, stdio transport, file-mode `DocumentStore`; 15 tools with `outputSchema` and annotations; resources and prompts; see [docs/how-to/use-the-mcp-server.md](docs/how-to/use-the-mcp-server.md).
- `.mcp.json` for Claude Code; README snippets for Cursor and Claude Desktop.
- CLI (`tensorcad`) sharing the same command set.
- Live bridge: UI writes a session file; MCP server attaches over local WebSocket; ops flow both ways; `resources/updated` on human edits.

**Done when**: from Claude Code, "open the Llama-3-8B preset, make it a 4-expert MoE with top-2, and tell me the new active params" works end to end and the change appears on the canvas.

**Result**: both halves work. That edit takes Llama-3-8B from 8.03B dense to 24.94B total and 13.67B active, verified over a real stdio pipe, and with `TENSORCAD_BRIDGE=1` it lands on a running editor's canvas as it is made.

The bridge is not the design that was written down here. That one had a `LiveStore` beside the `FileStore`, both behind `DocumentStore`, with the agent proxying to the editor — which gives a design two homes and no rule for deciding which is right when they differ. What is there instead observes the one store the tools already write to and mirrors it outward; an editor's edit comes back through the same `apply` a tool call uses, revision check and undo log included. There is one document and one revision counter, so "who is right" is a question that never has to be answered.

Three things guard it, in order of how much they do: it binds 127.0.0.1; it refuses an upgrade whose `Origin` is not a localhost one, because the same-origin policy does not stop a page you visited opening a WebSocket to your own machine; and it wants a token, which a browser gets from a loopback-only endpoint because it cannot read the file the token lives in. It is off unless `TENSORCAD_BRIDGE=1` asks for it, so every CI job that runs this server still opens no port.

Two Bun quirks are written into the tests, because each cost an afternoon: its `ws` shim does not implement `unexpected-response`, so a refused upgrade is a socket that never opens rather than an error; and *nothing* written to a socket taken off an `upgrade` event is ever delivered, so the 401 that arrives under Node is swallowed. A raw handshake that does succeed and is then dropped without a close takes the whole test process down with it — no error, no stack, exit 127.

### M4 — Test bench (simulation) · 2–3 weeks

Actually train designs, small.

- Tiny-track scaler: shrink a design to a target param budget while keeping ratios; emits config + `train.py`.
- Runner in `tensorcad-runtime`: fixed dataset (FineWeb-Edu sample or TinyStories), fixed seeds, bf16, `torch.compile`; logs loss/tokens/s/peak memory to `runs/*.jsonl`.
- Run registry and compare view in the UI (loss vs tokens, loss vs wall-clock, memory); baseline designs.
- Verified on the local RTX 5080 (16 GB): a ~20M-param design trains in ~10 minutes.

**Done when**: two designs (GPT-2-style vs Llama-style at equal params) can be trained back to back from the UI and compared on one chart, reproducibly.

**Result**: a 30.1M-parameter GPT-2 shrunk by `scaleDesign` trains 500 steps on the RTX 5080 in 27 seconds at 151,213 tokens per second, loss 10.85 down to 3.23, 6.5 GiB peak. A Llama-shaped design of 48.2M reaches 3.53 in 200 steps.

The runs are now side by side. `smoke-train` writes a record beside the step log it was already writing, and the editor's `Runs` tab opens either — the record when there is one, the `.jsonl` when the run stopped early, which is often the run you most want to look at. The curves go on one chart.

Two decisions in that chart are the whole point of it. It plots against **tokens**, not steps: a step is not a fixed amount of work, and two designs at the same batch size and different sequence lengths see different amounts of text per step, so plotting against steps quietly credits the longer one. And it **names what makes the comparison unfair** before drawing it — sequence length, batch, corpus, seed, learning rate, dtype. Pointed at the two runs actually on this machine it says "sequence length differs: 256, 512; batch size differs: 8, 16", which is the difference between an architecture result and a run that saw thirty-three times the tokens.

A field only one run reports is not a difference. A step log carries no seed and the record beside it does; one value and one silence is not two values, and a warning nobody can act on is how people learn to stop reading warnings.

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
- Jamba-v0.1, reproducing 51,570,323,328 exactly, which needed Mamba-1: a `selective_scan` primitive and a `mamba_block` around it. Not a variant of the Mamba-2 block already here — Mamba-1 reads its timestep at a rank and projects it back up, and that matrix in the middle is one the other does not have. Two periods run at once in the stack, attention every eighth layer and a mixture every second, so the repeating unit is eight layers and thirty-two is four of them. Twenty-eight of the thirty-two hold no cache that grows with the sequence. Jamba also norms the timestep and both gates before the scan, which the original does not: [256], [16] and [16], and being 8,064 short across the model is how that was found. The emitted scan runs a real forward pass and exports cleanly.
- Qwen3-Next-80B-A3B, reproducing 79,674,391,296 exactly — the first linear-attention hybrid, and the one the roadmap had been waiting on a settled config for. Three layers in four are gated DeltaNet, so only twelve of the forty-eight hold a cache that grows with the sequence. Building it against the real weights found three things the catalog had wrong or missing: a gated DeltaNet can have more value heads than key heads (Qwen3-Next has twice as many, which is most of why its input projection is 12,288 wide); its output norm is per head, [128], where the expansion made it as wide as every head together, though the block's own documented formula had said `v_head_dim` all along; and attention can gate its own output, which is why `q_proj` is [8192, 2048] rather than [4096, 2048]. A shared expert can be gated too, by a single learned direction — [1, 2048], 98,304 parameters across the model, and exactly the sort of row an estimate drops.
- Gemma-3-27B, reproducing 27,009,346,304 exactly. Not from a figure on a card — the card says 27.4B, which includes a SigLIP vision tower this does not model — but from the model's own safetensors headers, read with a range request rather than a download. Checking the arithmetic against those headers group by group is how `qk_norm` is known to be on: the 15,872 it came up short is exactly sixty-two layers times two 128-wide norms. Five layers in six attend within a 1024-token window, so at 128k context the cache is 10.41 GiB where treating every layer as global would give 62.
- Tensors as first-class objects in the drawing, the second third of E7. A wire is now
  selectable, and selecting one opens an inspector for the *tensor*: its shape, its dtype, the
  block that produced it, every block that reads it, and its share of the activation memory —
  with every other segment of the same net lit, because they are one tensor and not several.
  The analysis had always attributed activation memory to tensors rather than to blocks, but
  only reported it per block, which is the same rows under a coarser key for a plain
  transformer and a real loss everywhere a block fans out. `activationsByTensor` is the finer
  one. Nemotron-H's `split` holds 290 MiB across three output pins — 128, 160 and 2 — and the
  block's own number answers neither which of them is the big one nor what dropping one would
  save. A test pins both halves of that: where the finer key earns itself, and where it
  deliberately says nothing new.
- Configurations, the part of E7 that had a shape: `doc.configurations` are named sets of symbol values and `doc.active` says which the design is built at, so one architecture can carry four sizes rather than four files carrying one each. A configuration overrides symbols and nothing else, says only what differs — so `F` written as `4*D` follows an override of `D` rather than freezing — and never touches the document's own symbols, which is what makes switching twice end where it started. A test builds one design at all four GPT-2 sizes and reproduces every preset exactly.
- Operations as first-class objects, the last third of E7 — and with it E7 entire. `commit` was
  always the single door every document change went through; it took a closure, and a closure
  can be called and nothing else. It now takes a value: `{ kind: "setParam", path, key, value }`,
  with `applyEdit` the only thing that knows how to perform one. The history is a base document
  and a list of these, and the drawing is the fold. So a step can be **suppressed** — taken out
  of the middle while everything after it replays on top of what is left — which is the thing a
  stack of documents could never do, because by the time an edit is a document the operation
  that made it is gone. Seventeen call sites and one dispatcher; `cache[i]` is the document
  after `i` steps, so the ordinary edit is one apply and only a suppression replays a tail.
  Suppressing the step that added a block leaves the step that wired it with nothing to wire,
  which is not a bug to prevent but what taking a step out of the middle means: the step is
  marked, skipped, and says what it could not find. Writing that found three functions in
  `ops.ts` reporting success for work they had not done — `removeNode` filtering a node that was
  not there, `disconnect` an edge that was not there, `moveNodes` writing positions for paths
  that had gone — invisible while the result was only ever thrown away, and wrong the moment
  something read it. An agent's edit is a `replaceDoc` step, so what arrives over the live
  bridge is a row in the timeline like any other: labelled, jumpable, and able to be taken back
  out of the middle.

Remaining:
- Presets for 2026 models as their configs stabilize.

### M6 — Ship

- **Done.** The docs site: `mkdocs.yml` and a workflow that builds on every push and deploys from `main` to [docs.tensorcad.dev](https://docs.tensorcad.dev/), with GitHub Pages as a mirror. The editor is served the same way, from `packages/ui/wrangler.jsonc`, at [app.tensorcad.dev](https://app.tensorcad.dev/) — static assets and nothing else, the engine being WebAssembly that runs in the tab. The pages were already organised by Diátaxis and read the same in the repository; what a directory of Markdown could not give them is a navigation that states that split rather than leaving it implied, and a search box, which is what a reference is useless without. The build runs `--strict`, so a cross-reference to a page that does not exist fails rather than warning — which found four links to files outside `docs/` on the first run. The second guided tour is `Tune small, run big`: shrink a published architecture until it trains on one card, sweep there, carry the answer up the μP ladder, and price the real run, with every figure pasted from the command above it.
- **Block documentation done:** every block has a summary, every primitive that counts parameters gives the formula it counts them by, every parameter says what it means and every port declares what it carries — seventy-seven parameters said nothing, which is what the inspector showed on hover and what `get_block` answered with. Tests hold all four, because the ones that go undocumented are the ones whose names read plainly to whoever wrote them.
- **Done.** `server.json` for the MCP registry, checked against the registry's own schema and pinned against `package.json` by a test. An MCPB bundle for Claude Desktop, built for Node with the engine beside it and started under Node before it is attached to a release.
- Publish `@tensor-cad/engine` and `@tensor-cad/mcp` to npm, and `server.json` to the registry. **Prepared, not published.** Neither package was publishable as it sat: `main` pointed at TypeScript that Node cannot import, the server's `bin` carried a Bun shebang, and the dependency between them was written `workspace:*`, which npm rejects. `bun run build:dist` emits the publishable form, packs it, installs both tarballs into an empty directory, imports them under Node and starts the server — which found that the bundler tree-shakes the Go runtime's side-effect import and that the engine's wasm path assumed the repository's layout. The release workflow has a gated `npm` job: it builds and verifies on every tag, checks the tag against all three version numbers, and publishes only when `NPM_TOKEN` is set, saying in the run summary when it is not. Setting that secret and cutting a tag is the whole remaining step.
- Optional: MCP Apps canvas preview for Claude Desktop/Cursor.

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
