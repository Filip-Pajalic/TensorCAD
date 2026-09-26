# TensorCAD — Roadmap

Goal: a node-based CAD tool for designing neural network architectures at the pretraining level, with checks, analysis, code generation, a local test bench, and an MCP server. It began with language models and now covers vision transformers and convolutional classifiers on the same machinery. Primary use: learning, validating, and testing designs. See [docs/index.md](docs/index.md) for the documentation.

Effort estimates assume one developer working with an AI coding assistant, part-time. They are ranges, not commitments.

## Status as of 2026-09-25

| Milestone | State |
|---|---|
| M0 Sketch (core IR, symbolic shapes, catalog, params) | **Done.** 28 presets, exact parameter match on 25 of them. |
| M1 Check (design rules, full analysis) | **Done.** 23 rules, drawn on the canvas where the work happens; FLOPs, KV cache, memory, throughput, cost, Chinchilla. |
| M2 Manufacture (PyTorch codegen, verification) | **Done.** Every generated model's parameter count matches PyTorch exactly, and the FLOPs estimate matches a profiler once the causal mask is accounted for. |
| M3 Agent (MCP server, CLI) | **Done.** 19 MCP tools over stdio, 9 CLI commands, and the live editor bridge: an agent's edits land on the canvas as it makes them and the human's come back. |
| M4 Test bench | **Done.** `tensorcad-runtime smoke-train` trains a scaled design on the local GPU and writes a run record; the editor's `Runs` tab opens several and draws their loss curves on one chart, naming anything that makes the comparison unfair. |
| M5 Advanced parts | **Every block the plan named is there.** Mixture of experts, latent attention, state-space and linear-attention blocks; DeepSeek-V3, Nemotron-H-8B, Jamba, Qwen3-Next and Gemma-3 reproduce exactly. What is left is presets for models whose configs are still moving, which is not a thing that finishes. |
| M6 Ship | **Done.** Documentation at [docs.tensorcad.dev](https://docs.tensorcad.dev/) and the editor at [tensorcad.dev](https://tensorcad.dev/), both static-asset Workers deployed from `main`. `bun run release` moves every version and tags `main`; the tag builds the engine, the MCPB bundle and the desktop binaries, publishes `@tensor-cad/engine`, `@tensor-cad/mcp` and `@tensor-cad/ui` to npm, then lists the server in the MCP registry, and cuts a GitHub release. |
| Go engine | **Done.** The whole analysis is Go, compiled to WebAssembly, and the editor, the command line, the MCP server and the desktop shell all load the same module. The TypeScript it was ported from has been deleted; the golden files it wrote are the specification the engine is held to. |
| Editor UI | **Reworked against CAD convention.** Orthogonal wires, a grid with snap, schematic-style blocks, a model tree with locking, typed pins, a status bar, named refusals, a definition editor, a selectable tensor, a feature timeline, an SVG export of the sheet and a volume view whose stages are named. Twenty-seven passes, each with what was wrong and what replaced it, in `docs/explanation/interaction-design.md`. Every block and wire has a spoken name and the hierarchy can be walked by keyboard — Enter into a block, Escape back out onto it — and `bun run test:browser` drives the built editor in headless Chrome with real key events, in CI, because the unit suite has no browser to see what React Flow does with a key first. |
| M7 Legibility | **Done.** The sheet says *grouped-query attention* rather than `gqa_attention`, every part explains itself on hover, a key names every letter and mark, shapes can be read as English, the plumbing can be left out the way a published figure leaves it out, each preset's own paragraph is on screen in a browsable library, the editor opens on a model small enough to see every number of, and a walkthrough narrates whatever design is open — with its numbers, changing when it changes. [docs/explanation/legibility.md](docs/explanation/legibility.md). |
| M8 Real values | **Done.** `tensorcad-runtime trace` trains `nano-sort` to sort and records one run; the volume view draws its real values, the hover readout names each cell, and the walkthrough quotes the run. Everything else still draws decoration, labelled as such. |
| M9 Your own values | **Done.** Any design under a million parameters with a token embedding can be traced — trained to sort when its vocabulary is small enough, run as initialised and labelled untrained when it is not — and the trace is loaded with File > Load a trace, or made and loaded in one step from the desktop app. Rotary and grouped-query attention are recomputed and checked like nano-sort's. The desktop's *Verify against PyTorch* and *Smoke train* run the same way: a sentence back, and a loss curve on the Runs chart. |
| M10 Attention variants | **Done.** The analysis and the generated code describe the same kernel: softcapping and windows are counted as FlashAttention runs them and generated to use it. A design can now write a mask and a score expression on the fused primitive, FlexAttention-style, and see them counted, checked, drawn as a block mask and generated as `flex_attention`; a causal window is now counted at its width rather than half of it. BLOOM-7b1's ALiBi is a score expression and gpt-oss-20b brings attention sinks, both reproducing their published parameter counts exactly and both held against Hugging Face's own construction of what they add. Differential attention is two fused attentions and a combine, held against Microsoft's reference, and attention can be written out on purpose, with talking heads, at a cost a rule states. T5's relative bias came last, through M11's second sequence. [docs/explanation/attention-variants.md](docs/explanation/attention-variants.md). |
| M11 Encoder–decoder | **Done.** A second sequence, and cross-attention to it, so T5 can be drawn and M10 can close: a source length `S` beside `T`, each block counted at the tokens of its own stream, cross-attention, a repeat that hands every layer the same tensor, and expressions that read a tensor — T5's shared relative-bias table, which is Hugging Face's to the bit and learns through FlexAttention. `t5-small` and `flan-t5-base` reproduce their published counts exactly and compute what Hugging Face's T5 computes, weight for weight. Every decoder-only number is held unchanged. [docs/explanation/encoder-decoder.md](docs/explanation/encoder-decoder.md). |
| M13 The editor for someone new | **Done.** One top bar, with the account in the editor's own corner, and a Share button that works everywhere: through a store when signed in to one, and with the design carried in the link itself anywhere else. Every parameter has a plain label, with its code name beside it, the rare ones under Advanced and the ones that do not apply out of the way. A first visit opens on the drawing with nothing over it, four operating fields, four tabs and a Start here. |
| M12 Packed sequences | **Done.** Pretraining packs documents into one sequence and masks attention between them, which cuts Llama-3-8B's attention at 8k by eight times, and the engine assumes a sequence is one document. A documents input a mask can read, packing as a condition of training with a length distribution rather than a length, and a block mask built per batch. Phase 1 measures it: Llama-3-8B with its document mask at 8k in rows of 1,024-token documents keeps 491 keys a query, not 4,096, and trains 11% cheaper, while serving does not move. Phase 2 generates and verifies it: one block mask a batch shared by every layer, one document's tokens moving no other's outputs in PyTorch, and the kernel's whole blocks counted, 1.37 times the scores kept at 1,024 tokens, within 0.1% of FlexAttention's own count. Phase 3 restarts the positions at every document, and found that a rotary model under the mask computes the same either way while a learned position table does not. Phase 4 puts it in the editor: a packing control, the packed figures in the readout, the mask drawn over a sampled row, and two rules. [docs/explanation/packed-sequences.md](docs/explanation/packed-sequences.md). |

Two suites, reading the same files. `go test ./...` in `packages/core-go` checks
the Go source; `bun test packages` checks the compiled module through the
JavaScript boundary, along with the command line, the MCP server and the editor.
Between them: every preset's symbol table, inferred shapes at two expansion
settings, the full analysis and the design-rule check at three operating points,
and the generated PyTorch byte for byte. `go run ./cmd/golden` rewrites those
files, deliberately and never as part of a test.

28 presets, 25 matching their published parameter count exactly and 3 within a
stated tolerance, and every one of them confirmed against PyTorch 2.11 by
instantiating the generated model. Not all are language models:
`ijepa-vit-h14` is a vision transformer and `alexnet` a convolutional
classifier, on the same machinery.

## Principles

1. **Own the IR.** The design document is our JSON format; React Flow state is a view. (ComfyUI lesson.)
2. **One pure core.** Schema, validation, analysis and codegen are pure functions with no I/O — Go, compiled to WebAssembly — and the editor, the CLI, the MCP server and the desktop shell all load the same module. It began as TypeScript; the golden files that TypeScript wrote are what the Go is held to.
3. **Analyze primitives, present composites.** Param/FLOP/memory formulas live on 33 primitive ops; composite blocks (GQA attention, SwiGLU MLP, MoE layer) are subgraphs of primitives with exposed parameters. New blocks need no new math.
4. **Verify against PyTorch.** Every estimate can be cross-checked by instantiating the generated model on the meta device.
5. **Regression-test against real models.** Every preset carries the figure its authors published, and the tests hold the analysis to it — 21 of 24 to the parameter. It started as the 15-model table in [docs/reference/analysis-math.md](docs/reference/analysis-math.md).
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
- Logit softcapping, on the output logits and on the attention scores. The one on the scores was counted as eager, on the belief that no fused kernel could cap — 199 GiB of activations for Gemma-2-9B at 8k. FlashAttention has capped inside the kernel since 2.6, so M10's first phase counts it fused, 73 GiB, and the generated model uses that kernel when it is installed.
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

- **Done.** The docs site: `mkdocs.yml` and a workflow that builds on every push and deploys from `main` to [docs.tensorcad.dev](https://docs.tensorcad.dev/), with GitHub Pages as a mirror. The editor is served the same way, from `packages/ui/wrangler.jsonc`, at [tensorcad.dev](https://tensorcad.dev/) — static assets and nothing else, the engine being WebAssembly that runs in the tab. The pages were already organised by Diátaxis and read the same in the repository; what a directory of Markdown could not give them is a navigation that states that split rather than leaving it implied, and a search box, which is what a reference is useless without. The build runs `--strict`, so a cross-reference to a page that does not exist fails rather than warning — which found four links to files outside `docs/` on the first run. The second guided tour is `Tune small, run big`: shrink a published architecture until it trains on one card, sweep there, carry the answer up the μP ladder, and price the real run, with every figure pasted from the command above it.
- **Block documentation done:** every block has a summary, every primitive that counts parameters gives the formula it counts them by, every parameter says what it means and every port declares what it carries — seventy-seven parameters said nothing, which is what the inspector showed on hover and what `get_block` answered with. Tests hold all four, because the ones that go undocumented are the ones whose names read plainly to whoever wrote them.
- **Done.** `server.json` for the MCP registry, checked against the registry's own schema and pinned against `package.json` by a test. An MCPB bundle for Claude Desktop, built for Node with the engine beside it and started under Node before it is attached to a release.
- **Done.** `@tensor-cad/engine`, `@tensor-cad/mcp` and `@tensor-cad/ui` publish to npm on every tag, with provenance. Neither of the first two was publishable as it sat: `main` pointed at TypeScript that Node cannot import, the server's `bin` carried a Bun shebang, and the dependency between them was written `workspace:*`, which npm rejects. `bun run build:dist` emits the publishable form, packs it, installs both tarballs into an empty directory, imports them under Node and starts the server — which found that the bundler tree-shakes the Go runtime's side-effect import and that the engine's wasm path assumed the repository's layout. The release workflow has a gated `npm` job: it builds and verifies on every tag, checks the tag against all three version numbers, and publishes only when `NPM_TOKEN` is set, saying in the run summary when it is not. `bun run release bump` and `bun run release tag` are the two halves of cutting one, because `git push --follow-tags` silently pushes nothing for a lightweight tag.
- **Done.** The MCP registry: after npm, a `registry` job signs in with the workflow's own GitHub OIDC token and publishes `server.json`, having waited for npm to serve the version, since the registry reads the package's `mcpName` before it accepts the entry. The namespace is `io.github.Filip-Pajalic`, capitals included: the registry grants the owner's own casing and compares case-sensitively, so the lowercase namespace the file was written with would have been refused ([registry#689](https://github.com/modelcontextprotocol/registry/issues/689)). A test holds the namespace to the repository URL.
- Optional: MCP Apps canvas preview for Claude Desktop/Cursor.

### M7 — Legibility · Done

The analysis is right and the drawing is hard to read. Twenty-three architectures
are reproduced to the parameter, and the sheet labels them with the identifiers
the engine dispatches on — `gqa_attention`, `rmsnorm`, `lm_head`, `boundary_in` —
writes their shapes as `B T (H dh)` with no key anywhere, opens by default on an
eight-billion-parameter model, and renders the `meta.notes` paragraph each preset
carries about itself in no place at all. Meanwhile the volume view, being a port
of Bycroft's visualisation, calls the same parts *Token Embed* and *Attention
Matrix*: the vocabulary exists in this repository and one view has it.

Eight phases, ordered and estimated in
[docs/explanation/legibility.md](docs/explanation/legibility.md): a plain name on
every block, the summary reaching the drawing, a notation key, shapes in English,
the preset's own prose on screen, a model small enough to see every number of,
plumbing out of the drawing, and — the one that earns the comparison — a
walkthrough that narrates whatever design is open rather than a recorded one.

All eight phases are done. The twenty-first and twenty-second passes in
[interaction-design.md](docs/explanation/interaction-design.md) record what
happened, including five silent bugs the work walked into — a block a design
defines for itself could be listed in the palette, dragged onto the sheet, and
simply not appear — and the three rules it took to stop `dh` ("Head dimension")
rendering as "128 heads", which was false on every attention block of every
design.

The default document is now `nano-sort`: 85,728 parameters, GPT-2's structure
exactly, and small enough that every weight fits on the screen. The last preset
loaded is remembered, so only a first visit lands on the toy.

### M8 — Real values · Done

M7 made the drawing say what each block *is*. It still cannot say what one
*does* to a particular input, which is the half of Bycroft's visualisation that
makes it land: a token goes in, and every number it becomes is on screen.

Nothing here computes that yet, and the gap is precise. `tensorcad-runtime
smoke-train` trains a design and writes loss curves and throughput, and keeps no
weights. The volume view fills every cell from `speckle()`, a hash of the cell's
own coordinates, so what looks like data is decoration. And `nano-sort`, the
default design since M7, is exactly the size where a full trace is small: 85,728
parameters and eleven positions is a few hundred kilobytes of floats.

1. **A trace from the runtime.** Train `nano-sort` to sort — minGPT's task, three
   letters, six to read and five to write — then run one input through it and
   write every weight and every activation to a file, keyed by the block paths
   the editor already uses. Verified the way the rest of the runtime is: the
   attention matrix, which the fused kernel never materialises, is recomputed
   and multiplied back into what the kernel produced, and must agree to `1e-4`.
   *Done:* `tensorcad-runtime trace`; it agrees to `3e-8`.
2. **The volume view reads it.** Where a trace covers a tensor, the shader
   samples its values instead of `speckle()`. Where it does not, the speckle
   stays and the legend says it is decoration — a view that mixes real numbers
   with invented ones has to say which is which. *Done:* every box names the
   tensor it is a picture of, and the tests recompute the input embedding and
   every head's output from the boxes beside them, which fails on a transposed
   read.
3. **The walkthrough quotes it.** "Each token becomes a vector of 48 numbers"
   becomes *these* 48 numbers, and the attention step says which earlier token
   each position actually attended to. *Done.*

A trace lives in `packages/ui/src/three/traces`, one committed file for
`nano-sort`, regenerated deliberately by `bun run trace` like the goldens and
loaded only when something asks for it. It is shown only on a design that still
generates the `model.py` it was made from — the hash of that file is the
fingerprint, because it is the whole computation: an edit that changes what the
model does changes it, and one that does not leaves the numbers true. Tracing
an arbitrary edited design is not something a static site can do, and the view
says so rather than pretending.

**Done when** `nano-sort`, opened fresh, shows the real numbers for a real input
in the volume view; the attention matrix shows which letters attended to which;
and the walkthrough can name them. Anything larger than a trace covers still
draws, honestly labelled as structure only.

### M9 — Your own values · Done

M8 put real numbers in the volume view for one design, made ahead of time and
committed, because a static site cannot run Python. Every other design was
drawn with decoration. The desktop app can run Python, and anybody with the
runtime installed can run it from a terminal, so the missing pieces were a
trace that works on a design other than the one it was written for, and a way
into the editor for one that was not committed.

1. **A trace of any small design.** `tensorcad-runtime trace` no longer reads
   the design's symbol names, which are its author's choice. The vocabulary is
   the token embedding's size; an attention block is anything holding the four
   projections; its head layout is what the generated class says it was built
   with. A design that can learn to sort in seconds is trained to; anything
   else is run exactly as initialised and says so — `untrained`, in the legend,
   in the picture's label and in every sentence the walkthrough adds, because an
   untrained model's attention is close to even and a reader who did not know
   would draw the wrong conclusion from it. Rotary embedding and grouped-query
   attention are recomputed and checked, to 3e-8 on a small Llama. Retraced, the
   committed nano-sort trace comes out identical.
2. **Loading one.** File > Load a trace reads a trace file. It shows on the
   design whose generated model it fingerprints and on nothing else, like the
   committed one; a trace of another design is kept for when that design is
   open, and the message says so rather than appearing to do nothing. When a
   trace is shorter than the drawing — an untrained trace is 32 positions, and
   most designs default to thousands — the legend says so and offers to draw it
   at the trace's length.
3. **Making one from the desktop app.** Design > Trace this design generates
   the model, runs the trace in a scratch folder of the app's own, and loads the
   result into the volume view. The shell reads back only a trace it wrote.

Tested at each layer: the runtime against a rotary grouped-query design, with
the fingerprint it writes compared against the engine's model on the other side
of the language boundary; the editor's registry and wording in unit tests; and
File > Open then File > Load a trace in headless Chrome, with a design that is
not nano-sort.

Since done: the desktop app's *Verify against PyTorch* and *Smoke train*,
which had only said what to do first, run the way *Trace this design* does.
Verify answers in a sentence — PyTorch's count against the design's, whether a
forward pass ran, whether it exports — and a smoke run puts its loss curve on
the Runs chart. All three share one staging step and one reader, which returns
only a JSON result a job of the app left in its own folder. The desktop
window's TypeScript is now type-checked in CI; it had never been, and ten
errors had collected there unseen.

Since done too: a trace outlasts the session. The editor keeps the last eight
it has seen on a shelf in the browser's own storage, keyed by the fingerprint it
is matched by, so a design opened again shows its values with nothing to load —
in a tab, the hosted editor and the desktop window alike. The desktop also keeps
one beside the design file, `<name>.trace.json`, written when it is made and
when the design is saved and read back when the design is opened, so the pair
can be copied to another machine; a trace of an earlier version of the design
is loaded but said to be one. The browser suite reloads the page, opens only
the design, and finds its values; with the shelf taken away, that check fails.

### M10 — Attention variants · Done

The open question below, *how far to go op-level*, has a proposal:
[Attention variants](docs/explanation/attention-variants.md). Attention stays one
fused primitive, and a design says what it does to each score and which scores
count, as two small expressions — the shape PyTorch's FlexAttention takes and
compiles to a fused kernel. That covers ALiBi, relative position bias, prefix-LM,
document masking and softcapping without giving up the memory accounting that
makes the attention figures worth reading; sinks and differential attention get
a parameter and a primitive, and talking heads, which genuinely needs the score
matrix, gets an explicitly eager block.

Its first phase is a correction rather than a feature. Softcapping is counted as
eager because no fused kernel was thought to support it — FlashAttention has
since 2.6 — so Gemma-2-9B at 8k is counted at 198.9 GiB of activations where a
fused kernel keeps 72.9. And windowed attention is counted as a kernel that
skips the blocks outside the window while the generated code computes every
score under a dense mask. The analysis and the code must describe the same
kernel before anything is added.

1. **The numbers and the code agree** — softcap fused, windows block-sparse,
   verified by the runtime. *Done:* a window or a cap is generated as
   `fused_attention`, which calls FlashAttention on CUDA in half precision when
   it is installed and otherwise computes the same numbers the way it always
   did, warning once on a GPU. `SDPA-03` is a note naming the kernel, and
   Gemma-2-9B at 8k is counted at 72.9 GiB of activations, not 198.9.
   FlexAttention could not be the target yet: uncompiled it is unfused, and on
   Windows it does not compile at all, for want of a C++ compiler on the CPU
   and of Triton on CUDA; PyTorch's FLOP counter also refuses it, which the
   runtime's verification depends on. It is phase 2's problem, where it is
   needed.
2. **Mask and score expressions** on `sdpa`, parsed, costed and printed by the
   engine, edited in the inspector. *Done:* `mask` and `score` on `sdpa`,
   `gqa_attention` and `transformer_block`, in a small language with Python's
   operators over positions, the head and the design's symbols
   ([reference](docs/reference/attention-expressions.md)). A mask is counted
   by evaluating it; a score by its arithmetic. The inspector shows the
   expression as the engine understood it and the block mask it makes at the
   operating point. `SDPA-04` and `SDPA-05` catch a mask that keeps nothing or
   leaves a query nothing. Generated as FlexAttention's `mask_mod` and
   `score_mod`, compiled on CUDA and applied eagerly elsewhere, which a test
   holds against `flex_attention` itself. Treating `causal` and `window` as the
   masks they are found that a causal window was counted at half its width:
   Gemma 2, Gemma 3 and Mistral's windowed layers are now counted at
   `W - W²/2T` keys a query.
3. **Presets that need them** — ALiBi, relative bias, gpt-oss. *ALiBi done:*
   `bloom-7b1`, with no positions of any kind, its ALiBi written as the score
   expression `score - 2 ** (-8 * (h + 1) / heads) * (q - kv)`. It reproduces
   the 7,069,016,064 parameters Hugging Face reports, in the analysis and in
   PyTorch, and its FLOPs match the profiler's. The bias is held against a
   transcription of Hugging Face's own `build_alibi_tensor`: BLOOM adds slope ×
   key position rather than slope × distance, which differs by a constant along
   each row, and the weights agree to 1e-6. MPT-7B was the other candidate; its
   original repository is no longer public. *gpt-oss done:* `gpt-oss-20b`,
   alternating a 128-token band with full attention, 32 experts with four
   active, biases everywhere, and sinks. It reproduces the 20,914,757,184
   parameters Hugging Face reports in the analysis and in PyTorch, and OpenAI's
   3.61B active is the design's non-embedding active count. YaRN and the
   experts' clamped SwiGLU are recorded and not generated. *T5 done*, through M11: `t5-small` and `flan-t5-base`, each stack's learned `[buckets, heads]` table read by every layer's score expression. Both reproduce their published counts exactly, and each is held weight for weight against a transcription of Hugging Face's T5.
4. **Sinks, differential attention, and an eager block** for what is not a score.
   *Sinks done:* a learned score per head on `sdpa`, `gqa_attention` and
   `transformer_block`, counted as `heads` parameters and generated through
   FlexAttention — the output rescaled by `sigmoid(lse - sink)` — or, the long
   way, as one more column in the softmax, which is Hugging Face's form; a test
   holds the two against each other and against a transcription of Hugging
   Face's gpt-oss attention. The scaler, asked to shrink a design that repeats a
   pair of layers, now keeps the pairs whole. *Differential attention done:*
   `diff_attention`, and `attention: diff` on `transformer_block` — two fused
   attentions over shared values, a `diff_combine` that owns lambda's four
   vectors, a per-head RMSNorm and a `scale`, which is `softmax(Q1K1)V −
   λ·softmax(Q2K2)V` by linearity, so the memory is still the fused kernel's.
   There is no released model to regress against, so the arithmetic is pinned
   by tests, and the generated module is held against a transcription of
   Microsoft's reference given the same weights, to 3.6e-7. Building it found
   that `sdpa` counted the value product at the key width: DeepSeek-V3's
   attention FLOPs were a sixth too high, and now match the profiler.
   *Eager block done:* `eager_attention`, attention written out as scores,
   softmax and weighted values, with `head_mix` for talking heads; opted into
   with `written_out` or `talking_heads`, never by a preset. Every score is
   counted, the matrices kept are counted — activation memory now knows a
   tensor can be as long as the sequence twice — and the `eager-attention` rule
   says how many bytes that is at the operating point. Held against PyTorch's
   fused attention at the identity and the paper's formulation with the mixes
   learned. That finishes phase 4; T5, which needed encoder–decoder support and
   a learned table inside a score, was M11.

### M11 — Encoder–decoder · Done

M10's finishing line includes a relative-bias preset, and the relative-bias
model is T5, an encoder–decoder. The engine assumes one sequence in five
places: two runtime symbols, `B` and `T`; per token meaning per token of the
one stream; attention's keys as long as its queries; a repeat that carries
every input from one layer to the next; and expressions that read positions
but no tensors. The proposal, [Encoder–decoder](docs/explanation/encoder-decoder.md),
adds a second sequence without changing what any existing number means:

1. **The second sequence** — `S` beside `T`, the operating point's source
   length, and each block counted at the tokens of the stream it runs on.
   Every golden unchanged is the test. *Done:* not one golden byte moved; each
   stream is reported per token of its own and one example whole; a source is
   cached per request; and a block's declared `T` now binds to whichever
   sequence reaches it, so every block in the catalog runs along a source
   unchanged, while the source and the target meeting in one block is still
   an error.
2. **Cross-attention** — `sdpa` with keys `S` long, not causal, and a cache
   computed once per request; a repeat's *broadcast* inputs, which every layer
   reads unchanged; `scale` on `sdpa`, since T5's attention is unscaled; and
   generated code and verification with two inputs. *Done:*
   - `cross` on `sdpa`, and a `cross_attention` composite.
   - `cross_attention` on `transformer_block`, which adds a `memory` input.
   - A stack's input that its layer does not give back reaches every copy
     unchanged, with no new syntax, and is charged once in memory.
   - `forward(self, src, tgt)`, and a runtime that builds, profiles and exports
     both inputs.

   A small encoder-decoder's profiler count per target token equals the
   analysis's exactly. Along the way, an encoder turned out to cache nothing,
   and a pinned-shape port stopped taking part in batch binding.
3. **Tensors in expressions** — a score reading a port by name,
   `score + rel(t5_bucket(kv - q, 32, 128, true), h)`, with the table owned by a
   `position_bias` block outside the stack and FlexAttention capturing it. The
   same mechanism is what document masking will want. *Done:*
   - Any name called like a function is a table, read in a score; a mask
     cannot read one, because a mask is counted by evaluating it.
   - `t5_bucket` is a built-in, and `position_bias` a `[buckets, heads]` table.
   - Every block that carries a score down to its attention grows the input
     the score reads, and a port can take any shape.
   - A score that reads a table is generated as a factory that takes it.

   The bias equals Hugging Face's exactly, two-sided and one-sided, and a
   gradient reaches the table through `flex_attention` as well as the unfused
   form — the question the proposal left open. A small T5-like model verifies
   in PyTorch, count, FLOPs and export.
4. **The presets** — `t5-small` (60,506,624; its checkpoint holds 256 more, a
   table the model never reads) and `flan-t5-base` (247,577,856), exact.
   *Done:* both reproduce their counts in the analysis and in PyTorch, their
   profiled FLOPs match, and a transcription of Hugging Face's T5, given the
   generated model's own weights under Hugging Face's names, computes the same
   logits — using every weight exactly once. What they needed:
   - `scale` on `gqa_attention` and `transformer_block`, down to both
     attentions, since T5's is unscaled; `norm_eps`, since its norms add 1e-6.
   - `tied` on `embedding`: the decoder reads the encoder's table.
   - Not a new switch for t5-small's scaled tied head: the `scale` block
     before it says `1/sqrt(D)`.

   Three things surfaced. A symbol only an attention expression reads was
   reported unused. The walkthrough narrated two stacks as one, and every
   decoder's context cache as 0 B. And the Hoffmann loss differed between the
   goldens and the editor in its last bit, because `math.Exp` is assembly on
   amd64 and plain Go in WebAssembly; it is reported to twelve decimals now.

**Done when** both presets reproduce in the analysis and in PyTorch, their bias
matches Hugging Face's construction of it, a profiler agrees with a small
encoder–decoder's FLOPs, and every decoder-only golden is unchanged. M10 then
closes.

### M12 — Packed sequences · Done

Pretraining concatenates documents to fill the sequence, and Llama 3 masks
attention between them. The engine assumes a training sequence is one
document: a mask reads positions but never a tensor, training and serving are
measured at the same attention, and the generated attention keeps its block
masks as though they never changed. The proposal,
[Packed sequences](docs/explanation/packed-sequences.md), makes packing a
condition of training:

1. **Documents, measured** — a `documents` role on `input`, a mask that reads
   it (`doc(b, q) == doc(b, kv)`), and packing in the operating point as
   documents of mean length `μ` and spread `c`. The mask is still counted by
   evaluating it, with a packing drawn for each sampled row. Packing moves the
   training figures and not the serving ones. With packing off, every golden
   is unchanged. *Done:* a documents `role` on `input`; `packing` in the
   engine, `--pack` on the command line, and `packing` on the MCP tools;
   `flops.packed`, which the training cost is counted from; and a `documents`
   rule that follows the wire back to an input that says it holds documents.
   A row cut from a stream begins inside a document, so fixed 1,024-token
   documents at 8k keep 491.1 keys a query rather than 512, and that is what
   the analysis is held to. The sampler is within half a percent for fixed
   lengths and about two for spread-out ones, measured over 1,024 rows with
   keys banded back from each query. No analysis, rules or codegen golden
   moved.
2. **Generated and verified** — the mask as a factory over the documents, one
   block mask per forward pass rather than one cached per step, and a test that
   one document's tokens move no other document's outputs. FlexAttention's own
   block count is held against the engine's. *Done:* the block mask is built for
   the first layer that asks and shared by the rest, then replaced by the next
   batch's; the runtime draws packings the way the engine does; and
   `flops.packed.fwdAttentionBlocks` counts whole blocks, decided by interval
   arithmetic on the mask, which is exact for documents. In PyTorch, one
   document's tokens move no other's outputs by exactly zero, flex_attention
   agrees with the eager form to zero, and over 256 rows of the runtime's
   packing the engine's block figure is within 0.1% of what create_block_mask
   says a kernel computes. A stream's documents cut blocks at every boundary:
   1.37 times the scores kept at 1,024 tokens, 2.48 at 256, not the 1.12 and
   1.49 of documents aligned to blocks.
3. **Positions that restart** at each document, as Hugging Face's flattening
   collator does. *Done:* a `positions` role, read by `rope` (through
   `gqa_attention` and `transformer_block`) and by `pos_embedding`, and an
   `input-roles` rule that takes over from `documents`. Packed as the collator
   packs, a row computes what its documents compute alone, to 1e-6. For
   rotary positions that holds whether they restart or not, since attention
   sees only distances within a document; for a learned table only when they
   do, and otherwise it is off by 1.25.
4. **The editor and the rules** — the packing control, a sampled mask preview,
   and a rule for documents short enough that the kernel's blocks cost much
   more than the share. *Done:* the operating point's `docs` and `spread`,
   shown for a design with documents to pack; the packed training and
   attention in the readout; the mask preview drawn over one packed row; the
   walkthrough saying what the mask is for; `packing-unused`; and
   `document-blocks`, which measures against the whole forward pass so that
   it notes Llama-3-8B's 1.38x in 1,024-token documents and warns only where
   the blocks are a real part of a token's work.

**Done when** Llama-3-8B with a document mask, at 8,192 tokens and 1,024-token
documents, is counted at exactly 512 keys a query (attention falls from 12.5%
of the forward pass to 1.8%), gamma-distributed packings agree with the
size-biased mean, a generated model keeps its documents apart, FlexAttention's
block count matches the engine's, and with packing off no golden moves.

### M13 — The editor for someone new · Done

Twelve milestones went into what the engine can say, and the editor someone
meets first had drifted: a deployment with accounts showed two top bars, sharing
was four steps into a tab and absent from the public editor, the inspector named
every parameter by its code name, and the first screen opened with panels over
the drawing. This is the editor, for somebody who has not read the docs.

1. **One bar, and sharing in it.** The account moves into the editor's toolbar,
   from what the storage provider reports, so a deployment needs no bar of its
   own. A Share button beside it saves and links through a store when signed in
   to one, and otherwise carries the design in the link's fragment, which needs
   no server. *Done:* `signIn()`, `signOut()` and an account `problem` on the
   seam; the toolbar's account corner and Share dialog; `#design=` links opened
   by `openFromLocation` in any deployment. Every preset's link round-trips, the
   longest under sixteen thousand characters.
2. **Plain names.** Parameters with human labels and the code name beside them,
   parameters that do not apply hidden rather than listed as unused, the rare
   ones under Advanced, and the same plain names in the palette and the tree.
   *Done:* a label on every built-in parameter, in the engine
   (`catalog/labels.go`) so `explain` carries it too, with `advanced` and enum
   `valueLabels` beside it; a transformer block opens on fifteen fields rather
   than forty-two; the palette's headings are words, in the order a design is
   built, and its search reads the labels.
3. **A calm first screen.** Nothing over the drawing on first open, the
   operating point down to its essentials with the rest behind More, fewer tabs,
   and a Start here into the walkthrough. And a block clicked inside an unfolded
   frame inspected where it is, rather than "not on this level". *Done:* the key
   shut to its tab and the title block and minimap off on a first visit; batch,
   sequence, device and GPUs above a More line that names whatever under it has
   changed; Cluster, the width ladder and Runs as one Training tab; Start here
   in the empty inspector and at the top of the Help menu; a nested block
   inspected on its own level, read-only where that level is a built-in block's;
   and the toolbar fits at 1280, where Share used to fall off the edge.

## Sequencing and dependencies

```
M0 Sketch ──► M1 Check ──► M2 Manufacture ──► M3 Agent ──► M4 Test bench
                               │                 │
                               └──► M5 Advanced parts (starts after M2, runs alongside M3/M4)
                                                                 └──► M6 Ship ──► M7 Legibility ──► M8 Real values ──► M9 Your own values ──► M10 Attention variants ──► M11 Encoder–decoder ──► M12 Packed sequences ──► M13 The editor for someone new
```

M3 is deliberately short because the core is pure; the MCP server is a thin adapter. M5 is where most of the long-tail work lives and is driven by which architectures you want to learn next.

## Open questions

- ~~**Bun vs Node for the MCP binary**~~ Settled: developed with Bun, published for Node. `build:dist` installs the packed tarballs into an empty directory and starts the server under Node before anything is uploaded.
- **How far to go op-level**: the plan keeps `sdpa` and `ssd_scan` as primitives rather than decomposing to matmul/softmax. Decomposing further would let users invent new attention cores but makes the FLOP/memory model kernel-unaware. *Proposed answer, M10:* keep the fused primitive and give it a mask and a score expression, FlexAttention-style; decompose only in one explicitly eager block. See [Attention variants](docs/explanation/attention-variants.md).
- **Python dependency footprint**: `fla` and `mamba_ssm` need CUDA builds; keep them optional extras so the core runtime installs cleanly on CPU.
- **Tiny-track dataset**: FineWeb-Edu sample (closer to real pretraining) vs TinyStories (faster signal). Start with a 100M-token FineWeb-Edu shard.
