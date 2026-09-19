# TensorCAD — Feature proposal

A node-based CAD tool for designing the *internals* of transformer-family language models (pretraining-level architecture: embeddings, attention variants, norms, MLPs, MoE, SSM blocks, residual wiring, heads), with engineering checks, quantitative analysis, code generation, a small-scale test bench, and an MCP server so AI assistants can drive it.

Research backing this proposal is in [research/01-landscape.md](research/01-landscape.md), [research/02-analysis-math.md](research/02-analysis-math.md), and [research/03-mcp.md](research/03-mcp.md). Short version: nothing like this exists as a maintained tool in Sept 2026; all prior drag-and-drop NN builders are dead or MLP/CNN toys, and no calculator covers MoE + MLA + SSM + training + inference together.

## 1. Positioning: the CAD analogy, made concrete

| Mechanical CAD | TensorCAD | Concrete feature |
|---|---|---|
| Sketch / feature | Primitive op (Linear, Norm, SDPA, Add) | Op-level nodes with typed tensor ports |
| Part | Composite block (GQA attention, SwiGLU MLP, MoE layer, Mamba-2 block) | Catalog blocks defined as subgraphs of primitives; expand/collapse |
| Pattern / array | Repeat container ("Transformer Block ×N", hybrid layer patterns) | `repeat` node with count and variant pattern (e.g. `MMAMMA`) |
| Assembly | Whole model | Top-level graph: input → embedding → repeat → norm → head |
| Parameter table | Global symbols (D, H, Hkv, dh, F, L, V) | Symbol panel; every block param can be an expression |
| Constraint | Tensor shape / dtype interface | Symbolic shape inference on every edge |
| DRC / interference check | Design rules | Rule engine with severities and fix-its |
| Mass properties / FEA | Params, FLOPs, memory, KV cache, cost | Analysis panel with per-block breakdown |
| Simulation | Small-scale training run | Test bench on local GPU with loss-curve comparison |
| Manufacturing / CAM | Code generation | PyTorch module, litgpt/HF configs |
| Inspection / CMM | Verification | Cross-check generated model with torch.export and FlopCounterMode |
| Revision control | Design diff | Structural + numeric diff of two designs |
| CAD API / macros | MCP + CLI | Tools/resources/prompts for AI assistants; headless CLI for CI |

## 2. User goals this serves

- **Learning**: see shapes on every edge, hover to see why a block has `3·D·F` params, open Llama-3-8B and DeepSeek-V3 presets and compare them, expand a composite to see the primitive ops inside.
- **Validating**: catch mistakes before writing code (D not divisible by H, head_dim unsupported by FlashAttention, KV cache that does not fit the target GPU at the target context, tokens/param far from Chinchilla), and confirm estimates against PyTorch.
- **Testing**: generate a runnable model, smoke-train a scaled-down version on the local RTX 5080 (16 GB), compare loss curves against a baseline design, iterate.

## 3. Feature list

### 3.1 Block catalog (parts library)

**Primitives** (the analysis engine only needs formulas for these):
`input`, `output`, `embedding`, `linear`, `rmsnorm`, `layernorm`, `activation` (silu/gelu/relu²/…), `add`, `mul`, `rearrange` (einops pattern), `split`, `concat`, `rope`, `sdpa` (attention core; flags: causal, sliding window, flash), `softmax`, `topk_router`, `scatter_gather` (expert dispatch), `ssd_scan` (Mamba-2 core), `conv1d`, `dropout`, `lm_head` (linear with optional tying), `boundary_in`/`boundary_out` (repeat container ports).

**Composites** (subgraph templates with exposed parameters; ship with codegen templates):
`mha_attention`, `gqa_attention` (covers MQA), `mla_attention`, `sliding_window_attention`, `dense_mlp`, `gated_mlp` (SwiGLU/GeGLU), `moe_layer` (routed + shared experts, top-k, capacity factor, aux-loss flags), `mamba2_block`, `gated_deltanet_block` (via `fla`), `transformer_block_prenorm`, `transformer_block_postnorm`, `gemma_block` (pre+post norms), `mtp_head` (multi-token prediction), `value_embedding` and `unet_skip` (modded-nanogpt tricks, later).

**Containers**: `repeat` with `count` and optional `pattern` over named variants (Gemma 3 `5:1` local/global, Nemotron-H `M M M A` patterns, DeepSeek `first_k_dense`).

**Presets** seeded from Raschka's [LLM architecture gallery](https://github.com/rasbt/LLM-architecture-gallery) and HF configs: GPT-2 (4 sizes), Llama-2-7B, Llama-3-8B/70B, Mistral-7B, Mixtral-8x7B, Qwen2.5-7B, Qwen3-8B, Qwen3-30B-A3B, DeepSeek-V3, Gemma-2-9B, Gemma-3-27B, Nemotron-H-8B. Every preset is also a regression test.

**User-defined composites**: select nodes → "Collapse to block", give it parameters, save to a local library (ComfyUI subgraph model).

### 3.2 Editor (sketching)

- React Flow canvas: drag from palette, typed handles that refuse incompatible connections, edge labels showing the inferred shape (`B T D`), inline error/warning badges on nodes.
- Inspector panel: block parameters as numbers or symbol expressions (`F = 8/3*D` rounded to 256), enum choices, docs link, "explain" tab showing the formulas this block contributes.
- Symbols panel: global parameters with defaults and runtime symbols (`B`, `T`), so one edit rescales the whole model.
- Repeat container as a sub-flow with breadcrumb navigation; expand/collapse composites in place.
- Auto-layout (elkjs in a worker), minimap, keyboard shortcuts, undo/redo, multi-select, copy/paste, dark mode.
- Save/load `.tensorcad.json` with stable ids and deterministic key order (git-friendly); UI positions kept separate from semantics.
- Import: HF `config.json` (Llama/Mistral/Qwen/Gemma/Mixtral/DeepSeek families), litgpt `Config`.
- Export diagram as SVG/PNG and as a Markdown summary table.

### 3.3 Design rules (DRC)

Rule engine with `error` / `warning` / `info` severities, each with a fix-it action where possible.

- **Connectivity**: dangling required ports, cycles outside `repeat`, unreachable nodes, multiple writers to one port.
- **Shapes/dtypes**: every edge unifies; residual stream dimension is constant through a block; `D = H·dh` or explicit projection; `H % Hkv == 0`; `k ≤ E`; RoPE requires even `dh`; tied head requires matching `V, D`.
- **Kernel/hardware**: `dh ∈ {64, 128, 256}` for FlashAttention; `F`, `D`, `V` multiples of 64/128/256 for tensor cores; vocab padding hint; sliding window ≤ context.
- **Capacity**: KV cache at target context and concurrency fits the selected GPU; training memory fits with selected parallelism/optimizer; activation memory suggests recompute.
- **Scaling sanity**: tokens/param vs Chinchilla and typical over-training ratios; depth/width ratio outliers; MoE expert size vs active params; MLA latent dims vs `H·dh`.
- **Style**: unnamed blocks, unused symbols, presets diverging from reference (drift check).

Rules run on every edit (debounced) and in `tensorcad validate` for CI.

### 3.4 Analysis (mass properties)

Inputs: sequence length `T`, batch `B`, dtype, hardware profile, parallelism (DP/FSDP/TP/PP/EP), optimizer, recompute mode, target tokens.

Outputs, total and per block, with the formula shown on hover:
- Parameters: total, active (MoE), non-embedding, by block type; pie/treemap.
- FLOPs per token: forward, backward, train; attention share vs `T`; MoE active vs resident; per-layer breakdown.
- KV cache: bytes/token and bytes/sequence at `T`; MLA absorbed vs decompressed; sliding-window fixed cost; SSM fixed state.
- Memory: weights by dtype; training (weights + grads + optimizer + activations + logits) under chosen sharding; inference (weights + KV at `B, T`).
- Throughput: roofline decode tok/s and prefill time on the hardware profile; ridge-point indicator (memory-bound vs compute-bound).
- Training cost: GPU-hours, wall-clock, $ for given tokens, GPUs, MFU range; Chinchilla-optimal tokens and ratio.
- Hardware profiles: RTX 5080, RTX 4090, A100, H100, H200, B200, and a custom editor (peak TFLOPS, HBM, bandwidth, $/h).
- What-if: sliders on symbols with live re-analysis; "scale to N params" solver (adjust `D`/`L` under a chosen aspect ratio).

### 3.5 Code generation (manufacturing)

- **PyTorch single-file `nn.Module`** (primary): idiomatic, readable, no framework dependency; uses `F.scaled_dot_product_attention`, optional FlashAttention/`flex_attention` flags, imports `fla` / `mamba_ssm` for SSM blocks. Deterministic output, golden-file tested.
- **litgpt `Config`** when the design is expressible (dense/GQA/MoE with standard MLPs), which gives pretraining/finetuning/serving scripts for free.
- **HF `config.json` + modeling file** for interop with vLLM/HF tooling (later).
- **TorchTitan `ModelArgs`** (later) for scale-out training.
- Training-harness template: nanoGPT-style `train.py` wired to the generated model, with data prep script.

### 3.6 Verification (inspection)

A Python sidecar (`tensorcad-runtime`) instantiates the generated model on the `meta` device and reports back:
- exact parameter count vs the editor's estimate,
- shapes at every module boundary via `torch.export` with symbolic `Dim("B")`, `Dim("T")`,
- FLOPs via `FlopCounterMode` on fake tensors,
- a diff table in the UI: "editor says 8.030B, torch says 8.030B ✓; FLOPs differ by 1.8% (attention causal skip)".

### 3.7 Test bench (simulation)

- "Tiny track": scale a design down (keep ratios, shrink `D`, `L`, `T`) to a budget that trains in ~10 minutes on the local GPU; train on a fixed small dataset (FineWeb-Edu sample or TinyStories) with fixed seeds; record loss curve, tokens/s, peak memory.
- Compare runs across designs in the UI (loss vs tokens, loss vs wall-clock); baseline reference designs (GPT-2-small-style, Llama-style).
- Run registry (`runs/*.json`) with the design hash, so a run is reproducible from the design.
- Later: μP-aware width scaling so hyperparameters tuned on the tiny track transfer; export a modded-nanogpt-style script for 8×H100 runs.
- Lessons from [Parameter Golf](https://arxiv.org/abs/2607.01517) and [METR's NanoGPT note](https://metr.org/notes/2026-04-21-ai-rd-nanogpt-progress/): control the surrounding stack, report seeds and confidence intervals, and be honest that small-scale wins may not transfer.

### 3.8 Compare and version

- Diff two designs: structural (added/removed/changed blocks and params) and numeric (Δ params, Δ FLOPs, Δ KV cache, Δ memory).
- Git-friendly document format; optional design history inside the file (checkpoints).

### 3.9 MCP server and CLI

- MCP server (stdio by default) over the same core: `tensorcad_new_design`, `tensorcad_open_design`, `tensorcad_get_design` (full or outline), `tensorcad_get_block`, `tensorcad_search_catalog`, `tensorcad_apply_ops`, `tensorcad_validate`, `tensorcad_analyze`, `tensorcad_generate_code`, `tensorcad_render_preview`, `tensorcad_checkpoint`/`tensorcad_restore`, `tensorcad_save_design`, `tensorcad_run_script` (gated).
- Resources: `tensorcad://designs/{id}`, `/validation`, `/analysis`, `tensorcad://catalog`, `tensorcad://schema/design`. Prompts: `design_model`, `review_design`, `scale_design`, `explain_costs`.
- Live bridge: when the UI is running it writes a session file; the MCP server attaches over WebSocket so assistant edits appear on the canvas immediately and human edits are pushed back as resource updates.
- CLI: `tensorcad validate | analyze | codegen | import | diff | serve-mcp` for scripts and CI.
- Ships with `.mcp.json` for Claude Code, an MCPB bundle for Claude Desktop, and a Cursor deeplink.

### 3.10 Learning aids

- Explain tab per block: formulas with the current numbers substituted, links to the paper.
- Guided tour: "Build GPT-2 small from primitives", "Turn it into Llama", "Add MoE", "Swap in Mamba-2".
- Preset notes: what is distinctive about each reference architecture (from Raschka's gallery).

## 4. Non-goals (v1)

- Not a general deep-learning graph editor (no CNN/RNN zoo, no op-level autograd in the browser).
- No in-browser training; training runs in the Python runtime on a real GPU.
- No neural architecture search; the human (or the assistant via MCP) designs.
- No distributed training orchestration or serving; codegen targets existing frameworks for that.
- No weight editing or checkpoint conversion.

## 5. Prioritization

| Must (PoC, M0–M1) | Should (M2–M3) | Could (M4+) |
|---|---|---|
| React Flow canvas, typed ports, symbols, repeat container | PyTorch codegen + torch verification | Test bench with loss-curve compare |
| Primitives + composites for GPT-2/Llama/Mistral/Qwen | HF config import; litgpt export | MoE, MLA, Mamba-2, hybrids, MTP |
| Symbolic shapes, param count, DRC core rules | MCP server (file mode), CLI | Live MCP bridge, MCP Apps preview |
| FLOPs, KV cache, memory, cost, hardware profiles | Design diff | μP scale ladder, parallelism planner |
| Presets + regression tests | Explain tab, guided tour | User-defined composites library, MCPB/registry publishing |
