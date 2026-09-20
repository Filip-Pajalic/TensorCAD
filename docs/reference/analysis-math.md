# Research 02 — Analysis engine math: params, FLOPs, KV cache, memory, cost

Researched 2026-09-18. Formulas were validated numerically against published configs (see §4 and `params_reference.py`).

Notation: `L` layers, `D` d_model, `H` query heads, `Hkv` KV heads, `g = H/Hkv`, `dh` head dim, `F` FFN width, `V` vocab, `T` sequence length, `B` batch, `E` routed experts, `k` top-k, `n_s` shared experts, `Fe` expert FFN width, `N` parameters, `bytes` = dtype width (bf16 = 2).

## 1. Formulas and references

Canonical sources: EleutherAI [Transformer Math 101](https://blog.eleuther.ai/transformer-math/); [Kaplan scaling laws](https://arxiv.org/abs/2001.08361); [Chinchilla](https://arxiv.org/abs/2203.15556); [PaLM](https://arxiv.org/abs/2204.02311) App. B for the `6N + 12LHQT` rule; [Megatron activation recomputation, Korthikanti et al.](https://arxiv.org/abs/2205.05198); [ZeRO](https://arxiv.org/abs/1910.02054); [FlashAttention](https://arxiv.org/abs/2205.14135); [GQA](https://arxiv.org/abs/2305.13245); [MQA](https://arxiv.org/abs/1911.02150); [RoPE](https://arxiv.org/abs/2104.09864); [DeepSeek-V2](https://arxiv.org/abs/2405.04434) for MLA; [DeepSeek-V3](https://arxiv.org/abs/2412.19437); [DeepSeekMoE](https://arxiv.org/abs/2401.06066); [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060) and [reference code](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py); kipply [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) for roofline; [8-bit Adam](https://arxiv.org/abs/2110.02861).

### 1.1 Parameter count (validated, see §4)

| Block | Params |
|---|---|
| Token embedding | `V·D` |
| LM head | `V·D` if untied, `0` if tied (GPT-2, Gemma tie; Llama/Qwen/DeepSeek don't) |
| Learned positions (GPT-2) | `T_max·D`; RoPE/NoPE: 0 |
| Attention MHA/GQA/MQA | `D·H·dh + 2·D·Hkv·dh + H·dh·D` (+ biases `H·dh + 2·Hkv·dh + D` if `attention_bias`; Qwen2.5 has QKV bias, no O bias). MHA: `Hkv=H`; MQA: `Hkv=1`. QK-norm (Qwen3, Gemma 3): `+2·dh` |
| Attention MLA (DeepSeek-V2/V3) | Q path: `D·d_c' + d_c' + d_c'·H·(d_nope + d_rope)`; KV path: `D·(d_c + d_rope) + d_c + d_c·H·(d_nope + d_v)`; O: `H·d_v·D`. V3 (`D=7168, H=128, d_c'=1536, d_c=512, d_nope=128, d_rope=64, d_v=128`) → 187.1 M/layer |
| Dense MLP, 2-matrix (GELU/ReLU²) | `2·D·F` (+ `F + D` biases) |
| Gated MLP (SwiGLU/GeGLU) | `3·D·F` (Llama: `F = round_to_256(8/3·D)` → 11008, 14336) |
| MoE layer | router `D·E` (+`E` if bias, e.g. DeepSeek `e_score_correction_bias`) + routed `E·3·D·Fe` + shared `n_s·3·D·Fe`. Respect dense-layer overrides (`first_k_dense_replace`, `decoder_sparse_step`, `mlp_only_layers`) |
| Norms | RMSNorm `D` each; 2/layer pre-norm, 4/layer for Gemma (pre+post); +1 final. LayerNorm `2D` |
| Mamba-2 block (`d_inner=E·D`, `nheads=d_inner/P`, groups `G`, state `N_s`) | `in_proj D·(2·d_inner + 2·G·N_s + nheads)` + conv1d `(d_inner + 2·G·N_s)·(d_conv+1)` + `3·nheads` (dt_bias, A_log, D) + norm `d_inner` + `out_proj d_inner·D`. Defaults `N_s=128, P=64, E=2, d_conv=4` |
| **Active params (MoE)** | same, but `k·3·D·Fe + n_s·3·D·Fe` instead of `E·…`; embeddings/head count once |

### 1.2 FLOPs per token

- **Rule of thumb**: forward `≈ 2·N` (N = non-embedding *active* params, LM head included since it is a matmul; embedding lookup is a gather ≈ 0). Backward `≈ 4·N`. Train `6·N`; with full activation recompute `8·N`.
- **Attention (score + PV) term**, per token, per layer, forward: `4·T·H·dh` (QKᵀ `2·T·H·dh`, PV `2·T·H·dh`), independent of `Hkv`. Halve if the kernel skips masked causal blocks (FlashAttention does). Full training: `FLOPs/token = 6·N + 12·L·T·H·dh` (PaLM App. B form).
- **Where 6N/2N break**: (a) long context: with `N ≈ 12·L·D²` the attention share is `T/(6·D)` → 33% for Llama-3-8B at 8K, >100% at 32K; (b) MoE: use active `N`, but hardware FLOPs include padded/dropped tokens under capacity factors; (c) MLA decode with weight absorption scores in latent space: `2·T·H·(d_c + d_rope)` per token, larger than GQA's `2·T·H·dh`; (d) MTP adds ≈ one extra layer + head per depth; (e) sliding-window layers: replace `T` by `min(T, W)`; sparse attention by `topk + W`; (f) SSM layers: linear in `T`; SSD scan ≈ `6·d_inner·N_s` FLOPs per token plus a chunk-quadratic term ≈ `4·d_inner·Q` for chunk `Q=256`, small vs the `in/out_proj` matmuls (**approximate; paper gives asymptotics not constants**); (g) elementwise ops (norm, RoPE, softmax, act) are ~1–3% of FLOPs but memory-bound: ignore for FLOPs, not for latency; (h) inference decode per token: `2·N_active + 4·L·T_ctx·H·dh`.

### 1.3 KV-cache bytes per token

- **MHA/GQA/MQA**: `2·L·Hkv·dh·bytes`. Llama-2-7B (MHA) 512 KiB; Llama-3-8B (GQA g=4) 128 KiB; Llama-3-70B 320 KiB (the 160 KiB first reported here was wrong: 80 layers x 8 kv heads x 128 head dim x 2 tensors x 2 bytes = 320 KiB, confirmed by the implementation's regression test). DeepSeek-V4 uses `Hkv=1` with `head_dim=512` and K=V shared → `L·512·bytes`.
- **MLA**: `L·(d_c + d_rope)·bytes` (DeepSeek-V2 Table 1). V3: `61·576·2 = 70,272 B` ≈ 68.6 KiB/token vs 4 MiB uncompressed (57×). Engines that don't absorb weights cache decompressed K/V → MHA size; expose both.
- **Sliding-window layers**: cache `min(T, W)` tokens, so per-sequence cost `2·L_local·Hkv·dh·bytes·W` is constant. Gemma-3-27B (5:1 local:global, W=1024, 62 layers): ~11 global layers → 90 KB/token, plus ≈ 428 MB fixed for 51 local layers.
- **SSM layers**: fixed per sequence, not per token: `nheads·P·N_s·bytes + (d_inner + 2·G·N_s)·(d_conv−1)·bytes`. Nemotron-H-8B: ≈ 50 MB/sequence for 24 Mamba layers, plus 4 attention layers × 16 KiB/token.
- Serving total: `per_token × T_ctx × concurrent sequences` (+ paged-attention block waste).

### 1.4 Training memory

- **Per parameter** (mixed precision AdamW): bf16 weights 2 + bf16 grads 2 + fp32 master 4 + m 4 + v 4 = **16 B/param**. fp32 grads → 18 B. 8-bit Adam → 10 B. SGD+momentum 8 B optimizer; Muon ≈ 8 B. Pure-bf16 Adam without master weights: 8 B (unstable; flag in UI).
- **Activations per layer** (Megatron eq. 1, GPT-3-style block, bf16, F=4D, dropout): `s·b·h·(34 + 5·a·s/h)` bytes, i.e. per token `34·D + 5·H·T`. With tensor parallel `t`: `sbh(10 + 24/t + 5as/(ht))`; with sequence parallel: `sbh(34/t + 5as/(ht))`; **selective recompute or FlashAttention drops the `5as/h` term → `34·sbh/t`**; full recompute `2·sbh` (layer input only); recompute costs +33% FLOPs (8N).
- **Modern SwiGLU/no-dropout block** (own derivation from saved tensors; verify against a profiler): per token per layer ≈ `D·(14 + 4/g) + 6·F` bytes bf16 (norm inputs 4D, QKV input 2D, Q/K/V for FA backward 2D(1+2/g), FA output + o_proj input 4D, MLP input 2D, gate/up outputs 4F, act output 2F). Llama-3-8B: ≈ 147 KB/token/layer → 38 GB for one 8K sequence without recompute. **Implementation note:** this list omits the down-projection's input. Unfused PyTorch keeps four F-wide tensors (gate output, activation output, up output, product), not three, so the real figure is ≈ 176 KB/token/layer. A fused gated feed-forward reaches the 147 KB quoted here.
- **Logits**: `B·T·V·(2 + 4)` bytes (bf16 logits + fp32 softmax/CE); 8K × 128K vocab = 6.3 GB. Use chunked CE.
- **Sharding**: ZeRO-1 `W + G + Opt/N_dp + Act`; ZeRO-2 `W + (G+Opt)/N_dp + Act`; ZeRO-3/FSDP `(W+G+Opt)/N_dp + Act + live unsharded layer`. TP divides W/G/Opt by `t` and activations by `t` only with sequence parallel; PP divides layers by `p` but holds `p` micro-batches of activations in flight (1F1B); EP divides only expert weights. Add 10–20% workspace/fragmentation.

### 1.5 Inference memory and roofline

- Memory = `N_total·bytes` (all experts resident) + KV cache + small activations + ~20% overhead.
- Decode step time ≈ `max(bytes_moved / BW, FLOPs / (peak·eff))`; `bytes_moved = read weight bytes + KV bytes(B·T)`; `FLOPs = B·(2·N_active + 4·L·T·H·dh)`. Arithmetic intensity of weight streaming ≈ `B` FLOP/byte (bf16), so decode is compute-bound only when `B > ridge = peak/BW`: A100 ≈ 208; H100 SXM ≈ 989/3.35 ≈ 295; B200 ≈ 2250/8 ≈ 280 (**approximate spec values**). Batch-1 tok/s upper bound = `BW / weight_bytes` (Llama-3-8B bf16 on H100 ≈ 209 tok/s). Prefill is compute-bound: `2·N·T/(peak·MFU)`.

- **A sparse model at batch reads more than one token's share.** One token routes to `k` of `E` experts and reads `k/E` of them, which is the active count. A batch of `B` tokens routes independently, so an expert is read unless every one of them skipped it: the expected share read is `1 - (1 - k/E)^B`, climbing from `k/E` at `B=1` to essentially 1 well before `B` reaches `E`. Mixtral at batch 32 reads 43.49 GiB of the 43.50 GiB it holds, not the 12.9 GiB one token reads, and its decode rate is 929 tok/s where the active count alone says 1314. The implementation writes this per block as `Σ per · copies · (1 - (1 - active/copies)^B)`, which needs no knowledge of `k` or `E` and reduces to the active count at `B=1` and to the resident count for anything dense. **This assumes routing is independent and uniform;** a router trained towards balance spreads a batch over more experts still, so the figure is a floor on what is read and the rate above it a ceiling.

### 1.6 Training cost

`FLOPs_total = (6·N_active + 12·L·T·H·dh)·tokens` (×8/6 with full recompute); `GPU-hours = FLOPs_total / (peak_FLOPs · MFU · 3600)`; `wall-clock = GPU-hours / n_GPU`; `cost = GPU-hours · $/h`.

MFU anchors: Llama-3-405B 43%/41%/38% BF16 on 8K/16K/16K+CP H100s ([Llama 3 paper](https://ar5iv.labs.arxiv.org/html/2407.21783)); PaLM 46.2% on TPUv4; DeepSeek-V3: 2.664 M H800-hours for 14.8 T tokens → ≈ 38% BF16-equivalent (~19% of FP8 peak). Suggested UI ranges (synthesis): dense well-tuned 35–45%; MoE 20–35%; naive/small clusters 25–35%.

### 1.7 Chinchilla and over-training

`L(N,D) = E + A/N^α + B/D^β`; Hoffmann: `E=1.69, A=406.4, B=410.7, α=0.34, β=0.28`; [Epoch refit](https://epoch.ai/blog/chinchilla-scaling-a-replication-attempt) `E=1.8172, A=482.01, B=2085.43, α=0.3478, β=0.3658`. Compute-optimal `D_opt ≈ 20·N`, `N_opt = sqrt(C/120)`. Real tokens/param ratios ([Epoch](https://epoch.ai/data-insights/training-tokens-per-parameter)): Llama-2-7B 286; Llama-3-70B 214; Llama-3.1-405B 38.5; Llama-3-8B ≈ 1875; Qwen2.5-72B 248; DeepSeek-V3 22 (total) / 400 (active); Qwen3-0.6B ≈ 60,000. Trend: 3.1×/yr. Expose both total- and active-param ratios; cite inference-adjusted optima ([Sardana & Frankle](https://arxiv.org/abs/2401.00448)).

## 2. Existing calculators / libraries

| Tool | Covers | Gaps |
|---|---|---|
| [EleutherAI cookbook `calc/`](https://github.com/EleutherAI/cookbook/tree/main/calc) | params (GQA, MoE, tied emb), FLOPs (train/checkpointing, attention terms, MoE gating), memory (ZeRO 1-3, TP/PP, Adam/8-bit/SGD/Muon, Megatron activation formulas, KV cache) | no MLA, no SSM, no sliding window; KV formula may omit kv-ratio (verify) |
| [llm-analysis](https://github.com/cli99/llm-analysis) (Apache-2) | train+infer latency/memory/FLOPs, TP/PP/SP/EP/DP, dtype bits, efficiency knobs | GQA/MLA/SSM/FlashAttention not modelled |
| [LLM-para](https://github.com/dengls24/LLM-para) (MIT) | inference roofline over 13 operators; GQA, MoE, MLA, RoPE, SwiGLU, FA; 24 hardware targets | inference only; no SSM |
| [`torch.utils.flop_counter.FlopCounterMode`](https://docs.pytorch.org/docs/stable/generated/torch.utils.flop_counter.FlopCounterMode.html) | dispatch-level, shape-based; mm/bmm/conv/SDPA fwd+bwd; custom ops via `register_flop_formula`; works with fake/meta tensors | only registered aten ops. **Good validation backend.** |
| [DeepSpeed FLOPS profiler](https://www.deepspeed.ai/tutorials/flops-profiler/) | per-module breakdown | custom kernels unseen |
| [calflops](https://github.com/MrYxJ/calculate-flops.pytorch) | HF models; FLOPs/MACs/params | same limits |
| torchinfo, fvcore | params, mult-adds by hooks | miss functional matmuls, no training memory |
| [HF Model Memory Utility](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) | weight bytes, training ≈ 4× | no KV/activations, no FLOPs |
| [Vokturz can-it-run-llm](https://huggingface.co/spaces/Vokturz/can-it-run-llm), [apxml](https://apxml.com/tools/vram-calculator), NyxKrage, modelfit.io | inference/fine-tune memory vs GPU | opaque or HF-only |

Nothing covers MoE + MLA + SSM + training + inference together; SSM/hybrid accounting is the clearest gap.

## 3. Shape inference / symbolic approach

- **Representation**: each port carries `(dims: SymExpr[], dtype)` where `SymExpr` is a canonical integer polynomial over named symbols (`B, T, D, H, dh, F, V…`) with rational coefficients. Covers concat (`+`), split/reshape (`*`, `/` with a divisibility obligation like `D = H·dh`), and constants. Validation = unify producer shape with consumer pattern: bind free names, compare canonical forms, emit diffs ("expected `(B,T,D)`, got `(B,T,H·dh)` and `H·dh ≠ D`"). Keep a global constraint set (`F % 256 == 0`, `H % Hkv == 0`, `E ≥ k`) checked as design rules.
- **Port-type notation**: [jaxtyping](https://docs.kidger.site/jaxtyping/api/array/) strings (`"batch seq dim"`, expression axes, variadic `"*batch"`) and [einops](https://einops.rocks/api/parse_shape/) patterns (`"b t (h d) -> b h t d"`) are a ready-made port-typing DSL. Adopt that syntax.
- **Reference implementations**: `torch.export` uses SymPy-backed `SymInt`s in a `ShapeEnv` with guards, `Dim(name, min, max)` ([docs](https://docs.pytorch.org/docs/2.9/torch.compiler_dynamic_shapes.html)); ONNX `dim_param` strings with no arithmetic; onnxruntime [`SymbolicShapeInference`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/symbolic_shape_infer.py) uses SymPy.
- **JS/TS**: Algebrite, math.js, nerdamer are heavier than needed; a ~300-line polynomial canonicalizer is the pragmatic choice.
- **Validation backend**: code-gen PyTorch from the graph, run `torch.export` with `Dim("B"), Dim("T")` to cross-check shapes, and `FlopCounterMode` on meta/fake tensors to cross-check FLOPs (meta-device counting should work since counting is shape-based; verify).

## 4. Known-good architecture specs (regression set)

Configs from HF `config.json` or the paper; "calc" = `params_reference.py` in this folder.

| Model | L | D | H / Hkv / dh | F (Fe) | V | Experts / k / shared | Published | Calc |
|---|---|---|---|---|---|---|---|---|
| GPT-2 small/med/large/XL (tied, learned pos, biases, LN) | 12/24/36/48 | 768/1024/1280/1600 | 12/16/20/25, MHA | 4D | 50257 | – | 124M/355M/774M/1.5B | 124.4/354.8/774.0/1557.6M |
| [Llama-2-7B](https://huggingface.co/NousResearch/Llama-2-7b-hf) | 32 | 4096 | 32/32/128 | 11008 | 32000 | – | 6.74B | 6.738B |
| Mistral-7B | 32 | 4096 | 32/8/128 | 14336 | 32000 | – | 7.24B | 7.242B |
| [Llama-3-8B](https://huggingface.co/NousResearch/Meta-Llama-3-8B) | 32 | 4096 | 32/8/128 | 14336 | 128256 | – | 8.03B | 8.030B |
| Llama-3-70B | 80 | 8192 | 64/8/128 | 28672 | 128256 | – | 70.6B | 70.55B |
| Llama-3.1-405B | 126 | 16384 | 128/8/128 | 53248 | 128256 | – | 405B | 405.9B |
| [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | 32 | 4096 | 32/8/128 | (14336) | 32000 | 8 / 2 / 0 | 46.7B / 12.9B act | 46.70B / 12.88B |
| Qwen2.5-7B (QKV bias) | 28 | 3584 | 28/4/128 | 18944 | 152064 | – | 7.61B | 7.615B |
| Qwen3-8B (QK-norm) | 36 | 4096 | 32/8/128 | 12288 | 151936 | – | 8.2B | 8.19B |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 48 | 2048 | 32/4/128 | (768) | 151936 | 128 / 8 / 0 | 30.5B / 3.3B | 30.53B / 3.35B |
| Qwen3-235B-A22B | 94 | 4096 | 64/4/128 | (1536) | 151936 | 128 / 8 / 0 | 235B / 22B | 235.1B / 22.2B |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) MLA `d_c'=1536, d_c=512, nope 128, rope 64, d_v 128`; 3 dense layers F=18432; router bias | 61 | 7168 | 128 MLA | (2048) | 129280 | 256 / 8 / 1 | 671B / 37B; +14B MTP | 671.0B / 37.6B; MTP 13.5B |
| Gemma-2-9B (tied, 4 norms/layer, GeGLU) | 42 | 3584 | 16/8/256 | 14336 | 256000 | – | 9.24B | 9.242B |
| Gemma-3-27B (tied; W=1024, 5:1) | 62 | 5376 | 32/16/128 | 21504 | 262208 | – | text 27.0B | 27.01B |
| [Nemotron-H-8B](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K): 24×Mamba-2, 4×attn, 24×FFN(relu², 2-matrix); Mamba-2 `nheads 128, P 64, N_s 128, G 8` | 52 | 4096 | 32/8/128 | 21504 | 131072 | – | ~8B | 8.10B |
| [Jamba-v0.1](https://huggingface.co/ai21labs/Jamba-v0.1): attn every 8, MoE every 2, Mamba-1 | 32 | 4096 | 32/8/128 | 14336 | 65536 | 16 / 2 | 52B / 12B | not computed |

2026 models (from HF docs/config and secondary sources; **totals not recomputed**): DeepSeek-V4-Flash 284B/13B — 43 layers, D=4096, 256+1 experts, top-6, Fe=2048, shared-K=V MQA with `head_dim 512`, grouped low-rank O-proj, sliding window 128 + compressors ([HF docs](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)). [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — 93 layers, D=7168, 96-head MLA, 896 routed + 2 shared, top-16, KDA linear-attention 3:1, V=163840; reported 2.8T/104B. GLM-5 744B/40B — 78 layers, D=6144, 256 experts top-8 + 1 shared, MLA + DSA. Qwen3.5-397B-A17B and Qwen3-Coder-Next 80B-A3B — Gated DeltaNet + gated attention 3:1 ([Raschka](https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight)).

## Flags / unverified

- Mamba-2 SSD FLOP constants (§1.2 f) and the SwiGLU activation-bytes formula (§1.4) are derivations, not quoted from papers.
- H100/B200 ridge points use approximate spec-sheet values.
- 2026 model totals (Kimi-K3, GLM-5, DeepSeek-V4) come from vendor/secondary summaries; recompute before using as tests.
- Jamba was not recomputed.
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) is a useful memory/parallelism reference but could not be fetched (JS-rendered).
