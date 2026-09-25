# Attention variants: what to open up, and how far

*A proposal for M10. Phases 1 and 2 are built, and most of 3 and 4; the rest is not. The point of
writing it first was that the choice decides what every attention estimate in
the tool means. The language the two expressions are written in is
[its own reference page](../reference/attention-expressions.md).*

The roadmap has carried one open question since M2:

> **How far to go op-level**: the plan keeps `sdpa` and `ssd_scan` as primitives
> rather than decomposing to matmul/softmax. Decomposing further would let users
> invent new attention cores but makes the FLOP/memory model kernel-unaware.

That was the right trade when every model used the same attention. It is less
right now: ALiBi, relative position bias, attention sinks, prefix-LM and document
masking are all in shipped models, and none of them can be drawn here. This page
says what the engine does today, two places where it is already wrong, the three
ways to open attention up, and which one to take.

## What the engine does today

Attention is one primitive, `sdpa`, and a variant is a switch on it:

| Parameter | What it changes |
|---|---|
| `heads`, `kv_heads` | Multi-head, grouped-query, multi-query |
| `head_dim`, `v_head_dim` | Head widths; latent attention's narrower values |
| `causal` | Future positions masked |
| `window` | Sliding window; the cache stops growing past it |
| `flash` | Fused kernel: the score matrix is never materialised |
| `logit_softcap` | `cap·tanh(score/cap)` on the scores, as in Gemma 2 |
| `cache` | Whether this block owns the inference cache |

Everything around the core — rotary embedding, QK-norm, value embeddings, an
output gate, latent attention — is built from other primitives in the
composites, which is why those needed no new maths.

The accounting is where the choice matters. Per token, the primitive counts
`4·T_eff·heads·head_dim` FLOPs, halved when causal because a causal kernel skips
the masked blocks, with `T_eff` the window when there is one. For the backward
pass it keeps the output and the softmax's log-sum-exp when the kernel is fused,
and the score row and the softmax output — `T`-long per head per token — when it
is not. That difference is most of the story: it is what makes attention
activations linear in the sequence rather than quadratic, and it is the part a
decomposed graph would lose.

## Two places it is already wrong

Writing this found both. They are the same mistake: the analysis and the code
the tool generates describe different kernels.

**Softcapping.** `SDPA-03`, the `logit_softcap` documentation and Gemma 2's
preset notes all say a fused kernel cannot cap the scores, so a capped layer is
counted as eager. That stopped being true in July 2024: FlashAttention 2.6
added softcapping "as used in Gemma-2 and Grok models", and PyTorch's
FlexAttention does it with a two-line `score_mod`. The engine's own numbers for
Gemma-2-9B, batch 1:

| Context | Activations, softcap counted eager | Counted fused |
|---:|---:|---:|
| 4,096 | 78.4 GiB | 36.4 GiB |
| 8,192 | 198.9 GiB | 72.9 GiB |

The eager figure is true of the model TensorCAD *generates*, which calls an eager
`softcap_attention` helper. It is not true of Gemma 2 trained with a current
kernel, and that is what somebody pricing a run needs.

**Sliding windows.** The analysis counts a windowed layer at `T_eff = window`
keys per query, which is what a kernel that skips blocks outside the window
does. The generated code builds a dense `T×T` mask and passes it to
`F.scaled_dot_product_attention`, which computes every score and then masks.
At 8k context Gemma 2's local layers are counted at 4,096 keys a query, 2,048
on average once the causal half is skipped; the generated model computes all
8,192 for every query and masks what it should not have computed.

*Phase 2 found that the 2,048 was wrong too.* Halving for causal is right for
full attention and wrong inside a window, which already only looks back: a
query at position `q` keeps `min(q + 1, W)` keys, which averages
`W - W²/2T` — 3,072 at 8k, not 2,048. Writing the window out as a mask and
measuring it is what showed it, because the two spellings of the same layer
disagreed.

Whatever M10 does has to close this gap, not widen it: the numbers are only
worth reading if they describe the model the tool hands over.

## What cannot be drawn today

| Variant | Where it is used | What it changes |
|---|---|---|
| ALiBi | BLOOM, MPT | A per-head linear penalty on distance, added to each score |
| Relative position bias | T5 | A learned `[buckets, heads]` table, added to each score |
| Prefix-LM | T5, PaliGemma | Bidirectional over a prefix, causal after it |
| Document masking | Packed pretraining | No attention across document boundaries |
| Attention sinks | gpt-oss | A learned per-head term in the softmax's denominator |
| Differential attention | DIFF Transformer | Two attention maps, one subtracted from the other |
| Talking heads | Shazeer et al. 2020 | Scores mixed across heads, before and after the softmax |

They are not alike. The first four change each score, or which scores count,
using only that score and its position. Sinks change the normalisation.
Differential attention combines two whole attentions. Talking heads needs every
head's score for the same position at once. That split is what separates the
options below.

## Three ways to open it up

### A. More switches on the primitive

An `alibi` flag, a `sinks` flag, a `prefix` input, and so on. Cheap for each one,
and each is engine work: a formula, a code path, a kernel to target. The
FlexAttention announcement calls this the hypercube problem — the combinations
multiply, and a new variant has no support until somebody adds it. It is what
the tool does now, and it is why the table above is not empty.

### B. Decompose: matmul, scale, mask, softmax, matmul

New primitives for a batched matrix multiply, a mask and a softmax — none exists
today — and attention becomes a subgraph of them. Anything is drawable,
talking heads included.

The cost is the accounting. Drawn as five primitives, every attention is eager:
each keeps its `T×T` intermediate for the backward pass, and every attention
memory figure in the tool gets the Gemma 2 treatment above — 2.7 times too large
at 8k, and worse as the context grows. The fix is a fusion pass that recognises
"scores, mask, softmax, weighted sum" and counts it as one fused kernel, which
is a compiler: every variant either matches a known pattern or silently falls
back to eager, and somebody has to maintain the patterns. Code generation needs
the same recogniser, or it emits eager attention everywhere. The common case
gets worse to make the rare one possible.

### C. Keep the fused kernel, and let the design say what it does to a score

PyTorch's FlexAttention made an observation that fits this tool closely:
nearly every attention variant is a function applied to each score before the
softmax, or a rule for which scores to keep. It takes two small functions,

```python
score_mod(score, b, h, q_idx, kv_idx) -> score   # what happens to one score
mask_mod(b, h, q_idx, kv_idx) -> bool            # whether it counts at all
```

and compiles them into one fused kernel that never materialises the scores, with
the backward pass generated. A mask also yields a *block mask*: whole blocks
that are masked out are skipped, which is where the speed of causal and windowed
attention comes from.

For TensorCAD this becomes two new parameters on `sdpa`, each a small expression
the engine can read — not arbitrary Python:

- `mask`: a boolean expression over `q_idx`, `kv_idx`, `h`, runtime inputs such
  as a prefix length or document ids, and constants. `causal` and `window`
  become the two most common masks rather than special cases.
- `score`: an expression over `score` and the same indices, with `tanh`, `abs`,
  `min`, `max` and arithmetic, able to refer to learned tensors the block
  declares — ALiBi's per-head slopes, T5's bias table.

What that buys, point by point:

- **The accounting stays kernel-aware.** Memory is the fused kernel's: output
  and log-sum-exp. FLOPs are the matmuls over the fraction of blocks the mask
  keeps, which the engine can compute by evaluating the mask over the block
  grid at the operating point's `T`, plus the score expression's elementwise
  cost over the scores it touches. Causal comes out at a half, as now; a window
  at its width over `T`, as now; prefix-LM and document masks at whatever they
  actually are.
- **The code matches the numbers.** Code generation emits `flex_attention` with
  the two expressions as Python functions and a `create_block_mask`, and keeps
  `F.scaled_dot_product_attention(is_causal=True)` for the plain case. Both of
  the disagreements above disappear, because the kernel the analysis describes
  is the one the code calls.
- **The learned parts are parameters.** A bias table referenced by the score
  expression is a tensor the block owns, counted and initialised like any other,
  so a T5-style model's parameter count comes out right.
- **It can be checked.** M9's trace already recomputes attention from the
  captured query and key and holds it against what the fused kernel produced.
  With the same expressions evaluated eagerly, that check covers every variant,
  and a mask or score the kernel gets wrong is caught before it is drawn.

What it cannot express, and what to do about each:

- **Attention sinks** change the softmax, not a score: a parameter on the
  primitive, `sinks`, with its `heads` learned scalars. One more term in the
  normalisation, no change to the memory.
- **Differential attention** is two attentions combined:
  `(softmax(A₁) − λ·softmax(A₂))·V` is `softmax(A₁)·V − λ·softmax(A₂)·V`, so it is
  two `sdpa` blocks, each fused, and a weighted difference. The graph already has
  the parts except λ's reparameterisation, which is a small primitive.
- **Talking heads** genuinely needs the score matrix. For that, and for anything
  else that does, B's primitives exist as one explicitly eager block — "attention,
  written out" — with a design rule saying what it costs. Opt-in, labelled, and
  never what a preset reaches for by accident.

## Recommendation

**C, with B's primitives only as an explicitly eager block.** The reason to use
this tool over a spreadsheet is that its numbers know about the kernel. Option B
gives that up for every design to gain a variant most designs do not use; option
A keeps it but never catches up with the variants that are already shipping.
Option C keeps the fused kernel as the thing being described, puts the variant in
the design where it can be read, costed and checked, and fixes the two places the
analysis and the generated code currently disagree.

## The phases

1. **Make the numbers and the code agree.** Before anything is added.
   *Done*, with FlashAttention rather than FlexAttention as the kernel — see
   below. A window or a cap is generated as a `fused_attention` helper that calls
   `flash_attn_func` with `window_size` and `softcap` on CUDA in half precision
   when flash-attn is installed, and otherwise computes exactly what the code
   computed before, saying once on a GPU that it is running unfused. The cap is
   counted inside the fused kernel, over the scores it computes. `SDPA-03` is a
   note naming the kernel; Gemma 2's notes are corrected; three models'
   generated code and Gemma 2's analysis and findings changed in the goldens.
   Gemma 2's activations at 8k fall from 198.9 GiB to 72.9. On a CPU, where
   verification runs, the generated model is the one it was: the same
   parameters, the same FLOPs against the profiler, and it still exports. The
   fused branch is checked against the unfused one on a GPU, with a stand-in
   for FlashAttention written from its documented contract, since flash-attn
   does not install on Windows.
2. **The two expressions.** A small grammar the engine parses, evaluates for
   density and cost, and prints as Python; the inspector edits them, with the
   block mask drawn beside the expression, and design rules for one that does not
   parse, masks everything, or reads something the kernel cannot. `causal` and
   `window` become sugar for the masks they are. *Done.* `mask` and `score` are
   parameters on `sdpa`, `gqa_attention` and `transformer_block`, in a language
   with Python's operators and precedence over `q`, `kv`, `h`, `b`, `heads`,
   `score` and the design's symbols. A mask is counted by evaluating it —
   stratified, seeded, so it is reproducible — as a share of what causal and the
   window leave; a score expression by its arithmetic over the scores the mask
   keeps. `SDPA-04` and `SDPA-05` catch a mask that keeps nothing or leaves a
   query with nothing. Each expression is generated as a FlexAttention
   `mask_mod` or `score_mod`, with the switches folded in, and run through
   compiled `flex_attention` on CUDA or, anywhere it cannot compile, applied
   eagerly to the whole score matrix — the form verification uses, and one a
   test holds against `flex_attention` itself. Making the switches sugar is
   what corrected the window count above: a causal window of 1,024 at 8k is now
   counted at 960 keys a query, where it was 512. The unmasked figure, the one
   a profiler reproduces, now counts the whole sequence for a windowed layer as
   the profiler does. Both move Gemma 2, Gemma 3 and Mistral's attention FLOPs
   and what is computed from them, and nothing else.
3. **Presets that need it.** Models with a published parameter count to regress
   against: an ALiBi model (BLOOM or MPT), a relative-bias model (T5), and gpt-oss,
   which uses both sinks and banded attention. *The ALiBi model is done:*
   `bloom-7b1`, whose only sense of order is a score expression, reproduces its
   published count exactly, and its bias is checked against Hugging Face's own
   construction of it. *So is gpt-oss:* `gpt-oss-20b` reproduces its 20.9B to
   the parameter, sinks included.
4. **The three that are not a score.** Sinks as a parameter, differential attention
   as two blocks and a primitive, and the eager block with its rule. *Sinks are
   done*, as planned above: `sinks` on the primitive, `heads` learned scalars,
   no change to the memory. FlexAttention takes them through the log-sum-exp it
   returns — the output times `sigmoid(lse - sink)` — so they need no kernel of
   their own; the unfused form is one more column in the softmax, and the two
   are held against each other and against Hugging Face's gpt-oss attention.
   *Differential attention is done*, as two blocks and two small primitives:
   `diff_attention` draws two query and two key projections, one value
   projection twice as wide, two fused attentions over those values, and
   `diff_combine`, which owns lambda and takes the second output from the first,
   then a per-head RMSNorm and a constant `scale`. The reference implementation
   materialises both score matrices; this never does, and a test gives both the
   same weights and finds the same numbers. Two things the reference does are
   not carried: `lambda_init` is one value per block, where the paper schedules
   it by depth, and with grouped keys the reference pairs query heads with key
   heads in an interleaved order that this does not reproduce — the same
   parameters and FLOPs, a different pairing.

**Done when** Gemma 2's attention is generated fused and counted fused and a
profiler agrees with both; a windowed model's FLOPs match what its generated code
does; ALiBi and relative-bias presets reproduce their published counts; and a mask
typed into the inspector changes the cost it should.

## What this leaves open

- **Whether FlexAttention is a stable enough target.** Phase 1 found out, on
  PyTorch 2.11. Grouped-query attention works and the numbers match. But it is
  only fused when compiled, and compiling it needs a C++ compiler on the CPU and
  Triton on CUDA — neither available on Windows — and PyTorch's FLOP counter
  refuses it, which the runtime's verification depends on. So phase 1 used
  FlashAttention's `softcap` and `window_size`. Phase 2's answer for the
  expressions is to generate them as plain functions both forms can call:
  compiled `flex_attention` where it compiles, and the same functions over
  broadcast position grids everywhere else. Verification and the profiler see
  the second; the analysis counts the first; and a test holds the second
  against uncompiled `flex_attention`, to the last bit.
- **Block granularity.** A kernel computes whole blocks, 128 positions a side by
  default, so a mask that keeps a sliver of a block costs the whole block.
  Counting the kept fraction element by element undercounts by up to one block a
  row; counting it at the kernel's block size is what the kernel does. The
  second is right, and it makes the block size part of the operating point.
  Phase 2 counts the share element by element and *draws* the blocks: the
  preview under the mask shades each block by what it keeps, so a mask that
  keeps a sliver of every block is visible as one even where its count is not.
- **Inference.** Paged attention and decoding cost are unchanged by this; a mask
  that depends on a runtime input, like document ids, makes the cache question
  per-request, which the KV-cache figures would have to say.

## Sources

- FlexAttention: [the announcement](https://pytorch.org/blog/flexattention/) —
  the `score_mod` and `mask_mod` signatures, the block mask, and the soft-capping,
  ALiBi, relative-position, sliding-window, prefix-LM and document-masking
  examples used above.
- FlashAttention: [the repository](https://github.com/Dao-AILab/flash-attention)
  — "2.6: Softcapping. Support attention with softcapping, as used in Gemma-2 and
  Grok models", and sliding windows since 2.3.
- Attention sinks in gpt-oss: [Hugging Face's introduction](https://huggingface.co/blog/welcome-openai-gpt-oss)
  and [the model card](https://arxiv.org/html/2508.10925v1).
- The engine's own figures: `analyze(getPreset("gemma-2-9b"), { T, B: 1 })`, once
  as the preset is and once with `logit_softcap` set to 0.
