# Encoder–decoder: a second sequence

*A proposal for M11. Phases 1 and 2 are built; the rest is not. It was written first
for the same reason M10's was: the choice decides what "per token" means in
every number the tool reports.*

M10 set itself a finishing line it has not crossed: "ALiBi and relative-bias
presets reproduce their published counts". The relative-bias model is T5. T5 is
an encoder–decoder, and every part of the engine assumes there is one sequence.
This page says where that assumption lives, what T5 needs, and a design that
adds a second sequence without changing what any existing number means.

## What the engine assumes today

There is one stream of tokens, and the engine is built around it.

- **Two runtime symbols, `B` and `T`.** Every shape is a polynomial over them
  and the design's own symbols. Shape checking keeps them indeterminate, so a
  mismatch is a real difference, not a coincidence of numbers.
- **"Per token" means per token of that stream.** FLOPs, activation memory and
  the KV cache are computed per token and multiplied by `B·T`.
- **Attention's keys are `T` long.** `sdpa` takes `q`, `k` and `v` of the same
  length. Its FLOPs grow with `T`, and its cache grows one token at a time.
- **A repeat carries its inputs.** A stack of layers passes every input on to
  the next copy and takes every output back. There is no way to hand every copy
  the same tensor unchanged.
- **An expression reads positions, not tensors.** A `mask` or `score` can read
  `q`, `kv`, `h`, `b`, `heads`, `score` and the design's symbols, and nothing
  that flows along a wire.

None of these is wrong for a decoder-only model. T5 breaks all five.

## What T5 needs

| | t5-small | flan-t5-base |
|---|---|---|
| Width, heads × head width | 512, 8 × 64 | 768, 12 × 64 |
| Encoder layers, decoder layers | 6, 6 | 12, 12 |
| Feed-forward | ReLU, 2,048 | gated GELU, 2,048 |
| Vocabulary, shared by both stacks | 32,128 | 32,128 |
| Output head | tied to the embedding | its own |
| Parameters | **60,506,624** | **247,577,856** |

Both counts reproduce by hand from the configurations. flan-t5-base's is
exactly what its checkpoint holds. t5-small's checkpoint holds 256 more: a
relative-bias table on the decoder's first cross-attention, which the model
never reads and Hugging Face's loader ignores. That is the kind of discrepancy a
preset's notes exist for.

What is new:

1. **Two stacks over two sequences.** The encoder reads the source, `S` tokens
   long, with bidirectional self-attention. The decoder reads the target, `T`
   tokens long, with causal self-attention, and cross-attention from each target
   position to all `S` encoder outputs.
2. **Cross-attention.** Queries `T` long, keys and values `S` long. It is not
   causal, so it counts `S` keys for every query, not half of them. Its K/V
   cache is computed once per request, from the encoder, rather than growing
   with the output.
3. **Relative position bias.** Each stack has one learned table,
   `[32 buckets, heads]`. A score gets `table[bucket(kv − q), h]` added to it,
   where the bucket is exact for small distances and logarithmic out to 128.
   The encoder's buckets are two-sided and the decoder's one-sided.
   Cross-attention has none. The first layer owns the table and **every layer
   of the stack reads the same one**: 256 parameters for t5-small's encoder, not
   256 per layer.
4. **No `1/√d` on the scores.** T5 folds it into the initialisation, so its
   attention is unscaled.
5. **The encoder's output reaches every decoder layer**, unchanged.

## The design

### A second runtime symbol, `S`

The source length becomes a third reserved runtime symbol beside `B` and `T`.
The encoder's tensors are `B S D`, and cross-attention's keys are `B heads S dh`.
Shape checking treats `S` as it treats `T`, as indeterminate. So wiring an
encoder output into a decoder's self-attention is `S` against `T`, a
polynomial mismatch that the checker reports. The operating point gains a source
length beside the sequence length. It defaults to `T`, so a design that never
mentions `S` measures exactly as it does today.

### A token is a token of its own stream

The analysis already knows which stream a block runs on, from the shapes of its
tensors. It would multiply each block's per-token cost by the tokens of that
stream: `B·S` for the encoder, `B·T` for the decoder. Cross-attention is a
decoder block that reads `S` keys, so it costs `4·S·heads·head_dim` per target
token.

Every existing design has one stream, `T`. Nothing it reports changes, and a
test would hold every golden to that before anything else in M11 is merged.

The headline figures need a decision rather than a formula: "per token" is
ambiguous when a training example is `S + T` tokens of two different kinds.
The proposal is to report **per example** and **per target token**, with the
encoder's cost spread over the target tokens. Per example is what one training
step processes. Per target token is how a sequence-to-sequence trainer's
throughput is usually quoted. The 6N rule of thumb is shown against the
per-example figure, since that is the only one it is meaningful for.

### A repeat can hand every copy the same tensor

A repeat's inputs are *carried*: each copy's outputs become the next copy's
inputs. This adds *broadcast* inputs: a port marked as the same for every copy,
with no matching output. It is a property of the boundary, shown on the sheet
as a wire that enters the stack and reaches every layer. The generated loop
passes it to every call rather than threading it through. The decoder's
encoder output is one; T5's shared bias table is another.

### Expressions can read tensors

A `mask` or `score` expression may read a tensor wired into the attention, by
the port's name:

```
score + rel(t5_bucket(kv - q, 32, 128, true), h)
```

`rel` is a port on the attention. Here it is a `[buckets, heads]` table, owned
by a new `position_bias` block that counts its parameters once and sits outside
the repeat, broadcast to every layer. `t5_bucket` is a built-in, because its
definition — exact below half the buckets, logarithmic to the maximum distance,
clamped, and folded for two-sided attention — reads worse written out than the
five lines of Python it is. Its cost is counted like any other function's.

FlexAttention takes the same thing. A `score_mod` can capture a tensor and
index it, and its announcement's relative-position example does exactly that.
So the generated code is the design's expression with the table in scope, and
the eager fallback indexes the same table. Whether gradients reach a captured
table in the PyTorch version the runtime uses was the first thing phase 3
checked: they do, through `flex_attention` as through the fallback. The
same mechanism is what document masking wants, with a runtime input rather than
a learned table: `doc(q) == doc(kv)`, a mask for packed pretraining, left for
M12.

### An explicit scale on `sdpa`

`scale`, unset by default, meaning `1/√head_dim` as today. T5 sets it to one.
It is free: a scale does not change FLOPs or memory. It exists because the
generated code must compute what the model computes.

### Code generation and verification

- **The model:** the generated model takes the source and the target, runs the
  encoder once, and hands its output to every decoder layer as a broadcast
  input. Cross-attention is `sdpa` with keys of length `S`, which PyTorch's
  kernel takes as it is.
- **Verification:** the runtime makes two integer inputs rather than one, and
  exports with both lengths dynamic.
- **The editor:** it draws two inputs.
- **The volume view:** it stays a picture of one stream. It says so on an
  encoder–decoder, rather than drawing half a model as though it were all of
  one.

## The phases

1. **The second sequence.** `S`, the operating point's source length, and
   per-stream accounting. Every golden unchanged is the test, and a
   two-stream design with no cross-attention yet measures each stack at its own
   length.
   *Done.* A design with a source declares `S` as a runtime symbol. `S` is
   reserved for that and cannot be a design symbol, and a design that never
   declares it is untouched: every golden was rewritten and not one byte
   moved. The analysis finds which blocks run along the source from the
   shapes on their pins, measures each at its own stream's length, and
   spreads the source's per-token figures over the target's tokens. It
   reports each stream per token of its own and one example whole
   (`perStream`, `fwdPerExample`), and caches a source per request. The
   operating point, the CLI (`--S`) and the MCP tools take the source length.
   Building it found a sixth assumption the proposal had missed: every block
   *declares* its pins over `T` (`B heads T head_dim`, a rearrange from
   `B T (H dh)`), so an encoder built from them failed shape checking inside.
   In a declaration `T` now means *this block's sequence*, which is `S` for a
   block that everything arrives at `S` long. A block that receives both is
   left alone, so the source and the target meeting where a block takes one is
   still an error. Every block the catalog has runs along a source without
   being written again.
2. **Cross-attention.**
   - `sdpa` with keys of length `S`, not causal, counted at `S` keys a query,
     with a per-request cache.
   - Broadcast inputs on a repeat, and `scale` on `sdpa`.
   - Generated code with two inputs, verified with two.
   *Done.*
   - **Cross-attention.** `cross` on `sdpa` declares its keys and values `S`
     long. It is counted at every one of them for every target token, and its
     cache is `S` positions once per request. `SDPA-07` refuses causal, a
     window, a mask, a score, sinks or a cap on it.
   - **The composites.** `cross_attention` projects its keys and values from a
     `memory` input. `cross_attention` on `transformer_block` adds that input
     and a third norm between the block's own attention and its feed-forward.
     `scale` on `sdpa` is what T5 will set to one.
   - **A stack's shared input.** An input the stack's layer does not give back
     is handed to every copy unchanged. That needs no new syntax: the loop
     passes it as an argument rather than threading it through. Memory charges
     it once, as the tensor it is outside the stack, following it up through
     every boundary it crossed on the way in.
   - **Generated code.** A model with several inputs takes each by its block's
     name, `forward(self, src, tgt)`; one input is `ids`, byte for byte.
   - **Verification.** The runtime builds every input, takes `--source`, and
     exports with the batch, the target and the source all dynamic.

   Two things the proposal did not foresee:
   - **Batch binding.** A port that spells every axis out no longer takes part
     in agreeing a batch, so a block can read `B T D` and `B S D` together.
   - **The encoder's cache.** An encoder caches nothing: it runs once, before
     anything is generated, and what generation keeps of the source is
     cross-attention's keys and values. Phase 1 had counted an encoder's own
     cache per request, which would have counted the source twice.

   A small encoder-decoder verifies in PyTorch. The profiler's count per target
   token, 625,152, is the analysis's exactly: the encoder's share is spread
   over the target, and cross-attention reads all 24 source positions.
3. **Tensors in expressions.** Ports read by name, the `position_bias` block,
   `t5_bucket`, and FlexAttention's captured table. Held against a transcription
   of Hugging Face's own `_relative_position_bucket` and bias computation.
   *Done.*
   - **The language.** A name called like a function that is not a function is
     a table, read at the indices it is given, in a score and not a mask.
     `t5_bucket` is a built-in whose last three arguments must be constants.
   - **The ports.** `sdpa` has an input for each table its score reads, and
     `gqa_attention` and `transformer_block` have the same input and wire it
     through to the attention, so a table is wired to the layer. The input
     takes any shape, which is new: `*` is a pattern every shape matches.
   - **`position_bias`.** A `[buckets, heads]` table, `buckets × heads`
     parameters, no inputs. In generated code it is an `nn.Parameter`, and
     what flows along its wire is that parameter, so every layer reading it
     reads the same one.
   - **Generated code.** A score that reads a table is a factory, called with
     the table and returning the `score_mod`, with `t5_bucket` emitted beside
     it when it is used.

   Checked in PyTorch 2.11: the generated bias is Hugging Face's to the bit
   over 300 positions, two-sided and one-sided; uncompiled `flex_attention`
   gives what the unfused form gives; and a gradient reaches the table through
   both. The last was the open question, and it means the table is trained by
   the fused kernel, not only by the fallback. A small T5-like model, a table
   per stack handed to every layer, verifies: its count, its profiled FLOPs per
   target token, a forward pass and an export. The tables add their 256
   parameters and nothing to the matmuls; the bias is fifteen operations a
   score, counted with the elementwise work, where the profiler does not look.
4. **The presets.** `t5-small` and `flan-t5-base`, exact. The notes carry the
   checkpoint's unused table.

**Done when** both presets reproduce their counts in the analysis and in
PyTorch; their bias is checked against Hugging Face's construction of it; a
profiler agrees with the FLOPs of a small encoder–decoder; and every
decoder-only golden is unchanged. M10's relative-bias clause is then met, and
M10 closes.

## What this leaves open

- **What to call "a token" in the headline.** Per example and per target token
  is a proposal, not a finding. It should be checked against how the T5 and UL2
  papers quote compute before it is built.
- **Tied heads that are scaled.** t5-small multiplies the decoder's output by
  `d_model^-0.5` before its tied head, and flan-t5 does not. That is a switch on
  `lm_head`, and it changes what the model computes, not what it costs.
- **Prefix-LM.** T5's own paper compares an encoder–decoder with a decoder-only
  model that is bidirectional over a prefix. That is already drawable with
  M10's mask, `kv <= q or kv < P`, and the comparison between the two is the
  tool's to make once both exist.

## Sources

- T5: [Raffel et al. 2020](https://arxiv.org/abs/1910.10683), §2.1 on the
  relative position embedding, and Hugging Face's `modeling_t5.py` for the
  bucket function and the sharing across layers.
- Configurations and checkpoint counts:
  [google-t5/t5-small](https://huggingface.co/google-t5/t5-small) and
  [google/flan-t5-base](https://huggingface.co/google/flan-t5-base).
- FlexAttention's learned biases:
  [the announcement](https://pytorch.org/blog/flexattention/), on `score_mod`
  capturing tensors, with the relative-position example.
