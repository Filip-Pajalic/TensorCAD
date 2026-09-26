# Analysis outputs

`analyze(doc, options)` returns every number at once. `validate(doc, options)`
returns the same thing with [design-rule findings](design-rules.md) attached.

Nothing in the UI computes its own numbers: one call per document and operating
point feeds every panel.

## The operating point

Conditions the design is measured under, never properties of it:

| Option | Meaning |
|---|---|
| `B` | Micro-batch size |
| `T` | Sequence length in tokens |
| `bytes` | Bytes per element of the activation dtype |
| `hardware` | GPU profile id |
| `gpus` | How many |
| `sharding` | ZeRO stage, tensor and pipeline parallel degree |
| `recompute` | `none` / `selective` / `full` |
| `precision` | How a bf16 or fp16 dtype is trained. `mixed`, the default: bf16 weights and activations over an fp32 master copy, as Megatron does it. `autocast`: `torch.autocast`, fp32 weights cast to bf16 at each matrix multiply. It keeps the residual stream and the norms in fp32 and a bf16 copy of every weight for the backward pass (reported as `memory.train.castWeights`), and splits the same 16 bytes a parameter 4/4/8 rather than 2/2/12. Ignored in fp32 |
| `tokens` | Training token budget, for cost and Chinchilla |
| `packing` | Training rows packed with documents, `{ mean, spread }`: the mean length in tokens and the coefficient of variation, 0 for fixed lengths and 1 for exponential. Absent is one document a row |

## `params`

`total`, `active`, `byPath`, `byCategory`.

Sparse designs differ in the two totals: a container carries a `total` multiplier
that drives the parameter count and an `active` multiplier that drives FLOPs. That
single mechanism is what makes mixture-of-experts work.

`byPath` holds **leaf blocks only**. A container's contribution is the sum of the
paths beneath it.

## `flops`

Per token, unless stated.

| Field | Meaning |
|---|---|
| `fwdDense` | Matmul FLOPs that do not depend on sequence length |
| `fwdAttention` | Score and value products at the given `T` |
| `fwdAttentionUnmasked` | The same counted as if nothing were masked |
| `fwdTotal` | `fwdDense + fwdAttention` — what a fused causal kernel does |
| `fwdTotalUnmasked` | What a profiler reports |
| `elementwise` | Norms, activations, RoPE, residual adds — memory-bound, excluded above |
| `trainPerToken` | Forward plus backward, plus recomputation |
| `attentionShare` | Fraction of forward FLOPs inside attention |
| `ruleOfThumb2N` / `ruleOfThumb6N` | The usual approximations, for comparison |
| `perStream` | A design with two sequences only: each one's forward pass per token of its own, `S` then `T` |
| `fwdPerExample` | A design with two sequences only: one example's forward pass, both sequences' tokens |
| `packed` | Training under the operating point's `packing`, for a design whose mask keeps documents apart: `fwdAttention`, `fwdTotal`, `trainPerToken` and `attentionShare`, and `fwdAttentionBlocks`, the attention as a block-sparse kernel computes it, every 128 × 128 block holding a kept score computed whole |

With a `packing`, every figure above except `packed` is still one document a
row, which is what serving is. `packed` is training, and the training time and
cost are counted from it. It is present only when a mask reads the documents,
because a packing changes nothing else.

For a design with a second sequence every per-token figure above is per
*target* token: the source's blocks are measured at `S`, and their share is
spread over the target's tokens. `options.S` then says what `S` was.

The two totals exist because they are answers to different questions. See
[Verify a design against PyTorch](../how-to/verify-against-pytorch.md).

For a convolutional design the token is **one image**, so `T` is 1 and every
per-token figure reads as per-image.

## `kv`

`bytesPerToken`, and the total at the current batch and sequence length. A
state-space layer keeps a fixed state per sequence rather than one growing with
every token, which is most of why hybrid stacks are cheap to serve.

## `memory`

Weights, gradients, optimiser state and activations, before and after sharding,
per GPU and in total. Activation figures follow the Megatron formulas.

Activation memory is attributed to **tensors, not blocks**: a block declares
which inputs it must keep alive, and each producing tensor is counted once even
when several blocks read it.

## `throughput`

A roofline estimate: arithmetic intensity against the chosen hardware's compute
and bandwidth, and which of the two binds.

## `cost` and `chinchilla`

Training cost at the given token budget, and how far that budget is from
compute-optimal — about twenty tokens per parameter.

## `symbols`, `infer`, `expanded`, `flat`

The intermediate results, exposed because the UI needs them: the resolved symbol
table, shapes before and after composite expansion, and the flattened list of
primitive instances the totals are summed from.
