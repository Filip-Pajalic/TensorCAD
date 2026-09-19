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
| `tokens` | Training token budget, for cost and Chinchilla |

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
