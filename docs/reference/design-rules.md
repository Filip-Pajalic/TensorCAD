# Design rules

Twenty checks, run on every edit. `validate(doc, options)` returns the analysis
with its findings attached; the CLI's `validate` exits 1 on any error.

Each finding carries a severity — `error`, `warning` or `info` — and the path of
the block it is about.

## Structural

| Rule | Fires when |
|---|---|
| `shape` | An interface does not match. The message names the polynomial difference, not two numbers. |
| `symbols` | A symbol is undefined, cyclic, or does not evaluate to a number. |
| `dtype` | A port that says what it carries is given the other thing: token ids into a matmul, activations where an index was wanted. Shapes agree perfectly in both cases, so nothing else catches it. |
| `documents` | A mask that keeps [packed documents](../explanation/packed-sequences.md) apart is not given the documents: what reaches it, followed back through every stack it was handed into, is not an input whose `role` is `documents`. Token ids have the same shape and element type, so the two checks above pass them. |
| `block-constraints` | A block's own `constraints()` objects — for example heads not divisible by key/value heads, or an [attention mask](attention-expressions.md) that keeps nothing (`SDPA-04`) or leaves a query with no key to attend to (`SDPA-05`). |
| `user-blocks` | A definition in `doc.defs` failed to compile. It is dropped from the catalog rather than thrown, and this says why. |
| `dangling-output` | A block produces a tensor nothing reads. |
| `unused-symbol` | A symbol is declared and never referenced. |

## Efficiency

| Rule | Fires when |
|---|---|
| `flash-head-dim` | The head dimension is one a memory-efficient attention kernel will not take. |
| `tensor-core-multiples` | A width is not a multiple that tensor cores like, so the hardware runs below peak. |
| `vocab-padding` | The vocabulary is not padded to a friendly multiple. nanoGPT pads 50,257 to 50,304 for exactly this reason. |
| `window-vs-context` | A sliding window is declared wider than the context it slides over. |
| `eager-attention` | An attention is [written out](attention-expressions.md) rather than fused, and says how much its score matrices keep for the backward pass at the operating point. They grow with the square of the context, and a fused kernel keeps none of them. |
| `attention-share` | Attention dominates the FLOPs, which at long context means the design is spending its time in the wrong place. |
| `recompute-hint` | Activation memory would fall a long way under recomputation. |

## Fit

These are the ones that save money, and they depend on the operating point —
batch, sequence length, dtype, GPU model and count, sharding plan.

| Rule | Fires when |
|---|---|
| `inference-fits` | Weights plus KV cache exceed the chosen GPUs at the chosen batch. |
| `training-fits` | Weights, gradients, optimiser state and activations exceed them under the chosen sharding. |
| `logits-memory` | The logits tensor alone is a significant fraction of memory — large vocabulary times long sequence times batch. |

## Agreement

| Rule | Fires when |
|---|---|
| `published-drift` | The design no longer reproduces the `meta.published.params` it claims. |
| `active-params-drift` | The same, for active parameters in a sparse design. |
| `chinchilla` | The training token budget is far from compute-optimal for the parameter count. |

## Severity is not yet configurable

Every rule fires at its built-in severity. Per-rule `Error / Warning / Ignore`,
stored in the document with persisted per-finding exclusions, is planned — KiCad's
DRC severity matrix is the model. See [`ROADMAP.md`](https://github.com/Filip-Pajalic/TensorCAD/blob/main/ROADMAP.md).
