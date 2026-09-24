# Attention expressions

Two parameters say what an attention does beyond the switches it has always
had: `mask`, which scores count, and `score`, what each becomes before the
softmax. They are FlexAttention's `mask_mod` and `score_mod`, written in a
small language the engine reads rather than in Python, because the engine has
to evaluate a mask to count what it keeps, cost a score expression per score,
check both, and print them as Python for the generated model.

They are on `sdpa`, on `gqa_attention`, and on `transformer_block` when its
`attention` is `gqa`. A block passes them to the attention it expands into.

## Names

| Name | Is |
|---|---|
| `q` | The query's position, from 0 |
| `kv` | The key's position, from 0 |
| `h` | The head, from 0 |
| `b` | The sequence's index in the batch |
| `heads` | How many query heads the attention has; a constant once the expression is on a block |
| `score` | The score, already scaled — in `score` only |
| anything else | A symbol of the design, replaced by its value |

`B` and `T` are not available. They carry a default in the symbol table, but a
mask that read `T` would be evaluated at that default rather than at the length
it runs at.

## Operators

Python's, with Python's precedence, loosest first:

| Operators | |
|---|---|
| `or`, `\|\|` | |
| `and`, `&&` | |
| `not`, `!` | |
| `<` `<=` `>` `>=` `==` `!=` | Do not chain: write `a < b and b < c` |
| `+` `-` | |
| `*` `/` `%` | `/` is true division; `%` takes the sign of the divisor, as in Python |
| unary `-` | |
| `**` | Right-associative, and tighter than a unary minus on its left: `-2 ** 2` is `-4` |

`true` and `false` are constants.

## Functions

| Function | Cost counted per score |
|---|---:|
| `tanh(x)` | 6 |
| `exp(x)`, `log(x)` | 4 |
| `sqrt(x)` | 2 |
| `abs(x)`, `floor(x)` | 1 |
| `min(a, b)`, `max(a, b)` | 1 |
| `where(cond, a, b)` | 1 |

Every other operation counts one.

## What a design may write

A `mask` is true or false for each score: it compares something. It cannot read
`score` — it decides which scores are computed at all — and it has to read at
least one of `q`, `kv`, `h` or `b`, since a mask that is the same for every
score either masks nothing or keeps nothing.

A `score` is a number, and it has to read `score`: an expression that does not
has thrown the attention scores away.

## How they combine with the switches

A score counts only if it passes `causal` (`kv <= q`), `window` (`q - kv < W`)
and the mask, so a mask can narrow attention on its own but widening it needs
`causal` off. The design's score expression is applied first and
`logit_softcap` after it, so a capped score stays within the cap whatever the
expression added. The switches mean exactly the expressions they are, which is
what the preview under the field shows: every condition, and the whole score.

## Examples

| | `causal` | `mask` | `score` |
|---|---|---|---|
| Prefix-LM, the first 16 positions seen by all | off | `kv <= q or kv < 16` | |
| A sliding window that keeps four sink tokens | on | `q - kv < W or kv < 4` | |
| Chunked attention, as in Llama 4 | on | `floor(q / C) == floor(kv / C)` | |
| Every fourth key | on | `(q - kv) % 4 == 0` | |
| Half the heads global, half local | on | `h < heads / 2 or q - kv < W` | |
| ALiBi | on | | `score - 2 ** (-8 * (h + 1) / heads) * (q - kv)` |
| Gemma 2's cap, written out | on | | `50 * tanh(score / 50)` |

## What the engine does with them

**Resolves them.** Symbols are replaced by their values and constants folded;
the inspector shows the result beside the field when it differs from what was
typed, since it is what the kernel is given. An expression that does not parse
or check is an error on the block it was written on and goes no further.

**Counts them.** The attention's matmuls are counted over the keys each query
keeps: half of them when causal, `W - W²/2T` in a causal window of `W` once the
sequence outgrows it, and for a mask, whatever share of those it keeps. That
share is measured by evaluating the mask — up to 128 query rows, and within
each up to 512 keys drawn one from each of 512 equal strata, at random within
the stratum so a periodic mask does not alias — with a generator seeded by the
mask, so the same design always measures the same. A score expression's cost is
counted over the scores the mask keeps. Memory is a fused kernel's: the output
and the log-sum-exp.

**Checks them.** `SDPA-04` is an error for a mask that keeps nothing in the
first 128 positions; `SDPA-05` a warning for one that leaves some query with no
key, naming which. `SDPA-06` notes that the layer is counted as FlexAttention
runs it. The first two are checked without an operating point, which is why
they look at the positions every sequence has.

**Draws them.** Under the `mask` field, the attention's block mask at the
operating point's sequence length, for one head at a time when the mask reads
`h`: the sequence cut into blocks along both sides, each shaded by the share it
keeps. An empty block is one the kernel skips; a full one is computed without
the mask; anything between is computed and then masked.

**Generates them.** Each distinct expression becomes a module-level function in
FlexAttention's signature,

```python
def mask_mod_1(b, h, q_idx, kv_idx):
    """kv <= q and (q - kv < 1024 or kv < 4)"""
    return (kv_idx <= q_idx) & (((q_idx - kv_idx) < 1024) | (kv_idx < 4))
```

with causal, the window and the cap folded in, and the attention calls
`expression_attention`. On CUDA that compiles `flex_attention` the first time it
is called, with a block mask built once per shape. Anywhere else, or where it
does not compile — without Triton, which includes Windows — it applies the same
two functions to the whole score matrix and says once on a GPU that it is
running unfused. That form is what a CPU verifies, profiles and exports, and a
test holds it against `flex_attention` itself.

## What they cannot say yet

- **Learned tensors.** T5's relative-position bias reads a table the block
  owns; that is the next phase.
- **Inputs at run time.** Document masking needs each position's document id,
  which is an input, not a position.
- **The cache.** A mask that bounds how far back a query looks does not shrink
  the KV-cache estimate; `window` does, and is what to use for a sliding window.
- **Block granularity.** A kernel computes whole blocks, so a mask that keeps a
  sliver of every block costs more than its share says. The count is the share;
  the preview shows the blocks.
