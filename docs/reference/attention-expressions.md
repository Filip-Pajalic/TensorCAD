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
| `name(i, j, ...)` | A tensor wired into the attention's input `name`, read at those indices; see [Tables](#tables). A mask reads only a documents input, as `name(b, q)`: see [Documents](#documents) |
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
| `t5_bucket(rel, buckets, max_distance, bidirectional)` | 12 |
| a table read, `name(i, ...)` | 1 |

Every other operation counts one.

`t5_bucket` is T5's relative-position bucket for a distance `rel`, normally
`kv - q`: exact while the distance is under half the buckets, logarithmically
spaced out to `max_distance`, and the last bucket for everything further.
Two-sided (`true`, an encoder) spends half the buckets on each side; one-sided
(`false`, a decoder) puts every future key in bucket 0. It is Hugging Face's
`_relative_position_bucket`, and `buckets`, `max_distance` and `bidirectional`
have to be constants, since the kernel is built for one table. It is a built-in
rather than something to write out because written out it reads worse than the
twenty lines of Python it is.

## What a design may write

A `mask` is true or false for each score: it compares something. It cannot read
`score` — it decides which scores are computed at all — and it has to read at
least one of `q`, `kv`, `h` or `b`, since a mask that is the same for every
score either masks nothing or keeps nothing.

A `score` is a number, and it has to read `score`: an expression that does not
has thrown the attention scores away.

## Tables

A score expression can read a tensor. Any name called like a function that is
not one of the functions above is a table: `rel(t5_bucket(kv - q, 32, 128,
true), h)` reads the attention's input `rel` at row `t5_bucket(...)` and column
`h`. The attention grows an input of that name, of any shape, and so does every
block that carries the expression down to it, `gqa_attention` and
`transformer_block`, so the table is wired to the layer and passed on inside.

What goes into it is usually a `position_bias` block, a learned
`[buckets, heads]` table whose parameters are counted once. T5 keeps one per
stack, outside the repeat, handed unchanged to every layer:

```python
def score_mod_1(rel):
    """score + rel(t5_bucket(kv - q, 32, 128, true), h)"""

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + rel[t5_bucket(kv_idx - q_idx, 32, 128, True), h]

    return score_mod
```

A score that reads a table is generated as a factory: called with the tensor,
it returns the `score_mod`, with the tensor in scope, which is how FlexAttention
takes one. The attention calls `score_mod=score_mod_1(rel)` with whatever its
input is, and since that is the `nn.Parameter` itself the table is trained by
every layer that reads it. A gradient reaches it through `flex_attention` as
well as through the unfused form; a test holds both, and holds the bias against
a transcription of Hugging Face's own.

The indices are used as the generated code computes them. They have to be whole
numbers — positions, heads, `+`, `-`, `*`, `%` and `t5_bucket` are; `/` is not,
and neither is `floor` of it, which PyTorch keeps as a float — and one outside
the table is not checked. A mask cannot read a table: a mask's share is
measured by evaluating it, and a table's values are the model's.

## Documents

A mask can read one kind of tensor: each position's document, when training
rows are packed with several documents and attention is kept within each.

```
doc(b, q) == doc(b, kv)
```

`doc` is an input on the attention, `B T` integers, wired from an `input` whose
`role` is `documents`. It is read at a row and a position, which is why it takes
two indices. Anything else a mask might read is refused: a mask is counted by
evaluating it, and the engine can make up a packing but not a learned table.

Without a packing in the operating point, a row is one document and the mask
keeps everything causal does, so a design costs what it did before the mask was
written. With one, the mask is measured over rows cut from a stream of
documents drawn from it. See [Packed sequences](../explanation/packed-sequences.md).

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
| ALiBi, as `bloom-7b1` has it | on | | `score - 2 ** (-8 * (h + 1) / heads) * (q - kv)` |
| Gemma 2's cap, written out | on | | `50 * tanh(score / 50)` |
| T5's relative bias, in an encoder | off | | `score + rel(t5_bucket(kv - q, 32, 128, true), h)` |
| Packed documents kept apart, as Llama 3 trained | on | `doc(b, q) == doc(b, kv)` | |

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

## Sinks

`sinks` is not an expression but sits beside them: one learned score per head
in each row's softmax, beside the keys, so a query can put its attention
nowhere (gpt-oss). It adds `heads` parameters to the block and nothing to the
memory; a layer with sinks is FlexAttention's to run, which takes them through
the log-sum-exp it returns — the output times `sigmoid(lse - sink)` — and the
unfused form is one more column in the softmax that no value is read by. Unset,
its default, is none.

## Written out

What a single score or its position decides belongs here, on the fused kernel.
What needs every head's whole matrix at once — talking heads, which mixes the
maps across heads — cannot be fused, and `written_out` (or `talking_heads`,
which implies it) computes the attention as `attn_scores`, `attn_softmax` and
`attn_values` with the score matrix a tensor between them. Every score is
counted, masked or not, and the matrices it keeps are counted as activations;
the `eager-attention` design rule states the bytes at the operating point.
A written-out attention is causal attention and nothing else, so a mask, a
score, a window, a cap or sinks on one is an error rather than something
quietly dropped.

## What they cannot say yet

- **Other tensors in a mask.** A mask reads the documents and nothing else,
  since a mask is counted by evaluating it and the documents are the one tensor
  the engine can make up.
- **The cache.** A mask that bounds how far back a query looks does not shrink
  the KV-cache estimate; `window` does, and is what to use for a sliding window.
- **Block granularity.** A kernel computes whole blocks, so a mask that keeps a
  sliver of every block costs more than its share says. The count is the share;
  the preview shows the blocks.
