# Packed sequences: a mask that reads the batch

*A proposal for M12. Phase 1 is built; the rest is not. It was written first,
as M10's and M11's were, because it decides what the training figures mean.
Before it, they counted the attention of one document filling the whole
sequence. Pretraining is rarely run that way.*

Pretraining does not pad. It concatenates documents until the sequence is full
and trains on the result, so one row of a batch holds the end of one document,
several whole ones, and the start of another. Whether attention may cross from
one document into the next is a choice. Llama 3 chose not: its model section
says it masks attention between documents in the same sequence (§3.2). It found
the mask mattered little in ordinary pretraining and a great deal in continued
pretraining on very long sequences. Zhao et al. measure the same choice and
call it intra-document causal masking. Krell et al. packed BERT's sequences
this way for the speed.

The choice also changes what a step costs, and nothing in the tool can say so
yet.

## What the engine assumes today

- **A training sequence is one document.** The attention a step pays for is
  fixed by `T`, `causal`, `window` and a mask of positions. Every per-token
  figure, and so every training time and cost, is computed that way.
- **A mask reads positions, heads and the batch index, never a tensor.** A
  score can read one since M11, T5's bias table. A mask cannot, because a mask
  is counted by *evaluating* it, and a tensor's values are not the engine's to
  know. `attention-expressions.md` lists "inputs at run time" among the things
  the language cannot say yet, with document masking as the example.
- **Training and serving are measured at the same attention.** The operating
  point already separates them where it matters: a training dtype and a serving
  dtype. There is no way to say that the training batch is packed and a request
  is not.
- **The generated attention keeps its block masks.** `expression_attention`
  caches one per mask function and shape. That is right for a mask of positions
  and wrong for one that reads the batch.

None of this is wrong for what it describes. It describes one document at a time.

## What packing changes

### The keys a query keeps

Under a document mask, a query keeps only the tokens before it in its own
document. For documents of length `L` that tile the sequence exactly, a query
keeps `L/2` keys on average, where causal attention alone keeps `T/2`.

Llama-3-8B at 8,192 tokens:

| | Keys a query keeps | Attention per token | Share of the forward pass |
|---|---:|---:|---:|
| One document | 4,096 | 2.15 GFLOP | 12.5% |
| 1,024-token documents | 512 | 268 MFLOP | 1.8% |

Training FLOPs per token fall by 11%. That is the training-time estimate
getting an eighth of its attention term right, and it moves at every sequence
length a long-context run uses.

### The mean is not enough

A token is more likely to land in a long document than a short one, simply
because a long document holds more tokens. For lengths with mean `μ` and
coefficient of variation `c`, a query keeps on average

```
μ(1 + c²) / 2
```

keys, before the sequence's edges cut documents short. Fixed lengths (`c = 0`)
give `μ/2`. Exponentially distributed lengths (`c = 1`) give `μ`, twice as much
for the same mean. Real corpora are heavier-tailed than that. So the operating
point has to describe a distribution, not just a length.

### The kernel's blocks

FlexAttention computes the score matrix in 128 × 128 blocks. It skips a block
the mask removes entirely, computes a block it keeps entirely, and computes and
then masks a block that straddles a boundary. With a causal mask alone at 8k,
only the sixty-four blocks on the diagonal straddle, and the waste is under 2%. With
documents, every boundary makes straddlers:

| Documents (aligned to blocks) | Scores kept | Scores computed | Ratio |
|---|---:|---:|---:|
| 1,024 tokens | 524,800 | 589,824 | 1.12× |
| 256 tokens | 32,896 | 49,152 | 1.49× |

These are per document. Documents that do not start on a block boundary do
worse. `attention-expressions.md` already names block granularity as a gap, and
document masking is where it stops being small.

## The design

### A documents input

`input` gains a `role`. It is `tokens` by default, which is every design today.
The new role is `documents`: a `B T` integer tensor holding each position's
document index. A mask reads it the way M11's score reads a table, by the
port's name:

```
doc(b, q) == doc(b, kv)
```

wired from the documents input into the attention, through every block above it
as the tables were. A mask may read a tensor only when the engine knows how to
make one up. A documents input qualifies; a learned table still does not.

### Packing is a condition of training

The operating point gains **packing**. It is off by default: one document per
sequence, today's numbers exactly. When on, it gives documents of mean length
`μ` and spread `c`, drawn from a gamma distribution, so `c = 0` is fixed and
`c = 1` exponential.

It applies to training only: `trainPerToken`, the training time, the cost and
the token budget. The serving figures keep one document per sequence, because a
request is one. The analysis reports the packed attention beside the unpacked,
as `flops.packed`. Like `perStream`, it is omitted when there is nothing to say,
so a design without a documents mask reports what it does today.

Packing a design that has no document mask changes nothing, since attention
then crosses documents. A rule says so as a note, since Llama 3 found the mask
optional at ordinary lengths.

### Counting by evaluating, as before

The mask is still measured by sampling it, with the same seeded, stratified
draws. For each sampled batch row, the engine first draws a packing from the
operating point's distribution, then evaluates the mask against it. The closed
forms keep the sampler honest:
- exactly `L/2` for fixed documents that tile the sequence;
- the `μ(1 + c²)/2` above, corrected for the edges, for the rest.

Beside the share, the engine reports what the kernel computes: the same draws,
rounded out to whole 128 × 128 blocks.

### Generated code

- **The model's inputs.** The model takes the documents as an input:
  `forward(self, ids, documents)`.
- **The mask.** A mask that reads them is a factory, like a score that reads a
  table. Called with the documents, it returns the `mask_mod`.
- **The block mask.** It is built once per forward pass and reused by every
  layer, as FlexAttention's FAQ advises, and not kept past the batch it was
  built for. The generated helper's cache is keyed by the mask function, and a
  factory makes a new one every call. Left alone, it would miss every time and
  grow by one mask per step.
- **The eager fallback** indexes the same tensor.

### Verification

- **Isolation.** The runtime makes up a packing for the documents input. With
  it, a generated model must show that changing the tokens of one document
  moves no other document's outputs, in the fused form and the eager one.
- **Block counts.** FlexAttention's own `BlockMask`, built for the same
  packing, must agree with the engine's block figure.

### Positions that restart

Hugging Face's `DataCollatorWithFlattening` restarts `position_ids` at each
packed example. FlashAttention's variable-length kernel then treats every
example as its own sequence. The same belongs here: a `positions` role on
`input`, read by `rope` in place of the index. It changes what the model
computes and not what it costs, so it comes after the cost is right.

### The editor

The operating point gets the packing control. The mask preview in the inspector
draws a sampled packing, so the block-diagonal shape is visible. The walkthrough
says what the mask is for.

## The phases

1. **Documents, measured.**
   - The `documents` role, a mask that reads it, packing in the operating point
     (engine, CLI and MCP), counting by sampling, and the split between training
     and serving.
   - The test is the same as M11's first phase: every golden unchanged, because
     no design has a documents input and packing is off by default.
   - The fixed-length closed form must reproduce exactly, and the gamma forms
     within the sampler's error.

   *Done.*
   - **The input and the mask.** `input` has a `role`, `tokens` by default. A
     mask may read a tensor at two indices, a row and a position, and nothing
     else. `sdpa`, `gqa_attention` and `transformer_block` grow a `B T` integer
     input for it, as they grow one for a score's table.
   - **The operating point.** `packing: { mean, spread }` in the engine, `--pack`
     and `--pack-spread` on the command line, `packing` on the MCP tools. With
     it, `flops.packed` holds the training figures and the cost is counted from
     them. Everything else stays one document a row. It is present only when a
     mask reads the documents.
   - **The rule.** `documents` follows what reaches such a mask back through
     every stack it was handed into, and objects unless it starts at an input
     whose role is `documents`. Token ids have the same shape and element type,
     so nothing else would catch them wired there.
   - **Golden files.** Every analysis, rules and codegen golden is unchanged.
     Only the catalog's prose moved, for the new parameter.

   What building it found:
   - **A row begins inside a document.** A stream cut into rows starts each
     row wherever the last one stopped, so the pieces at both ends are shorter
     documents as far as the mask can tell. Fixed 1,024-token documents at
     8,192 keep **491.1** keys a query, averaged over every phase a row can
     begin at, not the 512 of a row that begins on a boundary. That is what
     the analysis reports and what the test holds it to, exactly; the
     proposal's 512 is the special case.
   - **The long documents are the noise.** The first sampler, 64 rows of 32
     queries each, was off by 5% for exponential lengths, because which queries
     land in long documents varied from draw to draw. Now it is:
     - 1,024 rows cut from a circular stream of at least 1,024 documents,
       whose lengths are one from each stratum of the distribution, shuffled;
     - 16 queries a row;
     - for each query, keys in bands that widen by four going back from it.

     That is the same quarter of a million evaluations. Fixed lengths are
     within half a percent of the exact count. Spread-out lengths are within
     about two percent of a brute-force stream of two hundred thousand
     documents, which itself agrees with the closed form for exponential
     lengths to the token. It takes about 100 ms in the editor the first time
     and nothing after, being cached.
   - **The reference needs documents, not rows.** A brute-force reference cut
     from three thousand documents was off by two percent through its long
     documents alone, however many rows it cut.
   - **Generated code came early.** A mask that reads the documents is a
     factory called with them, as a score that reads a table is, and the eager
     form keeps documents apart. On CUDA each call still builds, and keeps, a
     block mask of its own. The generated model says so in a warning, and
     building one per batch is what phase 2 is for.
2. **Generated and verified.**
   - The factory, the block mask built once per forward pass, and the runtime's
     made-up packing.
   - The isolation test in both forms, and `BlockMask`'s own count against the
     engine's block figure.
3. **Positions that restart.** The `positions` role and `rope` reading it, held
   against Hugging Face's flattening collator.
4. **The editor and the rules.**
   - The packing control, the sampled preview and the walkthrough.
   - Two rules: packing without a document mask, and documents short enough
     against the kernel's blocks that the share understates the work by more
     than a quarter.

**Done when:**
- Llama-3-8B with a document mask, at 8,192 tokens and fixed 1,024-token
  documents, is counted at exactly 512 keys a query. *Phase 1 found that a
  row cut from a stream begins inside a document, which makes it 491.1, and
  the analysis is held to that.*
- Gamma-distributed packings agree with the size-biased mean.
- A generated model keeps its documents apart in both forms.
- FlexAttention's block count matches the engine's.
- With packing off, every golden is unchanged.

## What this leaves open

- **Real length distributions.** A gamma distribution is two numbers. A corpus
  is a histogram, and FineWeb-Edu's is not a gamma. Reading one from a file is
  the obvious next step, and it is left out until the two numbers have been
  used.
- **Which figure is the headline.** The scores a mask keeps are what the design
  asks for; the blocks are what the kernel does. This proposal reports both and
  keeps the first as the headline, as every mask is counted today. If the gap
  is large for realistic packings, the headline should move, for windows too.
- **Llama 3's own presets.** The paper trained with the mask. Carrying it in
  `llama-3-8b` and its siblings would make them say what they did, and would
  change their generated `forward` to take documents. That is a change to three
  presets' code, for a mask the paper found made little difference at their
  length.
- **Context parallelism.** Splitting a packed sequence across devices balances
  badly when documents are uneven. The planner does not model that, and this
  does not change it.

## Sources

- Llama 3: [Grattafiori et al. 2024](https://arxiv.org/abs/2407.21783), §3.2,
  on the mask between documents and where it mattered.
- [Zhao et al. 2024, Analysing The Impact of Sequence Composition on Language
  Model Pre-Training](https://arxiv.org/abs/2402.13991), on intra-document
  causal masking.
- [Krell et al. 2021, Efficient Sequence Packing without
  Cross-contamination](https://arxiv.org/abs/2107.02027).
- FlexAttention: [the announcement](https://pytorch.org/blog/flexattention/),
  for the document mask as a `mask_mod`, and its FAQ on building a batch's
  `BlockMask` once and reusing it across layers.
- Hugging Face, [Improving Hugging Face Training Efficiency Through Packing
  with Flash Attention 2](https://huggingface.co/blog/packing-with-FA2), for
  positions that restart at each packed example.
