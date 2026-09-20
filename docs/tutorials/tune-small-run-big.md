# Tune small, run big

You want to train something the size of Llama-3-8B. You have one GPU to
experiment on and a cluster you get an hour of at a time. The learning rate is
the hyperparameter that matters most and the one you cannot afford to sweep at
full size.

This tutorial does the whole loop on the command line: shrink the architecture
to something that trains on the machine in front of you, build the ladder that
carries a swept learning rate back up, and price the real run before you queue
it. Every number below is what the tool actually prints.

It assumes you have run `bun run build:wasm` once. Nothing here needs a GPU —
the last step is arithmetic about one, which is the point.

## 1. Look at what you are aiming for

```bash
bun packages/cli/src/index.ts analyze llama-3-8b --T 8192 --gpus 64
```

Among the output:

```
Training memory (64 GPUs)
  weights          14.96 GiB
  gradients        14.96 GiB
  optimizer        89.75 GiB
  activations      49.03 GiB  B=1 T=8192, logits 5.87 GiB
  total per GPU   168.69 GiB  fits 80.00 GiB? no
```

168 GiB per device against 80. Unsharded, this does not train on an H100 — which
is the first thing worth knowing and the reason step 4 exists.

## 2. Shrink it to something you can actually run

```bash
bun run scripts/scale-demo.ts
```

```
llama-3-8b-67m: 34.4M non-embedding params (target 30.0M)
changes: D 4096->640, H 32->5, Hkv 8->1, L 32->5, V 128256->50304
symbols: L=5  D=640  H=5  Hkv=1  dh=128  F=3072  V=50304
note: The closest reachable size is 15% from the target. Rounding the width to
      whole heads limits how finely the size can be tuned.
note: The embedding table is 48% of this design's weights. Shrink the vocabulary
      or tie the output projection if you want the comparison to be about the
      transformer.
train memory on an RTX 5080 at batch 16, 1024 tokens: 8.24 GiB of 16.00 GiB
```

The depth and the width move together, so the aspect ratio survives: 5 layers of
640 rather than 32 of 4096. The head *dimension* stays at 128, so every head is
the shape it was. The notes are the interesting part — the tool says where the
proxy is imperfect rather than presenting 34.4M as though it were 30M.

8.24 GiB of 16. It trains on a desktop card.

## 3. Build the ladder

The small model is only useful if what you learn on it transfers. That is what
[μP](https://arxiv.org/abs/2203.03466) is for, and `mup` says exactly what to
change at each width:

```bash
bun packages/cli/src/index.ts mup llama-3-8b
```

```
llama-3-8b laddered by D
  4 rungs, tuned at 512, heads of 128 throughout.

  base  512 wide    4 heads      253.0M  m = 1
       1024 wide    8 heads      749.3M  m = 2
       2048 wide   16 heads       2.27B  m = 4
       4096 wide   32 heads       8.03B  m = 8

Multiply the base model's settings by
                   512          1024          2048          4096
  hidden
    init     unchanged       x0.7071          x0.5       x0.3536
    rate     unchanged          x0.5         x0.25        x0.125
  output
    init     unchanged          x0.5         x0.25        x0.125
    rate     unchanged          x0.5         x0.25        x0.125
```

Read it as: sweep the learning rate at 512 wide. Whatever you find, the 4096
model wants an eighth of it on every hidden matrix and on the readout, and
exactly the same rate on the embeddings and the norms.

Three things worth noticing.

**The heads get more numerous, not wider.** 4 heads at 512, 32 at 4096, and 128
wide throughout. That is the convention the paper is stated in, and it is the one
that leaves every head the same shape it had at the base — a rung whose attention
changed shape is not the same design measured wider.

**The input row is unchanged everywhere.** An embedding's fan_in is the
vocabulary, which does not grow with the width, so nothing about it scales. Norm
gains and biases have no fan_in at all.

**The rows are measured, not assumed.** Each rung is compared against the same
design at twice the width, and a weight that widened on both sides is a hidden
one. Run it on a mixture of experts and the router lands in the *output* row —
its fan_out is the expert count, which does not move — while the experts' own
matrices land in the hidden row. No list of block types would have got both.

### Laddering the small model instead

Point it at the shrunk design and the ladder goes the other way:

```
base width 640, head dim 128
640 (5 heads, 66.6M, m=1)   1280 (10 heads, 182.4M, m=2)   2560 (20 heads, 561.3M, m=4)
```

There is nothing below 640 worth sweeping at — the rungs stop at four heads,
because a two-head model is a poor proxy whatever else it gets right — so the
rungs go up. Same question, other direction: this is what you swept at, here is
what it carries to.

## 4. Price the real run

You have a learning rate. Now find out how to fit the model you meant to train:

```bash
bun packages/cli/src/index.ts plan llama-3-8b --gpus 64 --T 8192 --limit 4
```

```
llama-3-8b on 64 x H100 SXM (80 GB)
  72.00 GiB per device after headroom, of 80.00 GiB. 378 plans priced.

  DP 64, ZeRO-3                             50.90 GiB   71% of budget
  DP 64, ZeRO-2                             65.62 GiB   91% of budget
  DP 32 x TP 2, ZeRO-1, sequence parallel   40.87 GiB   57% of budget
  DP 32 x TP 2, ZeRO-2                      58.14 GiB   81% of budget

DP 64, ZeRO-3
  weights 239.32 MiB   gradients 239.32 MiB   optimizer 1.40 GiB   activations 49.03 GiB
```

The 168 GiB from step 1 is now 50.9. Note where it went: the weights, gradients
and optimizer state are nothing once they are sharded 64 ways, and **activations
are 96% of what is left**. That is the number to attack next — recomputation,
sequence parallelism, a smaller micro-batch — and the planner prices all three
if you ask it to.

The plans are ordered by how little they ask of you, not by speed. What the tool
claims is memory, which is arithmetic. Which plan is *fastest* turns on the
interconnect and the kernels, so each one carries a note about what it costs to
run and the choice stays yours.

## 5. Generate it

```bash
bun run scripts/codegen-demo.ts llama-3-8b
python -m tensorcad_runtime verify out/llama-3-8b/model.py
```

The generated model carries an `init_weights()` method rather than relying on
PyTorch's defaults, which matters here: `nn.Embedding` defaults to a unit normal,
and on GPT-2 small that is a first-step loss of 466 against the 10.94 proper
initialization gives. Apply the ladder's multipliers on top of that method and
the swept rate transfers.

## What to read next

- [The μP ladder](../reference/cli.md#mup) — every flag, and what the rows mean.
- [Analysis outputs](../reference/analysis.md) — every number the engine reports.
- [The analysis maths](../reference/analysis-math.md) — the formulas, with
  sources, including where 6N breaks down.
- [Verify against PyTorch](../how-to/verify-against-pytorch.md) — how the
  parameter counts are checked rather than asserted.
