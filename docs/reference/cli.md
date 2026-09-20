# Command line

```bash
bun packages/cli/src/index.ts <command> [args]
```

Every command takes a **file path or a preset name** in the same position, so
anything you can do to your design you can do to `llama-3-8b` first. The one
exception is `import`, whose argument is somebody else's file.

`--json` on any command gives machine-readable output.

## `list`

```bash
bun packages/cli/src/index.ts list [--json]
```

Presets and hardware profiles.

## `validate`

```bash
bun packages/cli/src/index.ts validate <file|preset> [--T n] [--hardware id] [--json]
```

Runs [the design rules](design-rules.md). **Exits 1 on any error**, which makes
it usable in CI.

## `analyze`

```bash
bun packages/cli/src/index.ts analyze <file|preset> \
  [--T n] [--B n] [--hardware id] [--gpus n] [--tokens n] [--json]
```

Parameters, FLOPs, KV cache, memory, throughput, cost and Chinchilla budget. See
[analysis outputs](analysis.md).

The flags are the operating point. They are conditions the design is measured
under, never stored in it.

## `codegen`

```bash
bun packages/cli/src/index.ts codegen <file|preset> --out <dir>
```

Writes `model.py` and the design alongside it. With `--json` it returns the files
without touching the disk.

## `show`

```bash
bun packages/cli/src/index.ts show <file|preset>
```

The block tree with inferred shapes, nested through repeat containers. The
fastest way to see what a design actually is.

## `diff`

```bash
bun packages/cli/src/index.ts diff <a> <b>
```

Structural and numeric difference between two designs: symbols, blocks, and what
moved in the analysis.

## `plan`

```bash
bun packages/cli/src/index.ts plan <file|preset> --gpus n
```

Every way of splitting the training across a cluster, and which of them fit.
Prices DP, TP, PP and EP, the four ZeRO stages, sequence parallelism and the
three recompute settings, then prints the plans that fit in the order of how
little they ask of you. Exits 1 when nothing fits, which makes it a check to run
before a job is queued.

```
llama-3-70b on 64 x H100 SXM (80 GB)
  72.00 GiB per device after headroom, of 80.00 GiB. 381 plans priced.

  DP 64, ZeRO-3, full recompute              32.55 GiB   45% of budget
  DP 8 x TP 8, ZeRO-2, sequence parallel     58.15 GiB   81% of budget
  DP 8 x TP 8, ZeRO-3, sequence parallel     43.77 GiB   61% of budget
```

| flag | meaning |
| --- | --- |
| `--gpus n` | how many devices there are. Required. |
| `--gpus-per-node n` | bounds the tensor-parallel degree, since splitting a matrix across a slower link than NVLink rarely pays. Default 8. |
| `--headroom f` | the fraction of device memory left for fragmentation, the allocator and the communication buffers. Default 0.1. |
| `--micro-batch 1,2,4` | micro-batch sizes to try. Default: whatever `--B` gives. |
| `--recompute k` | try only this setting rather than all three. |
| `--limit n` | how many plans to print. Default 8. |

What it claims is memory, which is arithmetic. What it does not claim is which
plan is fastest: that turns on interconnect topology, kernel implementations and
the shape of the communication schedule, none of which a parameter count knows.
Each plan carries a note about what it costs to run — the all-reduces tensor
parallelism needs, the bubble a pipeline has to fill, the all-to-all around a
sparse layer — so the choice among the plans that fit stays with you.

Expert parallelism is offered only to designs that have experts, and the result
says so when it is absent.

## `mup`

```bash
bun packages/cli/src/index.ts mup <file|preset>
```

The same design at several widths, and what to multiply the initialization and
the learning rate by at each. A learning rate tuned on a narrow model is the
right one for a wide model too, provided both are scaled by width the way
[Tensor Programs V](https://arxiv.org/abs/2203.03466) Table 3 says — so the
sweep can happen at a width that fits on one device.

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
```

| flag | meaning |
| --- | --- |
| `--widths 256,512,1024` | the rungs to build. Default: halve the design's own width down to four rungs, stopping at four heads. |
| `--base-width n` | the width the sweep happens at. Default: the narrowest rung. |

The head *dimension* is held and the head *count* grows, which is the
convention μP is stated in for transformers and the one that leaves every head
the same shape it had at the base. A width that is not a whole number of heads
is rounded to one, and the rung says so.

The classification is measured rather than asserted: each rung is compared
against the same design at twice the width, and a weight that widened on both
sides is a hidden one. That is how a mixture-of-experts router lands in the
output row — its fan_out is the expert count, which does not move — and how an
expert's own matrices land in the hidden row.

What it does not print is a learning rate. That is what the sweep at the base
rung is for.

## `import`

```bash
bun packages/cli/src/index.ts import <config.json> [--name name] [--out file] [--json]
```

Reads a Hugging Face `config.json` into a design. Knows `gpt2`, `llama`,
`mistral`, `mixtral`, `qwen2`, `qwen3`, `qwen3_moe`, `gemma`, `gemma2` and
`deepseek_v3`; refuses anything else by name rather than guessing.

`--name` names the design, which otherwise takes the config's `_name_or_path`.
`--out` is where the document goes, defaulting to `out/<name>.tensorcad.json`
and written the way `packages/core-go/presets/data` is, so an import can be
dropped in there unedited. With `--json` nothing is written and the whole
report, document included, comes back on stdout.

It prints the parameter count the analysis gets, to be compared against the
model card, and every warning about what the import could not represent
faithfully — a multi-token-prediction head left out, layers made sparse that
the model keeps dense. See [Add a preset](../how-to/add-a-preset.md).

## Scripts

Convenience wrappers around the same engine:

```bash
bun run scripts/report.ts             # the parameter regression table
bun run scripts/analyze-demo.ts       # full analysis for one preset
bun run scripts/codegen-demo.ts <p>   # writes out/<preset>/model.py
bun run scripts/scale-demo.ts         # shrink a design to a budget
bun run golden                        # regenerate the engine's golden files
```
