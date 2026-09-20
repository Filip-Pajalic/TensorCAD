# Command line

```bash
bun packages/cli/src/index.ts <command> [args]
```

Every command takes a **file path or a preset name** in the same position, so
anything you can do to your design you can do to `llama-3-8b` first.

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

## `import`

```bash
bun packages/cli/src/index.ts import <config.json>
```

Read a Hugging Face `config.json` into a design. The fastest honest way to get
one for a published model that is not among the twenty presets: point this at
the `config.json` from its model card.

```
Meta-Llama-3-8B  8.03B parameters
  non-embedding 7.50B
  rules 0 errors, 1 warnings
```

| flag | meaning |
| --- | --- |
| `--name n` | a name for the design. Otherwise the config's directory, which is what a download is called. |
| `--out dir\|file` | write it as `.tensorcad.json`. A directory gets `<name>.tensorcad.json`. |
| `--json` | the whole document on stdout. |

Anything the importer cannot model faithfully comes back as a warning rather
than being approximated silently, and warnings go to stderr so a redirected
`--json` stays a document and nothing else. It exits non-zero when the import
warned or when the design breaks a rule outright — an import that could not
model the architecture is not a success, whatever the parameter count says.

Known families: GPT-2, Llama, Mistral, Mixtral, Qwen2, Qwen3, Qwen3-MoE, Gemma,
Gemma 2 and DeepSeek-V3. The eight in `packages/core-go/testdata/hf-configs.json`
are held to reproducing the hand-written preset exactly, on the parameter count
and on the cache.

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

## Scripts

Convenience wrappers around the same engine:

```bash
bun run scripts/report.ts             # the parameter regression table
bun run scripts/analyze-demo.ts       # full analysis for one preset
bun run scripts/codegen-demo.ts <p>   # writes out/<preset>/model.py
bun run scripts/scale-demo.ts         # shrink a design to a budget
bun run golden                        # regenerate the engine's golden files
```
