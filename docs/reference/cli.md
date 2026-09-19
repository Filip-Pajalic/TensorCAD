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

## Scripts

Convenience wrappers around the same engine:

```bash
bun run scripts/report.ts             # the parameter regression table
bun run scripts/analyze-demo.ts       # full analysis for one preset
bun run scripts/codegen-demo.ts <p>   # writes out/<preset>/model.py
bun run scripts/scale-demo.ts         # shrink a design to a budget
bun run scripts/golden.ts             # regenerate the Go port's golden files
```
