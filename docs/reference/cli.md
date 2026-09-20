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
faithfully — an alternating attention pattern flattened, a multi-token
prediction head left out. See [Add a preset](../how-to/add-a-preset.md).

## Scripts

Convenience wrappers around the same engine:

```bash
bun run scripts/report.ts             # the parameter regression table
bun run scripts/analyze-demo.ts       # full analysis for one preset
bun run scripts/codegen-demo.ts <p>   # writes out/<preset>/model.py
bun run scripts/scale-demo.ts         # shrink a design to a budget
bun run scripts/golden.ts             # regenerate the Go port's golden files
```
