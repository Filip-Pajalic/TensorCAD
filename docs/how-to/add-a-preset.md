# Add a preset

A preset is an assertion about a real model, so it has to be checkable. The
twenty of them are the regression suite.

A preset is a *document*: `packages/core-go/presets/data/<name>.json`, named in
`index.json` beside it, embedded into the engine by `go:embed`. Adding one is
writing that document and then proving it.

## Import the config

The fastest honest way to get the document is to let the importer read the
model's own `config.json`:

```bash
bun packages/cli/src/index.ts import config.json \
  --name my-model-7b \
  --out packages/core-go/presets/data/my-model-7b.json
```

It prints two things and you want both. The parameter count is what the
analysis makes of the document, ready to be held against the model card. The
warnings are the importer saying where the document is not the model — Gemma 2's
alternating attention, a multi-token-prediction head it left out — and each one
either gets fixed by hand or written into `notes`.

The importer knows `gpt2`, `llama`, `mistral`, `mixtral`, `qwen2`, `qwen3`,
`qwen3_moe`, `gemma`, `gemma2` and `deepseek_v3`. It refuses anything else by
name rather than guessing, and a refusal is the case for copying the nearest
preset's JSON and editing it. Something that is no kind of decoder at all is
written block by block, as `ijepa-vit-h14` and `alexnet` are.

## `published` is the point

An import gives you `meta.name` and `meta.family`. The claim is the rest, and
you write it:

```json
"meta": {
  "name": "my-model-7b",
  "family": "my-family",
  "notes": "What distinguishes it, in a sentence or two.",
  "published": { "params": 8030261248, "source": "https://…" }
}
```

- `params` — what the authors report, with a `source` link to the config or paper.
- `tolerance` — **only** where the published figure is itself rounded ("22B
  active"). Say so in the notes.

If your number disagrees with the one people quote, find out why and write it
down. I-JEPA's ViT-H is 630.4M, not the 632M everyone cites, because its
positions are frozen sincos and it has no class token. That sentence is worth
more than the number.

Then add the name to `packages/core-go/presets/data/index.json`, which is both
the library's membership and the order it presents.

## Check it

From `packages/core-go`, the assertion itself — every preset against its
published count:

```bash
go test ./presets
```

The golden files are per preset, so a new preset needs new ones. Regenerate
them deliberately and read the diff: it should be your preset and nothing else.

```bash
go run ./cmd/golden
go test ./...
```

Then the compiled engine against those same answers:

```bash
bun run build:wasm
bun test packages
```

Finally against real PyTorch:

```bash
bun packages/cli/src/index.ts codegen my-model-7b --out out/my-model-7b
python -m tensorcad_runtime verify out/my-model-7b/model.py
```

See [Verify a design against PyTorch](verify-against-pytorch.md).
