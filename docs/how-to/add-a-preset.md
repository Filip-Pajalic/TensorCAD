# Add a preset

A preset is an assertion about a real model, so it has to be checkable. The
twenty of them are the regression suite.

## Write the document

The library is `packages/core-go/presets/data`: twenty JSON documents and an
`index.json` listing them, embedded into the binary by `go:embed`. There is no
builder to go through — a preset is a document in exactly the format the editor
saves, and the format `import` writes.

If the model has a Hugging Face `config.json`, start there:

```bash
bun packages/cli/src/index.ts import config.json \
  --name my-model-7b \
  --out packages/core-go/presets/data/my-model-7b.json
```

The importer reproduces eight of the presets to the parameter. It prints the
count the analysis gets, to be held against the model card, and every warning
about where the document is not the model — a multi-token-prediction head left
out, layers made sparse that the model keeps dense. Each warning gets fixed by
hand or written into the notes; none of them are swallowed. The families it
knows are `gpt2`, `llama`, `mistral`, `mixtral`, `qwen2`, `qwen3`, `qwen3_moe`,
`gemma`, `gemma2` and `deepseek_v3`, and it refuses anything else by name
rather than guessing.

`--out` puts the document straight into the library; the name still has to go
into `index.json`.

Otherwise start from the nearest preset rather than an empty file. Almost every
decoder-only model differs from `llama-3-8b.json` in seven numbers and a note:

```jsonc
"symbols": {
  "L":   { "kind": "design", "value": 32,     "doc": "Number of transformer layers" },
  "D":   { "kind": "design", "value": 4096,   "doc": "Residual stream width (d_model)" },
  "H":   { "kind": "design", "value": 32,     "doc": "Query heads" },
  "Hkv": { "kind": "design", "value": 8,      "doc": "Key/value heads" },
  "dh":  { "kind": "design", "value": "D/H",  "doc": "Head dimension" },
  "F":   { "kind": "design", "value": 14336,  "doc": "Feed-forward hidden width" },
  "V":   { "kind": "design", "value": 128256, "doc": "Vocabulary size" }
}
```

The graph below them is four or five nodes, because `transformer_block` and the
composites inside it carry the architecture. Change `meta.name`,
`meta.published`, the notes, and whatever the model does differently — the
normalization, the activation, the RoPE theta — and add the file to
`index.json`.

Two presets are not transformers at all and are written out block by block:
`ijepa-vit-h14.json` has bidirectional attention and no vocabulary, and
`alexnet.json` is convolutional with `B C H W` tensors. Copy those instead when
that is what you are describing.

## `published` is the point

An import gives you `meta.name` and `meta.family`. The claim is the rest, and
either way you write it yourself:

- `params` — what the authors report, with a `source` link to the config or paper.
- `tolerance` — **only** where the published figure is itself rounded ("22B
  active"). Say so in the notes.

If your number disagrees with the one people quote, find out why and write it
down. I-JEPA's ViT-H is 630.4M, not the 632M everyone cites, because its
positions are frozen sincos and it has no class token. That sentence is worth
more than the number.

## Check it

```bash
bun run scripts/report.ts
```

Your preset should say `exact`. Every other row must be unchanged.

Then against real PyTorch:

```bash
bun run scripts/codegen-demo.ts my-model-7b
python -m tensorcad_runtime verify out/my-model-7b/model.py
```

See [Verify a design against PyTorch](verify-against-pytorch.md).

## Write down what it says

The golden files are the specification, so a new preset needs its own:

```bash
cd packages/core-go
go run ./cmd/golden    # writes testdata/{golden,analysis,rules,codegen}/my-model-7b.json
go test ./...
```

Read the diff. Four new files appear and nothing else should move; a number
that changed under an existing preset means the preset was not the only thing
you touched.
