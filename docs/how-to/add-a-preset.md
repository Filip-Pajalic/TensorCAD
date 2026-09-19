# Add a preset

A preset is an assertion about a real model, so it has to be checkable. The
twenty of them are the regression suite.

## Write the spec

Most models are decoder-only and go in `SPECS` in
`packages/core/src/presets/index.ts`:

```ts
{
  name: "my-model-7b",
  family: "my-family",
  notes: "What distinguishes it, in a sentence or two.",
  layers: 32, dModel: 4096, heads: 32, kvHeads: 8,
  ffnHidden: 14336, vocab: 128256,
  norm: "rmsnorm", mlp: "gated", act: "silu",
  rope: { theta: 500000 }, tied: false,
  published: { params: 8_030_261_248, source: "https://…" },
}
```

Vision transformers use `JEPA_SPECS` and `presets/jepa.ts`; anything whose shape
does not fit a spec at all is written out block by block, as `presets/convnet.ts`
does for AlexNet.

## `published` is the point

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

## Regenerate the Go golden files

The Go port is proven against the TypeScript, so a new preset needs a new golden
file:

```bash
bun run scripts/golden.ts
go test ./packages/core-go/...
```
