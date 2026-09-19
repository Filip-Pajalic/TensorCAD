# Your first design

You will build a small transformer from an empty sheet, watch it fail a check,
fix it, and generate the PyTorch. About fifteen minutes.

You need [Bun](https://bun.sh). Python and PyTorch are optional and only used at
the last step.

```bash
git clone https://github.com/Filip-Pajalic/TensorCAD
cd TensorCAD
bun install
```

## 1. Look at something that already works

Before drawing anything, see what a finished design reports:

```bash
bun packages/cli/src/index.ts analyze gpt2-small
```

You get parameters, FLOPs per token, activation memory, KV cache and a cost
estimate. The parameter count is 124,439,808 — the number OpenAI published.
That agreement is not a coincidence and it is the point of the tool.

## 2. Start from a preset and change it

Open the editor:

```bash
bun run --cwd packages/ui dev
```

Pick `gpt2-small` from **Load preset**. You are looking at a schematic: blocks
with pins, tensors as wires, the repeated layer drawn once as a frame with a
`32×` bracket.

Open the **Symbols** tab on the right. These are the design's free variables —
`D` for the residual width, `H` for heads, `F` for the feed-forward width. Change
`D` from 768 to 1024 and watch every shape on the sheet, and every number in the
readout, move at once.

That is the whole idea: the drawing is the model, not a picture of it.

## 3. Break it on purpose

Set `H` to 7.

A finding appears: 1024 does not divide by 7, so the head dimension is not an
integer. The design rules ran on the keystroke.

Set `H` back to 16. `D / H` is 64, and the finding clears.

## 4. See what it would cost

Open the **Operating** panel. Set the batch to 8 and the sequence length to 4096,
pick an A100-80GB, and set the GPU count to 1.

The readout now says what this design needs to train at that operating point. Try
ZeRO-3 in the sharding control and watch the per-GPU number fall.

Nothing here is stored in the design. Batch size and hardware are conditions you
measure a design *under*, not properties of it.

## 5. Generate the model

```bash
bun packages/cli/src/index.ts codegen gpt2-small --out out/mine
```

`out/mine/model.py` is a runnable PyTorch module. Read it — it is the design,
compiled.

Note the `init_weights()` method. It is there because `nn.Embedding` defaults to
a unit normal, which gives GPT-2 small a next-token loss of 466 instead of 10.94
against a uniform baseline of `ln(50257) = 10.82`.

## 6. Check the generated model is really the design

If you have PyTorch:

```bash
pip install -e python/tensorcad_runtime
python -m tensorcad_runtime verify out/mine/model.py
```

This instantiates the model on the meta device and reports its true parameter
count, module by module, against what the analysis predicted. They should agree
exactly.

## Where to go next

- [Define a block inside a document](../how-to/define-a-block-in-a-document.md)
  — add your own composite without touching TypeScript
- [The shape algebra](../explanation/shape-algebra.md) — why step 3 caught that
  error rather than crashing later
- [Ports](../reference/ports.md) — what a pin actually declares
