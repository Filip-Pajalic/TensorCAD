# Verify a design against PyTorch

The analysis is arithmetic on a graph. This checks that arithmetic against a
model PyTorch actually built.

## Setup

```bash
pip install -e python/tensorcad_runtime
```

## Run it

```bash
bun run scripts/codegen-demo.ts llama-3-8b
python -m tensorcad_runtime verify out/llama-3-8b/model.py
```

The model is built on the **meta device**, so a 405B design costs no memory to
count. You get the real parameter total, a per-module breakdown, and — where the
model is small enough — a CPU forward pass, a FLOP count and a `torch.export`
check.

## Reading the output

`"matches": true` is the assertion that matters: PyTorch's parameter count equals
the design's.

`params_by_module` is where a disagreement becomes actionable. A mismatch is
almost always one module, and the breakdown names it.

`tied_parameters` lists weights that are shared. A tied embedding and output head
appear here, and the head contributes zero to the total.

## Options

```bash
python -m tensorcad_runtime verify <model.py> --batch 2 --seq 128
```

For a convolutional design pass `--seq 1`: its token is one image.

An encoder-decoder takes two inputs, and the runtime builds both. `--seq` is
the target's length and `--source` the source's, which defaults to the design's
own `S`. The report's `inputs` says what each was called with, and
`flops_per_token` is per target token, as the analysis's figures are.

A design that keeps packed documents apart takes its documents beside its
tokens, and the runtime makes up a packing for them: documents a quarter of the
row long on average, exponentially spread, cut into rows wherever they fall.
The profiled FLOPs do not depend on it, since a profiler counts attention as if
nothing were masked. A design that also restarts its positions at every
document is given the positions of the same packing.

The harness reads the design's own `input` block for the shape and dtype of the
tensor to feed, so a vision model gets `B C H W` floats rather than token ids.

## When the forward pass is skipped

Above a few billion parameters the message is `"forward": "skipped: too large"`.
That is fp32 weights not fitting in RAM, not a failure — the parameter count is
still verified, because that happens on the meta device.

## FLOPs will not always match exactly

For a convolutional model they do, to the FLOP.

For a transformer, `torch.utils.flop_counter` reports **more** than the analysis:
GPT-2 small measures 251.78 MFLOP/token against a reported 249.42. The whole
difference is the causal mask. A profiler counts the attention operator as if
nothing were masked, because the operator's shape does not depend on the mask.

`flops.fwdTotalUnmasked` reproduces the profiler exactly; `flops.fwdTotal` is
what a fused causal kernel actually does. Both are correct answers to different
questions, and a test pins both.
