# `tensorcad-runtime`

The Python half of TensorCAD. The engine computes parameter counts, FLOPs and
memory symbolically and emits PyTorch; this package is what actually runs that
PyTorch, so the estimates can be checked against the framework instead of against
themselves.

Three commands, each printing **one JSON object to stdout** and human-readable
progress to **stderr**, so the TypeScript side can parse stdout without filtering.

## Install

PyTorch is deliberately not a declared dependency: it has to come from the index
that matches your GPU. Install it first, then this package.

```bash
# Blackwell (RTX 50-series, sm_120) needs a CUDA 12.8+ build.
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -e python/tensorcad_runtime
```

Verified on Windows 11, Python 3.13.5, RTX 5080 (sm_120): that index resolved to
**torch 2.11.0+cu128**, and `torch.cuda.get_arch_list()` includes `sm_120`.

CPU-only machines can use the default index (`pip install torch`). `verify` never
needs a GPU — it runs on the meta device. `smoke-train` falls back to fp32 on CPU
(`--device cpu` forces it), which is fine for a few steps and far too slow for a
real run: roughly 1.5k tokens/s versus 150k on the 5080.

Optional, for `data prepare` against a real corpus:

```bash
pip install "datasets>=2.18" "tiktoken>=0.6"     # or: pip install -e "python/tensorcad_runtime[data]"
```

Without them (or without network access) `data prepare` falls back to a
deterministic synthetic corpus, so `smoke-train` always has something to train on.

## `verify`

```bash
tensorcad-runtime verify out/gpt2-small/model.py [--batch 2] [--seq 128] [--class-name Auto]
```

Imports the generated file, instantiates the model on `torch.device("meta")` and
reports:

| Field | Meaning |
|---|---|
| `params` | `sum(p.numel() for p in model.parameters())` — tied weights counted once |
| `params_by_module` | top-level module → parameter count; a shared tensor is attributed to the first module that owns it, so the values sum to `params` |
| `tied_parameters` | groups of names that share one tensor, e.g. `["embed.weight", "head.weight"]` |
| `expected` / `matches` | the `expected = N` value parsed from the file's `__main__` block, and whether torch agrees |
| `forward` / `shapes` | a real CPU forward pass and the logits shape, or `"skipped: too large"` above 2 GiB of fp32 weights (`--max-forward-bytes`) |
| `flops` / `flops_source` | forward FLOPs from `FlopCounterMode`, measured on meta tensors, or on fake tensors, or during the real forward pass when the graph has data-dependent shapes |
| `export_ok` / `export_shapes` | whether `torch.export.export` succeeds with dynamic batch and sequence `Dim`s, and the symbolic shapes it infers |
| `warnings` | everything that degraded instead of crashing |

Failures are captured as warnings rather than exceptions: a model whose export
fails still reports its parameter count. `--no-export` and `--no-flops` skip the
slow parts.

Export is tried first with named `Dim("batch")` / `Dim("seq")`, then with
`Dim.AUTO`. Grouped-query models trip a torch symbolic-shape limitation on the
named form — the stride arithmetic behind `permute(...).reshape(...)` generates a
`min(a*T, b*T)` guard the solver will not discharge — so they land on `Dim.AUTO`,
and `export_shapes.fallback_from` records that. Both forms keep B and T symbolic.

FLOP counting and export trace the forward graph, and a design with a Python-level
loop over the sequence traces one iteration per position per layer — the
generated Mamba-2 `SSDScan` is a readable sequential reference, so Nemotron-H at
the default `--seq 128` unrolls thousands of steps. Both phases are bounded by
`--phase-timeout` (default 300 s) and report a warning instead of hanging. For
state-space designs, verify at a small sequence length:

```bash
tensorcad-runtime verify out/nemotron-h-8b/model.py --seq 16      # full report in ~35 s
```

`ok` and the exit code say whether the *tool* ran, not whether the numbers agree —
a parameter mismatch is a successful run that reports `"matches": false` and exits
0. Check `matches`, not the exit code. A crash (missing file, unknown class, no
torch) gives `"ok": false`, an `error_kind`, and exit 1.

## `smoke-train`

```bash
tensorcad-runtime smoke-train python/fixtures/tiny-gpt2/model.py \
    --steps 500 --seq 512 --batch 16 --lr 6e-4 --log-every 10 \
    --out runs/tiny-gpt2-500.jsonl
```

bf16 autocast on CUDA, AdamW (betas 0.9/0.95, decay on matrices only), cosine
schedule with warmup, gradient clipping at 1.0, fixed seed. One JSON object per
logged step goes to the `.jsonl` (`step`, `loss`, `lr`, `tokens`,
`tokens_per_second`, `peak_memory_bytes`, `seconds`, `batch`, `seq`); the summary
on stdout adds `final_loss`, `best_loss`, `device`, and `design_hash` (sha256 of
the design JSON next to the model).

A CUDA OOM is caught, the batch is halved, and the step retried — recorded in
`oom_events` and `warnings` rather than crashing.

**Initialization.** Generated models carry no weight init, so they inherit
PyTorch's defaults — and `nn.Embedding` defaults to `N(0, 1)`, which starts
cross-entropy in the hundreds instead of near `ln(vocab)`. Like every reference
trainer, `smoke-train` re-initializes to `N(0, 0.02)` with zeroed biases and
GPT-2's `1/sqrt(2L)` residual scaling. Pass `--init default` to train from the
generated file's defaults instead.

Observed on an RTX 5080 (`python/fixtures/tiny-gpt2`, 30.1M params, 20M TinyStories
tokens, 500 steps of 16x512 in 27 s at ~151k tokens/s, 6.5 GiB peak):

```
step    10   30   50   70  110  150  200  300  400  500
loss  9.76 7.18 5.01 4.43 4.14 3.87 3.83 3.48 3.42 3.23
```

### The tiny fixture

`python/fixtures/tiny-gpt2/` is a 30M-parameter GPT-2 (6 layers, d=384, 6 heads,
vocab 50257) emitted by the same generator as the real presets. Regenerate it with:

```bash
bun run python/fixtures/make-tiny-gpt2.ts
```

## `trace`

```bash
tensorcad-runtime trace out/nano-sort/model.py --out trace.json
```

Trains a design small enough to look at on minGPT's sorting task — read `n`
symbols, write them back sorted — then runs one input through it and writes every
weight and every activation to `--out`, keyed by the block paths the editor uses
(`layers.0.block.attn.q_proj` becomes `layers/block/attn/q_proj`, layer 0). The
task is read off the design: the vocabulary is its `V` and the context `Tmax`
must be odd, `2n - 1` positions for the input and all but the last of the answer.
Training stops at the first check where every held-out input is sorted; for
`nano-sort` that is 250 steps and about four seconds on a CPU.

Tensors are base64 little-endian float32 in PyTorch's own axis order. Only leaf
modules are hooked, and an input that is some other module's output is recorded
once, as a `same_as` pointer to it.

Scaled dot-product attention never hands back its matrix, so the trace recomputes
it from the captured query and key, multiplies it by the captured values and
holds the result against what the fused kernel fed the output projection. If
they differ by more than `1e-4` the matrix is left out and `checks.attention_note`
says why; `ok` on stdout is false unless the input was sorted and the attention
checked. The summary carries `model_sha256`, the hash of the generated file, which
is what the editor matches a trace to a design by.

`bun run trace` regenerates the one trace the editor ships, for `nano-sort`.

## `data prepare`

```bash
tensorcad-runtime data prepare --dataset tinystories --tokens 20000000 --out python/data
```

Streams `roneneldan/TinyStories` (or `HuggingFaceFW/fineweb-edu`, `sample-10BT`),
tokenizes with tiktoken's gpt2 encoding (vocab 50257) and writes a flat
`uint16`/`uint32` `.bin` plus a manifest:

```json
{ "vocab_size": 50257, "tokens": 20000000, "dtype": "uint16",
  "source": "roneneldan/TinyStories", "tokenizer": "tiktoken:gpt2" }
```

If the download fails — no network, missing libraries, rate limit — it falls back
to a deterministic seeded corpus (a sparse Markov chain over a 512-word dictionary
of multi-token "words", so repeated n-grams give a model real signal) and records
`"source": "synthetic"`. Force it with `--synthetic`.

`smoke-train` reads `python/data/` by default (`--data`), preferring a real corpus
over the synthetic one, and synthesizes in memory if the directory is empty.

## As a library

```python
from tensorcad_runtime import verify_model, smoke_train, prepare_data
from tensorcad_runtime.trace import trace_model

report = verify_model("out/gpt2-small/model.py")
assert report["matches"]
```

`import tensorcad_runtime` does not import torch, so the module loads (and reports a
clean `error_kind: "torch_missing"`) on machines that have not installed it.

## Tests

`packages/engine/test/python.test.ts` shells out to `tensorcad-runtime verify` for
gpt2-small and asserts the counts agree. It skips — not fails — when the runtime
or torch is missing, so the suite stays green without a GPU.

```bash
bun test packages/engine/test/python.test.ts
```
