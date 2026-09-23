"""Train a design small enough to look at, and record what it computes.

Everything else the runtime does is about a design's structure or its cost. A
trace is about its behaviour: one model, trained on a task, run on one input,
with every weight and every activation written down under the block path the
editor already uses for it. The editor puts those numbers in the volume view's
cells, where until now it drew a hash of each cell's own coordinates.

The task is Karpathy's minGPT sorting demo, which is also the one in Brendan
Bycroft's visualisation: read `n` symbols, write them back sorted. A model
reads `2n - 1` positions — the input, then its own answer so far — and its
context length says what `n` is.

Nothing here is allowed to be approximately right. Scaled dot-product attention
is one fused operator and never hands back the attention matrix, so the matrix
is recomputed from the query and key the trace captured — and then multiplied by
the captured values and compared with what the fused kernel actually produced.
If they disagree the trace says so and leaves the matrix out rather than
shipping a picture of something the model did not do.

Tensors are written in PyTorch's own conventions and shapes: an `nn.Linear`
weight is `[out, in]`. Deciding how a block in the view lays one out is the
view's business, not the runtime's.
"""

from __future__ import annotations

import base64
import hashlib
import math
import struct
from typing import Any, Callable

from .loader import ModelFile, design_hash, find_model_class, load_module, read_design, resolve_symbols

# "Small enough to look at" is the whole premise. A trace of a model this size is
# a few hundred kilobytes; of anything much larger it is not a picture any more.
MAX_PARAMS = 1_000_000

# What the sort task can be stated over. Beyond this the vocabulary is not
# symbols a reader can hold in their head, and a letter per symbol runs out.
MAX_VOCAB = 26


def _floats(t) -> str:
    """A tensor as base64 little-endian float32, which is exact and compact."""
    values = t.detach().to("cpu").float().contiguous().flatten().tolist()
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")


def _tensor(t) -> dict[str, Any]:
    return {"shape": list(t.shape), "data": _floats(t)}


def _one(t):
    """The single sequence out of a batch of one.

    Not every tensor has a batch axis: the position embedding is looked up by
    position alone and is `[T, C]`, so only a leading axis of one is dropped.
    """
    return t[0] if t.dim() > 0 and t.shape[0] == 1 else t


def design_path(name: str) -> tuple[str, int | None]:
    """A torch module name as the editor's block path, and the layer it is in.

    Generated modules are named after the design's nodes, so the mapping is a
    matter of dropping the index a repeated stack puts in the name:
    `layers.0.block.attn.q_proj` is `layers/block/attn/q_proj` in layer 0.
    """
    parts: list[str] = []
    layer: int | None = None
    for segment in name.split("."):
        if segment.isdigit():
            layer = int(segment)
        else:
            parts.append(segment)
    return "/".join(parts), layer


def _sort_batch(gen, batch: int, length: int, vocab: int):
    """minGPT's SortDataset: the input, then its sorted form, predicted a token at a time."""
    import torch

    inp = torch.randint(vocab, (batch, length), generator=gen)
    sol = torch.sort(inp, dim=1).values
    cat = torch.cat([inp, sol], dim=1)
    x = cat[:, :-1].clone()
    y = cat[:, 1:].clone()
    # Only the answer is graded. The model is not asked to predict its input.
    y[:, : length - 1] = -1
    return x, y


def _generate(model, inp, length: int):
    """Greedy: the answer the model would actually give, a token at a time."""
    import torch

    x = inp
    for _ in range(length):
        logits = model(x)
        x = torch.cat([x, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    return x[:, length:]


def trace_model(
    model_path: str,
    *,
    seed: int = 1337,
    max_steps: int = 6000,
    batch: int = 64,
    lr: float = 5e-4,
    tokens: list[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    info = ModelFile(model_path)
    design = read_design(info)
    if design is None:
        raise ValueError("a trace needs the design beside the model, to know the task and the block paths")
    symbols = resolve_symbols(design)

    vocab = int(symbols.get("V") or 0)
    context = int(symbols.get("Tmax") or symbols.get("T") or 0)
    if not 2 <= vocab <= MAX_VOCAB:
        raise ValueError(f"the sort task needs a vocabulary of 2..{MAX_VOCAB} symbols, and this has {vocab}")
    if context < 3 or context % 2 == 0:
        raise ValueError(f"the sort task reads 2n-1 positions, so the context must be odd; it is {context}")
    length = (context + 1) // 2

    torch.manual_seed(seed)
    module = load_module(info.path)
    model = find_model_class(module, info)()
    if hasattr(model, "init_weights"):
        model.init_weights()
    params = sum(p.numel() for p in model.parameters())
    if params > MAX_PARAMS:
        raise ValueError(f"{params:,} parameters is not small enough to look at; the limit is {MAX_PARAMS:,}")

    # --- train ---------------------------------------------------------------
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    def accuracy(n: int = 500) -> float:
        model.eval()
        with torch.no_grad():
            inp = torch.randint(vocab, (n, length), generator=torch.Generator().manual_seed(seed + 1))
            got = _generate(model, inp, length)
            want = torch.sort(inp, dim=1).values
            acc = (got == want).all(dim=1).float().mean().item()
        model.train()
        return acc

    model.train()
    steps = 0
    last_loss = float("nan")
    acc = 0.0
    while steps < max_steps:
        x, y = _sort_batch(gen, batch, length, vocab)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1
        last_loss = loss.item()
        if steps % 250 == 0:
            acc = accuracy()
            say(f"step {steps}: loss {last_loss:.4f}, sorts {acc:.1%} of held-out inputs")
            if acc == 1.0:
                break
    if acc < 1.0:
        acc = accuracy()

    # --- one input -----------------------------------------------------------
    model.eval()
    if tokens is None:
        # Bycroft's own example when the task is his; otherwise a seeded draw.
        if vocab == 3 and length == 6:
            tokens = [2, 1, 0, 1, 0, 2]
        else:
            tokens = torch.randint(vocab, (length,), generator=torch.Generator().manual_seed(seed + 2)).tolist()
    if len(tokens) != length or any(not 0 <= t < vocab for t in tokens):
        raise ValueError(f"the input must be {length} symbols from 0..{vocab - 1}")

    with torch.no_grad():
        inp = torch.tensor([tokens])
        answer = _generate(model, inp, length)[0].tolist()
    # What the model reads when it writes its last symbol: the input and its own
    # answer so far. Every position it attends from is on screen.
    sequence = tokens + answer[:-1]

    # --- record --------------------------------------------------------------
    # Only the leaves. A container's output is its last leaf's output, and
    # recording both would store every tensor two or three times over.
    outputs: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    hooks = []
    for name, sub in model.named_modules():
        if not name or any(True for _ in sub.children()):
            continue

        def after(_m, _args, out, name=name):
            if torch.is_tensor(out):
                outputs[name] = out.detach().clone()

        def before(_m, args, name=name):
            if args and torch.is_tensor(args[0]):
                inputs[name] = args[0].detach().clone()

        hooks.append(sub.register_forward_hook(after))
        hooks.append(sub.register_forward_pre_hook(before))

    with torch.no_grad():
        logits = model(torch.tensor([sequence]))
    for h in hooks:
        h.remove()

    predicted = logits[0].argmax(-1).tolist()
    # Positions length-1 onward are where the answer is written; everything the
    # trace shows is only worth showing if the model actually got it right.
    sorted_correctly = predicted[length - 1 :] == sorted(tokens)

    # Every distinct tensor once. An input that is some other leaf's output —
    # the three projections all read the norm's — is recorded as that output,
    # so a reader can still ask for it by the name it has at either end. The
    # inputs that are nobody's output are the ones the model computes between
    # modules: the residual sums, the merged heads, the activation function.
    activations: dict[str, Any] = {}
    for name, t in outputs.items():
        path, layer = design_path(name)
        activations[f"{name}:out"] = {"path": path, "layer": layer, "role": "out", **_tensor(_one(t))}
    for name, t in inputs.items():
        path, layer = design_path(name)
        entry: dict[str, Any] = {"path": path, "layer": layer, "role": "in"}
        twin = next((o for o, u in outputs.items() if u.shape == t.shape and torch.equal(u, t)), None)
        if twin is not None:
            entry["same_as"] = f"{twin}:out"
        else:
            entry.update(_tensor(_one(t)))
        activations[f"{name}:in"] = entry

    weights: dict[str, Any] = {}
    for name, p in model.named_parameters():
        module_name, _, param = name.rpartition(".")
        path, layer = design_path(module_name)
        weights[name] = {"path": path, "layer": layer, "param": param, **_tensor(p)}

    # --- attention, recomputed and checked ----------------------------------
    heads = int(symbols.get("H") or 0)
    kv_heads = int(symbols.get("Hkv") or heads)
    head_dim = int(symbols.get("dh") or 0)
    attention: list[dict[str, Any]] = []
    attention_error: float | None = None
    attention_note: str | None = None

    q_names = sorted(n for n in outputs if n.endswith(".attn.q_proj"))
    if heads and head_dim and q_names:
        worst = 0.0
        found = []
        T = len(sequence)
        causal = torch.ones(T, T).triu(1).bool()
        for q_name in q_names:
            base = q_name[: -len(".q_proj")]
            if f"{base}.o_proj" not in inputs:
                continue
            q = outputs[f"{base}.q_proj"][0].view(T, heads, head_dim).transpose(0, 1)
            k = outputs[f"{base}.k_proj"][0].view(T, kv_heads, head_dim).transpose(0, 1)
            v = outputs[f"{base}.v_proj"][0].view(T, kv_heads, head_dim).transpose(0, 1)
            if kv_heads != heads:
                k = k.repeat_interleave(heads // kv_heads, dim=0)
                v = v.repeat_interleave(heads // kv_heads, dim=0)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            masked = scores.masked_fill(causal, float("-inf"))
            probs = torch.softmax(masked, dim=-1)
            merged = (probs @ v).transpose(0, 1).reshape(T, heads * head_dim)
            worst = max(worst, (merged - inputs[f"{base}.o_proj"][0]).abs().max().item())
            path, layer = design_path(base)
            # A masked score is not a zero, it is a position that was never
            # visible. NaN says that; the view draws it as empty.
            found.append(
                {
                    "path": path,
                    "layer": layer,
                    "scores": _tensor(scores.masked_fill(causal, float("nan"))),
                    "weights": _tensor(probs),
                }
            )
        attention_error = worst
        if worst < 1e-4:
            attention = found
        else:
            attention_note = (
                f"the recomputed attention differs from the fused kernel by {worst:.2e}, "
                "so it is left out rather than shown"
            )
    else:
        attention_note = "no attention block the trace knows how to recompute"

    return {
        "version": 1,
        "design": design.get("meta", {}).get("name"),
        "design_hash": design_hash(info),
        # What the numbers are numbers *of*. The generated model is the whole
        # computation, so a design that still generates these bytes is one this
        # trace describes, and any edit that changes what it computes changes them.
        "model_sha256": hashlib.sha256(info.source.encode("utf-8")).hexdigest(),
        "params": params,
        "symbols": {k: int(v) if float(v).is_integer() else v for k, v in sorted(symbols.items()) if k not in ("B", "T")},
        "task": {
            "name": "sort",
            "length": length,
            "vocab": vocab,
            "symbols": [chr(ord("A") + i) for i in range(vocab)],
        },
        "training": {
            "seed": seed,
            "steps": steps,
            "batch": batch,
            "lr": lr,
            "final_loss": last_loss,
            "held_out_accuracy": acc,
        },
        "input": tokens,
        "sequence": sequence,
        "answer": answer,
        "predicted": predicted,
        "checks": {
            "sorted_correctly": sorted_correctly,
            "attention_max_abs_error": attention_error,
            "attention_note": attention_note,
        },
        "weights": weights,
        "activations": activations,
        "attention": attention,
    }
