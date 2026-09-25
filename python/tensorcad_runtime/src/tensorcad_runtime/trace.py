"""Run a design small enough to look at, and record what it computes.

Everything else the runtime does is about a design's structure or its cost. A
trace is about its behaviour: one model, run on one input, with every weight and
every activation written down under the block path the editor already uses for
it. The editor puts those numbers in the volume view's cells, where otherwise it
draws a hash of each cell's own coordinates.

What the model is run *as* depends on what it can learn in a few seconds:

- **sort**, Karpathy's minGPT demo and the one in Brendan Bycroft's
  visualisation: read ``n`` symbols, write them back sorted. A design with a
  small vocabulary is trained until it sorts every held-out input, then run on
  one. Its context length says what ``n`` is.
- **untrained**, for everything else: the model exactly as ``init_weights()``
  leaves it, on a seeded random input. Real numbers — the ones this design
  computes on its first step — and said to be untrained wherever they are shown,
  because a trained model's attention and an untrained one's look nothing alike.

Nothing here is allowed to be approximately right. Scaled dot-product attention
is one fused operator and never hands back the attention matrix, so the matrix
is recomputed from the query and key the trace captured — after rotary position
embedding, when the design has it — and then multiplied by the captured values
and held against what the fused kernel actually produced. If they disagree, the
trace says so and leaves the matrix out rather than shipping a picture of
something the model did not do.

Nothing here reads the design's symbol names either, which are the author's own
choice. The vocabulary is the token embedding's size; an attention block is
anything holding ``q_proj``, ``k_proj``, ``v_proj`` and ``o_proj``; and its head
layout is what the generated class says it was built with.

Tensors are written in PyTorch's own conventions and shapes: an ``nn.Linear``
weight is ``[out, in]``. Deciding how a block in the view lays one out is the
view's business, not the runtime's.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
import struct
from typing import Any, Callable

from .loader import ModelFile, design_hash, find_model_class, load_module, read_design, resolve_symbols
from .packing import several_inputs

# "Small enough to look at" is the whole premise. A trace of a model this size is
# a few megabytes; of anything much larger it is not a picture any more.
MAX_PARAMS = 1_000_000

# What the sort task can be stated over. Beyond this the vocabulary is not
# symbols a reader can hold in their head, and a letter per symbol runs out.
MAX_VOCAB = 26

# How long an input an untrained trace runs, unless asked for more, and the
# most it will. Activations grow with it — a small Llama at 256 positions is a
# 19 MB file — and a drawing that long is too dense to read cell by cell anyway.
DEFAULT_POSITIONS = 32
MAX_POSITIONS = 256

TASKS = ("auto", "sort", "untrained")


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


def _vocabulary(model) -> int | None:
    """The token embedding's size: what the model can read, whatever the design calls it."""
    import torch.nn as nn

    for mod in model.modules():
        if isinstance(mod, nn.Embedding):
            return int(mod.num_embeddings)
    return None


_FIELD = re.compile(r"(\w+)=([\w.\-]+)")


def _built_with(module) -> dict[str, str]:
    """What a generated class says it was built with, read off its docstring.

    The emitter writes every resolved parameter there — `heads=3, kv_heads=3,
    head_dim=16, causal=true` — which is the layout as built, not as the design
    spelled it with symbols.
    """
    return dict(_FIELD.findall(type(module).__doc__ or ""))


def trace_model(
    model_path: str,
    *,
    task: str = "auto",
    seed: int = 1337,
    max_steps: int = 6000,
    batch: int = 64,
    lr: float = 5e-4,
    tokens: list[int] | None = None,
    positions: int | None = None,
    max_params: int = MAX_PARAMS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    if task not in TASKS:
        raise ValueError(f"the task is one of {', '.join(TASKS)}, not {task!r}")

    info = ModelFile(model_path)
    design = read_design(info)
    if design is None:
        raise ValueError("a trace needs the design beside the model, to know the block paths")
    symbols = resolve_symbols(design)
    several = several_inputs(design)
    if several:
        raise ValueError(
            f"this design takes {len(several)} inputs ({', '.join(several)}), and a trace runs token ids alone"
        )

    torch.manual_seed(seed)
    module = load_module(info.path)
    model = find_model_class(module, info)()
    if hasattr(model, "init_weights"):
        model.init_weights()
    params = sum(p.numel() for p in model.parameters())
    if params > max_params:
        raise ValueError(
            f"{params:,} parameters is not small enough to look at; the limit is {max_params:,}. "
            "Scale the design down first, with the Ladder or `tensorcad scale`."
        )

    vocab = _vocabulary(model)
    if vocab is None:
        raise ValueError("a trace reads token ids, and this design has no token embedding")
    # The longest input the design takes. `T` is reserved and always has a
    # default; a design with learned positions also says how many it has.
    context = int(symbols.get("Tmax") or symbols.get("T") or 0)
    if context < 1:
        raise ValueError("the design does not say how long an input it takes")

    sortable = 2 <= vocab <= MAX_VOCAB and context >= 3
    if task == "auto":
        task = "sort" if sortable else "untrained"
    if task == "sort" and not sortable:
        raise ValueError(
            f"the sort task needs a vocabulary of 2..{MAX_VOCAB} symbols and a context of at least 3; "
            f"this design reads {vocab} symbols over {context} positions. Use --task untrained."
        )

    steps = 0
    last_loss = float("nan")
    acc: float | None = None
    answer: list[int] | None = None
    length: int | None = None

    if task == "sort":
        length = (context + 1) // 2

        # --- train -------------------------------------------------------------
        gen = torch.Generator().manual_seed(seed)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

        def accuracy(n: int = 500) -> float:
            model.eval()
            with torch.no_grad():
                inp = torch.randint(vocab, (n, length), generator=torch.Generator().manual_seed(seed + 1))
                got = _generate(model, inp, length)
                want = torch.sort(inp, dim=1).values
                result = (got == want).all(dim=1).float().mean().item()
            model.train()
            return result

        model.train()
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
            answer = _generate(model, torch.tensor([tokens]), length)[0].tolist()
        # What the model reads when it writes its last symbol: the input and its
        # own answer so far. Every position it attends from is on screen.
        sequence = tokens + answer[:-1]
    else:
        # --- untrained: the model as initialised, on a seeded input --------------
        model.eval()
        n = min(positions or DEFAULT_POSITIONS, context, MAX_POSITIONS)
        if tokens is None:
            tokens = torch.randint(vocab, (n,), generator=torch.Generator().manual_seed(seed + 2)).tolist()
        if not 1 <= len(tokens) <= context or any(not 0 <= t < vocab for t in tokens):
            raise ValueError(f"the input must be 1..{context} token ids from 0..{vocab - 1}")
        sequence = list(tokens)
        say(f"untrained: {params:,} parameters as initialised, on {len(sequence)} seeded tokens")

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
    sorted_correctly = predicted[length - 1 :] == sorted(tokens) if task == "sort" else None

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
    attention: list[dict[str, Any]] = []
    attention_error: float | None = None
    notes: list[str] = []
    T = len(sequence)

    for name, mod in model.named_modules():
        kids = dict(mod.named_children())
        if not {"q_proj", "k_proj", "v_proj", "o_proj"} <= kids.keys():
            continue
        built = _built_with(mod)
        heads = int(built.get("heads", "0") or 0)
        kv_heads = int(built.get("kv_heads", "0") or 0) or heads
        head_dim = int(built.get("head_dim", "0") or 0)
        if not heads or not head_dim or f"{name}.o_proj" not in inputs:
            notes.append(f"{name}: the class does not say its head layout, so its attention is left out")
            continue

        def heads_of(proj: str, rope: str, count: int):
            # After rotary embedding when there is one: that is what the scores
            # are taken of. Its output is already `[B, heads, T, head_dim]`.
            if f"{name}.{rope}" in outputs:
                return outputs[f"{name}.{rope}"][0]
            return outputs[f"{name}.{proj}"][0].view(T, count, head_dim).transpose(0, 1)

        q = heads_of("q_proj", "rope_q", heads)
        k = heads_of("k_proj", "rope_k", kv_heads)
        v = outputs[f"{name}.v_proj"][0].view(T, kv_heads, head_dim).transpose(0, 1)
        if kv_heads != heads:
            k = k.repeat_interleave(heads // kv_heads, dim=0)
            v = v.repeat_interleave(heads // kv_heads, dim=0)

        # Which keys each query may see: none after it when causal, none
        # further back than the window when windowed.
        i = torch.arange(T).view(T, 1)
        j = torch.arange(T).view(1, T)
        hidden = torch.zeros(T, T, dtype=torch.bool)
        if built.get("causal", "true") == "true":
            hidden |= j > i
        window = int(built.get("window", "0") or 0)
        if window > 0:
            hidden |= (i - j) >= window

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        probs = torch.softmax(scores.masked_fill(hidden, float("-inf")), dim=-1)
        merged = (probs @ v).transpose(0, 1).reshape(T, heads * head_dim)
        actual = inputs[f"{name}.o_proj"][0]
        # Relative to the size of what is being compared: an untrained model's
        # values are small, a trained one's are not, and one fixed tolerance
        # would be too loose for one and too strict for the other.
        error = (merged - actual).abs().max().item() / max(1.0, actual.abs().max().item())
        attention_error = max(attention_error or 0.0, error)
        if error >= 1e-4:
            notes.append(
                f"{name}: the recomputed attention differs from the fused kernel by {error:.2e} "
                f"(built with {', '.join(f'{k}={v}' for k, v in built.items() if k in ('qk_norm', 'logit_softcap', 'value_embeddings', 'output_gate'))}), "
                "so it is left out rather than shown"
            )
            continue
        path, layer = design_path(name)
        # A masked score is not a zero, it is a position that was never
        # visible. NaN says that; the view draws it as empty.
        attention.append(
            {
                "path": path,
                "layer": layer,
                "scores": _tensor(scores.masked_fill(hidden, float("nan"))),
                "weights": _tensor(probs),
            }
        )

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
            "name": task,
            "length": length,
            "vocab": vocab,
            # Letters when there are few enough to read as letters; token ids otherwise.
            "symbols": [chr(ord("A") + i) for i in range(vocab)] if task == "sort" else [],
        },
        "training": {
            "seed": seed,
            "steps": steps,
            "batch": batch if task == "sort" else 0,
            "lr": lr if task == "sort" else 0,
            # None rather than NaN: JSON has no NaN, and a browser refuses the file.
            "final_loss": last_loss if steps else None,
            "held_out_accuracy": acc,
        },
        "input": tokens,
        "sequence": sequence,
        "answer": answer,
        "predicted": predicted,
        "checks": {
            "sorted_correctly": sorted_correctly,
            "attention_max_abs_error": attention_error,
            "attention_note": "; ".join(notes) if notes else None,
        },
        "weights": weights,
        "activations": activations,
        "attention": attention,
    }
