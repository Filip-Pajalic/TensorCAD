"""Measure what training a generated model holds in GPU memory.

The analysis says what a training step keeps: weights, gradients, optimizer
state, and the activations saved for the backward pass. This runs a few real
steps on the GPU and says the same things from PyTorch's own allocator, so the
two can be held against each other the way the parameter count and the FLOPs
already are.

It measures at rest and in flight. At rest, between steps, is the part that
does not depend on the batch: each tensor is counted directly, so weights,
gradients and optimizer state come out separately rather than as one total.
In flight is what one forward pass adds and what the whole step peaks at, from
the allocator's own counters.

Three recipes, because the answer depends on them:

``amp``
    fp32 weights, bf16 autocast, fp32 AdamW state. What ``smoke-train`` and most
    PyTorch training scripts do.
``bf16``
    The whole model in bf16, AdamW state included.
``fp32``
    Everything in fp32.

Random token ids stand in for data: memory does not depend on which tokens.
"""

from __future__ import annotations

import contextlib
from typing import Any

from .loader import ModelFile, find_model_class, load_module, read_design, vocab_from_design, vocab_from_model
from .packing import several_inputs

RECIPES = ("amp", "bf16", "fp32")


def _nbytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def measure_memory(
    model_path: str,
    batch: int = 2,
    seq: int = 512,
    recipe: str = "amp",
    steps: int = 3,
    class_name: str | None = None,
    seed: int = 1337,
    progress=None,
) -> dict[str, Any]:
    """Run ``steps`` AdamW steps on the GPU and report what they held."""
    import torch
    import torch.nn.functional as F

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; use one of {', '.join(RECIPES)}")
    if steps < 2:
        raise ValueError("measure needs at least two steps: the first one allocates what the rest reuse")
    if not torch.cuda.is_available():
        # Host memory is shared with everything else the process does, and a
        # peak read from it would not be a claim about the model.
        return {
            "ok": False,
            "error_kind": "no_cuda",
            "error": "measuring training memory needs a CUDA device, and there is none",
        }

    info = ModelFile(model_path)
    design = read_design(info)
    several = several_inputs(design)
    if several:
        raise ValueError(
            f"this design takes {len(several)} inputs ({', '.join(several)}), and measure feeds token ids alone"
        )

    torch.manual_seed(seed)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    module = load_module(info.path)
    cls = find_model_class(module, info, class_name)
    say(f"building {cls.__name__} on {torch.cuda.get_device_name(0)}, recipe {recipe}")
    with torch.device(device):
        model = cls()
    if hasattr(model, "init_weights"):
        model.init_weights()
    if recipe == "bf16":
        model = model.to(dtype=torch.bfloat16)
    model.train()
    params = sum(p.numel() for p in model.parameters())
    vocab = vocab_from_design(design) or vocab_from_model(model) or 32000

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if recipe == "amp" else contextlib.nullcontext()
    )

    readings: list[dict[str, int]] = []
    for step in range(steps):
        x = torch.randint(0, vocab, (batch, seq), device=device)
        y = torch.randint(0, vocab, (batch, seq), device=device)
        torch.cuda.synchronize()
        resting = torch.cuda.memory_allocated() - baseline
        torch.cuda.reset_peak_memory_stats()
        with autocast:
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
        torch.cuda.synchronize()
        forward = torch.cuda.memory_allocated() - baseline
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        readings.append(
            {
                "resting": resting,
                "saved": forward - resting,
                "peak": torch.cuda.max_memory_allocated() - baseline,
            }
        )
        del x, y, logits, loss

    # The last step is the steady one: the first allocates the optimizer's state
    # and the gradients, and says nothing about what a step costs thereafter.
    last = readings[-1]
    weights = _nbytes(model.parameters())
    grads = _nbytes(p.grad for p in model.parameters())
    state = _nbytes(
        v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v) and v.dim() > 0
    )
    report = {
        "ok": True,
        "recipe": recipe,
        "batch": batch,
        "seq": seq,
        "steps": steps,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "params": params,
        # At rest, each counted from the tensors themselves.
        "weights_bytes": weights,
        "grads_bytes": grads,
        "optimizer_bytes": state,
        # What the allocator holds at rest beyond those three: small buffers,
        # the step counters, and whatever the allocator rounds up to.
        "other_resting_bytes": last["resting"] - weights - grads - state,
        # In flight.
        "saved_bytes": last["saved"],
        "peak_bytes": last["peak"],
        "readings": readings,
    }
    say(
        f"{params:,} parameters: at rest {last['resting'] / 2**20:,.1f} MiB, "
        f"saved for backward {last['saved'] / 2**20:,.1f} MiB, peak {last['peak'] / 2**20:,.1f} MiB"
    )
    return report
