"""``tensorcad-runtime verify`` — instantiate a generated model and cross-check it.

Reports parameter counts (against the design's own ``expected``), a real forward
pass when the model is small enough to fit in RAM, forward FLOPs from
``FlopCounterMode``, and whether ``torch.export`` succeeds with dynamic batch and
sequence dimensions.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, TypeVar

from .loader import (
    ModelFile,
    design_hash,
    find_model_class,
    load_module,
    read_design,
    resolve_symbols,
    input_spec_from_design,
    vocab_from_design,
    vocab_from_model,
)

__all__ = ["verify_model", "DEFAULT_MAX_FORWARD_BYTES", "DEFAULT_PHASE_TIMEOUT"]

# A real forward pass materializes the weights in RAM. Above this, skip it.
DEFAULT_MAX_FORWARD_BYTES = 2 * 1024**3

# Tracing budget for FLOP counting and torch.export. A design with a Python-level
# loop over the sequence (the Mamba-2 SSD scan is a readable sequential
# reference) unrolls once per position per layer, so tracing can run for a very
# long time. Bound it rather than hang.
DEFAULT_PHASE_TIMEOUT = 300.0

T = TypeVar("T")


def _bounded(
    fn: Callable[[], T], seconds: float, label: str, warnings: list[str]
) -> tuple[T | None, bool]:
    """Run ``fn`` on a daemon thread and give up after ``seconds``.

    Tracing happens inside torch and cannot be interrupted, so a thread that
    overruns is abandoned; the process exits soon after. Returns
    ``(result, timed_out)`` — on timeout the result is ``None`` and a warning has
    been recorded.
    """
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=run, name=f"tensorcad-{label}", daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        warnings.append(
            f"{label} gave up after {seconds:.0f}s. Designs with a Python loop over the "
            f"sequence (such as the Mamba-2 scan) unroll while tracing; retry with a "
            f"smaller --seq, or skip the phase."
        )
        return None, True
    if "error" in box:
        raise box["error"]
    return box.get("value"), False


def _params_by_module(model) -> tuple[dict[str, int], list[list[str]]]:
    """Top-level module -> parameter count, plus the tied-parameter groups.

    Shared (tied) tensors are attributed to the first module that reports them,
    so the values sum to exactly ``sum(p.numel() for p in model.parameters())``.
    """
    groups: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(param), []).append(name)
    tied = [names for names in groups.values() if len(names) > 1]

    seen: set[int] = set()
    table: dict[str, int] = {}
    for name, child in model.named_children():
        total = 0
        for param in child.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            total += param.numel()
        table[name] = total

    direct = 0
    for _, param in model.named_parameters(recurse=False):
        if id(param) in seen:
            continue
        seen.add(id(param))
        direct += param.numel()
    if direct:
        table["_root"] = direct
    return table, tied


def _example_input(in_spec, vocab: int, batch: int, seq: int, *, device=None, concrete: bool = False):
    """The tensor the model is actually called with.

    ``in_spec`` comes from the design's own input block; without one this falls
    back to token ids, which is what every language preset wants.
    """
    import torch

    dims = [batch, seq]
    dtype = "int64"
    if in_spec:
        dims, dtype = in_spec[0], in_spec[1]

    if dtype in ("fp32", "bf16"):
        td = torch.float32 if dtype == "fp32" else torch.bfloat16
        if concrete:
            return torch.randn(*dims, dtype=td)
        return torch.zeros(*dims, dtype=td, device=device)
    if concrete:
        return torch.randint(0, max(vocab, 1), tuple(dims))
    return torch.zeros(*dims, dtype=torch.long, device=device)


def _count_flops(cls, meta_model, in_spec, vocab: int, batch: int, seq: int, warnings: list[str]):
    """Forward FLOPs via FlopCounterMode, on meta tensors, then fake tensors.

    ``meta_model`` is the instance already built by the caller; the fake-tensor
    fallback has to construct its own inside ``FakeTensorMode``.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    try:
        model = meta_model
        ids = _example_input(in_spec, vocab, batch, seq, device="meta")
        counter = FlopCounterMode(display=False)
        with counter:
            model(ids)
        total = counter.get_total_flops()
        if total:
            return int(total)
        warnings.append("FlopCounterMode returned 0 FLOPs on meta tensors; retrying on fake tensors")
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        warnings.append(f"FLOP counting on meta tensors failed: {type(exc).__name__}: {exc}")

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            model = cls()
            ids = _example_input(in_spec, vocab, batch, seq)
            counter = FlopCounterMode(display=False)
            with counter:
                model(ids)
            total = counter.get_total_flops()
        return int(total) if total else None
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"FLOP counting on fake tensors failed: {type(exc).__name__}: {exc}")
        return None


def _symbolic_shape(node) -> list[str] | None:
    val = getattr(node, "meta", {}).get("val", None)
    if val is None or not hasattr(val, "shape"):
        return None
    return [str(dim) for dim in val.shape]


def _try_export(meta_model, in_spec, vocab: int, batch: int, seq: int, seq_max: int, warnings: list[str]):
    """``torch.export`` with dynamic B and T. Returns (ok, shapes, error)."""
    import torch
    from torch.export import Dim, export

    model = meta_model
    model.eval()
    ids = _example_input(in_spec, vocab, batch, seq, device="meta")

    # Which axes are allowed to move is a property of the design, not an
    # assumption: `B T` has two, `B C H W` has one, and asking for a dynamic
    # channel count is how this used to fail on a convnet.
    atoms = in_spec[2] if in_spec else ["B", "T"]
    named: dict[int, Any] = {}
    auto: dict[int, Any] = {}
    for i, atom in enumerate(atoms):
        if atom == "B":
            named[i] = Dim("batch", min=1, max=8192)
            auto[i] = Dim.AUTO
        elif atom == "T" and seq_max >= 2 and seq >= 2:
            named[i] = Dim("seq", min=2, max=seq_max)
            auto[i] = Dim.AUTO

    attempts: list[tuple[str, Any]] = [
        ("Dim", {"ids": named}),
        ("Dim.AUTO", {"ids": auto}),
    ]

    last_error = None
    attempted: list[str] = []
    for label, dynamic_shapes in attempts:
        try:
            program = export(model, (ids,), dynamic_shapes=dynamic_shapes)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            attempted.append(f"{label}: {last_error[:300]}")
            continue

        outputs = [n for n in program.graph.nodes if n.op == "output"]
        shapes: dict[str, Any] = {"strategy": label, "input": _symbolic_shape_of_input(program)}
        if outputs:
            args = outputs[0].args[0]
            if not isinstance(args, (list, tuple)):
                args = [args]
            found = [_symbolic_shape(a) for a in args if hasattr(a, "meta")]
            found = [f for f in found if f]
            if found:
                shapes["output"] = found[0] if len(found) == 1 else found
        if attempted:
            # Export succeeded, just not on the first strategy. Worth recording,
            # not worth alarming about.
            shapes["fallback_from"] = attempted
        return True, shapes, None

    for note in attempted:
        warnings.append(f"torch.export failed with {note}")
    return False, None, last_error


def _symbolic_shape_of_input(program) -> list[str] | None:
    import torch

    for node in program.graph.nodes:
        if node.op != "placeholder":
            continue
        shape = _symbolic_shape(node)
        # Parameters arrive as placeholders too; the token-id input is the first
        # one with an integer dtype.
        val = getattr(node, "meta", {}).get("val", None)
        dtype = getattr(val, "dtype", None) if val is not None else None
        if shape and dtype in (torch.int64, torch.int32):
            return shape
    return None


def _run_forward(cls, in_spec, vocab: int, batch: int, seq: int, warnings: list[str], count_flops: bool):
    """A real CPU forward pass. Returns (status, shapes, flops).

    ``count_flops`` wraps the pass in ``FlopCounterMode``. That is the only way
    to get FLOPs for a design whose forward has a data-dependent shape (the MoE
    dispatch uses ``nonzero``, which has no meta or fake kernel).
    """
    import contextlib

    import torch

    try:
        model = cls()
        model.eval()
        ids = _example_input(in_spec, vocab, batch, seq, concrete=True)
        counter = None
        if count_flops:
            from torch.utils.flop_counter import FlopCounterMode

            counter = FlopCounterMode(display=False)
        with torch.no_grad(), (counter or contextlib.nullcontext()):
            out = model(ids)
        if isinstance(out, (tuple, list)):
            out = out[0]
        flops = int(counter.get_total_flops()) if counter is not None else None
        return (
            "ok",
            {
                "input": list(ids.shape),
                "logits": list(out.shape),
                "dtype": str(out.dtype).replace("torch.", ""),
            },
            flops or None,
        )
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        warnings.append(f"forward pass failed: {message[:400]}")
        return f"failed: {message[:200]}", None, None


def verify_model(
    model_path: str | os.PathLike[str],
    batch: int = 2,
    seq: int = 128,
    class_name: str | None = None,
    max_forward_bytes: int = DEFAULT_MAX_FORWARD_BYTES,
    skip_export: bool = False,
    skip_flops: bool = False,
    phase_timeout: float = DEFAULT_PHASE_TIMEOUT,
    progress=None,
) -> dict[str, Any]:
    """Verify a generated ``model.py``. Returns the JSON-ready report."""
    import torch

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    warnings: list[str] = []
    info = ModelFile(model_path)
    design = read_design(info)

    say(f"importing {info.path}")
    module = load_module(info.path)
    cls = find_model_class(module, info, class_name)
    say(f"model class: {cls.__name__}")

    say("instantiating on the meta device")
    with torch.device("meta"):
        model = cls()

    params = sum(p.numel() for p in model.parameters())
    table, tied = _params_by_module(model)
    buffers = sum(b.numel() for b in model.buffers())

    expected = info.expected_params
    matches = (expected is not None) and (expected == params)
    if expected is None:
        warnings.append("no `expected = N` assignment found in the file's __main__ block")
    elif not matches:
        warnings.append(f"parameter count differs: torch says {params}, design says {expected}")

    vocab = vocab_from_design(design)
    vocab_source = "design.symbols.V"
    in_spec = input_spec_from_design(design, batch, seq)
    if vocab is None:
        vocab = vocab_from_model(model)
        vocab_source = "final nn.Linear out_features"
    if vocab is None:
        vocab = 32000
        vocab_source = "fallback default"
        warnings.append("could not determine the vocabulary size; assuming 32000")

    report: dict[str, Any] = {
        "ok": True,
        "model_path": str(info.path),
        "class_name": cls.__name__,
        "design_path": str(info.design_path) if info.design_path else None,
        "design_hash": design_hash(info),
        "params": params,
        "params_by_module": table,
        "buffers": buffers,
        "expected": expected,
        "matches": matches,
        "tied_parameters": tied,
        "vocab_size": vocab,
        "vocab_size_source": vocab_source,
        "batch": batch,
        "seq": seq,
        "torch_version": torch.__version__,
    }

    # -- FLOPs on meta/fake tensors ----------------------------------------
    flops = None
    flops_source = None
    flops_timed_out = False
    if skip_flops:
        warnings.append("FLOP counting skipped by request")
    else:
        say("counting forward FLOPs")
        flops, flops_timed_out = _bounded(
            lambda: _count_flops(cls, model, in_spec, vocab, batch, seq, warnings),
            phase_timeout,
            "FLOP counting",
            warnings,
        )
        if flops is not None:
            flops_source = "meta/fake tensors"

    # -- forward pass ------------------------------------------------------
    approx_bytes = params * 4
    report["weight_bytes_fp32"] = approx_bytes
    if approx_bytes > max_forward_bytes:
        gib = approx_bytes / 1024**3
        report["forward"] = "skipped: too large"
        report["shapes"] = None
        warnings.append(
            f"forward pass skipped: {params:,} params would need about {gib:.1f} GiB in fp32"
        )
    else:
        # If the shape-only paths failed (a data-dependent forward, e.g. the MoE
        # dispatch), the real pass can still be counted.
        also_count = flops is None and not skip_flops
        say(f"running a CPU forward pass at batch={batch}, seq={seq}")
        status, shapes, forward_flops = _run_forward(cls, in_spec, vocab, batch, seq, warnings, also_count)
        report["forward"] = status
        report["shapes"] = shapes
        if forward_flops is not None:
            flops = forward_flops
            flops_source = "real CPU forward pass"

    report["flops"] = flops
    report["flops_source"] = flops_source
    if flops is not None:
        report["flops_per_token"] = flops / max(1, batch * seq)

    # -- export ------------------------------------------------------------
    if skip_export:
        report["export_ok"] = None
        report["export_shapes"] = None
        warnings.append("torch.export check skipped by request")
    elif flops_timed_out:
        # The abandoned FLOP thread is still tracing this same module. Exporting
        # now would run two traces over one model concurrently.
        report["export_ok"] = None
        report["export_shapes"] = None
        warnings.append(
            "torch.export skipped: the FLOP phase is still tracing this model after its "
            "timeout. Re-run with a smaller --seq, or with --no-flops."
        )
    else:
        symbols = resolve_symbols(design)
        seq_max = int(symbols.get("Tmax") or symbols.get("T") or 0) or 1 << 17
        seq_max = max(seq_max, seq)
        say("checking torch.export with dynamic batch and sequence dims")
        outcome, _ = _bounded(
            lambda: _try_export(model, in_spec, vocab, batch, seq, seq_max, warnings),
            phase_timeout,
            "torch.export",
            warnings,
        )
        export_ok, export_shapes, error = outcome if outcome is not None else (None, None, None)
        report["export_ok"] = export_ok
        report["export_shapes"] = export_shapes
        if error:
            report["export_error"] = error[:400]

    report["warnings"] = warnings
    return report
