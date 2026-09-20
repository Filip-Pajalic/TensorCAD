"""``tensorcad-runtime smoke-train`` — train a generated model briefly and log it.

bf16 autocast on CUDA, AdamW, cosine schedule with warmup, gradient clipping at
1.0, fixed seed. One JSON object per logged step goes to the ``.jsonl`` run file;
a single summary object goes to stdout.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_corpus
from .loader import (
    ModelFile,
    design_hash,
    find_model_class,
    load_module,
    read_design,
    vocab_from_design,
    vocab_from_model,
)

__all__ = ["smoke_train"]


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, getattr(torch, "OutOfMemoryError", ())):
        return True
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _lr_at(step: int, total: int, warmup: int, peak: float, floor_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay to ``floor_ratio * peak``."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    floor = peak * floor_ratio
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def apply_gpt2_init(model, std: float = 0.02) -> dict[str, Any]:
    """GPT-2 style initialization: N(0, 0.02), zero biases, scaled residuals.

    Generated models carry no initialization of their own, so they inherit
    PyTorch's defaults — and ``nn.Embedding`` defaults to ``N(0, 1)``, which puts
    the initial cross-entropy in the hundreds instead of near ``ln(vocab)``.
    Every reference trainer (nanoGPT, GPT-2, Llama) re-initializes, so the smoke
    trainer does too.
    """
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # Residual output projections get std / sqrt(2 * n_layers), as in GPT-2.
    residual_suffixes = ("o_proj", "down", "out_proj", "proj")
    residual = [
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in residual_suffixes
    ]
    n_layers = max(1, len(residual) // 2)
    scale = std / math.sqrt(2 * n_layers)
    for module in residual:
        nn.init.normal_(module.weight, mean=0.0, std=scale)
    return {"scheme": "gpt2", "std": std, "residual_std": scale, "residual_layers": n_layers}


def _peak_memory(device: str) -> int:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def smoke_train(
    model_path: str | os.PathLike[str],
    steps: int = 50,
    seq: int = 256,
    batch: int = 8,
    lr: float = 3e-4,
    out: str | os.PathLike[str] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    class_name: str | None = None,
    seed: int = 1337,
    log_every: int = 1,
    device: str | None = None,
    compile_model: bool = False,
    weight_decay: float = 0.1,
    init: str = "gpt2",
    progress=None,
) -> dict[str, Any]:
    """Train a generated model for ``steps`` optimizer steps and summarize."""
    import torch
    import torch.nn.functional as F

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    warnings: list[str] = []
    info = ModelFile(model_path)
    design = read_design(info)

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    rng = np.random.default_rng(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        warnings.append("CUDA requested but not available; falling back to CPU")
        device = "cpu"
    torch.manual_seed(seed)

    say(f"importing {info.path}")
    module = load_module(info.path)
    cls = find_model_class(module, info, class_name)

    say(f"building {cls.__name__} on {device}")
    with torch.device(device):
        model = cls()
    model.train()
    params = sum(p.numel() for p in model.parameters())
    say(f"{params:,} parameters")

    init_info: dict[str, Any] = {"scheme": "default"}
    if init == "gpt2":
        init_info = apply_gpt2_init(model)
        say(
            f"re-initialized weights N(0, {init_info['std']}) "
            f"(generated models carry no init; torch defaults nn.Embedding to N(0, 1))"
        )
    elif init != "default":
        raise ValueError(f"unknown init scheme {init!r}; use 'gpt2' or 'default'")
    else:
        warnings.append(
            "using PyTorch's default initialization; nn.Embedding defaults to N(0, 1), "
            "so the initial loss will be far above ln(vocab)"
        )

    model_vocab = vocab_from_design(design) or vocab_from_model(model) or 32000

    if compile_model:
        try:
            model = torch.compile(model)
            say("torch.compile enabled")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"torch.compile failed: {type(exc).__name__}: {exc}")

    # -- data --------------------------------------------------------------
    if data_dir is None:
        default = Path("python/data")
        data_dir = default if default.exists() else None
    needed = (seq + 1) * batch * 4
    data, manifest = load_corpus(data_dir, needed, progress=say)
    corpus_vocab = int(manifest.get("vocab_size", int(data.max()) + 1))
    if corpus_vocab > model_vocab:
        warnings.append(
            f"corpus vocab {corpus_vocab} exceeds the model's {model_vocab}; "
            "token ids are taken modulo the model vocab"
        )
        narrow = np.uint16 if model_vocab <= np.iinfo(np.uint16).max else np.int64
        data = (data.astype(np.int64) % model_vocab).astype(narrow)
    data = np.ascontiguousarray(data)
    max_start = len(data) - seq - 1
    if max_start < 1:
        raise ValueError(f"corpus too small: {len(data)} tokens for seq={seq}")

    def get_batch(size: int):
        starts = rng.integers(0, max_start, size=size)
        idx = starts[:, None] + np.arange(seq + 1)[None, :]
        chunk = data[idx].astype(np.int64)
        x = torch.from_numpy(np.ascontiguousarray(chunk[:, :-1]))
        y = torch.from_numpy(np.ascontiguousarray(chunk[:, 1:]))
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    # -- optimizer ---------------------------------------------------------
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    if device == "cuda" and not use_bf16:
        warnings.append("bf16 is not supported on this GPU; training in fp32")
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
    )

    warmup = max(1, min(steps // 10, 100))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # -- run file ----------------------------------------------------------
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = Path("runs") / f"{info.path.parent.name or info.path.stem}-{stamp}.jsonl"
    out_path = Path(out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_file = out_path.open("w", encoding="utf-8")

    say(
        f"training {steps} steps at batch={batch}, seq={seq}, lr={lr:g} "
        f"({'bf16 autocast' if use_bf16 else 'fp32'})"
    )

    losses: list[float] = []
    records: list[dict[str, Any]] = []
    tokens_seen = 0
    current_batch = batch
    oom_events = 0
    started = time.perf_counter()
    window_start = started
    window_tokens = 0
    first_step_end = None
    tokens_after_first = 0

    try:
        for step in range(steps):
            step_lr = _lr_at(step, steps, warmup, lr)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            loss_value = None
            for attempt in range(6):
                try:
                    x, y = get_batch(current_batch)
                    with autocast_ctx:
                        logits = model(x)
                        if isinstance(logits, (tuple, list)):
                            logits = logits[0]
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    loss_value = float(loss.detach().item())
                    break
                except Exception as exc:  # noqa: BLE001
                    if not _is_oom(exc) or current_batch <= 1:
                        raise
                    oom_events += 1
                    optimizer.zero_grad(set_to_none=True)
                    new_batch = max(1, current_batch // 2)
                    message = (
                        f"CUDA OOM at batch={current_batch} (step {step}); "
                        f"retrying at batch={new_batch}"
                    )
                    warnings.append(message)
                    say(message)
                    current_batch = new_batch
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        # Otherwise every later reading reports the high-water
                        # mark of the batch that did not fit.
                        torch.cuda.reset_peak_memory_stats()
            if loss_value is None:
                raise RuntimeError("out of memory even at batch=1")

            tokens_seen += current_batch * seq
            window_tokens += current_batch * seq
            losses.append(loss_value)
            if step == 0:
                if device == "cuda":
                    torch.cuda.synchronize()
                first_step_end = time.perf_counter()
                tokens_after_first = tokens_seen

            if (step + 1) % max(1, log_every) == 0 or step == steps - 1:
                if device == "cuda":
                    torch.cuda.synchronize()
                now = time.perf_counter()
                window = max(now - window_start, 1e-9)
                record = {
                    "step": step + 1,
                    "loss": loss_value,
                    "lr": step_lr,
                    "tokens": tokens_seen,
                    "tokens_per_second": window_tokens / window,
                    "peak_memory_bytes": _peak_memory(device),
                    "seconds": now - started,
                    "batch": current_batch,
                    "seq": seq,
                }
                run_file.write(json.dumps(record) + "\n")
                run_file.flush()
                records.append(record)
                say(
                    f"step {step + 1}/{steps} loss {loss_value:.4f} "
                    f"{record['tokens_per_second']:,.0f} tok/s"
                )
                window_start = now
                window_tokens = 0
    finally:
        run_file.close()

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    # Exclude the first step (kernel autotuning, allocator warmup) from the
    # headline throughput number when there is more than one step.
    if first_step_end is not None and steps > 1:
        steady_seconds = max(time.perf_counter() - first_step_end, 1e-9)
        throughput = max(tokens_seen - tokens_after_first, 0) / steady_seconds
    else:
        throughput = tokens_seen / max(elapsed, 1e-9)

    summary = {
        "ok": True,
        "model_path": str(info.path),
        "class_name": cls.__name__,
        "params": params,
        "steps": len(losses),
        "final_loss": losses[-1] if losses else None,
        "best_loss": min(losses) if losses else None,
        "initial_loss": losses[0] if losses else None,
        "tokens": tokens_seen,
        "tokens_per_second": throughput,
        "peak_memory_bytes": _peak_memory(device),
        "seconds": elapsed,
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "cpu"
        ),
        "dtype": "bfloat16" if use_bf16 else "float32",
        "init": init_info,
        "batch": current_batch,
        "requested_batch": batch,
        "seq": seq,
        "lr": lr,
        "seed": seed,
        "oom_events": oom_events,
        "design_hash": design_hash(info),
        "data_source": manifest.get("source"),
        "data_tokens": manifest.get("tokens"),
        "run_file": str(out_path),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "log": records,
        "warnings": warnings,
        "torch_version": torch.__version__,
    }

    # The run record, beside the step log it summarizes.
    #
    # The `.jsonl` is written as the run goes, so it survives an interrupted
    # one; this is the whole thing in one object, which is what the editor's
    # Runs panel opens and what makes a run comparable with another. Writing it
    # here rather than leaving it on stdout is the difference between a run you
    # can look at later and a run you had to be watching.
    record_path = out_path.with_suffix(".json")
    record_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    summary["record_file"] = str(record_path)
    return summary
