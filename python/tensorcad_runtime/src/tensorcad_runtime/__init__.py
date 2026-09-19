"""tensorcad-runtime — PyTorch verification and smoke training for TensorCAD designs.

The CLI is the contract with the TypeScript side (one JSON object on stdout,
progress on stderr), but everything is importable as a library too::

    from tensorcad_runtime import verify_model
    report = verify_model("out/gpt2-small/model.py")

torch is imported lazily so that ``import tensorcad_runtime`` works (and reports a
clean error) on machines that have not installed it yet.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "verify_model",
    "smoke_train",
    "prepare_data",
    "synthetic_corpus",
    "load_corpus",
    "load_module",
    "find_model_class",
    "ModelFile",
]

# Cheap, torch-free imports.
from .data import load_corpus, prepare_data, synthetic_corpus  # noqa: E402
from .loader import ModelFile, find_model_class, load_module  # noqa: E402


def __getattr__(name: str) -> Any:
    """Defer the torch-dependent entry points until they are actually used."""
    if name == "verify_model":
        from .verify import verify_model

        return verify_model
    if name == "smoke_train":
        from .smoke_train import smoke_train

        return smoke_train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
