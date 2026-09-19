"""Loading generated ``model.py`` files and the design JSON that sits next to them.

Everything in here is torch-optional: only :func:`find_model_class` needs torch,
so the metadata helpers can be unit-tested (and imported by the CLI for error
reporting) on a machine with no PyTorch.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = [
    "HELPER_CLASSES",
    "ModelFile",
    "load_module",
    "find_model_class",
    "read_design",
    "design_hash",
    "resolve_symbols",
    "vocab_from_design",
    "vocab_from_model",
]

# Classes the PyTorch emitter produces as building blocks. The model class is
# never one of these, so they are skipped by the "last class defined wins"
# heuristic.
HELPER_CLASSES = frozenset(
    {
        "RotaryEmbedding",
        "GqaAttention",
        "MhaAttention",
        "MqaAttention",
        "MlaAttention",
        "SlidingWindowAttention",
        "GatedMlp",
        "DenseMlp",
        "MoeLayer",
        "MoeMlp",
        "Expert",
        "Router",
        "TransformerBlock",
        "Block",
        "Layer",
        "Mamba2Block",
        "GatedDeltaNetBlock",
        "MtpHead",
        "RMSNorm",
        "LayerNorm",
    }
)


class ModelFile:
    """A generated ``model.py`` plus whatever metadata we can read off disk."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"no such model file: {self.path}")
        self.source = self.path.read_text(encoding="utf-8")
        self._tree = ast.parse(self.source, filename=str(self.path))

    # -- metadata parsed out of the source, no import required -------------

    @property
    def class_names(self) -> list[str]:
        """Top-level class names in definition order."""
        return [n.name for n in self._tree.body if isinstance(n, ast.ClassDef)]

    @property
    def main_block(self) -> list[ast.stmt]:
        """Body of ``if __name__ == "__main__":``, or ``[]``."""
        for node in self._tree.body:
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            ):
                return node.body
        return []

    @property
    def expected_params(self) -> int | None:
        """The ``expected = N`` assignment from the ``__main__`` block."""
        for node in self.main_block:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "expected":
                        value = node.value.value
                        if isinstance(value, int):
                            return value
        # Fall back to the "Parameters: 1,234" line in the module docstring.
        m = re.search(r"^Parameters:\s*([\d,_]+)\s*$", self.source, re.MULTILINE)
        if m:
            return int(m.group(1).replace(",", "").replace("_", ""))
        return None

    @property
    def main_class_hint(self) -> str | None:
        """Class instantiated as ``model = Something()`` in the ``__main__`` block."""
        for node in ast.walk(ast.Module(body=self.main_block, type_ignores=[])):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "model" not in targets:
                continue
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                return call.func.id
        return None

    @property
    def design_path(self) -> Path | None:
        """The design document emitted next to the model, if there is one."""
        folder = self.path.parent
        candidates = [
            folder / "design.tensorcad.json",
            folder / f"{self.path.stem}.tensorcad.json",
            folder / "design.json",
        ]
        candidates.extend(sorted(folder.glob("*.tensorcad.json")))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None


def load_module(path: str | os.PathLike[str], name: str | None = None) -> ModuleType:
    """Import a generated model file from an arbitrary path.

    The module is registered in ``sys.modules`` before execution so that
    dataclasses, ``typing.get_type_hints`` and pickling behave normally.
    """
    path = Path(path).resolve()
    name = name or f"tensorcad_generated_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Let the generated file import siblings if it ever wants to.
    folder = str(path.parent)
    added = folder not in sys.path
    if added:
        sys.path.insert(0, folder)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        if added:
            try:
                sys.path.remove(folder)
            except ValueError:
                pass
    return module


def find_model_class(module: ModuleType, info: ModelFile, class_name: str | None = None):
    """Locate the top-level model class inside an imported generated module.

    Resolution order:

    1. an explicit ``--class-name``;
    2. the class the file's own ``__main__`` block instantiates;
    3. the last ``nn.Module`` subclass defined in the file that is not a known
       helper (falling back to the last one overall if every class is a helper).
    """
    import torch.nn as nn

    def get(name: str):
        cls = getattr(module, name, None)
        if cls is None:
            raise AttributeError(f"class {name!r} not found in {info.path}")
        if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
            raise TypeError(f"{name!r} in {info.path} is not an nn.Module subclass")
        return cls

    if class_name:
        return get(class_name)

    hint = info.main_class_hint
    if hint and isinstance(getattr(module, hint, None), type):
        try:
            return get(hint)
        except TypeError:
            pass

    # Definition order, restricted to nn.Module subclasses actually defined here.
    defined = []
    for name in info.class_names:
        obj = getattr(module, name, None)
        if (
            isinstance(obj, type)
            and issubclass(obj, nn.Module)
            and getattr(obj, "__module__", None) == module.__name__
        ):
            defined.append((name, obj))
    if not defined:
        raise LookupError(f"no nn.Module subclass defined in {info.path}")

    for name, obj in reversed(defined):
        if name not in HELPER_CLASSES:
            return obj
    return defined[-1][1]


# -- design document ---------------------------------------------------------


def read_design(info: ModelFile) -> dict[str, Any] | None:
    """Parse the design JSON next to the model, if present."""
    path = info.design_path
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def design_hash(info: ModelFile) -> str | None:
    """sha256 of the design JSON bytes next to the model."""
    path = info.design_path
    if path is None:
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


_SAFE_EXPR = re.compile(r"^[0-9A-Za-z_+\-*/(). ]+$")


def resolve_symbols(design: dict[str, Any] | None) -> dict[str, float]:
    """Evaluate the design's symbol table to concrete numbers.

    Symbol values may be plain numbers or small arithmetic expressions over
    other symbols (``"4*D"``). Runtime symbols contribute their ``default``.
    """
    if not design:
        return {}
    raw = design.get("symbols") or {}
    if not isinstance(raw, dict):
        return {}

    pending: dict[str, Any] = {}
    for name, entry in raw.items():
        if isinstance(entry, (int, float)):
            pending[name] = entry
        elif isinstance(entry, dict):
            value = entry.get("value", entry.get("default"))
            if value is not None:
                pending[name] = value
        elif isinstance(entry, str):
            pending[name] = entry

    resolved: dict[str, float] = {k: v for k, v in pending.items() if isinstance(v, (int, float))}
    # Expressions may reference each other; a few passes settle any chain.
    for _ in range(len(pending) + 1):
        progressed = False
        for name, value in pending.items():
            if name in resolved or not isinstance(value, str):
                continue
            if not _SAFE_EXPR.match(value):
                continue
            try:
                out = eval(value, {"__builtins__": {}}, dict(resolved))  # noqa: S307
            except Exception:
                continue
            if isinstance(out, (int, float)):
                resolved[name] = out
                progressed = True
        if not progressed:
            break
    return resolved


def input_spec_from_design(
    design: dict[str, Any] | None, batch: int, seq: int
) -> tuple[list[int], str, list[str]] | None:
    """Dims and dtype of the model's input, from the design's own ``input`` block.

    Every design was a language model until a vision one arrived, so the harness
    assumed token ids: ``randint(0, vocab, (B, T))``. A JEPA takes ``B T P``
    floats, and guessing wrong fails inside the first Linear. The document
    already says which it is; this reads it instead.

    The atom names come back with the dims because they say which axis is which:
    a convnet's input is ``B C H W``, and marking axis 1 dynamic as though it
    were a sequence asks ``torch.export`` to vary the channel count.

    Returns None when the shape cannot be resolved, and the caller falls back to
    token ids.
    """
    if not design:
        return None
    node = _find_node(design.get("graph"), "input")
    if node is None:
        return None
    params = node.get("params") or {}
    pattern = str(params.get("shape") or "B T")
    dtype = str(params.get("dtype") or "int64")
    values = dict(resolve_symbols(design))
    values["B"] = batch
    values["T"] = seq
    dims: list[int] = []
    for atom in pattern.split():
        if atom in values:
            dims.append(int(values[atom]))
            continue
        try:
            dims.append(int(float(atom)))
        except ValueError:
            return None
    if not dims:
        return None
    return dims, dtype, pattern.split()


def _find_node(graph: Any, type_name: str) -> dict[str, Any] | None:
    """First node of a given type, anywhere in the nesting."""
    if not isinstance(graph, dict):
        return None
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("type") == type_name:
            return node
        found = _find_node(node.get("graph"), type_name)
        if found is not None:
            return found
    return None


def vocab_from_design(design: dict[str, Any] | None) -> int | None:
    """Read ``symbols.V`` out of a design document."""
    value = resolve_symbols(design).get("V")
    if isinstance(value, (int, float)) and value >= 1:
        return int(value)
    return None


def vocab_from_model(model) -> int | None:
    """Fall back to the ``out_features`` of the model's final ``nn.Linear``."""
    import torch.nn as nn

    last: int | None = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            last = mod.out_features
    if last is not None:
        return int(last)
    for mod in model.modules():
        if isinstance(mod, nn.Embedding):
            return int(mod.num_embeddings)
    return None
