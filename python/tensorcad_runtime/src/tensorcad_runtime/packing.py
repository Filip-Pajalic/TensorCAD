"""Made-up packings, for a model that keeps documents apart.

A design whose attention reads each position's document takes the documents
as an input beside the tokens. Verifying one needs a batch of them, and this
makes one the way a pretraining pipeline does: documents laid end to end, their
lengths gamma-distributed with a given mean and coefficient of variation, and
each row a window cut out of that stream wherever it happens to fall — so a row
usually begins and ends inside a document.

The ids count documents from zero within each row. Only equality between two
positions of one row means anything.
"""

from __future__ import annotations

from typing import Any


def draw_documents(batch: int, seq: int, mean: float, spread: float = 1.0, seed: int = 0) -> Any:
    """A (batch, seq) tensor of document ids, one row per window of a stream."""
    import numpy as np
    import torch

    if not mean >= 1:
        raise ValueError(f"documents have to be at least one token long on average, not {mean}")
    if not spread >= 0:
        raise ValueError(f"the spread is a coefficient of variation, zero or more, not {spread}")
    rng = np.random.default_rng(seed)
    need = max(2 * batch * seq, 64 * mean)
    chunks: list[Any] = []
    total = 0.0
    while total < need:
        count = int(np.ceil(need / mean)) + 1
        if spread == 0:
            lengths = np.full(count, mean)
        else:
            lengths = rng.gamma(1 / spread**2, mean * spread**2, size=count)
        lengths = np.maximum(1, np.round(lengths))
        chunks.append(lengths)
        total += float(lengths.sum())
    lengths = np.concatenate(chunks)
    begins = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    rows = []
    for _ in range(batch):
        start = int(rng.integers(0, int(total) - seq + 1))
        doc = np.searchsorted(begins, start + np.arange(seq), side="right") - 1
        rows.append(doc - doc[0])
    return torch.tensor(np.stack(rows), dtype=torch.long)


def several_inputs(design: dict[str, Any] | None) -> list[str]:
    """The names of a design's inputs, when it has more than one."""
    if not design:
        return []
    nodes = (design.get("graph") or {}).get("nodes") or []
    names = [str(n.get("id")) for n in nodes if isinstance(n, dict) and n.get("type") == "input"]
    return names if len(names) > 1 else []
