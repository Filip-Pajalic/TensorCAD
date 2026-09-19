"""``tensorcad-runtime data prepare`` — a small token corpus for smoke training.

Preferred path: stream a real corpus with ``datasets`` and tokenize it with
``tiktoken``'s gpt2 encoding (vocab 50257). When there is no network access or
either library is missing, fall back to a deterministic synthetic corpus built
from a seeded Markov chain over a small n-gram vocabulary — enough structure
that a tiny model's loss visibly drops, so ``smoke-train`` always has something
to train on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["prepare_data", "load_corpus", "synthetic_corpus", "DATASETS"]

DATASETS: dict[str, dict[str, Any]] = {
    "tinystories": {
        "repo": "roneneldan/TinyStories",
        "split": "train",
        "text_field": "text",
        "config": None,
    },
    "fineweb-edu": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "text_field": "text",
        "config": "sample-10BT",
    },
}

# The synthetic corpus stays well inside every preset's vocabulary.
SYNTHETIC_VOCAB = 1024
SYNTHETIC_WORDS = 512
SYNTHETIC_SEED = 1234


def _dtype_for(vocab_size: int) -> str:
    return "uint16" if vocab_size <= np.iinfo(np.uint16).max else "uint32"


def synthetic_corpus(
    tokens: int,
    vocab_size: int = SYNTHETIC_VOCAB,
    seed: int = SYNTHETIC_SEED,
) -> np.ndarray:
    """A deterministic corpus with learnable structure.

    Builds a fixed dictionary of multi-token "words", then walks a sparse Markov
    chain over them. Repeated n-grams and a skewed successor distribution give a
    language model real signal to fit, and the whole thing is vectorized so 20M
    tokens take well under a second.
    """
    rng = np.random.default_rng(seed)
    tokens = max(int(tokens), 64)

    # Fixed dictionary of words, each 2-6 tokens drawn from [1, vocab_size).
    lengths = rng.integers(2, 7, size=SYNTHETIC_WORDS)
    flat = rng.integers(1, vocab_size, size=int(lengths.sum()), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    # Each word has 4 likely successors with a skewed (Zipf-ish) distribution,
    # so bigrams and trigrams of words recur throughout the corpus.
    n_succ = 4
    succ = rng.integers(0, SYNTHETIC_WORDS, size=(SYNTHETIC_WORDS, n_succ))
    weights = np.array([0.55, 0.25, 0.13, 0.07])
    cum = np.cumsum(weights)

    # Walk many chains in parallel so the sequential step count stays small.
    n_chains = 128
    avg_len = float(lengths.mean()) + 1.0  # +1 for the separator token
    steps = int(tokens / avg_len / n_chains) + 2
    current = rng.integers(0, SYNTHETIC_WORDS, size=n_chains)
    walk = np.empty((steps, n_chains), dtype=np.int64)
    for i in range(steps):
        walk[i] = current
        draw = rng.random(n_chains)
        pick = np.searchsorted(cum, draw)
        np.clip(pick, 0, n_succ - 1, out=pick)
        current = succ[current, pick]
    word_ids = walk.T.reshape(-1)  # chain-major, so each chain stays contiguous

    # Ragged gather: expand the word-id walk into a flat token stream.
    word_lens = lengths[word_ids]
    ends = np.cumsum(word_lens)
    out_starts = ends - word_lens
    total = int(ends[-1])
    offsets = np.arange(total) - np.repeat(out_starts, word_lens)
    stream = flat[np.repeat(starts[word_ids], word_lens) + offsets]

    # A separator token (0) between words gives the model an easy first win.
    with_sep = np.zeros(total + len(word_ids), dtype=np.int64)
    sep_positions = ends + np.arange(len(word_ids))
    mask = np.ones(len(with_sep), dtype=bool)
    mask[sep_positions] = False
    with_sep[mask] = stream

    if len(with_sep) < tokens:
        reps = int(np.ceil(tokens / len(with_sep)))
        with_sep = np.tile(with_sep, reps)
    return with_sep[:tokens].astype(np.uint16 if vocab_size <= 65535 else np.uint32)


def _download_and_tokenize(dataset: str, tokens: int, progress) -> tuple[np.ndarray, dict[str, Any]]:
    """Stream a real corpus and tokenize it. Raises on any failure."""
    import tiktoken  # noqa: PLC0415 - optional dependency
    from datasets import load_dataset  # noqa: PLC0415

    spec = DATASETS[dataset]
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256

    progress(f"streaming {spec['repo']} ({dataset})")
    stream = load_dataset(
        spec["repo"],
        name=spec["config"],
        split=spec["split"],
        streaming=True,
    )

    chunks: list[np.ndarray] = []
    count = 0
    docs = 0
    records = iter(stream)
    try:
        for record in records:
            text = record.get(spec["text_field"]) or ""
            if not text:
                continue
            ids = enc.encode_ordinary(text)
            ids.append(eot)
            chunks.append(np.asarray(ids, dtype=np.uint16))
            count += len(ids)
            docs += 1
            if docs % 2000 == 0:
                progress(f"  {docs:,} documents, {count:,} tokens")
            if count >= tokens:
                break
    finally:
        # Close the HTTP stream explicitly; otherwise the reader thread logs
        # socket errors while the interpreter is already shutting down.
        close = getattr(records, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001, S110
                pass

    if not chunks:
        raise RuntimeError("the dataset stream yielded no documents")

    data = np.concatenate(chunks)[:tokens]
    meta = {
        "source": spec["repo"],
        "dataset": dataset,
        "tokenizer": "tiktoken:gpt2",
        "vocab_size": enc.n_vocab,
        "documents": docs,
    }
    return data, meta


def prepare_data(
    dataset: str = "tinystories",
    tokens: int = 20_000_000,
    out: str | os.PathLike[str] = "python/data",
    force_synthetic: bool = False,
    progress=None,
) -> dict[str, Any]:
    """Write ``<out>/<dataset>.bin`` plus a JSON manifest. Returns the manifest."""

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose one of {sorted(DATASETS)}")

    out_dir = Path(out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    data: np.ndarray | None = None
    meta: dict[str, Any] = {}

    if not force_synthetic:
        try:
            data, meta = _download_and_tokenize(dataset, tokens, say)
        except Exception as exc:  # noqa: BLE001 - offline is an expected outcome
            warnings.append(
                f"download failed ({type(exc).__name__}: {str(exc)[:200]}); "
                "falling back to a deterministic synthetic corpus"
            )
            say(warnings[-1])
            data = None

    if data is None:
        say(f"generating a synthetic corpus of {tokens:,} tokens")
        data = synthetic_corpus(tokens)
        meta = {
            "source": "synthetic",
            "dataset": dataset,
            "tokenizer": "synthetic-ngram",
            "vocab_size": SYNTHETIC_VOCAB,
            "seed": SYNTHETIC_SEED,
        }

    dtype = _dtype_for(int(meta["vocab_size"]))
    data = data.astype(dtype, copy=False)

    stem = "synthetic" if meta["source"] == "synthetic" else dataset
    bin_path = out_dir / f"{stem}.bin"
    manifest_path = out_dir / f"{stem}.json"

    say(f"writing {len(data):,} tokens to {bin_path}")
    data.tofile(bin_path)

    manifest = {
        "vocab_size": int(meta["vocab_size"]),
        "tokens": int(len(data)),
        "dtype": dtype,
        "source": meta["source"],
        "dataset": meta.get("dataset", dataset),
        "tokenizer": meta.get("tokenizer"),
        "bin": bin_path.name,
        "path": str(bin_path),
        "bytes": int(bin_path.stat().st_size),
    }
    if "documents" in meta:
        manifest["documents"] = meta["documents"]
    if "seed" in meta:
        manifest["seed"] = meta["seed"]
    if warnings:
        manifest["warnings"] = warnings

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def load_corpus(
    data_dir: str | os.PathLike[str] | None,
    min_tokens: int,
    progress=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a prepared corpus, or synthesize one in memory if none exists."""

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    candidates: list[Path] = []
    if data_dir is not None:
        path = Path(data_dir)
        if path.is_file() and path.suffix == ".json":
            candidates = [path]
        elif path.is_dir():
            # Prefer a real corpus over the synthetic one.
            real = sorted(p for p in path.glob("*.json") if p.stem != "synthetic")
            candidates = real + sorted(path.glob("synthetic.json"))

    for manifest_path in candidates:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            bin_path = Path(manifest.get("path") or (manifest_path.parent / manifest["bin"]))
            if not bin_path.is_file():
                bin_path = manifest_path.parent / manifest["bin"]
            data = np.fromfile(bin_path, dtype=np.dtype(manifest["dtype"]))
        except Exception as exc:  # noqa: BLE001
            say(f"ignoring {manifest_path.name}: {type(exc).__name__}: {exc}")
            continue
        if len(data) >= min_tokens:
            say(f"using {bin_path} ({len(data):,} tokens, source={manifest.get('source')})")
            manifest["tokens"] = int(len(data))
            return data, manifest
        say(f"{manifest_path.name} has only {len(data):,} tokens, need {min_tokens:,}")

    say(f"no prepared corpus found; synthesizing {max(min_tokens, 2_000_000):,} tokens in memory")
    tokens = max(min_tokens, 2_000_000)
    data = synthetic_corpus(tokens)
    manifest = {
        "vocab_size": SYNTHETIC_VOCAB,
        "tokens": int(len(data)),
        "dtype": _dtype_for(SYNTHETIC_VOCAB),
        "source": "synthetic",
        "tokenizer": "synthetic-ngram",
        "seed": SYNTHETIC_SEED,
        "in_memory": True,
    }
    return data, manifest
