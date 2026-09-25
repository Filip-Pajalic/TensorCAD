"""Hold a generated model that keeps packed documents apart to its word.

Four things, each against something the engine did not write:

- Isolation, in the eager form the model runs on a CPU: a whole forward pass,
  with one document's tokens changed, moves no other document's outputs at all.
- The fused form: PyTorch's own flex_attention, uncompiled, given the model's
  mask and a block mask create_block_mask built, computes what the eager form
  computes.
- The block mask is built once a batch: every layer handed the same documents
  shares one, the next batch's replaces it, and documents changed in place are
  noticed.
- What it costs: over many rows drawn by the runtime's own packing, the scores
  the mask keeps, counted one by one, and the blocks create_block_mask says a
  kernel computes, against the engine's two figures.

Arguments: <model.py> <T> <mean> <spread> <engine kept keys> <engine block keys>.
"""

import importlib.util
import inspect
import json
import sys

import torch

from tensorcad_runtime.packing import draw_documents  # noqa: E402

spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec)
sys.modules["gen"] = gen
spec.loader.exec_module(gen)

T, mean, spread = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
engine_kept, engine_blocks = float(sys.argv[5]), float(sys.argv[6])
out = {}
torch.manual_seed(0)

cls = next(
    c
    for c in vars(gen).values()
    if isinstance(c, type) and issubclass(c, torch.nn.Module) and c.__module__ == "gen" and "init_weights" in vars(c)
)
factory = next(
    f
    for name, f in vars(gen).items()
    if name.startswith("mask_mod_") and callable(f) and len(inspect.signature(f).parameters) == 1
)

# --- isolation, the whole model, eager ---------------------------------------
model = cls().init_weights().eval()
vocab = model.embed.num_embeddings
docs = draw_documents(2, T, mean, spread, seed=1)
tokens = torch.randint(0, vocab, (2, T))
row = docs[0]
target = int(row[T // 2])
inside = row == target
changed = tokens.clone()
changed[0, inside] = (changed[0, inside] + 1) % vocab
with torch.no_grad():
    a, b = model(tokens, docs), model(changed, docs)
out["other_documents_moved"] = float((a[0, ~inside] - b[0, ~inside]).abs().max())
out["other_rows_moved"] = float((a[1] - b[1]).abs().max())
out["own_document_moved"] = float((a[0, inside] - b[0, inside]).abs().max())

# --- the fused form ------------------------------------------------------------
from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa: E402

heads, dh = 4, 16
q, k, v = (torch.randn(2, heads, T, dh) for _ in range(3))
mask_mod = factory(docs)
eager = gen.expression_attention(q, k, v, mask_mod=mask_mod, mask_batch=True)
block_mask = create_block_mask(mask_mod, 2, None, T, T, device="cpu")
fused = flex_attention(q, k, v, block_mask=block_mask)
out["flex_vs_eager"] = float((eager - fused).abs().max())

# --- the block mask, once a batch ----------------------------------------------
built = []


def counting(mask_mod, batch, heads, q_len, kv_len, device=None):
    built.append(mask_mod)
    return object()


gen._BLOCK_MASKS.clear()
first = [gen._block_mask(counting, factory(docs), 2, None, T, T, "cpu") for _ in range(3)]
out["layers_share_one"] = len(built) == 1 and all(m is first[0] for m in first)
fresh = draw_documents(2, T, mean, spread, seed=2)
gen._block_mask(counting, factory(fresh), 2, None, T, T, "cpu")
out["next_batch_rebuilds"] = len(built) == 2
fresh[0, 0] += 0
gen._block_mask(counting, factory(fresh), 2, None, T, T, "cpu")
out["edited_documents_rebuild"] = len(built) == 3
out["one_kept_per_mask"] = len(gen._BLOCK_MASKS) == 1

# --- what it costs --------------------------------------------------------------
rows = 256
many = draw_documents(rows, T, mean, spread, seed=3)
mask_mod = factory(many)
b = torch.arange(rows).view(-1, 1, 1)
q_idx = torch.arange(T).view(1, -1, 1)
kv_idx = torch.arange(T).view(1, 1, -1)
kept = int(mask_mod(b, 0, q_idx, kv_idx).sum())
# The engine counts causal attention at T/2 keys a query, which is the kept
# scores over T + 1 rather than T: the same convention here.
out["kept_keys"] = kept / rows / (T + 1)
block_mask = create_block_mask(mask_mod, rows, None, T, T, device="cpu")
blocks = int(block_mask.kv_num_blocks.sum())
if getattr(block_mask, "full_kv_num_blocks", None) is not None:
    blocks += int(block_mask.full_kv_num_blocks.sum())
size = block_mask.BLOCK_SIZE[1] if isinstance(block_mask.BLOCK_SIZE, tuple) else block_mask.BLOCK_SIZE
per_row = -(-T // size)
out["block_size"] = size
out["block_keys"] = blocks * size / rows / per_row
out["kept_vs_engine"] = out["kept_keys"] / engine_kept - 1
out["blocks_vs_engine"] = out["block_keys"] / engine_blocks - 1

print(json.dumps(out))
