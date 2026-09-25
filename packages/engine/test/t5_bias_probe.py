"""Hold a generated relative-position bias against T5's own.

`compute_bias` and `relative_position_bucket` are transcribed from Hugging
Face's T5Attention (transformers/models/t5/modeling_t5.py). The generated
model's score function, handed the same table, has to add exactly that bias to
every score: two-sided in the encoder, one-sided in the decoder.

Then the same score function is handed to PyTorch's own flex_attention,
uncompiled, and has to give what the model's eager fallback gives — and this
records whether a gradient reaches a table FlexAttention captured, which is
the question the proposal left for this phase.
"""

import importlib.util
import json
import math
import sys

import torch

spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec)
sys.modules["gen"] = gen
spec.loader.exec_module(gen)


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )
    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


def compute_bias(table, query_length, key_length, bidirectional):
    context_position = torch.arange(query_length, dtype=torch.long)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position
    bucket = relative_position_bucket(relative_position, bidirectional=bidirectional, num_buckets=table.shape[0])
    values = torch.nn.functional.embedding(bucket, table)  # (q, k, heads)
    return values.permute([2, 0, 1]).unsqueeze(0)  # (1, heads, q, k)


torch.manual_seed(0)
T, heads = 300, int(sys.argv[2])
out = {}
b = torch.arange(1).view(-1, 1, 1, 1)
h = torch.arange(heads).view(1, -1, 1, 1)
q_idx = torch.arange(T).view(1, 1, -1, 1)
kv_idx = torch.arange(T).view(1, 1, 1, -1)
for name, bidirectional in (("score_mod_1", True), ("score_mod_2", False)):
    table = torch.randn(32, heads)
    ours = getattr(gen, name)(table)(torch.zeros(1, heads, T, T), b, h, q_idx, kv_idx)
    theirs = compute_bias(table, T, T, bidirectional)
    out[name] = (ours - theirs).abs().max().item()

# The same function through FlexAttention's hands, and back.
from torch.nn.attention.flex_attention import flex_attention  # noqa: E402

q, k, v = (torch.randn(2, heads, 40, 16) for _ in range(3))
table = torch.randn(32, heads, requires_grad=True)
score = gen.score_mod_1(table)
eager = gen.expression_attention(q, k, v, score_mod=score)
flex = flex_attention(q, k, v, score_mod=score)
out["flex_vs_eager"] = (eager - flex).abs().max().item()
eager.sum().backward()
out["eager_grad"] = table.grad is not None and bool(table.grad.abs().sum() > 0)
table.grad = None
try:
    flex_attention(q, k, v, score_mod=gen.score_mod_1(table)).sum().backward()
    out["flex_grad"] = table.grad is not None and bool(table.grad.abs().sum() > 0)
except Exception as error:  # noqa: BLE001 - what it says is the finding
    out["flex_grad"] = f"{type(error).__name__}: {str(error)[:160]}"

print(json.dumps(out))
