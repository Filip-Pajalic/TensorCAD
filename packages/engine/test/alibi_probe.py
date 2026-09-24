"""Hold a generated model's ALiBi against the bias BLOOM builds.

`build_alibi_tensor` below is transcribed from Hugging Face's BLOOM
(transformers/models/bloom/modeling_bloom.py), not written from the engine's
expression, so the two are independent statements of the same bias. BLOOM
adds slope × key position; the design writes -slope × distance. Along one
query's row those differ by slope × query position, a constant the softmax
cannot see — so the check is that the difference is constant along every row,
and that the attention weights come out the same.
"""

import importlib.util
import json
import math
import sys

import torch

spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
heads = int(sys.argv[2])


def build_alibi_tensor(attention_mask, num_heads, dtype):
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32)
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


T = 64
theirs = build_alibi_tensor(torch.ones(1, T), heads, torch.float32).view(heads, 1, T).expand(heads, T, T)
h = torch.arange(heads).view(-1, 1, 1)
q_idx = torch.arange(T).view(1, -1, 1)
kv_idx = torch.arange(T).view(1, 1, -1)
ours = gen.score_mod_1(torch.zeros(heads, T, T), 0, h, q_idx, kv_idx)

causal = kv_idx <= q_idx
difference = (ours - theirs).masked_fill(~causal, float("nan"))
spread = (difference.nan_to_num(float("-inf")).amax(-1) - difference.nan_to_num(float("inf")).amin(-1))

torch.manual_seed(0)
scores = torch.randn(heads, T, T)
weights = lambda bias: torch.softmax((scores + bias).masked_fill(~causal, float("-inf")), dim=-1)
print(json.dumps({
    "row_spread": spread.max().item(),
    "weights": (weights(ours) - weights(theirs)).abs().max().item(),
}))
