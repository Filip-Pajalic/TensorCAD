"""Hold a generated DiffAttention against the reference implementation.

`MultiheadDiffAttn` below is transcribed from microsoft/unilm's
Diff-Transformer/multihead_diffattn.py, with the parts that need a GPU
library replaced by what they compute: flash-attn's causal attention by the
explicit mask the reference builds for itself, apex's RMSNorm by PyTorch's,
and no rotary, which is the design's to leave out. It materialises both score
matrices and subtracts them; the generated module runs two fused attentions
over shared values and subtracts their outputs. By linearity those are the
same numbers, and this is where that is checked rather than asserted.

The reference packs a pair of query heads into one projection, interleaved
as heads 2i and 2i+1; the design draws them as two projections. Copying the
weights across is that reshuffle and nothing else.
"""

import importlib.util
import json
import math
import sys

import torch
import torch.nn.functional as F
from torch import nn

spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
depth = int(sys.argv[2])


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(self, embed_dim, depth, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.n_rep = 1
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.subln = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q * self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_mask = torch.triu(torch.zeros([tgt_len, src_len]).float().fill_(float("-inf")).type_as(attn_weights), 1)
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        return self.out_proj(attn)


torch.manual_seed(0)
ours = gen.DiffAttention()
heads = ours.q1_proj.out_features // ours.combine.shape[1]
d = ours.combine.shape[1]
embed = ours.q1_proj.in_features
ref = MultiheadDiffAttn(embed, depth, heads)
with torch.no_grad():
    # The reference's head 2i is the design's q1 head i, and 2i+1 its q2.
    for name in ("q", "k"):
        w = getattr(ref, name + "_proj").weight.view(heads, 2, d, embed)
        w[:, 0] = getattr(ours, name + "1_proj").weight.view(heads, d, embed)
        w[:, 1] = getattr(ours, name + "2_proj").weight.view(heads, d, embed)
    ref.v_proj.weight.copy_(ours.v_proj.weight)
    ref.out_proj.weight.copy_(ours.o_proj.weight)
    ref.lambda_q1.copy_(ours.combine[0])
    ref.lambda_k1.copy_(ours.combine[1])
    ref.lambda_q2.copy_(ours.combine[2])
    ref.lambda_k2.copy_(ours.combine[3])
    ref.subln.weight.copy_(torch.randn(2 * d) * 0.5 + 1)
    ours.subln.weight.copy_(ref.subln.weight)

x = torch.randn(2, 24, embed)
a, b = ours(x), ref(x)
print(json.dumps({
    "heads": heads,
    "head_dim": d,
    "lambda_init": ref.lambda_init,
    "max_abs": (a - b).abs().max().item(),
    "scale": b.abs().max().item(),
    "params_ours": sum(p.numel() for p in ours.parameters()),
    "params_reference": sum(p.numel() for p in ref.parameters()),
}))
