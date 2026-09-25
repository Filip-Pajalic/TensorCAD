"""Hold a generated written-out attention against two other statements of it.

- At its first step, with both mixing matrices the identity, talking heads is
  plain attention: the generated block has to give what PyTorch's own fused
  scaled_dot_product_attention gives.
- With the matrices random, it has to give talking-heads attention as the
  paper writes it (Shazeer et al. 2020, section 4, in einsum form): logits
  mixed across heads, the softmax, the weights mixed across heads, then the
  values. Transcribed here rather than read from the generated file.

Called with the generated model.py and the design's head width, which the
block's score scale was written for.
"""

import importlib.util
import json
import sys

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

torch.manual_seed(0)
block = gen.EagerAttention()
heads = block.mix_logits.shape[0]
B, T, D = 2, 16, int(sys.argv[2])
q = torch.randn(B, heads, T, D)
k = torch.randn(B, heads, T, D)
v = torch.randn(B, heads, T, D)

out = {"heads": heads}
with torch.no_grad():
    out["identity_vs_sdpa"] = (block(q=q, k=k, v=v) - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max().item()

    block.mix_logits.copy_(torch.randn(heads, heads))
    block.mix_weights.copy_(torch.randn(heads, heads))
    P_l, P_w = block.mix_logits, block.mix_weights

    # Section 4: logits = einsum(Q, K) / sqrt(d); logits = einsum(logits, P_l);
    # weights = softmax(logits); weights = einsum(weights, P_w); O = einsum(weights, V).
    logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * D**-0.5
    logits = torch.einsum("bhqk,hg->bgqk", logits, P_l)
    logits = logits.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    weights = torch.softmax(logits, dim=-1)
    weights = torch.einsum("bhqk,hg->bgqk", weights, P_w)
    paper = torch.einsum("bhqk,bhkd->bhqd", weights, v)
    ours = block(q=q, k=k, v=v)
    out["mixed_vs_paper"] = (ours - paper).abs().max().item()
    out["scale"] = paper.abs().max().item()
    out["mixing_matters"] = (ours - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max().item()

print(json.dumps(out))
