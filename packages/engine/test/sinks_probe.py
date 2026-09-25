"""Hold a generated model's attention sinks against two other statements of them.

The generated `expression_attention` applies sinks the long way on a CPU: one
more column in each row's softmax. Two independent forms are compared with it:

- FlexAttention's, the way the helper applies them on CUDA: the kernel's
  output rescaled by sigmoid(lse - sink), from the log-sum-exp it returns.
  Run here uncompiled, so the formula is checked where a GPU is not needed.
- Hugging Face's gpt-oss attention, transcribed from `eager_attention_forward`
  in transformers/models/gpt_oss/modeling_gpt_oss.py: the sinks concatenated
  to the scores, the row maximum taken off, the softmax, the sink column
  dropped.

With the file's own window mask, grouped key-value heads, and a query that a
mask leaves nothing but its sink.
"""

import importlib.util
import json
import sys
import warnings

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec)
sys.modules["gen"] = gen
spec.loader.exec_module(gen)

from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa: E402


def hugging_face(q, k, v, sinks, keep, scale):
    heads = q.shape[1]
    k = k.repeat_interleave(heads // k.shape[1], dim=1)
    v = v.repeat_interleave(heads // v.shape[1], dim=1)
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale
    attn_weights = attn_weights.masked_fill(~keep, float("-inf"))
    sinks = sinks.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    return torch.matmul(scores, v)


torch.manual_seed(0)
out = {"cases": {}}
masks = sorted(n for n in dir(gen) if n.startswith("mask_mod_"))
T, H, KV, D = 160, 8, 2, 16
scale = D**-0.5
for name in masks:
    mask = getattr(gen, name)
    q = torch.randn(2, H, T, D)
    k = torch.randn(2, KV, T, D)
    v = torch.randn(2, KV, T, D)
    sinks = torch.randn(H)
    ours = gen.expression_attention(q, k, v, mask_mod=mask, sinks=sinks)

    block = create_block_mask(mask, None, None, T, T, device="cpu")
    fused, lse = flex_attention(q, k, v, block_mask=block, enable_gqa=True, return_lse=True)
    fused = fused * torch.sigmoid(lse - sinks.view(1, -1, 1)).unsqueeze(-1)

    idx = torch.arange(T)
    keep = mask(0, 0, idx.view(-1, 1), idx.view(1, -1)).view(1, 1, T, T)
    theirs = hugging_face(q, k, v, sinks, keep, scale)
    out["cases"][name] = {
        "fused": (ours - fused).abs().max().item(),
        "hugging_face": (ours - theirs).abs().max().item(),
    }


# A query with nothing but its sink: all of its attention goes nowhere, and
# its output is zero in every form.
def strictly_causal(b, h, q_idx, kv_idx):
    return kv_idx < q_idx


q = torch.randn(1, 2, 8, 16)
sinks = torch.zeros(2)
ours = gen.expression_attention(q, q, q, mask_mod=strictly_causal, sinks=sinks)
fused, lse = flex_attention(q, q, q, block_mask=create_block_mask(strictly_causal, None, None, 8, 8, device="cpu"), return_lse=True)
fused = fused * torch.sigmoid(lse - sinks.view(1, -1, 1)).unsqueeze(-1)
out["empty_row"] = {"ours": ours[:, :, 0].abs().max().item(), "fused": fused[:, :, 0].abs().max().item()}
out["finite"] = bool(torch.isfinite(ours).all())

# A sink far below every score takes nothing, and the attention is the one
# without it.
q = torch.randn(1, 2, 8, 16)
plain = gen.expression_attention(q, q, q, mask_mod=masks and getattr(gen, masks[0]))
silent = gen.expression_attention(q, q, q, mask_mod=masks and getattr(gen, masks[0]), sinks=torch.full((2,), -1e4))
out["silent_sink"] = (plain - silent).abs().max().item()

out["cuda"] = torch.cuda.is_available()
if out["cuda"] and masks:
    q = torch.randn(2, H, 64, D)
    k = torch.randn(2, KV, 64, D)
    v = torch.randn(2, KV, 64, D)
    sinks = torch.randn(H)
    mask = getattr(gen, masks[0])
    cpu = gen.expression_attention(q, k, v, mask_mod=mask, sinks=sinks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpu = gen.expression_attention(q.cuda(), k.cuda(), v.cuda(), mask_mod=mask, sinks=sinks.cuda())
    out["cuda_vs_cpu"] = (gpu.cpu() - cpu).abs().max().item()
    out["flex_compiled"] = bool(gen._FLEX)

print(json.dumps(out))
