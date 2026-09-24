"""Hold the generated expression_attention against FlexAttention itself.

Given a generated model.py, every mask_mod_N and score_mod_N in it is run
through the file's own expression_attention and through
torch.nn.attention.flex_attention, uncompiled, on the CPU — where both are the
plain computation, so they have to agree to rounding. That is what says the
functions the engine printed mean in the eager form what they mean to the
kernel.

On a GPU the helper is also called for real, which compiles FlexAttention
where Triton is installed and falls back, saying so, where it is not; either
way it has to give what the CPU gave.
"""

import importlib.util
import json
import sys
import warnings

import torch

spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec)
# FlexAttention traces the functions it is given, which means importing the
# module they came from by name.
sys.modules["gen"] = gen
spec.loader.exec_module(gen)

from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa: E402

masks = sorted(n for n in dir(gen) if n.startswith("mask_mod_"))
scores = sorted(n for n in dir(gen) if n.startswith("score_mod_"))
out = {"masks": masks, "scores": scores, "cases": {}}

torch.manual_seed(0)
for mask_name in masks + [None]:
    for score_name in scores + [None]:
        if mask_name is None and score_name is None:
            continue
        mask = getattr(gen, mask_name) if mask_name else None
        score = getattr(gen, score_name) if score_name else None
        for heads, kv_heads in ((4, 4), (4, 2)):
            q = torch.randn(2, heads, 40, 16)
            k = torch.randn(2, kv_heads, 40, 16)
            v = torch.randn(2, kv_heads, 40, 16)
            ours = gen.expression_attention(q, k, v, mask_mod=mask, score_mod=score, mask_heads=True)
            block = create_block_mask(mask, 2, heads, 40, 40, device="cpu") if mask else None
            theirs = flex_attention(q, k, v, score_mod=score, block_mask=block, enable_gqa=kv_heads != heads)
            key = f"{mask_name}/{score_name}/{heads}x{kv_heads}"
            out["cases"][key] = (ours - theirs).abs().max().item()


# A query with nothing to attend to gets zeros, as FlexAttention gives it.
def strictly_causal(b, h, q_idx, kv_idx):
    return kv_idx < q_idx


q = torch.randn(1, 2, 8, 16)
ours = gen.expression_attention(q, q, q, mask_mod=strictly_causal)
theirs = flex_attention(q, q, q, block_mask=create_block_mask(strictly_causal, None, None, 8, 8, device="cpu"))
out["empty_row"] = {"ours": ours[:, :, 0].abs().max().item(), "theirs": theirs[:, :, 0].abs().max().item()}
out["finite"] = bool(torch.isfinite(ours).all())

out["cuda"] = torch.cuda.is_available()
if out["cuda"] and masks:
    q = torch.randn(2, 4, 64, 16)
    k = torch.randn(2, 2, 64, 16)
    v = torch.randn(2, 2, 64, 16)
    mask = getattr(gen, masks[0])
    score = getattr(gen, scores[0]) if scores else None
    cpu = gen.expression_attention(q, k, v, mask_mod=mask, score_mod=score, mask_heads=True)
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        gpu = gen.expression_attention(q.cuda(), k.cuda(), v.cuda(), mask_mod=mask, score_mod=score, mask_heads=True)
    out["cuda_vs_cpu"] = (gpu.cpu() - cpu).abs().max().item()
    out["flex_compiled"] = bool(gen._FLEX)
    out["said_unfused"] = any("running unfused" in str(w.message) for w in said)

print(json.dumps(out))
