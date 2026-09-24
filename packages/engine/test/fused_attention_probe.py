import importlib.util, json, sys
import torch

out = {"cuda": torch.cuda.is_available()}
if not out["cuda"]:
    print(json.dumps(out)); sys.exit(0)

spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

calls = []

def flash_attn_func(q, k, v, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, **_):
    # FlashAttention's documented contract, written independently of the helper:
    # (B, T, heads, head_dim); fewer key/value heads than query heads; a query at
    # i sees keys [i - left, i + right] inclusive, causal aligned to the bottom
    # right; the cap applied to the scaled scores before the softmax.
    calls.append({"window_size": list(window_size), "softcap": softcap, "causal": causal})
    B, Tq, H, D = q.shape
    Tk, Hk = k.shape[1], k.shape[2]
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    kf = kf.repeat_interleave(H // Hk, dim=1)
    vf = vf.repeat_interleave(H // Hk, dim=1)
    s = (qf @ kf.transpose(-2, -1)) * (softmax_scale if softmax_scale is not None else D ** -0.5)
    if softcap > 0:
        s = softcap * torch.tanh(s / softcap)
    i = torch.arange(Tq, device=q.device)[:, None] + (Tk - Tq)
    j = torch.arange(Tk, device=q.device)[None, :]
    keep = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device)
    if causal:
        keep &= j <= i
    left, right = window_size
    if left >= 0:
        keep &= j >= i - left
    if right >= 0:
        keep &= j <= i + right
    s = s.masked_fill(~keep, float("-inf"))
    return (torch.softmax(s, -1) @ vf).transpose(1, 2).to(q.dtype)

torch.manual_seed(0)
cases = {"window": dict(window=16), "cap": dict(softcap=50.0), "both": dict(window=16, softcap=50.0)}
worst = {}
for name, kw in cases.items():
    for gqa in (False, True):
        q = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
        kv = 2 if gqa else 4
        k = torch.randn(2, kv, 64, 32, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(2, kv, 64, 32, device="cuda", dtype=torch.bfloat16)
        # The reference is the fallback in float32, which never takes the fused
        # path; the fused path runs in bf16, as it would in training. Measured
        # relative to the output's size, so rounding is a few thousandths and a
        # wrong window or scale is several hundredths.
        gen._flash_attn_func = flash_attn_func
        ref = gen.fused_attention(q.float(), k.float(), v.float(), enable_gqa=gqa, **kw)
        fast = gen.fused_attention(q, k, v, enable_gqa=gqa, **kw)
        key = f"{name}{'-gqa' if gqa else ''}"
        worst[key] = ((fast.float() - ref).abs().max() / ref.abs().max()).item()
out["worst"] = worst

# A whole model on the fused path: every attention layer goes through it.
calls.clear()
model = gen.__dict__[sys.argv[2]]().cuda().to(torch.bfloat16)
if hasattr(model, "init_weights"):
    model.init_weights()
with torch.no_grad():
    logits = model(torch.randint(0, 256, (1, 64), device="cuda"))
out["model_calls"] = len(calls)
out["model_windows"] = sorted({tuple(c["window_size"]) for c in calls})
out["finite"] = bool(torch.isfinite(logits.float()).all())
print(json.dumps(out))
