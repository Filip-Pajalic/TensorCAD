"""Hold a generated T5 against Hugging Face's, weight for weight.

The generated model's parameters are renamed into Hugging Face's layout
(transformers/models/t5/modeling_t5.py: `shared`, `encoder.block.N.layer.M`,
`relative_attention_bias` on each stack's first layer) and run through a
transcription of that file's forward pass: T5LayerNorm, T5Attention with its
bias computed once per stack and handed down, cross-attention with none, the
ReLU or gated-GELU feed-forward, the final norms, and the output rescaled by
d_model ** -0.5 when the head is tied. The logits have to agree, and the
renaming has to use every one of the generated model's weights exactly once, so
the two are the same parameters and not merely the same count.
"""

import importlib.util
import json
import math
import sys

import torch
import torch.nn.functional as F

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
    bucket = relative_position_bucket(memory_position - context_position, bidirectional, table.shape[0])
    return F.embedding(bucket, table).permute([2, 0, 1]).unsqueeze(0)


def layer_norm(weight, hidden, eps=1e-6):
    variance = hidden.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return weight * (hidden * torch.rsqrt(variance + eps))


def attention(p, prefix, hidden, kv, bias, heads):
    b = hidden.shape[0]
    d_kv = p[prefix + ".q.weight"].shape[0] // heads

    def shape(x):
        return x.view(b, -1, heads, d_kv).transpose(1, 2)

    q = shape(F.linear(hidden, p[prefix + ".q.weight"]))
    k = shape(F.linear(kv, p[prefix + ".k.weight"]))
    v = shape(F.linear(kv, p[prefix + ".v.weight"]))
    scores = torch.matmul(q, k.transpose(3, 2)) + bias
    weights = F.softmax(scores.float(), dim=-1).type_as(scores)
    out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, -1, heads * d_kv)
    return F.linear(out, p[prefix + ".o.weight"])


def feed_forward(p, prefix, hidden, gated):
    if gated:
        h = F.gelu(F.linear(hidden, p[prefix + ".wi_0.weight"]), approximate="tanh")
        h = h * F.linear(hidden, p[prefix + ".wi_1.weight"])
    else:
        h = F.relu(F.linear(hidden, p[prefix + ".wi.weight"]))
    return F.linear(h, p[prefix + ".wo.weight"])


def t5(p, source, target, layers, heads, gated, tied):
    d_model = p["shared.weight"].shape[1]
    # The encoder: its bias from its first layer's table, for every layer.
    x = F.embedding(source, p["shared.weight"])
    bias = compute_bias(p["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
                        source.shape[1], source.shape[1], True)
    for i in range(layers):
        pre = f"encoder.block.{i}.layer."
        normed = layer_norm(p[pre + "0.layer_norm.weight"], x)
        x = x + attention(p, pre + "0.SelfAttention", normed, normed, bias, heads)
        x = x + feed_forward(p, pre + "1.DenseReluDense", layer_norm(p[pre + "1.layer_norm.weight"], x), gated)
    memory = layer_norm(p["encoder.final_layer_norm.weight"], x)

    # The decoder: one-sided buckets, the causal mask added to the bias, and
    # cross-attention with no bias at all.
    T = target.shape[1]
    y = F.embedding(target, p["shared.weight"])
    causal = torch.triu(torch.full((T, T), torch.finfo(y.dtype).min), 1)
    bias = compute_bias(p["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"], T, T, False)
    bias = bias + causal
    for i in range(layers):
        pre = f"decoder.block.{i}.layer."
        normed = layer_norm(p[pre + "0.layer_norm.weight"], y)
        y = y + attention(p, pre + "0.SelfAttention", normed, normed, bias, heads)
        y = y + attention(p, pre + "1.EncDecAttention", layer_norm(p[pre + "1.layer_norm.weight"], y),
                          memory, 0.0, heads)
        y = y + feed_forward(p, pre + "2.DenseReluDense", layer_norm(p[pre + "2.layer_norm.weight"], y), gated)
    y = layer_norm(p["decoder.final_layer_norm.weight"], y)
    if tied:
        y = y * (d_model**-0.5)
    return F.linear(y, p["shared.weight"] if tied else p["lm_head.weight"])


torch.manual_seed(0)
cls = next(c for c in vars(gen).values()
           if isinstance(c, type) and issubclass(c, torch.nn.Module) and c.__module__ == "gen"
           and "source" in c.forward.__code__.co_varnames[:3])
model = cls().init_weights().eval()
# Nothing starts at a value that would hide a mistake: the tables away from
# zero, the norm gains away from one.
with torch.no_grad():
    for name, param in model.named_parameters():
        if name.endswith("_bias"):
            param.normal_(0.0, 1.0)
        elif "norm" in name:
            param.normal_(1.0, 0.1)

ours = dict(model.named_parameters())
layers, heads = len(model.encoder), ours["enc_bias"].shape[1]
gated = "encoder.0.block.mlp.gate.weight" in ours
tied = model.head.weight is model.embed.weight
p = {"shared.weight": ours["embed.weight"]}
for stack, table in (("encoder", "enc_bias"), ("decoder", "dec_bias")):
    p[f"{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = ours[table]
    for i in range(layers):
        g, h = f"{stack}.{i}.block.", f"{stack}.block.{i}.layer."
        for w in "qkvo":
            p[f"{h}0.SelfAttention.{w}.weight"] = ours[f"{g}attn.{w}_proj.weight"]
        p[h + "0.layer_norm.weight"] = ours[g + "norm1.weight"]
        ff = "2" if stack == "decoder" else "1"
        if stack == "decoder":
            for w in "qkvo":
                p[f"{h}1.EncDecAttention.{w}.weight"] = ours[f"{g}cross.{w}_proj.weight"]
            p[h + "1.layer_norm.weight"] = ours[g + "norm_cross.weight"]
        if gated:
            p[f"{h}{ff}.DenseReluDense.wi_0.weight"] = ours[g + "mlp.gate.weight"]
            p[f"{h}{ff}.DenseReluDense.wi_1.weight"] = ours[g + "mlp.up.weight"]
        else:
            p[f"{h}{ff}.DenseReluDense.wi.weight"] = ours[g + "mlp.up.weight"]
        p[f"{h}{ff}.DenseReluDense.wo.weight"] = ours[g + "mlp.down.weight"]
        p[f"{h}{ff}.layer_norm.weight"] = ours[g + "norm2.weight"]
p["encoder.final_layer_norm.weight"] = ours["enc_norm.weight"]
p["decoder.final_layer_norm.weight"] = ours["final_norm.weight"]
if not tied:
    p["lm_head.weight"] = ours["head.weight"]

used = [id(t) for t in p.values()]
out = {
    "layers": layers,
    "heads": heads,
    "gated": gated,
    "tied": tied,
    "every_weight_once": sorted(used) == sorted(id(t) for t in model.parameters()),
    "hf_params": sum(t.numel() for t in p.values()),
}
source = torch.randint(0, p["shared.weight"].shape[0], (2, 40))
target = torch.randint(0, p["shared.weight"].shape[0], (2, 24))
with torch.no_grad():
    mine = model(source, target)
    theirs = t5(p, source, target, layers, heads, gated, tied)
out["logits_scale"] = theirs.abs().max().item()
out["logits_diff"] = (mine - theirs).abs().max().item()
print(json.dumps(out))
