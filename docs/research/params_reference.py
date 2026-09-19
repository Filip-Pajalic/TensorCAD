"""Regression check of parameter-count formulas against published configs."""

def dense_llama(L, D, H, Hkv, dh, F, V, tied=False, gated=True, norms_per_layer=2, bias=False):
    q = D * H * dh; k = D * Hkv * dh; v = k; o = H * dh * D
    attn = q + k + v + o
    mlp = (3 if gated else 2) * D * F
    norm = norms_per_layer * D
    layer = attn + mlp + norm
    if bias:
        layer += (H*dh + 2*Hkv*dh + D) + (F*(2 if gated else 1) + D)  # rough
    emb = V * D
    head = 0 if tied else V * D
    total = L * layer + emb + head + D  # final norm
    return total, dict(attn=attn, mlp=mlp, layer=layer, emb=emb)

def moe_llama(L, D, H, Hkv, dh, V, E, k, Fe, F_dense=0, n_shared=0, dense_layers=0, tied=False, router_bias=False):
    q = D * H * dh; kk = D * Hkv * dh; o = H * dh * D
    attn = q + 2*kk + o
    expert = 3 * D * Fe
    router = D * E + (E if router_bias else 0)
    moe_layer = attn + E*expert + n_shared*expert + router + 2*D
    dense_layer = attn + 3*D*F_dense + 2*D
    emb = V*D; head = 0 if tied else V*D
    Lm = L - dense_layers
    total = Lm*moe_layer + dense_layers*dense_layer + emb + head + D
    active = Lm*(attn + k*expert + n_shared*expert + router + 2*D) + dense_layers*dense_layer + emb + head + D
    return total, active

def mla_attn(D, H, q_lora, kv_lora, nope, rope, vh):
    # DeepSeek-V2/V3 style MLA
    q = D*q_lora + q_lora + q_lora*H*(nope+rope)          # W_DQ, q_norm, W_UQ
    kv = D*(kv_lora+rope) + kv_lora + kv_lora*H*(nope+vh)  # W_DKV(+W_KR), kv_norm, W_UKV
    o = H*vh*D
    return q+kv+o

def deepseek_v3():
    D=7168; L=61; H=128; V=129280; E=256; k=8; Fe=2048; F=18432
    attn = mla_attn(D,H,1536,512,128,64,128)
    expert = 3*D*Fe
    router = D*E + E  # gate + e_score_correction_bias
    moe_layer = attn + E*expert + 1*expert + router + 2*D
    dense_layer = attn + 3*D*F + 2*D
    emb = V*D; head = V*D
    total = 58*moe_layer + 3*dense_layer + emb + head + D
    active = 58*(attn + k*expert + expert + router + 2*D) + 3*dense_layer + emb + head + D
    # MTP module: 1 extra MoE layer + its own embedding/head share + proj (2D->D) + 2 norms
    mtp = moe_layer + 2*D*D + 2*D + emb + head  # HF checkpoint stores separate emb/head for MTP
    kv_per_token_bytes = L*(512+64)*2
    return total, active, attn, mtp, kv_per_token_bytes

def mamba2_block(D, expand=2, headdim=64, d_state=128, ngroups=1, d_conv=4, conv_bias=True):
    d_inner = expand*D; d_ssm = d_inner; nheads = d_ssm//headdim
    d_in_proj = 2*d_inner + 2*ngroups*d_state + nheads
    in_proj = D*d_in_proj
    conv_dim = d_ssm + 2*ngroups*d_state
    conv = conv_dim*d_conv + (conv_dim if conv_bias else 0)
    small = nheads*3  # dt_bias, A_log, D
    norm = d_ssm
    out_proj = d_inner*D
    return in_proj+conv+small+norm+out_proj, dict(state_per_token_layer=nheads*headdim*d_state)

def nemotron_h_8b():
    D=4096; V=131072; pattern="M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    nM=pattern.count('M'); nA=pattern.count('*'); nF=pattern.count('-')
    mamba,_ = mamba2_block(D, expand=2, headdim=64, d_state=128, ngroups=8)
    attn = D*32*128 + 2*D*8*128 + 32*128*D
    ffn = 2*D*21504  # relu^2, not gated
    total = nM*(mamba+D) + nA*(attn+D) + nF*(ffn+D) + 2*V*D + D
    return total, (nM,nA,nF), mamba

def gpt2(L, D, H, V=50257, ctx=1024):
    layer = 12*D*D + 13*D  # weights + biases(4D+ 2*4D... approx) + 2 LN (4D)
    # exact: attn 4D^2 + 4D bias; mlp 8D^2 + 5D bias; LN 2*(2D)
    layer = (4*D*D + 4*D) + (8*D*D + 4*D + D) + 4*D
    return L*layer + V*D + ctx*D + 2*D

if __name__ == "__main__":
    print("Llama-2-7B   ", dense_llama(32,4096,32,32,128,11008,32000)[0]/1e9, "pub 6.74B")
    print("Llama-3-8B   ", dense_llama(32,4096,32,8,128,14336,128256)[0]/1e9, "pub 8.03B")
    print("Llama-3-70B  ", dense_llama(80,8192,64,8,128,28672,128256)[0]/1e9, "pub 70.6B")
    print("Llama-3-405B ", dense_llama(126,16384,128,8,128,53248,128256)[0]/1e9, "pub 405B")
    print("Mistral-7B   ", dense_llama(32,4096,32,8,128,14336,32000)[0]/1e9, "pub 7.24B")
    print("Qwen2.5-7B   ", dense_llama(28,3584,28,4,128,18944,152064, bias=False)[0]/1e9, "pub 7.61B (has qkv bias)")
    print("Qwen3-8B     ", dense_llama(36,4096,32,8,128,12288,151936)[0]/1e9, "pub 8.2B (has q/k norm)")
    print("Gemma-2-9B   ", dense_llama(42,3584,16,8,256,14336,256000, tied=True, norms_per_layer=4)[0]/1e9, "pub 9.24B")
    print("Gemma-3-27B  ", dense_llama(62,5376,32,16,128,21504,262208, tied=True, norms_per_layer=4)[0]/1e9, "pub 27.4B incl vision")
    t,a = moe_llama(32,4096,32,8,128,32000,8,2,14336)
    print("Mixtral-8x7B ", t/1e9, a/1e9, "pub 46.7B / 12.9B active")
    t,a = moe_llama(48,2048,32,4,128,151936,128,8,768)
    print("Qwen3-30B-A3B", t/1e9, a/1e9, "pub 30.5B / 3.3B active")
    t,a = moe_llama(94,4096,64,4,128,151936,128,8,1536)
    print("Qwen3-235B   ", t/1e9, a/1e9, "pub 235B / 22B active")
    t,a,attn,mtp,kv = deepseek_v3()
    print("DeepSeek-V3  ", t/1e9, a/1e9, "attn/layer", attn/1e6, "MTP", mtp/1e9, "KV B/tok", kv, "pub 671B / 37B, +14B MTP, 70KB/tok")
    t,counts,mamba = nemotron_h_8b()
    print("Nemotron-H-8B", t/1e9, counts, "mamba blk", mamba/1e6, "pub ~8B")
    for name,(L,D,H) in {"gpt2":(12,768,12),"gpt2-medium":(24,1024,16),"gpt2-large":(36,1280,20),"gpt2-xl":(48,1600,25)}.items():
        print(name, gpt2(L,D,H)/1e6)
