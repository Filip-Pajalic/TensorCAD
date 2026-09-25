// Package codegen emits PyTorch from a design.
//
// One generic emitter walks a graph and produces an nn.Module. Composites and
// repeat containers become their own classes, named after the block, so the
// output reads like code somebody wrote rather than a flattened trace. Because
// there is a single code path, generated code cannot drift from the analysis:
// both consume the same expansion.
//
// The emitted file has no dependency beyond PyTorch itself.
package codegen

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/attnexpr"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/shapes"
)

// Options steer the emitter.
type Options struct {
	// ClassName is the name of the top-level module; empty derives it from the
	// design's own name.
	ClassName string
	// NoSmokeTest leaves out the __main__ block that instantiates the model and
	// checks its size. The block is emitted by default.
	NoSmokeTest bool
	// MoeDispatch is how a mixture-of-experts layer routes tokens.
	//
	// "sparse", the default, gathers the rows each expert was given, which is
	// what you want for speed but uses nonzero, whose output shape depends on
	// the data. That makes it untraceable by torch.export.
	//
	// "dense" runs every expert over every token and weights the results by
	// whether the expert was selected. It computes the same thing, costs
	// experts/top_k times as much, and traces cleanly. Use it to verify a
	// design, not to train one.
	MoeDispatch string
	// InitStd is the standard deviation for the weight initialization. Nil
	// takes the default of 0.02, and zero leaves PyTorch's own defaults alone.
	//
	// This matters more than it looks. nn.Embedding defaults to a unit normal,
	// which starts a language model at a cross-entropy of a few hundred instead
	// of ln(vocab), and it takes a long time to recover.
	InitStd *float64
}

// WireOptions is Options as a client sends it, and the only spelling of these
// settings that a client should know.
//
// It differs from Options in one way, deliberately: the smoke test is named for
// what a caller decides ("include it") rather than for what the emitter then
// skips. Everything is optional, so an absent field and a zero one have to be
// distinguishable, which is what the pointers are for.
type WireOptions struct {
	ClassName        string   `json:"className,omitempty"`
	IncludeSmokeTest *bool    `json:"includeSmokeTest,omitempty"`
	MoeDispatch      string   `json:"moeDispatch,omitempty"`
	InitStd          *float64 `json:"initStd,omitempty"`
}

// Options is what the emitter reads.
func (w WireOptions) Options() Options {
	o := Options{ClassName: w.ClassName, MoeDispatch: w.MoeDispatch, InitStd: w.InitStd}
	if w.IncludeSmokeTest != nil && !*w.IncludeSmokeTest {
		o.NoSmokeTest = true
	}
	return o
}

// Wire is the inverse, for writing down the settings an answer was produced
// under in the words a client would have used.
func Wire(o Options) WireOptions {
	w := WireOptions{ClassName: o.ClassName, MoeDispatch: o.MoeDispatch, InitStd: o.InitStd}
	if o.NoSmokeTest {
		no := false
		w.IncludeSmokeTest = &no
	}
	return w
}

// File is one emitted file.
type File struct {
	Path     string `json:"path"`
	Contents string `json:"contents"`
}

// Generated is everything the emitter produced.
type Generated struct {
	Files []File `json:"files"`
	// Warnings is never nil: an empty list means the design generated cleanly,
	// and null would mean the same thing in a way every reader has to handle.
	Warnings []string `json:"warnings"`
}

// ---------------------------------------------------------------------------
// Helpers emitted only when the design uses them
// ---------------------------------------------------------------------------

const helperRope = `class RotaryEmbedding(nn.Module):
    """Rotary position embedding over the last dimension of (B, H, T, head_dim)."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[-2]
        pos = torch.arange(seq, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(x.device))
        cos = freqs.cos().to(x.dtype)[None, None, :, :]
        sin = freqs.sin().to(x.dtype)[None, None, :, :]
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
`

// helperFused is attention with a sliding window, a cap on the scores, or
// both. The analysis counts these as one fused kernel that skips the key blocks
// outside the window and caps inside the kernel, so this uses that kernel —
// FlashAttention 2.6 or later — wherever it can, and computes the same numbers
// the long way everywhere else, saying so once on a GPU, where the difference
// is memory somebody is paying for.
//
// The long way is what the generated code did before, exactly: a window alone
// through F.scaled_dot_product_attention with an additive mask, a cap through
// the eager form. So verification on a CPU — the parameter count, the FLOP
// count against the profiler, the export — sees the same model it always did.
const helperFused = `try:
    from flash_attn import flash_attn_func as _flash_attn_func
except ImportError:  # an optional install, and only ever on CUDA
    _flash_attn_func = None

_UNFUSED_SAID = False


def fused_attention(q, k, v, window=0, softcap=0.0, causal=True, scale=None, enable_gqa=False):
    """Attention with a sliding window, a cap on the scores, or both.

    The design's analysis counts this as one fused kernel: it skips the keys
    outside the window and applies the cap inside the kernel, so the score
    matrix is never kept for the backward pass. FlashAttention, 2.6 or later,
    is that kernel, and it is used whenever it can be: on CUDA, in half
    precision, with flash_attn installed.

    Anywhere else the same numbers are computed the long way: every score,
    then the mask. A window alone still goes through a memory-efficient kernel;
    a cap is computed eagerly and keeps the score matrix. Either way it is more
    than the analysis counts, which is said once when it happens on a GPU.

    q is (B, heads, T, head_dim); k and v may have fewer heads than q.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if (
        _flash_attn_func is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and v.shape[-1] == q.shape[-1]
    ):
        # FlashAttention takes (B, T, heads, head_dim), repeats grouped key and
        # value heads itself, and counts a window as how far back and forward
        # a query may look: back window - 1, forward nothing.
        y = _flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            softmax_scale=scale,
            causal=causal,
            window_size=(window - 1, 0) if window > 0 else (-1, -1),
            softcap=softcap,
        )
        return y.transpose(1, 2)

    global _UNFUSED_SAID
    if q.is_cuda and not _UNFUSED_SAID:
        _UNFUSED_SAID = True
        import warnings

        warnings.warn(
            "attention with a window or a score cap is running unfused: install flash-attn 2.6 or "
            "later for the kernel the design's memory and FLOP figures assume",
            stacklevel=2,
        )

    seq = q.shape[-2]
    i = torch.arange(seq, device=q.device)
    allowed = torch.ones(seq, seq, dtype=torch.bool, device=q.device)
    if causal:
        allowed = allowed & (i[:, None] >= i[None, :])
    if window > 0:
        allowed = allowed & (i[:, None] - i[None, :] < window)
    mask = torch.zeros(seq, seq, device=q.device, dtype=q.dtype).masked_fill(~allowed, float("-inf"))
    if not softcap:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=enable_gqa)
    if enable_gqa and k.shape[-3] != q.shape[-3]:
        k = k.repeat_interleave(q.shape[-3] // k.shape[-3], dim=-3)
        v = v.repeat_interleave(q.shape[-3] // v.shape[-3], dim=-3)
    scores = torch.tanh((q @ k.transpose(-2, -1)) * scale / softcap) * softcap
    return torch.softmax(scores + mask, dim=-1).to(v.dtype) @ v
`

// helperExpression is attention with the design's own mask or score
// expressions, which is FlexAttention's shape: two small functions over a
// score and its position, compiled into one fused kernel. The analysis counts
// that kernel, so this uses it wherever it compiles, which is CUDA with
// Triton, and everywhere else applies the same two functions to the whole
// score matrix — the form a CPU verifies, profiles and exports, and the one
// FlexAttention's own documentation defines it by.
const helperExpression = `_FLEX = None
_BLOCK_MASKS = {}
_EXPRESSIONS_UNFUSED_SAID = False


def _flex_attention():
    """FlexAttention compiled once, with what builds its block mask, or False."""
    global _FLEX
    if _FLEX is None:
        try:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        except ImportError:  # PyTorch before 2.5
            _FLEX = False
        else:
            _FLEX = (torch.compile(flex_attention, dynamic=False), create_block_mask)
    return _FLEX


def _say_unfused(why):
    global _EXPRESSIONS_UNFUSED_SAID
    if not _EXPRESSIONS_UNFUSED_SAID:
        _EXPRESSIONS_UNFUSED_SAID = True
        import warnings

        warnings.warn(
            "attention with a mask or score expression is running unfused (" + why + "), which keeps "
            "the score matrix the design's memory figures assume a fused kernel never builds",
            stacklevel=3,
        )


def expression_attention(
    q, k, v, mask_mod=None, score_mod=None, mask_heads=False, mask_batch=False, scale=None, sinks=None
):
    """Attention whose scores the design's expressions mask or change.

    mask_mod(b, h, q_idx, kv_idx) says whether a score counts, and
    score_mod(score, b, h, q_idx, kv_idx) what it becomes before the softmax:
    FlexAttention's own signatures. The design's analysis counts this as
    FlexAttention runs it, one fused kernel that skips every block the mask
    removes and changes each score inside, so the score matrix is never kept
    for the backward pass. On CUDA it is that kernel, compiled on first use.

    Anywhere else, or where it does not compile, the same two functions are
    applied to the whole score matrix: every score, then the mask. A query the
    mask leaves nothing to attend to gets zeros, as FlexAttention gives it.

    sinks, one learned score per query head, sits in each row's softmax
    beside the keys, so a query can put its attention nowhere. Fused, that is
    the output rescaled by sigmoid(lse - sink), from the log-sum-exp the kernel
    returns; the long way, it is one more column in the softmax that no value
    is read by, which is how the model it comes from writes it.

    q is (B, heads, T, head_dim); k and v may have fewer heads than q.
    """
    global _FLEX
    if scale is None:
        scale = q.shape[-1] ** -0.5
    batch, heads, seq, _ = q.shape
    keys = k.shape[-2]
    grouped = k.shape[-3] != heads
    if q.is_cuda and _flex_attention():
        flex, create_block_mask = _FLEX
        try:
            block_mask = None
            if mask_mod is not None:
                key = (mask_mod, batch if mask_batch else None, heads if mask_heads else None, seq, keys, q.device)
                block_mask = _BLOCK_MASKS.get(key)
                if block_mask is None:
                    block_mask = create_block_mask(mask_mod, key[1], key[2], seq, keys, device=q.device)
                    _BLOCK_MASKS[key] = block_mask
            if sinks is None:
                return flex(q, k, v, score_mod=score_mod, block_mask=block_mask, scale=scale, enable_gqa=grouped)
            out, lse = flex(
                q, k, v, score_mod=score_mod, block_mask=block_mask, scale=scale, enable_gqa=grouped, return_lse=True
            )
            return out * torch.sigmoid(lse - sinks.float().view(1, -1, 1)).unsqueeze(-1).to(out.dtype)
        except Exception as error:  # most often, no Triton to compile it with
            _FLEX = False
            _say_unfused("FlexAttention did not compile: " + type(error).__name__)
    elif q.is_cuda:
        _say_unfused("this PyTorch has no FlexAttention, which arrived in 2.5")

    if grouped:
        k = k.repeat_interleave(heads // k.shape[-3], dim=-3)
        v = v.repeat_interleave(heads // v.shape[-3], dim=-3)
    b = torch.arange(batch, device=q.device).view(-1, 1, 1, 1)
    h = torch.arange(heads, device=q.device).view(1, -1, 1, 1)
    q_idx = torch.arange(seq, device=q.device).view(1, 1, -1, 1)
    kv_idx = torch.arange(keys, device=q.device).view(1, 1, 1, -1)
    scores = (q @ k.transpose(-2, -1)) * scale
    if score_mod is not None:
        scores = score_mod(scores, b, h, q_idx, kv_idx)
    if mask_mod is not None:
        scores = scores.masked_fill(~mask_mod(b, h, q_idx, kv_idx), float("-inf"))
    if sinks is not None:
        sink = sinks.view(1, -1, 1, 1).expand(scores.shape[0], -1, scores.shape[2], 1)
        scores = torch.cat([scores, sink.to(scores.dtype)], dim=-1)
    weights = torch.softmax(scores.float(), dim=-1).nan_to_num(0.0)
    if sinks is not None:
        weights = weights[..., :-1]
    return weights.to(v.dtype) @ v
`

// helperBucket is T5's relative-position bucket, for a score expression that
// calls t5_bucket. It is Hugging Face's _relative_position_bucket written over
// whatever tensors it is given: zero-dimensional ones inside FlexAttention's
// score_mod, broadcast grids in the eager fallback.
const helperBucket = `def t5_bucket(relative_position, num_buckets, max_distance, bidirectional):
    """T5's relative-position bucket for kv - q.

    Exact while the distance is small, then logarithmically spaced out to
    max_distance, and everything further shares the last bucket. Two-sided
    attention spends half its buckets on each side. This is Hugging Face's
    _relative_position_bucket, written for any tensor it is handed.
    """
    import math

    buckets = 0
    if bidirectional:
        num_buckets //= 2
        buckets = buckets + (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.clamp(relative_position, max=0)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    large = torch.clamp(large, max=num_buckets - 1)
    return buckets + torch.where(is_small, relative_position, large)
`

const helperShift = "def shift_sequence(x: torch.Tensor, by: int) -> torch.Tensor:\n" +
	"    \"\"\"Move a sequence `by` positions earlier, zero-filling the end.\n" +
	"\n" +
	"    What a multi-token predictor reads: at depth k it wants the embedding of\n" +
	"    the token k ahead, which over a whole training sequence is this. A slice\n" +
	"    and a concatenation rather than a roll, so nothing wraps around from the\n" +
	"    end of the sequence to the start of it.\n" +
	"    \"\"\"\n" +
	"    if by <= 0:\n" +
	"        return x\n" +
	"    pad = torch.zeros_like(x[..., :by, :])\n" +
	"    return torch.cat([x[..., by:, :], pad], dim=-2)\n"

const helperGatedDelta = "class GatedDeltaScan(nn.Module):\n" +
	"    \"\"\"Gated DeltaNet recurrence.\n" +
	"\n" +
	"        S_t = S_{t-1} (a_t (I - b_t k_t k_t^T)) + b_t v_t k_t^T\n" +
	"        o_t = S_t q_t\n" +
	"\n" +
	"    Mamba-2's decay gate and DeltaNet's write rule in one step. A readable\n" +
	"    sequential reference so the generated file runs unmodified; for a real\n" +
	"    training run, swap it for the chunked kernel in `fla`, which computes the\n" +
	"    same thing without a Python loop over the sequence.\n" +
	"    \"\"\"\n" +
	"\n" +
	"    def __init__(self, heads: int, head_dim: int, v_head_dim: int, value_heads: int):\n" +
	"        super().__init__()\n" +
	"        self.heads, self.head_dim, self.v_head_dim = heads, head_dim, v_head_dim\n" +
	"        # One state per value head, which is not always the number of key\n" +
	"        # heads: Qwen3-Next has twice as many.\n" +
	"        self.value_heads = value_heads\n" +
	"        self.A_log = nn.Parameter(torch.zeros(value_heads))\n" +
	"        self.dt_bias = nn.Parameter(torch.zeros(value_heads))\n" +
	"\n" +
	"    def forward(self, q, k, v, gates):\n" +
	"        b, h, t, _ = v.shape\n" +
	"        # One key head can serve several value heads, the way grouped-query\n" +
	"        # attention shares a key across a group. The recurrence runs per\n" +
	"        # value head, so the queries and keys are repeated up to it.\n" +
	"        if self.heads != self.value_heads:\n" +
	"            share = self.value_heads // self.heads\n" +
	"            q = q.repeat_interleave(share, dim=1)\n" +
	"            k = k.repeat_interleave(share, dim=1)\n" +
	"        beta_raw, alpha_raw = torch.split(gates, [self.value_heads, self.value_heads], dim=-1)\n" +
	"        # The write strength is a fraction; the decay is Mamba-2's, so that a\n" +
	"        # large timestep forgets more.\n" +
	"        beta = torch.sigmoid(beta_raw).transpose(1, 2)\n" +
	"        dt = F.softplus(alpha_raw + self.dt_bias).transpose(1, 2)\n" +
	"        alpha = torch.exp(-torch.exp(self.A_log)[None, :, None] * dt)\n" +
	"        # The delta rule needs a unit key, or the state it removes is not the\n" +
	"        # state the key wrote.\n" +
	"        k = F.normalize(k.float(), dim=-1)\n" +
	"        q = q.float()\n" +
	"        v = v.float()\n" +
	"        state = torch.zeros(b, h, self.v_head_dim, self.head_dim, device=q.device, dtype=torch.float32)\n" +
	"        out = []\n" +
	"        for i in range(t):\n" +
	"            ki = k[:, :, i]\n" +
	"            bi = beta[:, :, i, None]\n" +
	"            state = state * alpha[:, :, i, None, None]\n" +
	"            # Remove what this key already held, then write the new value.\n" +
	"            held = torch.einsum(\"bhvd,bhd->bhv\", state, ki)\n" +
	"            state = state + torch.einsum(\"bhv,bhd->bhvd\", bi * (v[:, :, i] - held), ki)\n" +
	"            out.append(torch.einsum(\"bhvd,bhd->bhv\", state, q[:, :, i]))\n" +
	"        return torch.stack(out, dim=2).to(v.dtype)\n"

const helperSelective = "class SelectiveScan(nn.Module):\n" +
	"    \"\"\"Mamba-1's selective scan.\n" +
	"\n" +
	"        h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t\n" +
	"        y_t = C_t h_t + D x_t\n" +
	"\n" +
	"    Selective because dt, B and C are read from the token rather than fixed,\n" +
	"    so what the state keeps depends on what it just saw. A readable sequential\n" +
	"    reference so the generated file runs unmodified; for a real training run,\n" +
	"    swap it for the fused kernel in `mamba_ssm`, which computes the same thing\n" +
	"    without a Python loop over the sequence.\n" +
	"    \"\"\"\n" +
	"\n" +
	"    def __init__(self, d_inner: int, state: int):\n" +
	"        super().__init__()\n" +
	"        self.d_inner, self.state = d_inner, state\n" +
	"        # A is held as a log so it stays negative and the recurrence decays.\n" +
	"        self.A_log = nn.Parameter(torch.log(torch.arange(1, state + 1, dtype=torch.float32)).repeat(d_inner, 1))\n" +
	"        self.D = nn.Parameter(torch.ones(d_inner))\n" +
	"\n" +
	"    def forward(self, x, dt, b, c):\n" +
	"        batch, t, _ = x.shape\n" +
	"        dt = F.softplus(dt.float())\n" +
	"        a = -torch.exp(self.A_log.float())\n" +
	"        h = torch.zeros(batch, self.d_inner, self.state, device=x.device, dtype=torch.float32)\n" +
	"        out = []\n" +
	"        for i in range(t):\n" +
	"            dti = dt[:, i][..., None]\n" +
	"            # Zero-order hold: the discount and the write both scale with dt.\n" +
	"            h = h * torch.exp(dti * a[None]) + dti * b[:, i].float()[:, None, :] * x[:, i].float()[..., None]\n" +
	"            out.append(torch.einsum(\"bdn,bn->bd\", h, c[:, i].float()))\n" +
	"        y = torch.stack(out, dim=1)\n" +
	"        return (y + x.float() * self.D).to(x.dtype)\n"

const helperSsd = "class SSDScan(nn.Module):\n" +
	"    \"\"\"Mamba-2 state-space scan.\n" +
	"\n" +
	"    A readable sequential reference so the generated file runs unmodified. For a\n" +
	"    real training run, swap this for the fused kernel in `mamba_ssm`; it is the\n" +
	"    same recurrence but orders of magnitude faster.\n" +
	"    \"\"\"\n" +
	"\n" +
	"    def __init__(self, heads: int, head_dim: int, state: int, groups: int):\n" +
	"        super().__init__()\n" +
	"        self.heads, self.head_dim, self.state, self.groups = heads, head_dim, state, groups\n" +
	"        self.A_log = nn.Parameter(torch.zeros(heads))\n" +
	"        self.D = nn.Parameter(torch.ones(heads))\n" +
	"        self.dt_bias = nn.Parameter(torch.zeros(heads))\n" +
	"\n" +
	"    def forward(self, xbc: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:\n" +
	"        b, t, _ = xbc.shape\n" +
	"        d_inner = self.heads * self.head_dim\n" +
	"        gs = self.groups * self.state\n" +
	"        x, bs, cs = torch.split(xbc, [d_inner, gs, gs], dim=-1)\n" +
	"        x = x.view(b, t, self.heads, self.head_dim)\n" +
	"        rep = self.heads // self.groups\n" +
	"        bs = bs.view(b, t, self.groups, self.state).repeat_interleave(rep, dim=2)\n" +
	"        cs = cs.view(b, t, self.groups, self.state).repeat_interleave(rep, dim=2)\n" +
	"        dt = F.softplus(dt + self.dt_bias)\n" +
	"        a = -torch.exp(self.A_log)\n" +
	"        h = torch.zeros(b, self.heads, self.head_dim, self.state, device=xbc.device, dtype=torch.float32)\n" +
	"        out = []\n" +
	"        for i in range(t):\n" +
	"            decay = torch.exp(dt[:, i] * a)[:, :, None, None].float()\n" +
	"            h = h * decay + (dt[:, i][:, :, None, None] * x[:, i][..., None] * bs[:, i][:, :, None, :]).float()\n" +
	"            y = (h * cs[:, i][:, :, None, :].float()).sum(-1).to(x.dtype) + self.D[None, :, None] * x[:, i]\n" +
	"            out.append(y)\n" +
	"        return torch.stack(out, dim=1).reshape(b, t, d_inner)\n"

// activationCall is how each nonlinearity is written in Python.
var activationCall = map[string]string{
	"silu":      "F.silu(%s)",
	"swish":     "F.silu(%s)",
	"gelu":      "F.gelu(%s)",
	"gelu_tanh": `F.gelu(%s, approximate="tanh")`,
	"relu":      "F.relu(%s)",
	"relu2":     "F.relu(%s).square()",
	"tanh":      "torch.tanh(%s)",
	"sigmoid":   "torch.sigmoid(%s)",
	"identity":  "%s",
}

func activation(kind, value string) string {
	tmpl, ok := activationCall[kind]
	if !ok {
		tmpl = activationCall["identity"]
	}
	return fmt.Sprintf(tmpl, value)
}

func activationOr(kind, fallback, value string) string {
	if _, ok := activationCall[kind]; !ok {
		kind = fallback
	}
	return activation(kind, value)
}

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

type emitted struct {
	name string
	code string
}

type ctx struct {
	doc *ir.Doc
	// cat is the document's catalog, so blocks it defines itself generate too.
	cat      catalog.Catalog
	symbols  *ir.SymbolTable
	warnings []string
	// classes are the deduplicated classes, in dependency order.
	classes    []emitted
	byKey      map[string]string
	usedNames  map[string]bool
	needsRope  bool
	needsFused bool
	// inputArgs names the model's arguments when it has more than one input:
	// each input node's own name.
	inputArgs map[string]string
	// needsBucket is T5's bucket function, for a score that calls it.
	needsBucket bool
	// needsExpression is the FlexAttention helper, and attnFuncs are the
	// mask and score functions it is handed, one per distinct expression.
	needsExpression bool
	attnFuncs       []string
	attnNames       map[string]string
	needsShift      bool
	needsSsd        bool
	needsSelective  bool
	needsGdn        bool
	moeDispatch     string
}

func (c *ctx) warn(format string, args ...any) {
	c.warnings = append(c.warnings, fmt.Sprintf(format, args...))
}

// attentionFunction is the name of a module-level mask_mod or score_mod
// computing an expression, emitting it the first time it is asked for. Two
// layers with the same expression share one function, as they share a class.
//
// A score that reads a tensor is a factory instead: called with the tensors,
// it returns the score_mod, which has them in scope. That is how FlexAttention
// takes a learned table, and the eager fallback calls the same function.
func (c *ctx) attentionFunction(kind string, n attnexpr.Node, heads float64) string {
	body := attnexpr.Python(n, heads)
	if strings.Contains(body, "t5_bucket(") {
		c.needsBucket = true
	}
	tables := attnexpr.Tables(n)
	key := kind + "|" + strings.Join(tables, ",") + "|" + body
	if name, ok := c.attnNames[key]; ok {
		return name
	}
	if c.attnNames == nil {
		c.attnNames = map[string]string{}
	}
	count := 1
	for k := range c.attnNames {
		if strings.HasPrefix(k, kind+"|") {
			count++
		}
	}
	name := fmt.Sprintf("%s_mod_%d", kind, count)
	c.attnNames[key] = name
	signature := "b, h, q_idx, kv_idx"
	if kind == "score" {
		signature = "score, " + signature
	}
	if len(tables) > 0 {
		c.attnFuncs = append(c.attnFuncs, fmt.Sprintf(
			"def %s(%s):\n    \"\"\"%s\"\"\"\n\n    def %s_mod(%s):\n        return %s\n\n    return %s_mod\n",
			name, strings.Join(tables, ", "), attnexpr.String(n), kind, signature, body, kind))
		return name
	}
	c.attnFuncs = append(c.attnFuncs, fmt.Sprintf("def %s(%s):\n    \"\"\"%s\"\"\"\n    return %s\n",
		name, signature, attnexpr.String(n), body))
	return name
}

func (c *ctx) className(base string) string {
	p := pascal(base)
	name := p
	if name == "" {
		name = "Block"
	}
	for n := 2; c.usedNames[name]; n++ {
		name = fmt.Sprintf("%s%d", p, n)
	}
	c.usedNames[name] = true
	return name
}

// order is a topological order of a graph's nodes.
func order(graph *ir.Graph) []ir.NodeDef {
	byID := make(map[string]*ir.NodeDef, len(graph.Nodes))
	for i := range graph.Nodes {
		byID[graph.Nodes[i].ID] = &graph.Nodes[i]
	}
	deps := make(map[string][]string, len(graph.Nodes))
	seen := map[string]bool{}
	for _, e := range graph.Edges {
		f, errF := ir.SplitEndpoint(e.From())
		t, errT := ir.SplitEndpoint(e.To())
		if errF != nil || errT != nil {
			continue
		}
		_, okF := byID[f.Node]
		_, okT := byID[t.Node]
		if !okF || !okT {
			continue
		}
		key := t.Node + "\x00" + f.Node
		if seen[key] {
			continue
		}
		seen[key] = true
		deps[t.Node] = append(deps[t.Node], f.Node)
	}

	var out []ir.NodeDef
	state := map[string]bool{}
	var visit func(id string)
	visit = func(id string) {
		if state[id] {
			return
		}
		state[id] = true
		for _, d := range deps[id] {
			visit(d)
		}
		if n, ok := byID[id]; ok {
			out = append(out, *n)
		}
	}
	for _, n := range graph.Nodes {
		visit(n.ID)
	}
	return out
}

type graphEmit struct {
	init    []string
	forward []string
	outputs map[string]string
	// outputOrder is the top-level output nodes in the order the graph lists
	// them, which is the order a model returns them in.
	outputOrder []string
}

type classInfo struct {
	name    string
	inputs  []string
	outputs []string
}

// portNames lists a boundary node's ports in the order the document wrote them.
//
// A Go map has no order, and here the order is the signature of the generated
// class: the arguments of its forward and the tuple it returns. It is recovered
// from the raw JSON the same way the symbol table's order is.
func portNames(node *ir.NodeDef, fallback string) []string {
	if node == nil {
		return []string{fallback}
	}
	names := orderedKeys(node.Params["ports"])
	if len(names) == 0 {
		return []string{fallback}
	}
	return names
}

// emitGraph emits the body of one graph. inputs maps the boundary port names to
// the variable names they arrive in.
func emitGraph(graph *ir.Graph, prefix string, inputs map[string]string, c *ctx) graphEmit {
	out := graphEmit{outputs: map[string]string{}}

	producers := map[string]string{}
	for _, e := range graph.Edges {
		if _, taken := producers[e.To()]; !taken {
			producers[e.To()] = e.From()
		}
	}

	// vars holds the variable carrying the value on a given output endpoint.
	vars := map[string]string{}
	valueOf := func(nodeID, port string) string {
		if v, ok := vars[nodeID+":"+port]; ok {
			return v
		}
		c.warn("%s: no value for port %q.", ir.JoinPath(prefix, nodeID), port)
		return "None"
	}
	inputVar := func(nodeID, port string) string {
		from, ok := producers[nodeID+":"+port]
		if !ok {
			c.warn("%s: input %q is not connected.", ir.JoinPath(prefix, nodeID), port)
			return "None"
		}
		f, err := ir.SplitEndpoint(from)
		if err != nil {
			return "None"
		}
		return valueOf(f.Node, f.Port)
	}

	for _, node := range order(graph) {
		path := ir.JoinPath(prefix, node.ID)
		def, known := c.cat[node.Type]
		if !known {
			c.warn("%s: unknown block type %q.", path, node.Type)
			continue
		}
		r := catalog.ResolveNodeParams(def, node.Params, c.symbols)
		attr := pyName(node.ID)
		outName := func(port string) string { return pyName(node.ID + "_" + port) }
		set := func(port, value string) { vars[node.ID+":"+port] = value }
		p := r.P

		switch node.Type {
		case "input":
			// A model with one input takes it as ids, as it always has; one
			// with several — an encoder-decoder's source and target — takes
			// each by its node's name.
			if name, ok := c.inputArgs[node.ID]; ok && prefix == "" {
				set("x", name)
			} else {
				set("x", "ids")
			}

		case "output":
			// Keyed by the node, not by the port: a design may have several
			// output nodes and they all call their port "x". Multi-token
			// prediction is the first thing that does — the model's own logits
			// and one more set per prediction depth — and keying by port meant
			// the last one silently won.
			out.outputs[node.ID] = inputVar(node.ID, "x")
			out.outputOrder = append(out.outputOrder, node.ID)

		case "boundary_in":
			for _, port := range orderedKeys(node.Params["ports"]) {
				v, ok := inputs[port]
				if !ok {
					v = "None"
				}
				set(port, v)
			}

		case "boundary_out":
			for _, port := range orderedKeys(node.Params["ports"]) {
				out.outputs[port] = inputVar(node.ID, port)
			}

		case "embedding":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Embedding(%s, %s)",
				attr, pyValue(p["vocab"]), pyValue(p["dim"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "ids")))
			set("y", outName("y"))

		case "pos_embedding":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Embedding(%s, %s)",
				attr, pyValue(p["max_seq"]), pyValue(p["dim"])))
			src := inputVar(node.ID, "x")
			out.forward = append(out.forward, fmt.Sprintf(
				"%s = %s + self.%s(torch.arange(%s.shape[1], device=%s.device))",
				outName("y"), src, attr, src, src))
			set("y", outName("y"))

		case "conv2d":
			groups := ""
			if r.Num("groups") > 1 {
				groups = ", groups=" + pyValue(p["groups"])
			}
			out.init = append(out.init, fmt.Sprintf(
				"self.%s = nn.Conv2d(%s, %s, kernel_size=%s, stride=%s, padding=%s%s, bias=%s)",
				attr, pyValue(p["in_channels"]), pyValue(p["out_channels"]),
				pyValue(p["kernel"]), pyValue(p["stride"]), pyValue(p["padding"]),
				groups, pyBool(p["bias"])))
			src := inputVar(node.ID, "x")
			out.forward = append(out.forward, fmt.Sprintf("%s = %s",
				outName("y"), activationOr(r.Str("act"), "identity", fmt.Sprintf("self.%s(%s)", attr, src))))
			set("y", outName("y"))

		case "maxpool2d":
			out.init = append(out.init, fmt.Sprintf(
				"self.%s = nn.MaxPool2d(kernel_size=%s, stride=%s, padding=%s)",
				attr, pyValue(p["kernel"]), pyValue(p["stride"]), pyValue(p["padding"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "shift":
			by := r.Num("by")
			if by == 0 {
				set("y", inputVar(node.ID, "x"))
				break
			}
			c.needsShift = true
			out.forward = append(out.forward, fmt.Sprintf("%s = shift_sequence(%s, %s)",
				outName("y"), inputVar(node.ID, "x"), pyNum(by)))
			set("y", outName("y"))

		case "gate":
			// Broadcast, not elementwise: the gate is one value per token and
			// PyTorch spreads it across the width on its own.
			out.forward = append(out.forward, fmt.Sprintf("%s = %s * %s",
				outName("y"), inputVar(node.ID, "g"), inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "mix":
			// Two scalars in one parameter, so a checkpoint carries them as one
			// tensor and the count is unambiguous.
			out.init = append(out.init, fmt.Sprintf(
				"self.%s = nn.Parameter(torch.tensor([1.0, 0.0]))", attr))
			out.forward = append(out.forward, fmt.Sprintf(
				"%s = self.%s[0] * %s + self.%s[1] * %s",
				outName("y"), attr, inputVar(node.ID, "a"), attr, inputVar(node.ID, "b")))
			set("y", outName("y"))

		case "diff_combine":
			// lambda's four vectors in one parameter: q1, k1, q2, k2. Drawn
			// from a narrow normal as the paper's are, not zeros: at zero the
			// dot products have no gradient and lambda never leaves its
			// starting value.
			d := pyValue(p["head_dim"])
			out.init = append(out.init, fmt.Sprintf(
				"self.%s = nn.Parameter(torch.randn(4, %s) * 0.1)", attr, d))
			lambda := outName("lambda")
			out.forward = append(out.forward,
				fmt.Sprintf("%s = torch.exp((self.%s[0] * self.%s[1]).sum().float()) - torch.exp((self.%s[2] * self.%s[3]).sum().float()) + %s",
					lambda, attr, attr, attr, attr, pyValue(p["lambda_init"])),
				fmt.Sprintf("%s = %s - %s.to(%s.dtype) * %s",
					outName("y"), inputVar(node.ID, "a"), lambda, inputVar(node.ID, "a"), inputVar(node.ID, "b")))
			set("y", outName("y"))

		case "attn_scores", "attn_values":
			// Grouped keys and values are repeated out to the query heads, which
			// is what an eager matmul has to do and a fused kernel does not.
			var left, right string
			if node.Type == "attn_scores" {
				left, right = inputVar(node.ID, "q"), inputVar(node.ID, "k")
			} else {
				left, right = inputVar(node.ID, "p"), inputVar(node.ID, "v")
			}
			if h, kv := r.Num("heads"), r.Num("kv_heads"); kv > 0 && h != kv {
				right = fmt.Sprintf("%s.repeat_interleave(%s, dim=1)", right, pyNum(h/kv))
			}
			if node.Type == "attn_scores" {
				out.forward = append(out.forward, fmt.Sprintf("%s = (%s @ %s.transpose(-2, -1)) * %s",
					outName("y"), left, right, jsToPrecision(1/math.Sqrt(r.Num("head_dim")), 12)))
			} else {
				out.forward = append(out.forward, fmt.Sprintf("%s = %s @ %s", outName("y"), left, right))
			}
			set("y", outName("y"))

		case "attn_softmax":
			x := inputVar(node.ID, "x")
			if r.Bool("causal") {
				x = fmt.Sprintf("%s.masked_fill(torch.ones(%s.shape[-2], %s.shape[-1], dtype=torch.bool, device=%s.device).triu(1), float(\"-inf\"))",
					x, x, x, x)
			}
			out.forward = append(out.forward, fmt.Sprintf("%s = torch.softmax(%s.float(), dim=-1).to(%s.dtype)",
				outName("y"), x, inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "head_mix":
			// The identity to begin with: talking heads starts as plain
			// attention and learns to talk.
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Parameter(torch.eye(%s))", attr, pyValue(p["heads"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = torch.einsum(\"bhqk,hg->bgqk\", %s, self.%s)",
				outName("y"), inputVar(node.ID, "x"), attr))
			set("y", outName("y"))

		case "position_bias":
			// The table is the parameter, and what flows along the wire is the
			// parameter itself: every attention that reads it reads the same
			// one, and its gradient is theirs added up.
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Parameter(torch.zeros(%s, %s))",
				attr, pyValue(p["buckets"]), pyValue(p["heads"])))
			set("table", "self."+attr)

		case "scale":
			out.forward = append(out.forward, fmt.Sprintf("%s = %s * %s",
				outName("y"), inputVar(node.ID, "x"), pyValue(p["by"])))
			set("y", outName("y"))

		case "flatten2d":
			out.forward = append(out.forward, fmt.Sprintf("%s = torch.flatten(%s, 1)",
				outName("y"), inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "learned_tokens":
			// A parameter, not a module: registered so it trains and moves with
			// the model, expanded to the batch at use. expand rather than repeat
			// keeps it a view, which is what the reference relies on.
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Parameter(torch.zeros(1, %s, %s))",
				attr, pyValue(p["count"]), pyValue(p["dim"])))
			// Expanded against the model's own input, which is the only tensor a
			// block with no inputs can learn the batch size from. `ids` is the
			// forward's parameter, whatever the design feeds it.
			tokens := "-1"
			if v, ok := p["tokens"]; ok && truthy(v) {
				tokens = pyValue(v)
			}
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s.expand(ids.shape[0], %s, -1)",
				outName("y"), attr, tokens))
			set("y", outName("y"))

		case "linear":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Linear(%s, %s, bias=%s)",
				attr, pyValue(p["in_features"]), pyValue(p["out_features"]), pyBool(p["bias"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "lm_head":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Linear(%s, %s, bias=%s)",
				attr, pyValue(p["dim"]), pyValue(p["vocab"]), pyBool(p["bias"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			if cap := r.Num("softcap"); cap != 0 {
				// Gemma's bound on the logits. It changes the loss, not the
				// shapes, so it is one line after the projection.
				out.forward = append(out.forward, fmt.Sprintf(
					"%s = torch.tanh(%s / %s) * %s",
					outName("y"), outName("y"), pyNum(cap), pyNum(cap)))
			}
			set("y", outName("y"))

		case "rmsnorm":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.RMSNorm(%s, eps=%s)",
				attr, pyValue(p["dim"]), pyValue(p["eps"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "layernorm":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.LayerNorm(%s, eps=%s, bias=%s)",
				attr, pyValue(p["dim"]), pyValue(p["eps"]), pyBool(p["bias"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "activation":
			out.forward = append(out.forward, fmt.Sprintf("%s = %s",
				outName("y"), activationOr(r.Str("kind"), "silu", inputVar(node.ID, "x"))))
			set("y", outName("y"))

		case "add":
			out.forward = append(out.forward, fmt.Sprintf("%s = %s + %s",
				outName("y"), inputVar(node.ID, "a"), inputVar(node.ID, "b")))
			set("y", outName("y"))

		case "mul":
			out.forward = append(out.forward, fmt.Sprintf("%s = %s * %s",
				outName("y"), inputVar(node.ID, "a"), inputVar(node.ID, "b")))
			set("y", outName("y"))

		case "rearrange":
			out.forward = append(out.forward,
				emitRearrange(r, c, inputVar(node.ID, "x"), outName("y"), path)...)
			set("y", outName("y"))

		case "rope":
			c.needsRope = true
			out.init = append(out.init, fmt.Sprintf("self.%s = RotaryEmbedding(%s, theta=%s)",
				attr, pyValue(p["head_dim"]), pyValue(p["theta"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "conv1d":
			src := inputVar(node.ID, "x")
			channels := pyValue(p["channels"])
			out.init = append(out.init, fmt.Sprintf(
				"self.%s = nn.Conv1d(%s, %s, %s, groups=%s, padding=%s, bias=%s)",
				attr, channels, channels, pyValue(p["kernel"]), channels,
				pyNum(r.Num("kernel")-1), pyBool(p["bias"])))
			// Convolve over time, then drop the padding the causal kernel added.
			out.forward = append(out.forward, fmt.Sprintf(
				"%s = self.%s(%s.transpose(1, 2))[..., : %s.shape[1]].transpose(1, 2)",
				outName("y"), attr, src, src))
			set("y", outName("y"))

		case "selective_scan":
			c.needsSelective = true
			out.init = append(out.init, fmt.Sprintf("self.%s = SelectiveScan(%s, %s)",
				attr, pyValue(p["d_inner"]), pyValue(p["state"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s, %s, %s, %s)",
				outName("y"), attr, inputVar(node.ID, "x"), inputVar(node.ID, "dt"),
				inputVar(node.ID, "b"), inputVar(node.ID, "c")))
			set("y", outName("y"))

		case "ssd_scan":
			c.needsSsd = true
			out.init = append(out.init, fmt.Sprintf("self.%s = SSDScan(%s, %s, %s, %s)",
				attr, pyValue(p["heads"]), pyValue(p["head_dim"]),
				pyValue(p["state"]), pyValue(p["groups"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s, %s)",
				outName("y"), attr, inputVar(node.ID, "xbc"), inputVar(node.ID, "dt")))
			set("y", outName("y"))

		case "gated_delta_scan":
			c.needsGdn = true
			vd := r.Num("v_head_dim")
			if vd == 0 {
				vd = r.Num("head_dim")
			}
			hv := r.Num("value_heads")
			if hv == 0 {
				hv = r.Num("heads")
			}
			out.init = append(out.init, fmt.Sprintf("self.%s = GatedDeltaScan(%s, %s, %s, %s)",
				attr, pyValue(p["heads"]), pyValue(p["head_dim"]), pyNum(vd), pyNum(hv)))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s, %s, %s, %s)",
				outName("y"), attr,
				inputVar(node.ID, "q"), inputVar(node.ID, "k"),
				inputVar(node.ID, "v"), inputVar(node.ID, "gates")))
			set("y", outName("y"))

		case "kv_latent_cache":
			// Analysis only: it marks what an inference engine would cache. In a
			// forward pass the compressed vector simply flows through.
			out.forward = append(out.forward, fmt.Sprintf("%s = %s  # cached latent",
				outName("y"), inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "split":
			// Sizes are expressions over the design symbols, not plain numbers.
			splitCtx := infer.EvalCtxFor(r, c.symbols)
			raw, _ := p["sizes"].([]any)
			sizes := make([]string, len(raw))
			bad := false
			for i, v := range raw {
				n, ok := evalSize(v, splitCtx)
				if !ok {
					bad = true
					sizes[i] = "NaN"
					continue
				}
				sizes[i] = pyNum(n)
			}
			if bad {
				c.warn("%s: could not evaluate the split widths.", path)
			}
			names := make([]string, len(sizes))
			for i := range sizes {
				names[i] = outName(fmt.Sprintf("y%d", i))
			}
			out.forward = append(out.forward, fmt.Sprintf("%s = torch.split(%s, [%s], dim=-1)",
				strings.Join(names, ", "), inputVar(node.ID, "x"), strings.Join(sizes, ", ")))
			for i := range sizes {
				set(fmt.Sprintf("y%d", i), names[i])
			}

		case "concat":
			raw, _ := p["sizes"].([]any)
			parts := make([]string, len(raw))
			for i := range raw {
				parts[i] = inputVar(node.ID, fmt.Sprintf("y%d", i))
			}
			axis := "-1"
			if v, ok := asNumber(p["axis"]); ok {
				axis = pyNum(v)
			}
			out.forward = append(out.forward, fmt.Sprintf("%s = torch.cat([%s], dim=%s)",
				outName("y"), strings.Join(parts, ", "), axis))
			set("y", outName("y"))

		case "expand_heads":
			src := inputVar(node.ID, "x")
			out.forward = append(out.forward, fmt.Sprintf("%s = %s.unsqueeze(1).expand(-1, %s, -1, -1)",
				outName("y"), src, pyValue(p["heads"])))
			set("y", outName("y"))

		case "sdpa":
			q, k, v := inputVar(node.ID, "q"), inputVar(node.ID, "k"), inputVar(node.ID, "v")
			gqa := ""
			if r.Num("heads") != r.Num("kv_heads") {
				gqa = ", enable_gqa=True"
			}
			// A narrower value head means the kernel cannot infer the scale from
			// the query width, so it is passed explicitly.
			scale := ""
			if vd := r.Num("v_head_dim"); vd != 0 && vd != r.Num("head_dim") {
				scale = ", scale=" + jsToPrecision(1/math.Sqrt(r.Num("head_dim")), 12)
			}
			// A design that says what the scale is — T5, whose attention is
			// unscaled — is taken at its word.
			if s, ok := p["scale"].(float64); ok {
				scale = ", scale=" + pyNum(s)
			}
			w := r.Num("window")
			cap := r.Num("logit_softcap")
			a := catalog.AttentionOf(r)
			switch {
			case a.Flex():
				// Causal, the window and the cap are folded into the two
				// functions, so the kernel is told everything once.
				c.needsExpression = true
				args := []string{q, k, v}
				if a.Sinks {
					// Zero to begin with: a sink at zero adds one to the
					// denominator, which is a start rather than a choice.
					out.init = append(out.init, fmt.Sprintf(
						"self.%s_sinks = nn.Parameter(torch.zeros(%s))", attr, pyNum(a.Heads)))
					args = append(args, fmt.Sprintf("sinks=self.%s_sinks", attr))
				}
				if m := a.Mask(); m != nil {
					args = append(args, "mask_mod="+c.attentionFunction("mask", m, a.Heads))
					if attnexpr.Uses(m, "h") {
						args = append(args, "mask_heads=True")
					}
					if attnexpr.Uses(m, "b") {
						args = append(args, "mask_batch=True")
					}
				}
				if s := a.Score(); s != nil {
					fn := c.attentionFunction("score", s, a.Heads)
					if tables := attnexpr.Tables(s); len(tables) > 0 {
						vars := make([]string, len(tables))
						for i, name := range tables {
							vars[i] = inputVar(node.ID, name)
						}
						fn += "(" + strings.Join(vars, ", ") + ")"
					}
					args = append(args, "score_mod="+fn)
				}
				if scale != "" {
					args = append(args, strings.TrimPrefix(scale, ", "))
				}
				out.forward = append(out.forward, fmt.Sprintf("%s = expression_attention(%s)",
					outName("y"), strings.Join(args, ", ")))
			case cap != 0 || w > 0:
				// A window or a cap is what the plain kernel cannot skip or
				// apply, so both go through the helper that finds one that can.
				c.needsFused = true
				args := []string{q, k, v}
				if w > 0 {
					args = append(args, "window="+pyNum(w))
				}
				if cap != 0 {
					args = append(args, "softcap="+pyNum(cap))
				}
				if w == 0 {
					args = append(args, "causal="+pyBool(p["causal"]))
				}
				if scale != "" {
					args = append(args, strings.TrimPrefix(scale, ", "))
				}
				if gqa != "" {
					args = append(args, "enable_gqa=True")
				}
				out.forward = append(out.forward, fmt.Sprintf("%s = fused_attention(%s)",
					outName("y"), strings.Join(args, ", ")))
			default:
				out.forward = append(out.forward, fmt.Sprintf(
					"%s = F.scaled_dot_product_attention(%s, %s, %s, is_causal=%s%s%s)",
					outName("y"), q, k, v, pyBool(p["causal"]), gqa, scale))
			}
			set("y", outName("y"))

		default:
			switch {
			case catalog.IsContainer(def) && node.Graph != nil:
				inner := emitClassForGraph(node.Graph, path, "Layer", c, nil)
				out.init = append(out.init, fmt.Sprintf("self.%s = nn.ModuleList([%s() for _ in range(%s)])",
					attr, inner.name, pyValue(p["count"])))
				// An input the layer gives back is carried from copy to copy;
				// one it does not — a decoder's view of the encoder — is handed
				// to every copy as it is.
				returned := map[string]bool{}
				for _, port := range inner.outputs {
					returned[port] = true
				}
				var seed, loopVars, args []string
				for _, port := range inner.inputs {
					if !returned[port] {
						args = append(args, inputVar(node.ID, port))
						continue
					}
					seed = append(seed, inputVar(node.ID, port))
					loopVars = append(loopVars, outName(port))
					args = append(args, outName(port))
				}
				out.forward = append(out.forward,
					strings.Join(loopVars, ", ")+" = "+strings.Join(seed, ", "),
					"for layer in self."+attr+":",
					"    "+strings.Join(loopVars, ", ")+" = layer("+strings.Join(args, ", ")+")")
				for _, port := range inner.outputs {
					set(port, outName(port))
				}

			case catalog.IsComposite(def) && def.Type == "moe_layer":
				inner := emitMoeClass(r, path, c)
				out.init = append(out.init, fmt.Sprintf("self.%s = %s()", attr, inner.name))
				out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
					outName("y"), attr, inputVar(node.ID, "x")))
				set("y", outName("y"))

			case catalog.IsComposite(def):
				exp, ok := catalog.Expand(def, r.RawFull, r)
				if !ok {
					c.warn("%s: expansion failed: %s has no expansion.", path, node.Type)
					break
				}
				inner := emitClassForGraph(&ir.Graph{Nodes: exp.Nodes, Edges: exp.Edges}, path, def.Type, c, r)
				args := make([]string, len(inner.inputs))
				for i, port := range inner.inputs {
					args[i] = inputVar(node.ID, port)
				}
				rets := make([]string, len(inner.outputs))
				for i, port := range inner.outputs {
					rets[i] = outName(port)
				}
				out.init = append(out.init, fmt.Sprintf("self.%s = %s()", attr, inner.name))
				out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
					strings.Join(rets, ", "), attr, strings.Join(args, ", ")))
				for _, port := range inner.outputs {
					set(port, outName(port))
				}

			default:
				c.warn("%s: no code generator for block type %q.", path, node.Type)
				ports := catalog.Ports{}
				if !catalog.IsContainer(def) {
					ports = catalog.PortsOf(def, r)
				}
				for port := range ports.Out {
					set(port, "None")
				}
			}
		}
	}

	return out
}

// emitClassForGraph generates, or reuses, a class for a graph and returns its
// name and signature.
func emitClassForGraph(graph *ir.Graph, path, baseName string, c *ctx, resolved *catalog.Resolved) classInfo {
	var bIn, bOut *ir.NodeDef
	for i := range graph.Nodes {
		switch graph.Nodes[i].Type {
		case "boundary_in":
			if bIn == nil {
				bIn = &graph.Nodes[i]
			}
		case "boundary_out":
			if bOut == nil {
				bOut = &graph.Nodes[i]
			}
		}
	}
	inputs := portNames(bIn, "x")
	outputs := portNames(bOut, "y")

	// Two blocks of the same type with the same numbers share a class.
	var params any
	if resolved != nil {
		params = resolved.P
	}
	key := dedupKey(map[string]any{"baseName": baseName, "params": params, "graph": graph})
	if existing, ok := c.byKey[key]; ok {
		return classInfo{name: existing, inputs: inputs, outputs: outputs}
	}

	name := c.className(baseName)
	c.byKey[key] = name

	argMap := make(map[string]string, len(inputs))
	for _, port := range inputs {
		argMap[port] = pyName(port)
	}
	body := emitGraph(graph, path, argMap, c)

	returned := make([]string, len(outputs))
	args := make([]string, len(inputs))
	for i, port := range outputs {
		if v, ok := body.outputs[port]; ok {
			returned[i] = v
		} else {
			returned[i] = "None"
		}
	}
	for i, port := range inputs {
		args[i] = pyName(port)
	}

	lines := []string{fmt.Sprintf("class %s(nn.Module):", name)}
	if resolved != nil {
		if summary := paramSummary(resolved); summary != "" {
			lines = append(lines, fmt.Sprintf(`    """%s: %s"""`, baseName, summary))
		}
	}
	lines = append(lines, "", "    def __init__(self):", "        super().__init__()")
	for _, l := range body.init {
		lines = append(lines, "        "+l)
	}
	if len(body.init) == 0 {
		lines = append(lines, "        pass")
	}
	lines = append(lines, "", fmt.Sprintf("    def forward(self, %s):", strings.Join(args, ", ")))
	for _, l := range body.forward {
		lines = append(lines, "        "+l)
	}
	lines = append(lines, "        return "+strings.Join(returned, ", "), "")

	c.classes = append(c.classes, emitted{name: name, code: strings.Join(lines, "\n")})
	return classInfo{name: name, inputs: inputs, outputs: outputs}
}

// emitMoeClass is the mixture-of-experts dispatch.
//
// This is the one block whose runtime behaviour is not a dataflow graph of our
// primitives: routing sends different tokens to different modules. The expert
// body still comes from the generic emitter, so only the dispatch is bespoke.
func emitMoeClass(r *catalog.Resolved, path string, c *ctx) classInfo {
	raw := r.RawFull
	d := catalog.Ex(raw["d_model"], "0")
	fe := catalog.Ex(raw["expert_hidden"], "0")
	shared := r.Num("shared_experts")

	mlpDef := catalog.Builtin["gated_mlp"]
	expertParams := map[string]any{
		"d_model": d, "hidden": fe, "act": r.P["act"], "bias": r.P["bias"] == true,
	}
	expertResolved := catalog.ResolveNodeParams(mlpDef, expertParams, c.symbols)
	expertGraph, _ := catalog.Expand(mlpDef, expertResolved.RawFull, expertResolved)
	expert := emitClassForGraph(
		&ir.Graph{Nodes: expertGraph.Nodes, Edges: expertGraph.Edges},
		path+"/expert", "Expert", c, expertResolved)

	sharedClass := ""
	if shared > 0 {
		sharedParams := map[string]any{
			"d_model": d, "hidden": fmt.Sprintf("%s*%s", pyNum(shared), fe),
			"act": r.P["act"], "bias": r.P["bias"] == true,
		}
		sharedResolved := catalog.ResolveNodeParams(mlpDef, sharedParams, c.symbols)
		sharedGraph, _ := catalog.Expand(mlpDef, sharedResolved.RawFull, sharedResolved)
		sharedClass = emitClassForGraph(
			&ir.Graph{Nodes: sharedGraph.Nodes, Edges: sharedGraph.Edges},
			path+"/shared", "SharedExpert", c, sharedResolved).name
	}

	var sharedKey any
	if sharedClass != "" {
		sharedKey = sharedClass
	}
	key := dedupKey(map[string]any{"moe": r.P, "expert": expert.name, "sharedClass": sharedKey})
	if existing, ok := c.byKey[key]; ok {
		return classInfo{name: existing, inputs: []string{"x"}, outputs: []string{"y"}}
	}

	name := c.className("MoeLayer")
	c.byKey[key] = name

	experts, topK := r.Num("experts"), r.Num("top_k")
	doc := fmt.Sprintf("%s experts, top %s per token", pyNum(experts), pyNum(topK))
	if shared > 0 {
		doc += fmt.Sprintf(", plus %s shared expert(s)", pyNum(shared))
	}
	doc += "."
	if c.moeDispatch == "dense" {
		doc += fmt.Sprintf(" Dense dispatch: traceable, but %sx the work.",
			pyNum(jsRound(experts/topK)))
	}

	lines := []string{
		fmt.Sprintf("class %s(nn.Module):", name),
		fmt.Sprintf(`    """%s"""`, doc),
		"",
		"    def __init__(self):",
		"        super().__init__()",
		fmt.Sprintf("        self.top_k = %s", pyNum(topK)),
		fmt.Sprintf("        self.router = nn.Linear(%s, %s, bias=%s)",
			pyValue(r.P["d_model"]), pyNum(experts), pyBool(r.P["router_bias"])),
		fmt.Sprintf("        self.experts = nn.ModuleList([%s() for _ in range(%s)])",
			expert.name, pyNum(experts)),
	}
	if sharedClass != "" {
		lines = append(lines, fmt.Sprintf("        self.shared = %s()", sharedClass))
		if r.Bool("shared_expert_gate") {
			// One learned direction, not a matrix: Qwen's is [1, d_model], and
			// it weighs 2048 where the expert beside it weighs three million.
			lines = append(lines, fmt.Sprintf(
				"        self.shared_gate = nn.Linear(%s, 1, bias=False)", pyValue(r.P["d_model"])))
		}
	}
	lines = append(lines,
		"",
		"    def forward(self, x):",
		"        shape = x.shape",
		fmt.Sprintf("        flat = x.reshape(-1, %s)", pyValue(r.P["d_model"])),
		"        scores = F.softmax(self.router(flat).float(), dim=-1)",
		"        weight, index = torch.topk(scores, self.top_k, dim=-1)")
	if r.P["normalize"] != false {
		lines = append(lines, "        weight = weight / weight.sum(dim=-1, keepdim=True)")
	}
	lines = append(lines, "        weight = weight.to(x.dtype)")
	if c.moeDispatch == "dense" {
		// Every expert sees every token, weighted by whether it was chosen. Same
		// result, no data-dependent shapes, so torch.export can trace it.
		lines = append(lines,
			fmt.Sprintf("        gate = torch.zeros(flat.shape[0], %s, device=x.device, dtype=x.dtype)",
				pyNum(experts)),
			"        gate = gate.scatter(1, index, weight)",
			"        out = torch.zeros_like(flat)",
			"        for e, expert in enumerate(self.experts):",
			"            out = out + expert(flat) * gate[:, e : e + 1]",
			"        y = out.reshape(shape)")
	} else {
		lines = append(lines,
			"        out = torch.zeros_like(flat)",
			"        for e, expert in enumerate(self.experts):",
			"            rows, slot = (index == e).nonzero(as_tuple=True)",
			"            if rows.numel() == 0:",
			"                continue",
			"            out.index_add_(0, rows, expert(flat[rows]) * weight[rows, slot, None])",
			"        y = out.reshape(shape)")
	}
	if sharedClass != "" {
		if r.Bool("shared_expert_gate") {
			lines = append(lines, "        y = y + torch.sigmoid(self.shared_gate(x)) * self.shared(x)")
		} else {
			lines = append(lines, "        y = y + self.shared(x)")
		}
	}
	lines = append(lines, "        return y", "")

	c.classes = append(c.classes, emitted{name: name, code: strings.Join(lines, "\n")})
	return classInfo{name: name, inputs: []string{"x"}, outputs: []string{"y"}}
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

// totalLayers is the number of stacked layers, used to scale the residual
// initialization by depth.
func totalLayers(flat *analysis.FlatResult) float64 {
	n := 0.0
	for _, rep := range flat.Repeats {
		if rep.Type == "repeat" {
			n += rep.Count
		}
	}
	if n == 0 {
		return 1
	}
	return n
}

// GenerateTorch emits a PyTorch module for a design.
func GenerateTorch(doc *ir.Doc, options Options) *Generated {
	symbols := ir.ResolveSymbols(doc)
	dispatch := options.MoeDispatch
	if dispatch == "" {
		dispatch = "sparse"
	}
	c := &ctx{
		doc:         doc,
		cat:         catalog.Of(doc),
		symbols:     symbols,
		warnings:    append([]string{}, symbols.Errors...),
		byKey:       map[string]string{},
		usedNames:   map[string]bool{},
		moeDispatch: dispatch,
	}

	modelName := options.ClassName
	if modelName == "" {
		base := doc.Meta.Name
		if base == "" {
			base = "Model"
		}
		modelName = c.className(base)
	}

	var inputNodes []string
	for _, n := range doc.Graph.Nodes {
		if n.Type == "input" {
			inputNodes = append(inputNodes, n.ID)
		}
	}
	modelArgs := "ids"
	if len(inputNodes) > 1 {
		c.inputArgs = map[string]string{}
		names := make([]string, len(inputNodes))
		for i, id := range inputNodes {
			names[i] = pyName(id)
			c.inputArgs[id] = names[i]
		}
		modelArgs = strings.Join(names, ", ")
	}
	body := emitGraph(&doc.Graph, "", map[string]string{}, c)

	// Weight tying, which the graph expresses as a flag rather than an edge.
	//
	// Every tied head, not the first one: a design with multi-token prediction
	// has one output head per prediction depth and they all share the
	// embedding. Counting them as shared and then emitting them untied is a
	// model that does not have the parameters the analysis said it does, which
	// is exactly what `verify` is for and exactly what it caught.
	var tie []string
	var embed *ir.NodeDef
	for i := range doc.Graph.Nodes {
		if doc.Graph.Nodes[i].Type == "embedding" && embed == nil {
			embed = &doc.Graph.Nodes[i]
		}
	}
	if embed != nil {
		for i := range doc.Graph.Nodes {
			head := &doc.Graph.Nodes[i]
			if head.Type != "lm_head" {
				continue
			}
			r := catalog.ResolveNodeParams(c.cat[head.Type], head.Params, symbols)
			if r.P["tied"] == true {
				tie = append(tie, fmt.Sprintf("self.%s.weight = self.%s.weight",
					pyName(head.ID), pyName(embed.ID)))
			}
		}
	}

	flatResult := analysis.Flatten(doc, symbols)
	params := analysis.CountParams(flatResult)

	header := []string{
		fmt.Sprintf(`"""%s — generated by TensorCAD.`, doc.Meta.Name),
		"",
		"Do not edit by hand: change the design and regenerate.",
		"",
		"Symbols:",
	}
	for _, n := range symbols.Order {
		if symbols.Runtime[n] {
			continue
		}
		header = append(header, fmt.Sprintf("    %s = %s", n, analysis.JSNumber(symbols.Values[n])))
	}
	header = append(header,
		"",
		fmt.Sprintf("Parameters: %s", localeInt(params.Total)),
		`"""`,
		"",
		"import torch",
		"import torch.nn as nn",
		"import torch.nn.functional as F",
		"",
		"")

	var helpers []string
	if c.needsRope {
		helpers = append(helpers, helperRope, "")
	}
	if c.needsFused {
		helpers = append(helpers, helperFused, "")
	}
	if c.needsBucket {
		helpers = append(helpers, helperBucket, "")
	}
	if c.needsExpression {
		helpers = append(helpers, helperExpression, "")
		for _, f := range c.attnFuncs {
			helpers = append(helpers, f, "")
		}
	}
	if c.needsShift {
		helpers = append(helpers, helperShift, "")
	}
	if c.needsSelective {
		helpers = append(helpers, helperSelective, "")
	}
	if c.needsSsd {
		helpers = append(helpers, helperSsd, "")
	}
	if c.needsGdn {
		helpers = append(helpers, helperGatedDelta, "")
	}

	modelLines := []string{
		fmt.Sprintf("class %s(nn.Module):", modelName),
		"",
		"    def __init__(self):",
		"        super().__init__()",
	}
	for _, l := range body.init {
		modelLines = append(modelLines, "        "+l)
	}
	for _, l := range tie {
		modelLines = append(modelLines, "        "+l)
	}

	// Weight initialization. Residual projections are scaled down by the depth
	// so the residual stream does not grow as layers are added, which is what
	// GPT-2 does and what every model since has kept.
	initStd := 0.02
	if options.InitStd != nil {
		initStd = *options.InitStd
	}
	var initLines []string
	// The blank line belongs to the forward that follows, whether or not there
	// is an init_weights between them.
	modelLines = append(modelLines, "")
	if initStd > 0 {
		residualStd := initStd / math.Sqrt(2*math.Max(1, totalLayers(flatResult)))
		std := analysis.JSNumber(initStd)
		modelLines = append(modelLines,
			"    @torch.no_grad()",
			"    def init_weights(self):",
			fmt.Sprintf(`        """Normal(0, %s), with residual projections scaled by depth.`, std),
			"",
			"        PyTorch's defaults leave nn.Embedding at a unit normal, which starts",
			"        a language model near a cross-entropy of a few hundred rather than",
			"        ln(vocab). Call this after moving the model to its device.",
			`        """`,
			"        for module in self.modules():",
			"            if isinstance(module, (nn.Linear, nn.Embedding)):",
			fmt.Sprintf("                nn.init.normal_(module.weight, mean=0.0, std=%s)", std),
			`                bias = getattr(module, "bias", None)`,
			"                if bias is not None:",
			"                    nn.init.zeros_(bias)",
			"        for name, param in self.named_parameters():",
			"            if name.endswith(RESIDUAL_PROJECTIONS):",
			fmt.Sprintf("                nn.init.normal_(param, mean=0.0, std=%s)", jsToPrecision(residualStd, 8)),
			"        return self",
			"")
		initLines = append(initLines,
			"",
			"# Projections that write back into the residual stream. Their initial scale",
			"# is divided by sqrt(2 * layers) so depth does not inflate the stream.",
			`RESIDUAL_PROJECTIONS = ("o_proj.weight", "down.weight", "out_proj.weight")`,
			"")
	}

	modelLines = append(modelLines, "    def forward(self, "+modelArgs+"):")
	for _, l := range body.forward {
		modelLines = append(modelLines, "        "+l)
	}
	// One output returns a tensor; several return a tuple, in the order the
	// graph lists them.
	returned := "None"
	switch len(body.outputOrder) {
	case 0:
	case 1:
		returned = body.outputs[body.outputOrder[0]]
	default:
		parts := make([]string, len(body.outputOrder))
		for i, id := range body.outputOrder {
			parts[i] = body.outputs[id]
		}
		returned = strings.Join(parts, ", ")
	}
	modelLines = append(modelLines, "        return "+returned, "")

	var smoke []string
	if !options.NoSmokeTest {
		smoke = append(smoke,
			"",
			`if __name__ == "__main__":`,
			fmt.Sprintf("    expected = %s", analysis.JSNumber(params.Total)),
			`    with torch.device("meta"):`,
			fmt.Sprintf("        model = %s()", modelName),
			"    actual = sum(p.numel() for p in model.parameters())",
			`    print(f"parameters: {actual:,} (design says {expected:,})")`,
			`    assert actual == expected, f"parameter count differs: {actual} vs {expected}"`,
			"")
	}

	classCode := make([]string, len(c.classes))
	for i, cl := range c.classes {
		classCode[i] = cl.code
	}

	sections := []string{
		strings.Join(header, "\n"),
		strings.Join(initLines, "\n"),
		strings.Join(helpers, "\n"),
		strings.Join(classCode, "\n"),
		strings.Join(modelLines, "\n"),
		strings.Join(smoke, "\n"),
	}
	var kept []string
	for _, s := range sections {
		if strings.TrimSpace(s) != "" {
			kept = append(kept, s)
		}
	}
	contents := strings.Join(kept, "\n")
	if !strings.HasSuffix(contents, "\n") {
		contents += "\n"
	}

	design, _ := json.MarshalIndent(doc, "", "  ")

	return &Generated{
		Files: []File{
			{Path: "model.py", Contents: contents},
			{Path: "design.tensorcad.json", Contents: string(design) + "\n"},
		},
		Warnings: c.warnings,
	}
}

func asNumber(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	}
	return 0, false
}

// truthy is JavaScript's notion of it, which the emitter leans on where the
// TypeScript writes `x ? x : -1`.
func truthy(v any) bool {
	switch n := v.(type) {
	case nil:
		return false
	case bool:
		return n
	case float64:
		return n != 0 && !math.IsNaN(n)
	case int:
		return n != 0
	case string:
		return n != ""
	}
	return true
}

// jsRound rounds half towards positive infinity, as Math.round does.
func jsRound(v float64) float64 { return math.Floor(v + 0.5) }

func evalSize(v any, ctx shapes.EvalCtx) (float64, bool) {
	if n, ok := asNumber(v); ok {
		return n, true
	}
	s, ok := v.(string)
	if !ok {
		return 0, false
	}
	sym, err := shapes.EvalExpr(s, ctx)
	if err != nil {
		return 0, false
	}
	n, ok := sym.ToNumber(ctx.Values)
	return n, ok
}
