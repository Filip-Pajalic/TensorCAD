// Package golden holds the questions the engine is checked against, and writes
// down its answers.
//
// The answers live in testdata as JSON. They started as a record of what the
// TypeScript engine said, which is how the Go port was proved correct; now that
// the TypeScript is gone they are a record of what this engine said last time,
// and a diff in one is either a bug being fixed or a behaviour being changed.
// Both want to be seen in review, which is why they are files rather than
// assertions spread through the tests.
//
// The questions are here rather than in the tests so the generator and the
// tests cannot ask different ones.
package golden

import (
	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/codegen"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/plan"
	"github.com/tensorcad/core/scale"
)

// Env is the symbol environment the expression and pattern cases evaluate in.
var Env = map[string]float64{
	"B": 4, "T": 2048, "D": 4096, "H": 32, "Hkv": 8,
	"dh": 128, "F": 14336, "V": 128256, "L": 32,
}

// Expressions are the corners of the algebra.
//
// A symbol table only exercises the paths its presets happen to take. These are
// the rest: rational coefficients that do not divide, exact monomial division,
// eager folding of functions, and the printed form of a polynomial — which is
// what a shape label on the canvas says, so a divergence there is visible
// rather than theoretical.
var Expressions = []string{
	"D", "-D", "D + 1", "4*D", "D*4", "H*dh", "B*T*D",
	"D/H", "D/4", "D/3", "1/3*D", "1.3*8/3*D",
	"ceil_mult(1.3*8/3*D, 1024)", "(H + 2*Hkv)*dh", "H*dh - D",
	"D^2", "2^10", "(D + D)/2", "D*dh/dh", "B*T*(H*dh)/H",
	"floor(D/H)", "ceil(D/3)", "round(D/3)", "min(D, F)", "max(D, F)",
	"floor_mult(F, 1024)", "round_mult(D*2/3, 128)",
	"log2(D)", "sqrt(D)", "abs(0 - D)",
	"1_000 + D", "1e3 + D", "0.5*D", "D - D", "3*D - 2*D",
}

// ExpressionErrors are the ones that must be refused, and the message matters:
// it is what a person reads out of the inspector.
var ExpressionErrors = []string{
	"D +", "D $ 1", "nope", "floor(D, 2)", "notafunction(D)",
	"D^(0 - 1)", "D/0", "B/T",
}

// Patterns are where a shape stops being a string and becomes something the
// engine can disagree with: groups, ellipses that bind nothing, ellipses that
// bind three dimensions.
var Patterns = []string{
	"B T D", "B T (H dh)", "... D", "...", "B H T dh",
	"... 2*D", "B (H dh) T", "B 3 224 224", "B T (H dh) D",
}

// PatternErrors are the malformed ones.
var PatternErrors = []string{"B T (H dh", "B () D", "... T ... D"}

// Match is an actual shape, written as a pattern and instantiated, checked
// against a pattern.
type Match struct{ Actual, Pattern string }

// Matches include the mismatches, because the message a mismatch produces is
// what a person reads off the canvas when a design is wrong.
var Matches = []Match{
	{"B T D", "B T D"},
	{"B T D", "... D"},
	{"B T D", "B T (H dh)"},
	{"B H T dh", "... dh"},
	{"B T D", "B T D D"},
	{"B T D", "B H T dh"},
	{"D", "... D"},
	{"B T F", "... D"},
}

// BlockCase is one block at one set of parameters.
type BlockCase struct {
	Type   string
	Params map[string]any
}

// PrimitiveCtx is the operating point the primitive formulas are evaluated at.
var PrimitiveCtx = catalog.AnalysisCtx{T: 4096, B: 1, Bytes: 2, Flash: true}

// PrimitiveSymbols is the symbol table the primitive cases resolve against.
func PrimitiveSymbols() *ir.SymbolTable {
	return ir.ResolveSymbols(&ir.Doc{
		Version: ir.DocVersion,
		Symbols: map[string]ir.SymbolDef{
			"B":   {Kind: "literal", Number: 4, HasNumber: true},
			"T":   {Kind: "literal", Number: 2048, HasNumber: true},
			"D":   {Kind: "literal", Number: 4096, HasNumber: true},
			"H":   {Kind: "literal", Number: 32, HasNumber: true},
			"Hkv": {Kind: "literal", Number: 8, HasNumber: true},
			"dh":  {Kind: "literal", Number: 128, HasNumber: true},
			"F":   {Kind: "literal", Number: 14336, HasNumber: true},
			"V":   {Kind: "literal", Number: 128256, HasNumber: true},
			"L":   {Kind: "literal", Number: 32, HasNumber: true},
			"Tc":  {Kind: "literal", Number: 218, HasNumber: true},
			"Dp":  {Kind: "literal", Number: 384, HasNumber: true},
		},
	})
}

// Primitives walks the catalog at a fixed set of parameters, because a preset
// only exercises the blocks its architecture happens to use, at the sizes that
// architecture happens to pick.
var Primitives = []BlockCase{
	{"input", map[string]any{"shape": "B T", "dtype": "int64"}},
	{"output", map[string]any{}},
	{"boundary_in", map[string]any{"ports": map[string]any{"x": "B T D"}}},
	{"boundary_out", map[string]any{"ports": map[string]any{"x": "B T D"}}},
	{"embedding", map[string]any{"vocab": 128256.0, "dim": 4096.0}},
	{"pos_embedding", map[string]any{"max_seq": 1024.0, "dim": 768.0}},
	{"learned_tokens", map[string]any{"count": 1.0, "dim": 384.0, "tokens": 0.0}},
	{"learned_tokens", map[string]any{"count": 1.0, "dim": 384.0, "tokens": 256.0}},
	{"linear", map[string]any{"in_features": 4096.0, "out_features": 14336.0, "bias": false}},
	{"linear", map[string]any{"in_features": 4096.0, "out_features": 14336.0, "bias": true}},
	{"lm_head", map[string]any{"vocab": 128256.0, "dim": 4096.0, "tied": false, "bias": false}},
	{"lm_head", map[string]any{"vocab": 128256.0, "dim": 4096.0, "tied": true, "bias": false}},
	{"conv2d", map[string]any{"in_channels": 3.0, "out_channels": 64.0, "kernel": 11.0,
		"stride": 4.0, "padding": 2.0, "in_h": 224.0, "in_w": 224.0, "act": "relu"}},
	{"conv2d", map[string]any{"in_channels": 192.0, "out_channels": 384.0, "kernel": 3.0,
		"stride": 1.0, "padding": 1.0, "in_h": 13.0, "in_w": 13.0, "act": "relu"}},
	{"maxpool2d", map[string]any{"channels": 64.0, "kernel": 3.0, "stride": 2.0, "in_h": 55.0, "in_w": 55.0}},
	{"flatten2d", map[string]any{"channels": 256.0, "in_h": 6.0, "in_w": 6.0}},
	{"rmsnorm", map[string]any{"dim": 4096.0}},
	{"rmsnorm", map[string]any{"dim": 4096.0, "scale": false}},
	{"layernorm", map[string]any{"dim": 768.0, "bias": true}},
	{"layernorm", map[string]any{"dim": 768.0, "bias": false}},
	{"activation", map[string]any{"kind": "silu", "dim": 14336.0}},
	{"activation", map[string]any{"kind": "gelu", "dim": 3072.0}},
	{"add", map[string]any{"dim": 4096.0}},
	{"mul", map[string]any{"dim": 14336.0}},
	{"rearrange", map[string]any{"from": "B T (H dh)", "to": "B H T dh"}},
	{"rope", map[string]any{"heads": 32.0, "head_dim": 128.0, "theta": 500000.0}},
	{"rope", map[string]any{"heads": 32.0, "head_dim": 127.0, "theta": 10000.0}},
	{"sdpa", map[string]any{"heads": 32.0, "kv_heads": 8.0, "head_dim": 128.0, "causal": true}},
	{"sdpa", map[string]any{"heads": 32.0, "kv_heads": 8.0, "head_dim": 128.0, "causal": false}},
	{"sdpa", map[string]any{"heads": 32.0, "kv_heads": 8.0, "head_dim": 128.0, "window": 4096.0}},
	{"sdpa", map[string]any{"heads": 32.0, "kv_heads": 7.0, "head_dim": 128.0}},
	{"sdpa", map[string]any{"heads": 32.0, "kv_heads": 8.0, "head_dim": 128.0, "cache": false}},
	{"topk_router", map[string]any{"d_model": 7168.0, "experts": 256.0, "top_k": 8.0, "bias": true}},
	{"topk_router", map[string]any{"d_model": 7168.0, "experts": 8.0, "top_k": 9.0}},
	{"weighted_sum", map[string]any{"dim": 4096.0, "n": 8.0}},
	{"split", map[string]any{"from": "B T D", "sizes": []any{512.0, 64.0}}},
	{"concat", map[string]any{"to": "B T D", "sizes": []any{512.0, 64.0}}},
	{"concat", map[string]any{"to": "B T Dp", "sizes": []any{"Tc", "T-Tc"}, "axis": 1.0}},
	{"expand_heads", map[string]any{"heads": 32.0, "dim": 64.0}},
	{"kv_latent_cache", map[string]any{"dim": 576.0}},
	{"conv1d", map[string]any{"channels": 8192.0, "kernel": 4.0, "bias": true}},
	{"ssd_scan", map[string]any{"d_inner": 8192.0, "heads": 128.0, "head_dim": 64.0,
		"state": 128.0, "groups": 8.0, "xbc_width": 10240.0}},
	{"ssd_scan", map[string]any{"d_inner": 8192.0, "heads": 128.0, "head_dim": 65.0,
		"state": 128.0, "groups": 7.0, "xbc_width": 10240.0}},
}

// Composites are pinned as the subgraph they stand for, node for node.
//
// A composite is nothing but that subgraph, so it is the specification.
// Comparing parameter totals would let a wrong expansion pass whenever two
// wrong numbers happened to cancel; comparing the graph cannot.
var Composites = []BlockCase{
	{"gqa_attention", map[string]any{"d_model": "D", "heads": "H", "kv_heads": "Hkv", "head_dim": "dh"}},
	{"gqa_attention", map[string]any{"d_model": "D", "heads": "H", "kv_heads": "Hkv", "head_dim": "dh",
		"bias": true, "o_bias": false}},
	{"gqa_attention", map[string]any{"d_model": "D", "heads": "H", "kv_heads": "Hkv", "head_dim": "dh",
		"qk_norm": true, "rope": map[string]any{"theta": 500000.0}}},
	{"gqa_attention", map[string]any{"d_model": "D", "heads": "H", "kv_heads": "Hkv", "head_dim": "dh",
		"window": 4096.0, "causal": false}},
	{"gated_mlp", map[string]any{"d_model": "D", "hidden": "F", "act": "silu"}},
	{"gated_mlp", map[string]any{"d_model": "D", "hidden": "F", "act": "gelu", "bias": true}},
	{"dense_mlp", map[string]any{"d_model": "D", "hidden": "4*D", "act": "gelu", "bias": true}},
	{"mla_attention", map[string]any{"d_model": 7168.0, "heads": 128.0, "q_lora": 1536.0,
		"kv_lora": 512.0, "nope_dim": 128.0, "rope_dim": 64.0, "v_dim": 128.0,
		"rope": map[string]any{"theta": 10000.0}}},
	{"moe_layer", map[string]any{"d_model": "D", "experts": 8.0, "top_k": 2.0, "expert_hidden": "F"}},
	{"moe_layer", map[string]any{"d_model": "D", "experts": 256.0, "top_k": 8.0,
		"expert_hidden": 2048.0, "shared_experts": 1.0, "router_bias": true}},
	{"mamba2_block", map[string]any{"d_model": "D", "expand": 2.0, "head_dim": 64.0,
		"state": 128.0, "groups": 8.0, "conv_kernel": 4.0}},
	{"transformer_block", map[string]any{"d_model": "D", "heads": "H", "kv_heads": "Hkv",
		"head_dim": "dh", "ffn_hidden": "F"}},
	{"transformer_block", map[string]any{"d_model": "D", "heads": "H", "kv_heads": "Hkv",
		"head_dim": "dh", "ffn_hidden": "F", "norm": "layernorm", "norm_bias": true,
		"mlp": "dense", "act": "gelu", "attn_bias": true, "mlp_bias": true, "rope": nil}},
	{"transformer_block", map[string]any{"d_model": "D", "heads": "H", "kv_heads": "Hkv",
		"head_dim": "dh", "ffn_hidden": "F", "post_norm": true}},
	{"transformer_block", map[string]any{"d_model": "D", "heads": "H", "kv_heads": "Hkv",
		"head_dim": "dh", "ffn_hidden": "F", "mlp": "moe", "experts": 8.0, "top_k": 2.0,
		"expert_hidden": 2048.0}},
	{"transformer_block", map[string]any{"d_model": 7168.0, "heads": 128.0, "kv_heads": 128.0,
		"head_dim": 128.0, "ffn_hidden": 18432.0, "attention": "mla", "q_lora": 1536.0,
		"kv_lora": 512.0, "nope_dim": 128.0, "rope_dim": 64.0, "v_dim": 128.0}},
}

func f(v float64) *float64 { return &v }
func b(v bool) *bool       { return &v }
func i(v int) *int         { return &v }

// OperatingPoint is one set of conditions a design is measured under.
type OperatingPoint struct {
	Label   string
	Options analysis.Options
}

// OperatingPoints are the three the analysis and the rules are checked at.
//
// One would not be enough: the default never exercises sharding, full
// recomputation or an eager attention kernel, and those are three of the places
// the arithmetic is easiest to get subtly wrong.
var OperatingPoints = []OperatingPoint{
	{"default", analysis.Options{}},
	{"sharded", analysis.Options{
		Dtype:          "fp8",
		InferenceDtype: "fp8",
		Recompute:      "full",
		Optimizer:      "adamw8bit",
		Parallel: &analysis.PartialParallel{
			TP: f(8), PP: f(2), DP: f(4), Zero: i(3), SequenceParallel: b(true),
		},
		GPUs:        f(64),
		Concurrency: f(32),
		Tokens:      f(15e12),
		MFU:         f(0.4),
	}},
	{"eager", analysis.Options{
		T: f(8192), B: f(4), Flash: b(false),
		Recompute: "selective", KvDtype: "fp8", GPUs: f(8),
	}},
}

// CodegenVariant is one set of generation settings.
type CodegenVariant struct {
	Label   string
	Options codegen.Options
}

// CodegenVariants take three different paths through the emitter.
var CodegenVariants = []CodegenVariant{
	{"default", codegen.Options{}},
	{"dense", codegen.Options{MoeDispatch: "dense"}},
	{"bare", codegen.Options{InitStd: f(0), NoSmokeTest: true, ClassName: "Net"}},
}

// ScaleCase is one design shrunk towards a budget.
type ScaleCase struct {
	Label   string
	Preset  string
	Options scale.Options
}

// ScaleCases each exercise a different corner: a dense model, one scaled by its
// non-embedding count with a tied head, one at fixed depth, a mixture of
// experts, latent attention, and a state-space model.
var ScaleCases = []ScaleCase{
	{"30m", "gpt2-small", scale.Options{TargetParams: 30e6}},
	{"30m-nonembed", "llama-3-8b", scale.Options{
		TargetParams: 30e6, TargetBasis: "non-embedding", Vocab: f(8192), TieHead: b(true)}},
	{"100m-keepdepth", "llama-3-8b", scale.Options{TargetParams: 100e6, KeepDepth: true}},
	{"50m-moe", "mixtral-8x7b", scale.Options{TargetParams: 50e6, Vocab: f(4096)}},
	{"20m-mla", "deepseek-v3", scale.Options{TargetParams: 20e6, Vocab: f(4096)}},
	{"10m-mamba", "nemotron-h-8b", scale.Options{TargetParams: 10e6, Vocab: f(4096)}},
}

// PlanCase is one design fitted to one cluster.
type PlanCase struct {
	Label   string
	Preset  string
	Seq     float64
	Cluster plan.Request
}

// PlanCases cover the four answers the planner can give: a model that fits on
// one node, one that needs a big cluster and a choice about how to use it, a
// mixture of experts where expert parallelism is on the table, and a model that
// does not fit at all.
var PlanCases = []PlanCase{
	{"8b-on-8", "llama-3-8b", 8192, plan.Request{GPUs: 8, Limit: 6}},
	{"70b-on-64", "llama-3-70b", 8192, plan.Request{GPUs: 64, Limit: 6}},
	{"moe-on-64", "mixtral-8x7b", 4096, plan.Request{GPUs: 64, Limit: 6}},
	{"405b-on-8", "llama-3.1-405b", 8192, plan.Request{GPUs: 8, Limit: 6}},
	{"8b-on-16-tight", "llama-3-8b", 8192, plan.Request{GPUs: 16, Headroom: 0.25, Limit: 4}},
}
