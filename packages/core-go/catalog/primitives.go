// Primitive blocks. These carry every formula in the system.
//
// Conventions:
//   - B and T are reserved runtime symbols (batch, sequence length).
//   - FLOPs are counted the standard way: a multiply-accumulate is 2, and only
//     matmuls are counted in the headline number. Elementwise work is reported
//     separately because the 6N convention excludes it and because those
//     operations are memory-bound.
//   - Activation memory is attributed to tensors, not to blocks: a block lists
//     the input ports it must keep alive, so a tensor read by several blocks
//     costs once.
//
// Sources for each formula are in docs/reference/analysis-math.md.
package catalog

import (
	"fmt"
	"math"

	"github.com/tensorcad/core/shapes"
)

// softcapCost is the arithmetic in `tanh(x / cap) * cap`: a divide, a tanh and
// a multiply, per element.
const softcapCost = 1 + 6 + 1

var elementwiseCost = map[string]float64{
	"relu":      1,
	"relu2":     2,
	"gelu":      8,
	"gelu_tanh": 8,
	"silu":      5,
	"swish":     5,
	"tanh":      6,
	"sigmoid":   4,
	"identity":  0,
}

// convOut is the spatial extent a convolution or pooling window leaves behind.
func convOut(size, kernel, stride, padding int) int {
	return (size+2*padding-kernel)/stride + 1
}

// axisIndex resolves a possibly-negative axis against a rank, as NumPy does.
func axisIndex(axis any, rank int) int {
	n := -1
	switch v := axis.(type) {
	case float64:
		n = int(v)
	case int:
		n = v
	}
	i := n
	if n < 0 {
		i = rank + n
	}
	if i < 0 {
		i = 0
	}
	if rank > 0 && i > rank-1 {
		i = rank - 1
	}
	return i
}

func fmin(a, b float64) float64 { return math.Min(a, b) }

// intp and nump build the pointers ParamSpec wants for its bounds.
func intp(v float64) *float64 { return &v }

func pInt(min float64, doc string) ParamSpec {
	return ParamSpec{Type: ParamInt, Min: intp(min), Doc: doc}
}
func pIntD(def, min float64, doc string) ParamSpec {
	return ParamSpec{Type: ParamInt, Default: def, HasDefault: true, Min: intp(min), Doc: doc}
}
func pNum(def float64, doc string) ParamSpec {
	return ParamSpec{Type: ParamNum, Default: def, HasDefault: true, Doc: doc}
}
func pBool(def bool, doc string) ParamSpec {
	return ParamSpec{Type: ParamBool, Default: def, HasDefault: true, Doc: doc}
}
func pEnum(values []string, def string, doc string) ParamSpec {
	return ParamSpec{Type: ParamEnum, Values: values, Default: def, HasDefault: true, Doc: doc}
}
func pPattern(def string, doc string) ParamSpec {
	return ParamSpec{Type: ParamPattern, Default: def, HasDefault: true, Doc: doc}
}
func pObj(def any, doc string) ParamSpec {
	return ParamSpec{Type: ParamObj, Default: def, HasDefault: true, Doc: doc}
}

func noParams(*Resolved) float64                   { return 0 }
func noFlops(*Resolved, AnalysisCtx) FlopsPerToken { return FlopsPerToken{} }
func noRetains(*Resolved) []string                 { return nil }

// Primitives is every block that carries a formula.
var Primitives = []*BlockDef{
	// --- graph boundaries ---------------------------------------------------
	{
		Kind: "primitive", Type: "input", Category: "io",
		Params: ParamList{
			{"shape", pPattern("B T", "Shape pattern of the model input")},
			{"dtype", pEnum([]string{"int64", "int32", "bf16", "fp32"}, "int64", "")},
		},
		PortsFn: func(r *Resolved) Ports {
			s := r.Str("shape")
			if s == "" {
				s = "B T"
			}
			return Ports{In: map[string]PortSpec{}, Out: map[string]PortSpec{"x": Port(s)}}
		},
		Docs: BlockDocs{Summary: "Model input, usually a batch of token ids."},
	},
	{
		Kind: "primitive", Type: "output", Category: "io",
		Params: ParamList{},
		Ports:  Ports{In: map[string]PortSpec{"x": Port("...")}, Out: map[string]PortSpec{}},
		Docs:   BlockDocs{Summary: "Model output."},
	},
	{
		Kind: "primitive", Type: "boundary_in", Category: "io",
		Params:  ParamList{{"ports", pObj(map[string]any{"x": "B T D"}, "Port name -> shape pattern")}},
		PortsFn: func(r *Resolved) Ports { return Ports{In: map[string]PortSpec{}, Out: portsParam(r)} },
		Docs:    BlockDocs{Summary: "Entry point of a container subgraph."},
	},
	{
		Kind: "primitive", Type: "boundary_out", Category: "io",
		Params:  ParamList{{"ports", pObj(map[string]any{"x": "B T D"}, "Port name -> shape pattern")}},
		PortsFn: func(r *Resolved) Ports { return Ports{In: portsParam(r), Out: map[string]PortSpec{}} },
		Docs:    BlockDocs{Summary: "Exit point of a container subgraph."},
	},

	// --- embeddings ---------------------------------------------------------
	{
		Kind: "primitive", Type: "embedding", Category: "embedding",
		Params: ParamList{
			{"vocab", pInt(1, "Vocabulary size")},
			{"dim", pInt(1, "Embedding width")},
		},
		Ports:      Ports{In: map[string]PortSpec{"ids": Port("...")}, Out: map[string]PortSpec{"y": Port("... dim")}},
		ParamCount: func(r *Resolved) float64 { return r.Num("vocab") * r.Num("dim") },
		Flops:      noFlops,
		Retains:    noRetains,
		Docs: BlockDocs{
			Summary: "Token embedding table.",
			Formula: "params = vocab * dim; FLOPs ~ 0 (a gather, not a matmul)",
		},
	},
	{
		Kind: "primitive", Type: "pos_embedding", Category: "embedding",
		Params: ParamList{
			{"max_seq", pInt(1, "Maximum position index")},
			{"dim", pInt(1, "")},
		},
		Ports:      Ports{In: map[string]PortSpec{"x": Port("... dim")}, Out: map[string]PortSpec{"y": Port("... dim")}},
		ParamCount: func(r *Resolved) float64 { return r.Num("max_seq") * r.Num("dim") },
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			return FlopsPerToken{Elementwise: r.Num("dim")}
		},
		Retains: noRetains,
		Docs: BlockDocs{
			Summary: "Learned absolute position embedding, added to the token embedding (GPT-2 style).",
			Formula: "params = max_seq * dim",
		},
	},
	{
		Kind: "primitive", Type: "learned_tokens", Category: "embedding",
		Params: ParamList{
			{"count", pIntD(1, 1, "How many distinct vectors are learned")},
			{"dim", pInt(1, "")},
			{"tokens", pIntD(0, 0, "Sequence length it is broadcast to; 0 means one position per learned vector")},
		},
		// Nothing goes in. This is a weight that is also an activation:
		// broadcast across the batch, and often across the sequence as well.
		PortsFn: func(r *Resolved) Ports {
			span := "count"
			if r.Num("tokens") != 0 {
				span = "tokens"
			}
			return Ports{
				In:  map[string]PortSpec{},
				Out: map[string]PortSpec{"y": Port("B " + span + " dim")},
			}
		},
		ParamCount: func(r *Resolved) float64 { return r.Num("count") * r.Num("dim") },
		Flops:      noFlops,
		Retains:    noRetains,
		Docs: BlockDocs{
			Summary: "A learned tensor with no input: a mask token, a CLS token, register tokens, " +
				"learned queries. I-JEPA's predictor stands one of these in for every patch " +
				"it has to predict.",
			Formula: "params = count * dim",
			Refs:    []string{"https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py"},
		},
	},

	// --- linear algebra -----------------------------------------------------
	{
		Kind: "primitive", Type: "linear", Category: "linear",
		Params: ParamList{
			{"in_features", pInt(1, "")},
			{"out_features", pInt(1, "")},
			{"bias", pBool(false, "")},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": Port("... in_features")},
			Out: map[string]PortSpec{"y": Port("... out_features")},
		},
		ParamCount: func(r *Resolved) float64 {
			n := r.Num("in_features") * r.Num("out_features")
			if r.Bool("bias") {
				n += r.Num("out_features")
			}
			return n
		},
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			return FlopsPerToken{Fwd: 2 * r.Num("in_features") * r.Num("out_features")}
		},
		Retains: func(*Resolved) []string { return []string{"x"} },
		Docs: BlockDocs{
			Summary: "Dense projection.",
			Formula: "params = in*out (+out with bias); FLOPs/token = 2*in*out; saves its input for the weight gradient",
		},
	},
	{
		Kind: "primitive", Type: "lm_head", Category: "head",
		Params: ParamList{
			{"vocab", pInt(1, "")},
			{"dim", pInt(1, "")},
			{"tied", pBool(false, "Share weights with the token embedding")},
			{"bias", pBool(false, "")},
			{"softcap", ParamSpec{Type: ParamNum, Default: 0.0, HasDefault: true,
				Doc: "Bound the logits to this magnitude with tanh; 0 leaves them alone"}},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": Port("... dim")},
			Out: map[string]PortSpec{"y": Port("... vocab")},
		},
		ParamCount: func(r *Resolved) float64 {
			n := 0.0
			if !r.Bool("tied") {
				n = r.Num("vocab") * r.Num("dim")
			}
			if r.Bool("bias") {
				n += r.Num("vocab")
			}
			return n
		},
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			f := FlopsPerToken{Fwd: 2 * r.Num("vocab") * r.Num("dim")}
			if r.Num("softcap") != 0 {
				f.Elementwise = softcapCost * r.Num("vocab")
			}
			return f
		},
		Retains: func(*Resolved) []string { return []string{"x"} },
		// bf16 logits plus the fp32 softmax/cross-entropy buffer.
		ExtraActivationBytes: func(r *Resolved, c AnalysisCtx) float64 {
			return r.Num("vocab") * (c.Bytes + 4)
		},
		Docs: BlockDocs{
			Summary: "Output projection to vocabulary logits.",
			Formula: "params = 0 when tied, else vocab*dim; FLOPs/token = 2*vocab*dim; " +
				"logits cost vocab*(bytes+4) per token",
			Refs: []string{"https://blog.eleuther.ai/transformer-math/"},
		},
	},

	// --- convolution --------------------------------------------------------
	//
	// A convnet is the one architecture here whose tensors are not a sequence
	// of vectors. Its shapes are B C H W and its spatial extent shrinks layer
	// by layer, so each block is told what arrives and works out what leaves.
	// The per-token FLOP convention holds with one image as the token.
	{
		Kind: "primitive", Type: "conv2d", Category: "linear",
		Params: ParamList{
			{"in_channels", pInt(1, "")},
			{"out_channels", pInt(1, "")},
			{"kernel", pIntD(3, 1, "")},
			{"stride", pIntD(1, 1, "")},
			{"padding", pIntD(0, 0, "")},
			{"groups", pIntD(1, 1, "Grouped convolution; equal to in_channels is depthwise")},
			{"bias", pBool(true, "")},
			{"in_h", pInt(1, "Height of the incoming feature map")},
			{"in_w", pInt(1, "Width of the incoming feature map")},
			{"act", pEnum([]string{"identity", "relu", "relu2", "gelu", "gelu_tanh", "silu"}, "identity",
				"Activation applied to the output, as a convnet always does")},
		},
		PortsFn: func(r *Resolved) Ports {
			oh := convOut(r.Int("in_h"), r.Int("kernel"), r.Int("stride"), r.Int("padding"))
			ow := convOut(r.Int("in_w"), r.Int("kernel"), r.Int("stride"), r.Int("padding"))
			return Ports{
				In: map[string]PortSpec{"x": Port(fmt.Sprintf("B %d %d %d",
					r.Int("in_channels"), r.Int("in_h"), r.Int("in_w")))},
				Out: map[string]PortSpec{"y": Port(fmt.Sprintf("B %d %d %d",
					r.Int("out_channels"), oh, ow))},
			}
		},
		ParamCount: func(r *Resolved) float64 {
			n := (r.Num("in_channels") / r.Num("groups")) * r.Num("out_channels") *
				r.Num("kernel") * r.Num("kernel")
			if r.Bool("bias") {
				n += r.Num("out_channels")
			}
			return n
		},
		// Every output position is a dot product over one kernel window, so the
		// cost is the weight count times the positions it is applied at. This
		// is the whole image, not one token: a convnet's token is the image.
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			oh := float64(convOut(r.Int("in_h"), r.Int("kernel"), r.Int("stride"), r.Int("padding")))
			ow := float64(convOut(r.Int("in_w"), r.Int("kernel"), r.Int("stride"), r.Int("padding")))
			positions := oh * ow
			perPosition := 2 * (r.Num("in_channels") / r.Num("groups")) * r.Num("out_channels") *
				r.Num("kernel") * r.Num("kernel")
			return FlopsPerToken{
				Fwd:         perPosition * positions,
				Elementwise: elementwiseCost[r.Str("act")] * r.Num("out_channels") * positions,
			}
		},
		Retains: func(*Resolved) []string { return []string{"x"} },
		Docs: BlockDocs{
			Summary: "Two-dimensional convolution, optionally with the activation a convnet always follows it with.",
			Formula: "params = (in/groups)*out*k*k (+out with bias); " +
				"FLOPs = 2*(in/groups)*out*k*k*H_out*W_out; " +
				"H_out = floor((H + 2*padding - k)/stride) + 1",
			Refs: []string{"https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html"},
		},
	},
	{
		Kind: "primitive", Type: "maxpool2d", Category: "shape",
		Params: ParamList{
			{"channels", pInt(1, "")},
			{"kernel", pIntD(2, 1, "")},
			{"stride", pIntD(2, 1, "")},
			{"padding", pIntD(0, 0, "")},
			{"in_h", pInt(1, "")},
			{"in_w", pInt(1, "")},
		},
		PortsFn: func(r *Resolved) Ports {
			oh := convOut(r.Int("in_h"), r.Int("kernel"), r.Int("stride"), r.Int("padding"))
			ow := convOut(r.Int("in_w"), r.Int("kernel"), r.Int("stride"), r.Int("padding"))
			return Ports{
				In: map[string]PortSpec{"x": Port(fmt.Sprintf("B %d %d %d",
					r.Int("channels"), r.Int("in_h"), r.Int("in_w")))},
				Out: map[string]PortSpec{"y": Port(fmt.Sprintf("B %d %d %d", r.Int("channels"), oh, ow))},
			}
		},
		ParamCount: noParams,
		// A comparison per element of each window: no multiply-accumulates, so
		// it belongs with the memory-bound work rather than the headline number.
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			oh := float64(convOut(r.Int("in_h"), r.Int("kernel"), r.Int("stride"), r.Int("padding")))
			ow := float64(convOut(r.Int("in_w"), r.Int("kernel"), r.Int("stride"), r.Int("padding")))
			return FlopsPerToken{Elementwise: r.Num("channels") * oh * ow * r.Num("kernel") * r.Num("kernel")}
		},
		Retains: func(*Resolved) []string { return []string{"x"} },
		Docs: BlockDocs{
			Summary: "Spatial max pooling.",
			Formula: "no parameters; H_out = floor((H + 2*padding - k)/stride) + 1",
		},
	},
	{
		Kind: "primitive", Type: "flatten2d", Category: "shape",
		Params: ParamList{
			{"channels", pInt(1, "")},
			{"in_h", pInt(1, "")},
			{"in_w", pInt(1, "")},
		},
		// Where a convnet stops being spatial and becomes a vector, which in
		// AlexNet is where nine tenths of its parameters live.
		PortsFn: func(r *Resolved) Ports {
			return Ports{
				In: map[string]PortSpec{"x": Port(fmt.Sprintf("B %d %d %d",
					r.Int("channels"), r.Int("in_h"), r.Int("in_w")))},
				Out: map[string]PortSpec{"y": Port(fmt.Sprintf("B %d",
					r.Int("channels")*r.Int("in_h")*r.Int("in_w")))},
			}
		},
		ParamCount: noParams, Flops: noFlops, Retains: noRetains,
		Docs: BlockDocs{Summary: "Folds a feature map into one vector per sample."},
	},

	// --- normalisation and elementwise --------------------------------------
	{
		Kind: "primitive", Type: "rmsnorm", Category: "norm",
		Params: ParamList{
			{"dim", pInt(1, "")},
			{"eps", pNum(1e-5, "")},
			{"scale", pBool(true, "Learned per-channel gain")},
		},
		Ports: Ports{In: map[string]PortSpec{"x": Port("... dim")}, Out: map[string]PortSpec{"y": Port("... dim")}},
		ParamCount: func(r *Resolved) float64 {
			if r.Bool("scale") {
				return r.Num("dim")
			}
			return 0
		},
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			return FlopsPerToken{Elementwise: 4 * r.Num("dim")}
		},
		Retains: func(*Resolved) []string { return []string{"x"} },
		Docs:    BlockDocs{Summary: "Root-mean-square layer norm.", Formula: "params = dim (scale only, no bias)"},
	},
	{
		Kind: "primitive", Type: "layernorm", Category: "norm",
		Params: ParamList{
			{"dim", pInt(1, "")},
			{"eps", pNum(1e-5, "")},
			{"bias", pBool(true, "")},
		},
		Ports: Ports{In: map[string]PortSpec{"x": Port("... dim")}, Out: map[string]PortSpec{"y": Port("... dim")}},
		ParamCount: func(r *Resolved) float64 {
			n := r.Num("dim")
			if r.Bool("bias") {
				n += r.Num("dim")
			}
			return n
		},
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			return FlopsPerToken{Elementwise: 6 * r.Num("dim")}
		},
		Retains: func(*Resolved) []string { return []string{"x"} },
		Docs:    BlockDocs{Summary: "Standard layer norm.", Formula: "params = 2*dim with bias, dim without"},
	},
	{
		Kind: "primitive", Type: "activation", Category: "elementwise",
		Params: ParamList{
			{"kind", pEnum([]string{"silu", "gelu", "gelu_tanh", "relu", "relu2", "tanh", "sigmoid", "identity"}, "silu", "")},
			{"dim", pInt(1, "Width, used for the memory and elementwise estimates")},
		},
		Ports:      Ports{In: map[string]PortSpec{"x": Port("... dim")}, Out: map[string]PortSpec{"y": Port("... dim")}},
		ParamCount: noParams,
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			cost, ok := elementwiseCost[r.Str("kind")]
			if !ok {
				cost = 4
			}
			return FlopsPerToken{Elementwise: cost * r.Num("dim")}
		},
		Retains: func(*Resolved) []string { return []string{"x"} },
		Docs:    BlockDocs{Summary: "Pointwise nonlinearity."},
	},
	{
		Kind: "primitive", Type: "add", Category: "elementwise",
		Params: ParamList{
			{"dim", pInt(1, "")},
		},
		Ports: Ports{
			In: map[string]PortSpec{
				"a": Port("... dim"),
				// The bypass. A figure draws this running alongside the main
				// path, never entering from above, because from above it reads
				// as the main path rather than the one that skips it.
				"b": {Shape: "... dim", Anchor: "side"},
			},
			Out: map[string]PortSpec{"y": Port("... dim")},
		},
		ParamCount: noParams,
		Flops:      func(r *Resolved, _ AnalysisCtx) FlopsPerToken { return FlopsPerToken{Elementwise: r.Num("dim")} },
		// The gradient of an add is the identity, so nothing needs saving.
		Retains: noRetains,
		Docs:    BlockDocs{Summary: "Elementwise sum, the residual connection."},
	},
	{
		Kind: "primitive", Type: "mul", Category: "elementwise",
		Params: ParamList{
			{"dim", pInt(1, "")},
		},
		Ports: Ports{
			In: map[string]PortSpec{
				"a": Port("... dim"),
				"b": {Shape: "... dim", Anchor: "side"},
			},
			Out: map[string]PortSpec{"y": Port("... dim")},
		},
		ParamCount: noParams,
		Flops:      func(r *Resolved, _ AnalysisCtx) FlopsPerToken { return FlopsPerToken{Elementwise: r.Num("dim")} },
		// Each operand is needed to differentiate the other.
		Retains: func(*Resolved) []string { return []string{"a", "b"} },
		Docs:    BlockDocs{Summary: "Elementwise product, the gate in a gated MLP."},
	},
	{
		Kind: "primitive", Type: "rearrange", Category: "shape",
		Params: ParamList{
			{"from", pPattern("B T (H dh)", "")},
			{"to", pPattern("B H T dh", "")},
		},
		PortsFn: func(r *Resolved) Ports {
			return Ports{
				In:  map[string]PortSpec{"x": Port(r.Str("from"))},
				Out: map[string]PortSpec{"y": Port(r.Str("to"))},
			}
		},
		ParamCount: noParams, Flops: noFlops, Retains: noRetains,
		Docs: BlockDocs{
			Summary: "Reshape or permute, written in einops notation.",
			Formula: "No parameters and no FLOPs; the product of the dimensions must be preserved",
		},
	},

	// --- attention ----------------------------------------------------------
	{
		Kind: "primitive", Type: "rope", Category: "position",
		Params: ParamList{
			{"heads", pInt(1, "")},
			{"head_dim", pInt(2, "")},
			{"theta", pNum(10000, "")},
			{"scaling", pObj(nil, "Optional RoPE scaling spec (linear, NTK, YaRN)")},
		},
		Ports: Ports{
			In: map[string]PortSpec{"x": Port("B heads T head_dim")},
			// An accessory beside the line rather than a stage on it.
			Out: map[string]PortSpec{"y": {Shape: "B heads T head_dim", Anchor: "side"}},
		},
		ParamCount: noParams,
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			return FlopsPerToken{Elementwise: 6 * r.Num("heads") * r.Num("head_dim")}
		},
		// The rotation is reconstructed from the position, so nothing is saved.
		Retains: noRetains,
		Constraints: func(r *Resolved) []BlockFinding {
			if int(r.Num("head_dim"))%2 == 0 {
				return nil
			}
			return []BlockFinding{{
				ID: "ROPE-01", Severity: "error", Param: "head_dim",
				Message: fmt.Sprintf("RoPE needs an even head_dim, got %s", num(r.Num("head_dim"))),
				Hint:    "Rotary embedding rotates pairs of channels, so an odd width leaves one unpaired.",
			}}
		},
		Docs: BlockDocs{
			Summary: "Rotary position embedding applied to queries or keys.",
			Refs:    []string{"https://arxiv.org/abs/2104.09864"},
		},
	},
	{
		Kind: "primitive", Type: "sdpa", Category: "attention",
		Params: ParamList{
			{"heads", pInt(1, "Query heads")},
			{"kv_heads", pInt(1, "Key/value heads; equal to heads for MHA, 1 for MQA")},
			{"head_dim", pInt(1, "Width of a query/key head")},
			{"v_head_dim", pIntD(0, 0, "Width of a value head; 0 means the same as head_dim")},
			{"causal", pBool(true, "")},
			{"window", ParamSpec{Type: ParamInt, Default: 0.0, HasDefault: true, Doc: "Sliding-window width; 0 means full attention"}},
			{"flash", pBool(true, "Memory-efficient kernel that never materializes the score matrix")},
			{"cache", pBool(true, "Whether this block owns the inference cache. Latent attention caches a compressed vector instead.")},
			{"logit_softcap", ParamSpec{Type: ParamNum, Default: 0.0, HasDefault: true,
				Doc: "Bound the attention scores to this magnitude with tanh; 0 leaves them alone. A fused kernel cannot do this, so a layer that caps runs eager."}},
		},
		PortsFn: func(r *Resolved) Ports {
			v := "head_dim"
			if r.Num("v_head_dim") != 0 {
				v = "v_head_dim"
			}
			return Ports{
				In: map[string]PortSpec{
					"q": Port("B heads T head_dim"),
					"k": Port("B kv_heads T head_dim"),
					"v": Port("B kv_heads T " + v),
				},
				Out: map[string]PortSpec{"y": Port("B heads T " + v)},
			}
		},
		ParamCount: noParams,
		Flops: func(r *Resolved, c AnalysisCtx) FlopsPerToken {
			tEff := c.T
			if w := r.Num("window"); w > 0 {
				tEff = fmin(c.T, w)
			}
			// A causal kernel skips masked blocks, so on average each token
			// attends to half the window.
			causal := 1.0
			if r.Bool("causal") {
				causal = 0.5
			}
			unmasked := 4 * tEff * r.Num("heads") * r.Num("head_dim")
			f := FlopsPerToken{FwdSeq: unmasked * causal, FwdSeqUnmasked: unmasked}
			if r.Num("logit_softcap") != 0 {
				// Over the whole score matrix, not half of it: capping forces
				// the eager form, which computes every score and then masks.
				f.Elementwise = softcapCost * tEff * r.Num("heads")
			}
			return f
		},
		// q, k and v arrive on edges and are counted there; the output and the
		// kernel's own statistics are not on any edge the backward pass reads.
		Retains: func(*Resolved) []string { return []string{"q", "k", "v"} },
		ExtraActivationBytes: func(r *Resolved, c AnalysisCtx) float64 {
			tEff := c.T
			if w := r.Num("window"); w > 0 {
				tEff = fmin(c.T, w)
			}
			vDim := r.Num("v_head_dim")
			if vDim == 0 {
				vDim = r.Num("head_dim")
			}
			output := r.Num("heads") * vDim * c.Bytes
			if c.Flash && r.Bool("flash") && r.Num("logit_softcap") == 0 {
				// A fused kernel keeps the output and the log-sum-exp
				// statistics only.
				return output + r.Num("heads")*4
			}
			// Otherwise the score matrix row and the softmax output are both
			// kept.
			return output + 2*r.Num("heads")*tEff*c.Bytes
		},
		StateBytes: func(r *Resolved, c AnalysisCtx) StateBytes {
			if v, ok := r.P["cache"].(bool); ok && !v {
				return StateBytes{}
			}
			vDim := r.Num("v_head_dim")
			if vDim == 0 {
				vDim = r.Num("head_dim")
			}
			perTokenFull := r.Num("kv_heads") * (r.Num("head_dim") + vDim) * c.Bytes
			if w := r.Num("window"); w > 0 {
				return StateBytes{PerSequence: perTokenFull * w}
			}
			return StateBytes{PerToken: perTokenFull}
		},
		Constraints: func(r *Resolved) []BlockFinding {
			var out []BlockFinding
			if int(r.Num("heads"))%int(r.Num("kv_heads")) != 0 {
				out = append(out, BlockFinding{
					ID: "SDPA-01", Severity: "error", Param: "kv_heads",
					Message: fmt.Sprintf("heads (%s) must be divisible by kv_heads (%s)",
						num(r.Num("heads")), num(r.Num("kv_heads"))),
				})
			}
			if r.Num("window") < 0 {
				out = append(out, BlockFinding{
					ID: "SDPA-02", Severity: "error", Param: "window",
					Message: "window must be non-negative",
				})
			}
			if r.Num("logit_softcap") != 0 && r.Bool("flash") {
				out = append(out, BlockFinding{
					ID: "SDPA-03", Severity: "warning", Param: "logit_softcap",
					Message: "capping the attention scores rules out a fused kernel, which never " +
						"materializes them to cap. This layer is counted as eager attention, so " +
						"the score matrix is held for the backward pass.",
				})
			}
			return out
		},
		Docs: BlockDocs{
			Summary: "Scaled dot-product attention core. Covers MHA, GQA and MQA through kv_heads.",
			Formula: "FLOPs/token = 4*T_eff*heads*head_dim (halved when causal); KV cache = 2*kv_heads*head_dim*bytes per token",
			Refs:    []string{"https://arxiv.org/abs/2305.13245", "https://arxiv.org/abs/2205.14135"},
		},
	},

	// --- mixture of experts -------------------------------------------------
	{
		Kind: "primitive", Type: "topk_router", Category: "moe",
		Params: ParamList{
			{"d_model", pInt(1, "")},
			{"experts", pInt(1, "Routed experts to choose from")},
			{"top_k", pInt(1, "Experts each token is sent to")},
			{"bias", pBool(false, "Per-expert routing bias (DeepSeek's score correction)")},
			{"normalize", pBool(true, "Renormalize the chosen weights to sum to one")},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": Port("... d_model")},
			Out: map[string]PortSpec{"weights": {Shape: "... top_k", Dtype: "float", Anchor: "side"}},
		},
		ParamCount: func(r *Resolved) float64 {
			n := r.Num("d_model") * r.Num("experts")
			if r.Bool("bias") {
				n += r.Num("experts")
			}
			return n
		},
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			return FlopsPerToken{Fwd: 2 * r.Num("d_model") * r.Num("experts")}
		},
		Retains: func(*Resolved) []string { return []string{"x"} },
		Constraints: func(r *Resolved) []BlockFinding {
			if r.Num("top_k") <= r.Num("experts") {
				return nil
			}
			return []BlockFinding{{
				ID: "ROUTER-01", Severity: "error", Param: "top_k",
				Message: fmt.Sprintf("top_k (%s) cannot exceed the number of experts (%s)",
					num(r.Num("top_k")), num(r.Num("experts"))),
			}}
		},
		Docs: BlockDocs{
			Summary: "Chooses which experts each token is sent to.",
			Formula: "params = d_model * experts (+ experts with a routing bias)",
			Refs:    []string{"https://arxiv.org/abs/2401.06066"},
		},
	},
	{
		Kind: "primitive", Type: "weighted_sum", Category: "moe",
		Params: ParamList{
			{"dim", pInt(1, "")},
			{"n", pInt(1, "How many contributions are combined")},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": Port("... dim"), "weights": Port("... n")},
			Out: map[string]PortSpec{"y": Port("... dim")},
		},
		ParamCount: noParams,
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			return FlopsPerToken{Elementwise: r.Num("dim") * r.Num("n")}
		},
		Retains: noRetains,
		Docs:    BlockDocs{Summary: "Combines the chosen experts' outputs using the router's weights."},
	},

	// --- tensor plumbing ----------------------------------------------------
	{
		Kind: "primitive", Type: "split", Category: "shape",
		Params: ParamList{
			{"from", pPattern("B T D", "Shape of the incoming tensor")},
			{"sizes", pObj([]any{}, "Widths of the pieces, along the last dimension")},
			{"axis", ParamSpec{Type: ParamInt, Default: -1.0, HasDefault: true, Doc: "Which dimension to cut. Only the last is supported."}},
		},
		PortsFn: func(r *Resolved) Ports {
			atoms := patternAtoms(r.Str("from"))
			out := map[string]PortSpec{}
			for i, size := range sizeList(r.P["sizes"]) {
				copyAtoms := append([]string{}, atoms...)
				if len(copyAtoms) > 0 {
					copyAtoms[len(copyAtoms)-1] = "(" + size + ")"
				}
				out[fmt.Sprintf("y%d", i)] = Port(joinAtoms(copyAtoms))
			}
			return Ports{In: map[string]PortSpec{"x": Port(r.Str("from"))}, Out: out}
		},
		ParamCount: noParams, Flops: noFlops, Retains: noRetains,
		Docs: BlockDocs{
			Summary: "Cuts a tensor into pieces along its last dimension.",
			Formula: "No parameters and no FLOPs; the pieces must add up to the incoming width",
		},
	},
	{
		Kind: "primitive", Type: "concat", Category: "shape",
		Params: ParamList{
			{"to", pPattern("B T D", "Shape of the combined tensor")},
			{"sizes", pObj([]any{}, "Extents of the pieces, along `axis`")},
			{"axis", ParamSpec{Type: ParamInt, Default: -1.0, HasDefault: true, Doc: "Which dimension to join on; negative counts from the end"}},
		},
		// Any axis, not only the last. Joining along the channel dimension is
		// what a fused QKV projection does; joining along the sequence is what a
		// predictor does when it stands mask tokens beside its context.
		PortsFn: func(r *Resolved) Ports {
			atoms := patternAtoms(r.Str("to"))
			axis := axisIndex(r.P["axis"], len(atoms))
			in := map[string]PortSpec{}
			for i, size := range sizeList(r.P["sizes"]) {
				copyAtoms := append([]string{}, atoms...)
				if axis < len(copyAtoms) {
					copyAtoms[axis] = "(" + size + ")"
				}
				in[fmt.Sprintf("y%d", i)] = Port(joinAtoms(copyAtoms))
			}
			return Ports{In: in, Out: map[string]PortSpec{"y": Port(r.Str("to"))}}
		},
		ParamCount: noParams, Flops: noFlops, Retains: noRetains,
		Docs: BlockDocs{Summary: "Joins tensors along one dimension."},
	},
	{
		Kind: "primitive", Type: "expand_heads", Category: "shape",
		Params: ParamList{
			{"heads", pInt(1, "")},
			{"dim", pInt(1, "")},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": Port("B T dim")},
			Out: map[string]PortSpec{"y": Port("B heads T dim")},
		},
		ParamCount: noParams, Flops: noFlops,
		// A broadcast view costs nothing to keep.
		Retains: noRetains,
		Docs: BlockDocs{
			Summary: "Shares one tensor across every attention head, as latent attention does with its rotary key.",
		},
	},
	{
		Kind: "primitive", Type: "kv_latent_cache", Category: "attention",
		Params: ParamList{
			{"dim", pInt(1, "Width of the cached vector per token per layer")},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": Port("... dim")},
			Out: map[string]PortSpec{"y": Port("... dim")},
		},
		ParamCount: noParams, Flops: noFlops, Retains: noRetains,
		StateBytes: func(r *Resolved, c AnalysisCtx) StateBytes {
			return StateBytes{PerToken: r.Num("dim") * c.Bytes}
		},
		Docs: BlockDocs{
			Summary: "Marks the compressed vector that latent attention caches instead of keys and values.",
			Formula: "cache = layers * dim * bytes per token, against 2 * layers * kv_heads * head_dim * bytes for GQA",
			Refs:    []string{"https://arxiv.org/abs/2405.04434"},
		},
	},

	// --- state-space models -------------------------------------------------
	{
		Kind: "primitive", Type: "conv1d", Category: "ssm",
		Params: ParamList{
			{"channels", pInt(1, "Width of the stream, convolved per channel")},
			{"kernel", pIntD(4, 1, "Kernel width")},
			{"bias", pBool(true, "")},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": Port("... channels")},
			Out: map[string]PortSpec{"y": Port("... channels")},
		},
		ParamCount: func(r *Resolved) float64 {
			n := r.Num("channels") * r.Num("kernel")
			if r.Bool("bias") {
				n += r.Num("channels")
			}
			return n
		},
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			return FlopsPerToken{Fwd: 2 * r.Num("channels") * r.Num("kernel")}
		},
		Retains: func(*Resolved) []string { return []string{"x"} },
		// Generation keeps the last kernel-1 tokens per channel.
		StateBytes: func(r *Resolved, c AnalysisCtx) StateBytes {
			return StateBytes{PerSequence: r.Num("channels") * (r.Num("kernel") - 1) * c.Bytes}
		},
		Docs: BlockDocs{
			Summary: "Short depthwise convolution over time, used before a state-space scan.",
			Formula: "params = channels * kernel (+ channels with bias)",
			Refs:    []string{"https://arxiv.org/abs/2405.21060"},
		},
	},
	{
		Kind: "primitive", Type: "ssd_scan", Category: "ssm",
		Params: ParamList{
			{"d_inner", pInt(1, "Width of the state-space stream")},
			{"heads", pInt(1, "")},
			{"head_dim", pInt(1, "Width per state-space head (Mamba-2's P)")},
			{"state", pInt(1, "Recurrent state width per head (Mamba-2's N)")},
			{"groups", pInt(1, "How many heads share one B/C projection")},
			{"xbc_width", pInt(1, "Width of the combined x, B and C stream")},
			{"chunk", ParamSpec{Type: ParamInt, Default: 256.0, HasDefault: true, Doc: "Chunk length of the chunked scan"}},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"xbc": Port("... xbc_width"), "dt": Port("... heads")},
			Out: map[string]PortSpec{"y": Port("... d_inner")},
		},
		// Per-head decay, skip and timestep-bias scalars.
		ParamCount: func(r *Resolved) float64 { return 3 * r.Num("heads") },
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			// The scan is linear in sequence length. The constants are
			// approximate: the Mamba-2 paper gives asymptotics, not counts.
			return FlopsPerToken{
				Fwd: 6*r.Num("d_inner")*r.Num("state") + 4*r.Num("d_inner")*r.Num("chunk"),
			}
		},
		Retains: func(*Resolved) []string { return []string{"xbc", "dt"} },
		// The recurrent state is fixed per sequence: this is why a state-space
		// layer has no cache that grows with context.
		StateBytes: func(r *Resolved, c AnalysisCtx) StateBytes {
			return StateBytes{PerSequence: r.Num("heads") * r.Num("head_dim") * r.Num("state") * c.Bytes}
		},
		Constraints: func(r *Resolved) []BlockFinding {
			var out []BlockFinding
			if r.Num("heads")*r.Num("head_dim") != r.Num("d_inner") {
				out = append(out, BlockFinding{
					ID: "SSD-01", Severity: "error", Param: "head_dim",
					Message: fmt.Sprintf("heads (%s) times head_dim (%s) must equal d_inner (%s)",
						num(r.Num("heads")), num(r.Num("head_dim")), num(r.Num("d_inner"))),
				})
			}
			if int(r.Num("heads"))%int(r.Num("groups")) != 0 {
				out = append(out, BlockFinding{
					ID: "SSD-02", Severity: "error", Param: "groups",
					Message: fmt.Sprintf("heads (%s) must be divisible by groups (%s)",
						num(r.Num("heads")), num(r.Num("groups"))),
				})
			}
			return out
		},
		Docs: BlockDocs{
			Summary: "Mamba-2 state-space scan. Linear in sequence length, and its state is fixed per sequence rather than growing per token.",
			Formula: "params = 3*heads; state = heads*head_dim*state*bytes per sequence; FLOPs are approximate",
			Refs:    []string{"https://arxiv.org/abs/2405.21060"},
		},
	},
}

// portsParam reads the `ports` object a boundary node carries.
func portsParam(r *Resolved) map[string]PortSpec {
	out := map[string]PortSpec{}
	m, ok := r.P["ports"].(map[string]any)
	if !ok {
		return out
	}
	for k, v := range m {
		if s, ok := v.(string); ok {
			out[k] = Port(s)
		}
	}
	return out
}

// sizeList reads the `sizes` array of a split or concat, which the document may
// write as numbers or as expressions.
func sizeList(v any) []string {
	list, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(list))
	for _, item := range list {
		switch s := item.(type) {
		case string:
			out = append(out, s)
		case float64:
			out = append(out, num(s))
		default:
			out = append(out, fmt.Sprint(item))
		}
	}
	return out
}

func patternAtoms(src string) []string {
	p, err := shapes.ParsePattern(src)
	if err != nil {
		return nil
	}
	out := make([]string, len(p.Atoms))
	for i, a := range p.Atoms {
		out[i] = shapes.AtomToString(a)
	}
	return out
}

func joinAtoms(atoms []string) string {
	out := ""
	for i, a := range atoms {
		if i > 0 {
			out += " "
		}
		out += a
	}
	return out
}
