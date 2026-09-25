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

	"github.com/tensorcad/core/attnexpr"
	"github.com/tensorcad/core/shapes"
)

// softcapCost is the arithmetic in `tanh(x / cap) * cap`: a divide, a tanh and
// a multiply, per element. An attention score's cap is costed as the
// expression it is, which comes to the same.
const softcapCost = 1 + 6 + 1

// maskSpec and scoreSpec are the two attention expressions, declared once for
// every block that carries them down to sdpa.
func maskSpec() ParamSpec {
	return ParamSpec{Type: ParamMask, Default: "", HasDefault: true,
		Doc: "Which scores count, as an expression over q and kv (the query's and the key's positions), " +
			"h (the head) and the design's symbols. A score has to pass causal, window and this, so a " +
			"mask that widens attention needs causal off: kv <= q or kv < 16 is causal with the first " +
			"16 positions seen by all. Empty adds nothing."}
}

// sinksSpec is a learned score per head that every query can attend to instead
// of any key. Unset rather than false by default, so a design that never
// mentions it resolves, documents and generates exactly as it did before.
func sinksSpec() ParamSpec {
	return ParamSpec{Type: ParamBool, Default: nil, HasDefault: true,
		Doc: "Learn one score per head that sits in the softmax's denominator beside the keys, so a " +
			"query can put its attention nowhere rather than spreading it over keys it has no use for " +
			"(gpt-oss). Adds heads parameters; unset is none."}
}

func scoreSpec() ParamSpec {
	return ParamSpec{Type: ParamScore, Default: "", HasDefault: true,
		Doc: "What each score becomes before the softmax, over score, q, kv, h, heads and the " +
			"design's symbols: score - 2 ** (-8 * (h + 1) / heads) * (q - kv) is ALiBi. Applied " +
			"before logit_softcap; empty leaves the scores alone."}
}

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

// grouped puts a parameter under a heading in the inspector.
func grouped(spec ParamSpec, group string) ParamSpec {
	spec.Group = group
	return spec
}

// when says which values of another parameter make this one meaningful.
func when(spec ParamSpec, param string, is ...string) ParamSpec {
	spec.When = &ParamWhen{Param: param, Is: is}
	return spec
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
			{"dtype", pEnum([]string{"int64", "int32", "bf16", "fp32"}, "int64", "What the tensor holds; token ids are integers, everything else is not")},
		},
		PortsFn: func(r *Resolved) Ports {
			s := r.Str("shape")
			if s == "" {
				s = "B T"
			}
			// The `dtype` parameter already says what this carries; the port
			// said `inherit` and an input inherits from nothing, so the one
			// place the answer was known was the one place it was not written.
			return Ports{
				In:  map[string]PortSpec{},
				Out: map[string]PortSpec{"x": {Shape: s, Dtype: dtypeOfName(r.Str("dtype")), Anchor: "flow"}},
			}
		},
		Docs: BlockDocs{Name: "input", Summary: "Model input, usually a batch of token ids."},
	},
	{
		Kind: "primitive", Type: "output", Category: "io",
		Params: ParamList{},
		Ports:  Ports{In: map[string]PortSpec{"x": Port("...")}, Out: map[string]PortSpec{}},
		Docs:   BlockDocs{Name: "output", Summary: "Model output."},
	},
	{
		Kind: "primitive", Type: "boundary_in", Category: "io",
		Params:  ParamList{{"ports", pObj(map[string]any{"x": "B T D"}, "Port name -> shape pattern")}},
		PortsFn: func(r *Resolved) Ports { return Ports{In: map[string]PortSpec{}, Out: portsParam(r)} },
		Docs:    BlockDocs{Name: "block input", Summary: "Entry point of a container subgraph."},
	},
	{
		Kind: "primitive", Type: "boundary_out", Category: "io",
		Params:  ParamList{{"ports", pObj(map[string]any{"x": "B T D"}, "Port name -> shape pattern")}},
		PortsFn: func(r *Resolved) Ports { return Ports{In: portsParam(r), Out: map[string]PortSpec{}} },
		Docs:    BlockDocs{Name: "block output", Summary: "Exit point of a container subgraph."},
	},

	// --- embeddings ---------------------------------------------------------
	{
		Kind: "primitive", Type: "embedding", Category: "embedding",
		Params: ParamList{
			{"vocab", pInt(1, "Vocabulary size")},
			{"dim", pInt(1, "Embedding width")},
		},
		Ports: Ports{
			// An index in, a vector out. This is the block where the integral
			// side of a design stops and the real side begins, so both ends of
			// it have to say so — without the output declaring `float`, `int`
			// would propagate through the entire model.
			In:  map[string]PortSpec{"ids": {Shape: "...", Dtype: "int", Anchor: "flow"}},
			Out: map[string]PortSpec{"y": {Shape: "... dim", Dtype: "float", Anchor: "flow"}},
		},
		ParamCount: func(r *Resolved) float64 { return r.Num("vocab") * r.Num("dim") },
		Flops:      noFlops,
		Retains:    noRetains,
		Docs: BlockDocs{
			Name:    "token embedding",
			Summary: "Token embedding table.",
			Formula: "params = vocab * dim; FLOPs ~ 0 (a gather, not a matmul)",
		},
	},
	{
		Kind: "primitive", Type: "pos_embedding", Category: "embedding",
		Params: ParamList{
			{"max_seq", pInt(1, "Maximum position index")},
			{"dim", pInt(1, "Width of each position's vector, matching the stream it is added to")},
		},
		Ports:      Ports{In: map[string]PortSpec{"x": Port("... dim")}, Out: map[string]PortSpec{"y": Port("... dim")}},
		ParamCount: func(r *Resolved) float64 { return r.Num("max_seq") * r.Num("dim") },
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			return FlopsPerToken{Elementwise: r.Num("dim")}
		},
		Retains: noRetains,
		Docs: BlockDocs{
			Name:    "position embedding",
			Summary: "Learned absolute position embedding, added to the token embedding (GPT-2 style).",
			Formula: "params = max_seq * dim",
		},
	},
	{
		Kind: "primitive", Type: "learned_tokens", Category: "embedding",
		Params: ParamList{
			{"count", pIntD(1, 1, "How many distinct vectors are learned")},
			{"dim", pInt(1, "Width of each learned vector, matching the stream it joins")},
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
			Name: "learned tokens",
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
			{"in_features", pInt(1, "Width in")},
			{"out_features", pInt(1, "Width out")},
			{"bias", pBool(false, "Learn a per-output constant as well as the matrix")},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": {Shape: "... in_features", Dtype: "real", Anchor: "flow"}},
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
			Name:    "linear",
			Summary: "Dense projection.",
			Formula: "params = in*out (+out with bias); FLOPs/token = 2*in*out; saves its input for the weight gradient",
		},
	},
	{
		Kind: "primitive", Type: "lm_head", Category: "head",
		Params: ParamList{
			{"vocab", pInt(1, "How many tokens it scores")},
			{"dim", pInt(1, "Width of the stream it reads")},
			{"tied", pBool(false, "Share weights with the token embedding")},
			{"bias", pBool(false, "Learn a per-token constant as well as the matrix")},
			{"softcap", ParamSpec{Type: ParamNum, Default: 0.0, HasDefault: true,
				Doc: "Bound the logits to this magnitude with tanh; 0 leaves them alone"}},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": {Shape: "... dim", Dtype: "real", Anchor: "flow"}},
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
			Name:    "output projection",
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
			{"in_channels", pInt(1, "Channels arriving")},
			{"out_channels", pInt(1, "Channels produced, one per filter")},
			{"kernel", pIntD(3, 1, "Side of the square filter")},
			{"stride", pIntD(1, 1, "How far the filter moves between positions; 2 halves the feature map")},
			{"padding", pIntD(0, 0, "Zeros added around each edge before convolving")},
			{"groups", pIntD(1, 1, "Grouped convolution; equal to in_channels is depthwise")},
			{"bias", pBool(true, "Learn a per-output-channel constant as well as the filters")},
			{"in_h", pInt(1, "Height of the incoming feature map")},
			{"in_w", pInt(1, "Width of the incoming feature map")},
			{"act", pEnum([]string{"identity", "relu", "relu2", "gelu", "gelu_tanh", "silu"}, "identity",
				"Activation applied to the output, as a convnet always does")},
		},
		PortsFn: func(r *Resolved) Ports {
			oh := convOut(r.Int("in_h"), r.Int("kernel"), r.Int("stride"), r.Int("padding"))
			ow := convOut(r.Int("in_w"), r.Int("kernel"), r.Int("stride"), r.Int("padding"))
			return Ports{
				In: map[string]PortSpec{"x": {
					Shape: fmt.Sprintf("B %d %d %d",
						r.Int("in_channels"), r.Int("in_h"), r.Int("in_w")),
					Dtype: "real", Anchor: "flow",
				}},
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
			Name:    "2D convolution",
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
			{"channels", pInt(1, "Channels passing through; pooling does not change them")},
			{"kernel", pIntD(2, 1, "Side of the square window")},
			{"stride", pIntD(2, 1, "How far the window moves between positions")},
			{"padding", pIntD(0, 0, "Cells added around each edge before pooling")},
			{"in_h", pInt(1, "Height of the incoming feature map")},
			{"in_w", pInt(1, "Width of the incoming feature map")},
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
			Name:    "max pool",
			Summary: "Spatial max pooling.",
			Formula: "no parameters; H_out = floor((H + 2*padding - k)/stride) + 1",
		},
	},
	{
		Kind: "primitive", Type: "flatten2d", Category: "shape",
		Params: ParamList{
			{"channels", pInt(1, "Channels arriving from the convolution stack")},
			{"in_h", pInt(1, "Height of the incoming feature map")},
			{"in_w", pInt(1, "Width of the incoming feature map")},
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
		Docs: BlockDocs{Name: "flatten", Summary: "Folds a feature map into one vector per sample."},
	},

	// --- normalisation and elementwise --------------------------------------
	{
		Kind: "primitive", Type: "rmsnorm", Category: "norm",
		Params: ParamList{
			{"dim", pInt(1, "Width normalized over")},
			{"eps", pNum(1e-5, "Added to the mean square before the square root, against dividing by zero")},
			{"scale", pBool(true, "Learned per-channel gain")},
		},
		Ports: Ports{In: map[string]PortSpec{"x": {Shape: "... dim", Dtype: "real", Anchor: "flow"}}, Out: map[string]PortSpec{"y": Port("... dim")}},
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
		Docs:    BlockDocs{Name: "RMS norm", Summary: "Root-mean-square layer norm.", Formula: "params = dim (scale only, no bias)"},
	},
	{
		Kind: "primitive", Type: "layernorm", Category: "norm",
		Params: ParamList{
			{"dim", pInt(1, "Width normalized over")},
			{"eps", pNum(1e-5, "Added to the variance before the square root, against dividing by zero")},
			{"bias", pBool(true, "Learn a shift as well as a gain")},
		},
		Ports: Ports{In: map[string]PortSpec{"x": {Shape: "... dim", Dtype: "real", Anchor: "flow"}}, Out: map[string]PortSpec{"y": Port("... dim")}},
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
		Docs:    BlockDocs{Name: "layer norm", Summary: "Standard layer norm.", Formula: "params = 2*dim with bias, dim without"},
	},
	{
		Kind: "primitive", Type: "activation", Category: "elementwise",
		Params: ParamList{
			{"kind", pEnum([]string{"silu", "gelu", "gelu_tanh", "relu", "relu2", "tanh", "sigmoid", "identity"}, "silu", "Which nonlinearity")},
			{"dim", pInt(1, "Width, used for the memory and elementwise estimates")},
		},
		Ports:      Ports{In: map[string]PortSpec{"x": {Shape: "... dim", Dtype: "real", Anchor: "flow"}}, Out: map[string]PortSpec{"y": Port("... dim")}},
		ParamCount: noParams,
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			cost, ok := elementwiseCost[r.Str("kind")]
			if !ok {
				cost = 4
			}
			return FlopsPerToken{Elementwise: cost * r.Num("dim")}
		},
		Retains: func(*Resolved) []string { return []string{"x"} },
		Docs:    BlockDocs{Name: "activation", Summary: "Pointwise nonlinearity."},
	},
	{
		Kind: "primitive", Type: "add", Category: "elementwise",
		Params: ParamList{
			{"dim", pInt(1, "Width of both operands, for the elementwise cost")},
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
		Docs:    BlockDocs{Name: "add", Summary: "Elementwise sum, the residual connection."},
	},
	{
		Kind: "primitive", Type: "gate", Category: "elementwise",
		Params: ParamList{
			{"dim", pInt(1, "Width of the stream being scaled")},
		},
		Ports: Ports{
			In: map[string]PortSpec{
				"x": Port("... dim"),
				// One value per token, broadcast across the width. This is why
				// it is not `mul`: that one is elementwise and says both sides
				// are the same width, which here they are not.
				"g": {Shape: "... 1", Anchor: "side",
					Doc: "One value per token, scaling the whole stream"},
			},
			Out: map[string]PortSpec{"y": Port("... dim")},
		},
		ParamCount: noParams,
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			return FlopsPerToken{Elementwise: r.Num("dim")}
		},
		// Each side is needed to differentiate the other.
		Retains: func(*Resolved) []string { return []string{"x", "g"} },
		Docs: BlockDocs{
			Name: "gate",
			Summary: "Scale a stream by one value per token. What a shared expert's gate does, " +
				"where the gate is a single learned direction rather than a matrix.",
			Formula: "y = g * x, g broadcast across the width",
		},
	},
	{
		Kind: "primitive", Type: "mix", Category: "elementwise",
		Params: ParamList{
			{"dim", pInt(1, "Width of both streams")},
		},
		Ports: Ports{
			In: map[string]PortSpec{
				"a": Port("... dim"),
				// The one being mixed in, which a figure draws arriving from
				// the side the way it draws a residual.
				"b": {Shape: "... dim", Anchor: "side"},
			},
			Out: map[string]PortSpec{"y": Port("... dim")},
		},
		// Two scalars, whatever the width: the mixture is learned, not the map.
		ParamCount: func(*Resolved) float64 { return 2 },
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			// Two multiplies and an add, per element.
			return FlopsPerToken{Elementwise: 3 * r.Num("dim")}
		},
		// Each operand is needed to differentiate the other's weight.
		Retains: func(*Resolved) []string { return []string{"a", "b"} },
		Docs: BlockDocs{
			Name: "learned mix",
			Summary: "Weighted sum of two streams, with both weights learned. What mixes a value " +
				"embedding into the values, and what mixes a U-net skip back into a later layer.",
			Formula: "y = w0 * a + w1 * b, w0 and w1 scalars",
			Refs:    []string{"https://github.com/KellerJordan/modded-nanogpt"},
		},
	},
	{
		Kind: "primitive", Type: "mul", Category: "elementwise",
		Params: ParamList{
			{"dim", pInt(1, "Width of both operands, for the elementwise cost")},
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
		Docs:    BlockDocs{Name: "multiply", Summary: "Elementwise product, the gate in a gated MLP."},
	},
	{
		Kind: "primitive", Type: "rearrange", Category: "shape",
		Params: ParamList{
			{"from", pPattern("B T (H dh)", "The shape arriving, named")},
			{"to", pPattern("B H T dh", "The same axes, regrouped")},
		},
		PortsFn: func(r *Resolved) Ports {
			return Ports{
				In:  map[string]PortSpec{"x": Port(r.Str("from"))},
				Out: map[string]PortSpec{"y": Port(r.Str("to"))},
			}
		},
		ParamCount: noParams, Flops: noFlops, Retains: noRetains,
		Docs: BlockDocs{
			Name:    "reshape",
			Summary: "Reshape or permute, written in einops notation.",
			Formula: "No parameters and no FLOPs; the product of the dimensions must be preserved",
		},
	},

	// --- attention ----------------------------------------------------------
	{
		Kind: "primitive", Type: "rope", Category: "position",
		Params: ParamList{
			{"heads", pInt(1, "Heads the rotation is applied across")},
			{"head_dim", pInt(2, "Width of one head; the rotation pairs its dimensions")},
			{"theta", pNum(10000, "Base of the frequency ladder; a larger one reaches further before wrapping")},
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
			Name:    "rotary positions",
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
			{"causal", pBool(true, "Mask out every position after the current one")},
			{"window", ParamSpec{Type: ParamInt, Default: 0.0, HasDefault: true, Doc: "Sliding-window width; 0 means full attention"}},
			{"flash", pBool(true, "Memory-efficient kernel that never materializes the score matrix")},
			{"cache", pBool(true, "Whether this block owns the inference cache. Latent attention caches a compressed vector instead.")},
			{"logit_softcap", ParamSpec{Type: ParamNum, Default: 0.0, HasDefault: true,
				Doc: "Bound the attention scores to this magnitude with tanh; 0 leaves them alone. FlashAttention 2.6 and later caps inside the fused kernel; PyTorch's own scaled_dot_product_attention cannot."}},
			{"mask", maskSpec()},
			{"score", scoreSpec()},
			{"sinks", sinksSpec()},
		},
		PortsFn: func(r *Resolved) Ports {
			v := "head_dim"
			if r.Num("v_head_dim") != 0 {
				v = "v_head_dim"
			}
			return Ports{
				In: map[string]PortSpec{
					"q": {Shape: "B heads T head_dim", Dtype: "real", Anchor: "flow"},
					"k": {Shape: "B kv_heads T head_dim", Dtype: "real", Anchor: "flow"},
					"v": {Shape: "B kv_heads T " + v, Dtype: "real", Anchor: "flow"},
				},
				Out: map[string]PortSpec{"y": Port("B heads T " + v)},
			}
		},
		// A sink is one learned score per query head.
		ParamCount: func(r *Resolved) float64 {
			if r.Bool("sinks") {
				return r.Num("heads")
			}
			return 0
		},
		Flops: func(r *Resolved, c AnalysisCtx) FlopsPerToken {
			a := AttentionOf(r)
			// A kernel that skips what the mask removes multiplies each query
			// by the keys it keeps: half of them when causal, the window when
			// there is one, and whatever share the design's own mask leaves.
			// A profiler counts the operator as if nothing were masked, since
			// its shape does not depend on the mask.
			keys := a.KeysPerQuery(c.T, c.B)
			perKey := 4 * r.Num("heads") * r.Num("head_dim")
			f := FlopsPerToken{FwdSeq: perKey * keys, FwdSeqUnmasked: perKey * c.T}
			if s := a.Score(); s != nil {
				// On the scores the kernel computes: a fused one changes them
				// inside the blocks it does not skip, so the mask counts here
				// exactly as it does for the matmuls. A cap is a divide, a
				// tanh and a multiply, which is softcapCost.
				f.Elementwise = attnexpr.Cost(s) * keys * r.Num("heads")
			}
			if a.Sinks {
				// The sink joins the denominator after the kernel: each output
				// is rescaled by sigmoid(lse - sink), one multiply an element.
				vDim := r.Num("v_head_dim")
				if vDim == 0 {
					vDim = r.Num("head_dim")
				}
				f.Elementwise += r.Num("heads") * vDim
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
			if c.Flash && r.Bool("flash") {
				// A fused kernel keeps the output and the log-sum-exp
				// statistics only — with a cap as well, since FlashAttention
				// 2.6 applies it inside the kernel.
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
			a := AttentionOf(r)
			out = append(out, a.constraints(r)...)
			// With an expression the cap is part of the score function
			// FlexAttention compiles, which SDPA-06 says; without one it is
			// FlashAttention's.
			if r.Num("logit_softcap") != 0 && r.Bool("flash") && !a.Flex() {
				out = append(out, BlockFinding{
					ID: "SDPA-03", Severity: "info", Param: "logit_softcap",
					Message: "capping the attention scores needs a kernel that caps inside it: " +
						"FlashAttention 2.6 or later does, PyTorch's own scaled_dot_product_attention " +
						"does not. This layer is counted as that fused kernel, and the generated model " +
						"uses it when flash-attn is installed.",
					Hint: "Without it the cap is computed eagerly and the score matrix is held for the " +
						"backward pass, which is the memory the fused count leaves out.",
				})
			}
			return out
		},
		Docs: BlockDocs{
			Name:    "scaled dot-product attention",
			Summary: "Scaled dot-product attention core. Covers MHA, GQA and MQA through kv_heads.",
			Formula: "FLOPs/token = 4*keys*heads*head_dim, keys = T/2 causal, W - W^2/2T in a causal window of W, times the share a mask keeps; KV cache = 2*kv_heads*head_dim*bytes per token",
			Refs:    []string{"https://arxiv.org/abs/2305.13245", "https://arxiv.org/abs/2205.14135"},
		},
	},

	// --- mixture of experts -------------------------------------------------
	{
		Kind: "primitive", Type: "topk_router", Category: "moe",
		Params: ParamList{
			{"d_model", pInt(1, "Width of the stream it scores from")},
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
			Name:    "router",
			Summary: "Chooses which experts each token is sent to.",
			Formula: "params = d_model * experts (+ experts with a routing bias)",
			Refs:    []string{"https://arxiv.org/abs/2401.06066"},
		},
	},
	{
		Kind: "primitive", Type: "weighted_sum", Category: "moe",
		Params: ParamList{
			{"dim", pInt(1, "Width of each expert's output")},
			{"n", pInt(1, "How many contributions are combined")},
		},
		Ports: Ports{
			In: map[string]PortSpec{
				"x": Port("... dim"),
				// What the router produced, which the router declares `float`.
				"weights": {Shape: "... n", Dtype: "float", Anchor: "flow"},
			},
			Out: map[string]PortSpec{"y": Port("... dim")},
		},
		ParamCount: noParams,
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			return FlopsPerToken{Elementwise: r.Num("dim") * r.Num("n")}
		},
		Retains: noRetains,
		Docs:    BlockDocs{Name: "combine experts", Summary: "Combines the chosen experts' outputs using the router's weights."},
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
			Name:    "split",
			Summary: "Cuts a tensor into pieces along its last dimension.",
			Formula: "No parameters and no FLOPs; the pieces must add up to the incoming width",
		},
	},
	{
		Kind: "primitive", Type: "shift", Category: "shape",
		Params: ParamList{
			{"dim", pInt(1, "Width of the stream, for the memory estimate")},
			{"by", ParamSpec{Type: ParamInt, Default: 1.0, HasDefault: true,
				Doc: "How many positions to move by; positive brings later tokens earlier, and the end is zero-filled"}},
		},
		// Shape-preserving, which is why nothing needed this until multi-token
		// prediction: a module that predicts the token after next reads the
		// embedding of the token after next, and over a whole training sequence
		// that is the same tensor moved along.
		Ports: Ports{
			In:  map[string]PortSpec{"x": Port("... dim")},
			Out: map[string]PortSpec{"y": Port("... dim")},
		},
		ParamCount: noParams,
		Flops:      noFlops,
		// A copy, so the backward pass needs nothing kept: the gradient of a
		// shift is the opposite shift.
		Retains: noRetains,
		Constraints: func(r *Resolved) []BlockFinding {
			if r.Num("by") < 0 {
				return []BlockFinding{{
					ID: "SHIFT-01", Severity: "error", Param: "by",
					Message: "shifting backwards would let a token see its own future; use a positive amount",
				}}
			}
			return nil
		},
		Docs: BlockDocs{
			Name:    "shift",
			Summary: "Moves a sequence along by a fixed number of positions, zero-filling the end.",
			Formula: "y[t] = x[t + by], and zero where t + by runs past the end",
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
		Docs: BlockDocs{Name: "concatenate", Summary: "Joins tensors along one dimension."},
	},
	{
		Kind: "primitive", Type: "expand_heads", Category: "shape",
		Params: ParamList{
			{"heads", pInt(1, "How many copies of each key/value head to make")},
			{"dim", pInt(1, "Width of one head")},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": Port("B T dim")},
			Out: map[string]PortSpec{"y": Port("B heads T dim")},
		},
		ParamCount: noParams, Flops: noFlops,
		// A broadcast view costs nothing to keep.
		Retains: noRetains,
		Docs: BlockDocs{
			Name:    "share across heads",
			Summary: "Shares one tensor across every attention head, as latent attention does with its rotary key.",
		},
	},
	{
		Kind: "primitive", Type: "kv_latent_cache", Category: "attention",
		Params: ParamList{
			{"dim", pInt(1, "Width of the cached vector per token per layer")},
			{"decompressed_dim", pIntD(0, 0,
				"Width an engine holds instead when it materializes keys and values per head; 0 means there is no other way to hold it")},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": Port("... dim")},
			Out: map[string]PortSpec{"y": Port("... dim")},
		},
		ParamCount: noParams, Flops: noFlops, Retains: noRetains,
		StateBytes: func(r *Resolved, c AnalysisCtx) StateBytes {
			return StateBytes{
				PerToken:             r.Num("dim") * c.Bytes,
				PerTokenDecompressed: r.Num("decompressed_dim") * c.Bytes,
			}
		},
		Docs: BlockDocs{
			Name: "cached latent",
			Summary: "Marks the compressed vector that latent attention caches instead of keys and values. " +
				"The compression only pays off under a kernel that scores in latent space; an engine that " +
				"decompresses holds the full multi-head cache, which the analysis reports beside it.",
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
			{"bias", pBool(true, "Learn a per-channel constant as well as the kernel")},
		},
		Ports: Ports{
			In:  map[string]PortSpec{"x": {Shape: "... channels", Dtype: "real", Anchor: "flow"}},
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
			Name:    "depthwise convolution",
			Summary: "Short depthwise convolution over time, used before a state-space scan.",
			Formula: "params = channels * kernel (+ channels with bias)",
			Refs:    []string{"https://arxiv.org/abs/2405.21060"},
		},
	},
	{
		Kind: "primitive", Type: "gated_delta_scan", Category: "ssm",
		Params: ParamList{
			{"heads", pInt(1, "Linear-attention heads, which is the number of key heads")},
			{"head_dim", pInt(1, "Width of a query/key head")},
			{"v_head_dim", pIntD(0, 0, "Width of a value head; 0 means the same as head_dim")},
			{"value_heads", pIntD(0, 0,
				"Value heads, when there are more of them than key heads; 0 means the same number")},
		},
		PortsFn: func(r *Resolved) Ports {
			v := "head_dim"
			if r.Num("v_head_dim") != 0 {
				v = "v_head_dim"
			}
			// The value side counts value heads, which need not be the key
			// heads: the recurrence carries one state per value head, and
			// Qwen3-Next has twice as many of those as it has keys.
			hv := "heads"
			if r.Num("value_heads") != 0 {
				hv = "value_heads"
			}
			return Ports{
				In: map[string]PortSpec{
					"q": Port("B heads T head_dim"),
					"k": Port("B heads T head_dim"),
					"v": Port("B " + hv + " T " + v),
					// The decay and the write strength, one scalar each per
					// value head. Written out rather than with an ellipsis,
					// because the other three ports name their batch and a
					// pattern that binds `...` to `B T` would disagree with a
					// pattern that binds nothing.
					"gates": {Shape: "B T (2*" + hv + ")", Anchor: "side"},
				},
				Out: map[string]PortSpec{"y": Port("B " + hv + " T " + v)},
			}
		},
		// A decay bias and a log-decay scale per head, the way Mamba-2 carries
		// its dt_bias and A_log.
		// Two scalars per *value* head: the recurrence runs one state per value
		// head, so a design with twice as many of those has twice as many.
		ParamCount: func(r *Resolved) float64 { return 2 * scanHeads(r) },
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			// Per head per token: S k (2 dk dv), the outer product that removes
			// it (dk dv), the scale and subtract (2 dk dv), the write (dk dv),
			// and the read S q (2 dk dv). Eight, and approximate in the same way
			// ssd_scan's constants are: the paper gives a recurrence, not a
			// count, and a chunked implementation trades some of these for
			// matmuls.
			return FlopsPerToken{Fwd: 8 * r.Num("heads") * r.Num("head_dim") * vHeadDim(r)}
		},
		Retains: func(*Resolved) []string { return []string{"q", "k", "v", "gates"} },
		// A matrix per head, and it does not grow: this is the whole point of
		// linear attention, and why a hybrid stack caches so much less.
		StateBytes: func(r *Resolved, c AnalysisCtx) StateBytes {
			return StateBytes{PerSequence: r.Num("heads") * r.Num("head_dim") * vHeadDim(r) * c.Bytes}
		},
		Docs: BlockDocs{
			Name:    "gated delta rule",
			Summary: "Gated DeltaNet recurrence: linear attention whose state is overwritten by a delta rule and decayed by a gate. Linear in sequence length, with a state fixed per sequence.",
			Formula: "S_t = S_{t-1}(a_t (I - b_t k_t k_t^T)) + b_t v_t k_t^T; o_t = S_t q_t",
			Refs:    []string{"https://arxiv.org/abs/2412.06464"},
		},
	},
	{
		Kind: "primitive", Type: "selective_scan", Category: "ssm",
		Params: ParamList{
			{"d_inner", pInt(1, "Width of the state-space stream")},
			{"state", pInt(1, "Recurrent state width per channel (Mamba's N)")},
		},
		Ports: Ports{
			In: map[string]PortSpec{
				"x":  Port("... d_inner"),
				"dt": Port("... d_inner"),
				// B and C are shared by every channel, which is what makes the
				// state cheap: one pair per token rather than one per channel.
				"b": {Shape: "... state", Anchor: "side", Doc: "Input gate, shared across channels"},
				"c": {Shape: "... state", Anchor: "side", Doc: "Output gate, shared across channels"},
			},
			Out: map[string]PortSpec{"y": Port("... d_inner")},
		},
		// The decay per channel and state, and the skip per channel: Mamba's
		// A_log and D.
		ParamCount: func(r *Resolved) float64 {
			return r.Num("d_inner")*r.Num("state") + r.Num("d_inner")
		},
		Flops: func(r *Resolved, _ AnalysisCtx) FlopsPerToken {
			// Linear in the sequence: each step discounts the state, writes into
			// it and reads one vector out. The constant is approximate, as the
			// paper gives asymptotics rather than counts.
			return FlopsPerToken{Fwd: 6 * r.Num("d_inner") * r.Num("state")}
		},
		Retains: func(*Resolved) []string { return []string{"x", "dt", "b", "c"} },
		// Fixed per sequence, which is the whole point: no cache grows with the
		// context.
		StateBytes: func(r *Resolved, c AnalysisCtx) StateBytes {
			return StateBytes{PerSequence: r.Num("d_inner") * r.Num("state") * c.Bytes}
		},
		Docs: BlockDocs{
			Name: "selective scan",
			Summary: "Mamba's selective scan: a recurrence whose decay and gates are read from the " +
				"token rather than fixed, so what it keeps depends on what it sees.",
			Formula: "params = d_inner*state + d_inner (A_log and D); state = d_inner*state per sequence",
			Refs:    []string{"https://arxiv.org/abs/2312.00752"},
		},
	},
	{
		Kind: "primitive", Type: "ssd_scan", Category: "ssm",
		Params: ParamList{
			{"d_inner", pInt(1, "Width of the state-space stream")},
			{"heads", pInt(1, "State-space heads, each with a recurrent state of its own")},
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
			Name:    "state-space scan",
			Summary: "Mamba-2 state-space scan. Linear in sequence length, and its state is fixed per sequence rather than growing per token.",
			Formula: "params = 3*heads; state = heads*head_dim*state*bytes per sequence; FLOPs are approximate",
			Refs:    []string{"https://arxiv.org/abs/2405.21060"},
		},
	},
}

// vHeadDim is the value head width, which defaults to the query/key width.
func vHeadDim(r *Resolved) float64 {
	if v := r.Num("v_head_dim"); v != 0 {
		return v
	}
	return r.Num("head_dim")
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

// scanHeads is how many states a gated delta scan carries: one per value head,
// which is not always the number of key heads.
func scanHeads(r *Resolved) float64 {
	if n := r.Num("value_heads"); n > 0 {
		return n
	}
	return r.Num("heads")
}
