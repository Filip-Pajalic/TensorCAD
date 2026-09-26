package catalog

// What a parameter is called where a person reads it.
//
// A parameter's name is an identifier: it is what a document writes, what a
// path and an MCP call use, and what a generated class takes as an argument,
// so it stays `ffn_hidden` and `kv_heads`. None of that is a reason to make
// somebody meeting the editor for the first time learn it. The label is what
// the inspector leads with and the name sits beside it, smaller, for the
// person who has to type it.
//
// Labels are keyed by the name, because a name means the same thing on nearly
// every block that has it — `heads` is heads everywhere — and a table keyed by
// block would say that eighty times. The few names that mean different things
// on different blocks (`dim`, `count`, `by`, `scale`) are overridden per block.
//
// A label is short, in sentence case, and names the quantity rather than
// describing it: the documentation line under the field is where the
// description goes.

var paramLabels = map[string]string{
	"act":                "Activation",
	"attention":          "Attention",
	"attn_bias":          "Attention bias",
	"attn_o_bias":        "Output projection bias",
	"axis":               "Axis",
	"bias":               "Bias",
	"buckets":            "Distance buckets",
	"by":                 "By",
	"cache":              "Owns the cache",
	"causal":             "Causal",
	"channels":           "Channels",
	"chunk":              "Chunk length",
	"conv_bias":          "Convolution bias",
	"conv_kernel":        "Convolution width",
	"count":              "Count",
	"cross":              "Cross-attention",
	"cross_attention":    "Cross-attention",
	"d_inner":            "Inner width",
	"d_model":            "Model width",
	"decompressed_dim":   "Decompressed width",
	"dim":                "Width",
	"dt_rank":            "Timestep rank",
	"dtype":              "Data type",
	"eps":                "Epsilon",
	"expand":             "Expansion",
	"expert_hidden":      "Expert width",
	"experts":            "Experts",
	"ffn_hidden":         "Feed-forward width",
	"flash":              "Fused kernel",
	"from":               "Shape in",
	"groups":             "Groups",
	"head_dim":           "Head width",
	"heads":              "Heads",
	"hidden":             "Hidden width",
	"in_channels":        "Channels in",
	"in_features":        "Width in",
	"in_h":               "Height in",
	"in_w":               "Width in",
	"kernel":             "Kernel size",
	"kind":               "Function",
	"kv_heads":           "Key/value heads",
	"kv_lora":            "Key/value latent width",
	"lambda_init":        "Starting lambda",
	"logit_softcap":      "Score cap",
	"mask":               "Mask",
	"max_seq":            "Longest sequence",
	"memory_dim":         "Memory width",
	"mlp":                "Feed-forward",
	"mlp_bias":           "Feed-forward bias",
	"n":                  "Inputs",
	"nope_dim":           "Unrotated head width",
	"norm":               "Norm",
	"norm_bias":          "Norm bias",
	"norm_eps":           "Norm epsilon",
	"norm_inputs":        "Normalize the gates",
	"normalize":          "Renormalize weights",
	"o_bias":             "Output projection bias",
	"out_channels":       "Channels out",
	"out_features":       "Width out",
	"output_gate":        "Output gate",
	"padding":            "Padding",
	"ports":              "Ports",
	"positions":          "Positions per document",
	"post_norm":          "Norm after each sublayer",
	"q_lora":             "Query latent width",
	"qk_norm":            "Normalize queries and keys",
	"role":               "Role",
	"rope":               "Rotary positions",
	"rope_dim":           "Rotary head width",
	"router_bias":        "Router bias",
	"scale":              "Score scale",
	"scaling":            "Rotary scaling",
	"score":              "Score",
	"shape":              "Shape",
	"shared_expert_gate": "Gate the shared expert",
	"shared_experts":     "Shared experts",
	"shared_values":      "Shared values",
	"sinks":              "Attention sinks",
	"sizes":              "Piece sizes",
	"softcap":            "Logit cap",
	"state":              "State width",
	"stride":             "Stride",
	"talking_heads":      "Talking heads",
	"theta":              "Frequency base",
	"tied":               "Tied",
	"to":                 "Shape out",
	"tokens":             "Length",
	"top_k":              "Experts per token",
	"v_dim":              "Value head width",
	"v_head_dim":         "Value head width",
	"value_embeddings":   "Value embeddings",
	"value_heads":        "Value heads",
	"vocab":              "Vocabulary",
	"window":             "Window",
	"written_out":        "Written out",
	"xbc_width":          "Combined x, B and C width",
}

// blockParamLabels are the names that mean something different on one block.
var blockParamLabels = map[string]map[string]string{
	"attn_scores":     {"kv_heads": "Key heads"},
	"attn_values":     {"kv_heads": "Value heads"},
	"conv1d":          {"kernel": "Kernel width"},
	"diff_combine":    {"dim": "Value head width"},
	"embedding":       {"dim": "Embedding width", "tied": "Share the first embedding"},
	"expand_heads":    {"dim": "Head width", "heads": "Copies of each head"},
	"kv_latent_cache": {"dim": "Cached width"},
	"learned_tokens":  {"count": "Vectors"},
	"lm_head":         {"tied": "Tied to the embedding"},
	"mtp_head":        {"by": "Tokens ahead"},
	"repeat":          {"count": "Repeats"},
	"rmsnorm":         {"scale": "Learned gain"},
	"scale":           {"by": "Multiply by"},
	"shift":           {"by": "Positions"},
	"weighted_sum":    {"n": "Contributions"},
}

// advancedParams are the parameters most designs never touch.
//
// The inspector puts them under a closed Advanced heading, which opens by
// itself for a block that sets one, so a design that uses attention sinks
// still shows them without a click. Only parameters with a default can be
// advanced: a field a block cannot do without is not one to hide.
var advancedParams = map[string]bool{
	"attn_o_bias":        true,
	"axis":               true,
	"cache":              true,
	"chunk":              true,
	"conv_bias":          true,
	"cross":              true,
	"cross_attention":    true,
	"decompressed_dim":   true,
	"eps":                true,
	"flash":              true,
	"lambda_init":        true,
	"logit_softcap":      true,
	"mask":               true,
	"memory_dim":         true,
	"norm_eps":           true,
	"norm_inputs":        true,
	"o_bias":             true,
	"output_gate":        true,
	"positions":          true,
	"post_norm":          true,
	"qk_norm":            true,
	"router_bias":        true,
	"scale":              true,
	"scaling":            true,
	"score":              true,
	"shared_expert_gate": true,
	"shared_values":      true,
	"sinks":              true,
	"softcap":            true,
	"talking_heads":      true,
	"v_head_dim":         true,
	"value_embeddings":   true,
	"written_out":        true,
}

// blockBasics are advanced parameters that are ordinary on one block: an
// RMSNorm's gain is the only thing it has to say.
var blockBasics = map[string]map[string]bool{
	"rmsnorm":   {"scale": true, "eps": true},
	"layernorm": {"eps": true},
}

// activationLabels are how a nonlinearity is written in a paper.
var activationLabels = map[string]string{
	"silu":      "SiLU",
	"gelu":      "GELU",
	"gelu_tanh": "GELU, tanh approximation",
	"relu":      "ReLU",
	"relu2":     "squared ReLU",
	"tanh":      "tanh",
	"sigmoid":   "sigmoid",
	"identity":  "none",
}

// valueLabels are what an enum's values are called, by parameter name.
var valueLabels = map[string]map[string]string{
	"act":  activationLabels,
	"kind": activationLabels,
	"attention": {
		"gqa":  "grouped-query",
		"mla":  "latent (MLA)",
		"diff": "differential",
	},
	"mlp": {
		"gated": "gated",
		"dense": "plain",
		"moe":   "mixture of experts",
	},
	"norm": {
		"rmsnorm":   "RMS norm",
		"layernorm": "layer norm",
	},
	"role": {
		"tokens":    "tokens",
		"documents": "which document each position is in",
		"positions": "each position's place in its document",
	},
}

// label fills in what a person reads for every parameter of a built-in block.
func label(def *BlockDef) {
	for i := range def.Params {
		p := &def.Params[i]
		if p.Spec.Label == "" {
			p.Spec.Label = blockParamLabels[def.Type][p.Name]
		}
		if p.Spec.Label == "" {
			p.Spec.Label = paramLabels[p.Name]
		}
		if advancedParams[p.Name] && !blockBasics[def.Type][p.Name] && p.Spec.HasDefault {
			p.Spec.Advanced = true
		}
		if p.Spec.Type == ParamEnum && p.Spec.ValueLabels == nil {
			if labels, ok := valueLabels[p.Name]; ok {
				p.Spec.ValueLabels = map[string]string{}
				for _, v := range p.Spec.Values {
					if l, ok := labels[v]; ok {
						p.Spec.ValueLabels[v] = l
					}
				}
			}
		}
	}
}
