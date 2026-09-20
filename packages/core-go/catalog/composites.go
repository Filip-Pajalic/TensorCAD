// Composite blocks: named subgraphs of primitives.
//
// A composite carries no formulas. The analysis expands it and sums the
// primitives, which is why adding a new attention variant or feed-forward shape
// never requires new maths. Each expansion is a self-contained graph with
// boundary_in and boundary_out nodes matching the composite's declared ports,
// so the very same shape inference runs inside it.
package catalog

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/tensorcad/core/ir"
)

// Ex renders a parameter as an expression, parenthesising only when it has to.
//
// Composites pass parameters into the composites they expand into, so
// unconditional wrapping compounds: three levels turned "H" into "(((H)))". It
// counts the same and reads like line noise everywhere it is shown.
func Ex(v any, fallback string) string {
	switch t := v.(type) {
	case nil:
		return fallback
	case float64:
		return num(t)
	case int:
		return num(float64(t))
	case bool:
		if t {
			return "1"
		}
		return "0"
	case string:
		text := strings.TrimSpace(t)
		if text == "" {
			return fallback
		}
		if atomRe.MatchString(text) || numberRe.MatchString(text) || wrapped(text) {
			return text
		}
		return "(" + text + ")"
	}
	return fallback
}

var (
	atomRe   = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	numberRe = regexp.MustCompile(`^\d+(?:\.\d+)?$`)
)

// wrapped reports whether the text is already one balanced parenthesised group.
func wrapped(text string) bool {
	if len(text) < 2 || text[0] != '(' || text[len(text)-1] != ')' {
		return false
	}
	depth := 0
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 && i != len(text)-1 {
				return false
			}
		}
	}
	return depth == 0
}

// Expansion is the subgraph a composite stands for.
type Expansion struct {
	Nodes []ir.NodeDef
	Edges []ir.Edge
}

// boundary builds the entry and exit nodes of a composite expansion. The port
// names must match the composite's own declared ports, or the inner shapes
// never reach the outer graph.
func boundary(in, out map[string]any) (ir.NodeDef, ir.NodeDef) {
	return ir.NodeDef{ID: "_in", Type: "boundary_in", Params: map[string]any{"ports": in}},
		ir.NodeDef{ID: "_out", Type: "boundary_out", Params: map[string]any{"ports": out}}
}

// streamBoundary is the standard single-stream boundary: input x, output y.
func streamBoundary(width string) (ir.NodeDef, ir.NodeDef) {
	return boundary(
		map[string]any{"x": "... " + width},
		map[string]any{"y": "... " + width},
	)
}

func node(id, typ string, params map[string]any) ir.NodeDef {
	return ir.NodeDef{ID: id, Type: typ, Params: params}
}

func edge(from, to string) ir.Edge { return ir.Edge{from, to} }

// ropeSpec is shared by every block that may carry rotary embedding.
func ropeSpec() ParamSpec {
	return ParamSpec{Type: ParamObj, Default: nil, HasDefault: true,
		Doc: "RoPE settings, or null for no rotary embedding"}
}

// ropeOf reads the rope object a block was given, and its theta.
func ropeOf(r *Resolved) (map[string]any, bool) {
	m, ok := r.P["rope"].(map[string]any)
	return m, ok
}

func thetaOf(rope map[string]any) any {
	if t, ok := rope["theta"]; ok && t != nil {
		return t
	}
	return 10000.0
}

func scalingOf(rope map[string]any) any {
	if s, ok := rope["scaling"]; ok {
		return s
	}
	return nil
}

// Composites is every block that is an arrangement rather than a formula.
var Composites = []*BlockDef{gqaAttention, gatedMlp, denseMlp, mlaAttention, moeLayer, mamba2Block, transformerBlock, mtpHeadComposite, gatedDeltanetBlock}

// Containers carry the multipliers that make sparsity and stacking work.
var Containers = []*BlockDef{moeExperts, repeatContainer}

// Multipliers is what a container contributes to the totals.
//
// Two numbers, not one: total drives the parameter count and active drives
// FLOPs and activation memory. That single mechanism is the whole of what makes
// a mixture of experts sparse.
type Multipliers struct{ Total, Active float64 }

// MultipliersOf reads a container's multipliers.
func MultipliersOf(def *BlockDef, r *Resolved) Multipliers {
	switch def.Type {
	case "moe_experts":
		return Multipliers{Total: r.Num("experts"), Active: r.Num("top_k")}
	case "repeat":
		return Multipliers{Total: r.Num("count"), Active: r.Num("count")}
	}
	return Multipliers{Total: 1, Active: 1}
}

// --- attention --------------------------------------------------------------

var gqaAttention = &BlockDef{
	Kind: "composite", Type: "gqa_attention", Category: "attention",
	Params: ParamList{
		{"d_model", pInt(1, "Residual stream width")},
		{"heads", pInt(1, "Query heads")},
		{"kv_heads", pInt(1, "Key/value heads. Equal to heads gives MHA, 1 gives MQA")},
		{"head_dim", pInt(1, "Width of one head; query and key heads share it")},
		{"bias", pBool(false, "Bias on the q/k/v/o projections")},
		{"o_bias", ParamSpec{Type: ParamBool, Default: nil, HasDefault: true,
			Doc: "Override the output-projection bias (Qwen2.5 has qkv bias only)"}},
		{"causal", pBool(true, "Mask out every position after the current one")},
		{"window", ParamSpec{Type: ParamInt, Default: 0.0, HasDefault: true, Doc: "Sliding-window width; 0 means full attention"}},
		{"qk_norm", pBool(false, "RMSNorm on the query and key heads (Qwen3, Gemma 3)")},
		{"rope", ropeSpec()},
		{"flash", pBool(true, "Assume a memory-efficient kernel, which never builds the score matrix")},
		{"logit_softcap", ParamSpec{Type: ParamNum, Default: 0.0, HasDefault: true,
			Doc: "Bound the attention scores to this magnitude with tanh (Gemma 2); 0 leaves them alone"}},
		{"value_embeddings", pBool(false,
			"Take a second embedding of the same tokens on `ve` and mix it into the values (nanoGPT speedrun)")},
		{"output_gate", pBool(false,
			"Project a second query-width tensor and use it as a sigmoid gate on the attention output (Qwen3-Next)")},
	},
	PortsFn: func(r *Resolved) Ports {
		in := map[string]PortSpec{"x": Port("... d_model")}
		if r.Bool("value_embeddings") {
			// As wide as the values, not as wide as the stream: it is mixed
			// into `v` after the projection, so a design whose d_model differs
			// from kv_heads * head_dim gets told by the shape checker.
			in["ve"] = PortSpec{Shape: "... (kv_heads head_dim)", Anchor: "side",
				Doc: "A second embedding of the same tokens, mixed into the values"}
		}
		return Ports{In: in, Out: map[string]PortSpec{"y": Port("... d_model")}}
	},
	Constraints: func(r *Resolved) []BlockFinding {
		if int(r.Num("heads"))%int(r.Num("kv_heads")) == 0 {
			return nil
		}
		return []BlockFinding{{
			ID: "ATTN-01", Severity: "error", Param: "kv_heads",
			Message: fmt.Sprintf("heads (%s) must be divisible by kv_heads (%s)",
				num(r.Num("heads")), num(r.Num("kv_heads"))),
			Hint: "Grouped-query attention shares one key/value head across a whole group of query heads, so the groups have to come out even.",
		}}
	},
	Docs: BlockDocs{
		Summary: "Grouped-query attention. kv_heads = heads gives multi-head attention, kv_heads = 1 gives multi-query.",
		Formula: "params = d_model*heads*head_dim + 2*d_model*kv_heads*head_dim + heads*head_dim*d_model",
		Refs:    []string{"https://arxiv.org/abs/2305.13245"},
	},
}

func expandGQA(raw map[string]any, r *Resolved) Expansion {
	D := Ex(raw["d_model"], "0")
	H := Ex(raw["heads"], "0")
	KV := Ex(raw["kv_heads"], "0")
	dh := Ex(raw["head_dim"], "0")
	bias := r.Bool("bias")
	// A null o_bias means "whatever the other projections use".
	oBias := bias
	if v, ok := r.P["o_bias"].(bool); ok {
		oBias = v
	}
	qkNorm := r.Bool("qk_norm")
	rope, hasRope := ropeOf(r)

	ve := r.Bool("value_embeddings")
	inNode, outNode := streamBoundary(D)
	if ve {
		inNode, outNode = boundary(
			map[string]any{"x": "... " + D, "ve": fmt.Sprintf("B T (%s %s)", KV, dh)},
			map[string]any{"y": "... " + D},
		)
	}
	nodes := []ir.NodeDef{
		inNode,
		node("q_proj", "linear", map[string]any{"in_features": D, "out_features": H + "*" + dh, "bias": bias}),
		node("k_proj", "linear", map[string]any{"in_features": D, "out_features": KV + "*" + dh, "bias": bias}),
		node("v_proj", "linear", map[string]any{"in_features": D, "out_features": KV + "*" + dh, "bias": bias}),
		node("q_heads", "rearrange", map[string]any{
			"from": fmt.Sprintf("B T (%s %s)", H, dh), "to": fmt.Sprintf("B %s T %s", H, dh)}),
		node("k_heads", "rearrange", map[string]any{
			"from": fmt.Sprintf("B T (%s %s)", KV, dh), "to": fmt.Sprintf("B %s T %s", KV, dh)}),
		node("v_heads", "rearrange", map[string]any{
			"from": fmt.Sprintf("B T (%s %s)", KV, dh), "to": fmt.Sprintf("B %s T %s", KV, dh)}),
	}
	edges := []ir.Edge{
		edge("_in:x", "q_proj:x"), edge("_in:x", "k_proj:x"), edge("_in:x", "v_proj:x"),
		edge("q_proj:y", "q_heads:x"), edge("k_proj:y", "k_heads:x"), edge("v_proj:y", "v_heads:x"),
	}

	qTail, kTail := "q_heads:y", "k_heads:y"
	vTail := "v_heads:y"

	if ve {
		// A second look at the tokens, mixed into the values by two learned
		// scalars rather than added: the mixture is what is learned.
		nodes = append(nodes,
			node("ve_heads", "rearrange", map[string]any{
				"from": fmt.Sprintf("B T (%s %s)", KV, dh), "to": fmt.Sprintf("B %s T %s", KV, dh)}),
			node("v_mix", "mix", map[string]any{"dim": dh}))
		edges = append(edges,
			edge("_in:ve", "ve_heads:x"),
			edge(vTail, "v_mix:a"), edge("ve_heads:y", "v_mix:b"))
		vTail = "v_mix:y"
	}

	if qkNorm {
		nodes = append(nodes,
			node("q_norm", "rmsnorm", map[string]any{"dim": dh}),
			node("k_norm", "rmsnorm", map[string]any{"dim": dh}))
		edges = append(edges, edge(qTail, "q_norm:x"), edge(kTail, "k_norm:x"))
		qTail, kTail = "q_norm:y", "k_norm:y"
	}

	if hasRope {
		theta := thetaOf(rope)
		scaling := scalingOf(rope)
		nodes = append(nodes,
			node("rope_q", "rope", map[string]any{"heads": H, "head_dim": dh, "theta": theta, "scaling": scaling}),
			node("rope_k", "rope", map[string]any{"heads": KV, "head_dim": dh, "theta": theta, "scaling": scaling}))
		edges = append(edges, edge(qTail, "rope_q:x"), edge(kTail, "rope_k:x"))
		qTail, kTail = "rope_q:y", "rope_k:y"
	}

	nodes = append(nodes,
		node("attn", "sdpa", map[string]any{
			"heads": H, "kv_heads": KV, "head_dim": dh,
			"causal":        r.Bool("causal"),
			"window":        Ex(raw["window"], "0"),
			"flash":         !isFalse(r.P["flash"]),
			"logit_softcap": Ex(raw["logit_softcap"], "0"),
		}),
		node("o_merge", "rearrange", map[string]any{
			"from": fmt.Sprintf("B %s T %s", H, dh), "to": fmt.Sprintf("B T (%s %s)", H, dh)}),
		node("o_proj", "linear", map[string]any{
			"in_features": H + "*" + dh, "out_features": D, "bias": oBias}),
		outNode)

	edges = append(edges,
		edge(qTail, "attn:q"), edge(kTail, "attn:k"), edge(vTail, "attn:v"),
		edge("attn:y", "o_merge:x"))

	// A sigmoid gate on the merged output, from a projection of the stream as
	// wide as the queries. Qwen3-Next fuses it into `q_proj`, which is why that
	// matrix is twice the query width; counted separately here, because a
	// second matrix and a wider one weigh the same and only one of them says
	// what it is for.
	oTail := "o_merge:y"
	if r.Bool("output_gate") {
		qWidth := H + "*" + dh
		nodes = append(nodes,
			node("gate_proj", "linear", map[string]any{
				"in_features": D, "out_features": qWidth, "bias": bias}),
			node("gate_act", "activation", map[string]any{"kind": "sigmoid", "dim": qWidth}),
			node("out_gate", "mul", map[string]any{"dim": qWidth}))
		edges = append(edges,
			edge("_in:x", "gate_proj:x"), edge("gate_proj:y", "gate_act:x"),
			edge(oTail, "out_gate:a"), edge("gate_act:y", "out_gate:b"))
		oTail = "out_gate:y"
	}
	edges = append(edges, edge(oTail, "o_proj:x"), edge("o_proj:y", "_out:y"))

	return Expansion{Nodes: nodes, Edges: edges}
}

// --- feed-forward -----------------------------------------------------------

var gatedMlp = &BlockDef{
	Kind: "composite", Type: "gated_mlp", Category: "mlp",
	Params: ParamList{
		{"d_model", pInt(1, "Width of the residual stream")},
		{"hidden", pInt(1, "Intermediate width")},
		{"act", pEnum([]string{"silu", "gelu", "gelu_tanh", "relu", "relu2"}, "silu", "Which nonlinearity the gate passes through")},
		{"bias", pBool(false, "Learn a constant on each projection")},
	},
	Ports: Ports{
		In:  map[string]PortSpec{"x": Port("... d_model")},
		Out: map[string]PortSpec{"y": Port("... d_model")},
	},
	Docs: BlockDocs{
		Summary: "Gated feed-forward network (SwiGLU when act is silu, GeGLU when gelu).",
		Formula: "params = 3 * d_model * hidden",
		Refs:    []string{"https://arxiv.org/abs/2002.05202"},
	},
}

func expandGatedMlp(raw map[string]any, r *Resolved) Expansion {
	D := Ex(raw["d_model"], "0")
	F := Ex(raw["hidden"], "0")
	bias := r.Bool("bias")
	inNode, outNode := streamBoundary(D)
	return Expansion{
		Nodes: []ir.NodeDef{
			inNode,
			node("gate", "linear", map[string]any{"in_features": D, "out_features": F, "bias": bias}),
			node("up", "linear", map[string]any{"in_features": D, "out_features": F, "bias": bias}),
			node("act", "activation", map[string]any{"kind": r.Str("act"), "dim": F}),
			node("gated", "mul", map[string]any{"dim": F}),
			node("down", "linear", map[string]any{"in_features": F, "out_features": D, "bias": bias}),
			outNode,
		},
		Edges: []ir.Edge{
			edge("_in:x", "gate:x"), edge("_in:x", "up:x"),
			edge("gate:y", "act:x"), edge("act:y", "gated:a"), edge("up:y", "gated:b"),
			edge("gated:y", "down:x"), edge("down:y", "_out:y"),
		},
	}
}

var denseMlp = &BlockDef{
	Kind: "composite", Type: "dense_mlp", Category: "mlp",
	Params: ParamList{
		{"d_model", pInt(1, "Width of the residual stream")},
		{"hidden", pInt(1, "Width in the middle, where the nonlinearity is")},
		{"act", pEnum([]string{"gelu", "gelu_tanh", "relu", "relu2", "silu"}, "gelu", "Which nonlinearity")},
		{"bias", pBool(true, "Learn a constant on each projection")},
	},
	Ports: Ports{
		In:  map[string]PortSpec{"x": Port("... d_model")},
		Out: map[string]PortSpec{"y": Port("... d_model")},
	},
	Docs: BlockDocs{
		Summary: "Classic two-matrix feed-forward network (GPT-2, Nemotron-H).",
		Formula: "params = 2 * d_model * hidden (+ hidden + d_model with bias)",
	},
}

func expandDenseMlp(raw map[string]any, r *Resolved) Expansion {
	D := Ex(raw["d_model"], "0")
	F := Ex(raw["hidden"], "0")
	bias := r.Bool("bias")
	inNode, outNode := streamBoundary(D)
	return Expansion{
		Nodes: []ir.NodeDef{
			inNode,
			node("up", "linear", map[string]any{"in_features": D, "out_features": F, "bias": bias}),
			node("act", "activation", map[string]any{"kind": r.Str("act"), "dim": F}),
			node("down", "linear", map[string]any{"in_features": F, "out_features": D, "bias": bias}),
			outNode,
		},
		Edges: []ir.Edge{
			edge("_in:x", "up:x"), edge("up:y", "act:x"),
			edge("act:y", "down:x"), edge("down:y", "_out:y"),
		},
	}
}

// --- latent attention -------------------------------------------------------

var mlaAttention = &BlockDef{
	Kind: "composite", Type: "mla_attention", Category: "attention",
	Params: ParamList{
		{"d_model", pInt(1, "Width of the residual stream")},
		{"heads", pInt(1, "Query heads; latent attention has no separate key/value count")},
		{"q_lora", pInt(1, "Width of the compressed query (DeepSeek's q_lora_rank)")},
		{"kv_lora", pInt(1, "Width of the compressed key/value latent, which is what gets cached")},
		{"nope_dim", pInt(1, "Per-head width that carries no position information")},
		{"rope_dim", pInt(2, "Per-head width that carries the rotary embedding")},
		{"v_dim", pInt(1, "Per-head value width")},
		{"causal", pBool(true, "Mask out every position after the current one")},
		{"rope", ropeSpec()},
		{"bias", pBool(false, "Learn a constant on each projection")},
	},
	Ports: Ports{
		In:  map[string]PortSpec{"x": Port("... d_model")},
		Out: map[string]PortSpec{"y": Port("... d_model")},
	},
	Docs: BlockDocs{
		Summary: "Multi-head latent attention. Keys and values are compressed to one small vector per token, and only that vector is cached.",
		Formula: "params = d_model*q_lora + q_lora + q_lora*heads*(nope+rope) + " +
			"d_model*(kv_lora+rope) + kv_lora + kv_lora*heads*(nope+v_dim) + " +
			"heads*v_dim*d_model; cache = layers*(kv_lora+rope)*bytes per token",
		Refs: []string{"https://arxiv.org/abs/2405.04434", "https://arxiv.org/abs/2412.19437"},
	},
}

func expandMLA(raw map[string]any, r *Resolved) Expansion {
	D := Ex(raw["d_model"], "0")
	H := Ex(raw["heads"], "0")
	QL := Ex(raw["q_lora"], "0")
	KL := Ex(raw["kv_lora"], "0")
	NOPE := Ex(raw["nope_dim"], "0")
	ROPE := Ex(raw["rope_dim"], "0")
	VD := Ex(raw["v_dim"], "0")
	QK := NOPE + "+" + ROPE
	LATENT := KL + "+" + ROPE
	bias := r.Bool("bias")
	rope, _ := ropeOf(r)
	theta := thetaOf(rope)

	inNode, outNode := streamBoundary(D)
	nodes := []ir.NodeDef{
		inNode,
		// Query path: compress, normalise, expand back into heads.
		node("q_down", "linear", map[string]any{"in_features": D, "out_features": QL, "bias": bias}),
		node("q_norm", "rmsnorm", map[string]any{"dim": QL}),
		node("q_up", "linear", map[string]any{"in_features": QL, "out_features": fmt.Sprintf("%s*(%s)", H, QK), "bias": bias}),
		node("q_heads", "rearrange", map[string]any{
			"from": fmt.Sprintf("B T (%s (%s))", H, QK), "to": fmt.Sprintf("B %s T (%s)", H, QK)}),

		// Key/value path: one compressed vector per token is all that is cached.
		node("kv_down", "linear", map[string]any{"in_features": D, "out_features": LATENT, "bias": bias}),
		// What an engine that does not absorb the up-projections holds instead:
		// a key and a value per head, plus the rotary part, which is shared by
		// every head and so is cached once however many there are.
		node("latent", "kv_latent_cache", map[string]any{
			"dim":              LATENT,
			"decompressed_dim": fmt.Sprintf("%s*((%s)+(%s))+(%s)", H, NOPE, VD, ROPE),
		}),
		node("kv_split", "split", map[string]any{
			"from": fmt.Sprintf("B T (%s)", LATENT), "sizes": []any{KL, ROPE}}),
		node("kv_norm", "rmsnorm", map[string]any{"dim": KL}),
		node("k_up", "linear", map[string]any{"in_features": KL, "out_features": fmt.Sprintf("%s*(%s)", H, NOPE), "bias": bias}),
		node("k_nope_heads", "rearrange", map[string]any{
			"from": fmt.Sprintf("B T (%s (%s))", H, NOPE), "to": fmt.Sprintf("B %s T (%s)", H, NOPE)}),
		node("v_up", "linear", map[string]any{"in_features": KL, "out_features": fmt.Sprintf("%s*(%s)", H, VD), "bias": bias}),
		node("v_heads", "rearrange", map[string]any{
			"from": fmt.Sprintf("B T (%s (%s))", H, VD), "to": fmt.Sprintf("B %s T (%s)", H, VD)}),

		// The rotary part of the key is shared by every head.
		node("k_rope_shared", "expand_heads", map[string]any{"heads": H, "dim": ROPE}),
		node("k_rope", "rope", map[string]any{"heads": H, "head_dim": ROPE, "theta": theta}),
		node("k_cat", "concat", map[string]any{
			"to": fmt.Sprintf("B %s T (%s)", H, QK), "sizes": []any{NOPE, ROPE}}),

		node("attn", "sdpa", map[string]any{
			"heads": H, "kv_heads": H, "head_dim": QK, "v_head_dim": VD,
			"causal": r.Bool("causal"),
			// The latent node above owns the cache; counting it here too would
			// double-count it, and at the uncompressed size.
			"cache": false,
		}),
		node("o_merge", "rearrange", map[string]any{
			"from": fmt.Sprintf("B %s T (%s)", H, VD), "to": fmt.Sprintf("B T (%s (%s))", H, VD)}),
		node("o_proj", "linear", map[string]any{
			"in_features": fmt.Sprintf("%s*(%s)", H, VD), "out_features": D, "bias": bias}),
		outNode,
	}

	edges := []ir.Edge{
		edge("_in:x", "q_down:x"), edge("q_down:y", "q_norm:x"),
		edge("q_norm:y", "q_up:x"), edge("q_up:y", "q_heads:x"),
		edge("_in:x", "kv_down:x"), edge("kv_down:y", "latent:x"),
		edge("latent:y", "kv_split:x"), edge("kv_split:y0", "kv_norm:x"),
		edge("kv_norm:y", "k_up:x"), edge("k_up:y", "k_nope_heads:x"),
		edge("kv_norm:y", "v_up:x"), edge("v_up:y", "v_heads:x"),
		edge("kv_split:y1", "k_rope_shared:x"), edge("k_rope_shared:y", "k_rope:x"),
		edge("k_nope_heads:y", "k_cat:y0"), edge("k_rope:y", "k_cat:y1"),
		edge("q_heads:y", "attn:q"), edge("k_cat:y", "attn:k"), edge("v_heads:y", "attn:v"),
		edge("attn:y", "o_merge:x"), edge("o_merge:y", "o_proj:x"), edge("o_proj:y", "_out:y"),
	}
	return Expansion{Nodes: nodes, Edges: edges}
}

// --- mixture of experts -----------------------------------------------------

// moeExperts is a bank of experts. This is the whole of what makes a model
// sparse: every expert holds weights, but a token only pays for top_k of them.
var moeExperts = &BlockDef{
	Kind: "container", Type: "moe_experts", Category: "moe",
	Params: ParamList{
		{"experts", pInt(1, "How many expert copies exist")},
		{"top_k", pInt(1, "How many a single token passes through")},
	},
	Docs: BlockDocs{
		Summary: "A bank of experts. Its subgraph describes one expert.",
		Formula: "total params scale with experts; FLOPs and activations scale with top_k",
	},
}

var moeLayer = &BlockDef{
	Kind: "composite", Type: "moe_layer", Category: "moe",
	Params: ParamList{
		{"d_model", pInt(1, "Width of the residual stream")},
		{"experts", pInt(1, "Routed experts")},
		{"top_k", pInt(1, "Experts each token is sent to")},
		{"expert_hidden", pInt(1, "Hidden width of one expert")},
		{"shared_experts", pIntD(0, 0, "Experts every token always passes through")},
		{"shared_expert_gate", pBool(false,
			"Weight the shared expert's output by a sigmoid of one learned direction (Qwen)")},
		{"act", pEnum([]string{"silu", "gelu", "gelu_tanh", "relu", "relu2"}, "silu", "Which nonlinearity inside an expert")},
		{"bias", pBool(false, "Learn a constant on each expert's projections")},
		{"router_bias", pBool(false, "Give the router a per-expert bias, which is how DeepSeek balances load")},
		{"normalize", pBool(true, "Renormalize the chosen weights to sum to one")},
	},
	Ports: Ports{
		In:  map[string]PortSpec{"x": Port("... d_model")},
		Out: map[string]PortSpec{"y": Port("... d_model")},
	},
	Constraints: func(r *Resolved) []BlockFinding {
		if r.Num("top_k") <= r.Num("experts") {
			return nil
		}
		return []BlockFinding{{
			ID: "MOE-01", Severity: "error", Param: "top_k",
			Message: fmt.Sprintf("top_k (%s) cannot exceed the number of experts (%s)",
				num(r.Num("top_k")), num(r.Num("experts"))),
		}}
	},
	Docs: BlockDocs{
		Summary: "Sparse feed-forward layer: a router picks top_k of the experts for each token.",
		Formula: "total = router + experts*3*d_model*expert_hidden + shared*3*d_model*expert_hidden " +
			"(+ d_model when the shared expert is gated); active swaps experts for top_k",
		Refs: []string{"https://arxiv.org/abs/2401.06066", "https://arxiv.org/abs/2412.19437"},
	},
}

func expandMoeLayer(raw map[string]any, r *Resolved) Expansion {
	D := Ex(raw["d_model"], "0")
	E := Ex(raw["experts"], "0")
	K := Ex(raw["top_k"], "0")
	Fe := Ex(raw["expert_hidden"], "0")
	shared := r.Num("shared_experts")
	inNode, outNode := streamBoundary(D)

	expertIn, expertOut := boundary(
		map[string]any{"x": "... " + D}, map[string]any{"y": "... " + D})

	nodes := []ir.NodeDef{
		inNode,
		node("router", "topk_router", map[string]any{
			"d_model": D, "experts": E, "top_k": K,
			"bias":      r.Bool("router_bias"),
			"normalize": !isFalse(r.P["normalize"]),
		}),
		{
			ID: "experts", Type: "moe_experts",
			Params: map[string]any{"experts": E, "top_k": K},
			Graph: &ir.Graph{
				Nodes: []ir.NodeDef{
					expertIn,
					node("expert", "gated_mlp", map[string]any{
						"d_model": D, "hidden": Fe, "act": r.Str("act"), "bias": r.Bool("bias")}),
					expertOut,
				},
				Edges: []ir.Edge{edge("_in:x", "expert:x"), edge("expert:y", "_out:y")},
			},
		},
		node("combine", "weighted_sum", map[string]any{"dim": D, "n": K}),
	}
	edges := []ir.Edge{
		edge("_in:x", "router:x"), edge("_in:x", "experts:x"),
		edge("experts:y", "combine:x"), edge("router:weights", "combine:weights"),
	}

	if shared > 0 {
		// Several shared experts are one wider feed-forward network, which is
		// how DeepSeek implements them and gives the same parameter count.
		nodes = append(nodes,
			node("shared", "gated_mlp", map[string]any{
				"d_model": D, "hidden": fmt.Sprintf("%s*%s", num(shared), Fe),
				"act": r.Str("act"), "bias": r.Bool("bias")}),
			node("merge", "add", map[string]any{"dim": D}),
			outNode)

		sharedTail := "shared:y"
		if r.Bool("shared_expert_gate") {
			// One learned direction, squashed, scaling the whole shared output:
			// Qwen's `shared_expert_gate` is [1, d_model], which is a row rather
			// than a matrix and easy to leave out of an estimate by mistake.
			nodes = append(nodes,
				node("shared_gate", "linear", map[string]any{
					"in_features": D, "out_features": "1", "bias": false}),
				node("shared_gate_act", "activation", map[string]any{"kind": "sigmoid", "dim": "1"}),
				node("shared_scale", "gate", map[string]any{"dim": D}))
			edges = append(edges,
				edge("_in:x", "shared_gate:x"), edge("shared_gate:y", "shared_gate_act:x"),
				edge("shared:y", "shared_scale:x"), edge("shared_gate_act:y", "shared_scale:g"))
			sharedTail = "shared_scale:y"
		}
		edges = append(edges,
			edge("_in:x", "shared:x"), edge("combine:y", "merge:a"),
			edge(sharedTail, "merge:b"), edge("merge:y", "_out:y"))
	} else {
		nodes = append(nodes, outNode)
		edges = append(edges, edge("combine:y", "_out:y"))
	}
	return Expansion{Nodes: nodes, Edges: edges}
}

// --- state-space blocks -----------------------------------------------------

var mamba2Block = &BlockDef{
	Kind: "composite", Type: "mamba2_block", Category: "ssm",
	Params: ParamList{
		{"d_model", pInt(1, "Width of the residual stream")},
		{"expand", pIntD(2, 1, "Inner width as a multiple of d_model")},
		{"head_dim", pIntD(64, 1, "Width per state-space head")},
		{"state", pIntD(128, 1, "Recurrent state width per head")},
		{"groups", pIntD(1, 1, "Heads sharing one B/C projection")},
		{"conv_kernel", pIntD(4, 1, "Width of the short depthwise convolution before the scan")},
		{"conv_bias", pBool(true, "Learn a per-channel constant on that convolution")},
		{"bias", pBool(false, "Bias on the input and output projections")},
	},
	Ports: Ports{
		In:  map[string]PortSpec{"x": Port("... d_model")},
		Out: map[string]PortSpec{"y": Port("... d_model")},
	},
	Docs: BlockDocs{
		Summary: "Mamba-2 block. Its cost is linear in sequence length and it keeps a fixed state per sequence instead of a growing cache.",
		Formula: "params = d_model*(2*d_inner + 2*groups*state + heads) + conv + 3*heads + d_inner + d_inner*d_model",
		Refs:    []string{"https://arxiv.org/abs/2405.21060"},
	},
}

func expandMamba2(raw map[string]any, r *Resolved) Expansion {
	D := Ex(raw["d_model"], "0")
	E := Ex(raw["expand"], "2")
	P := Ex(raw["head_dim"], "64")
	N := Ex(raw["state"], "128")
	G := Ex(raw["groups"], "1")
	bias := r.Bool("bias")

	inner := E + "*" + D
	heads := fmt.Sprintf("(%s)/(%s)", inner, P)
	bc := fmt.Sprintf("2*(%s)*(%s)", G, N)
	xbc := fmt.Sprintf("(%s)+(%s)", inner, bc)
	inProj := fmt.Sprintf("2*(%s)+(%s)+(%s)", inner, bc, heads)

	inNode, outNode := streamBoundary(D)
	return Expansion{
		Nodes: []ir.NodeDef{
			inNode,
			node("in_proj", "linear", map[string]any{"in_features": D, "out_features": inProj, "bias": bias}),
			node("split", "split", map[string]any{
				"from": fmt.Sprintf("B T (%s)", inProj), "sizes": []any{inner, xbc, heads}}),
			node("gate_act", "activation", map[string]any{"kind": "silu", "dim": inner}),
			node("conv", "conv1d", map[string]any{
				"channels": xbc, "kernel": Ex(raw["conv_kernel"], "4"), "bias": !isFalse(r.P["conv_bias"])}),
			node("conv_act", "activation", map[string]any{"kind": "silu", "dim": xbc}),
			node("scan", "ssd_scan", map[string]any{
				"d_inner": inner, "heads": heads, "head_dim": P,
				"state": N, "groups": G, "xbc_width": xbc}),
			node("norm", "rmsnorm", map[string]any{"dim": inner}),
			node("gate", "mul", map[string]any{"dim": inner}),
			node("out_proj", "linear", map[string]any{"in_features": inner, "out_features": D, "bias": bias}),
			outNode,
		},
		Edges: []ir.Edge{
			edge("_in:x", "in_proj:x"), edge("in_proj:y", "split:x"),
			edge("split:y0", "gate_act:x"), edge("split:y1", "conv:x"),
			edge("conv:y", "conv_act:x"), edge("conv_act:y", "scan:xbc"),
			edge("split:y2", "scan:dt"), edge("scan:y", "norm:x"),
			edge("norm:y", "gate:a"), edge("gate_act:y", "gate:b"),
			edge("gate:y", "out_proj:x"), edge("out_proj:y", "_out:y"),
		},
	}
}

var gatedDeltanetBlock = &BlockDef{
	Kind: "composite", Type: "gated_deltanet_block", Category: "ssm",
	Params: ParamList{
		{"d_model", pInt(1, "Residual stream width")},
		{"heads", pInt(1, "Linear-attention heads, which is the number of key heads")},
		{"head_dim", pInt(1, "Width of a query/key head")},
		{"v_head_dim", pIntD(0, 0, "Width of a value head; 0 means the same as head_dim")},
		{"value_heads", pIntD(0, 0,
			"Value heads, when there are more of them than key heads; 0 means the same number")},
		{"conv_kernel", pIntD(4, 1, "Depthwise convolution over q, k and v, as Mamba-2 has one")},
		{"bias", pBool(false, "Bias on the input and output projections")},
		{"conv_bias", pBool(false, "Bias on the depthwise convolution")},
	},
	Ports: Ports{
		In:  map[string]PortSpec{"x": Port("... d_model")},
		Out: map[string]PortSpec{"y": Port("... d_model")},
	},
	Constraints: func(r *Resolved) []BlockFinding {
		if r.Num("v_head_dim") < 0 {
			return []BlockFinding{{
				ID: "GDN-01", Severity: "error", Param: "v_head_dim",
				Message: "v_head_dim must be non-negative; 0 means the same as head_dim",
			}}
		}
		return nil
	},
	Docs: BlockDocs{
		Summary: "Gated DeltaNet: linear attention in place of softmax attention, with a delta rule writing the state and a gate decaying it. Its state is a matrix per head, fixed per sequence, so a layer of these caches nothing that grows with context.",
		Formula: "params = d_model*(2*heads*head_dim + 2*value_heads*v_head_dim + 2*value_heads) + " +
			"value_heads*v_head_dim*d_model + conv + 2*value_heads + v_head_dim",
		Refs: []string{"https://arxiv.org/abs/2412.06464"},
	},
}

// --- transformer block ------------------------------------------------------

var transformerBlock = &BlockDef{
	Kind: "composite", Type: "transformer_block", Category: "block",
	// Grouped and conditioned, because twenty-five fields in one list is a list
	// nobody reads. About ten of them mean nothing at any given moment — a
	// dense block has no `expert_hidden`, grouped-query attention has no
	// `kv_lora`, an RMSNorm has no bias — and `when` is how the inspector knows
	// which ten.
	// Grouped and conditioned, because twenty-five fields in one list is a list
	// nobody reads. About ten of them mean nothing at any given moment — a
	// dense block has no `expert_hidden`, grouped-query attention has no
	// `kv_lora`, an RMSNorm has no bias — and `when` is how the inspector knows
	// which ten.
	//
	// The declared order is left alone. It is also the order a generated class
	// documents its parameters in, and reordering it to suit one panel would
	// rewrite sixty docstrings for a layout decision; `group` is what the
	// inspector reads.
	Params: ParamList{
		{"d_model", grouped(pInt(1, "Width of the residual stream"), "Shape")},
		{"heads", when(grouped(pInt(1, "Query heads"), "Attention"), "attention", "gqa", "mla")},
		{"kv_heads", when(grouped(pInt(1, "Key/value heads; fewer than the query heads is grouped-query attention"), "Attention"), "attention", "gqa")},
		{"head_dim", when(grouped(pInt(1, "Width of one head"), "Attention"), "attention", "gqa")},
		{"ffn_hidden", when(grouped(pInt(1, "Width in the middle of the feed-forward"), "Feed-forward"), "mlp", "gated", "dense")},
		{"attention", grouped(pEnum([]string{"gqa", "mla"}, "gqa", "Grouped-query attention, or DeepSeek's latent attention"), "Attention")},
		{"q_lora", when(grouped(pIntD(0, 0, "Latent attention: compressed query width"), "Attention"), "attention", "mla")},
		{"kv_lora", when(grouped(pIntD(0, 0, "Latent attention: cached latent width"), "Attention"), "attention", "mla")},
		{"nope_dim", when(grouped(pIntD(0, 0, "Per-head width without position"), "Attention"), "attention", "mla")},
		{"rope_dim", when(grouped(pIntD(0, 0, "Per-head rotary width"), "Attention"), "attention", "mla")},
		{"v_dim", when(grouped(pIntD(0, 0, "Per-head value width"), "Attention"), "attention", "mla")},
		{"norm", grouped(pEnum([]string{"rmsnorm", "layernorm"}, "rmsnorm", "Which normalization"), "Normalization")},
		{"norm_bias", when(grouped(pBool(true, "Bias on layernorm; ignored for rmsnorm"), "Normalization"), "norm", "layernorm")},
		{"post_norm", grouped(pBool(false, "Also normalize each sublayer's output before the residual add (Gemma 2/3)"), "Normalization")},
		{"mlp", grouped(pEnum([]string{"gated", "dense", "moe"}, "gated", "Gated, plain, or a mixture of experts"), "Feed-forward")},
		{"experts", when(grouped(pIntD(0, 0, "Routed experts, when mlp is moe"), "Feed-forward"), "mlp", "moe")},
		{"top_k", when(grouped(pIntD(1, 1, "Experts each token is routed to"), "Feed-forward"), "mlp", "moe")},
		{"expert_hidden", when(grouped(pIntD(0, 0, "Hidden width of one expert"), "Feed-forward"), "mlp", "moe")},
		{"shared_experts", when(grouped(pIntD(0, 0, "Always-on experts, beside the routed ones"), "Feed-forward"), "mlp", "moe")},
		{"shared_expert_gate", when(grouped(pBool(false,
			"Weight the shared expert's output by a sigmoid of one learned direction (Qwen)"),
			"Feed-forward"), "mlp", "moe")},
		{"router_bias", when(grouped(pBool(false, "A learned bias on the router's scores"), "Feed-forward"), "mlp", "moe")},
		{"act", grouped(pEnum([]string{"silu", "gelu", "gelu_tanh", "relu", "relu2"}, "silu", "Which nonlinearity in the feed-forward"), "Feed-forward")},
		{"attn_bias", grouped(pBool(false, "Bias on the q/k/v projections"), "Attention")},
		{"attn_o_bias", grouped(ParamSpec{Type: ParamBool, Default: nil, HasDefault: true,
			Doc: "Override the output-projection bias; null follows attn_bias"}, "Attention")},
		{"mlp_bias", grouped(pBool(false, "Bias on the feed-forward projections"), "Feed-forward")},
		{"qk_norm", when(grouped(pBool(false, "RMSNorm on the query and key heads"), "Attention"), "attention", "gqa")},
		{"causal", grouped(pBool(true, "Whether a token may attend to what follows it"), "Attention")},
		{"window", grouped(ParamSpec{Type: ParamInt, Default: 0.0, HasDefault: true,
			Doc: "Sliding-window width; 0 means full attention"}, "Attention")},
		{"rope", grouped(ropeSpec(), "Attention")},
		{"logit_softcap", grouped(ParamSpec{Type: ParamNum, Default: 0.0, HasDefault: true,
			Doc: "Bound the attention scores to this magnitude with tanh (Gemma 2)"}, "Attention")},
		{"value_embeddings", when(grouped(pBool(false,
			"Take a second embedding of the same tokens on `ve` and mix it into the values"), "Attention"),
			"attention", "gqa")},
		{"output_gate", when(grouped(pBool(false,
			"Sigmoid gate on the attention output, from a projection as wide as the queries (Qwen3-Next)"),
			"Attention"), "attention", "gqa")},
	},
	// The value-embedding port only exists when the block asks for it, which is
	// why these are computed rather than declared.
	PortsFn: func(r *Resolved) Ports {
		in := map[string]PortSpec{"x": Port("... d_model")}
		if r.Bool("value_embeddings") && r.Str("attention") != "mla" {
			in["ve"] = PortSpec{Shape: "... (kv_heads head_dim)", Anchor: "side",
				Doc: "A second embedding of the same tokens, mixed into the values"}
		}
		return Ports{In: in, Out: map[string]PortSpec{"y": Port("... d_model")}}
	},
	Docs: BlockDocs{
		Summary: "Pre-norm transformer block: norm, attention, residual, norm, feed-forward, residual.",
		Formula: "params = attention + mlp + 2 norms",
	},
}

func expandTransformerBlock(raw map[string]any, r *Resolved) Expansion {
	D := Ex(raw["d_model"], "0")
	normType := r.Str("norm")
	normParams := func() map[string]any {
		if normType == "layernorm" {
			return map[string]any{"dim": D, "bias": r.Bool("norm_bias")}
		}
		return map[string]any{"dim": D}
	}

	// The value-embedding stream passes straight through to the attention,
	// which is the only thing in here that reads the tokens twice.
	ve := r.Bool("value_embeddings") && r.Str("attention") != "mla"
	inNode, outNode := streamBoundary(D)
	if ve {
		inNode, outNode = boundary(
			map[string]any{
				"x": "... " + D,
				"ve": fmt.Sprintf("B T (%s %s)",
					Ex(raw["kv_heads"], "0"), Ex(raw["head_dim"], "0")),
			},
			map[string]any{"y": "... " + D},
		)
	}

	var mlpNode ir.NodeDef
	if r.Str("mlp") == "moe" {
		mlpNode = node("mlp", "moe_layer", map[string]any{
			"d_model":            D,
			"experts":            Ex(raw["experts"], "0"),
			"top_k":              Ex(raw["top_k"], "1"),
			"expert_hidden":      Ex(raw["expert_hidden"], "0"),
			"shared_experts":     Ex(raw["shared_experts"], "0"),
			"shared_expert_gate": r.Bool("shared_expert_gate"),
			"act":                r.Str("act"),
			"bias":               r.Bool("mlp_bias"),
			"router_bias":        r.Bool("router_bias"),
		})
	} else {
		typ := "gated_mlp"
		if r.Str("mlp") == "dense" {
			typ = "dense_mlp"
		}
		mlpNode = node("mlp", typ, map[string]any{
			"d_model": D, "hidden": Ex(raw["ffn_hidden"], "0"),
			"act": r.Str("act"), "bias": r.Bool("mlp_bias")})
	}

	var attnNode ir.NodeDef
	if r.Str("attention") == "mla" {
		attnNode = node("attn", "mla_attention", map[string]any{
			"d_model":  D,
			"heads":    Ex(raw["heads"], "0"),
			"q_lora":   Ex(raw["q_lora"], "0"),
			"kv_lora":  Ex(raw["kv_lora"], "0"),
			"nope_dim": Ex(raw["nope_dim"], "0"),
			"rope_dim": Ex(raw["rope_dim"], "0"),
			"v_dim":    Ex(raw["v_dim"], "0"),
			"causal":   r.Bool("causal"),
			"rope":     r.P["rope"],
			"bias":     r.Bool("attn_bias"),
		})
	} else {
		attnNode = node("attn", "gqa_attention", map[string]any{
			"d_model":          D,
			"heads":            Ex(raw["heads"], "0"),
			"kv_heads":         Ex(raw["kv_heads"], "0"),
			"head_dim":         Ex(raw["head_dim"], "0"),
			"bias":             r.Bool("attn_bias"),
			"o_bias":           r.P["attn_o_bias"],
			"causal":           r.Bool("causal"),
			"window":           Ex(raw["window"], "0"),
			"qk_norm":          r.Bool("qk_norm"),
			"rope":             r.P["rope"],
			"logit_softcap":    Ex(raw["logit_softcap"], "0"),
			"value_embeddings": ve,
			"output_gate":      r.Bool("output_gate"),
		})
	}

	nodes := []ir.NodeDef{
		inNode,
		node("norm1", normType, normParams()),
		attnNode,
		node("resid1", "add", map[string]any{"dim": D}),
		node("norm2", normType, normParams()),
		mlpNode,
		node("resid2", "add", map[string]any{"dim": D}),
		outNode,
	}
	edges := []ir.Edge{
		edge("_in:x", "norm1:x"), edge("norm1:y", "attn:x"), edge("_in:x", "resid1:b"),
		edge("resid1:y", "norm2:x"), edge("norm2:y", "mlp:x"),
		edge("resid1:y", "resid2:b"), edge("resid2:y", "_out:y"),
	}
	if ve {
		edges = append(edges, edge("_in:ve", "attn:ve"))
	}

	if r.Bool("post_norm") {
		nodes = append(nodes,
			node("post_attn_norm", normType, normParams()),
			node("post_mlp_norm", normType, normParams()))
		edges = append(edges,
			edge("attn:y", "post_attn_norm:x"), edge("post_attn_norm:y", "resid1:a"),
			edge("mlp:y", "post_mlp_norm:x"), edge("post_mlp_norm:y", "resid2:a"))
	} else {
		edges = append(edges, edge("attn:y", "resid1:a"), edge("mlp:y", "resid2:a"))
	}

	return Expansion{Nodes: nodes, Edges: edges}
}

var mtpHeadComposite = &BlockDef{
	Kind: "composite", Type: "mtp_head", Category: "head",
	Params: ParamList{
		{"d_model", pInt(1, "Residual stream width")},
		{"by", ParamSpec{Type: ParamInt, Default: 1.0, HasDefault: true,
			Doc: "How far ahead this module predicts: 1 is the token after next"}},
		{"norm", pEnum([]string{"rmsnorm", "layernorm"}, "rmsnorm", "Which normalization before the join")},
		{"norm_bias", pBool(true, "Bias on layernorm; ignored for rmsnorm")},
		{"bias", pBool(false, "Bias on the projection")},
	},
	Ports: Ports{
		In: map[string]PortSpec{
			// The hidden state from the stack below, and the token embeddings
			// this module shifts for itself.
			"x": Port("... d_model"),
			"e": {Shape: "... d_model", Anchor: "side"},
		},
		Out: map[string]PortSpec{"y": Port("... d_model")},
	},
	Docs: BlockDocs{
		Summary: "One multi-token prediction module's projection: normalizes the hidden state " +
			"and the embedding of the token `by` ahead, joins them, and projects 2*d_model back " +
			"down to d_model. A whole module is this, then a transformer block, then the " +
			"model's own output head, which is shared and so costs FLOPs and no parameters; " +
			"wire a tied lm_head after the block to count them. Depth is how many modules are " +
			"stacked, each reading the one below.",
		Formula: "h' = W [ norm(h) ; norm(Emb(t+by)) ], W in R^(d x 2d)",
		Refs:    []string{"https://arxiv.org/abs/2412.19437"},
	},
}

// expandGatedDeltanet lays out the block around the recurrence.
//
// One projection produces q, k, v and the output gate together, which is what
// the reference does and what makes the parameter count a single matrix rather
// than four. The two per-head scalars come from their own much smaller
// projection, because they are two numbers per head against thousands for
// everything else.
func expandGatedDeltanet(raw map[string]any, r *Resolved) Expansion {
	D := Ex(raw["d_model"], "0")
	H := Ex(raw["heads"], "0")
	dk := Ex(raw["head_dim"], "0")
	dv := dk
	if r.Num("v_head_dim") != 0 {
		dv = Ex(raw["v_head_dim"], "0")
	}
	// The value heads need not be the key heads. Qwen3-Next has twice as many,
	// which is most of why its input projection is as wide as it is.
	HV := H
	if r.Num("value_heads") != 0 {
		HV = Ex(raw["value_heads"], "0")
	}
	bias := r.Bool("bias")

	qWidth := fmt.Sprintf("(%s)*(%s)", H, dk)
	vWidth := fmt.Sprintf("(%s)*(%s)", HV, dv)
	// q, k and v go through the convolution; the gate does not.
	convWidth := fmt.Sprintf("2*%s+%s", qWidth, vWidth)
	inWidth := fmt.Sprintf("(%s)+(%s)", convWidth, vWidth)
	gates := fmt.Sprintf("2*(%s)", HV)

	inNode, outNode := streamBoundary(D)
	return Expansion{
		Nodes: []ir.NodeDef{
			inNode,
			node("in_proj", "linear", map[string]any{
				"in_features": D, "out_features": inWidth, "bias": bias}),
			node("split_gate", "split", map[string]any{
				"from": fmt.Sprintf("B T (%s)", inWidth), "sizes": []any{convWidth, vWidth}}),
			node("conv", "conv1d", map[string]any{
				"channels": convWidth, "kernel": Ex(raw["conv_kernel"], "4"),
				"bias": r.Bool("conv_bias")}),
			node("conv_act", "activation", map[string]any{"kind": "silu", "dim": convWidth}),
			node("split_qkv", "split", map[string]any{
				"from":  fmt.Sprintf("B T (%s)", convWidth),
				"sizes": []any{qWidth, qWidth, vWidth}}),
			node("q_heads", "rearrange", map[string]any{
				"from": fmt.Sprintf("B T (%s %s)", H, dk), "to": fmt.Sprintf("B %s T %s", H, dk)}),
			node("k_heads", "rearrange", map[string]any{
				"from": fmt.Sprintf("B T (%s %s)", H, dk), "to": fmt.Sprintf("B %s T %s", H, dk)}),
			node("v_heads", "rearrange", map[string]any{
				"from": fmt.Sprintf("B T (%s %s)", HV, dv), "to": fmt.Sprintf("B %s T %s", HV, dv)}),
			// Two scalars per head: how much the state decays, and how hard this
			// token writes to it.
			node("ba_proj", "linear", map[string]any{
				"in_features": D, "out_features": gates, "bias": false}),
			node("scan", "gated_delta_scan", map[string]any{
				"heads": H, "head_dim": dk, "v_head_dim": dv, "value_heads": HV}),
			// The norm and the gate are per head, so they sit inside the head
			// layout and the merge comes after them. Qwen3-Next's
			// `linear_attn.norm.weight` is [128] for a 128-wide value head, not
			// [4096] for thirty-two of them; the formula here always said
			// `v_head_dim`, and it was the expansion that disagreed.
			node("norm", "rmsnorm", map[string]any{"dim": dv}),
			node("gate_heads", "rearrange", map[string]any{
				"from": fmt.Sprintf("B T (%s %s)", HV, dv), "to": fmt.Sprintf("B %s T %s", HV, dv)}),
			node("gate_act", "activation", map[string]any{"kind": "silu", "dim": dv}),
			node("gate", "mul", map[string]any{"dim": dv}),
			node("merge", "rearrange", map[string]any{
				"from": fmt.Sprintf("B %s T %s", HV, dv), "to": fmt.Sprintf("B T (%s %s)", HV, dv)}),
			node("out_proj", "linear", map[string]any{
				"in_features": vWidth, "out_features": D, "bias": bias}),
			outNode,
		},
		Edges: []ir.Edge{
			edge("_in:x", "in_proj:x"), edge("_in:x", "ba_proj:x"),
			edge("in_proj:y", "split_gate:x"),
			edge("split_gate:y0", "conv:x"), edge("conv:y", "conv_act:x"),
			edge("conv_act:y", "split_qkv:x"),
			edge("split_qkv:y0", "q_heads:x"),
			edge("split_qkv:y1", "k_heads:x"),
			edge("split_qkv:y2", "v_heads:x"),
			edge("q_heads:y", "scan:q"), edge("k_heads:y", "scan:k"),
			edge("v_heads:y", "scan:v"), edge("ba_proj:y", "scan:gates"),
			edge("scan:y", "norm:x"),
			edge("split_gate:y1", "gate_heads:x"), edge("gate_heads:y", "gate_act:x"),
			edge("norm:y", "gate:a"), edge("gate_act:y", "gate:b"),
			edge("gate:y", "merge:x"), edge("merge:y", "out_proj:x"),
			edge("out_proj:y", "_out:y"),
		},
	}
}

// --- containers -------------------------------------------------------------

var repeatContainer = &BlockDef{
	Kind: "container", Type: "repeat", Category: "container",
	Params: ParamList{
		{"count", pInt(1, "How many times the subgraph is stacked")},
	},
	Docs: BlockDocs{
		Summary: "Stacks its subgraph count times. The subgraph's input and output shapes must match.",
		// There was a `pattern` parameter here, for hybrid stacks written as one
		// character per layer. Nothing ever read it: the inspector offered the
		// field, the analysis ignored what was typed into it, and the answer was
		// wrong without saying so. The two things it was meant for both work
		// without it. A stack that alternates on a fixed period is a repeat of
		// the *group* — Gemma's local and global layers are one repeat of L/2
		// holding both blocks in series. A stack with no period at all is
		// written out, as Nemotron-H's fifty-two layers are.
	},
}

// Expand builds the subgraph a composite stands for.
//
// A switch rather than a field on BlockDef: the expansions close over the raw
// parameter map as well as the resolved one, because an expression such as "D"
// has to survive into the inner graph so inner shapes keep showing symbol
// names rather than the numbers they evaluate to.
func Expand(def *BlockDef, raw map[string]any, r *Resolved) (Expansion, bool) {
	if raw == nil {
		raw = map[string]any{}
	}
	// A parameter the document left out still has to reach the expansion, or a
	// default-valued width would come through as an empty string.
	full := map[string]any{}
	for k, v := range r.RawFull {
		full[k] = v
	}
	for k, v := range raw {
		full[k] = v
	}

	switch def.Type {
	case "gqa_attention":
		return expandGQA(full, r), true
	case "gated_mlp":
		return expandGatedMlp(full, r), true
	case "dense_mlp":
		return expandDenseMlp(full, r), true
	case "mla_attention":
		return expandMLA(full, r), true
	case "moe_layer":
		return expandMoeLayer(full, r), true
	case "mamba2_block":
		return expandMamba2(full, r), true
	case "transformer_block":
		return expandTransformerBlock(full, r), true
	case "mtp_head":
		return expandMtpHead(full, r), true
	case "gated_deltanet_block":
		return expandGatedDeltanet(full, r), true
	}
	// A block the document defined for itself expands from its template.
	return ExpandUser(def, full)
}

// expandMtpHead is the four steps of an MTP module's projection.
func expandMtpHead(raw map[string]any, r *Resolved) Expansion {
	D := Ex(raw["d_model"], "0")
	normType := r.Str("norm")
	normParams := func() map[string]any {
		if normType == "layernorm" {
			return map[string]any{"dim": D, "bias": r.Bool("norm_bias")}
		}
		return map[string]any{"dim": D}
	}

	inNode, outNode := boundary(
		map[string]any{"x": "... " + D, "e": "... " + D},
		map[string]any{"y": "... " + D},
	)

	nodes := []ir.NodeDef{
		inNode,
		node("norm_h", normType, normParams()),
		// The module reads the embedding of the token it is predicting, which
		// over a whole training sequence is the embedding stream moved along.
		node("ahead", "shift", map[string]any{"dim": D, "by": Ex(raw["by"], "1")}),
		node("norm_e", normType, normParams()),
		node("join", "concat", map[string]any{
			"to":    "... (2*" + D + ")",
			"sizes": []any{D, D},
			"axis":  -1.0,
		}),
		node("proj", "linear", map[string]any{
			"in_features": "2*" + D, "out_features": D, "bias": r.Bool("bias"),
		}),
		outNode,
	}
	edges := []ir.Edge{
		edge("_in:x", "norm_h:x"),
		edge("_in:e", "ahead:x"),
		edge("ahead:y", "norm_e:x"),
		edge("norm_h:y", "join:y0"),
		edge("norm_e:y", "join:y1"),
		edge("join:y", "proj:x"),
		edge("proj:y", "_out:y"),
	}
	return Expansion{Nodes: nodes, Edges: edges}
}

func isFalse(v any) bool {
	b, ok := v.(bool)
	return ok && !b
}
