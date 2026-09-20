// Package design builds documents.
//
// Most designs are read rather than built: a preset is a JSON file and an edit
// is an operation on one. This is for the cases where a document has to be
// constructed from a description — importing someone else's config, or starting
// a new design from a family rather than from an empty canvas.
package design

import (
	"fmt"

	"github.com/tensorcad/core/ir"
)

// Rope is a rotary position embedding's setting.
type Rope struct {
	Theta float64
}

// Mamba is a state-space layer's shape, for a hybrid stack.
type Mamba struct {
	Expand     *float64
	HeadDim    *float64
	State      *float64
	Groups     *float64
	ConvKernel *float64
}

// Hybrid is an irregular stack, written as one character per layer: M for a
// state-space layer, A for attention, F for feed-forward. It is for models
// whose layer pattern does not repeat on a fixed period.
type Hybrid struct {
	Pattern string
	Mamba   Mamba
}

// Alternating is a stack whose attention layers are not all the same: most
// attend within a bounded window and one in every Period attends to everything.
//
// Gemma 2 alternates every other layer; Gemma 3 makes every sixth one global.
// The repeating unit is the group, not the layer, which is why this becomes a
// repeat of L/Period with Period blocks inside it rather than anything new.
// When Period does not divide L the leftover layers are all local, and they
// follow as a second stack.
type Alternating struct {
	// Window is how far back a local layer may attend.
	Window float64
	// Period is layers per group. The last one in each group is the global one,
	// which is the order every model doing this uses: local first.
	Period float64
}

// MLA is latent attention. When present, KVHeads and HeadDim are unused.
type MLA struct {
	QLora   float64
	KVLora  float64
	NopeDim float64
	RopeDim float64
	VDim    float64
}

// MoE is a sparse feed-forward. Omit it for a dense model.
type MoE struct {
	Experts       float64
	TopK          float64
	ExpertHidden  float64
	SharedExperts float64
	RouterBias    bool
	// DenseLayers are leading layers that keep a dense feed-forward, as
	// DeepSeek and Qwen do.
	DenseLayers float64
}

// DecoderSpec describes a decoder-only transformer.
type DecoderSpec struct {
	Name   string
	Family string
	Notes  string

	Layers  float64
	DModel  float64
	Heads   float64
	KVHeads *float64
	// HeadDim is a number or an expression; nil is DModel/Heads.
	HeadDim any
	// FFNHidden is the feed-forward width, a number or an expression over the
	// other symbols.
	FFNHidden any
	Vocab     float64

	// Norm is "rmsnorm" or "layernorm"; empty means rmsnorm.
	Norm     string
	NormBias *bool
	PostNorm *bool
	// MLP is "gated" or "dense"; empty means gated.
	MLP string
	Act string

	// Rope is the rotary setting. Left nil with RopeGiven false it is the usual
	// theta of 10000; nil with RopeGiven true means no rotary embedding, which
	// is what a model with learned absolute positions wants.
	Rope      *Rope
	RopeGiven bool
	// MaxSeq is the number of learned absolute positions, when there are any.
	MaxSeq *float64

	Tied     *bool
	AttnBias *bool
	// AttnOBias is a tri-state: nil leaves the output projection's bias to
	// follow AttnBias, which is what most families want.
	AttnOBias *bool
	MLPBias   *bool
	QKNorm    *bool
	Window    *float64
	// Alternating, when set, overrides Window: the stack is local and global
	// layers in a fixed cycle rather than one kind throughout.
	Alternating *Alternating

	Hybrid *Hybrid
	MLA    *MLA
	MoE    *MoE

	DefaultSeq *float64
	Published  *ir.Published
}

func orNum(v *float64, fallback float64) float64 {
	if v == nil {
		return fallback
	}
	return *v
}

func orBool(v *bool, fallback bool) bool {
	if v == nil {
		return fallback
	}
	return *v
}

func orStr(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}

// symbolWriter builds the symbol table in the order it is written, because the
// order is what the symbol panel lists and what a person reads back.
type symbolWriter struct {
	doc *ir.Doc
}

func (w symbolWriter) design(name string, value any, doc string) {
	def := ir.SymbolDef{Kind: "design", Doc: doc}
	switch v := value.(type) {
	case float64:
		def.Number, def.HasNumber = v, true
	case int:
		def.Number, def.HasNumber = float64(v), true
	case string:
		def.Expr = v
	default:
		def.Expr = fmt.Sprint(v)
	}
	w.doc.Symbols[name] = def
	w.doc.SymbolOrder = append(w.doc.SymbolOrder, name)
}

func (w symbolWriter) runtime(name string, def float64, doc string) {
	w.doc.Symbols[name] = ir.SymbolDef{Kind: "runtime", Number: def, HasNumber: true, Doc: doc}
	w.doc.SymbolOrder = append(w.doc.SymbolOrder, name)
}

// DecoderOnly builds a decoder-only transformer from a description of it.
func DecoderOnly(spec DecoderSpec) *ir.Doc {
	kvHeads := orNum(spec.KVHeads, spec.Heads)
	var headDim any = spec.HeadDim
	if headDim == nil {
		headDim = spec.DModel / spec.Heads
	}
	norm := orStr(spec.Norm, "rmsnorm")
	mlp := orStr(spec.MLP, "gated")
	act := spec.Act
	if act == "" {
		act = "gelu"
		if mlp == "gated" {
			act = "silu"
		}
	}
	rope := spec.Rope
	if !spec.RopeGiven && rope == nil {
		rope = &Rope{Theta: 10000}
	}
	tied := orBool(spec.Tied, false)

	doc := &ir.Doc{
		Version: ir.DocVersion,
		Meta: ir.DocMeta{
			Name: spec.Name, Family: spec.Family, Notes: spec.Notes, Published: spec.Published,
		},
		Symbols: map[string]ir.SymbolDef{},
		Ui:      &ir.UiState{},
	}
	s := symbolWriter{doc}
	s.runtime("B", 1, "Batch size")
	s.runtime("T", orNum(spec.DefaultSeq, 4096), "Sequence length in tokens")
	s.design("L", spec.Layers, "Number of transformer layers")
	s.design("D", spec.DModel, "Residual stream width (d_model)")
	s.design("H", spec.Heads, "Query heads")
	s.design("Hkv", kvHeads, "Key/value heads")
	s.design("dh", headDim, "Head dimension")
	s.design("F", spec.FFNHidden, "Feed-forward hidden width")
	s.design("V", spec.Vocab, "Vocabulary size")
	if spec.MaxSeq != nil && *spec.MaxSeq != 0 {
		s.design("Tmax", *spec.MaxSeq, "Maximum position index")
	}
	if spec.Window != nil && *spec.Window != 0 {
		s.design("W", *spec.Window, "Sliding-window width")
	}
	if spec.Alternating != nil {
		s.design("W", spec.Alternating.Window, "Sliding-window width on the local attention layers")
	}
	if spec.MLA != nil {
		s.design("Ql", spec.MLA.QLora, "Compressed query width")
		s.design("Kl", spec.MLA.KVLora, "Cached latent width")
		s.design("dnope", spec.MLA.NopeDim, "Per-head width without position")
		s.design("drope", spec.MLA.RopeDim, "Per-head rotary width")
		s.design("dv", spec.MLA.VDim, "Per-head value width")
	}
	if spec.MoE != nil {
		s.design("E", spec.MoE.Experts, "Routed experts per layer")
		s.design("K", spec.MoE.TopK, "Experts each token is routed to")
		s.design("Fe", spec.MoE.ExpertHidden, "Hidden width of one expert")
		if spec.MoE.SharedExperts != 0 {
			s.design("Ns", spec.MoE.SharedExperts, "Always-on shared experts")
		}
		if spec.MoE.DenseLayers != 0 {
			s.design("Ld", spec.MoE.DenseLayers, "Leading dense layers")
			s.design("Lm", "L - Ld", "Sparse layers")
		}
	}

	var ropeParam any
	if rope != nil {
		ropeParam = map[string]any{"theta": rope.Theta}
	}
	window := any(0.0)
	if spec.Window != nil && *spec.Window != 0 {
		window = "W"
	}

	blockParams := map[string]any{
		"d_model":     "D",
		"heads":       "H",
		"kv_heads":    "Hkv",
		"head_dim":    "dh",
		"ffn_hidden":  "F",
		"norm":        norm,
		"norm_bias":   orBool(spec.NormBias, true),
		"post_norm":   orBool(spec.PostNorm, false),
		"mlp":         mlp,
		"act":         act,
		"attn_bias":   orBool(spec.AttnBias, false),
		"mlp_bias":    orBool(spec.MLPBias, false),
		"qk_norm":     orBool(spec.QKNorm, false),
		"causal":      true,
		"window":      window,
		"rope":        ropeParam,
		"attn_o_bias": nil,
	}
	if spec.AttnOBias != nil {
		blockParams["attn_o_bias"] = *spec.AttnOBias
	}
	if spec.MLA != nil {
		blockParams["attention"] = "mla"
		blockParams["q_lora"] = "Ql"
		blockParams["kv_lora"] = "Kl"
		blockParams["nope_dim"] = "dnope"
		blockParams["rope_dim"] = "drope"
		blockParams["v_dim"] = "dv"
	}

	moeParams := blockParams
	if spec.MoE != nil {
		moeParams = map[string]any{}
		for k, v := range blockParams {
			moeParams[k] = v
		}
		moeParams["mlp"] = "moe"
		moeParams["experts"] = "E"
		moeParams["top_k"] = "K"
		moeParams["expert_hidden"] = "Fe"
		moeParams["router_bias"] = spec.MoE.RouterBias
		moeParams["shared_experts"] = any(0.0)
		if spec.MoE.SharedExperts != 0 {
			moeParams["shared_experts"] = "Ns"
		}
	}

	var stacks []ir.NodeDef
	switch {
	case spec.Hybrid != nil:
		stacks = append(stacks, hybridStack(spec, act, rope))
	case spec.MoE != nil && spec.MoE.DenseLayers != 0:
		stacks = append(stacks,
			stack("dense_layers", "Ld", blockParams,
				fmt.Sprintf("Dense block x%s", num(spec.MoE.DenseLayers))),
			stack("layers", "Lm", moeParams,
				fmt.Sprintf("Sparse block x%s", num(spec.Layers-spec.MoE.DenseLayers))))
	case spec.Alternating != nil:
		stacks = append(stacks, alternatingStacks(spec, moeParams)...)
	default:
		stacks = append(stacks,
			stack("layers", "L", moeParams, fmt.Sprintf("Transformer block x%s", num(spec.Layers))))
	}

	nodes := []ir.NodeDef{
		{ID: "tokens", Type: "input", Params: map[string]any{"shape": "B T", "dtype": "int64"}},
		{ID: "embed", Type: "embedding", Params: map[string]any{"vocab": "V", "dim": "D"}},
	}
	edges := []ir.Edge{{"tokens:x", "embed:ids"}}
	tail := "embed:y"

	if spec.MaxSeq != nil && *spec.MaxSeq != 0 {
		nodes = append(nodes, ir.NodeDef{ID: "pos", Type: "pos_embedding",
			Params: map[string]any{"max_seq": "Tmax", "dim": "D"}})
		edges = append(edges, ir.Edge{tail, "pos:x"})
		tail = "pos:y"
	}

	for _, st := range stacks {
		nodes = append(nodes, st)
		edges = append(edges, ir.Edge{tail, st.ID + ":x"})
		// A stack's own port, not its output: a repeat container's boundary
		// carries one name in and the same name out.
		tail = st.ID + ":x"
	}

	finalParams := map[string]any{"dim": "D"}
	if norm == "layernorm" {
		finalParams["bias"] = orBool(spec.NormBias, true)
	}
	nodes = append(nodes,
		ir.NodeDef{ID: "final_norm", Type: norm, Params: finalParams},
		ir.NodeDef{ID: "head", Type: "lm_head",
			Params: map[string]any{"vocab": "V", "dim": "D", "tied": tied}},
		ir.NodeDef{ID: "logits", Type: "output"})
	edges = append(edges,
		ir.Edge{tail, "final_norm:x"},
		ir.Edge{"final_norm:y", "head:x"},
		ir.Edge{"head:y", "logits:x"})

	doc.Graph = ir.Graph{Nodes: nodes, Edges: edges}
	return doc
}

// alternatingStacks writes a local/global cycle as one repeat of the group.
//
// A group is Period-1 windowed layers and then one full-attention layer, in
// series, which is exactly what the cycle is; repeating that L/Period times
// gives back the stack. Everything downstream — the parameter count, the cache,
// the attention FLOPs — falls out of the blocks themselves, because a windowed
// layer already reports a cache bounded by its window rather than one that
// grows.
func alternatingStacks(spec DecoderSpec, params map[string]any) []ir.NodeDef {
	period := spec.Alternating.Period
	if period < 2 {
		period = 2
	}
	locals := int(period) - 1
	groups := int(spec.Layers / period)
	leftover := int(spec.Layers) - groups*int(period)

	local := func(params map[string]any) map[string]any { return withWindow(params, "W") }
	global := func(params map[string]any) map[string]any { return withWindow(params, 0.0) }

	var nodes []ir.NodeDef
	var edges []ir.Edge
	nodes = append(nodes, ir.NodeDef{ID: "_in", Type: "boundary_in",
		Params: map[string]any{"ports": map[string]any{"x": "B T D"}}})
	tail := "_in:x"
	for i := 0; i < locals; i++ {
		id := "local"
		if locals > 1 {
			id = fmt.Sprintf("local%d", i+1)
		}
		nodes = append(nodes, ir.NodeDef{ID: id, Type: "transformer_block", Params: local(params)})
		edges = append(edges, ir.Edge{tail, id + ":x"})
		tail = id + ":y"
	}
	nodes = append(nodes, ir.NodeDef{ID: "global", Type: "transformer_block", Params: global(params)})
	edges = append(edges, ir.Edge{tail, "global:x"})
	nodes = append(nodes, ir.NodeDef{ID: "_out", Type: "boundary_out",
		Params: map[string]any{"ports": map[string]any{"x": "B T D"}}})
	edges = append(edges, ir.Edge{"global:y", "_out:x"})

	out := []ir.NodeDef{{
		ID: "layers", Type: "repeat",
		Label:  fmt.Sprintf("%d local + 1 global x%d", locals, groups),
		Params: map[string]any{"count": fmt.Sprintf("L/%s", num(period))},
		Graph:  &ir.Graph{Nodes: nodes, Edges: edges},
	}}
	if leftover > 0 {
		// The cycle does not divide the depth, so the tail of the stack is
		// local layers with no global one after them.
		out = append(out, stack("tail_layers", num(float64(leftover)), local(params),
			fmt.Sprintf("Local block x%d", leftover)))
	}
	return out
}

// withWindow is params with one value changed, leaving the original alone.
func withWindow(params map[string]any, window any) map[string]any {
	out := make(map[string]any, len(params))
	for k, v := range params {
		out[k] = v
	}
	out["window"] = window
	return out
}

func stack(id, count string, params map[string]any, label string) ir.NodeDef {
	return ir.NodeDef{
		ID: id, Type: "repeat", Label: label,
		Params: map[string]any{"count": count},
		Graph: &ir.Graph{
			Nodes: []ir.NodeDef{
				{ID: "_in", Type: "boundary_in", Params: map[string]any{"ports": map[string]any{"x": "B T D"}}},
				{ID: "block", Type: "transformer_block", Params: params},
				{ID: "_out", Type: "boundary_out", Params: map[string]any{"ports": map[string]any{"x": "B T D"}}},
			},
			Edges: []ir.Edge{{"_in:x", "block:x"}, {"block:y", "_out:x"}},
		},
	}
}

// hybridStack lays out one layer per character of the pattern, each with its
// own norm and residual add, because the pattern does not repeat.
func hybridStack(spec DecoderSpec, act string, rope *Rope) ir.NodeDef {
	m := spec.Hybrid.Mamba
	nodes := []ir.NodeDef{
		{ID: "_in", Type: "boundary_in", Params: map[string]any{"ports": map[string]any{"x": "B T D"}}},
	}
	var edges []ir.Edge
	tail := "_in:x"

	var ropeParam any
	if rope != nil {
		ropeParam = map[string]any{"theta": rope.Theta}
	}

	for i, kind := range spec.Hybrid.Pattern {
		normID := fmt.Sprintf("norm%d", i)
		blockID := fmt.Sprintf("blk%d", i)
		addID := fmt.Sprintf("add%d", i)
		nodes = append(nodes, ir.NodeDef{ID: normID, Type: "rmsnorm", Params: map[string]any{"dim": "D"}})

		switch kind {
		case 'M':
			nodes = append(nodes, ir.NodeDef{
				ID: blockID, Type: "mamba2_block", Label: fmt.Sprintf("Mamba-2 %d", i),
				Params: map[string]any{
					"d_model":     "D",
					"expand":      orNum(m.Expand, 2),
					"head_dim":    orNum(m.HeadDim, 64),
					"state":       orNum(m.State, 128),
					"groups":      orNum(m.Groups, 1),
					"conv_kernel": orNum(m.ConvKernel, 4),
				},
			})
		case 'A':
			nodes = append(nodes, ir.NodeDef{
				ID: blockID, Type: "gqa_attention", Label: fmt.Sprintf("Attention %d", i),
				Params: map[string]any{
					"d_model": "D", "heads": "H", "kv_heads": "Hkv", "head_dim": "dh",
					"causal": true, "rope": ropeParam, "bias": orBool(spec.AttnBias, false),
				},
			})
		default:
			nodes = append(nodes, ir.NodeDef{
				ID: blockID, Type: "dense_mlp", Label: fmt.Sprintf("Feed-forward %d", i),
				Params: map[string]any{
					"d_model": "D", "hidden": "F", "act": act, "bias": orBool(spec.MLPBias, false),
				},
			})
		}

		nodes = append(nodes, ir.NodeDef{ID: addID, Type: "add", Params: map[string]any{"dim": "D"}})
		edges = append(edges,
			ir.Edge{tail, normID + ":x"},
			ir.Edge{normID + ":y", blockID + ":x"},
			ir.Edge{blockID + ":y", addID + ":a"},
			ir.Edge{tail, addID + ":b"})
		tail = addID + ":y"
	}

	nodes = append(nodes, ir.NodeDef{ID: "_out", Type: "boundary_out",
		Params: map[string]any{"ports": map[string]any{"x": "B T D"}}})
	edges = append(edges, ir.Edge{tail, "_out:x"})

	return ir.NodeDef{
		ID: "layers", Type: "repeat",
		Label:  fmt.Sprintf("Hybrid stack, %d layers", len([]rune(spec.Hybrid.Pattern))),
		Params: map[string]any{"count": 1},
		Graph:  &ir.Graph{Nodes: nodes, Edges: edges},
	}
}

// num writes a count into a label the way the other engine does.
func num(v float64) string {
	if v == float64(int64(v)) {
		return fmt.Sprintf("%d", int64(v))
	}
	return fmt.Sprint(v)
}
