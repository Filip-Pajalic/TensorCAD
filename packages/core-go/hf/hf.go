// Package hf imports a Hugging Face config.json.
//
// config.json is the de facto interchange format for open model architectures,
// so reading it is the fastest way to get a real design onto the canvas. Each
// family is mapped explicitly rather than guessed: an unknown model_type is an
// error, not a silent approximation.
package hf

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/tensorcad/core/design"
	"github.com/tensorcad/core/ir"
)

// Config is a parsed config.json, kept loose because every family adds fields.
type Config map[string]any

// Result is the imported design and what the import could not represent.
type Result struct {
	Doc      *ir.Doc  `json:"doc"`
	Warnings []string `json:"warnings"`
}

// SupportedModelTypes are the families whose layout this importer knows.
var SupportedModelTypes = []string{
	"gpt2",
	"llama",
	"mistral",
	"mixtral",
	"qwen2",
	"qwen3",
	"qwen3_moe",
	"gemma",
	"gemma2",
	"deepseek_v3",
}

// num reads the first of several spellings of a numeric field.
func num(c Config, keys ...string) (float64, bool) {
	for _, k := range keys {
		switch v := c[k].(type) {
		case float64:
			if !math.IsNaN(v) && !math.IsInf(v, 0) {
				return v, true
			}
		case int:
			return float64(v), true
		}
	}
	return 0, false
}

func flag(c Config, keys ...string) (bool, bool) {
	for _, k := range keys {
		if v, ok := c[k].(bool); ok {
			return v, true
		}
	}
	return false, false
}

func required(c Config, field string, keys ...string) (float64, error) {
	if len(keys) == 0 {
		keys = []string{field}
	}
	if v, ok := num(c, keys...); ok {
		return v, nil
	}
	return 0, fmt.Errorf("config.json is missing %q", field)
}

func f(v float64) *float64 { return &v }
func b(v bool) *bool       { return &v }

// Import reads a config.json into a design. An empty name takes the model's
// own _name_or_path.
func Import(config Config, name string) (*Result, error) {
	var warnings []string

	typ, _ := config["model_type"].(string)
	typ = strings.ToLower(typ)
	if typ == "" {
		return nil, fmt.Errorf("config.json has no %q", "model_type")
	}
	known := false
	for _, t := range SupportedModelTypes {
		if t == typ {
			known = true
			break
		}
	}
	if !known {
		return nil, fmt.Errorf("unsupported model_type %q. This importer knows: %s.",
			typ, strings.Join(SupportedModelTypes, ", "))
	}

	modelName := name
	if modelName == "" {
		if s, ok := config["_name_or_path"].(string); ok {
			modelName = s
		} else {
			modelName = typ
		}
	}

	// --- GPT-2 uses its own field names -------------------------------------
	if typ == "gpt2" {
		layers, err := required(config, "n_layer", "n_layer", "num_hidden_layers")
		if err != nil {
			return nil, err
		}
		dModel, err := required(config, "n_embd", "n_embd", "hidden_size")
		if err != nil {
			return nil, err
		}
		heads, err := required(config, "n_head", "n_head", "num_attention_heads")
		if err != nil {
			return nil, err
		}
		vocab, err := required(config, "vocab_size")
		if err != nil {
			return nil, err
		}

		var ffn any = "4*D"
		if v, ok := num(config, "n_inner"); ok {
			ffn = v
		}
		maxSeq := 1024.0
		if v, ok := num(config, "n_positions", "n_ctx"); ok {
			maxSeq = v
		}
		defaultSeq := 1024.0
		if v, ok := num(config, "n_positions"); ok {
			defaultSeq = v
		}
		tied := true
		if v, ok := flag(config, "tie_word_embeddings"); ok {
			tied = v
		}

		spec := design.DecoderSpec{
			Name: modelName, Family: "gpt2",
			Layers: layers, DModel: dModel, Heads: heads,
			FFNHidden: ffn, Vocab: vocab,
			MaxSeq: f(maxSeq),
			Norm:   "layernorm", MLP: "dense", Act: "gelu",
			// GPT-2 learns its positions, so there is no rotary embedding.
			RopeGiven: true,
			Tied:      b(tied), AttnBias: b(true), MLPBias: b(true),
			DefaultSeq: f(defaultSeq),
		}
		return &Result{Doc: design.DecoderOnly(spec), Warnings: warnings}, nil
	}

	// --- Everything else follows the Llama field names ----------------------
	layers, err := required(config, "num_hidden_layers")
	if err != nil {
		return nil, err
	}
	dModel, err := required(config, "hidden_size")
	if err != nil {
		return nil, err
	}
	heads, err := required(config, "num_attention_heads")
	if err != nil {
		return nil, err
	}
	vocab, err := required(config, "vocab_size")
	if err != nil {
		return nil, err
	}
	ffn, err := required(config, "intermediate_size")
	if err != nil {
		return nil, err
	}
	kvHeads := heads
	if v, ok := num(config, "num_key_value_heads"); ok {
		kvHeads = v
	}
	headDim := dModel / heads
	if v, ok := num(config, "head_dim"); ok {
		headDim = v
	}
	theta := 10000.0
	if v, ok := num(config, "rope_theta"); ok {
		theta = v
	}
	maxPos := 4096.0
	if v, ok := num(config, "max_position_embeddings"); ok {
		maxPos = v
	}

	spec := design.DecoderSpec{
		Name: modelName, Family: typ,
		Layers: layers, DModel: dModel, Heads: heads,
		KVHeads: f(kvHeads), HeadDim: headDim,
		FFNHidden: ffn, Vocab: vocab,
		Rope:       &design.Rope{Theta: theta},
		RopeGiven:  true,
		Tied:       b(false),
		AttnBias:   b(false),
		MLPBias:    b(false),
		DefaultSeq: f(math.Min(maxPos, 32768)),
	}
	if v, ok := flag(config, "tie_word_embeddings"); ok {
		spec.Tied = b(v)
	}
	if v, ok := flag(config, "attention_bias"); ok {
		spec.AttnBias = b(v)
	}
	if v, ok := flag(config, "mlp_bias"); ok {
		spec.MLPBias = b(v)
	}

	// Qwen 2.5 puts a bias on q/k/v but not on the output projection.
	_, hasWindowFlag := config["use_sliding_window"]
	_, hasDropout := config["attention_dropout"]
	if typ == "qwen2" && (hasWindowFlag || hasDropout) {
		spec.AttnBias = b(true)
		spec.AttnOBias = b(false)
	}
	if _, given := flag(config, "attention_bias"); typ == "qwen2" && !given {
		spec.AttnBias = b(true)
		spec.AttnOBias = b(false)
	}

	// Qwen 3 normalizes each attention head's queries and keys.
	if typ == "qwen3" || typ == "qwen3_moe" {
		spec.QKNorm = b(true)
	}

	// Gemma normalizes both the input and the output of every sublayer.
	if typ == "gemma2" || typ == "gemma" {
		spec.PostNorm = b(typ == "gemma2")
		spec.Act = "gelu_tanh"
		spec.Tied = b(true)
		if v, ok := flag(config, "tie_word_embeddings"); ok {
			spec.Tied = b(v)
		}
		if w, ok := num(config, "sliding_window"); ok && w != 0 && typ == "gemma2" {
			warnings = append(warnings,
				"Gemma 2 alternates sliding-window and full-attention layers. This import makes every layer full-attention, "+
					"which is right for the parameter count but understates how much the cache is reduced.")
		}
	}

	if window, ok := num(config, "sliding_window"); ok && window != 0 && typ == "mistral" {
		if use, given := flag(config, "use_sliding_window"); !given || use {
			spec.Window = f(window)
		}
	}

	// --- Sparse feed-forward -------------------------------------------------
	experts, hasExperts := num(config, "num_local_experts", "num_experts", "n_routed_experts")
	topK, hasTopK := num(config, "num_experts_per_tok")
	if hasExperts && experts != 0 && hasTopK && topK != 0 {
		expertHidden := ffn
		if v, ok := num(config, "moe_intermediate_size"); ok {
			expertHidden = v
		}
		sharedExperts, _ := num(config, "n_shared_experts")
		if sharedExperts == 0 {
			if intermediate, ok := num(config, "shared_expert_intermediate_size"); ok && intermediate != 0 {
				sharedExperts = math.Floor(intermediate/expertHidden + 0.5)
			}
		}
		denseLayers, _ := num(config, "first_k_dense_replace")
		spec.MoE = &design.MoE{
			Experts: experts, TopK: topK, ExpertHidden: expertHidden,
			SharedExperts: sharedExperts,
			RouterBias:    typ == "deepseek_v3",
			DenseLayers:   denseLayers,
		}
		if step, ok := num(config, "decoder_sparse_step"); ok && step != 1 {
			warnings = append(warnings, fmt.Sprintf(
				"decoder_sparse_step is %s, so only every %sth layer is sparse. This import makes them all sparse.",
				jsNum(step), jsNum(step)))
		}
		if only, ok := config["mlp_only_layers"].([]any); ok && len(only) > 0 {
			warnings = append(warnings, fmt.Sprintf(
				"mlp_only_layers lists %d dense layer(s) that this import does not reproduce.", len(only)))
		}
	}

	// --- Latent attention ----------------------------------------------------
	if kvLora, ok := num(config, "kv_lora_rank"); ok && kvLora != 0 {
		qLora, hasQ := num(config, "q_lora_rank")
		if !hasQ || qLora == 0 {
			warnings = append(warnings,
				"This model compresses keys and values but not queries. The import uses a full-width query path, "+
					"which changes the parameter count.")
			qLora = dModel
		}
		nope, err := required(config, "qk_nope_head_dim")
		if err != nil {
			return nil, err
		}
		ropeDim, err := required(config, "qk_rope_head_dim")
		if err != nil {
			return nil, err
		}
		vDim := headDim
		if v, ok := num(config, "v_head_dim"); ok {
			vDim = v
		}
		spec.MLA = &design.MLA{
			QLora: qLora, KVLora: kvLora, NopeDim: nope, RopeDim: ropeDim, VDim: vDim,
		}
		spec.HeadDim = nope + ropeDim
	}

	if v, ok := num(config, "num_nextn_predict_layers"); ok && v != 0 {
		warnings = append(warnings,
			"This model has a multi-token-prediction module, which the import leaves out. "+
				"Its published total may include those weights.")
	}

	return &Result{Doc: design.DecoderOnly(spec), Warnings: warnings}, nil
}

// ImportJSON reads a config.json from its text.
func ImportJSON(text string, name string) (*Result, error) {
	var parsed Config
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		return nil, fmt.Errorf("could not parse config.json: %w", err)
	}
	if parsed == nil {
		return nil, fmt.Errorf("config.json is not an object")
	}
	return Import(parsed, name)
}

// jsNum writes a number into a message the way the other engine does.
func jsNum(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e21 {
		return fmt.Sprintf("%d", int64(v))
	}
	return fmt.Sprint(v)
}
