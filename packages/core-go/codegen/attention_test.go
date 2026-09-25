package codegen_test

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/tensorcad/core/codegen"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
)

// withAttention is a preset with its stacked block's attention expressions
// set, the way the inspector sets them.
func withAttention(t *testing.T, preset string, params map[string]any) *ir.Doc {
	t.Helper()
	doc := presets.MustGet(preset)
	for i := range doc.Graph.Nodes {
		n := &doc.Graph.Nodes[i]
		if n.ID != "layers" || n.Graph == nil {
			continue
		}
		for j := range n.Graph.Nodes {
			inner := &n.Graph.Nodes[j]
			if inner.ID == "block" {
				for k, v := range params {
					inner.Params[k] = v
				}
				return doc
			}
		}
	}
	t.Fatalf("%s has no layers/block", preset)
	return nil
}

// An expression is generated as FlexAttention's two functions, with causal,
// the window and the cap folded into them, and the same functions are what
// the eager form applies — so there is one statement of what the layer does.
func TestExpressionsBecomeFlexAttentionFunctions(t *testing.T) {
	doc := withAttention(t, "nano-sort", map[string]any{
		"mask":          "kv < P or q - kv < W",
		"score":         "score - 2 ** (-8 * (h + 1) / heads) * (q - kv)",
		"logit_softcap": 30,
	})
	doc.Symbols["P"] = ir.SymbolDef{Kind: "literal", Number: 2, HasNumber: true}
	doc.Symbols["W"] = ir.SymbolDef{Kind: "literal", Number: 4, HasNumber: true}
	model := modelOf(t, doc)
	for _, want := range []string{
		"def expression_attention(\n    q, k, v, mask_mod=None, score_mod=None, mask_heads=False, mask_batch=False, scale=None, sinks=None\n):",
		"def mask_mod_1(b, h, q_idx, kv_idx):\n" +
			"    \"\"\"kv <= q and (kv < 2 or q - kv < 4)\"\"\"\n" +
			"    return (kv_idx <= q_idx) & ((kv_idx < 2) | ((q_idx - kv_idx) < 4))\n",
		"def score_mod_1(score, b, h, q_idx, kv_idx):\n" +
			"    \"\"\"30 * tanh((score - 2 ** (-8 * (h + 1) / 3) * (q - kv)) / 30)\"\"\"\n",
		"= expression_attention(",
		"mask_mod=mask_mod_1, score_mod=score_mod_1)",
	} {
		if !strings.Contains(model, want) {
			t.Errorf("missing:\n%s", want)
		}
	}
	// Three layers, one class, one pair of functions.
	if n := strings.Count(model, "def mask_mod_"); n != 1 {
		t.Errorf("%d mask functions, want 1", n)
	}
	// The window and the cap went into the functions, not to FlashAttention.
	if strings.Contains(model, "fused_attention(") {
		t.Error("an expression layer also calls the FlashAttention helper")
	}
}

// A mask that reads the head is built per head, and one that does not is
// built once and broadcast, which is the difference between a block mask of
// heads × blocks and one of blocks.
func TestAMaskThatReadsTheHeadIsBuiltPerHead(t *testing.T) {
	model := modelOf(t, withAttention(t, "nano-sort", map[string]any{"mask": "h == 0 or kv == q"}))
	if !strings.Contains(model, "mask_heads=True") {
		t.Error("a per-head mask is not built per head")
	}
	model = modelOf(t, withAttention(t, "nano-sort", map[string]any{"mask": "kv < 3 or kv == q"}))
	if strings.Contains(model, "mask_heads=True") {
		t.Error("a mask that does not read the head is built per head")
	}
}

// With no expression nothing changes: the plain case keeps PyTorch's own
// kernel, and a window or a cap alone keeps FlashAttention's.
func TestNoExpressionNoFlexAttention(t *testing.T) {
	for _, preset := range []string{"gpt2-small", "gemma-2-9b"} {
		if model := modelOf(t, presets.MustGet(preset)); strings.Contains(model, "expression_attention") {
			t.Errorf("%s uses the expression helper without an expression", preset)
		}
	}
}

// An encoder-decoder takes its source and its target by name, and its decoder
// stack hands the encoder's output to every layer rather than threading it
// through as though each layer made a new one.
func TestAnEncoderDecoderTakesBothInputs(t *testing.T) {
	var doc ir.Doc
	raw, err := os.ReadFile("testdata/seq2seq.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	model := modelOf(t, &doc)
	for _, want := range []string{
		"    def forward(self, src, tgt):",
		"        for layer in self.decoder:\n            decoder_x = layer(enc_norm_y, decoder_x)",
		"        for layer in self.encoder:\n            encoder_x = layer(encoder_x)",
		// Cross-attention's keys and values are the source's, every one seen.
		"attn_y = F.scaled_dot_product_attention(q_heads_y, k_heads_y, v_heads_y, is_causal=False)",
	} {
		if !strings.Contains(model, want) {
			t.Errorf("missing:\n%s", want)
		}
	}
	// And a design with one input takes it as ids, as it always has.
	if one := modelOf(t, presets.MustGet("gpt2-small")); !strings.Contains(one, "def forward(self, ids):") {
		t.Error("a one-input model no longer takes ids")
	}
}

// A score that reads a table is generated as a factory: handed the table, it
// returns the score_mod, which has the table in scope. T5 keeps one table per
// stack, a parameter outside it, handed to every layer.
func TestAScoreReadsItsTableThroughAFactory(t *testing.T) {
	var doc ir.Doc
	raw, err := os.ReadFile("testdata/t5ish.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	model := modelOf(t, &doc)
	for _, want := range []string{
		"def t5_bucket(relative_position, num_buckets, max_distance, bidirectional):",
		"def score_mod_1(rel):\n" +
			"    \"\"\"score + rel(t5_bucket(kv - q, 32, 128, true), h)\"\"\"\n\n" +
			"    def score_mod(score, b, h, q_idx, kv_idx):\n" +
			"        return score + rel[t5_bucket(kv_idx - q_idx, 32, 128, True), h]\n\n" +
			"    return score_mod\n",
		"score_mod=score_mod_1(rel)",
		"self.enc_bias = nn.Parameter(torch.zeros(32, 4))",
		"encoder_x = layer(self.enc_bias, encoder_x)",
		"decoder_x = layer(enc_norm_y, self.dec_bias, decoder_x)",
	} {
		if !strings.Contains(model, want) {
			t.Errorf("missing:\n%s", want)
		}
	}
}

// A mask that keeps documents apart reads them the way a score reads a table:
// through a factory the attention calls with this batch's documents, indexed
// by row and position. The factory marks what it returns with what it read,
// which is how the block mask is built once for a batch and shared by every
// layer rather than built by each.
func TestAMaskReadsTheDocumentsThroughAFactory(t *testing.T) {
	doc := withAttention(t, "nano-sort", map[string]any{"mask": "doc(b, q) == doc(b, kv)"})
	doc.Graph.Nodes = append(doc.Graph.Nodes, ir.NodeDef{ID: "docs", Type: "input",
		Params: map[string]any{"shape": "B T", "dtype": "int64", "role": "documents"}})
	doc.Graph.Edges = append(doc.Graph.Edges, ir.Edge{"docs:x", "layers:doc"})
	for i := range doc.Graph.Nodes {
		if stack := &doc.Graph.Nodes[i]; stack.ID == "layers" {
			for j := range stack.Graph.Nodes {
				if n := &stack.Graph.Nodes[j]; n.ID == "_in" {
					n.Params["ports"].(map[string]any)["doc"] = "B T"
				}
			}
			stack.Graph.Edges = append(stack.Graph.Edges, ir.Edge{"_in:doc", "block:doc"})
		}
	}
	out := codegen.GenerateTorch(doc, codegen.Options{})
	model := modelOf(t, doc)
	for _, want := range []string{
		"def forward(self, tokens, docs):",
		"def mask_mod_1(doc):",
		"return (kv_idx <= q_idx) & (doc[b, q_idx] == doc[b, kv_idx])",
		"mask_mod=mask_mod_1(doc), mask_batch=True",
		"    mask_mod.reads = (doc,)\n    return mask_mod\n",
		"def _block_mask(create_block_mask, mask_mod, batch, heads, seq, keys, device):",
	} {
		if !strings.Contains(model, want) {
			t.Errorf("missing: %s", want)
		}
	}
	if len(out.Warnings) != 0 {
		t.Errorf("warnings: %v", out.Warnings)
	}
}

// A rotation that restarts at every document is turned by the positions the
// model is given, through a helper of its own, so a design that rotates by
// the index generates exactly what it did.
func TestARotationTurnsByThePositionsItIsGiven(t *testing.T) {
	doc := withAttention(t, "llama-3-8b", map[string]any{"positions": true})
	doc.Graph.Nodes = append(doc.Graph.Nodes, ir.NodeDef{ID: "pos", Type: "input",
		Params: map[string]any{"shape": "B T", "dtype": "int64", "role": "positions"}})
	doc.Graph.Edges = append(doc.Graph.Edges, ir.Edge{"pos:x", "layers:pos"})
	for i := range doc.Graph.Nodes {
		if stack := &doc.Graph.Nodes[i]; stack.ID == "layers" {
			for j := range stack.Graph.Nodes {
				if n := &stack.Graph.Nodes[j]; n.ID == "_in" {
					n.Params["ports"].(map[string]any)["pos"] = "B T"
				}
			}
			stack.Graph.Edges = append(stack.Graph.Edges, ir.Edge{"_in:pos", "block:pos"})
		}
	}
	model := modelOf(t, doc)
	for _, want := range []string{
		"def rotary_at(rope, x, positions):",
		"rope_q_y = rotary_at(self.rope_q, q_heads_y, pos)",
		"rope_k_y = rotary_at(self.rope_k, k_heads_y, pos)",
		"def forward(self, tokens, pos):",
	} {
		if !strings.Contains(model, want) {
			t.Errorf("missing: %s", want)
		}
	}
	if strings.Contains(modelOf(t, presets.MustGet("llama-3-8b")), "rotary_at") {
		t.Error("a design that rotates by the index has the helper for positions")
	}
}
