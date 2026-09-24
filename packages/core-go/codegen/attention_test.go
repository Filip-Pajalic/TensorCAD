package codegen_test

import (
	"strings"
	"testing"

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
		"def expression_attention(q, k, v, mask_mod=None, score_mod=None, mask_heads=False, mask_batch=False, scale=None):",
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
