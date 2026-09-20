package catalog_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/shapes"
)

// Multi-token prediction, wired the way DeepSeek-V3 describes it.
//
// A module is four things in a row and then two more blocks: normalize the
// hidden state, normalize the embedding of the token ahead, join, project 2D
// down to D, run a transformer block, and go through the model's own output
// head. The head is shared, which is the part worth being careful about — it
// costs a second pass over the vocabulary and no weights at all.
//
// The arithmetic is pinned from first principles rather than against a
// published figure, because the figures everyone quotes for DeepSeek-V3 leave
// the MTP module out.

const (
	mtpD     = 1024.0
	mtpV     = 32000.0
	mtpF     = 4096.0
	mtpHeads = 16.0
	mtpDh    = 64.0
)

// mtpDoc is a small model with one MTP module hanging off the main stack.
func mtpDoc(t *testing.T, withMtp bool) *ir.Doc {
	t.Helper()
	doc := &ir.Doc{
		Version: ir.DocVersion,
		Meta:    ir.DocMeta{Name: "mtp-probe"},
		Symbols: map[string]ir.SymbolDef{
			"B": {Kind: "runtime", Number: 1, HasNumber: true},
			"T": {Kind: "runtime", Number: 512, HasNumber: true},
			"D": {Kind: "design", Number: mtpD, HasNumber: true},
			"V": {Kind: "design", Number: mtpV, HasNumber: true},
		},
		SymbolOrder: []string{"B", "T", "D", "V"},
	}
	block := map[string]any{
		"d_model": "D", "heads": mtpHeads, "kv_heads": mtpHeads, "head_dim": mtpDh,
		"ffn_hidden": mtpF, "norm": "rmsnorm", "mlp": "gated", "act": "silu",
	}
	nodes := []ir.NodeDef{
		{ID: "tokens", Type: "input", Params: map[string]any{"shape": "B T", "dtype": "int64"}},
		{ID: "embed", Type: "embedding", Params: map[string]any{"vocab": "V", "dim": "D"}},
		{ID: "block", Type: "transformer_block", Params: block},
		{ID: "final_norm", Type: "rmsnorm", Params: map[string]any{"dim": "D"}},
		{ID: "head", Type: "lm_head", Params: map[string]any{"vocab": "V", "dim": "D", "tied": true}},
		{ID: "logits", Type: "output"},
	}
	edges := []ir.Edge{
		{"tokens:x", "embed:ids"},
		{"embed:y", "block:x"},
		{"block:y", "final_norm:x"},
		{"final_norm:y", "head:x"},
		{"head:y", "logits:x"},
	}
	if withMtp {
		nodes = append(nodes,
			ir.NodeDef{ID: "mtp", Type: "mtp_head", Params: map[string]any{"d_model": "D", "by": 1.0}},
			ir.NodeDef{ID: "mtp_block", Type: "transformer_block", Params: block},
			ir.NodeDef{ID: "mtp_head_out", Type: "lm_head",
				Params: map[string]any{"vocab": "V", "dim": "D", "tied": true}},
			ir.NodeDef{ID: "mtp_logits", Type: "output"})
		edges = append(edges,
			ir.Edge{"block:y", "mtp:x"},
			ir.Edge{"embed:y", "mtp:e"},
			ir.Edge{"mtp:y", "mtp_block:x"},
			ir.Edge{"mtp_block:y", "mtp_head_out:x"},
			ir.Edge{"mtp_head_out:y", "mtp_logits:x"})
	}
	doc.Graph = ir.Graph{Nodes: nodes, Edges: edges}
	return doc
}

func analyzed(t *testing.T, doc *ir.Doc) *analysis.Result {
	t.Helper()
	res, err := analysis.Analyze(doc, analysis.Options{}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	return res
}

func TestMtpModuleCostsOneBlockAndOneProjection(t *testing.T) {
	plain := analyzed(t, mtpDoc(t, false))
	withMtp := analyzed(t, mtpDoc(t, true))

	if n := len(withMtp.Params.Errors); n != 0 {
		t.Fatalf("errors: %q", withMtp.Params.Errors)
	}

	// One transformer block, one 2D x D projection, and two norms of D each.
	// The output head is shared, so it adds nothing.
	// ByPath is per primitive, so a composite is the sum of what it expanded to.
	blockParams := 0.0
	for path, n := range plain.Params.ByPath {
		if strings.HasPrefix(path, "block/") {
			blockParams += n
		}
	}
	if blockParams <= 0 {
		t.Fatalf("the plain model has no block: %v", plain.Params.ByPath)
	}
	wantExtra := blockParams + 2*mtpD*mtpD + 2*mtpD
	gotExtra := withMtp.Params.Total - plain.Params.Total
	if gotExtra != wantExtra {
		t.Errorf("the module adds %s parameters, want %s (one block %s + a %v x %v projection + two norms)",
			analysis.FormatCount(gotExtra), analysis.FormatCount(wantExtra),
			analysis.FormatCount(blockParams), 2*mtpD, mtpD)
	}

	// The shared head costs a second pass over the vocabulary and no weights.
	if got := withMtp.Params.ByPath["mtp_head_out"]; got != 0 {
		t.Errorf("the shared head counted %v parameters", got)
	}
	headFlops := plain.Flops.ByPath["head"]
	if headFlops != 2*mtpV*mtpD {
		t.Fatalf("the head is %v FLOPs per token, expected %v", headFlops, 2*mtpV*mtpD)
	}
	if got := withMtp.Flops.ByPath["mtp_head_out"]; got != headFlops {
		t.Errorf("the second pass over the vocabulary is %v FLOPs, want %v", got, headFlops)
	}
}

// The module reads the embedding of the token it is predicting, and the shift
// that makes that true is inside the block rather than left to whoever wires it.
func TestMtpModuleShiftsTheEmbeddingItself(t *testing.T) {
	doc := mtpDoc(t, true)
	res := infer.Shapes(doc, ir.ResolveSymbols(doc), infer.Options{ExpandComposites: true})
	for _, issue := range res.Issues {
		if issue.Severity == "error" {
			t.Errorf("%s %s: %s", issue.Path, issue.Port, issue.Message)
		}
	}
	before, ok := res.Outputs["mtp/norm_h:y"]
	if !ok {
		t.Fatal("the module did not expand")
	}
	after, ok := res.Outputs["mtp/ahead:y"]
	if !ok {
		t.Fatal("no shift inside the module")
	}
	// A shift is the identity on shapes, which is the whole reason nothing
	// needed one until now.
	if shapes.ShapeToString(after) != shapes.ShapeToString(before) {
		t.Errorf("the shift changed the shape: %s became %s",
			shapes.ShapeToString(before), shapes.ShapeToString(after))
	}
	// And the projection reads twice the width, which is what makes it a join.
	proj, ok := res.Resolved["mtp/proj"]
	if !ok {
		t.Fatal("no projection in the expansion")
	}
	if got := proj.Num("in_features"); got != 2*mtpD {
		t.Errorf("the projection reads %v, want %v", got, 2*mtpD)
	}
}

// Depth is stacking, not a parameter: each module reads the one below.
func TestMtpModulesStack(t *testing.T) {
	doc := mtpDoc(t, true)
	block := map[string]any{
		"d_model": "D", "heads": mtpHeads, "kv_heads": mtpHeads, "head_dim": mtpDh,
		"ffn_hidden": mtpF, "norm": "rmsnorm", "mlp": "gated", "act": "silu",
	}
	doc.Graph.Nodes = append(doc.Graph.Nodes,
		ir.NodeDef{ID: "mtp2", Type: "mtp_head", Params: map[string]any{"d_model": "D", "by": 2.0}},
		ir.NodeDef{ID: "mtp2_block", Type: "transformer_block", Params: block},
		ir.NodeDef{ID: "mtp2_head", Type: "lm_head",
			Params: map[string]any{"vocab": "V", "dim": "D", "tied": true}},
		ir.NodeDef{ID: "mtp2_logits", Type: "output"})
	doc.Graph.Edges = append(doc.Graph.Edges,
		ir.Edge{"mtp_block:y", "mtp2:x"},
		ir.Edge{"embed:y", "mtp2:e"},
		ir.Edge{"mtp2:y", "mtp2_block:x"},
		ir.Edge{"mtp2_block:y", "mtp2_head:x"},
		ir.Edge{"mtp2_head:y", "mtp2_logits:x"})

	one := analyzed(t, mtpDoc(t, true))
	two := analyzed(t, doc)
	first := one.Params.Total - analyzed(t, mtpDoc(t, false)).Params.Total
	second := two.Params.Total - one.Params.Total
	if math.Abs(first-second) > 0.5 {
		t.Errorf("the second module costs %s, the first cost %s; they are the same shape",
			analysis.FormatCount(second), analysis.FormatCount(first))
	}
	// And the second one looks two tokens ahead rather than one.
	res := infer.Shapes(doc, ir.ResolveSymbols(doc), infer.Options{ExpandComposites: true})
	ahead, ok := res.Resolved["mtp2/ahead"]
	if !ok {
		t.Fatal("no shift in the second module")
	}
	if got := ahead.Num("by"); got != 2 {
		t.Errorf("the second module looks %v ahead, want 2", got)
	}
}

// Shifting the other way would let a token see its own future.
func TestABackwardShiftIsRefused(t *testing.T) {
	doc := mtpDoc(t, true)
	for i := range doc.Graph.Nodes {
		if doc.Graph.Nodes[i].ID == "mtp" {
			doc.Graph.Nodes[i].Params = map[string]any{"d_model": "D", "by": -1.0}
		}
	}
	res := infer.Shapes(doc, ir.ResolveSymbols(doc), infer.Options{ExpandComposites: true})
	found := false
	for _, issue := range res.Issues {
		if issue.Severity == "error" {
			found = true
		}
	}
	if !found {
		t.Error("a backward shift was accepted")
	}
}
