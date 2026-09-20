package catalog_test

import (
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
)

// Gated DeltaNet, which is linear attention and therefore caches nothing.
//
// The parameter count is checked against the shape the reference implementation
// has — one projection for q, k, v and the gate together, a small one for the
// two per-head scalars, a depthwise convolution over q, k and v, a norm and an
// output projection — because there is no published figure for a layer on its
// own. What is not approximate, and is the reason the block exists, is that its
// state is a matrix per head that does not grow with the sequence.

const (
	gdnD  = 2048.0
	gdnH  = 16.0
	gdnDk = 128.0
	gdnDv = 128.0
	gdnK  = 4.0
)

func gdnDoc(t *testing.T, params map[string]any) *ir.Doc {
	t.Helper()
	return &ir.Doc{
		Version: ir.DocVersion,
		Meta:    ir.DocMeta{Name: "gdn-probe"},
		Symbols: map[string]ir.SymbolDef{
			"B": {Kind: "runtime", Number: 1, HasNumber: true},
			"T": {Kind: "runtime", Number: 4096, HasNumber: true},
			"D": {Kind: "design", Number: gdnD, HasNumber: true},
		},
		SymbolOrder: []string{"B", "T", "D"},
		Graph: ir.Graph{
			Nodes: []ir.NodeDef{
				{ID: "tokens", Type: "input", Params: map[string]any{"shape": "B T D"}},
				{ID: "blk", Type: "gated_deltanet_block", Params: params},
				{ID: "out", Type: "output"},
			},
			Edges: []ir.Edge{{"tokens:x", "blk:x"}, {"blk:y", "out:x"}},
		},
	}
}

func gdnParams() map[string]any {
	return map[string]any{
		"d_model": "D", "heads": gdnH, "head_dim": gdnDk,
		"v_head_dim": gdnDv, "conv_kernel": gdnK,
	}
}

func TestGatedDeltanetCountsItsWeights(t *testing.T) {
	doc := gdnDoc(t, gdnParams())
	res, err := analysis.Analyze(doc, analysis.Options{}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	if len(res.Params.Errors) != 0 {
		t.Fatalf("errors: %q", res.Params.Errors)
	}

	q := gdnH * gdnDk
	v := gdnH * gdnDv
	want := map[string]float64{
		// q, k, v and the output gate in one matrix.
		"blk/in_proj": gdnD * (2*q + 2*v),
		// Two scalars per head: how much the state decays, and how hard this
		// token writes to it.
		"blk/ba_proj": gdnD * 2 * gdnH,
		// Depthwise over q, k and v; the gate does not go through it.
		"blk/conv": (2*q + v) * gdnK,
		// The decay bias and log-decay scale the recurrence carries.
		"blk/scan":     2 * gdnH,
		"blk/norm":     v,
		"blk/out_proj": v * gdnD,
	}
	total := 0.0
	for path, n := range want {
		total += n
		if got := res.Params.ByPath[path]; got != n {
			t.Errorf("%s is %v, want %v", path, got, n)
		}
	}
	block := 0.0
	for path, n := range res.Params.ByPath {
		if strings.HasPrefix(path, "blk/") {
			block += n
		}
	}
	if block != total {
		t.Errorf("the block is %s parameters against %s accounted for; something else is in there: %v",
			analysis.FormatCount(block), analysis.FormatCount(total), res.Params.ByPath)
	}
}

// The point of the whole block.
func TestGatedDeltanetCachesNothingPerToken(t *testing.T) {
	doc := gdnDoc(t, gdnParams())
	res, err := analysis.Analyze(doc, analysis.Options{}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	if res.Kv.BytesPerToken != 0 {
		t.Errorf("a linear-attention layer holds %v bytes per token", res.Kv.BytesPerToken)
	}
	// A matrix per head, plus what the depthwise convolution has to remember.
	bytes := 2.0
	wantState := gdnH*gdnDk*gdnDv*bytes + (2*gdnH*gdnDk+gdnH*gdnDv)*(gdnK-1)*bytes
	if got := res.Kv.BytesPerSequenceFixed; got != wantState {
		t.Errorf("the state is %s per sequence, want %s",
			analysis.FormatBytes(got), analysis.FormatBytes(wantState))
	}

	// And it stays that size however long the sequence is, which is what
	// distinguishes it from attention.
	long := 131072.0
	far, err := analysis.Analyze(doc, analysis.Options{T: &long}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	if far.Kv.BytesPerSequenceFixed != wantState {
		t.Errorf("at 128k tokens the state is %s, want the same %s",
			analysis.FormatBytes(far.Kv.BytesPerSequenceFixed), analysis.FormatBytes(wantState))
	}
}

// Every wire inside the block lines up, including the two that a reader would
// most easily get backwards: the gate skips the convolution, and the per-head
// scalars come from the residual stream rather than from the projection.
func TestGatedDeltanetWiresUp(t *testing.T) {
	doc := gdnDoc(t, gdnParams())
	res := infer.Shapes(doc, ir.ResolveSymbols(doc), infer.Options{ExpandComposites: true})
	for _, issue := range res.Issues {
		if issue.Severity == "error" {
			t.Errorf("%s %s: %s", issue.Path, issue.Port, issue.Message)
		}
	}
	proj, ok := res.Resolved["blk/in_proj"]
	if !ok {
		t.Fatal("the block did not expand")
	}
	// q, k, v and the gate, and nothing else.
	if got, want := proj.Num("out_features"), 2*gdnH*gdnDk+2*gdnH*gdnDv; got != want {
		t.Errorf("the input projection produces %v, want %v", got, want)
	}
	conv, ok := res.Resolved["blk/conv"]
	if !ok {
		t.Fatal("no convolution")
	}
	if got, want := conv.Num("channels"), 2*gdnH*gdnDk+gdnH*gdnDv; got != want {
		t.Errorf("the convolution runs over %v channels, want %v — the gate should skip it", got, want)
	}
	ba, ok := res.Resolved["blk/ba_proj"]
	if !ok {
		t.Fatal("no gate projection")
	}
	if got := ba.Num("in_features"); got != gdnD {
		t.Errorf("the gate projection reads %v, want the residual stream's %v", got, gdnD)
	}
}

// A value head may be narrower than a key head, which several of the 2025
// hybrids do, and the state follows it.
func TestGatedDeltanetTakesANarrowerValueHead(t *testing.T) {
	params := gdnParams()
	params["v_head_dim"] = 64.0
	res, err := analysis.Analyze(gdnDoc(t, params), analysis.Options{}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	for _, e := range res.Params.Errors {
		t.Errorf("error: %s", e)
	}
	bytes := 2.0
	wantState := gdnH*gdnDk*64*bytes + (2*gdnH*gdnDk+gdnH*64)*(gdnK-1)*bytes
	if got := res.Kv.BytesPerSequenceFixed; got != wantState {
		t.Errorf("the state is %s, want %s", analysis.FormatBytes(got), analysis.FormatBytes(wantState))
	}
	// Leaving it out means the same width as a key head.
	same := gdnParams()
	delete(same, "v_head_dim")
	plain, err := analysis.Analyze(gdnDoc(t, same), analysis.Options{}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	full, err := analysis.Analyze(gdnDoc(t, gdnParams()), analysis.Options{}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	if plain.Params.Total != full.Params.Total {
		t.Errorf("an absent v_head_dim gave %s, spelling it out gave %s",
			analysis.FormatCount(plain.Params.Total), analysis.FormatCount(full.Params.Total))
	}
}
