package catalog_test

import (
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/shapes"
)

// Two things the nanoGPT speedrun does, which turn out to be one primitive.
//
// Value embeddings mix a second look at the tokens into the value stream. A
// U-net skip mixes an earlier layer's output back into a later layer's input.
// Both are `w0 * a + w1 * b` with the weights learned, which is not `add` (no
// weights) and not `mul` (no parameters).

const (
	mixD  = 384.0
	mixV  = 4096.0
	mixF  = 1536.0
	mixH  = 6.0
	mixDh = 64.0
)

func mixBlock(ve bool) map[string]any {
	p := map[string]any{
		"d_model": "D", "heads": mixH, "kv_heads": mixH, "head_dim": mixDh,
		"ffn_hidden": "F", "norm": "rmsnorm", "mlp": "dense", "act": "relu2",
	}
	if ve {
		p["value_embeddings"] = true
	}
	return p
}

// speedrunDoc is `layers` blocks, the first `withVe` of them reading the tokens
// twice, and the second half mixing the first half back in.
func speedrunDoc(t *testing.T, layers, withVe int, skips bool) *ir.Doc {
	t.Helper()
	doc := &ir.Doc{
		Version: ir.DocVersion,
		Meta:    ir.DocMeta{Name: "speedrun"},
		Symbols: map[string]ir.SymbolDef{
			"B": {Kind: "runtime", Number: 2, HasNumber: true},
			"T": {Kind: "runtime", Number: 128, HasNumber: true},
			"D": {Kind: "design", Number: mixD, HasNumber: true},
			"F": {Kind: "design", Number: mixF, HasNumber: true},
			"V": {Kind: "design", Number: mixV, HasNumber: true},
		},
		SymbolOrder: []string{"B", "T", "D", "F", "V"},
	}
	nodes := []ir.NodeDef{
		{ID: "tokens", Type: "input", Params: map[string]any{"shape": "B T", "dtype": "int64"}},
		{ID: "embed", Type: "embedding", Params: map[string]any{"vocab": "V", "dim": "D"}},
	}
	edges := []ir.Edge{{"tokens:x", "embed:ids"}}
	if withVe > 0 {
		nodes = append(nodes, ir.NodeDef{ID: "value_embed", Type: "embedding",
			Params: map[string]any{"vocab": "V", "dim": "D"}})
		edges = append(edges, ir.Edge{"tokens:x", "value_embed:ids"})
	}
	tail := "embed:y"
	for i := 0; i < layers; i++ {
		id := blockID(i)
		if skips && i >= layers/2 {
			skip := "skip" + itoa(i)
			nodes = append(nodes, ir.NodeDef{ID: skip, Type: "mix", Params: map[string]any{"dim": "D"}})
			edges = append(edges,
				ir.Edge{tail, skip + ":a"},
				ir.Edge{blockID(layers-1-i) + ":y", skip + ":b"})
			tail = skip + ":y"
		}
		nodes = append(nodes, ir.NodeDef{ID: id, Type: "transformer_block", Params: mixBlock(i < withVe)})
		edges = append(edges, ir.Edge{tail, id + ":x"})
		if i < withVe {
			edges = append(edges, ir.Edge{"value_embed:y", id + ":ve"})
		}
		tail = id + ":y"
	}
	nodes = append(nodes,
		ir.NodeDef{ID: "final_norm", Type: "rmsnorm", Params: map[string]any{"dim": "D"}},
		ir.NodeDef{ID: "head", Type: "lm_head",
			Params: map[string]any{"vocab": "V", "dim": "D", "tied": true}},
		ir.NodeDef{ID: "logits", Type: "output"})
	edges = append(edges,
		ir.Edge{tail, "final_norm:x"},
		ir.Edge{"final_norm:y", "head:x"},
		ir.Edge{"head:y", "logits:x"})
	doc.Graph = ir.Graph{Nodes: nodes, Edges: edges}
	return doc
}

func blockID(i int) string { return "blk" + itoa(i) }

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

func mixAnalyzed(t *testing.T, doc *ir.Doc) *analysis.Result {
	t.Helper()
	res, err := analysis.Analyze(doc, analysis.Options{}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	for _, e := range res.Params.Errors {
		t.Errorf("error: %s", e)
	}
	return res
}

// A mix is two scalars, whatever it is mixing.
func TestAMixIsTwoScalars(t *testing.T) {
	doc := speedrunDoc(t, 4, 0, true)
	res := mixAnalyzed(t, doc)
	skips := 0.0
	for path, n := range res.Params.ByPath {
		if strings.HasPrefix(path, "skip") {
			skips += n
			if n != 2 {
				t.Errorf("%s costs %v parameters, want 2", path, n)
			}
		}
	}
	// Four layers, so the second two mix the first two back in.
	if skips != 4 {
		t.Errorf("the skips cost %v parameters in total, want 4", skips)
	}
}

// Value embeddings add a port, a mix, and no table: the table is at the top
// level and shared, which is the whole reason it arrives on a wire.
func TestValueEmbeddingsCostAMixPerLayerAndOneTable(t *testing.T) {
	plain := mixAnalyzed(t, speedrunDoc(t, 4, 0, false))
	withVe := mixAnalyzed(t, speedrunDoc(t, 4, 3, false))

	// One table, plus two scalars in each of the three layers that use it.
	want := mixV*mixD + 3*2
	if got := withVe.Params.Total - plain.Params.Total; got != want {
		t.Errorf("value embeddings cost %v parameters, want %v (one %v x %v table + 3 mixes)",
			got, want, mixV, mixD)
	}
	// And the table is counted once however many layers read it.
	if got := withVe.Params.ByPath["value_embed"]; got != mixV*mixD {
		t.Errorf("the table counts %v, want %v", got, mixV*mixD)
	}
}

// The port exists only when the block asks for it, because the expansion is
// static: a port that might be connected would mean a block that always carries
// the mix and only sometimes uses it.
func TestTheValueEmbeddingPortAppearsOnlyWhenAsked(t *testing.T) {
	doc := speedrunDoc(t, 4, 3, false)
	res := infer.Shapes(doc, ir.ResolveSymbols(doc), infer.Options{ExpandComposites: true})
	for _, issue := range res.Issues {
		if issue.Severity == "error" {
			t.Errorf("%s %s: %s", issue.Path, issue.Port, issue.Message)
		}
	}
	on, ok := res.Ports["blk0"]
	if !ok {
		t.Fatal("no ports on the first block")
	}
	if _, has := on.In["ve"]; !has {
		t.Error("a block that asked for value embeddings has no ve port")
	}
	off, ok := res.Ports["blk3"]
	if !ok {
		t.Fatal("no ports on the last block")
	}
	if _, has := off.In["ve"]; has {
		t.Error("a block that did not ask has a ve port anyway")
	}
	// And the mix is inside the attention, on the value path.
	if _, has := res.Outputs["blk0/attn/v_mix:y"]; !has {
		t.Errorf("no mix inside the first block's attention: %d outputs", len(res.Outputs))
	}
	if _, has := res.Outputs["blk3/attn/v_mix:y"]; has {
		t.Error("a block that did not ask for value embeddings has a mix anyway")
	}
}

// A U-net skip is not a repeating unit — layer i feeds layer L-1-i — so a stack
// that has them is written out. This is the check that it can be.
func TestUnetSkipsWireAcrossTheStack(t *testing.T) {
	doc := speedrunDoc(t, 6, 3, true)
	res := infer.Shapes(doc, ir.ResolveSymbols(doc), infer.Options{ExpandComposites: true})
	for _, issue := range res.Issues {
		if issue.Severity == "error" {
			t.Errorf("%s %s: %s", issue.Path, issue.Port, issue.Message)
		}
	}
	// Three skips, each carrying the stream's width.
	for _, id := range []string{"skip3", "skip4", "skip5"} {
		shape, ok := res.Outputs[id+":y"]
		if !ok {
			t.Errorf("%s produced nothing", id)
			continue
		}
		if got := shapes.ShapeToString(shape); got != "B T D" {
			t.Errorf("%s produces %q, want the stream's own shape", id, got)
		}
	}
	mixAnalyzed(t, doc)
}
