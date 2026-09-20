package codegen_test

import (
	"strings"
	"testing"

	"github.com/tensorcad/core/codegen"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
)

// What a generated model gives back, and what it shares.
//
// Both of these were found by pointing `tensorcad-runtime verify` at a design
// with multi-token prediction, which is the first thing to have two output
// heads. Nothing else in the library had either, so nothing else noticed.

func modelOf(t *testing.T, doc *ir.Doc) string {
	t.Helper()
	got := codegen.GenerateTorch(doc, codegen.Options{})
	for _, file := range got.Files {
		if file.Path == "model.py" {
			return file.Contents
		}
	}
	t.Fatal("no model.py")
	return ""
}

// A design with two outputs returns both. I-JEPA has had two since it was
// added — a prediction and the target it is compared against — and the emitted
// model returned only the target, which makes the file useless for the loss it
// exists to compute.
func TestEveryOutputIsReturned(t *testing.T) {
	model := modelOf(t, presets.MustGet("ijepa-vit-h14"))
	var ret string
	for _, line := range strings.Split(model, "\n") {
		if strings.HasPrefix(line, "        return ") {
			ret = strings.TrimSpace(line)
		}
	}
	if !strings.Contains(ret, ",") {
		t.Errorf("the model returns %q; I-JEPA has a prediction and a target", ret)
	}
	// And a design with one output still returns a tensor rather than a tuple
	// of one.
	one := modelOf(t, presets.MustGet("gpt2-small"))
	for _, line := range strings.Split(one, "\n") {
		if strings.HasPrefix(line, "        return ") && strings.Contains(line, ",") {
			t.Errorf("a single-output model returns %q", strings.TrimSpace(line))
		}
	}
}

// Every head that says it is tied is tied. A multi-token predictor has one
// output head per prediction depth and they all share the embedding; emitting
// the first as shared and the rest as their own weights is a model that does
// not have the parameters the analysis counted.
func TestEveryTiedHeadIsTied(t *testing.T) {
	doc := twoHeads(t)
	model := modelOf(t, doc)
	for _, want := range []string{
		"self.head.weight = self.embed.weight",
		"self.second.weight = self.embed.weight",
	} {
		if !strings.Contains(model, want) {
			t.Errorf("missing %q", want)
		}
	}

	// An untied head keeps its own weights, and is not tied by accident.
	for i := range doc.Graph.Nodes {
		if doc.Graph.Nodes[i].ID == "second" {
			doc.Graph.Nodes[i].Params = map[string]any{"vocab": "V", "dim": "D", "tied": false}
		}
	}
	if got := modelOf(t, doc); strings.Contains(got, "self.second.weight = self.embed.weight") {
		t.Error("an untied head was tied")
	}
}

// twoHeads is the smallest design with two output heads sharing one embedding.
func twoHeads(t *testing.T) *ir.Doc {
	t.Helper()
	head := func(id string) ir.NodeDef {
		return ir.NodeDef{ID: id, Type: "lm_head",
			Params: map[string]any{"vocab": "V", "dim": "D", "tied": true}}
	}
	return &ir.Doc{
		Version: ir.DocVersion,
		Meta:    ir.DocMeta{Name: "two-heads"},
		Symbols: map[string]ir.SymbolDef{
			"B": {Kind: "runtime", Number: 1, HasNumber: true},
			"T": {Kind: "runtime", Number: 128, HasNumber: true},
			"D": {Kind: "design", Number: 256, HasNumber: true},
			"V": {Kind: "design", Number: 1024, HasNumber: true},
		},
		SymbolOrder: []string{"B", "T", "D", "V"},
		Graph: ir.Graph{
			Nodes: []ir.NodeDef{
				{ID: "tokens", Type: "input", Params: map[string]any{"shape": "B T", "dtype": "int64"}},
				{ID: "embed", Type: "embedding", Params: map[string]any{"vocab": "V", "dim": "D"}},
				head("head"),
				{ID: "logits", Type: "output"},
				head("second"),
				{ID: "more", Type: "output"},
			},
			Edges: []ir.Edge{
				{"tokens:x", "embed:ids"},
				{"embed:y", "head:x"},
				{"head:y", "logits:x"},
				{"embed:y", "second:x"},
				{"second:y", "more:x"},
			},
		},
	}
}
