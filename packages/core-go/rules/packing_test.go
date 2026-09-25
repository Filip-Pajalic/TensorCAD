package rules_test

import (
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
)

func findingsOf(t *testing.T, doc *ir.Doc, T float64, pack *catalog.Packing, rule string) []rules.Finding {
	t.Helper()
	rep, err := rules.Validate(doc, analysis.Options{T: &T, Packing: pack})
	if err != nil {
		t.Fatal(err)
	}
	var out []rules.Finding
	for _, f := range rep.Findings {
		if f.Rule == rule {
			out = append(out, f)
		}
	}
	return out
}

// A packing on a design that attends across documents changes nothing, and
// is told so; on one that keeps them apart it is the point.
func TestAPackingWithNothingToChangeIsNoted(t *testing.T) {
	pack := &catalog.Packing{Mean: 1024}
	found := findingsOf(t, presets.MustGet("llama-3-8b"), 8192, pack, "packing-unused")
	if len(found) != 1 || found[0].Severity != "info" {
		t.Errorf("no mask: %+v", found)
	}
	if found := findingsOf(t, keepsDocumentsApart(t, "documents"), 8192, pack, "packing-unused"); len(found) > 0 {
		t.Errorf("with the mask: %+v", found)
	}
	if found := findingsOf(t, presets.MustGet("llama-3-8b"), 8192, nil, "packing-unused"); len(found) > 0 {
		t.Errorf("no packing at all: %+v", found)
	}
}

// Whole blocks against the scores kept: noted past a quarter, warned about
// once the difference is a twentieth of what a token costs. Llama-3-8B in
// 1,024-token documents computes 1.37 times its kept attention, which is
// under a percent of its forward pass; nano-sort, whose attention is a large
// share of it, in 128-token documents is another matter.
func TestShortDocumentsAgainstTheBlocks(t *testing.T) {
	found := findingsOf(t, keepsDocumentsApart(t, "documents"), 8192, &catalog.Packing{Mean: 1024}, "document-blocks")
	if len(found) != 1 || found[0].Severity != "info" || !strings.Contains(found[0].Message, "computes 1.3") {
		t.Errorf("llama-3-8b: %+v", found)
	}
	if found := findingsOf(t, keepsDocumentsApart(t, "documents"), 8192, &catalog.Packing{Mean: 4096}, "document-blocks"); len(found) > 0 {
		t.Errorf("long documents: %+v", found)
	}

	small := presets.MustGet("nano-sort")
	small.Graph.Nodes = append(small.Graph.Nodes, ir.NodeDef{ID: "docs", Type: "input",
		Params: map[string]any{"shape": "B T", "dtype": "int64", "role": "documents"}})
	small.Graph.Edges = append(small.Graph.Edges, ir.Edge{"docs:x", "layers:doc"})
	for i := range small.Graph.Nodes {
		if stack := &small.Graph.Nodes[i]; stack.ID == "layers" {
			for j := range stack.Graph.Nodes {
				switch n := &stack.Graph.Nodes[j]; n.ID {
				case "_in":
					n.Params["ports"].(map[string]any)["doc"] = "B T"
				case "block":
					n.Params["mask"] = "doc(b, q) == doc(b, kv)"
				}
			}
			stack.Graph.Edges = append(stack.Graph.Edges, ir.Edge{"_in:doc", "block:doc"})
		}
	}
	found = findingsOf(t, small, 2048, &catalog.Packing{Mean: 128}, "document-blocks")
	if len(found) != 1 || found[0].Severity != "warning" {
		t.Errorf("nano-sort: %+v", found)
	}
}
