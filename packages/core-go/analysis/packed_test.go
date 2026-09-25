package analysis_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
)

// Llama 3 trained with a mask between the documents of a packed row. This is
// llama-3-8b with that mask written in: a documents input, wired through the
// stack to every layer's attention.
func documentMasked(t *testing.T) *ir.Doc {
	t.Helper()
	doc := presets.MustGet("llama-3-8b")
	doc.Graph.Nodes = append(doc.Graph.Nodes, ir.NodeDef{ID: "docs", Type: "input",
		Params: map[string]any{"shape": "B T", "dtype": "int64", "role": "documents"}})
	doc.Graph.Edges = append(doc.Graph.Edges, ir.Edge{"docs:x", "layers:doc"})
	for i := range doc.Graph.Nodes {
		stack := &doc.Graph.Nodes[i]
		if stack.ID != "layers" {
			continue
		}
		for j := range stack.Graph.Nodes {
			n := &stack.Graph.Nodes[j]
			switch n.ID {
			case "_in":
				n.Params["ports"].(map[string]any)["doc"] = "B T"
			case "block":
				n.Params["mask"] = "doc(b, q) == doc(b, kv)"
			}
		}
		stack.Graph.Edges = append(stack.Graph.Edges, ir.Edge{"_in:doc", "block:doc"})
	}
	return doc
}

func analyzeAt(t *testing.T, doc *ir.Doc, T float64, pack *catalog.Packing) *analysis.Result {
	t.Helper()
	res, err := analysis.Analyze(doc, analysis.Options{T: &T, Packing: pack}, analysis.Inputs{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) > 0 {
		t.Fatalf("errors: %v", res.Errors)
	}
	return res
}

// The documents reach every layer's attention, and the wiring checks.
func TestADocumentMaskIsWired(t *testing.T) {
	doc := documentMasked(t)
	res := infer.Shapes(doc, ir.ResolveSymbols(doc), infer.Options{ExpandComposites: true})
	for _, issue := range res.Issues {
		if issue.Severity == "error" {
			t.Errorf("%s: %s", issue.Path, issue.Message)
		}
	}
	for _, path := range []string{"layers/block", "layers/block/attn", "layers/block/attn/attn"} {
		if _, ok := res.Ports[path].In["doc"]; !ok {
			t.Errorf("%s has no doc input", path)
		}
	}
}

// Without a packing a row is one document, and a design that keeps documents
// apart costs exactly what it did without the mask.
func TestWithoutAPackingNothingMoves(t *testing.T) {
	plain := analyzeAt(t, presets.MustGet("llama-3-8b"), 8192, nil)
	masked := analyzeAt(t, documentMasked(t), 8192, nil)
	if masked.Flops.FwdTotal != plain.Flops.FwdTotal || masked.Flops.TrainPerToken != plain.Flops.TrainPerToken {
		t.Errorf("unpacked %v against %v", masked.Flops.FwdTotal, plain.Flops.FwdTotal)
	}
	if masked.Flops.Packed != nil {
		t.Error("packed figures with no packing")
	}
	// And a packing changes nothing for a design that attends across
	// documents anyway.
	packed := analyzeAt(t, presets.MustGet("llama-3-8b"), 8192, &catalog.Packing{Mean: 1024})
	if packed.Flops.Packed != nil || packed.Cost.GpuHours != plain.Cost.GpuHours {
		t.Error("a packing moved a design with no document mask")
	}
}

// Llama-3-8B at 8,192 tokens in rows of 1,024-token documents. A row cut from
// a stream keeps 491.1 keys a query where one document keeps 4,096, so the
// attention is an eighth of what it was, and training is eleven percent
// cheaper. Serving is not packed, and does not move.
func TestPackingIsTrainingOnly(t *testing.T) {
	one := analyzeAt(t, documentMasked(t), 8192, nil)
	res := analyzeAt(t, documentMasked(t), 8192, &catalog.Packing{Mean: 1024})
	p := res.Flops.Packed
	if p == nil {
		t.Fatal("no packed figures")
	}
	keys := 4096 * p.FwdAttention / one.Flops.FwdAttention
	if math.Abs(keys-491.1) > 0.005*491.1 {
		t.Errorf("packed at %v keys a query, want 491.1", keys)
	}
	if res.Flops.FwdTotal != one.Flops.FwdTotal || res.Throughput.DecodeTokensPerSecond != one.Throughput.DecodeTokensPerSecond {
		t.Error("the serving figures moved with the packing")
	}
	if got := 1 - p.TrainPerToken/one.Flops.TrainPerToken; math.Abs(got-0.110) > 0.002 {
		t.Errorf("training cheaper by %v", got)
	}
	// The training cost is counted from the packed figure.
	if got := res.Cost.GpuHours / one.Cost.GpuHours; math.Abs(got-p.TrainPerToken/one.Flops.TrainPerToken) > 1e-12 {
		t.Errorf("GPU-hours moved by %v", got)
	}
	// FlexAttention computes whole blocks, and 1,024-token documents cut a
	// stream's rows through blocks at every boundary: 1.37 times the scores
	// kept, exactly, for fixed lengths at every phase.
	if ratio := p.FwdAttentionBlocks / p.FwdAttention; math.Abs(ratio-1.37) > 0.03 {
		t.Errorf("the kernel computes %.3fx the scores kept", ratio)
	}
	if res.Options.Packing == nil || res.Options.Packing.Mean != 1024 {
		t.Error("the report does not say what it assumed")
	}
	// The same mean with exponential lengths costs nearly twice as much:
	// a token lands in a long document more often than a short one.
	spread := analyzeAt(t, documentMasked(t), 8192, &catalog.Packing{Mean: 1024, Spread: 1}).Flops.Packed
	if ratio := spread.FwdAttention / p.FwdAttention; ratio < 1.75 || ratio > 1.9 {
		t.Errorf("exponential against fixed: %v", ratio)
	}
}

func TestANonsensePackingIsRefused(t *testing.T) {
	T := 8192.0
	for _, p := range []catalog.Packing{{Mean: 0}, {Mean: 512, Spread: -1}, {Mean: math.NaN()}} {
		_, err := analysis.Analyze(documentMasked(t), analysis.Options{T: &T, Packing: &p}, analysis.Inputs{})
		if err == nil || !strings.Contains(err.Error(), "packing") {
			t.Errorf("%+v: %v", p, err)
		}
	}
}

// Positions that restart change what the model computes and not what it
// costs: the same rotation, turned by a different number.
func TestRestartingPositionsCostNothing(t *testing.T) {
	doc := presets.MustGet("llama-3-8b")
	doc.Graph.Nodes = append(doc.Graph.Nodes, ir.NodeDef{ID: "pos", Type: "input",
		Params: map[string]any{"shape": "B T", "dtype": "int64", "role": "positions"}})
	doc.Graph.Edges = append(doc.Graph.Edges, ir.Edge{"pos:x", "layers:pos"})
	for i := range doc.Graph.Nodes {
		if stack := &doc.Graph.Nodes[i]; stack.ID == "layers" {
			for j := range stack.Graph.Nodes {
				switch n := &stack.Graph.Nodes[j]; n.ID {
				case "_in":
					n.Params["ports"].(map[string]any)["pos"] = "B T"
				case "block":
					n.Params["positions"] = true
				}
			}
			stack.Graph.Edges = append(stack.Graph.Edges, ir.Edge{"_in:pos", "block:pos"})
		}
	}
	plain := analyzeAt(t, presets.MustGet("llama-3-8b"), 8192, nil)
	restarted := analyzeAt(t, doc, 8192, nil)
	if restarted.Params.Total != plain.Params.Total || restarted.Flops.FwdTotal != plain.Flops.FwdTotal ||
		restarted.Flops.Elementwise != plain.Flops.Elementwise || restarted.Kv.BytesPerToken != plain.Kv.BytesPerToken {
		t.Errorf("restarting the positions moved the numbers")
	}
}
