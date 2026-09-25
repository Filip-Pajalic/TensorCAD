package analysis_test

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
)

// Two sequences, measured each at its own length.
//
// An encoder over a source S tokens long and a decoder over a target T long,
// with no cross-attention between them yet: two stacks that share nothing but
// a batch. What M11's first phase claims is that each is counted at the length
// of the sequence it runs along — so moving S moves the encoder's figures and
// nothing of the decoder's — and that every per-token figure is per target
// token, the encoder's share spread over the target's tokens.

const twoStreams = `{
  "version": 1,
  "meta": {"name": "two-streams"},
  "symbols": {
    "B": {"kind": "runtime", "default": 1},
    "T": {"kind": "runtime", "default": 64},
    "S": {"kind": "runtime", "default": 256},
    "D": {"kind": "design", "value": 64},
    "H": {"kind": "design", "value": 4},
    "dh": {"kind": "design", "value": 16},
    "V": {"kind": "design", "value": 100}
  },
  "graph": {
    "nodes": [
      {"id": "src", "type": "input", "params": {"shape": "B S", "dtype": "int64"}},
      {"id": "src_embed", "type": "embedding", "params": {"vocab": "V", "dim": "D"}},
      {"id": "encoder", "type": "transformer_block", "params": {"d_model": "D", "heads": "H", "kv_heads": "H",
        "head_dim": "dh", "ffn_hidden": "4*D", "mlp": "dense", "act": "gelu", "causal": false}},
      {"id": "encoded", "type": "output"},
      {"id": "tgt", "type": "input", "params": {"shape": "B T", "dtype": "int64"}},
      {"id": "tgt_embed", "type": "embedding", "params": {"vocab": "V", "dim": "D"}},
      {"id": "decoder", "type": "transformer_block", "params": {"d_model": "D", "heads": "H", "kv_heads": "H",
        "head_dim": "dh", "ffn_hidden": "4*D", "mlp": "dense", "act": "gelu", "causal": true}},
      {"id": "head", "type": "lm_head", "params": {"vocab": "V", "dim": "D", "tied": false}},
      {"id": "logits", "type": "output"}
    ],
    "edges": [
      ["src:x", "src_embed:ids"], ["src_embed:y", "encoder:x"], ["encoder:y", "encoded:x"],
      ["tgt:x", "tgt_embed:ids"], ["tgt_embed:y", "decoder:x"], ["decoder:y", "head:x"],
      ["head:y", "logits:x"]
    ]
  }
}`

func twoStreamDoc(t *testing.T) *ir.Doc {
	t.Helper()
	var doc ir.Doc
	if err := json.Unmarshal([]byte(twoStreams), &doc); err != nil {
		t.Fatal(err)
	}
	return &doc
}

func at(t *testing.T, doc *ir.Doc, T, S float64) *analysis.Result {
	t.Helper()
	res, err := analysis.Analyze(doc, analysis.Options{T: &T, S: &S}, analysis.Inputs{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) > 0 {
		t.Fatalf("errors: %v", res.Errors)
	}
	return res
}

func sumUnder(m map[string]float64, prefix string) float64 {
	total := 0.0
	for path, v := range m {
		if strings.HasPrefix(path, prefix) {
			total += v
		}
	}
	return total
}

// Moving the source moves the encoder, and only the encoder.
func TestEachStackIsMeasuredAtItsOwnLength(t *testing.T) {
	doc := twoStreamDoc(t)
	short, long := at(t, doc, 64, 256), at(t, doc, 64, 512)

	// Per token of its own stream, the encoder's dense work does not depend on
	// how long its stream is, and its attention reads every one of S keys.
	enc := func(r *analysis.Result) analysis.StreamFlops { return r.Flops.PerStream[0] }
	dec := func(r *analysis.Result) analysis.StreamFlops { return r.Flops.PerStream[1] }
	if enc(short).Symbol != "S" || dec(short).Symbol != "T" {
		t.Fatalf("streams %+v", short.Flops.PerStream)
	}
	const H, dh = 4.0, 16.0
	if got := enc(long).Fwd - enc(short).Fwd; got != 4*(512-256)*H*dh {
		t.Errorf("doubling S adds %v to the encoder's per-token FLOPs, want its attention's %v", got, 4*(512-256)*H*dh)
	}
	if dec(long).Fwd != dec(short).Fwd {
		t.Errorf("moving S moved the decoder: %v then %v", dec(short).Fwd, dec(long).Fwd)
	}

	// Per target token, the encoder's work is spread over the target's tokens.
	want := dec(short).Fwd + enc(short).Fwd*256/64
	if math.Abs(short.Flops.FwdTotal-want) > 1e-6*want {
		t.Errorf("per target token %v, want %v", short.Flops.FwdTotal, want)
	}
	if got := short.Flops.FwdPerExample; got != enc(short).Fwd*256+dec(short).Fwd*64 {
		t.Errorf("per example %v", got)
	}

	// The encoder's activations are its stream's tokens' worth, so doubling S
	// doubles them; the decoder's do not move.
	encShort := sumUnder(short.Memory.Train.ActivationsByPath, "encoder/") + short.Memory.Train.ActivationsByPath["src_embed"]
	encLong := sumUnder(long.Memory.Train.ActivationsByPath, "encoder/") + long.Memory.Train.ActivationsByPath["src_embed"]
	if encShort == 0 || encLong != 2*encShort {
		t.Errorf("encoder activations %v at S=256 and %v at S=512, want double", encShort, encLong)
	}
	if a, b := sumUnder(short.Memory.Train.ActivationsByPath, "decoder/"), sumUnder(long.Memory.Train.ActivationsByPath, "decoder/"); a != b || a == 0 {
		t.Errorf("decoder activations %v then %v", a, b)
	}
	if short.Options.S != 256 {
		t.Errorf("the operating point says S = %v", short.Options.S)
	}
}

// A source keeps no cache. It is read once, before anything is generated,
// and an encoder's own keys and values are not needed after it; what
// generation keeps of the source is cross-attention's, which is the decoder's.
func TestASourceKeepsNoCache(t *testing.T) {
	res := at(t, twoStreamDoc(t), 64, 256)
	const H, dh, bytes = 4.0, 16.0, 2.0
	if want := H * 2 * dh * bytes; res.Kv.BytesPerToken != want {
		t.Errorf("per target token %v, want the decoder's %v", res.Kv.BytesPerToken, want)
	}
	if res.Kv.BytesPerSequenceFixed != 0 {
		t.Errorf("the encoder holds %v bytes a request", res.Kv.BytesPerSequenceFixed)
	}
}

// A design with one sequence reports no streams and no S: nothing about it has
// changed, and the goldens say so for all of them.
func TestOneStreamSaysNothingAboutStreams(t *testing.T) {
	doc := twoStreamDoc(t)
	delete(doc.Symbols, "S")
	for i, n := range doc.Graph.Nodes {
		if n.ID == "src" {
			doc.Graph.Nodes[i].Params = map[string]any{"shape": "B T", "dtype": "int64"}
		}
	}
	T := 64.0
	res, err := analysis.Analyze(doc, analysis.Options{T: &T}, analysis.Inputs{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Flops.PerStream != nil || res.Flops.FwdPerExample != 0 || res.Options.S != 0 {
		t.Errorf("a one-stream design reports %+v, %v, S=%v", res.Flops.PerStream, res.Flops.FwdPerExample, res.Options.S)
	}
}

// A block that receives the source and the target together is where two
// sequences meet, and one that takes only one of them is told so: S against T,
// a real difference rather than two numbers that happened to match. A whole
// stack fed the source is not an error — it runs along the source — which is
// what lets every existing block be an encoder's without being written again.
func TestTheTwoLengthsDoNotMix(t *testing.T) {
	doc := twoStreamDoc(t)
	doc.Graph.Nodes = append(doc.Graph.Nodes,
		ir.NodeDef{ID: "mix", Type: "add", Params: map[string]any{"dim": "D"}},
		ir.NodeDef{ID: "mixed", Type: "output"})
	doc.Graph.Edges = append(doc.Graph.Edges,
		ir.Edge{"tgt_embed:y", "mix:a"}, ir.Edge{"src_embed:y", "mix:b"}, ir.Edge{"mix:y", "mixed:x"})
	res := infer.Shapes(doc, ir.ResolveSymbols(doc), infer.Options{})
	found := false
	for _, issue := range res.Issues {
		if issue.Path == "mix" && issue.Severity == "error" {
			found = true
		}
	}
	if !found {
		t.Errorf("the source and the target added together went unremarked: %+v", res.Issues)
	}
	// And the encoder, fed the source, has nothing to say.
	clean := infer.Shapes(twoStreamDoc(t), ir.ResolveSymbols(doc), infer.Options{ExpandComposites: true})
	for _, issue := range clean.Issues {
		t.Errorf("%s: %s", issue.Path, issue.Message)
	}
}

// S is a length, so it cannot be a number the design fixes.
func TestSIsOnlyARuntimeSymbol(t *testing.T) {
	doc := twoStreamDoc(t)
	doc.Symbols["S"] = ir.SymbolDef{Kind: "design", Number: 256, HasNumber: true}
	table := ir.ResolveSymbols(doc)
	if len(table.Errors) == 0 || !strings.Contains(table.Errors[0], "runtime symbol") {
		t.Errorf("errors %v", table.Errors)
	}
}
