package catalog_test

import (
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
)

// Mamba-1, against Jamba-v0.1's own weights.
//
// Not a variant of the Mamba-2 block: the timestep is read at a rank and
// projected back up, and that matrix in the middle is what the other one does
// not have. Every shape below was read from the checkpoint's safetensors
// headers rather than from a description of them.
func TestMambaBlockMatchesJamba(t *testing.T) {
	const (
		d      = 4096.0
		inner  = 8192.0 // expand 2
		state  = 16.0
		rank   = 256.0
		kernel = 4.0
	)
	doc := &ir.Doc{
		Version: ir.DocVersion,
		Meta:    ir.DocMeta{Name: "mamba-probe"},
		Symbols: map[string]ir.SymbolDef{
			"B": {Kind: "runtime", Number: 1, HasNumber: true},
			"T": {Kind: "runtime", Number: 4096, HasNumber: true},
			"D": {Kind: "design", Number: d, HasNumber: true},
		},
		SymbolOrder: []string{"B", "T", "D"},
		Graph: ir.Graph{
			Nodes: []ir.NodeDef{
				{ID: "tokens", Type: "input", Params: map[string]any{"shape": "B T D"}},
				{ID: "blk", Type: "mamba_block", Params: map[string]any{
					"d_model": "D", "expand": 2.0, "state": state, "dt_rank": rank,
					"conv_kernel": kernel, "conv_bias": true, "bias": false,
					"norm_inputs": true,
				}},
				{ID: "out", Type: "output"},
			},
			Edges: []ir.Edge{{"tokens:x", "blk:x"}, {"blk:y", "out:x"}},
		},
	}

	res, err := analysis.Analyze(doc, analysis.Options{}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	for _, e := range res.Params.Errors {
		t.Errorf("error: %s", e)
	}

	want := map[string]float64{
		"blk/in_proj":  2 * inner * d,            // in_proj [16384, 4096]
		"blk/conv":     inner*kernel + inner,     // conv1d [8192, 1, 4] with a bias
		"blk/x_proj":   (rank + 2*state) * inner, // x_proj [288, 8192]
		"blk/dt_proj":  rank*inner + inner,       // dt_proj [8192, 256] with a bias
		"blk/scan":     inner*state + inner,      // A_log [8192, 16] and D [8192]
		"blk/dt_norm":  rank,                     // dt_layernorm [256]
		"blk/b_norm":   state,                    // b_layernorm [16]
		"blk/c_norm":   state,                    // c_layernorm [16]
		"blk/out_proj": inner * d,                // out_proj [4096, 8192]
	}
	total := 0.0
	for path, n := range want {
		total += n
		if got := res.Params.ByPath[path]; got != n {
			t.Errorf("%s is %v, want %v", path, got, n)
		}
	}
	if got := res.Params.Total; got != total {
		t.Errorf("the block is %v parameters, want %v: something is counted that should not be",
			got, total)
	}

	// The state is fixed per sequence, which is the reason to use one at all.
	if got, want := res.Kv.BytesPerToken, 0.0; got != want {
		t.Errorf("it caches %v bytes per token; a state-space layer caches none", got)
	}
	if got := res.Kv.BytesPerSequenceFixed; got <= 0 {
		t.Error("it holds no state per sequence, so nothing carries the recurrence")
	}
}
