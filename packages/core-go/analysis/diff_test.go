package analysis_test

import (
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
)

// Differential attention, against the paper's arithmetic.
//
// There is no released DIFF Transformer to hold a preset to, so the arithmetic
// is pinned here instead: the parameters of Microsoft's reference module, the
// FLOPs of the attention it draws, and the cache it keeps.

// diffDesign is GPT-2 small's stack with its attention made differential at
// the paper's proportions: half the heads, each a pair of half-width maps, so
// the projections are exactly as wide as the baseline's.
func diffDesign(t *testing.T) *ir.Doc {
	t.Helper()
	doc := presets.MustGet("gpt2-small")
	doc.Symbols["H"] = ir.SymbolDef{Kind: "literal", Number: 6, HasNumber: true}
	doc.Symbols["Hkv"] = ir.SymbolDef{Kind: "literal", Number: 6, HasNumber: true}
	doc.Symbols["dh"] = ir.SymbolDef{Kind: "literal", Number: 64, HasNumber: true}
	for i := range doc.Graph.Nodes {
		if doc.Graph.Nodes[i].ID != "layers" {
			continue
		}
		for j := range doc.Graph.Nodes[i].Graph.Nodes {
			n := &doc.Graph.Nodes[i].Graph.Nodes[j]
			if n.ID == "block" {
				n.Params["attention"] = "diff"
				n.Params["attn_bias"] = false
			}
		}
	}
	return doc
}

func analyzeDoc(t *testing.T, doc *ir.Doc, opts analysis.Options) *analysis.Result {
	t.Helper()
	res, err := analysis.Analyze(doc, opts, analysis.Inputs{})
	if err != nil {
		t.Fatal(err)
	}
	return res
}

// The reference module has four D×D projections, lambda's four head_dim
// vectors and a norm over a value head, and nothing else.
func TestDifferentialAttentionHasTheReferenceParameters(t *testing.T) {
	res := analyzeDoc(t, diffDesign(t), analysis.Options{})
	const D, d, L = 768.0, 64.0, 12.0
	want := 4*D*D + 4*d + 2*d
	var got float64
	for path, n := range res.Params.ByPath {
		if len(path) > len("layers/block/attn/") && path[:len("layers/block/attn/")] == "layers/block/attn/" {
			got += n
		}
	}
	if got != want*L {
		t.Errorf("attention holds %v parameters over %v layers, the reference %v", got, L, want*L)
	}
}

// Two maps per head, each scoring at head_dim and summing values twice as
// wide: 2·(d + 2d) per key per map. The same design with grouped-query
// attention of the full head count scores 12 heads at 64 against values of 64,
// which is two thirds of that.
func TestDifferentialAttentionCostsTwoMaps(t *testing.T) {
	opts := analysis.Options{T: f(2048), B: f(1)}
	diff := analyzeDoc(t, diffDesign(t), opts)
	base := analyzeDoc(t, presets.MustGet("gpt2-small"), opts)
	const H, d, T, L = 6.0, 64.0, 2048.0, 12.0
	want := 2 * (2 * H * (d + 2*d)) * (T / 2) * L
	if diff.Flops.FwdAttention != want {
		t.Errorf("attention is %v FLOPs a token, want %v", diff.Flops.FwdAttention, want)
	}
	if ratio := diff.Flops.FwdAttention / base.Flops.FwdAttention; ratio != 1.5 {
		t.Errorf("differential attention costs %vx the baseline's, want 1.5", ratio)
	}
}

// The cache holds both maps' keys and the values once: the second map reads
// the first one's values and caches only its keys.
func TestDifferentialAttentionCachesTheValuesOnce(t *testing.T) {
	res := analyzeDoc(t, diffDesign(t), analysis.Options{})
	const Hkv, d, L, bytes = 6.0, 64.0, 12.0, 2.0
	want := (Hkv*d + Hkv*d + Hkv*2*d) * L * bytes
	if res.Kv.BytesPerToken != want {
		t.Errorf("the cache is %v bytes a token, want %v", res.Kv.BytesPerToken, want)
	}
}

// A narrower value head is counted at its own width. Latent attention's
// values are 128 wide against keys of 192, and the value product was counted
// at 192, a sixth too much of DeepSeek-V3's attention.
func TestTheValueProductIsCountedAtTheValueWidth(t *testing.T) {
	res := analyzeDoc(t, presets.MustGet("deepseek-v3"), analysis.Options{T: f(8192), B: f(1)})
	const H, qk, v, T, L = 128.0, 192.0, 128.0, 8192.0, 61.0
	want := 2 * H * (qk + v) * (T / 2) * L
	if res.Flops.FwdAttention != want {
		t.Errorf("DeepSeek-V3 attention is %v FLOPs a token, want %v", res.Flops.FwdAttention, want)
	}
}
