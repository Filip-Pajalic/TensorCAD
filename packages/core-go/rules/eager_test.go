package rules_test

import (
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
)

// Attention written out, against what it has to cost.
//
// GPT-2 small at 1,024 tokens: twelve heads, twelve layers, a score matrix of
// 1,024 by 1,024 per head, kept at two bytes an element.

func writtenOut(t *testing.T, params map[string]any) *ir.Doc {
	t.Helper()
	doc := presets.MustGet("gpt2-small")
	for i := range doc.Graph.Nodes {
		if doc.Graph.Nodes[i].ID != "layers" {
			continue
		}
		for j := range doc.Graph.Nodes[i].Graph.Nodes {
			n := &doc.Graph.Nodes[i].Graph.Nodes[j]
			if n.ID == "block" {
				for k, v := range params {
					n.Params[k] = v
				}
			}
		}
	}
	return doc
}

func validateAt(t *testing.T, doc *ir.Doc, T float64) *rules.Report {
	t.Helper()
	rep, err := rules.Validate(doc, analysis.Options{T: &T})
	if err != nil {
		t.Fatal(err)
	}
	return rep
}

const (
	heads, layers, headDim = 12.0, 12.0, 64.0
	seq                    = 1024.0
	scoreMatrix            = heads * seq * seq * 2 * layers // one per layer, bf16
)

// Written out, every score is computed, masked or not, so the attention costs
// what a profiler counts; and the weights the softmax makes are kept, one
// matrix a layer, which the rule states in bytes.
func TestAttentionWrittenOutKeepsItsWeights(t *testing.T) {
	rep := validateAt(t, writtenOut(t, map[string]any{"written_out": true}), seq)
	a := rep.Analysis
	if want := 4 * seq * heads * headDim * layers; a.Flops.FwdAttention != want || a.Flops.FwdAttentionUnmasked != want {
		t.Errorf("attention is %v FLOPs a token (%v unmasked), want %v for both",
			a.Flops.FwdAttention, a.Flops.FwdAttentionUnmasked, want)
	}
	// The block's own tensors are the weights, exactly; its queries, keys
	// and values are made outside it. It keeps no output of its own, where a
	// fused kernel keeps its output and log-sum-exp — which is why the whole
	// model grows by a little less than this.
	own := 0.0
	for path, bytes := range a.Memory.Train.ActivationsByPath {
		if rest, in := strings.CutPrefix(path, "layers/block/attn/attn/"); in && !strings.HasPrefix(rest, "_") {
			own += bytes
		}
	}
	if own != scoreMatrix {
		t.Errorf("the written-out attention keeps %s, want one score matrix a layer, %s",
			analysis.FormatBytes(own), analysis.FormatBytes(scoreMatrix))
	}
	found := false
	for _, f := range rep.Findings {
		if f.Rule == "eager-attention" {
			found = true
			if !strings.Contains(f.Message, analysis.FormatBytes(scoreMatrix)) || f.Severity != "warning" {
				t.Errorf("%s: %s", f.Severity, f.Message)
			}
		}
	}
	if !found {
		t.Error("nothing says what writing the attention out costs")
	}
}

// Talking heads mixes before the softmax and after it: two heads by heads
// matrices a layer, a matmul across heads for every score, and three score
// matrices kept instead of one — the scores, and the weights either side of
// the second mix.
func TestTalkingHeadsCostsWhatItMixes(t *testing.T) {
	plain := validateAt(t, writtenOut(t, map[string]any{"written_out": true}), seq).Analysis
	rep := validateAt(t, writtenOut(t, map[string]any{"talking_heads": true}), seq)
	a := rep.Analysis
	if got := a.Params.Total - plain.Params.Total; got != 2*heads*heads*layers {
		t.Errorf("talking heads adds %v parameters, want %v", got, 2*heads*heads*layers)
	}
	if got := a.Flops.FwdAttention - plain.Flops.FwdAttention; got != 2*2*heads*heads*seq*layers {
		t.Errorf("talking heads adds %v FLOPs a token, want %v", got, 2*2*heads*heads*seq*layers)
	}
	for _, f := range rep.Findings {
		if f.Rule == "eager-attention" && !strings.Contains(f.Message, analysis.FormatBytes(3*scoreMatrix)) {
			t.Errorf("says %q, want three score matrices, %s", f.Message, analysis.FormatBytes(3*scoreMatrix))
		}
	}
}

// Writing an attention out keeps nothing but causal attention, so asking it to
// keep a window as well is refused rather than silently dropped.
func TestWrittenOutRefusesWhatOnlyTheKernelDoes(t *testing.T) {
	rep := validateAt(t, writtenOut(t, map[string]any{"written_out": true, "window": 256.0}), seq)
	for _, f := range rep.Findings {
		if f.Rule == "ATTN-02" {
			if f.Severity != "error" || !strings.Contains(f.Message, "window") {
				t.Errorf("%s: %s", f.Severity, f.Message)
			}
			return
		}
	}
	t.Error("a window on a written-out attention is dropped without a word")
}
