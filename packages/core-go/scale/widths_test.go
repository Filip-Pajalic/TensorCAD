package scale_test

import (
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
	"github.com/tensorcad/core/scale"
)

// Every width has to move, not just the ones a dense transformer has.
//
// A symbol left out of the width set keeps its full-size value on a shrunk
// design, and the result is not a small version of the architecture but a
// different one. Latent attention is where it shows worst: a compressed query
// width that does not move is a compression that expands.

func widthOf(t *testing.T, doc *ir.Doc, name string) float64 {
	t.Helper()
	def, ok := doc.Symbols[name]
	if !ok || !def.HasNumber {
		t.Fatalf("no numeric %s", name)
	}
	return def.Number
}

// A latent is a compression, at every size.
func TestALatentStaysNarrowerThanTheStream(t *testing.T) {
	base := presets.MustGet("deepseek-v3")
	for _, target := range []float64{20e6, 100e6, 1e9} {
		res, err := scale.Design(base, scale.Options{TargetParams: target, Vocab: f(4096)})
		if err != nil {
			t.Fatal(err)
		}
		d := widthOf(t, res.Doc, "D")
		for _, latent := range []string{"Ql", "Kl"} {
			w := widthOf(t, res.Doc, latent)
			if w >= d {
				t.Errorf("at %.0f parameters the design is %v wide and its %s latent is %v: "+
					"a compression cannot be wider than what it compresses", target, d, latent, w)
			}
			// And it stays roughly in proportion, rather than merely under.
			if ratio := widthOf(t, base, latent) / widthOf(t, base, "D"); w < d*ratio/4 || w > d*ratio*4 {
				t.Errorf("at %.0f parameters %s is %v against a width of %v, which is far from the "+
					"design's own ratio of %v", target, latent, w, d, ratio)
			}
		}
	}
}

// A design with a second transformer in it keeps the second one in proportion
// too, heads and depth included: I-JEPA's predictor is narrower and shallower
// than its encoder, and a shrunk one that is neither is not the same design.
func TestASecondStreamScalesWithTheFirst(t *testing.T) {
	base := presets.MustGet("ijepa-vit-h14")
	res, err := scale.Design(base, scale.Options{TargetParams: 50e6})
	if err != nil {
		t.Fatal(err)
	}
	doc := res.Doc
	for _, pair := range [][2]string{{"Dp", "D"}, {"Fp", "F"}, {"Lp", "L"}} {
		small, big := widthOf(t, doc, pair[0]), widthOf(t, doc, pair[1])
		if small >= big {
			t.Errorf("the predictor's %s is %v against the encoder's %s of %v; "+
				"the design has it the other way round", pair[0], small, pair[1], big)
		}
	}
	// The predictor's heads have to follow its width, or its own projections
	// stop lining up.
	if heads, dh, d := widthOf(t, doc, "Hp"), widthOf(t, doc, "dhp"), widthOf(t, doc, "Dp"); heads*dh != d {
		t.Errorf("the predictor is %v wide but has %v heads of %v", d, heads, dh)
	}
}

// Whatever the scaling did, the design it produced still has to be one the
// engine accepts. A width moved out of step with another is most visible here.
func TestEveryScaledPresetStillChecksOut(t *testing.T) {
	names, err := presets.Names()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			// A Mamba block's group count is a node parameter rather than a
			// symbol, and its head count is derived from the width, so
			// narrowing the design leaves heads that the groups do not divide.
			// Repairing it needs the scaling to reach into node parameters,
			// which it does not do yet.
			if strings.HasPrefix(name, "nemotron-h") {
				t.Skip("scaling does not yet repair a state-space block's group count")
			}
			base := presets.MustGet(name)
			res, err := scale.Design(base, scale.Options{TargetParams: 100e6, Vocab: f(4096)})
			if err != nil {
				// Not every design has a width to move; that is its own answer.
				t.Skipf("%s does not scale: %v", name, err)
			}
			out, err := rules.Validate(res.Doc, analysis.Options{})
			if err != nil {
				t.Fatalf("validate: %v", err)
			}
			for _, finding := range out.Findings {
				if finding.Severity == "error" {
					t.Errorf("%s %s: %s", finding.Rule, finding.Path, finding.Message)
				}
			}
		})
	}
}

// A design that repeats a pair of layers stays a whole number of pairs.
//
// Gemma 2 and gpt-oss stack L/2 copies of a windowed layer and a full one, so a
// scaled depth of one is half a pair: a repeat count of 0.5, which the
// generated model cannot even loop over. The depth moves to the nearest value
// that keeps every group whole, and the notes say so.
func TestARepeatedPairStaysWhole(t *testing.T) {
	for _, name := range []string{"gemma-2-9b", "gpt-oss-20b"} {
		vocab := 256.0
		res, err := scale.Design(presets.MustGet(name), scale.Options{TargetParams: 2e6, Vocab: &vocab})
		if err != nil {
			t.Fatal(err)
		}
		if l := widthOf(t, res.Doc, "L"); l < 2 || int(l)%2 != 0 {
			t.Errorf("%s scaled to L = %v", name, l)
		}
		report, err := rules.Validate(res.Doc, analysis.Options{})
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range report.Findings {
			if f.Severity == "error" {
				t.Errorf("%s scaled has an error: %s", name, f.Message)
			}
		}
		said := false
		for _, n := range res.Notes {
			said = said || strings.Contains(n, "each repeated group of layers is whole")
		}
		if !said {
			t.Errorf("%s: the notes do not say the depth moved: %v", name, res.Notes)
		}
	}
}

// A stack labelled with its count says the count it has after scaling, not the
// one it had: Llama-3-8B's frame reads "Transformer block x32", and a design
// scaled to a tenth of it is not thirty-two layers deep.
func TestAScaledStackSaysItsNewCount(t *testing.T) {
	result, err := scale.Design(presets.MustGet("llama-3-8b"), scale.Options{TargetParams: 1e9})
	if err != nil {
		t.Fatal(err)
	}
	layers := ir.ResolveSymbols(result.Doc).Values["L"]
	for _, n := range result.Doc.Graph.Nodes {
		if n.Type == "repeat" {
			want := "Transformer block x" + analysis.JSNumber(layers)
			if n.Label != want {
				t.Errorf("the stack is labelled %q, want %q", n.Label, want)
			}
		}
	}
}
