package mup_test

import (
	"math"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/mup"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
	"github.com/tensorcad/core/scale"
)

// Pointers, because an unset option and one set to zero are different things.
// Named for what they hold rather than f and b, because f is already a finding
// in a loop below and one letter meaning two things in one file reads badly.
func num(v float64) *float64 { return &v }
func flag(v bool) *bool      { return &v }

var laddered = []string{"gpt2-small", "llama-3-8b", "mixtral-8x7b", "ijepa-vit-h14"}

func build(t *testing.T, name string) *mup.Ladder {
	t.Helper()
	ladder, err := mup.Build(presets.MustGet(name), mup.Options{})
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return ladder
}

// The premise: the heads get more numerous, not wider.
//
// A ladder that widened the head instead would be comparing designs whose
// attention has a different shape at every rung, and whatever transferred
// between them would not be evidence about the original.
func TestTheHeadDimensionNeverMoves(t *testing.T) {
	for _, name := range laddered {
		t.Run(name, func(t *testing.T) {
			base := presets.MustGet(name)
			want := base.Symbols["dh"].Number
			ladder := build(t, name)
			if ladder.HeadDim != want {
				t.Errorf("the ladder reports a head dimension of %v, want %v", ladder.HeadDim, want)
			}
			for _, r := range ladder.Rungs {
				if got := r.Doc.Symbols["dh"].Number; got != want {
					t.Errorf("at width %v the head dimension is %v, want %v", r.Width, got, want)
				}
				if r.Heads*want != r.Width {
					t.Errorf("width %v is not %v heads of %v", r.Width, r.Heads, want)
				}
			}
		})
	}
}

// Every weight the design holds is in exactly one row of the table.
//
// A weight in none is one nobody was told how to initialize; a weight in two
// has contradictory instructions. Both are worse than a wrong row, because a
// wrong row is at least visible.
func TestEveryWeightIsClassifiedOnce(t *testing.T) {
	for _, name := range laddered {
		t.Run(name, func(t *testing.T) {
			for _, r := range build(t, name).Rungs {
				want := map[string]bool{}
				flat := analysis.Flatten(r.Doc, ir.ResolveSymbols(r.Doc))
				for i := range flat.Nodes {
					node := &flat.Nodes[i]
					if node.Def.ParamCount == nil {
						continue
					}
					// A tied readout holds no weights of its own and is still
					// where the output rule applies.
					tied := node.Type == "lm_head" && node.Resolved.Bool("tied")
					if node.Def.ParamCount(node.Resolved) > 0 || tied {
						want[node.Path] = true
					}
				}
				seen := map[string]int{}
				for _, s := range r.Scaling {
					for _, p := range s.Paths {
						seen[p]++
					}
				}
				for p := range want {
					switch seen[p] {
					case 1:
					case 0:
						t.Errorf("at width %v, %s holds weights and is in no class", r.Width, p)
					default:
						t.Errorf("at width %v, %s is in %d classes at once", r.Width, p, seen[p])
					}
				}
				for p := range seen {
					if !want[p] {
						t.Errorf("at width %v, %s is classified but holds no weights", r.Width, p)
					}
				}
			}
		})
	}
}

// What the multipliers are for, computed rather than restated.
//
// Initialize a weight at a standard deviation of 1/sqrt(fan_in) times the
// ladder's multiplier, and a hidden layer's output has the same scale at every
// width — which is the property that lets a learning rate transfer. The readout
// is the deliberate exception: its scale falls as 1/m, which is what keeps the
// logits from growing with the width.
//
// The rounding is why this is a tolerance rather than an equality. A
// feed-forward width lands on a multiple of 64, so it is not exactly m times
// the base's, and the invariant inherits that error.
func TestAHiddenLayerKeepsItsScaleAcrossTheLadder(t *testing.T) {
	for _, name := range laddered {
		t.Run(name, func(t *testing.T) {
			ladder := build(t, name)
			base := map[string]float64{}
			for _, r := range ladder.Rungs {
				if r.Base {
					base = fanIns(r.Doc)
				}
			}
			if len(base) == 0 {
				t.Fatal("no base rung")
			}

			for _, r := range ladder.Rungs {
				fans := fanIns(r.Doc)
				for _, s := range r.Scaling {
					want, ok := map[mup.Class]float64{Hidden: 1, Output: 1 / r.Multiplier}[s.Class]
					if !ok {
						continue
					}
					for _, path := range s.Paths {
						in, here := fans[path]
						from, there := base[path]
						if !here || !there || in == 0 || from == 0 {
							continue
						}
						// sigma is what the initializer is handed: the base
						// model's 1/sqrt(fan_in), times this rung's multiplier.
						sigma := s.InitStd / math.Sqrt(from)
						got := in * sigma * sigma
						if math.Abs(got-want) > 0.15*want {
							t.Errorf("at width %v, %s puts out %.3f times what it did at the base, "+
								"where a %s weight should put out %.3f",
								r.Width, path, got, s.Class, want)
						}
					}
				}
			}
		})
	}
}

// fanIns is what each weight-holding block reads, by path.
func fanIns(doc *ir.Doc) map[string]float64 {
	out := map[string]float64{}
	flat := analysis.Flatten(doc, ir.ResolveSymbols(doc))
	for i := range flat.Nodes {
		node := &flat.Nodes[i]
		switch node.Type {
		case "linear":
			out[node.Path] = node.Resolved.Num("in_features")
		case "lm_head":
			out[node.Path] = node.Resolved.Num("dim")
		case "topk_router":
			out[node.Path] = node.Resolved.Num("d_model")
		}
	}
	return out
}

// At the rung the sweep happens on, nothing is multiplied by anything.
func TestTheBaseRungIsTheBaseModel(t *testing.T) {
	for _, name := range laddered {
		t.Run(name, func(t *testing.T) {
			bases := 0
			for _, r := range build(t, name).Rungs {
				if !r.Base {
					continue
				}
				bases++
				if r.Multiplier != 1 {
					t.Errorf("the base rung has a multiplier of %v", r.Multiplier)
				}
				for _, s := range r.Scaling {
					if s.InitStd != 1 || s.AdamLR != 1 {
						t.Errorf("at the base, %s weights are multiplied by %v and %v",
							s.Class, s.InitStd, s.AdamLR)
					}
				}
			}
			if bases != 1 {
				t.Errorf("%d rungs are the base, want exactly one", bases)
			}
		})
	}
}

// A rung is a design, so it has to be one the engine accepts. The heads and the
// widths move together, and a rung that did not check out would mean they had
// come apart.
func TestEveryRungIsADesignThatChecksOut(t *testing.T) {
	for _, name := range laddered {
		t.Run(name, func(t *testing.T) {
			for _, r := range build(t, name).Rungs {
				out, err := rules.Validate(r.Doc, analysis.Options{})
				if err != nil {
					t.Fatalf("at width %v: %v", r.Width, err)
				}
				for _, f := range out.Findings {
					if f.Severity == "error" {
						t.Errorf("at width %v, %s %s: %s", r.Width, f.Rule, f.Path, f.Message)
					}
				}
			}
		})
	}
}

// A ladder of one rung transfers nothing: it is one model, with every
// multiplier at 1 and nothing to carry an answer to.
//
// A design already shrunk to a bench budget has nowhere below it to go — the
// rungs stop at four heads, because a two-head model proxies badly for a
// thirty-two-head one — so the ladder goes up instead. That is the same
// question the other way round: this is what you swept at, here is what it
// carries to.
func TestALadderNeverHasOneRung(t *testing.T) {
	bench, err := scale.Design(presets.MustGet("llama-3-8b"), scale.Options{
		TargetParams: 30e6, TargetBasis: "non-embedding", Vocab: num(50304), TieHead: flag(true),
	})
	if err != nil {
		t.Fatal(err)
	}
	ladder, err := mup.Build(bench.Doc, mup.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(ladder.Rungs) < 3 {
		t.Fatalf("a %v-wide design gets %d rung(s); a ladder of one carries nothing",
			ladder.BaseWidth, len(ladder.Rungs))
	}
	// It is the design's own width that is swept at, and the rest are above it.
	if !ladder.Rungs[0].Base {
		t.Error("the narrowest rung is not the base")
	}
	for i := 1; i < len(ladder.Rungs); i++ {
		if ladder.Rungs[i].Width <= ladder.Rungs[i-1].Width {
			t.Errorf("rung %d is %v wide, no wider than the %v before it",
				i, ladder.Rungs[i].Width, ladder.Rungs[i-1].Width)
		}
	}
	// And every preset gets a ladder worth the name.
	names, err := presets.Names()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range names {
		l, err := mup.Build(presets.MustGet(name), mup.Options{})
		if err != nil {
			continue // Not every design has a width to move; that is its own answer.
		}
		if len(l.Rungs) < 2 {
			t.Errorf("%s gets %d rung(s)", name, len(l.Rungs))
		}
	}
}

// A design the ladder cannot be built over gets told why, in terms of what it
// is missing rather than of what failed.
func TestADesignWithNoWidthIsRefused(t *testing.T) {
	for _, name := range []string{"alexnet"} {
		if _, err := mup.Build(presets.MustGet(name), mup.Options{}); err == nil {
			t.Errorf("%s has no residual width and a ladder was built over it anyway", name)
		} else if got := err.Error(); !contains(got, "D") {
			t.Errorf("%s was refused with %q, which does not say what is missing", name, got)
		}
	}
	doc := presets.MustGet("gpt2-small")
	if _, err := mup.Build(doc, mup.Options{Widths: []float64{-1}}); err == nil {
		t.Error("a negative width was accepted")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// Aliases so the table above reads as the paper's rows do.
const (
	Hidden = mup.Hidden
	Output = mup.Output
)
