package presets_test

import (
	"math"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/presets"
)

// The presets are the regression suite, and this is the claim they exist to
// make: the analysis reproduces the parameter count the model's authors
// published. Fourteen of them match to the parameter; the rest are checked
// against a rounded vendor figure and say so with an explicit tolerance.

func TestEveryPresetReproducesItsPublishedCount(t *testing.T) {
	names, err := presets.Names()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) < 20 {
		t.Fatalf("the library has %d presets; it had 20", len(names))
	}

	checked := 0
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			doc, err := presets.Get(name)
			if err != nil {
				t.Fatal(err)
			}
			result, err := analysis.Analyze(doc, analysis.Options{}, analysis.Inputs{})
			if err != nil {
				t.Fatal(err)
			}
			for _, e := range result.Errors {
				t.Errorf("analysis error: %s", e)
			}

			pub := doc.Meta.Published
			if pub == nil || pub.Params == 0 {
				t.Fatalf("%s has no published parameter count to check against", name)
			}
			checked++
			tolerance := pub.Tolerance
			if tolerance == 0 {
				tolerance = 0.005
			}
			delta := math.Abs(result.Params.Total-pub.Params) / pub.Params
			if delta >= tolerance {
				t.Errorf("computes %s parameters, published %s: %s%% off, tolerance %s%%",
					analysis.FormatCount(result.Params.Total), analysis.FormatCount(pub.Params),
					analysis.JSToFixed(delta*100, 3), analysis.JSToFixed(tolerance*100, 1))
			}

			if pub.ActiveParams > 0 {
				delta := math.Abs(result.Params.Active-pub.ActiveParams) / pub.ActiveParams
				if delta >= tolerance {
					t.Errorf("activates %s parameters, published %s: %s%% off",
						analysis.FormatCount(result.Params.Active),
						analysis.FormatCount(pub.ActiveParams),
						analysis.JSToFixed(delta*100, 3))
				}
			}
		})
	}
	if checked != len(names) {
		t.Errorf("only %d of %d presets carried a published figure", checked, len(names))
	}
}

// TestExactPresets names the ones that have to match to the parameter.
//
// A tolerance is a claim about the source, not about the engine: it belongs on
// the presets whose published figure is itself a rounded headline number. A
// preset that quietly grew one would hide a real drift, so the list is written
// down rather than derived.
func TestExactPresets(t *testing.T) {
	// A tolerance is a claim about the source, not about the engine, so the
	// list is short and each entry is a published figure that is itself
	// rounded: "671B total, 37B active".
	approximate := map[string]bool{
		"deepseek-v3":     true,
		"qwen3-235b-a22b": true,
		"qwen3-30b-a3b":   true,
	}
	names, err := presets.Names()
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	exact := 0
	for _, name := range names {
		doc, err := presets.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		pub := doc.Meta.Published
		if pub == nil {
			continue
		}
		switch {
		case pub.Tolerance > 0 && !approximate[name]:
			t.Errorf("%s carries a tolerance of %g but is not on the list of presets "+
				"whose published figure is itself approximate", name, pub.Tolerance)
		case pub.Tolerance == 0 && approximate[name]:
			t.Errorf("%s is on the approximate list but carries no tolerance; "+
				"if it now matches exactly, take it off the list", name)
		case pub.Tolerance == 0:
			exact++
		}
		seen[name] = true
	}
	for name := range approximate {
		if !seen[name] {
			t.Errorf("%s is on the approximate list but is not in the library", name)
		}
	}
	if exact != len(names)-len(approximate) {
		t.Errorf("%d presets match exactly, expected %d", exact, len(names)-len(approximate))
	}
}

// TestGetReturnsAFreshCopy: a caller that edits a preset must not change what
// the next caller sees. The library is embedded and shared.
func TestGetReturnsAFreshCopy(t *testing.T) {
	first := presets.MustGet("gpt2-small")
	first.Meta.Name = "edited"
	first.Graph.Nodes = nil

	second := presets.MustGet("gpt2-small")
	if second.Meta.Name != "gpt2-small" {
		t.Errorf("the second copy is named %q", second.Meta.Name)
	}
	if len(second.Graph.Nodes) == 0 {
		t.Error("the second copy has no nodes")
	}
}

func TestUnknownPresetIsRefused(t *testing.T) {
	if _, err := presets.Get("no-such-model"); err == nil {
		t.Fatal("an unknown preset name was accepted")
	}
	if _, err := presets.Get("../../etc/passwd"); err == nil {
		t.Fatal("a path was accepted as a preset name")
	}
}
