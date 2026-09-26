package start_test

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/jsonx"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
	"github.com/tensorcad/core/start"
)

// Every size a kind offers makes a design that is that size, builds, and is
// of that kind.
//
// This is the promise the dialog makes by listing the size at all: a choice
// that came out a fifth off, or with a design rule in error, is a choice the
// editor should not have offered.
func TestEveryOfferedSizeMakesADesignOfThatSize(t *testing.T) {
	for _, family := range start.Families {
		if _, err := presets.Get(family.Base); err != nil {
			t.Fatalf("%s starts from %s, which is not a preset: %v", family.ID, family.Base, err)
		}
		if family.Name == "" || family.Summary == "" || len(family.Sizes) == 0 {
			t.Errorf("%s is missing a name, a summary or its sizes", family.ID)
		}
		for _, params := range family.Sizes {
			result, err := start.New(start.Request{Family: family.ID, Params: params}, analysis.Options{})
			if err != nil {
				t.Errorf("%s at %g: %v", family.ID, params, err)
				continue
			}
			if off := math.Abs(result.Params-params) / params; off > 0.1 {
				t.Errorf("%s at %g came out at %g, %.0f%% off", family.ID, params, result.Params, off*100)
			}
			report, err := rules.Validate(result.Doc, analysis.Options{})
			if err != nil {
				t.Errorf("%s at %g: %v", family.ID, params, err)
				continue
			}
			for _, f := range report.Findings {
				if f.Severity == "error" {
					t.Errorf("%s at %g: %s: %s", family.ID, params, f.Rule, f.Message)
				}
			}
			if result.Doc.Meta.Published != nil {
				t.Errorf("%s at %g still claims %s's published figure", family.ID, params, family.Base)
			}
			if !strings.Contains(result.Doc.Meta.Notes, family.Base) {
				t.Errorf("%s at %g does not say what it started from: %q", family.ID, params, result.Doc.Meta.Notes)
			}
		}
	}
}

// Query heads stay in whole groups over the key heads, the reference's ratio.
func TestHeadsStayInTheirGroups(t *testing.T) {
	for _, params := range []float64{350e6, 1e9, 3e9} {
		result, err := start.New(start.Request{Family: "dense", Params: params}, analysis.Options{})
		if err != nil {
			t.Fatal(err)
		}
		h, kv := result.Symbols["H"], result.Symbols["Hkv"]
		if kv == 0 || h/kv != 4 {
			t.Errorf("a Llama-style design at %g has %g query heads over %g, not four to one", params, h, kv)
		}
	}
}

// A small design with a large vocabulary ties its head, as small models do.
func TestASmallDesignTiesItsHead(t *testing.T) {
	result, err := start.New(start.Request{Family: "dense", Params: 350e6}, analysis.Options{})
	if err != nil {
		t.Fatal(err)
	}
	tied := false
	for _, n := range result.Doc.Graph.Nodes {
		if n.Type == "lm_head" {
			tied, _ = n.Params["tied"].(bool)
		}
	}
	if !tied {
		t.Error("a 350M design with Llama 3's vocabulary should share its embedding with its head")
	}
	large, err := start.New(start.Request{Family: "dense", Params: 8e9}, analysis.Options{})
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range large.Doc.Graph.Nodes {
		if n.Type == "lm_head" {
			if v, _ := n.Params["tied"].(bool); v {
				t.Error("at eight billion the vocabulary is a small share, and Llama 3 does not tie")
			}
		}
	}
}

// Fitting a device finds the largest design whose training fits, and says so.
func TestFittingADeviceFindsTheLargestThatFits(t *testing.T) {
	for _, family := range []string{"dense", "classic", "moe"} {
		result, err := start.New(start.Request{Family: family, Fit: "rtx4090"}, analysis.Options{})
		if err != nil {
			t.Fatalf("%s: %v", family, err)
		}
		if !result.Fits || result.TrainBytes > result.Budget {
			t.Errorf("%s: %g bytes to train against a budget of %g", family, result.TrainBytes, result.Budget)
		}
		if result.Budget != 24*(1<<30)*(1-start.Headroom) {
			t.Errorf("%s: the budget should be a 4090 less the planner's headroom, not %g", family, result.Budget)
		}
		// A fifth larger does not fit, or the search stopped short.
		bigger, err := start.New(start.Request{Family: family, Params: result.Params * 1.2}, analysis.Options{Hardware: "rtx4090"})
		if err != nil {
			t.Fatal(err)
		}
		if bigger.Fits {
			t.Errorf("%s: %g fits but the search settled on %g", family, bigger.Params, result.Params)
		}
	}
}

// The new design trains at its kind's sequence length, not the reference's
// serving context: Qwen3's is 32,768, and a first design measured there fits
// nowhere.
func TestANewDesignTrainsAtItsKindsSequence(t *testing.T) {
	result, err := start.New(start.Request{Family: "moe", Params: 1e9}, analysis.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if got := ir.ResolveSymbols(result.Doc).Values["T"]; got != 4096 {
		t.Errorf("T is %g, not 4096", got)
	}
	t5, err := start.New(start.Request{Family: "encoder-decoder", Params: 60e6}, analysis.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if got := ir.ResolveSymbols(t5.Doc).Values["T"]; got != 114 {
		t.Errorf("an encoder-decoder keeps T5's own target length, not %g", got)
	}
}

func TestNonsenseIsRefusedWithAReason(t *testing.T) {
	for _, c := range []struct {
		req  start.Request
		want string
	}{
		{start.Request{Family: "rnn", Params: 1e9}, "no kind of model"},
		{start.Request{Family: "dense"}, "say how big"},
		{start.Request{Family: "dense", Params: 1000}, "smaller than"},
		{start.Request{Family: "dense", Fit: "abacus"}, "abacus"},
	} {
		_, err := start.New(c.req, analysis.Options{})
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%+v: got %v, want an error saying %q", c.req, err, c.want)
		}
	}
}

// What crosses to the editor has its lists as lists, even the empty ones: a
// design scaled with nothing to say has no notes, and null is not an empty
// list to a panel that asks for its length.
func TestTheResultCrossesWhole(t *testing.T) {
	result, err := start.New(start.Request{Family: "classic", Params: 125e6}, analysis.Options{})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := jsonx.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"doc", "notes", "symbols", "trainBytes", "budget", "fits"} {
		if decoded[key] == nil {
			t.Errorf("%s crossed as null", key)
		}
	}
}
