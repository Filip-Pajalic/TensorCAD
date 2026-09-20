package diff_test

import (
	"encoding/json"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/diff"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
)

func get(t *testing.T, name string) *ir.Doc {
	t.Helper()
	doc, err := presets.Get(name)
	if err != nil {
		t.Fatalf("preset %s: %v", name, err)
	}
	return doc
}

func compare(t *testing.T, a, b *ir.Doc) *diff.Result {
	t.Helper()
	res, err := diff.Designs(a, b, analysis.Options{})
	if err != nil {
		t.Fatalf("diff: %v", err)
	}
	return res
}

// A design against itself is the base case, and the one most likely to be
// wrong: a diff that finds changes where there are none is worse than no diff.
func TestADesignAgainstItselfIsIdentical(t *testing.T) {
	for _, name := range []string{"gpt2-small", "mixtral-8x7b", "deepseek-v3", "alexnet"} {
		t.Run(name, func(t *testing.T) {
			res := compare(t, get(t, name), get(t, name))
			if !res.Identical {
				t.Errorf("a design differs from itself: %+v", res)
			}
			for _, m := range res.Metrics {
				if m.Delta != 0 {
					t.Errorf("%s moved by %v against itself", m.Metric, m.Delta)
				}
			}
		})
	}
}

// Two sizes of the same architecture differ in their symbols and in almost
// nothing else, which is what makes the diff readable at all.
func TestTwoSizesDifferInTheirSymbols(t *testing.T) {
	res := compare(t, get(t, "gpt2-small"), get(t, "gpt2-medium"))
	if res.Identical {
		t.Fatal("two different models compared identical")
	}
	changed := map[string]bool{}
	for _, c := range res.Symbols.Changed {
		changed[c.Name] = true
	}
	for _, want := range []string{"D", "H", "L"} {
		if !changed[want] {
			t.Errorf("%s did not change between small and medium", want)
		}
	}
	if len(res.Blocks.Added) != 0 || len(res.Blocks.Removed) != 0 {
		t.Errorf("the graph changed shape: +%d -%d", len(res.Blocks.Added), len(res.Blocks.Removed))
	}
	if len(res.Edges.Added) != 0 || len(res.Edges.Removed) != 0 {
		t.Errorf("the wiring changed: +%d -%d", len(res.Edges.Added), len(res.Edges.Removed))
	}
	// And the numbers say how much bigger, which is the half a structural diff
	// cannot give you.
	params := metric(res, "parameters")
	if params == nil || params.Delta <= 0 || params.Ratio == nil || *params.Ratio < 2 {
		t.Errorf("parameters: %+v", params)
	}
}

// Both sides are measured at one operating point, or the attention terms and
// the activation memory are not comparable.
func TestBothSidesAreMeasuredTheSameWay(t *testing.T) {
	// gpt2-small defaults to 1024 tokens; llama-3-8b to 8192. The longer wins.
	res := compare(t, get(t, "gpt2-small"), get(t, "llama-3-8b"))
	if res.At.T != 8192 {
		t.Errorf("measured at T=%v, expected the longer of the two defaults", res.At.T)
	}
	// And an explicit request overrides both.
	seq := 512.0
	fixed, err := diff.Designs(get(t, "gpt2-small"), get(t, "llama-3-8b"), analysis.Options{T: &seq})
	if err != nil {
		t.Fatalf("diff: %v", err)
	}
	if fixed.At.T != 512 {
		t.Errorf("asked for T=512, measured at %v", fixed.At.T)
	}
}

// A block that gained a parameter, lost one and changed one, all at once.
func TestOneBlockThreeWays(t *testing.T) {
	before := get(t, "gpt2-small")
	after := get(t, "gpt2-small")
	for i := range after.Graph.Nodes {
		if after.Graph.Nodes[i].ID == "head" {
			after.Graph.Nodes[i].Params = map[string]any{
				"vocab": "V", "dim": "D", "tied": false, "softcap": 30.0,
			}
			after.Graph.Nodes[i].Label = "renamed"
		}
	}
	res := compare(t, before, after)
	if len(res.Blocks.Changed) != 1 {
		t.Fatalf("%d blocks changed, expected 1", len(res.Blocks.Changed))
	}
	c := res.Blocks.Changed[0]
	if c.Path != "head" {
		t.Errorf("changed %q", c.Path)
	}
	if c.Label == nil {
		t.Error("the label change was not reported")
	}
	keys := map[string]ParamPair{}
	for _, p := range c.Params {
		keys[p.Key] = ParamPair{string(p.From), string(p.To)}
	}
	if got, ok := keys["softcap"]; !ok || got.From != "null" {
		t.Errorf("a gained parameter should come from null; got %+v", got)
	}
	if got, ok := keys["tied"]; !ok || got.From != "true" || got.To != "false" {
		t.Errorf("tied: %+v", got)
	}
	// A parameter that did not move is not reported.
	if _, ok := keys["dim"]; ok {
		t.Error("an unchanged parameter was reported as changed")
	}
	// The numbers followed: untying the head adds a whole vocabulary matrix.
	params := metric(res, "parameters")
	if params == nil || params.Delta <= 0 {
		t.Errorf("untying the head did not add parameters: %+v", params)
	}
}

// Structure and numbers are independent: two designs can be structurally
// identical and cost different amounts, because the numbers are measured.
func TestIdenticalMeansStructurallyIdentical(t *testing.T) {
	doc := get(t, "gpt2-small")
	seq := 4096.0
	res, err := diff.Designs(doc, doc, analysis.Options{T: &seq})
	if err != nil {
		t.Fatalf("diff: %v", err)
	}
	if !res.Identical {
		t.Error("a design compared against itself at one operating point is not identical")
	}
}

// Everything comes back as a list rather than a null, because a client reading
// `.length` on the answer should not have to check first.
func TestNothingComesBackNull(t *testing.T) {
	res := compare(t, get(t, "gpt2-small"), get(t, "gpt2-small"))
	raw, err := json.Marshal(res)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var walk func(v any, path string)
	walk = func(v any, path string) {
		switch t2 := v.(type) {
		case nil:
			// `ratio` is null on purpose when the denominator is zero.
			if path != ".metrics[].ratio" {
				t.Errorf("%s is null", path)
			}
		case map[string]any:
			for k, inner := range t2 {
				walk(inner, path+"."+k)
			}
		case []any:
			for _, inner := range t2 {
				walk(inner, path+"[]")
			}
		}
	}
	var parsed any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	walk(parsed, "")
}

type ParamPair struct{ From, To string }

func metric(res *diff.Result, name string) *diff.Delta {
	for i := range res.Metrics {
		if res.Metrics[i].Metric == name {
			return &res.Metrics[i]
		}
	}
	return nil
}
