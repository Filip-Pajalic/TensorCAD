package scale_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/scale"
)

// Scaling against the TypeScript, symbol for symbol.
//
// A search is where two engines drift most easily: the same binary search over
// the same rounding has to land on the same widths, not merely near them. Six
// cases, chosen so each exercises a different corner — a dense model, one
// scaled by its non-embedding count with a tied head, one at fixed depth, a
// mixture of experts, latent attention, and a state-space model. The variants
// must stay in step with SCALE_CASES in scripts/golden.ts. Regenerate with
// `bun run scripts/golden.ts`.

func f(v float64) *float64 { return &v }
func b(v bool) *bool       { return &v }

func cases() map[string]scale.Options {
	return map[string]scale.Options{
		"30m":            {TargetParams: 30e6},
		"30m-nonembed":   {TargetParams: 30e6, TargetBasis: "non-embedding", Vocab: f(8192), TieHead: b(true)},
		"100m-keepdepth": {TargetParams: 100e6, KeepDepth: true},
		"50m-moe":        {TargetParams: 50e6, Vocab: f(4096)},
		"20m-mla":        {TargetParams: 20e6, Vocab: f(4096)},
		"10m-mamba":      {TargetParams: 10e6, Vocab: f(4096)},
	}
}

type goldenScale struct {
	Label      string               `json:"label"`
	Preset     string               `json:"preset"`
	Achieved   float64              `json:"achieved"`
	Target     float64              `json:"target"`
	Changes    [][3]json.RawMessage `json:"changes"`
	Notes      []string             `json:"notes"`
	Name       string               `json:"name"`
	NotesOnDoc string               `json:"notesOnDoc"`
	Published  json.RawMessage      `json:"published"`
	Symbols    [][2]json.RawMessage `json:"symbols"`
}

func TestScaledDesignsMatchTypeScript(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "scale.json"))
	if err != nil {
		t.Fatalf("read scale golden: %v", err)
	}
	var g struct {
		Cases []goldenScale `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse scale golden: %v", err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("scale golden is empty")
	}

	opts := cases()
	for _, c := range g.Cases {
		t.Run(c.Label, func(t *testing.T) {
			o, ok := opts[c.Label]
			if !ok {
				t.Fatalf("no Go case named %q", c.Label)
			}
			base, err := presets.Get(c.Preset)
			if err != nil {
				t.Fatal(err)
			}
			got, err := scale.Design(base, o)
			if err != nil {
				t.Fatal(err)
			}

			if got.Achieved != c.Achieved {
				t.Errorf("achieved %.0f parameters, the TypeScript reached %.0f", got.Achieved, c.Achieved)
			}
			if got.Target != c.Target {
				t.Errorf("target: got %.0f, want %.0f", got.Target, c.Target)
			}
			if got.Doc.Meta.Name != c.Name {
				t.Errorf("name: got %q, want %q", got.Doc.Meta.Name, c.Name)
			}
			if got.Doc.Meta.Notes != c.NotesOnDoc {
				t.Errorf("notes on the document:\n got  %q\n want %q", got.Doc.Meta.Notes, c.NotesOnDoc)
			}
			// The published figure belongs to the original, not to a design
			// scaled away from it.
			if got.Doc.Meta.Published != nil {
				t.Error("the scaled design kept the original's published figure")
			}
			if string(c.Published) != "null" {
				t.Errorf("the TypeScript kept a published figure: %s", c.Published)
			}

			if len(got.Notes) != len(c.Notes) {
				t.Errorf("notes: got %d, want %d\n go   %q\n want %q",
					len(got.Notes), len(c.Notes), got.Notes, c.Notes)
			} else {
				for i := range got.Notes {
					if got.Notes[i] != c.Notes[i] {
						t.Errorf("note %d:\n got  %q\n want %q", i, got.Notes[i], c.Notes[i])
					}
				}
			}

			// The changes are the answer: which symbols moved and to what.
			wantChanges := map[string][2]float64{}
			for _, ch := range c.Changes {
				var name string
				var from, to float64
				mustDecode(t, ch[0], &name)
				mustDecode(t, ch[1], &from)
				mustDecode(t, ch[2], &to)
				wantChanges[name] = [2]float64{from, to}
			}
			for name, w := range wantChanges {
				g, ok := got.Changes[name]
				if !ok {
					t.Errorf("%s did not change; the TypeScript moved it %g -> %g", name, w[0], w[1])
					continue
				}
				if g.From != w[0] || g.To != w[1] {
					t.Errorf("%s: got %g -> %g, want %g -> %g", name, g.From, g.To, w[0], w[1])
				}
			}
			for name, g := range got.Changes {
				if _, ok := wantChanges[name]; !ok {
					t.Errorf("%s changed %g -> %g, which the TypeScript did not do", name, g.From, g.To)
				}
			}

			// And the symbol table the scaled document resolves to, which is
			// what a bench run would actually build.
			table := ir.ResolveSymbols(got.Doc)
			wantSymbols := map[string]float64{}
			for _, s := range c.Symbols {
				var name string
				var value float64
				mustDecode(t, s[0], &name)
				mustDecode(t, s[1], &value)
				wantSymbols[name] = value
			}
			for name, w := range wantSymbols {
				if v, ok := table.DesignValues[name]; !ok || v != w {
					t.Errorf("symbol %s: got %g, want %g", name, v, w)
				}
			}
			for name, v := range table.DesignValues {
				if _, ok := wantSymbols[name]; !ok {
					t.Errorf("symbol %s = %g is not in the TypeScript", name, v)
				}
			}
		})
	}
}

// TestScaledDesignsStillCheckOut: a design that shrinks into something the rule
// engine rejects is not a bench proxy, it is a broken document.
func TestScaledDesignsStillCheckOut(t *testing.T) {
	for label, o := range cases() {
		t.Run(label, func(t *testing.T) {
			base := presets.MustGet(presetFor(label))
			got, err := scale.Design(base, o)
			if err != nil {
				t.Fatal(err)
			}
			result, err := analysis.Analyze(got.Doc, analysis.Options{}, analysis.Inputs{})
			if err != nil {
				t.Fatal(err)
			}
			for _, e := range result.Errors {
				t.Errorf("the scaled design does not analyse: %s", e)
			}
			if result.Params.Total <= 0 {
				t.Error("the scaled design has no parameters")
			}
		})
	}
}

func presetFor(label string) string {
	switch label {
	case "30m":
		return "gpt2-small"
	case "50m-moe":
		return "mixtral-8x7b"
	case "20m-mla":
		return "deepseek-v3"
	case "10m-mamba":
		return "nemotron-h-8b"
	}
	return "llama-3-8b"
}

func TestTargetMustBePositive(t *testing.T) {
	if _, err := scale.Design(presets.MustGet("gpt2-small"), scale.Options{}); err == nil {
		t.Fatal("a target of zero was accepted")
	}
}

func mustDecode(t *testing.T, raw json.RawMessage, into any) {
	t.Helper()
	if err := json.Unmarshal(raw, into); err != nil {
		t.Fatalf("bad golden entry %s: %v", raw, err)
	}
}
