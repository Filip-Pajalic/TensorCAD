package explain_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/explain"
	"github.com/tensorcad/core/presets"
)

// What the tool says about a block, against the TypeScript.
//
// Explain is the learning surface, so the prose matters as much as the numbers:
// the expression a person wrote beside the value it came to, the block's own
// summary, and the formula with its source. A wrong share of the parameter
// count is a wrong number; a missing formula is a tool that stopped teaching.
// Regenerate with `bun run scripts/golden.ts`.

// A parameter is written as [name, expression, value, doc], which keeps the
// golden readable next to the block it describes.
type goldenParam [4]json.RawMessage

type goldenBlock struct {
	Path string `json:"path"`
	Type string `json:"type"`
	Kind string `json:"kind"`
	Docs struct {
		Summary string   `json:"summary"`
		Formula string   `json:"formula"`
		Refs    []string `json:"refs"`
	} `json:"docs"`
	Copies struct {
		Total  float64 `json:"total"`
		Active float64 `json:"active"`
	} `json:"copies"`
	Params []goldenParam `json:"params"`
	Shapes struct {
		In  map[string]string `json:"in"`
		Out map[string]string `json:"out"`
	} `json:"shapes"`
	Contributes struct {
		Params                float64 `json:"params"`
		ActiveParams          float64 `json:"activeParams"`
		ShareOfParams         float64 `json:"shareOfParams"`
		FlopsPerToken         float64 `json:"flopsPerToken"`
		ShareOfFlops          float64 `json:"shareOfFlops"`
		ActivationBytes       float64 `json:"activationBytes"`
		CacheBytesPerToken    float64 `json:"cacheBytesPerToken"`
		CacheBytesPerSequence float64 `json:"cacheBytesPerSequence"`
	} `json:"contributes"`
	Breakdown []struct {
		Path   string  `json:"path"`
		Type   string  `json:"type"`
		Params float64 `json:"params"`
	} `json:"breakdown"`
}

func TestExplanationsMatchTypeScript(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "explain.json"))
	if err != nil {
		t.Fatalf("read explain golden: %v", err)
	}
	var g struct {
		Cases []struct {
			Preset string        `json:"preset"`
			Blocks []goldenBlock `json:"blocks"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse explain golden: %v", err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("explain golden is empty")
	}

	for _, c := range g.Cases {
		t.Run(c.Preset, func(t *testing.T) {
			doc, err := presets.Get(c.Preset)
			if err != nil {
				t.Fatal(err)
			}
			all, err := explain.All(doc, analysis.Options{})
			if err != nil {
				t.Fatal(err)
			}
			if len(all) < len(c.Blocks) {
				t.Fatalf("explained %d blocks, the TypeScript explained at least %d",
					len(all), len(c.Blocks))
			}

			// The order is the answer too: largest contribution first is what
			// puts the thing worth looking at at the top of the panel.
			for i, want := range c.Blocks {
				got := all[i]
				if got.Path != want.Path {
					t.Errorf("block %d: got %q, want %q", i, got.Path, want.Path)
					continue
				}
				t.Run(want.Path, func(t *testing.T) { compareBlock(t, got, want) })
			}
		})
	}
}

func compareBlock(t *testing.T, got *explain.Explanation, want goldenBlock) {
	t.Helper()
	if got.Type != want.Type {
		t.Errorf("type: got %q, want %q", got.Type, want.Type)
	}
	if got.Kind != want.Kind {
		t.Errorf("kind: got %q, want %q", got.Kind, want.Kind)
	}
	if got.Docs.Summary != want.Docs.Summary {
		t.Errorf("summary: got %q, want %q", got.Docs.Summary, want.Docs.Summary)
	}
	if got.Docs.Formula != want.Docs.Formula {
		t.Errorf("formula: got %q, want %q", got.Docs.Formula, want.Docs.Formula)
	}
	if len(got.Docs.Refs) != len(want.Docs.Refs) {
		t.Errorf("refs: got %v, want %v", got.Docs.Refs, want.Docs.Refs)
	} else {
		for i := range got.Docs.Refs {
			if got.Docs.Refs[i] != want.Docs.Refs[i] {
				t.Errorf("ref %d: got %q, want %q", i, got.Docs.Refs[i], want.Docs.Refs[i])
			}
		}
	}
	exact(t, "copies.total", got.Copies.Total, want.Copies.Total)
	exact(t, "copies.active", got.Copies.Active, want.Copies.Active)

	if len(got.Params) != len(want.Params) {
		t.Errorf("parameters: got %d, want %d", len(got.Params), len(want.Params))
	}
	for _, p := range want.Params {
		var name string
		if err := json.Unmarshal(p[0], &name); err != nil {
			t.Fatalf("bad golden parameter: %v", err)
		}
		g, ok := got.Params[name]
		if !ok {
			t.Errorf("parameter %q missing", name)
			continue
		}
		var wantExpr string
		_ = json.Unmarshal(p[1], &wantExpr) // null decodes to ""
		if g.Expression != wantExpr {
			t.Errorf("parameter %q expression: got %q, want %q", name, g.Expression, wantExpr)
		}
		if !sameJSON(g.Value, p[2]) {
			t.Errorf("parameter %q value: got %#v, want %s", name, g.Value, p[2])
		}
		var wantDoc string
		_ = json.Unmarshal(p[3], &wantDoc)
		if g.Doc != wantDoc {
			t.Errorf("parameter %q doc: got %q, want %q", name, g.Doc, wantDoc)
		}
	}

	compareShapes(t, "in", got.Shapes.In, want.Shapes.In)
	compareShapes(t, "out", got.Shapes.Out, want.Shapes.Out)

	exact(t, "params", got.Contributes.Params, want.Contributes.Params)
	exact(t, "activeParams", got.Contributes.ActiveParams, want.Contributes.ActiveParams)
	closeTo(t, "shareOfParams", got.Contributes.ShareOfParams, want.Contributes.ShareOfParams)
	closeTo(t, "flopsPerToken", got.Contributes.FlopsPerToken, want.Contributes.FlopsPerToken)
	closeTo(t, "shareOfFlops", got.Contributes.ShareOfFlops, want.Contributes.ShareOfFlops)
	closeTo(t, "activationBytes", got.Contributes.ActivationBytes, want.Contributes.ActivationBytes)
	closeTo(t, "cacheBytesPerToken", got.Contributes.CacheBytesPerToken, want.Contributes.CacheBytesPerToken)
	closeTo(t, "cacheBytesPerSequence", got.Contributes.CacheBytesPerSequence, want.Contributes.CacheBytesPerSequence)

	if len(got.Breakdown) != len(want.Breakdown) {
		t.Errorf("breakdown: got %d lines, want %d", len(got.Breakdown), len(want.Breakdown))
		return
	}
	for i := range got.Breakdown {
		g, w := got.Breakdown[i], want.Breakdown[i]
		if g.Path != w.Path || g.Type != w.Type || g.Params != w.Params {
			t.Errorf("breakdown %d: got %+v, want %+v", i, g, w)
		}
	}
}

func compareShapes(t *testing.T, side string, got, want map[string]string) {
	t.Helper()
	for name, shape := range want {
		if got[name] != shape {
			t.Errorf("%s port %q: got %q, want %q", side, name, got[name], shape)
		}
	}
	for name := range got {
		if _, ok := want[name]; !ok {
			t.Errorf("%s port %q is not in the TypeScript", side, name)
		}
	}
}

func sameJSON(got any, want json.RawMessage) bool {
	a, err := json.Marshal(got)
	if err != nil {
		return false
	}
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(want, &y) != nil {
		return false
	}
	b1, _ := json.Marshal(x)
	b2, _ := json.Marshal(y)
	return string(b1) == string(b2)
}

func exact(t *testing.T, label string, got, want float64) {
	t.Helper()
	if got != want {
		t.Errorf("%s: got %.17g, want %.17g", label, got, want)
	}
}

func closeTo(t *testing.T, label string, got, want float64) {
	t.Helper()
	if got == want {
		return
	}
	scale := math.Max(math.Abs(got), math.Abs(want))
	if scale > 0 && math.Abs(got-want)/scale <= 1e-12 {
		return
	}
	t.Errorf("%s: got %.17g, want %.17g", label, got, want)
}
