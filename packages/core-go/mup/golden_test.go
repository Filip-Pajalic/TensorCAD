package mup_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/golden"
	"github.com/tensorcad/core/mup"
	"github.com/tensorcad/core/presets"
)

// The ladder against the goldens, rung for rung.
//
// The classification is as much the answer as the multipliers are: a weight
// that moves from the hidden row to the output row is being initialized a
// factor of the width differently, which is a thing to review rather than to
// discover after a sweep transferred badly. Regenerate with
// `go run ./cmd/golden`.

type goldenScaling struct {
	Class   string   `json:"class"`
	InitStd string   `json:"initStd"`
	AdamLR  string   `json:"adamLr"`
	Paths   []string `json:"paths"`
}

type goldenRung struct {
	Width      float64         `json:"width"`
	Multiplier string          `json:"multiplier"`
	Heads      float64         `json:"heads"`
	Params     float64         `json:"params"`
	Base       bool            `json:"base"`
	Scaling    []goldenScaling `json:"scaling"`
	Notes      []string        `json:"notes"`
	Symbols    [][2]any        `json:"symbols"`
}

type goldenLadder struct {
	Label       string       `json:"label"`
	Preset      string       `json:"preset"`
	Options     mup.Options  `json:"options"`
	WidthSymbol string       `json:"widthSymbol"`
	BaseWidth   float64      `json:"baseWidth"`
	HeadDim     float64      `json:"headDim"`
	Rungs       []goldenRung `json:"rungs"`
	Notes       []string     `json:"notes"`
}

func loadLadders(t *testing.T) []goldenLadder {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "mup.json"))
	if err != nil {
		t.Fatalf("read the ladder goldens: %v", err)
	}
	var file struct {
		Cases []goldenLadder `json:"cases"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatalf("parse the ladder goldens: %v", err)
	}
	if len(file.Cases) == 0 {
		t.Fatal("the ladder goldens are empty")
	}
	return file.Cases
}

func TestLaddersMatchTheGoldens(t *testing.T) {
	for _, want := range loadLadders(t) {
		t.Run(want.Label, func(t *testing.T) {
			got, err := mup.Build(presets.MustGet(want.Preset), want.Options)
			if err != nil {
				t.Fatal(err)
			}
			if got.WidthSymbol != want.WidthSymbol {
				t.Errorf("moved %q, want %q", got.WidthSymbol, want.WidthSymbol)
			}
			if got.BaseWidth != want.BaseWidth {
				t.Errorf("base width %v, want %v", got.BaseWidth, want.BaseWidth)
			}
			if got.HeadDim != want.HeadDim {
				t.Errorf("head dimension %v, want %v", got.HeadDim, want.HeadDim)
			}
			sameStrings(t, "the ladder's notes", got.Notes, want.Notes)
			if len(got.Rungs) != len(want.Rungs) {
				t.Fatalf("%d rungs, want %d", len(got.Rungs), len(want.Rungs))
			}
			for i, w := range want.Rungs {
				g := got.Rungs[i]
				if g.Width != w.Width || g.Heads != w.Heads || g.Params != w.Params {
					t.Errorf("rung %d: %v wide, %v heads, %v parameters; want %v, %v, %v",
						i, g.Width, g.Heads, g.Params, w.Width, w.Heads, w.Params)
				}
				if m := analysis.JSNumber(g.Multiplier); m != w.Multiplier {
					t.Errorf("rung %d: multiplier %s, want %s", i, m, w.Multiplier)
				}
				if g.Base != w.Base {
					t.Errorf("rung %d: base %v, want %v", i, g.Base, w.Base)
				}
				sameStrings(t, "rung "+g.Doc.Meta.Name+"'s notes", g.Notes, w.Notes)
				if len(g.Scaling) != len(w.Scaling) {
					t.Fatalf("rung %d: %d classes, want %d", i, len(g.Scaling), len(w.Scaling))
				}
				for j, ws := range w.Scaling {
					gs := g.Scaling[j]
					if string(gs.Class) != ws.Class {
						t.Errorf("rung %d class %d: %q, want %q", i, j, gs.Class, ws.Class)
					}
					if s := analysis.JSNumber(gs.InitStd); s != ws.InitStd {
						t.Errorf("rung %d, %s: init x%s, want x%s", i, ws.Class, s, ws.InitStd)
					}
					if s := analysis.JSNumber(gs.AdamLR); s != ws.AdamLR {
						t.Errorf("rung %d, %s: rate x%s, want x%s", i, ws.Class, s, ws.AdamLR)
					}
					sameStrings(t, "rung "+analysis.JSNumber(g.Width)+"'s "+ws.Class+" weights",
						gs.Paths, ws.Paths)
				}
			}
		})
	}
}

// The generator's cases and the goldens have to stay in step: a case added to
// one and not written to the other would test nothing and say nothing.
func TestEveryCaseIsInTheGoldens(t *testing.T) {
	have := map[string]bool{}
	for _, c := range loadLadders(t) {
		have[c.Label] = true
	}
	for _, c := range golden.MupCases {
		if !have[c.Label] {
			t.Errorf("case %q is not in the goldens; run `go run ./cmd/golden`", c.Label)
		}
	}
	if len(golden.MupCases) != len(have) {
		t.Errorf("%d cases against %d goldens", len(golden.MupCases), len(have))
	}
}

func sameStrings(t *testing.T, what string, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: %d entries, want %d\n got: %v\nwant: %v", what, len(got), len(want), got, want)
		return
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("%s, %d: %q, want %q", what, i, got[i], want[i])
		}
	}
}

// Sanity that the goldens are not empty answers: a ladder of one rung with no
// weights in it would match itself forever.
func TestTheGoldensSaySomething(t *testing.T) {
	for _, c := range loadLadders(t) {
		if len(c.Rungs) < 2 {
			t.Errorf("%s has %d rungs; a ladder of one transfers nothing", c.Label, len(c.Rungs))
		}
		for _, r := range c.Rungs {
			hidden := 0
			for _, s := range r.Scaling {
				hidden += len(s.Paths)
			}
			if hidden == 0 {
				t.Errorf("%s at width %v classifies no weights at all", c.Label, r.Width)
			}
		}
	}
}
