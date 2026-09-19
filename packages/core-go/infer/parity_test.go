package infer_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/shapes"
)

// Shape inference against the TypeScript, port by port.
//
// Every preset is walked twice, once as written and once with composites
// expanded, because both are used: the rule engine and the canvas read the
// graph a person drew, and the analysis reads it expanded. A shape is compared
// as the text the editor would print, so agreeing means agreeing on the
// polynomial and on how it reads: `B T 4096` where the TypeScript says
// `B T D` is a right number and a wrong answer. Regenerate with
// `bun run scripts/golden.ts`.

type goldenIssue struct {
	Path     string `json:"path"`
	Port     string `json:"port,omitempty"`
	Message  string `json:"message"`
	Severity string `json:"severity"`
	Rule     string `json:"rule,omitempty"`
	Param    string `json:"param,omitempty"`
}

type goldenInfer struct {
	Outputs    [][2]string   `json:"outputs"`
	Inputs     [][2]string   `json:"inputs"`
	ProducerOf [][2]string   `json:"producerOf"`
	Issues     []goldenIssue `json:"issues"`
}

type goldenFile struct {
	Preset        string      `json:"preset"`
	Infer         goldenInfer `json:"infer"`
	InferExpanded goldenInfer `json:"inferExpanded"`
}

func presetNames(t *testing.T) []string {
	t.Helper()
	names, err := presets.Names()
	if err != nil {
		t.Fatalf("read preset index: %v", err)
	}
	if len(names) == 0 {
		t.Fatal("the preset library is empty")
	}
	return names
}

func loadDoc(t *testing.T, name string) *ir.Doc {
	t.Helper()
	doc, err := presets.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	return doc
}

func loadGolden(t *testing.T, name string) goldenFile {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "golden", name+".json"))
	if err != nil {
		t.Fatalf("read golden %s: %v", name, err)
	}
	var g goldenFile
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("parse golden %s: %v", name, err)
	}
	return g
}

func TestInferredShapesMatchTypeScript(t *testing.T) {
	for _, name := range presetNames(t) {
		g := loadGolden(t, name)
		for _, mode := range []struct {
			label string
			opts  infer.Options
			want  goldenInfer
		}{
			{"flat", infer.Options{}, g.Infer},
			{"expanded", infer.Options{ExpandComposites: true}, g.InferExpanded},
		} {
			t.Run(name+"/"+mode.label, func(t *testing.T) {
				doc := loadDoc(t, name)
				got := infer.Shapes(doc, ir.ResolveSymbols(doc), mode.opts)

				compareShapes(t, "outputs", got.Outputs, mode.want.Outputs)
				compareShapes(t, "inputs", got.Inputs, mode.want.Inputs)
				compareStrings(t, "producerOf", got.ProducerOf, mode.want.ProducerOf)
				compareIssues(t, got.Issues, mode.want.Issues)
			})
		}
	}
}

// TestEveryPresetIsFullyResolved is the standing claim the presets make: they
// are the regression suite, so a preset with a dangling port or an unknown
// block is a broken suite whatever the goldens say.
func TestEveryPresetIsFullyResolved(t *testing.T) {
	for _, name := range presetNames(t) {
		t.Run(name, func(t *testing.T) {
			doc := loadDoc(t, name)
			symbols := ir.ResolveSymbols(doc)
			for _, opts := range []infer.Options{{}, {ExpandComposites: true}} {
				res := infer.Shapes(doc, symbols, opts)
				for _, issue := range res.Issues {
					if issue.Severity == "error" {
						t.Errorf("expand=%v %s %s: %s",
							opts.ExpandComposites, issue.Path, issue.Port, issue.Message)
					}
				}
				if len(res.Outputs) == 0 {
					t.Fatalf("expand=%v: nothing was inferred", opts.ExpandComposites)
				}
			}
		})
	}
}

func compareShapes(t *testing.T, label string, got map[string]shapes.Shape, want [][2]string) {
	t.Helper()
	as := make(map[string]string, len(got))
	for k, v := range got {
		as[k] = shapes.ShapeToString(v)
	}
	compareStrings(t, label, as, want)
}

func compareStrings(t *testing.T, label string, got map[string]string, want [][2]string) {
	t.Helper()
	seen := make(map[string]bool, len(want))
	for _, kv := range want {
		key, w := kv[0], kv[1]
		seen[key] = true
		g, ok := got[key]
		if !ok {
			t.Errorf("%s: %s missing, want %q", label, key, w)
			continue
		}
		if g != w {
			t.Errorf("%s: %s = %q, want %q", label, key, g, w)
		}
	}
	// An extra entry is a failure too: a port the Go walk invented is a port
	// the editor would draw a wire to and the analysis would count.
	extra := make([]string, 0)
	for k := range got {
		if !seen[k] {
			extra = append(extra, k)
		}
	}
	sort.Strings(extra)
	for _, k := range extra {
		t.Errorf("%s: %s = %q is not in the TypeScript", label, k, got[k])
	}
}

// compareIssues compares the findings as a set.
//
// The order is not the contract: the TypeScript visits a block's ports in the
// order they were declared, and a Go map has no such order, so the two engines
// can report the same findings in a different sequence. What they may not do is
// report a different set of them.
func compareIssues(t *testing.T, got []infer.Issue, want []goldenIssue) {
	t.Helper()
	g := make([]string, len(got))
	for i, x := range got {
		g[i] = issueKey(x.Path, x.Port, x.Message, x.Severity, x.Rule, x.Param)
	}
	w := make([]string, len(want))
	for i, x := range want {
		w[i] = issueKey(x.Path, x.Port, x.Message, x.Severity, x.Rule, x.Param)
	}
	sort.Strings(g)
	sort.Strings(w)

	for _, missing := range difference(w, g) {
		t.Errorf("issue not reported: %s", missing)
	}
	for _, extra := range difference(g, w) {
		t.Errorf("issue not in the TypeScript: %s", extra)
	}
}

func issueKey(parts ...string) string {
	quoted := make([]string, len(parts))
	for i, p := range parts {
		quoted[i] = fmt.Sprintf("%q", p)
	}
	return strings.Join(quoted, " | ")
}

// difference is a minus b, counting duplicates.
func difference(a, b []string) []string {
	remaining := map[string]int{}
	for _, x := range b {
		remaining[x]++
	}
	var out []string
	for _, x := range a {
		if remaining[x] > 0 {
			remaining[x]--
			continue
		}
		out = append(out, x)
	}
	return out
}
