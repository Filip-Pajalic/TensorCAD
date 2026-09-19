package ir_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
)

// The TypeScript engine is the specification until it is gone. It writes the
// document it builds and the answer it gets for every preset; this reads the
// same documents and requires the same answers, to the last symbol. Regenerate
// with `bun run scripts/golden.ts`.

type goldenSymbols struct {
	Order        []string           `json:"order"`
	Values       map[string]float64 `json:"values"`
	DesignValues map[string]float64 `json:"designValues"`
	Runtime      []string           `json:"runtime"`
	Docs         map[string]string  `json:"docs"`
	Errors       []string           `json:"errors"`
}

type golden struct {
	Preset  string        `json:"preset"`
	Symbols goldenSymbols `json:"symbols"`
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

func loadGolden(t *testing.T, name string) golden {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "golden", name+".json"))
	if err != nil {
		t.Fatalf("read golden %s: %v", name, err)
	}
	var g golden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("parse golden %s: %v", name, err)
	}
	return g
}

func TestEveryPresetDecodes(t *testing.T) {
	for _, name := range presetNames(t) {
		doc := loadDoc(t, name)
		if doc.Version != ir.DocVersion {
			t.Errorf("%s: version %d, want %d", name, doc.Version, ir.DocVersion)
		}
		if doc.Meta.Name == "" {
			t.Errorf("%s: no name", name)
		}
		if len(doc.Graph.Nodes) == 0 {
			t.Errorf("%s: no nodes", name)
		}
		if len(doc.Symbols) == 0 {
			t.Errorf("%s: no symbols", name)
		}
		// The order has to survive the round trip, or the symbol panel lists a
		// person's own symbols in an order they did not choose.
		if len(doc.SymbolOrder) != len(doc.Symbols) {
			t.Errorf("%s: recovered %d symbol names for %d symbols",
				name, len(doc.SymbolOrder), len(doc.Symbols))
		}
	}
}

func TestSymbolTableMatchesTypeScript(t *testing.T) {
	for _, name := range presetNames(t) {
		t.Run(name, func(t *testing.T) {
			got := ir.ResolveSymbols(loadDoc(t, name))
			want := loadGolden(t, name).Symbols

			if diff := diffStrings(got.Order, want.Order); diff != "" {
				t.Errorf("evaluation order: %s", diff)
			}
			compareValues(t, "values", got.Values, want.Values)
			compareValues(t, "designValues", got.DesignValues, want.DesignValues)

			runtime := make([]string, 0, len(got.Runtime))
			for k := range got.Runtime {
				runtime = append(runtime, k)
			}
			sort.Strings(runtime)
			if diff := diffStrings(runtime, want.Runtime); diff != "" {
				t.Errorf("runtime symbols: %s", diff)
			}

			for k, v := range want.Docs {
				if got.Docs[k] != v {
					t.Errorf("doc for %s: got %q, want %q", k, got.Docs[k], v)
				}
			}
			if diff := diffStrings(got.Errors, want.Errors); diff != "" {
				t.Errorf("errors: %s", diff)
			}
		})
	}
}

func compareValues(t *testing.T, label string, got, want map[string]float64) {
	t.Helper()
	for k, w := range want {
		g, ok := got[k]
		if !ok {
			t.Errorf("%s: %s missing, want %g", label, k, w)
			continue
		}
		// These are dimensions and counts: exact, or the port is wrong.
		if g != w && !(math.IsNaN(g) && math.IsNaN(w)) {
			t.Errorf("%s: %s = %g, want %g", label, k, g, w)
		}
	}
	for k, g := range got {
		if _, ok := want[k]; !ok {
			t.Errorf("%s: unexpected %s = %g", label, k, g)
		}
	}
}

func diffStrings(got, want []string) string {
	if len(got) == 0 && len(want) == 0 {
		return ""
	}
	if len(got) != len(want) {
		return "got " + sprintList(got) + ", want " + sprintList(want)
	}
	for i := range got {
		if got[i] != want[i] {
			return "got " + sprintList(got) + ", want " + sprintList(want)
		}
	}
	return ""
}

func sprintList(v []string) string {
	b, _ := json.Marshal(v)
	return string(b)
}
