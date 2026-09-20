package catalog_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorcad/core/catalog"
)

// The catalog's prose, against the goldens.
//
// Every block's summary, formula and sources, and every parameter's one-line
// documentation. This is what the inspector shows and what explain reads out,
// so it is part of the engine's output rather than decoration around it — and
// it is the easiest thing to lose in a port, because nothing computes with it
// and every other test would still pass. Regenerate with
// `go run ./cmd/golden`.

// A parameter is [name, type, doc, values], which keeps the golden readable.
type goldenParamDoc [4]json.RawMessage

type goldenBlockDocs struct {
	Type     string           `json:"type"`
	Kind     string           `json:"kind"`
	Category string           `json:"category"`
	Summary  string           `json:"summary"`
	Formula  string           `json:"formula"`
	Refs     []string         `json:"refs"`
	Params   []goldenParamDoc `json:"params"`
}

func TestCatalogProseMatchesTheGoldens(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "catalog-docs.json"))
	if err != nil {
		t.Fatalf("read catalog prose golden: %v", err)
	}
	var g struct {
		Blocks []goldenBlockDocs `json:"blocks"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse catalog prose golden: %v", err)
	}
	if len(g.Blocks) == 0 {
		t.Fatal("catalog prose golden is empty")
	}

	seen := map[string]bool{}
	for _, want := range g.Blocks {
		seen[want.Type] = true
		t.Run(want.Type, func(t *testing.T) {
			def, err := catalog.Builtin.Get(want.Type)
			if err != nil {
				t.Fatal(err)
			}
			for _, s := range []struct{ label, got, want string }{
				{"kind", def.Kind, want.Kind},
				{"category", def.Category, want.Category},
				{"summary", def.Docs.Summary, want.Summary},
				{"formula", def.Docs.Formula, want.Formula},
			} {
				if s.got != s.want {
					t.Errorf("%s:\n got  %q\n want %q", s.label, s.got, s.want)
				}
			}
			if len(def.Docs.Refs) != len(want.Refs) {
				t.Errorf("sources: got %v, want %v", def.Docs.Refs, want.Refs)
			} else {
				for i := range want.Refs {
					if def.Docs.Refs[i] != want.Refs[i] {
						t.Errorf("source %d: got %q, want %q", i, def.Docs.Refs[i], want.Refs[i])
					}
				}
			}

			// The order is the answer too: it is the order the inspector lays
			// the fields out in and the order a generated class documents them.
			if len(def.Params) != len(want.Params) {
				t.Errorf("parameters: got %d, want %d (%v)", len(def.Params), len(want.Params), names(def.Params))
				return
			}
			for i, w := range want.Params {
				var name, kind, doc string
				var values []string
				mustDecode(t, w[0], &name)
				mustDecode(t, w[1], &kind)
				mustDecode(t, w[2], &doc)
				_ = json.Unmarshal(w[3], &values) // null decodes to nil

				got := def.Params[i]
				if got.Name != name {
					t.Errorf("parameter %d: got %q, want %q", i, got.Name, name)
					continue
				}
				if string(got.Spec.Type) != kind {
					t.Errorf("parameter %q type: got %q, want %q", name, got.Spec.Type, kind)
				}
				if got.Spec.Doc != doc {
					t.Errorf("parameter %q doc:\n got  %q\n want %q", name, got.Spec.Doc, doc)
				}
				if len(got.Spec.Values) != len(values) {
					t.Errorf("parameter %q values: got %v, want %v", name, got.Spec.Values, values)
					continue
				}
				for k := range values {
					if got.Spec.Values[k] != values[k] {
						t.Errorf("parameter %q value %d: got %q, want %q", name, k, got.Spec.Values[k], values[k])
					}
				}
			}
		})
	}

	for name := range catalog.Builtin {
		if !seen[name] {
			t.Errorf("block %q is in the catalog but not the golden", name)
		}
	}
}

// TestEveryBlockSaysWhatItIs: a block with no summary is a block the inspector
// has nothing to say about, and a primitive with no formula is a number with no
// derivation. Both are the whole point of the tool.
func TestEveryBlockSaysWhatItIs(t *testing.T) {
	for name, def := range catalog.Builtin {
		if def.Docs.Summary == "" {
			t.Errorf("%s has no summary", name)
		}
		if catalog.IsPrimitive(def) && def.ParamCount != nil && def.Docs.Formula == "" {
			// A block that carries a parameter count should say where it comes
			// from; one that has none has nothing to derive.
			if def.ParamCount(&catalog.Resolved{P: map[string]any{}}) != 0 {
				t.Errorf("%s counts parameters but gives no formula", name)
			}
		}
	}
}

func names(l catalog.ParamList) []string {
	out := make([]string, len(l))
	for i, e := range l {
		out[i] = e.Name
	}
	return out
}

func mustDecode(t *testing.T, raw json.RawMessage, into any) {
	t.Helper()
	if err := json.Unmarshal(raw, into); err != nil {
		t.Fatalf("bad golden entry %s: %v", raw, err)
	}
}
