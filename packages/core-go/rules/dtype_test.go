package rules_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
)

// The element-type check exists for one thing: the wiring error whose shapes
// agree. So the tests are the two halves of that — it fires on the mistake, and
// it is silent on every design in the library.

// A design that feeds token ids straight into a matmul. The shapes agree
// exactly — `B T D` into a linear that declares `... in_features` — which is
// why nothing but this rule can object to it.
const idsIntoMatmul = `{
  "version": 1,
  "meta": { "name": "ids-into-matmul" },
  "symbols": { "B": {"kind":"runtime","default":1}, "T": {"kind":"runtime","default":8},
               "D": {"kind":"design","value":16} },
  "graph": {
    "nodes": [
      { "id": "tokens", "type": "input", "params": { "shape": "B T D", "dtype": "int64" } },
      { "id": "proj", "type": "linear", "params": { "in_features": "D", "out_features": "D", "bias": false } },
      { "id": "out", "type": "output" }
    ],
    "edges": [["tokens:x", "proj:x"], ["proj:y", "out:x"]]
  }
}`

// The same design with the input declared as activations, which is the fix the
// hint points at. Nothing else about it changes.
const actsIntoMatmul = `{
  "version": 1,
  "meta": { "name": "acts-into-matmul" },
  "symbols": { "B": {"kind":"runtime","default":1}, "T": {"kind":"runtime","default":8},
               "D": {"kind":"design","value":16} },
  "graph": {
    "nodes": [
      { "id": "tokens", "type": "input", "params": { "shape": "B T D", "dtype": "bf16" } },
      { "id": "proj", "type": "linear", "params": { "in_features": "D", "out_features": "D", "bias": false } },
      { "id": "out", "type": "output" }
    ],
    "edges": [["tokens:x", "proj:x"], ["proj:y", "out:x"]]
  }
}`

func dtypeFindings(t *testing.T, source string) []rules.Finding {
	t.Helper()
	var doc ir.Doc
	if err := json.Unmarshal([]byte(source), &doc); err != nil {
		t.Fatalf("parse: %v", err)
	}
	report, err := rules.Validate(&doc, analysis.Options{})
	if err != nil {
		t.Fatalf("validate: %v", err)
	}
	var out []rules.Finding
	for _, f := range report.Findings {
		if f.Rule == "dtype" {
			out = append(out, f)
		}
	}
	return out
}

func TestDtypeCatchesIdsInAMatmul(t *testing.T) {
	found := dtypeFindings(t, idsIntoMatmul)
	if len(found) != 1 {
		t.Fatalf("got %d dtype findings, want 1: %+v", len(found), found)
	}
	f := found[0]
	if f.Severity != "error" {
		t.Errorf("severity: got %q, want error", f.Severity)
	}
	if f.Path != "proj" || f.Port != "x" {
		t.Errorf("pointed at %q port %q, want proj/x", f.Path, f.Port)
	}
	// The message has to name both sides, or it says a thing is wrong without
	// saying what would have been right.
	if !strings.Contains(f.Message, "integer") || !strings.Contains(f.Message, "tokens:x") {
		t.Errorf("message does not name what arrived from where: %q", f.Message)
	}
}

func TestDtypeIsQuietWhenTheWiringIsRight(t *testing.T) {
	if found := dtypeFindings(t, actsIntoMatmul); len(found) != 0 {
		t.Errorf("fired on a correct design: %+v", found)
	}
}

// The check that matters most. A rule that fires on a shipped design is a rule
// that teaches people to read past it, and the walk behind this one is timid
// precisely so that cannot happen: an input it cannot resolve makes the whole
// block unknown rather than a guess.
func TestDtypeIsQuietOnEveryPreset(t *testing.T) {
	names, err := presets.Names()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			doc, err := presets.Get(name)
			if err != nil {
				t.Fatal(err)
			}
			report, err := rules.Validate(doc, analysis.Options{})
			if err != nil {
				t.Fatal(err)
			}
			for _, f := range report.Findings {
				if f.Rule == "dtype" {
					t.Errorf("%s %s — %s", f.Path, f.Port, f.Message)
				}
			}
		})
	}
}
