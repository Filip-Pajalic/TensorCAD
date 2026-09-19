package report_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/jsonx"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/report"
)

// What the editor actually receives.
//
// A nil Go slice marshals as null, and null is not a list: a panel that asks an
// empty list for its length gets an answer, and one that asks null throws. That
// failure appears in the window rather than in a test, which is why this walks
// the whole report and objects to every null it was not told to expect.

// nullIsMeaningful names the places a null says something rather than being a
// list nobody filled in. Each is a field that is genuinely optional: a finding
// with no hint, a parameter a block does not set, a design with no published
// figure to be checked against.
var nullIsMeaningful = []string{
	".hint",
	".param",
	".port",
	".published",
	".notes",
	".family",
	".label",
	// A resolved parameter's value: a tri-state flag is legitimately null, and
	// so is an object-valued parameter the design left out.
	".p.",
	".default",
	".scaling",
	".rope",
	".variants",
	".ui",
	".defs",
	".graph",
}

func expected(path string) bool {
	for _, suffix := range nullIsMeaningful {
		if strings.HasSuffix(path, strings.TrimSuffix(suffix, ".")) || strings.Contains(path, suffix) {
			return true
		}
	}
	return false
}

// walk reports every null in a decoded document, by the path it sits at.
func walk(value any, path string, out *[]string) {
	switch v := value.(type) {
	case nil:
		if !expected(path) {
			*out = append(*out, path)
		}
	case map[string]any:
		for key, inner := range v {
			walk(inner, path+"."+key, out)
		}
	case []any:
		for i, inner := range v {
			if i > 2 {
				// Three of anything is enough to find a shape problem, and a
				// failure listing five thousand paths helps nobody.
				break
			}
			walk(inner, path+"[]", out)
		}
	}
}

func TestDerivedReportHasNoAccidentalNulls(t *testing.T) {
	for _, name := range []string{"gpt2-small", "llama-3-8b", "mixtral-8x7b", "deepseek-v3", "alexnet"} {
		t.Run(name, func(t *testing.T) {
			derived, err := report.Derive(presets.MustGet(name), analysis.Options{})
			if err != nil {
				t.Fatal(err)
			}
			raw, err := jsonx.Marshal(derived)
			if err != nil {
				t.Fatal(err)
			}
			var decoded any
			if err := json.Unmarshal(raw, &decoded); err != nil {
				t.Fatal(err)
			}

			var nulls []string
			walk(decoded, "", &nulls)
			seen := map[string]bool{}
			for _, path := range nulls {
				if seen[path] {
					continue
				}
				seen[path] = true
				t.Errorf("null at %s; if that is a list nobody filled in, give it an empty one", path)
			}
		})
	}
}

// A design that is wrong is the case the editor most needs to survive, because
// it is where a person is when they need the panels working.
func TestBrokenDesignStillProducesAWholeReport(t *testing.T) {
	doc := presets.MustGet("gpt2-small")
	// A symbol that cannot evaluate: every number downstream becomes NaN.
	doc.Symbols["D"] = mustSymbol(t, `{"kind":"design","value":"D + 1"}`)

	derived, err := report.Derive(doc, analysis.Options{})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := jsonx.Marshal(derived)
	if err != nil {
		t.Fatalf("a broken design could not be reported: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("the report is not JSON: %v", err)
	}
	if len(derived.Report.Findings) == 0 {
		t.Error("nothing was reported about a design with a symbol cycle")
	}
	if !strings.Contains(string(raw), "Symbol cycle") {
		t.Error("the cycle is not named in the report")
	}
}

func mustSymbol(t *testing.T, text string) (def ir.SymbolDef) {
	t.Helper()
	if err := json.Unmarshal([]byte(text), &def); err != nil {
		t.Fatal(err)
	}
	return def
}
