package rules_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
)

// The design rules against the TypeScript, finding for finding.
//
// A rule is a sentence a person reads and acts on, so the message and the hint
// are as much the specification as the severity is: "the cache dominates" and
// "quantize the weights" send someone to different parts of their design. The
// comparison is therefore on the whole finding, including the rounding in its
// percentages. Regenerate with `go run ./cmd/golden`.

type goldenFinding struct {
	Rule     string `json:"rule"`
	Severity string `json:"severity"`
	Path     string `json:"path"`
	Port     string `json:"port"`
	Param    string `json:"param"`
	Message  string `json:"message"`
	Hint     string `json:"hint"`
}

type goldenRuleCase struct {
	Label    string          `json:"label"`
	OK       bool            `json:"ok"`
	Counts   map[string]int  `json:"counts"`
	Findings []goldenFinding `json:"findings"`
}

func loadRules(t *testing.T, name string) []goldenRuleCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "rules", name+".json"))
	if err != nil {
		t.Fatalf("read rules golden %s: %v", name, err)
	}
	var g struct {
		Cases []goldenRuleCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse rules golden %s: %v", name, err)
	}
	if len(g.Cases) == 0 {
		t.Fatalf("rules golden %s is empty", name)
	}
	return g.Cases
}

func TestFindingsMatchTheGoldens(t *testing.T) {
	opts := variants()
	for _, name := range presetNames(t) {
		for _, c := range loadRules(t, name) {
			t.Run(name+"/"+c.Label, func(t *testing.T) {
				o, ok := opts[c.Label]
				if !ok {
					t.Fatalf("no Go variant named %q", c.Label)
				}
				report, err := rules.Validate(loadDoc(t, name), o)
				if err != nil {
					t.Fatal(err)
				}

				if report.OK != c.OK {
					t.Errorf("ok: got %v, want %v", report.OK, c.OK)
				}
				for _, sev := range []string{"error", "warning", "info"} {
					if report.Counts[sev] != c.Counts[sev] {
						t.Errorf("counts[%s]: got %d, want %d", sev, report.Counts[sev], c.Counts[sev])
					}
				}

				// The order is part of the answer: it is the order the panel
				// lists them in, and both engines sort by severity and then by
				// path, stably.
				if len(report.Findings) != len(c.Findings) {
					t.Errorf("got %d findings, want %d", len(report.Findings), len(c.Findings))
					for k, f := range report.Findings {
						t.Logf("  go   [%d] %s/%s %s: %s", k, f.Severity, f.Rule, f.Path, f.Message)
					}
					for k, f := range c.Findings {
						t.Logf("  ts   [%d] %s/%s %s: %s", k, f.Severity, f.Rule, f.Path, f.Message)
					}
					return
				}
				for k := range report.Findings {
					got, want := report.Findings[k], c.Findings[k]
					if got.Rule != want.Rule || got.Severity != want.Severity ||
						got.Path != want.Path || got.Port != want.Port || got.Param != want.Param ||
						got.Message != want.Message || got.Hint != want.Hint {
						t.Errorf("finding %d:\n got  %+v\n want %+v", k, got, want)
					}
				}
			})
		}
	}
}

// TestEveryPresetPassesItsOwnRules is the standing claim the presets make.
// They are the regression suite, so a preset that no longer passes the check is
// a broken suite whatever the goldens say it produces.
func TestEveryPresetPassesItsOwnRules(t *testing.T) {
	for _, name := range presetNames(t) {
		t.Run(name, func(t *testing.T) {
			report, err := rules.Validate(loadDoc(t, name), analysis.Options{})
			if err != nil {
				t.Fatal(err)
			}
			for _, f := range report.Findings {
				if f.Severity == "error" {
					t.Errorf("%s %s: %s", f.Rule, f.Path, f.Message)
				}
			}
		})
	}
}

// TestEveryRuleHasIdentityKeeps the list honest: a rule with no id or no
// description cannot be excluded, documented or pointed at from the panel.
func TestEveryRuleHasIdentity(t *testing.T) {
	seen := map[string]bool{}
	for _, r := range rules.Rules {
		switch {
		case r.ID == "":
			t.Error("a rule has no id")
		case seen[r.ID]:
			t.Errorf("two rules share the id %q", r.ID)
		case r.Title == "" || r.Description == "":
			t.Errorf("rule %q has no title or description", r.ID)
		case r.Run == nil:
			t.Errorf("rule %q does nothing", r.ID)
		}
		seen[r.ID] = true
	}
	if len(rules.Rules) != 21 {
		t.Errorf("got %d rules, want 21; add the new one to the docs too", len(rules.Rules))
	}
}

// ---------------------------------------------------------------------------
// Shared with the analysis parity test, which lives in another package.
// ---------------------------------------------------------------------------

func f(v float64) *float64 { return &v }
func b(v bool) *bool       { return &v }
func i(v int) *int         { return &v }

// variants must stay in step with OperatingPoints in golden/cases.go.
func variants() map[string]analysis.Options {
	return map[string]analysis.Options{
		"default": {},
		"sharded": {
			Dtype:          "fp8",
			InferenceDtype: "fp8",
			Recompute:      "full",
			Optimizer:      "adamw8bit",
			Parallel: &analysis.PartialParallel{
				TP: f(8), PP: f(2), DP: f(4), Zero: i(3), SequenceParallel: b(true),
			},
			GPUs:        f(64),
			Concurrency: f(32),
			Tokens:      f(15e12),
			MFU:         f(0.4),
		},
		"eager": {
			T: f(8192), B: f(4), Flash: b(false),
			Recompute: "selective", KvDtype: "fp8", GPUs: f(8),
		},
	}
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

// TestBrokenDesignsMatchTheGoldens covers the rules a correct preset never
// reaches.
//
// Eight of the eighteen rules only fire on a mistake, and their messages are
// what a person sees when their design is broken — the most important sentences
// the engine writes. The documents live in the golden file next to the findings
// they produce, so a fixture and its expectations cannot drift apart.
func TestBrokenDesignsMatchTheGoldens(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "broken.json"))
	if err != nil {
		t.Fatalf("read broken golden: %v", err)
	}
	var g struct {
		Cases []struct {
			Name     string          `json:"name"`
			Note     string          `json:"note"`
			Doc      json.RawMessage `json:"doc"`
			OK       bool            `json:"ok"`
			Counts   map[string]int  `json:"counts"`
			Findings []goldenFinding `json:"findings"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse broken golden: %v", err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("broken golden is empty")
	}

	fired := map[string]bool{}
	for _, c := range g.Cases {
		t.Run(c.Name, func(t *testing.T) {
			var doc ir.Doc
			if err := json.Unmarshal(c.Doc, &doc); err != nil {
				t.Fatalf("parse %s: %v", c.Name, err)
			}
			report, err := rules.Validate(&doc, analysis.Options{})
			if err != nil {
				t.Fatal(err)
			}
			if report.OK != c.OK {
				t.Errorf("ok: got %v, want %v", report.OK, c.OK)
			}
			for _, sev := range []string{"error", "warning", "info"} {
				if report.Counts[sev] != c.Counts[sev] {
					t.Errorf("counts[%s]: got %d, want %d", sev, report.Counts[sev], c.Counts[sev])
				}
			}
			if len(report.Findings) != len(c.Findings) {
				t.Errorf("got %d findings, want %d", len(report.Findings), len(c.Findings))
				for k, f := range report.Findings {
					t.Logf("  go   [%d] %s/%s %s: %s", k, f.Severity, f.Rule, f.Path, f.Message)
				}
				for k, f := range c.Findings {
					t.Logf("  ts   [%d] %s/%s %s: %s", k, f.Severity, f.Rule, f.Path, f.Message)
				}
				return
			}
			for k := range report.Findings {
				got, want := report.Findings[k], c.Findings[k]
				if got.Rule != want.Rule || got.Severity != want.Severity ||
					got.Path != want.Path || got.Port != want.Port || got.Param != want.Param ||
					got.Message != want.Message || got.Hint != want.Hint {
					t.Errorf("finding %d:\n got  %+v\n want %+v", k, got, want)
				}
			}
		})
		for _, f := range c.Findings {
			fired[f.Rule] = true
		}
	}

	// The fixtures exist to reach the rules the presets cannot. If one stops
	// reaching its rule, the rule silently stops being tested anywhere.
	for _, id := range []string{"symbols", "graph", "user-blocks", "dangling-output",
		"published-drift", "active-params-drift", "window-vs-context", "shape"} {
		if !fired[id] {
			t.Errorf("no broken design reaches the %q rule any more", id)
		}
	}
}
