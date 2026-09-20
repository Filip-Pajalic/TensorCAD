package plan_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/plan"
	"github.com/tensorcad/core/presets"
)

// The planner against the goldens, plan for plan and in order.
//
// The ranking is as much the answer as the numbers are. A planner that starts
// recommending a pipeline where it used to recommend sharding the optimizer has
// changed its advice, and that is a thing to review rather than to discover.
// Regenerate with `go run ./cmd/golden`.

type goldenPlan struct {
	Label      string           `json:"label"`
	Preset     string           `json:"preset"`
	Seq        float64          `json:"seq"`
	Cluster    plan.Request     `json:"cluster"`
	Budget     float64          `json:"budget"`
	Considered int              `json:"considered"`
	Fits       []plan.Candidate `json:"fits"`
	Closest    *plan.Candidate  `json:"closest"`
	Notes      []string         `json:"notes"`
}

func loadPlans(t *testing.T) []goldenPlan {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "plans.json"))
	if err != nil {
		t.Fatalf("read the plan goldens: %v", err)
	}
	var file struct {
		Cases []goldenPlan `json:"cases"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatalf("parse the plan goldens: %v", err)
	}
	if len(file.Cases) == 0 {
		t.Fatal("no plan goldens")
	}
	return file.Cases
}

func TestPlansMatchTheGoldens(t *testing.T) {
	for _, c := range loadPlans(t) {
		t.Run(c.Label, func(t *testing.T) {
			seq := c.Seq
			got, err := plan.Search(presets.MustGet(c.Preset),
				analysis.Options{T: &seq, Hardware: "h100-sxm"}, c.Cluster)
			if err != nil {
				t.Fatalf("search: %v", err)
			}
			if got.Considered != c.Considered {
				t.Errorf("priced %d plans, want %d", got.Considered, c.Considered)
			}
			if got.Budget != c.Budget {
				t.Errorf("budget %v, want %v", got.Budget, c.Budget)
			}
			if len(got.Fits) != len(c.Fits) {
				t.Fatalf("%d plans fit, want %d", len(got.Fits), len(c.Fits))
			}
			for i := range c.Fits {
				g, w := got.Fits[i], c.Fits[i]
				if g.Summary != w.Summary {
					t.Errorf("plan %d is %q, want %q", i, g.Summary, w.Summary)
				}
				if g.PerGpu.Total != w.PerGpu.Total {
					t.Errorf("%s holds %s, want %s", w.Summary,
						analysis.FormatBytes(g.PerGpu.Total), analysis.FormatBytes(w.PerGpu.Total))
				}
				if g.Used != w.Used {
					t.Errorf("%s uses %v of the budget, want %v", w.Summary, g.Used, w.Used)
				}
				if !sameStrings(g.Notes, w.Notes) {
					t.Errorf("%s notes: got %q, want %q", w.Summary, g.Notes, w.Notes)
				}
			}
			switch {
			case c.Closest == nil && got.Closest != nil:
				t.Errorf("a nearest miss appeared: %q", got.Closest.Summary)
			case c.Closest != nil && got.Closest == nil:
				t.Errorf("the nearest miss went away; it was %q", c.Closest.Summary)
			case c.Closest != nil && got.Closest.Summary != c.Closest.Summary:
				t.Errorf("nearest miss is %q, want %q", got.Closest.Summary, c.Closest.Summary)
			}
			if !sameStrings(got.Notes, c.Notes) {
				t.Errorf("notes: got %q, want %q", got.Notes, c.Notes)
			}
		})
	}
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
