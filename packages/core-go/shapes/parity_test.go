package shapes_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorcad/core/shapes"
)

// The expression algebra against the TypeScript it replaces. A symbol table
// only exercises the paths its presets happen to take; these are the corners.
// Regenerate with `go run ./cmd/golden`.

type exprCase struct {
	Src     string   `json:"src"`
	Text    string   `json:"text"`
	Value   *float64 `json:"value"`
	Symbols []string `json:"symbols"`
}

type exprError struct {
	Src     string `json:"src"`
	Message string `json:"message"`
	Text    string `json:"text"`
}

type exprGolden struct {
	Env    map[string]float64 `json:"env"`
	OK     []exprCase         `json:"ok"`
	Errors []exprError        `json:"errors"`
}

func loadExpressions(t *testing.T) exprGolden {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "expressions.json"))
	if err != nil {
		t.Fatalf("read expression golden: %v", err)
	}
	var g exprGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("parse expression golden: %v", err)
	}
	if len(g.OK) == 0 {
		t.Fatal("expression golden is empty")
	}
	return g
}

func ctxOf(g exprGolden) shapes.EvalCtx {
	known := make(map[string]bool, len(g.Env))
	for k := range g.Env {
		known[k] = true
	}
	return shapes.EvalCtx{Values: g.Env, Known: known}
}

func TestExpressionsMatchTheGoldens(t *testing.T) {
	g := loadExpressions(t)
	ctx := ctxOf(g)

	for _, c := range g.OK {
		t.Run(c.Src, func(t *testing.T) {
			sym, err := shapes.EvalExpr(c.Src, ctx)
			if err != nil {
				t.Fatalf("evaluate: %v", err)
			}
			// The printed form is the shape label on the canvas, so it has to
			// agree character for character, not just numerically.
			if got := sym.String(); got != c.Text {
				t.Errorf("text: got %q, want %q", got, c.Text)
			}
			n, ok := sym.ToNumber(g.Env)
			switch {
			case c.Value == nil && ok:
				t.Errorf("value: got %g, want indeterminate", n)
			case c.Value != nil && !ok:
				t.Errorf("value: indeterminate, want %g", *c.Value)
			case c.Value != nil && !closeEnough(n, *c.Value):
				t.Errorf("value: got %g, want %g", n, *c.Value)
			}
			if got := sym.Symbols(); !sameStrings(got, c.Symbols) {
				t.Errorf("symbols: got %v, want %v", got, c.Symbols)
			}
		})
	}
}

func TestExpressionErrorsMatchTheGoldens(t *testing.T) {
	g := loadExpressions(t)
	ctx := ctxOf(g)

	for _, c := range g.Errors {
		t.Run(c.Src, func(t *testing.T) {
			sym, err := shapes.EvalExpr(c.Src, ctx)
			if c.Message == "" {
				// TypeScript accepted it, so Go has to as well.
				if err != nil {
					t.Fatalf("rejected %q that TypeScript accepts: %v", c.Src, err)
				}
				if got := sym.String(); got != c.Text {
					t.Errorf("text: got %q, want %q", got, c.Text)
				}
				return
			}
			if err == nil {
				t.Fatalf("accepted %q, which TypeScript rejects with %q", c.Src, c.Message)
			}
		})
	}
}

// closeEnough compares two doubles that took different routes to the same
// arithmetic. Dimensions come out exact; sqrt and log2 are the two places a
// last-bit difference is possible and does not mean anything.
func closeEnough(a, b float64) bool {
	if a == b {
		return true
	}
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	scale := math.Max(math.Abs(a), math.Abs(b))
	return math.Abs(a-b) <= 1e-12*math.Max(1, scale)
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
