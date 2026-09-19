package shapes_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorcad/core/shapes"
)

// Shape patterns against the TypeScript they replace. The mismatch *messages*
// matter as much as the verdicts: they are what a person reads off the canvas
// when a design is wrong. Regenerate with `bun run scripts/golden.ts`.

type patternParse struct {
	Src          string   `json:"src"`
	Atoms        []string `json:"atoms"`
	Instantiated *string  `json:"instantiated"`
	Errors       []string `json:"errors"`
}

type patternError struct {
	Src     string `json:"src"`
	Message string `json:"message"`
}

type patternMatch struct {
	Actual  string   `json:"actual"`
	Pattern string   `json:"pattern"`
	OK      bool     `json:"ok"`
	Batch   string   `json:"batch"`
	Errors  []string `json:"errors"`
}

type patternGolden struct {
	Parse       []patternParse `json:"parse"`
	ParseErrors []patternError `json:"parseErrors"`
	Matches     []patternMatch `json:"matches"`
}

func loadPatterns(t *testing.T) patternGolden {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "patterns.json"))
	if err != nil {
		t.Fatalf("read pattern golden: %v", err)
	}
	var g patternGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("parse pattern golden: %v", err)
	}
	if len(g.Parse) == 0 {
		t.Fatal("pattern golden is empty")
	}
	return g
}

func TestPatternsMatchTypeScript(t *testing.T) {
	g := loadPatterns(t)
	env := loadExpressions(t).Env
	ctx := shapes.EvalCtx{Values: env, Known: knownOf(env)}
	// The TypeScript binds "..." to B and T when instantiating.
	batch := shapes.Shape{shapes.V("B"), shapes.V("T")}

	for _, c := range g.Parse {
		t.Run("parse/"+c.Src, func(t *testing.T) {
			p, err := shapes.ParsePattern(c.Src)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			got := make([]string, len(p.Atoms))
			for i, a := range p.Atoms {
				if a.Kind == shapes.AtomEllipsis {
					got[i] = "..."
					continue
				}
				got[i] = joinParts(a.Parts)
			}
			if !sameStrings(got, c.Atoms) {
				t.Errorf("atoms: got %v, want %v", got, c.Atoms)
			}

			shape, errs := shapes.Instantiate(p, ctx, batch, true)
			if len(errs) == 0 && c.Instantiated != nil {
				if s := shapes.ShapeToString(shape); s != *c.Instantiated {
					t.Errorf("instantiated: got %q, want %q", s, *c.Instantiated)
				}
			}
			if (len(errs) > 0) != (len(c.Errors) > 0) {
				t.Errorf("errors: got %v, want %v", errs, c.Errors)
			}
		})
	}

	for _, c := range g.ParseErrors {
		t.Run("reject/"+c.Src, func(t *testing.T) {
			_, err := shapes.ParsePattern(c.Src)
			if c.Message == "" {
				if err != nil {
					t.Fatalf("rejected %q that TypeScript accepts: %v", c.Src, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("accepted %q, which TypeScript rejects with %q", c.Src, c.Message)
			}
		})
	}

	for _, c := range g.Matches {
		t.Run("match/"+c.Actual+"|"+c.Pattern, func(t *testing.T) {
			ap, err := shapes.ParsePattern(c.Actual)
			if err != nil {
				t.Fatalf("parse actual: %v", err)
			}
			actual, _ := shapes.Instantiate(ap, ctx, nil, true)
			pp, err := shapes.ParsePattern(c.Pattern)
			if err != nil {
				t.Fatalf("parse pattern: %v", err)
			}
			r := shapes.MatchPattern(actual, pp, ctx, env)

			if r.OK != c.OK {
				t.Errorf("ok: got %v, want %v (errors: %v)", r.OK, c.OK, r.Errors)
			}
			if got := shapes.ShapeToString(r.Batch); got != c.Batch {
				t.Errorf("batch: got %q, want %q", got, c.Batch)
			}
			// The message is the product, not a by-product.
			if !sameStrings(r.Errors, c.Errors) {
				t.Errorf("errors:\n got  %v\n want %v", r.Errors, c.Errors)
			}
		})
	}
}

func joinParts(parts []string) string {
	out := ""
	for i, p := range parts {
		if i > 0 {
			out += " "
		}
		out += p
	}
	return out
}

func knownOf(env map[string]float64) map[string]bool {
	known := make(map[string]bool, len(env))
	for k := range env {
		known[k] = true
	}
	return known
}
