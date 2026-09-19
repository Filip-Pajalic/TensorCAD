package codegen_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tensorcad/core/codegen"
	"github.com/tensorcad/core/ir"
)

// The generated PyTorch against the TypeScript, byte for byte.
//
// The strongest test in the set. A model.py is the whole engine's output as one
// artifact — every parameter it resolved, every composite it expanded, every
// number it formatted — and a file that differs by one character is a file that
// was generated differently. Three variants per preset, because each takes a
// different path through the emitter. Regenerate with
// `bun run scripts/golden.ts`.

func f(v float64) *float64 { return &v }

// variants must stay in step with CODEGEN_VARIANTS in scripts/golden.ts.
func variants() map[string]codegen.Options {
	return map[string]codegen.Options{
		"default": {},
		"dense":   {MoeDispatch: "dense"},
		"bare":    {InitStd: f(0), NoSmokeTest: true, ClassName: "Net"},
	}
}

type goldenCase struct {
	Label    string   `json:"label"`
	Warnings []string `json:"warnings"`
	Model    string   `json:"model"`
}

func TestGeneratedTorchMatchesTypeScript(t *testing.T) {
	opts := variants()
	for _, name := range presetNames(t) {
		for _, c := range loadCodegen(t, name) {
			t.Run(name+"/"+c.Label, func(t *testing.T) {
				o, ok := opts[c.Label]
				if !ok {
					t.Fatalf("no Go variant named %q", c.Label)
				}
				got := codegen.GenerateTorch(loadDoc(t, name), o)

				if len(got.Warnings) != len(c.Warnings) {
					t.Errorf("warnings: got %d, want %d\n go   %q\n want %q",
						len(got.Warnings), len(c.Warnings), got.Warnings, c.Warnings)
				} else {
					for i := range got.Warnings {
						if got.Warnings[i] != c.Warnings[i] {
							t.Errorf("warning %d: got %q, want %q", i, got.Warnings[i], c.Warnings[i])
						}
					}
				}

				var model string
				for _, file := range got.Files {
					if file.Path == "model.py" {
						model = file.Contents
					}
				}
				if model != c.Model {
					t.Error(firstDifference(model, c.Model))
				}
			})
		}
	}
}

// TestDesignTravelsWithTheCode pins the second file: a generated model is not
// much use without the design it came from, and regenerating it is the only
// supported way to change it.
func TestDesignTravelsWithTheCode(t *testing.T) {
	name := presetNames(t)[0]
	got := codegen.GenerateTorch(loadDoc(t, name), codegen.Options{})
	if len(got.Files) != 2 {
		t.Fatalf("got %d files, want 2", len(got.Files))
	}
	if got.Files[1].Path != "design.tensorcad.json" {
		t.Errorf("second file is %q", got.Files[1].Path)
	}
	var doc ir.Doc
	if err := json.Unmarshal([]byte(got.Files[1].Contents), &doc); err != nil {
		t.Fatalf("the emitted design does not parse: %v", err)
	}
	if doc.Meta.Name == "" || len(doc.Graph.Nodes) == 0 {
		t.Error("the emitted design is empty")
	}
}

// firstDifference reports where two files diverge, with the lines around it.
// A whole model.py in a failure message is unreadable; the line that changed
// and its neighbours is what a person needs.
func firstDifference(got, want string) string {
	g := strings.Split(got, "\n")
	w := strings.Split(want, "\n")
	for i := 0; i < len(g) || i < len(w); i++ {
		gl, wl := "<end of file>", "<end of file>"
		if i < len(g) {
			gl = g[i]
		}
		if i < len(w) {
			wl = w[i]
		}
		if gl == wl {
			continue
		}
		var b strings.Builder
		b.WriteString("model.py differs at line ")
		b.WriteString(itoa(i + 1))
		b.WriteString(" of ")
		b.WriteString(itoa(len(w)))
		b.WriteString("\n")
		for k := max(0, i-3); k < i; k++ {
			b.WriteString("       " + w[k] + "\n")
		}
		b.WriteString("  go   " + gl + "\n")
		b.WriteString("  ts   " + wl + "\n")
		return b.String()
	}
	return "the files differ but no line does; check the trailing newline"
}

func itoa(v int) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func presetNames(t *testing.T) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "presets.json"))
	if err != nil {
		t.Fatalf("read preset index: %v", err)
	}
	var names []string
	if err := json.Unmarshal(raw, &names); err != nil {
		t.Fatalf("parse preset index: %v", err)
	}
	return names
}

func loadDoc(t *testing.T, name string) *ir.Doc {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "presets", name+".json"))
	if err != nil {
		t.Fatalf("read preset %s: %v", name, err)
	}
	var doc ir.Doc
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("parse preset %s: %v", name, err)
	}
	return &doc
}

func loadCodegen(t *testing.T, name string) []goldenCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "codegen", name+".json"))
	if err != nil {
		t.Fatalf("read codegen golden %s: %v", name, err)
	}
	var g struct {
		Cases []goldenCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse codegen golden %s: %v", name, err)
	}
	if len(g.Cases) == 0 {
		t.Fatalf("codegen golden %s is empty", name)
	}
	return g.Cases
}
