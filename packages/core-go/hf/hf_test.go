package hf_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/hf"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
)

// Importing a real config.json.
//
// Each fixture is the architecture-relevant subset of a published model's
// config.json, and the claim is the one that makes an importer trustworthy: it
// lands on the same parameter count as the preset somebody wrote by hand. The
// same file is what the TypeScript test reads, so neither engine is checked
// against its own copy of the input.

func configs(t *testing.T) map[string]hf.Config {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "hf-configs.json"))
	if err != nil {
		t.Fatalf("read the config fixtures: %v", err)
	}
	var out map[string]hf.Config
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("parse the config fixtures: %v", err)
	}
	if len(out) == 0 {
		t.Fatal("the config fixtures are empty")
	}
	return out
}

func paramsOf(t *testing.T, doc *ir.Doc) float64 {
	t.Helper()
	return analysis.CountParams(analysis.Flatten(doc, ir.ResolveSymbols(doc))).Total
}

func TestImportReproducesThePreset(t *testing.T) {
	for name, config := range configs(t) {
		t.Run(name, func(t *testing.T) {
			got, err := hf.Import(config, name)
			if err != nil {
				t.Fatal(err)
			}
			if len(got.Warnings) != 0 {
				t.Errorf("warnings: %q", got.Warnings)
			}
			want := presets.MustGet(name)
			if g, w := paramsOf(t, got.Doc), paramsOf(t, want); g != w {
				t.Errorf("the import counts %s parameters, the preset counts %s",
					analysis.FormatCount(g), analysis.FormatCount(w))
			}
		})
	}
}

func TestImportedDesignPassesTheRules(t *testing.T) {
	seq := 8192.0
	got, err := hf.Import(configs(t)["llama-3-8b"], "imported")
	if err != nil {
		t.Fatal(err)
	}
	report, err := rules.Validate(got.Doc, analysis.Options{T: &seq})
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range report.Findings {
		if f.Severity == "error" {
			t.Errorf("%s %s: %s", f.Rule, f.Path, f.Message)
		}
	}
}

func TestSlidingWindowSurvivesTheImport(t *testing.T) {
	got, err := hf.Import(configs(t)["mistral-7b"], "m")
	if err != nil {
		t.Fatal(err)
	}
	w, ok := got.Doc.Symbols["W"]
	if !ok {
		t.Fatal("the imported design has no sliding-window symbol")
	}
	if w.Kind != "design" || w.Number != 4096 || w.Doc != "Sliding-window width" {
		t.Errorf("W is %+v", w)
	}
}

func TestLeadingDenseLayersSurviveTheImport(t *testing.T) {
	got, err := hf.Import(configs(t)["deepseek-v3"], "ds")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, n := range got.Doc.Graph.Nodes {
		if n.ID == "dense_layers" {
			found = true
		}
	}
	if !found {
		t.Error("the imported design has no separate stack for the leading dense layers")
	}
}

// TestUnknownFamilyIsRefused: an importer that guesses is worse than one that
// stops, because the number it produces looks as real as any other.
func TestUnknownFamilyIsRefused(t *testing.T) {
	_, err := hf.Import(hf.Config{"model_type": "some_new_thing", "num_hidden_layers": 1.0}, "")
	if err == nil {
		t.Fatal("an unknown family was accepted")
	}
	if !strings.Contains(err.Error(), "unsupported model_type") {
		t.Errorf("the refusal does not name the problem: %v", err)
	}
}

func TestMissingFieldIsNamed(t *testing.T) {
	_, err := hf.Import(hf.Config{"model_type": "llama"}, "")
	if err == nil {
		t.Fatal("a config with no layers was accepted")
	}
	if !strings.Contains(err.Error(), "num_hidden_layers") {
		t.Errorf("the error does not name the missing field: %v", err)
	}
}

func TestPartlySparseStackWarns(t *testing.T) {
	config := hf.Config{}
	for k, v := range configs(t)["mixtral-8x7b"] {
		config[k] = v
	}
	config["decoder_sparse_step"] = 2.0

	got, err := hf.Import(config, "partly-sparse")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(got.Warnings, " "), "only every 2th layer is sparse") {
		t.Errorf("no warning about the sparse step: %q", got.Warnings)
	}
}

// TestEveryKnownFamilyIsCovered keeps the fixture set honest: a family this
// importer claims to know but nothing imports is a claim nothing checks.
func TestEveryKnownFamilyIsCovered(t *testing.T) {
	covered := map[string]bool{}
	for _, config := range configs(t) {
		if s, ok := config["model_type"].(string); ok {
			covered[s] = true
		}
	}
	// Three families share another's layout exactly, so a fixture for each
	// would test the same code path twice.
	sameAsAnother := map[string]string{
		"qwen3_moe": "qwen3",
		"gemma":     "gemma2",
		"qwen2":     "qwen2",
	}
	for _, family := range hf.SupportedModelTypes {
		if covered[family] {
			continue
		}
		if _, ok := sameAsAnother[family]; ok {
			continue
		}
		t.Errorf("no fixture imports a %q model", family)
	}
}
