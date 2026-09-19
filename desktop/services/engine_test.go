package services_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/desktop/services"
)

// The engine service is a boundary, and a boundary is where an answer gets
// quietly lost: an operating point that does not survive being written down, a
// result that marshals to something the window cannot read. These check the
// round trip rather than the arithmetic, which the engine's own tests cover.

func documentOf(t *testing.T, name string) string {
	t.Helper()
	doc, err := presets.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestAnalyzeCrossesTheBoundaryIntact(t *testing.T) {
	engine := services.NewEngineService()
	document := documentOf(t, "gpt2-small")

	text, err := engine.Analyze(document, "")
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Name   string `json:"name"`
		Params struct {
			Total float64 `json:"total"`
		} `json:"params"`
		Options struct {
			T        float64 `json:"T"`
			Hardware struct {
				ID string `json:"id"`
			} `json:"hardware"`
		} `json:"options"`
	}
	if err := json.Unmarshal([]byte(text), &got); err != nil {
		t.Fatalf("the window could not read the result: %v", err)
	}

	want, err := analysis.Analyze(presets.MustGet("gpt2-small"), analysis.Options{}, analysis.Inputs{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Params.Total != want.Params.Total {
		t.Errorf("parameters: got %g, want %g", got.Params.Total, want.Params.Total)
	}
	if got.Name != "gpt2-small" {
		t.Errorf("name: got %q", got.Name)
	}
	if got.Options.Hardware.ID == "" {
		t.Error("the resolved hardware profile did not come through")
	}
}

// TestOperatingPointSurvivesTheCrossing: the operating point is the whole
// reason the same design gives different numbers, so every field of it has to
// arrive. A dropped one would read as a default and look plausible.
func TestOperatingPointSurvivesTheCrossing(t *testing.T) {
	engine := services.NewEngineService()
	document := documentOf(t, "llama-3-8b")

	point := `{
		"T": 8192, "B": 4, "dtype": "fp8", "inferenceDtype": "fp8", "kvDtype": "bf16",
		"hardware": "a100-80", "gpus": 64, "optimizer": "adamw8bit", "recompute": "full",
		"flash": false, "tokens": 15000000000000, "mfu": 0.4, "decodeEfficiency": 0.25,
		"concurrency": 32,
		"parallel": {"dp": 4, "tp": 8, "pp": 2, "ep": 1, "zero": 3, "sequenceParallel": true}
	}`
	text, err := engine.Analyze(document, point)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Options struct {
			T                float64 `json:"T"`
			B                float64 `json:"B"`
			Dtype            string  `json:"dtype"`
			InferenceDtype   string  `json:"inferenceDtype"`
			KvDtype          string  `json:"kvDtype"`
			GPUs             float64 `json:"gpus"`
			Optimizer        string  `json:"optimizer"`
			Recompute        string  `json:"recompute"`
			Flash            bool    `json:"flash"`
			Tokens           float64 `json:"tokens"`
			MFU              float64 `json:"mfu"`
			DecodeEfficiency float64 `json:"decodeEfficiency"`
			Concurrency      float64 `json:"concurrency"`
			Hardware         struct {
				ID string `json:"id"`
			} `json:"hardware"`
			Parallel struct {
				DP               float64 `json:"dp"`
				TP               float64 `json:"tp"`
				PP               float64 `json:"pp"`
				Zero             int     `json:"zero"`
				SequenceParallel bool    `json:"sequenceParallel"`
			} `json:"parallel"`
		} `json:"options"`
	}
	if err := json.Unmarshal([]byte(text), &got); err != nil {
		t.Fatal(err)
	}
	o := got.Options
	for _, c := range []struct {
		label string
		got   any
		want  any
	}{
		{"T", o.T, 8192.0}, {"B", o.B, 4.0},
		{"dtype", o.Dtype, "fp8"}, {"inferenceDtype", o.InferenceDtype, "fp8"},
		{"kvDtype", o.KvDtype, "bf16"}, {"hardware", o.Hardware.ID, "a100-80"},
		{"gpus", o.GPUs, 64.0}, {"optimizer", o.Optimizer, "adamw8bit"},
		{"recompute", o.Recompute, "full"}, {"flash", o.Flash, false},
		{"tokens", o.Tokens, 15e12}, {"mfu", o.MFU, 0.4},
		{"decodeEfficiency", o.DecodeEfficiency, 0.25}, {"concurrency", o.Concurrency, 32.0},
		{"parallel.dp", o.Parallel.DP, 4.0}, {"parallel.tp", o.Parallel.TP, 8.0},
		{"parallel.pp", o.Parallel.PP, 2.0}, {"parallel.zero", o.Parallel.Zero, 3},
		{"parallel.sequenceParallel", o.Parallel.SequenceParallel, true},
	} {
		if c.got != c.want {
			t.Errorf("%s: got %v, want %v", c.label, c.got, c.want)
		}
	}
}

// TestFlashFalseIsNotTheSameAsUnset is the reason every option is a pointer: a
// boolean that defaults to true cannot tell "off" from "not mentioned" unless
// the absence is representable.
func TestFlashFalseIsNotTheSameAsUnset(t *testing.T) {
	engine := services.NewEngineService()
	document := documentOf(t, "gpt2-small")

	off, err := engine.Analyze(document, `{"flash": false}`)
	if err != nil {
		t.Fatal(err)
	}
	unset, err := engine.Analyze(document, `{}`)
	if err != nil {
		t.Fatal(err)
	}
	if flashOf(t, off) != false {
		t.Error("flash: false arrived as true")
	}
	if flashOf(t, unset) != true {
		t.Error("an unmentioned flash did not default to true")
	}
}

func flashOf(t *testing.T, text string) bool {
	t.Helper()
	var got struct {
		Options struct {
			Flash bool `json:"flash"`
		} `json:"options"`
	}
	if err := json.Unmarshal([]byte(text), &got); err != nil {
		t.Fatal(err)
	}
	return got.Options.Flash
}

func TestValidateReturnsFindingsAndTheAnalysisTogether(t *testing.T) {
	engine := services.NewEngineService()
	text, err := engine.Validate(documentOf(t, "gpt2-small"), "")
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		OK       bool           `json:"ok"`
		Counts   map[string]int `json:"counts"`
		Findings []struct {
			Rule     string `json:"rule"`
			Severity string `json:"severity"`
			Message  string `json:"message"`
		} `json:"findings"`
		Analysis struct {
			Params struct {
				Total float64 `json:"total"`
			} `json:"params"`
		} `json:"analysis"`
	}
	if err := json.Unmarshal([]byte(text), &got); err != nil {
		t.Fatal(err)
	}
	if !got.OK {
		t.Errorf("gpt2-small does not pass its own rules: %+v", got.Findings)
	}
	if got.Analysis.Params.Total != 124439808 {
		t.Errorf("the analysis did not come back with the findings: %g", got.Analysis.Params.Total)
	}
	if got.Counts == nil {
		t.Error("no counts")
	}
}

func TestGenerateTorchCrossesWhole(t *testing.T) {
	engine := services.NewEngineService()
	text, err := engine.GenerateTorch(documentOf(t, "gpt2-small"), `{"moeDispatch":"dense"}`)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Files []struct {
			Path     string `json:"path"`
			Contents string `json:"contents"`
		} `json:"files"`
		Warnings []string `json:"warnings"`
	}
	if err := json.Unmarshal([]byte(text), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Files) != 2 {
		t.Fatalf("got %d files", len(got.Files))
	}
	if !strings.Contains(got.Files[0].Contents, "class Gpt2Small(nn.Module):") {
		t.Error("the generated model does not define its class")
	}
	if !strings.HasSuffix(got.Files[0].Contents, "\n") {
		t.Error("the generated file lost its trailing newline")
	}
}

func TestExplainAndCatalogAndPresets(t *testing.T) {
	engine := services.NewEngineService()

	names, err := engine.Presets()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) < 20 {
		t.Errorf("the library has %d presets", len(names))
	}

	doc, err := engine.Preset("gpt2-small")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(doc, `"gpt2-small"`) {
		t.Error("the preset did not come through")
	}

	text, err := engine.Explain(doc, "layers/block/attn", "")
	if err != nil {
		t.Fatal(err)
	}
	var e struct {
		Type string `json:"type"`
		Docs struct {
			Summary string `json:"summary"`
		} `json:"docs"`
		Contributes struct {
			Params float64 `json:"params"`
		} `json:"contributes"`
	}
	if err := json.Unmarshal([]byte(text), &e); err != nil {
		t.Fatal(err)
	}
	if e.Type != "gqa_attention" {
		t.Errorf("explained a %q", e.Type)
	}
	if e.Docs.Summary == "" {
		t.Error("the explanation carries no summary")
	}
	if e.Contributes.Params <= 0 {
		t.Error("the explanation counts no parameters")
	}

	catalogText, err := engine.Catalog()
	if err != nil {
		t.Fatal(err)
	}
	var blocks []struct {
		Type   string `json:"type"`
		Kind   string `json:"kind"`
		Params map[string]struct {
			Type string `json:"type"`
			Doc  string `json:"doc"`
		} `json:"params"`
		ParamOrder []string `json:"paramOrder"`
	}
	if err := json.Unmarshal([]byte(catalogText), &blocks); err != nil {
		t.Fatal(err)
	}
	if len(blocks) < 30 {
		t.Errorf("the catalog has %d blocks", len(blocks))
	}
	// The palette needs the declaration order, which a Go map does not keep, so
	// it travels beside the parameters rather than as their arrangement.
	for _, b := range blocks {
		if b.Type != "gqa_attention" {
			continue
		}
		if len(b.ParamOrder) == 0 || b.ParamOrder[0] != "d_model" {
			t.Errorf("gqa_attention's parameters start with %v", b.ParamOrder)
		}
		if len(b.Params) != len(b.ParamOrder) {
			t.Errorf("%d parameters but %d in the order", len(b.Params), len(b.ParamOrder))
		}
		if b.Params["d_model"].Doc == "" {
			t.Error("d_model has no documentation")
		}
	}
}

func TestBadInputIsRefusedWithAReason(t *testing.T) {
	engine := services.NewEngineService()
	for _, c := range []struct {
		label string
		run   func() (string, error)
		says  string
	}{
		{"not JSON", func() (string, error) { return engine.Analyze("{", "") }, "could not read the design"},
		{"no version", func() (string, error) { return engine.Analyze(`{"meta":{}}`, "") }, "version"},
		{"bad operating point", func() (string, error) {
			return engine.Analyze(documentOf(t, "gpt2-small"), "{")
		}, "operating point"},
		{"unknown hardware", func() (string, error) {
			return engine.Analyze(documentOf(t, "gpt2-small"), `{"hardware":"made-up"}`)
		}, "hardware"},
		{"scale with no target", func() (string, error) {
			return engine.Scale(documentOf(t, "gpt2-small"), "")
		}, "target"},
		{"unknown preset", func() (string, error) { return engine.Preset("no-such-model") }, "unknown preset"},
	} {
		t.Run(c.label, func(t *testing.T) {
			_, err := c.run()
			if err == nil {
				t.Fatal("accepted")
			}
			if !strings.Contains(strings.ToLower(err.Error()), c.says) {
				t.Errorf("the refusal does not say what is wrong: %v", err)
			}
		})
	}
}
