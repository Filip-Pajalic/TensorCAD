package analysis_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
)

// The analysis against the TypeScript, number for number.
//
// Three operating points per preset, because the default one never exercises
// sharding, full recomputation or an eager attention kernel, and those are
// three of the places the arithmetic is easiest to get subtly wrong. The
// variants below must stay in step with ANALYSIS_VARIANTS in scripts/golden.ts;
// each case records its resolved options, and those are compared first, so
// drift shows up as "the two engines were asked different questions" rather
// than as an unexplained number. Regenerate with `bun run scripts/golden.ts`.

func f(v float64) *float64 { return &v }
func b(v bool) *bool       { return &v }
func i(v int) *int         { return &v }

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

type goldenOptions struct {
	T              float64 `json:"T"`
	B              float64 `json:"B"`
	Dtype          string  `json:"dtype"`
	InferenceDtype string  `json:"inferenceDtype"`
	KvDtype        string  `json:"kvDtype"`
	Hardware       string  `json:"hardware"`
	GPUs           float64 `json:"gpus"`
	Parallel       struct {
		DP               float64 `json:"dp"`
		TP               float64 `json:"tp"`
		PP               float64 `json:"pp"`
		EP               float64 `json:"ep"`
		Zero             int     `json:"zero"`
		SequenceParallel bool    `json:"sequenceParallel"`
	} `json:"parallel"`
	Optimizer           string  `json:"optimizer"`
	Recompute           string  `json:"recompute"`
	Flash               bool    `json:"flash"`
	Tokens              float64 `json:"tokens"`
	TokensWereDefaulted bool    `json:"tokensWereDefaulted"`
	MFU                 float64 `json:"mfu"`
	DecodeEfficiency    float64 `json:"decodeEfficiency"`
	Concurrency         float64 `json:"concurrency"`
}

type pair struct {
	Key   string
	Value float64
}

// UnmarshalJSON reads a ["key", value] pair, which is how the generator writes
// a map so neither engine's key order matters.
func (p *pair) UnmarshalJSON(data []byte) error {
	var raw []json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if len(raw) != 2 {
		return fmt.Errorf("expected a [key, value] pair, got %d entries", len(raw))
	}
	if err := json.Unmarshal(raw[0], &p.Key); err != nil {
		return err
	}
	return json.Unmarshal(raw[1], &p.Value)
}

type goldenCase struct {
	Label   string        `json:"label"`
	Options goldenOptions `json:"options"`
	Params  struct {
		Total              float64 `json:"total"`
		Active             float64 `json:"active"`
		Embedding          float64 `json:"embedding"`
		Head               float64 `json:"head"`
		NonEmbedding       float64 `json:"nonEmbedding"`
		NonEmbeddingActive float64 `json:"nonEmbeddingActive"`
		ByPath             []pair  `json:"byPath"`
		ByCategory         []pair  `json:"byCategory"`
		ByType             []pair  `json:"byType"`
	} `json:"params"`
	Flops struct {
		FwdDense             float64 `json:"fwdDense"`
		FwdAttention         float64 `json:"fwdAttention"`
		FwdAttentionUnmasked float64 `json:"fwdAttentionUnmasked"`
		FwdTotal             float64 `json:"fwdTotal"`
		FwdTotalUnmasked     float64 `json:"fwdTotalUnmasked"`
		Elementwise          float64 `json:"elementwise"`
		TrainPerToken        float64 `json:"trainPerToken"`
		AttentionShare       float64 `json:"attentionShare"`
		RuleOfThumb2N        float64 `json:"ruleOfThumb2N"`
		RuleOfThumb6N        float64 `json:"ruleOfThumb6N"`
		ByPath               []pair  `json:"byPath"`
		ByCategory           []pair  `json:"byCategory"`
	} `json:"flops"`
	Kv struct {
		BytesPerToken         float64 `json:"bytesPerToken"`
		BytesPerSequenceFixed float64 `json:"bytesPerSequenceFixed"`
		ByPath                []pair  `json:"byPath"`
	} `json:"kv"`
	Memory struct {
		WeightsBytes float64 `json:"weightsBytes"`
		Train        struct {
			Weights     float64 `json:"weights"`
			Grads       float64 `json:"grads"`
			Optimizer   float64 `json:"optimizer"`
			Activations float64 `json:"activations"`
			Logits      float64 `json:"logits"`
			Total       float64 `json:"total"`
			PerGpu      struct {
				Weights     float64 `json:"weights"`
				Grads       float64 `json:"grads"`
				Optimizer   float64 `json:"optimizer"`
				Activations float64 `json:"activations"`
				Total       float64 `json:"total"`
			} `json:"perGpu"`
			ActivationsByPath []pair `json:"activationsByPath"`
		} `json:"train"`
		Infer struct {
			Weights  float64 `json:"weights"`
			Kv       float64 `json:"kv"`
			Overhead float64 `json:"overhead"`
			Total    float64 `json:"total"`
		} `json:"infer"`
		OptimizerLabel string   `json:"optimizerLabel"`
		Notes          []string `json:"notes"`
	} `json:"memory"`
	Throughput struct {
		RidgePoint            float64  `json:"ridgePoint"`
		DecodeBytesPerStep    float64  `json:"decodeBytesPerStep"`
		DecodeFlopsPerStep    float64  `json:"decodeFlopsPerStep"`
		DecodeSecondsPerStep  float64  `json:"decodeSecondsPerStep"`
		DecodeTokensPerSecond float64  `json:"decodeTokensPerSecond"`
		MemoryBound           bool     `json:"memoryBound"`
		PrefillSeconds        float64  `json:"prefillSeconds"`
		Notes                 []string `json:"notes"`
	} `json:"throughput"`
	Cost struct {
		TotalFlops     float64 `json:"totalFlops"`
		GpuHours       float64 `json:"gpuHours"`
		WallClockHours float64 `json:"wallClockHours"`
		Dollars        float64 `json:"dollars"`
		Tokens         float64 `json:"tokens"`
		MFU            float64 `json:"mfu"`
	} `json:"cost"`
	Chinchilla struct {
		OptimalTokens        float64 `json:"optimalTokens"`
		TokensPerParam       float64 `json:"tokensPerParam"`
		TokensPerActiveParam float64 `json:"tokensPerActiveParam"`
		OverTrainingRatio    float64 `json:"overTrainingRatio"`
		PredictedLoss        []pair  `json:"predictedLoss"`
		Verdict              string  `json:"verdict"`
	} `json:"chinchilla"`
	Errors []string `json:"errors"`
	Flat   struct {
		Nodes   int `json:"nodes"`
		Blocks  int `json:"blocks"`
		Repeats []struct {
			Path   string  `json:"path"`
			Type   string  `json:"type"`
			Count  float64 `json:"count"`
			Active float64 `json:"active"`
		} `json:"repeats"`
		Errors []string `json:"errors"`
	} `json:"flat"`
}

func loadAnalysis(t *testing.T, name string) []goldenCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "analysis", name+".json"))
	if err != nil {
		t.Fatalf("read analysis golden %s: %v", name, err)
	}
	var g struct {
		Cases []goldenCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse analysis golden %s: %v", name, err)
	}
	if len(g.Cases) == 0 {
		t.Fatalf("analysis golden %s is empty", name)
	}
	return g.Cases
}

func TestAnalysisMatchesTypeScript(t *testing.T) {
	opts := variants()
	for _, name := range presetNames(t) {
		for _, c := range loadAnalysis(t, name) {
			t.Run(name+"/"+c.Label, func(t *testing.T) {
				o, ok := opts[c.Label]
				if !ok {
					t.Fatalf("no Go variant named %q", c.Label)
				}
				doc := loadDoc(t, name)
				got, err := analysis.Analyze(doc, o, analysis.Inputs{})
				if err != nil {
					t.Fatal(err)
				}

				// The options first: if these disagree the two engines were
				// asked different questions and nothing below means anything.
				compareOptions(t, got.Options, c.Options)

				p, w := got.Params, c.Params
				// A parameter count is a whole number of weights, so it is
				// exact or it is wrong.
				exact(t, "params.total", p.Total, w.Total)
				exact(t, "params.active", p.Active, w.Active)
				exact(t, "params.embedding", p.Embedding, w.Embedding)
				exact(t, "params.head", p.Head, w.Head)
				exact(t, "params.nonEmbedding", p.NonEmbedding, w.NonEmbedding)
				exact(t, "params.nonEmbeddingActive", p.NonEmbeddingActive, w.NonEmbeddingActive)
				exactMap(t, "params.byPath", p.ByPath, w.ByPath)
				exactMap(t, "params.byCategory", p.ByCategory, w.ByCategory)
				exactMap(t, "params.byType", p.ByType, w.ByType)

				fl, wf := got.Flops, c.Flops
				close(t, "flops.fwdDense", fl.FwdDense, wf.FwdDense)
				close(t, "flops.fwdAttention", fl.FwdAttention, wf.FwdAttention)
				close(t, "flops.fwdAttentionUnmasked", fl.FwdAttentionUnmasked, wf.FwdAttentionUnmasked)
				close(t, "flops.fwdTotal", fl.FwdTotal, wf.FwdTotal)
				close(t, "flops.fwdTotalUnmasked", fl.FwdTotalUnmasked, wf.FwdTotalUnmasked)
				close(t, "flops.elementwise", fl.Elementwise, wf.Elementwise)
				close(t, "flops.trainPerToken", fl.TrainPerToken, wf.TrainPerToken)
				close(t, "flops.attentionShare", fl.AttentionShare, wf.AttentionShare)
				close(t, "flops.ruleOfThumb2N", fl.RuleOfThumb2N, wf.RuleOfThumb2N)
				close(t, "flops.ruleOfThumb6N", fl.RuleOfThumb6N, wf.RuleOfThumb6N)
				closeMap(t, "flops.byPath", fl.ByPath, wf.ByPath)
				closeMap(t, "flops.byCategory", fl.ByCategory, wf.ByCategory)

				close(t, "kv.bytesPerToken", got.Kv.BytesPerToken, c.Kv.BytesPerToken)
				close(t, "kv.bytesPerSequenceFixed", got.Kv.BytesPerSequenceFixed, c.Kv.BytesPerSequenceFixed)
				closeMap(t, "kv.byPath", got.Kv.ByPath, c.Kv.ByPath)

				m, wm := got.Memory, c.Memory
				close(t, "memory.weightsBytes", m.WeightsBytes, wm.WeightsBytes)
				close(t, "memory.train.weights", m.Train.Weights, wm.Train.Weights)
				close(t, "memory.train.grads", m.Train.Grads, wm.Train.Grads)
				close(t, "memory.train.optimizer", m.Train.Optimizer, wm.Train.Optimizer)
				close(t, "memory.train.activations", m.Train.Activations, wm.Train.Activations)
				close(t, "memory.train.logits", m.Train.Logits, wm.Train.Logits)
				close(t, "memory.train.total", m.Train.Total, wm.Train.Total)
				close(t, "memory.train.perGpu.weights", m.Train.PerGpu.Weights, wm.Train.PerGpu.Weights)
				close(t, "memory.train.perGpu.grads", m.Train.PerGpu.Grads, wm.Train.PerGpu.Grads)
				close(t, "memory.train.perGpu.optimizer", m.Train.PerGpu.Optimizer, wm.Train.PerGpu.Optimizer)
				close(t, "memory.train.perGpu.activations", m.Train.PerGpu.Activations, wm.Train.PerGpu.Activations)
				close(t, "memory.train.perGpu.total", m.Train.PerGpu.Total, wm.Train.PerGpu.Total)
				closeMap(t, "memory.train.activationsByPath", m.Train.ActivationsByPath, wm.Train.ActivationsByPath)
				close(t, "memory.infer.weights", m.Infer.Weights, wm.Infer.Weights)
				close(t, "memory.infer.kv", m.Infer.Kv, wm.Infer.Kv)
				close(t, "memory.infer.overhead", m.Infer.Overhead, wm.Infer.Overhead)
				close(t, "memory.infer.total", m.Infer.Total, wm.Infer.Total)
				if m.OptimizerLabel != wm.OptimizerLabel {
					t.Errorf("memory.optimizerLabel: got %q, want %q", m.OptimizerLabel, wm.OptimizerLabel)
				}
				// The notes are prose a person reads under the numbers, and
				// they are the only place the model explains itself.
				sameLines(t, "memory.notes", m.Notes, wm.Notes)

				th, wt := got.Throughput, c.Throughput
				close(t, "throughput.ridgePoint", th.RidgePoint, wt.RidgePoint)
				close(t, "throughput.decodeBytesPerStep", th.DecodeBytesPerStep, wt.DecodeBytesPerStep)
				close(t, "throughput.decodeFlopsPerStep", th.DecodeFlopsPerStep, wt.DecodeFlopsPerStep)
				close(t, "throughput.decodeSecondsPerStep", th.DecodeSecondsPerStep, wt.DecodeSecondsPerStep)
				close(t, "throughput.decodeTokensPerSecond", th.DecodeTokensPerSecond, wt.DecodeTokensPerSecond)
				if th.MemoryBound != wt.MemoryBound {
					t.Errorf("throughput.memoryBound: got %v, want %v", th.MemoryBound, wt.MemoryBound)
				}
				close(t, "throughput.prefillSeconds", th.PrefillSeconds, wt.PrefillSeconds)
				sameLines(t, "throughput.notes", th.Notes, wt.Notes)

				close(t, "cost.totalFlops", got.Cost.TotalFlops, c.Cost.TotalFlops)
				close(t, "cost.gpuHours", got.Cost.GpuHours, c.Cost.GpuHours)
				close(t, "cost.wallClockHours", got.Cost.WallClockHours, c.Cost.WallClockHours)
				close(t, "cost.dollars", got.Cost.Dollars, c.Cost.Dollars)
				close(t, "cost.tokens", got.Cost.Tokens, c.Cost.Tokens)
				close(t, "cost.mfu", got.Cost.MFU, c.Cost.MFU)

				ch, wc := got.Chinchilla, c.Chinchilla
				close(t, "chinchilla.optimalTokens", ch.OptimalTokens, wc.OptimalTokens)
				close(t, "chinchilla.tokensPerParam", ch.TokensPerParam, wc.TokensPerParam)
				close(t, "chinchilla.tokensPerActiveParam", ch.TokensPerActiveParam, wc.TokensPerActiveParam)
				close(t, "chinchilla.overTrainingRatio", ch.OverTrainingRatio, wc.OverTrainingRatio)
				closeMap(t, "chinchilla.predictedLoss", ch.PredictedLoss, wc.PredictedLoss)
				if ch.Verdict != wc.Verdict {
					t.Errorf("chinchilla.verdict: got %q, want %q", ch.Verdict, wc.Verdict)
				}

				sameLines(t, "errors", got.Errors, c.Errors)
				if len(got.Flat.Nodes) != c.Flat.Nodes {
					t.Errorf("flat.nodes: got %d, want %d", len(got.Flat.Nodes), c.Flat.Nodes)
				}
				if len(got.Flat.Blocks) != c.Flat.Blocks {
					t.Errorf("flat.blocks: got %d, want %d", len(got.Flat.Blocks), c.Flat.Blocks)
				}
				sameLines(t, "flat.errors", got.Flat.Errors, c.Flat.Errors)
				if len(got.Flat.Repeats) != len(c.Flat.Repeats) {
					t.Errorf("flat.repeats: got %d, want %d", len(got.Flat.Repeats), len(c.Flat.Repeats))
				} else {
					for k, r := range got.Flat.Repeats {
						w := c.Flat.Repeats[k]
						if r.Path != w.Path || r.Type != w.Type || r.Count != w.Count || r.Active != w.Active {
							t.Errorf("flat.repeats[%d]: got %v, want %v", k, r, w)
						}
					}
				}
			})
		}
	}
}

// TestUnknownHardwareIsRefused pins the one thing Analyze can fail at: a
// profile id nobody knows is an error, not a silent fall back to the default.
func TestUnknownHardwareIsRefused(t *testing.T) {
	doc := loadDoc(t, presetNames(t)[0])
	if _, err := analysis.Analyze(doc, analysis.Options{Hardware: "made-up"}, analysis.Inputs{}); err == nil {
		t.Fatal("an unknown hardware profile was accepted")
	}
}

func compareOptions(t *testing.T, got analysis.ResolvedOptions, want goldenOptions) {
	t.Helper()
	exact(t, "options.T", got.T, want.T)
	exact(t, "options.B", got.B, want.B)
	for _, s := range []struct{ label, got, want string }{
		{"options.dtype", got.Dtype, want.Dtype},
		{"options.inferenceDtype", got.InferenceDtype, want.InferenceDtype},
		{"options.kvDtype", got.KvDtype, want.KvDtype},
		{"options.hardware", got.Hardware.ID, want.Hardware},
		{"options.optimizer", got.Optimizer, want.Optimizer},
		{"options.recompute", got.Recompute, want.Recompute},
	} {
		if s.got != s.want {
			t.Errorf("%s: got %q, want %q", s.label, s.got, s.want)
		}
	}
	exact(t, "options.gpus", got.GPUs, want.GPUs)
	exact(t, "options.parallel.dp", got.Parallel.DP, want.Parallel.DP)
	exact(t, "options.parallel.tp", got.Parallel.TP, want.Parallel.TP)
	exact(t, "options.parallel.pp", got.Parallel.PP, want.Parallel.PP)
	exact(t, "options.parallel.ep", got.Parallel.EP, want.Parallel.EP)
	if got.Parallel.Zero != want.Parallel.Zero {
		t.Errorf("options.parallel.zero: got %d, want %d", got.Parallel.Zero, want.Parallel.Zero)
	}
	if got.Parallel.SequenceParallel != want.Parallel.SequenceParallel {
		t.Errorf("options.parallel.sequenceParallel: got %v, want %v",
			got.Parallel.SequenceParallel, want.Parallel.SequenceParallel)
	}
	if got.Flash != want.Flash {
		t.Errorf("options.flash: got %v, want %v", got.Flash, want.Flash)
	}
	exact(t, "options.tokens", got.Tokens, want.Tokens)
	if got.TokensWereDefault != want.TokensWereDefaulted {
		t.Errorf("options.tokensWereDefaulted: got %v, want %v",
			got.TokensWereDefault, want.TokensWereDefaulted)
	}
	close(t, "options.mfu", got.MFU, want.MFU)
	close(t, "options.decodeEfficiency", got.DecodeEfficiency, want.DecodeEfficiency)
	exact(t, "options.concurrency", got.Concurrency, want.Concurrency)
}

func exact(t *testing.T, label string, got, want float64) {
	t.Helper()
	if got != want {
		t.Errorf("%s: got %s, want %s", label, show(got), show(want))
	}
}

// close allows a relative difference of 1e-12, which is roughly four thousand
// times the spacing of a double and far tighter than any formula error could
// hide in. It exists for the handful of quantities that go through pow, where
// the two runtimes' libm need not agree in the last bit.
func close(t *testing.T, label string, got, want float64) {
	t.Helper()
	if got == want {
		return
	}
	scale := math.Max(math.Abs(got), math.Abs(want))
	if scale > 0 && math.Abs(got-want)/scale <= 1e-12 {
		return
	}
	t.Errorf("%s: got %s, want %s", label, show(got), show(want))
}

func exactMap(t *testing.T, label string, got map[string]float64, want []pair) {
	t.Helper()
	compareMap(t, label, got, want, exact)
}

func closeMap(t *testing.T, label string, got map[string]float64, want []pair) {
	t.Helper()
	compareMap(t, label, got, want, close)
}

func compareMap(t *testing.T, label string, got map[string]float64, want []pair,
	cmp func(*testing.T, string, float64, float64)) {
	t.Helper()
	seen := make(map[string]bool, len(want))
	for _, w := range want {
		seen[w.Key] = true
		g, ok := got[w.Key]
		if !ok {
			t.Errorf("%s: %s missing, want %s", label, w.Key, show(w.Value))
			continue
		}
		cmp(t, label+"["+w.Key+"]", g, w.Value)
	}
	extra := make([]string, 0)
	for k := range got {
		if !seen[k] {
			extra = append(extra, k)
		}
	}
	sort.Strings(extra)
	for _, k := range extra {
		t.Errorf("%s: %s = %s is not in the TypeScript", label, k, show(got[k]))
	}
}

func sameLines(t *testing.T, label string, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: got %d lines, want %d\n got  %q\n want %q", label, len(got), len(want), got, want)
		return
	}
	for k := range got {
		if got[k] != want[k] {
			t.Errorf("%s[%d]: got %q, want %q", label, k, got[k], want[k])
		}
	}
}

func show(v float64) string { return fmt.Sprintf("%.17g", v) }

// loadDoc and presetNames live in the infer package's test too; the analysis
// package cannot import them, so they are repeated here.
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
	if len(names) == 0 {
		t.Fatal("preset index is empty")
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
