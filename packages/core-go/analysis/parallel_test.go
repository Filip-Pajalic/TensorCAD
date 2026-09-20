package analysis_test

import (
	"math"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
)

// Expert parallelism, which the options carried for a long time and nothing
// read.
//
// A sparse model's weights are not one pool. Mixtral is 46.70B parameters of
// which 45.10B are experts, so a plan that divides the experts eight ways and a
// plan that divides everything eight ways hold very different amounts, and
// before this they reported the same number.
func TestExpertParallelismDividesTheExperts(t *testing.T) {
	doc := presets.MustGet("mixtral-8x7b")
	seq := 4096.0

	at := func(tp, ep float64) analysis.TrainPerGpu {
		t.Helper()
		gpus := math.Max(tp*ep, 1)
		res, err := analysis.Analyze(doc, analysis.Options{
			T: &seq, GPUs: &gpus,
			Parallel: &analysis.PartialParallel{TP: &tp, EP: &ep},
		}, analysis.Inputs{})
		if err != nil {
			t.Fatalf("analyze: %v", err)
		}
		return res.Memory.Train.PerGpu
	}

	params := analysis.CountParams(analysis.Flatten(doc, ir.ResolveSymbols(doc)))
	if params.Expert <= 0 || params.Expert >= params.Total {
		t.Fatalf("experts are %v of %v parameters", params.Expert, params.Total)
	}
	// Mixtral is almost all experts; a dense model would be none.
	if share := params.Expert / params.Total; share < 0.9 {
		t.Errorf("experts are %.1f%% of Mixtral, want over 90%%", share*100)
	}

	whole := at(1, 1)
	byExperts := at(1, 8)
	byTensors := at(8, 1)
	both := at(8, 8)

	// Sharding the experts eight ways leaves the dense weights whole, so it
	// divides by less than eight.
	dense := params.Total - params.Expert
	want := dense + params.Expert/8
	if got := byExperts.Weights / (whole.Weights / params.Total); math.Abs(got-want) > 1 {
		t.Errorf("expert parallelism holds %.0f parameters per device, want %.0f", got, want)
	}
	if !(byExperts.Weights < whole.Weights) {
		t.Error("expert parallelism did not reduce the weights at all")
	}
	// Tensor parallelism divides everything, so it wins on weights alone.
	if !(byTensors.Weights < byExperts.Weights) {
		t.Errorf("tensor parallelism holds %v, expert parallelism %v; tensor divides more",
			byTensors.Weights, byExperts.Weights)
	}
	// Together they compound.
	if !(both.Weights < byTensors.Weights/2) {
		t.Errorf("together they hold %v; alone tensor parallelism holds %v",
			both.Weights, byTensors.Weights)
	}
	// And the optimizer follows the weights, because it is per parameter.
	if r := both.Optimizer / both.Weights; math.Abs(r-whole.Optimizer/whole.Weights) > 1e-9 {
		t.Errorf("the optimizer no longer tracks the weights: %v vs %v",
			r, whole.Optimizer/whole.Weights)
	}
}

// A dense design has no experts, so the degree has nothing to divide and the
// analysis says so rather than silently reporting a smaller number.
func TestExpertParallelismOnADenseDesignSaysSo(t *testing.T) {
	doc := presets.MustGet("llama-3-8b")
	ep, gpus := 8.0, 8.0
	res, err := analysis.Analyze(doc, analysis.Options{
		GPUs: &gpus, Parallel: &analysis.PartialParallel{EP: &ep},
	}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	plain, err := analysis.Analyze(doc, analysis.Options{GPUs: &gpus}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}
	if res.Memory.Train.PerGpu.Weights != plain.Memory.Train.PerGpu.Weights {
		t.Errorf("expert parallelism changed a dense design's weights: %v vs %v",
			res.Memory.Train.PerGpu.Weights, plain.Memory.Train.PerGpu.Weights)
	}
	found := false
	for _, n := range res.Memory.Notes {
		if n == "Expert parallelism has nothing to divide: this design has no experts." {
			found = true
		}
	}
	if !found {
		t.Errorf("no note about the empty pool; notes were %q", res.Memory.Notes)
	}
}
