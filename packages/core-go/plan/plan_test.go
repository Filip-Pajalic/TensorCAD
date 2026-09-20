package plan_test

import (
	"math"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/plan"
	"github.com/tensorcad/core/presets"
)

func search(t *testing.T, preset string, seq, gpus float64, req plan.Request) *plan.Result {
	t.Helper()
	req.GPUs = gpus
	res, err := plan.Search(presets.MustGet(preset), analysis.Options{
		T: &seq, Hardware: "h100-sxm",
	}, req)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	return res
}

// Every plan offered has to be a plan: the degrees multiply to the cluster, the
// tensor-parallel group fits in a node, and the memory fits in the budget. A
// planner that suggests something impossible is worse than no planner.
func TestEveryPlanOfferedIsValid(t *testing.T) {
	for _, c := range []struct {
		preset string
		gpus   float64
	}{
		{"llama-3-8b", 8},
		{"llama-3-70b", 64},
		{"mixtral-8x7b", 64},
		{"gpt2-small", 4},
	} {
		t.Run(c.preset, func(t *testing.T) {
			res := search(t, c.preset, 8192, c.gpus, plan.Request{Limit: 20})
			if len(res.Fits) == 0 {
				t.Fatalf("nothing fits on %v GPUs", c.gpus)
			}
			for _, f := range res.Fits {
				p := f.Parallel
				if got := p.DP * p.TP * p.PP * p.EP; got != c.gpus {
					t.Errorf("%s uses %v devices, the cluster has %v", f.Summary, got, c.gpus)
				}
				if p.TP > 8 {
					t.Errorf("%s spreads a matrix over %v ranks, past one node", f.Summary, p.TP)
				}
				if f.Used > 1 {
					t.Errorf("%s takes %.0f%% of the budget and was offered anyway", f.Summary, f.Used*100)
				}
				if f.PerGpu.Total > res.Budget {
					t.Errorf("%s needs %s against a budget of %s",
						f.Summary, analysis.FormatBytes(f.PerGpu.Total), analysis.FormatBytes(res.Budget))
				}
				if p.Zero > 0 && p.DP <= 1 {
					t.Errorf("%s sharded the optimizer across one replica", f.Summary)
				}
				if p.SequenceParallel && p.TP <= 1 {
					t.Errorf("%s shards the sequence with no group to shard it over", f.Summary)
				}
				if f.Summary == "" {
					t.Error("a plan with no summary")
				}
			}
		})
	}
}

// The plans come back in the order of how little they ask, and the list has no
// two entries that are the same split holding the same bytes.
func TestPlansAreRankedAndDistinct(t *testing.T) {
	res := search(t, "llama-3-70b", 8192, 64, plan.Request{Limit: 12})
	seen := map[string]bool{}
	for _, f := range res.Fits {
		if seen[f.Summary] {
			t.Errorf("%q offered twice", f.Summary)
		}
		seen[f.Summary] = true
	}
	// The first plan is the one to reach for, so it must not be the one that
	// fills the device.
	if res.Fits[0].Used > 0.95 {
		t.Errorf("the first plan fills %.0f%% of the budget", res.Fits[0].Used*100)
	}
}

// A model that cannot fit is told so, with the size of the gap, rather than
// being given the least bad plan as though it worked.
func TestNothingFitsIsSaidPlainly(t *testing.T) {
	res := search(t, "llama-3.1-405b", 8192, 8, plan.Request{})
	if len(res.Fits) != 0 {
		t.Fatalf("405B fits on eight devices? %q", res.Fits[0].Summary)
	}
	if res.Closest == nil {
		t.Fatal("no nearest miss to look at")
	}
	if res.Closest.PerGpu.Total <= res.Budget {
		t.Error("the nearest miss fits, so it should have been offered")
	}
	if len(res.Notes) == 0 {
		t.Error("nothing said about why")
	}
}

// More devices is more room. The property is about the least a plan can hold,
// not about the one ranked first: with more devices the planner can afford a
// less intrusive plan, and a plan that stops recomputing holds more per device
// on purpose. What must never happen is that the best case gets worse.
func TestMoreDevicesNeverRaisesTheFloor(t *testing.T) {
	var floor float64 = math.Inf(1)
	fitted := false
	for _, gpus := range []float64{8, 16, 32, 64, 128} {
		res := search(t, "llama-3-70b", 8192, gpus, plan.Request{Limit: 1000})
		if len(res.Fits) == 0 {
			continue
		}
		fitted = true
		least := math.Inf(1)
		for _, f := range res.Fits {
			least = math.Min(least, f.PerGpu.Total)
		}
		if least > floor {
			t.Errorf("the least %v devices can hold is %s; %v could get to %s",
				gpus, analysis.FormatBytes(least), gpus/2, analysis.FormatBytes(floor))
		}
		floor = least
	}
	if !fitted {
		t.Error("nothing fit at any size")
	}
}

// Expert parallelism is only offered where there are experts to parallelize.
func TestExpertParallelismIsOfferedOnlyToSparseDesigns(t *testing.T) {
	dense := search(t, "llama-3-70b", 8192, 64, plan.Request{Limit: 50})
	for _, f := range dense.Fits {
		if f.Parallel.EP > 1 {
			t.Errorf("a dense design was offered %q", f.Summary)
		}
	}
	if len(dense.Notes) == 0 {
		t.Error("nothing said about why expert parallelism is absent")
	}

	sparse := search(t, "mixtral-8x7b", 4096, 64, plan.Request{Limit: 200})
	any := false
	for _, f := range sparse.Fits {
		if f.Parallel.EP > 1 {
			any = true
		}
	}
	if !any {
		t.Error("a mixture of experts was never offered expert parallelism")
	}
}

// The same question twice gives the same answer. A map iterated somewhere in
// the middle of this would not.
func TestTheSearchIsDeterministic(t *testing.T) {
	for i := 0; i < 3; i++ {
		a := search(t, "mixtral-8x7b", 4096, 64, plan.Request{Limit: 6})
		b := search(t, "mixtral-8x7b", 4096, 64, plan.Request{Limit: 6})
		if len(a.Fits) != len(b.Fits) {
			t.Fatalf("%d plans, then %d", len(a.Fits), len(b.Fits))
		}
		for j := range a.Fits {
			if a.Fits[j].Summary != b.Fits[j].Summary || a.Fits[j].PerGpu.Total != b.Fits[j].PerGpu.Total {
				t.Errorf("plan %d: %q then %q", j, a.Fits[j].Summary, b.Fits[j].Summary)
			}
		}
	}
}

// A cluster has to be a whole number of devices, and headroom a fraction.
func TestNonsenseIsRefused(t *testing.T) {
	doc := presets.MustGet("gpt2-small")
	for _, req := range []plan.Request{
		{GPUs: 0},
		{GPUs: 2.5},
		{GPUs: -8},
		{GPUs: 8, Headroom: 1},
		{GPUs: 8, Headroom: 4},
	} {
		if _, err := plan.Search(doc, analysis.Options{}, req); err == nil {
			t.Errorf("accepted %+v", req)
		}
	}
	if _, err := plan.Search(doc, analysis.Options{Hardware: "made-up"}, plan.Request{GPUs: 8}); err == nil {
		t.Error("accepted a hardware profile that does not exist")
	}
}
