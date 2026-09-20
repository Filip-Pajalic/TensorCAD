package analysis_test

import (
	"math"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
)

// What a decode step reads, which for a mixture of experts is neither the
// active count nor the resident one.
//
// One token routes to top_k of the experts and reads that share. A batch does
// not: its tokens route independently, and an expert one token skipped is read
// anyway if any other token in the step wanted it. Treating a batch as though
// it read one token's share is the difference between a decode rate that is
// plausible and one that is several times too fast.

func flatten(t *testing.T, name string) *analysis.FlatResult {
	t.Helper()
	doc := presets.MustGet(name)
	return analysis.Flatten(doc, ir.ResolveSymbols(doc))
}

var sparse = []string{"mixtral-8x7b", "deepseek-v3", "qwen3-30b-a3b"}
var dense = []string{"gpt2-small", "llama-3-8b", "ijepa-vit-h14"}

// A single token reads exactly what it is active in. That is the definition,
// and it is what keeps every batch-1 figure in the library where it was.
func TestOneTokenReadsTheActiveCount(t *testing.T) {
	for _, name := range append(append([]string{}, sparse...), dense...) {
		t.Run(name, func(t *testing.T) {
			flat := flatten(t, name)
			want := analysis.CountParams(flat).Active
			if got := analysis.StreamedParams(flat, 1); math.Abs(got-want) > 1e-6 {
				t.Errorf("one token reads %v weights, want the active count of %v", got, want)
			}
		})
	}
}

// A dense model reads all of itself whatever the batch: there is nothing for a
// second token to add.
func TestADenseModelReadsItselfAtEveryBatch(t *testing.T) {
	for _, name := range dense {
		t.Run(name, func(t *testing.T) {
			flat := flatten(t, name)
			want := analysis.CountParams(flat).Total
			for _, batch := range []float64{1, 8, 1024} {
				if got := analysis.StreamedParams(flat, batch); math.Abs(got-want) > 1e-6 {
					t.Errorf("at batch %v it reads %v weights, want %v", batch, got, want)
				}
			}
		})
	}
}

// A sparse model reads more as the batch grows, and never more than it holds.
func TestASparseModelClimbsTowardsItsResidentCount(t *testing.T) {
	for _, name := range sparse {
		t.Run(name, func(t *testing.T) {
			flat := flatten(t, name)
			counts := analysis.CountParams(flat)
			last := 0.0
			for _, batch := range []float64{1, 2, 4, 16, 64, 4096} {
				got := analysis.StreamedParams(flat, batch)
				if got <= last {
					t.Errorf("batch %v reads %v, no more than the %v a smaller batch read", batch, got, last)
				}
				if got > counts.Total+1e-6 {
					t.Errorf("batch %v reads %v, more than the %v the model holds", batch, got, counts.Total)
				}
				last = got
			}
			// Sparsity is the point: one token has to read materially less than
			// the whole model, or there is nothing here to model.
			if one := analysis.StreamedParams(flat, 1); one > 0.9*counts.Total {
				t.Errorf("one token reads %v of %v, which is not a sparse model", one, counts.Total)
			}
			// And a large batch has to arrive: an expert missed by four
			// thousand tokens is an expert that is never used.
			if many := analysis.StreamedParams(flat, 4096); many < 0.999*counts.Total {
				t.Errorf("4096 tokens read %v of %v, leaving experts no token ever asked for",
					many, counts.Total)
			}
		})
	}
}

// The consequence, which is the reason to model it at all: a sparse model's
// decode rate at batch is lower than treating the batch as one token says.
func TestBatchedSparseDecodeIsSlowerThanOneTokenSuggests(t *testing.T) {
	doc := presets.MustGet("mixtral-8x7b")
	batch := 32.0
	res, err := analysis.Analyze(doc, analysis.Options{Concurrency: &batch}, analysis.Inputs{})
	if err != nil {
		t.Fatal(err)
	}
	streamed := res.Throughput.DecodeWeightBytes
	resident := res.Throughput.ResidentWeightBytes
	active := res.Params.Active * analysis.DtypeBytes[res.Options.InferenceDtype]

	if streamed <= active {
		t.Errorf("a batch of %v reads %v of weights, no more than the %v one token reads",
			batch, streamed, active)
	}
	if streamed > resident {
		t.Errorf("a batch of %v reads %v, more than the %v held", batch, streamed, resident)
	}
	// Thirty-two tokens picking two of eight experts each leave almost nothing
	// unread, which is the finding: this is nearly a dense read, not a sparse one.
	if streamed < 0.99*resident {
		t.Errorf("a batch of %v reads only %v of the %v held", batch, streamed, resident)
	}
	if len(res.Throughput.Notes) < 2 {
		t.Error("nothing in the notes says the batch reads more than one token's share")
	}
}
