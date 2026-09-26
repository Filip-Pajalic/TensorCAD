package analysis_test

import (
	"math"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/presets"
)

func atPrecision(t *testing.T, name, precision, dtype string) *analysis.Result {
	t.Helper()
	T, B, one := 512.0, 2.0, 1.0
	r, err := analysis.Analyze(presets.MustGet(name), analysis.Options{
		T: &T, B: &B, GPUs: &one, Recompute: "none", Optimizer: "adamw", Dtype: dtype, Precision: precision,
	}, analysis.Inputs{})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// Autocast holds the same sixteen bytes a parameter as mixed precision, split
// the other way: fp32 weights that are their own master copy.
func TestAutocastSplitsTheSameSixteenBytesDifferently(t *testing.T) {
	mixed := atPrecision(t, "gpt2-small", "", "bf16").Memory.Train
	auto := atPrecision(t, "gpt2-small", "autocast", "bf16").Memory.Train
	n := atPrecision(t, "gpt2-small", "", "bf16").Params.Total

	if got, want := mixed.Weights+mixed.Grads+mixed.Optimizer, 16*n; got != want {
		t.Errorf("mixed precision holds %g bytes at rest, want %g", got, want)
	}
	if got, want := auto.Weights+auto.Grads+auto.Optimizer, 16*n; got != want {
		t.Errorf("autocast holds %g bytes at rest, want %g", got, want)
	}
	if label := atPrecision(t, "gpt2-small", "autocast", "bf16").Memory.OptimizerLabel; label != "AdamW, under autocast (16 B/param)" {
		t.Errorf("the optimizer is labelled %q", label)
	}
	if auto.Weights != 4*n || auto.Grads != 4*n || auto.Optimizer != 8*n {
		t.Errorf("autocast splits %g/%g/%g, want 4/4/8 bytes a parameter",
			auto.Weights/n, auto.Grads/n, auto.Optimizer/n)
	}
}

// What autocast saves: more than mixed precision, and among it a bf16 copy of
// every weight a matrix multiply reads — GPT-2's tied head included, which
// owns no weights of its own and still casts the table it shares.
func TestAutocastSavesWeightCopiesAndWiderTensors(t *testing.T) {
	r := atPrecision(t, "gpt2-small", "autocast", "bf16")
	mixed := atPrecision(t, "gpt2-small", "", "bf16")
	if !(r.Memory.Train.Activations > mixed.Memory.Train.Activations) {
		t.Errorf("autocast saves %g, no more than mixed precision's %g",
			r.Memory.Train.Activations, mixed.Memory.Train.Activations)
	}
	// Every parameter but the position table and the norms passes through a
	// matrix multiply, the shared token table included, via the head.
	params := r.Params
	const positions, norms = 1024 * 768, (12*2 + 1) * 2 * 768
	want := 2 * (params.Total - positions - norms)
	if math.Abs(r.Memory.Train.CastWeights-want) > 1 {
		t.Errorf("weight copies: %g bytes, want %g", r.Memory.Train.CastWeights, want)
	}
	if r.Options.Precision != "autocast" {
		t.Errorf("the options say %q; the numbers assume autocast", r.Options.Precision)
	}
}

// The default is untouched, and in fp32 there is nothing to cast to.
func TestAutocastIsOptInAndHalfPrecisionOnly(t *testing.T) {
	mixed := atPrecision(t, "gpt2-small", "", "bf16")
	named := atPrecision(t, "gpt2-small", "mixed", "bf16")
	if mixed.Memory.Train.Activations != named.Memory.Train.Activations ||
		mixed.Memory.Train.Optimizer != named.Memory.Train.Optimizer {
		t.Error("naming the default changed the numbers")
	}
	if mixed.Memory.Train.CastWeights != 0 || mixed.Options.Precision != "" {
		t.Error("the default should carry no weight copies and name no recipe")
	}
	full := atPrecision(t, "gpt2-small", "autocast", "fp32")
	plain := atPrecision(t, "gpt2-small", "", "fp32")
	if full.Memory.Train.Activations != plain.Memory.Train.Activations || full.Options.Precision != "" {
		t.Error("autocast in fp32 should be fp32 training, unchanged")
	}
}
