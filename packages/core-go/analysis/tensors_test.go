package analysis_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/presets"
)

// Activation memory charged to the tensor rather than to the block.
//
// The same bytes, twice, under two keys — which is only useful if the two
// really are the same bytes. A block with several outputs is one row in
// ActivationsByPath and several here, and the second answers the question
// somebody pointing at a wire is asking: not "what does this block cost" but
// "is *this* the expensive net".

func memoryOf(t *testing.T, preset string) analysis.TrainMemory {
	t.Helper()
	doc := presets.MustGet(preset)
	result, err := analysis.Analyze(doc, analysis.Options{}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("analyze %s: %v", preset, err)
	}
	return result.Memory.Train
}

func sum(m map[string]float64) float64 {
	var total float64
	for _, v := range m {
		total += v
	}
	return total
}

func TestActivationsByTensorTotalsTheSame(t *testing.T) {
	// A convnet, a vision transformer, a dense language model and a sparse one:
	// every shape of design the catalog knows how to draw.
	for _, preset := range []string{"alexnet", "ijepa-vit-h14", "gpt2-small", "llama-3-8b", "mixtral-8x7b"} {
		t.Run(preset, func(t *testing.T) {
			train := memoryOf(t, preset)
			byPath, byTensor := sum(train.ActivationsByPath), sum(train.ActivationsByTensor)
			if math.Abs(byPath-byTensor) > 1 {
				t.Fatalf("the two attributions disagree: %.0f by path, %.0f by tensor", byPath, byTensor)
			}
			if math.Abs(byTensor-train.Activations) > 1 {
				t.Fatalf("by tensor is %.0f, the reported total is %.0f", byTensor, train.Activations)
			}
		})
	}
}

func TestActivationsByTensorNamesAPort(t *testing.T) {
	train := memoryOf(t, "gpt2-small")
	if len(train.ActivationsByTensor) == 0 {
		t.Fatal("nothing was charged to a tensor")
	}

	// Most rows name a wire, "path:port". A few name a block — the extra bytes
	// a block holds that never leave it, like an attention score matrix — and
	// those are its own path, which is what they are.
	var wires int
	for key := range train.ActivationsByTensor {
		if strings.Contains(key, ":") {
			wires++
		}
	}
	if wires == 0 {
		t.Fatalf("no row names a port; keys were %v", keys(train.ActivationsByTensor))
	}
}

// Where the finer attribution is actually finer.
//
// In a plain transformer it is not: every block that holds an activation holds
// exactly one, so the two maps carry the same rows under longer keys. The
// difference shows up wherever a block fans out — `split`, which is how a
// selective scan gets its Δ, B and C out of one projection. Those are three
// wires of three different sizes, and charging them to the block that made
// them gives one number that answers neither "which of these is the big one"
// nor "what would dropping C save".
func TestSplitIsWhereTheTensorKeyEarnsItself(t *testing.T) {
	byTensor := memoryOf(t, "nemotron-h-8b").ActivationsByTensor

	perPath := map[string][]float64{}
	for key, bytes := range byTensor {
		at := strings.LastIndex(key, ":")
		if at < 0 {
			perPath[key] = append(perPath[key], bytes)
			continue
		}
		perPath[key[:at]] = append(perPath[key[:at]], bytes)
	}

	var found string
	for path, charges := range perPath {
		if len(charges) < 2 {
			continue
		}
		// Several charges of the same size would still be a fan-out, but an
		// uneven one is the case that makes the distinction worth having.
		low, high := charges[0], charges[0]
		for _, c := range charges {
			low, high = math.Min(low, c), math.Max(high, c)
		}
		if high > low {
			found = path
			t.Logf("%s holds %d tensors, %.0f to %.0f bytes", path, len(charges), low, high)
			break
		}
	}
	if found == "" {
		t.Fatal("no block fans out into differently sized tensors, so the tensor key says nothing the path key did not")
	}
}

// And the other half of that: in a plain transformer the two agree row for
// row, which is why a design made only of them shows nothing new. Written down
// so the next person does not conclude the field is broken.
func TestAPlainTransformerHoldsOneTensorPerBlock(t *testing.T) {
	byTensor := memoryOf(t, "gpt2-small").ActivationsByTensor
	perPath := map[string]int{}
	for key := range byTensor {
		at := strings.LastIndex(key, ":")
		if at < 0 {
			perPath[key]++
			continue
		}
		perPath[key[:at]]++
	}
	for path, n := range perPath {
		if n > 1 {
			t.Fatalf("%s holds %d tensors; GPT-2 has no block that fans out", path, n)
		}
	}
}

func keys(m map[string]float64) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
