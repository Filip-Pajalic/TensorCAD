package analysis_test

import (
	"math"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/presets"
)

// What latent attention costs to serve, both ways.
//
// The compressed vector is the whole point of the design, and it is only what
// gets cached under a kernel that scores in latent space — one that absorbs the
// up-projections into the query and output matrices and never materializes a
// key or a value. An engine that does not do that holds the full multi-head
// cache, and the difference is nearly two orders of magnitude, so which one is
// being quoted matters more than most numbers in the analysis.

func kvOf(t *testing.T, name string) *analysis.KvResult {
	t.Helper()
	res, err := analysis.Analyze(presets.MustGet(name), analysis.Options{}, analysis.Inputs{})
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return res.Kv
}

// DeepSeek-V2 Table 1 gives the compression as 57x. This reproduces it, which
// is the check that the decompressed figure is the one the paper means: a key
// and a value per head, plus the rotary part, which every head shares and so is
// held once however many there are.
func TestLatentAttentionCompressesTheCacheFiftySevenFold(t *testing.T) {
	kv := kvOf(t, "deepseek-v3")
	ratio := kv.BytesPerTokenDecompressed / kv.BytesPerToken
	if math.Abs(ratio-57) > 0.5 {
		t.Errorf("the cache compresses %.1fx, where DeepSeek-V2 Table 1 gives 57x "+
			"(%s absorbed against %s decompressed)",
			ratio, analysis.FormatBytes(kv.BytesPerToken),
			analysis.FormatBytes(kv.BytesPerTokenDecompressed))
	}
	// And the absorbed figure is the one the preset has always reported.
	if want := 70272.0; kv.BytesPerToken != want {
		t.Errorf("the compressed cache is %v bytes per token, want %v", kv.BytesPerToken, want)
	}
}

// Everything else has one cache, not two. A design with no latent attention in
// it must report the same number both ways, or the second figure would look
// like a claim about designs it says nothing about.
func TestADesignWithoutLatentAttentionHasOneCache(t *testing.T) {
	for _, name := range []string{"llama-3-8b", "gpt2-small", "mixtral-8x7b", "nemotron-h-8b", "gemma-2-9b"} {
		t.Run(name, func(t *testing.T) {
			kv := kvOf(t, name)
			if kv.BytesPerTokenDecompressed != kv.BytesPerToken {
				t.Errorf("%s caches %v bytes per token but %v decompressed, and it has nothing to decompress",
					name, kv.BytesPerToken, kv.BytesPerTokenDecompressed)
			}
		})
	}
}

// The compressed cache is never the larger one: a compression that expands is
// the sort of thing that survives a refactor unnoticed.
func TestTheCompressedCacheIsNeverTheLargerOne(t *testing.T) {
	names, err := presets.Names()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range names {
		kv := kvOf(t, name)
		if kv.BytesPerTokenDecompressed < kv.BytesPerToken {
			t.Errorf("%s: the decompressed cache is %v, smaller than the %v compressed",
				name, kv.BytesPerTokenDecompressed, kv.BytesPerToken)
		}
	}
}
