package catalog

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// A mask that keeps documents apart, and what a packing makes it cost.
//
// The references here share nothing with the sampler: an exact count over
// every phase a row can start at, and a brute-force stream of documents drawn
// by a different generator, a different method and many times as many rows.

const documentMask = "doc(b, q) == doc(b, kv)"

// keysPacked is the keys a causal query keeps under the document mask.
func keysPacked(t *testing.T, T float64, pack *Packing) float64 {
	t.Helper()
	def, r := sdpaWith(t, map[string]any{"causal": true, "mask": documentMask})
	f := def.Flops(r, AnalysisCtx{T: T, B: 1, Bytes: 2, Flash: true, Packing: pack})
	return f.FwdSeq / (4 * 8 * 64)
}

// kept is what a causal document mask keeps of one row: each piece of a
// document the row holds, a triangle of its own.
func kept(pieces []float64) float64 {
	sum := 0.0
	for _, l := range pieces {
		sum += l * (l + 1) / 2
	}
	return sum
}

// asKeys turns what rows kept into the engine's count: the share of the
// causal triangle, times the half of T it counts causal attention at.
func asKeys(keptSum, rows, T float64) float64 {
	return keptSum / rows / (T * (T + 1) / 2) * (T / 2)
}

// Documents of one length, cut into rows at every phase a row can begin at.
func exactFixed(T, L float64) float64 {
	sum := 0.0
	for phase := 0.0; phase < L; phase++ {
		var pieces []float64
		at := 0.0
		for next := L - phase; at < T; next = math.Min(T, at+L) {
			next = math.Min(T, next)
			pieces = append(pieces, next-at)
			at = next
		}
		sum += kept(pieces)
	}
	return asKeys(sum, L, T)
}

// A stream of gamma-distributed documents, drawn the textbook way —
// Marsaglia and Tsang's method from normal draws — and cut into many rows at
// random. Exponential lengths of mean 1,024 at 8,192 come out at 896.0, which
// is the continuous closed form μ - μ²(1 - e^(-T/μ))/T to the token.
func bruteGamma(T, mean, cv float64, rows int) float64 {
	g := rand.New(rand.NewSource(7))
	shape, scale := 1/(cv*cv), mean*cv*cv
	draw := func() float64 {
		a, boost := shape, 1.0
		if a < 1 {
			boost = math.Pow(g.Float64(), 1/a)
			a++
		}
		d := a - 1.0/3
		c := 1 / math.Sqrt(9*d)
		for {
			x := g.NormFloat64()
			v := 1 + c*x
			if v <= 0 {
				continue
			}
			v = v * v * v
			if math.Log(g.Float64()) < 0.5*x*x+d-d*v+d*math.Log(v) {
				return math.Max(1, math.Round(d*v*scale*boost))
			}
		}
	}
	// A stream of a few hundred thousand documents, and rows cut from it at
	// random, which overlap. The documents are what have to be many: a
	// reference drawn from three thousand of them is off by two percent
	// through its long ones alone, whatever the number of rows.
	var lengths []float64
	total := 0.0
	for total < math.Max(400*T, 200000*mean) {
		l := draw()
		lengths = append(lengths, l)
		total += l
	}
	begins := make([]float64, len(lengths))
	acc := 0.0
	for i, l := range lengths {
		begins[i] = acc
		acc += l
	}
	sum := 0.0
	for r := 0; r < rows; r++ {
		start := math.Floor(g.Float64() * (total - T))
		i := sort.SearchFloat64s(begins, start+1) - 1
		var pieces []float64
		for at := start; at < start+T; i++ {
			end := math.Min(start+T, begins[i]+lengths[i])
			pieces = append(pieces, end-at)
			at = end
		}
		sum += kept(pieces)
	}
	return asKeys(sum, float64(rows), T)
}

// With no packing a row is one document, and the mask keeps everything
// causal does: a design that keeps documents apart costs what it always did
// until it is told how its batches are packed.
func TestOneDocumentKeepsEverything(t *testing.T) {
	if got := keysPacked(t, 8192, nil); got != 4096 {
		t.Errorf("unpacked: %v keys a query, want 4096", got)
	}
}

// Fixed 1,024-token documents at 8,192. A row that began on a document
// boundary would keep 512.4 keys a query; a stream cut into rows begins one
// anywhere, and the pieces at each end are shorter documents as far as the
// mask can tell: 491.1, averaged over every phase.
func TestFixedDocumentsAtEveryPhase(t *testing.T) {
	want := exactFixed(8192, 1024)
	if math.Abs(want-491.1) > 0.05 {
		t.Fatalf("the reference itself: %v", want)
	}
	near(t, "fixed 1,024 at 8k", keysPacked(t, 8192, &Packing{Mean: 1024}), want, 0.01)
	near(t, "fixed 300 at 4k", keysPacked(t, 4096, &Packing{Mean: 300}), exactFixed(4096, 300), 0.01)
}

// Spread-out lengths cost more at the same mean, because a token lands in a
// long document more often than a short one: the mean a query keeps is
// E[L²]/2E[L], which is μ(1 + c²)/2 — double for exponential lengths.
//
// The sampler is within about two percent of these for spread-out lengths,
// depending on the mask's seed: a packing is a random thing, and it is
// measured from about a quarter of a million evaluations, which is twenty
// milliseconds. Fixed lengths are within half a percent.
func TestSpreadCostsWhatTheSizeBiasSays(t *testing.T) {
	for _, c := range []struct{ T, mean, cv float64 }{
		{8192, 1024, 1}, {8192, 1024, 0.5}, {8192, 256, 2}, {2048, 512, 1},
	} {
		got := keysPacked(t, c.T, &Packing{Mean: c.mean, Spread: c.cv})
		want := bruteGamma(c.T, c.mean, c.cv, 20000)
		near(t, "packed", got, want, 0.025)
	}
	// Far from the edges, the formula itself.
	got := keysPacked(t, 16384, &Packing{Mean: 64, Spread: 1})
	near(t, "exponential, short documents", got, (64.0*(1+1)+1)/2*16384/16385, 0.025)
}

// The same design, the same packing, the same number.
func TestAPackingIsReproducible(t *testing.T) {
	pack := &Packing{Mean: 700, Spread: 1.3}
	first := keysPacked(t, 8192, pack)
	clear(shareCache)
	if again := keysPacked(t, 8192, pack); again != first {
		t.Errorf("%v then %v", first, again)
	}
}

// The quantile is the gamma's: its CDF at the quantile is the share asked for,
// to a part in a thousand. Bisection from a bracket as wide as the mean does no
// better than that in the far lower tail of a very spread distribution, where
// the quantile is a small fraction of a token and a document is one token long
// however it is rounded.
func TestGammaQuantile(t *testing.T) {
	for _, cv := range []float64{0.3, 0.8, 1, 1.7, 3} {
		shape, scale := 1/(cv*cv), 100*cv*cv
		for _, u := range []float64{0.01, 0.25, 0.5, 0.9, 0.999} {
			x := gammaQuantile(u, 100, cv)
			if p := lowerGamma(shape, x/scale); math.Abs(p-u) > 1e-3*u {
				t.Errorf("cv %v, u %v: P at the quantile is %v", cv, u, p)
			}
		}
	}
	// Exponential, where the quantile has a closed form.
	if got, want := gammaQuantile(0.5, 100, 1), 100*math.Ln2; math.Abs(got-want) > 1e-9 {
		t.Errorf("median of an exponential: %v, want %v", got, want)
	}
}

// Fixed documents at every phase, counted as a kernel counts them: the
// diagonal block always, and an earlier block whenever one document reaches
// from inside it to inside this one. Returned as keys a query.
func exactBlocksFixed(T, L float64) float64 {
	blocks := int(math.Ceil(T / kernelBlock))
	sum := 0.0
	for phase := 0.0; phase < L; phase++ {
		doc := func(pos float64) float64 { return math.Floor((pos + phase) / L) }
		for i := 0; i < blocks; i++ {
			first := doc(float64(i) * kernelBlock)
			computed := kernelBlock
			for j := 0; j < i; j++ {
				if doc(float64(j+1)*kernelBlock-1) == first {
					computed += kernelBlock
				}
			}
			sum += float64(computed)
		}
	}
	return sum / L / float64(blocks)
}

// What FlexAttention computes for a mask that keeps documents apart: whole
// blocks, so a boundary through a block costs the part the mask throws away.
// Documents that start on block boundaries would cost 1.12x the scores kept
// at 1,024 tokens and 1.49x at 256; a stream starts them anywhere, and the
// exact counts are 1.37x and 2.48x.
func TestTheKernelComputesWholeBlocks(t *testing.T) {
	def, r := sdpaWith(t, map[string]any{"causal": true, "mask": documentMask})
	a := AttentionOf(r)
	for _, c := range []struct{ T, L, ratio float64 }{{8192, 1024, 1.37}, {8192, 256, 2.48}, {4096, 300, 0}} {
		want := exactBlocksFixed(c.T, c.L)
		got := a.KernelKeys(c.T, Packing{Mean: c.L})
		near(t, "blocks", got, want, 0.02)
		kept := keysPacked(t, c.T, &Packing{Mean: c.L})
		if c.ratio > 0 && math.Abs(got/kept-c.ratio) > 0.05 {
			t.Errorf("fixed %v at %v: the kernel computes %.2fx the scores kept", c.L, c.T, got/kept)
		}
	}
	// And the flops carry it, for the packed training to report.
	f := def.Flops(r, AnalysisCtx{T: 8192, B: 1, Bytes: 2, Flash: true, Packing: &Packing{Mean: 256}})
	if f.FwdSeqBlocks <= f.FwdSeq {
		t.Errorf("blocks %v against scores %v", f.FwdSeqBlocks, f.FwdSeq)
	}
	if g := def.Flops(r, AnalysisCtx{T: 8192, B: 1, Bytes: 2, Flash: true}); g.FwdSeqBlocks != 0 {
		t.Error("a block figure with no packing")
	}
}
