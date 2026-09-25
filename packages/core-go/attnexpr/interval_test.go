package attnexpr

import (
	"math/rand"
	"testing"
)

// Possible never says no where a point says yes: over random boxes, every
// point in the box is evaluated and held against what Possible claimed.
func TestPossibleNeverMissesAPoint(t *testing.T) {
	masks := []string{
		"kv <= q",
		"q - kv < 7 and kv <= q",
		"kv <= q or kv < 3",
		"(q - kv) % 4 == 0",
		"floor(q / 5) == floor(kv / 5)",
		"h < heads / 2 or q - kv < 3",
		"not kv > q",
		"doc(b, q) == doc(b, kv)",
		"doc(b, q) == doc(b, kv) and kv <= q and q - kv < 9",
		"doc(b, q) != doc(b, kv) or kv == q",
		"abs(q - kv) <= 2 and max(q, kv) > 10",
	}
	g := rand.New(rand.NewSource(3))
	// Rows of monotone document ids, as a packing makes them.
	docs := make([][]float64, 3)
	for r := range docs {
		id := 0.0
		for p := 0; p < 64; p++ {
			if g.Intn(7) == 0 {
				id++
			}
			docs[r] = append(docs[r], id)
		}
	}
	read := func(_ string, at []float64) float64 { return docs[int(at[0])][int(at[1])] }
	span := func(_ string, at []Span) Span {
		row := docs[int(at[0].Lo)]
		return Span{row[int(at[1].Lo)], row[int(at[1].Hi)]}
	}
	for _, src := range masks {
		n := compile(t, src, Mask, nil)
		n = With(n, "heads", 4)
		for trial := 0; trial < 400; trial++ {
			q0, k0 := g.Intn(56), g.Intn(56)
			box := Box{
				Q:     Span{float64(q0), float64(q0 + g.Intn(8))},
				KV:    Span{float64(k0), float64(k0 + g.Intn(8))},
				H:     Span{0, float64(g.Intn(4))},
				B:     point(float64(g.Intn(3))),
				Heads: 4, Table: span,
			}
			canTrue, canFalse := Possible(n, box)
			sawTrue, sawFalse := false, false
			for q := box.Q.Lo; q <= box.Q.Hi; q++ {
				for kv := box.KV.Lo; kv <= box.KV.Hi; kv++ {
					for h := box.H.Lo; h <= box.H.Hi; h++ {
						v := Eval(n, Env{Q: q, KV: kv, H: h, B: box.B.Lo, Heads: 4, Table: read})
						sawTrue = sawTrue || v != 0
						sawFalse = sawFalse || v == 0
					}
				}
			}
			if sawTrue && !canTrue || sawFalse && !canFalse {
				t.Fatalf("%q over %+v: said %v/%v, saw %v/%v", src, box, canTrue, canFalse, sawTrue, sawFalse)
			}
		}
	}
}

// For a mask that keeps documents apart, read against ids that never fall
// along a row, the answer is exact: a block is possible exactly when some
// document reaches into both of its ranges.
func TestPossibleIsExactForDocuments(t *testing.T) {
	n := compile(t, "doc(b, q) == doc(b, kv) and kv <= q", Mask, nil)
	docs := []float64{0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3}
	span := func(_ string, at []Span) Span { return Span{docs[int(at[1].Lo)], docs[int(at[1].Hi)]} }
	read := func(_ string, at []float64) float64 { return docs[int(at[1])] }
	for q0 := 0; q0 < 16; q0 += 4 {
		for k0 := 0; k0 <= q0; k0 += 4 {
			box := Box{Q: Span{float64(q0), float64(q0 + 3)}, KV: Span{float64(k0), float64(k0 + 3)}, Table: span}
			canTrue, _ := Possible(n, box)
			saw := false
			for q := q0; q < q0+4; q++ {
				for kv := k0; kv < k0+4; kv++ {
					saw = saw || Eval(n, Env{Q: float64(q), KV: float64(kv), Table: read}) != 0
				}
			}
			if canTrue != saw {
				t.Errorf("block %d,%d: said %v, is %v", q0/4, k0/4, canTrue, saw)
			}
		}
	}
}

func TestBoundsOfArithmetic(t *testing.T) {
	box := Box{Q: Span{2, 5}, KV: Span{-1, 3}}
	for src, want := range map[string]Span{
		"q - kv":       {-1, 6},
		"q * kv":       {-5, 15},
		"abs(kv)":      {0, 3},
		"min(q, kv)":   {-1, 3},
		"floor(q / 2)": {1, 2},
		"(q + 1) % 4":  {0, 4},
		"2 ** q":       {4, 32},
	} {
		n, err := Parse(src)
		if err != nil {
			t.Fatal(err)
		}
		if got := Bounds(n, box); got != want {
			t.Errorf("%s: %+v, want %+v", src, got, want)
		}
	}
}
