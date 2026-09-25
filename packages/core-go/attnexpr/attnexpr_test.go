package attnexpr

import (
	"math"
	"strings"
	"testing"
)

func compile(t *testing.T, src string, kind Kind, symbols map[string]float64) Node {
	t.Helper()
	n, err := Compile(src, kind, symbols)
	if err != nil {
		t.Fatalf("%q: %v", src, err)
	}
	return n
}

func TestAMaskMeansWhatItSays(t *testing.T) {
	cases := []struct {
		src    string
		q, kv  float64
		expect bool
	}{
		{"kv <= q", 3, 2, true},
		{"kv <= q", 2, 3, false},
		{"kv <= q and q - kv < 4", 10, 7, true},
		{"kv <= q and q - kv < 4", 10, 6, false},
		{"kv <= q or kv < 2", 0, 1, true},
		{"kv <= q or kv < 2", 0, 3, false},
		{"not kv > q", 3, 3, true},
		{"kv <= q && !(q - kv >= 4)", 5, 1, false},
		{"(q + 1) % 2 == 0 or kv == q", 1, 0, true},
	}
	for _, c := range cases {
		n := compile(t, c.src, Mask, nil)
		got := Eval(n, Env{Q: c.q, KV: c.kv}) != 0
		if got != c.expect {
			t.Errorf("%q at q=%v kv=%v: %v, want %v", c.src, c.q, c.kv, got, c.expect)
		}
	}
}

func TestPrecedenceIsPythons(t *testing.T) {
	cases := map[string]float64{
		"-2 ** 2":     -4,
		"2 ** 3 ** 2": 512,
		"2 ** -1":     0.5,
		"1 + 2 * 3":   7,
		"(1 + 2) * 3": 9,
		"7 % 4 * 2":   6,
		"10 - 4 - 3":  3,
		"8 / 4 / 2":   1,
	}
	for src, want := range cases {
		n, err := Parse(src)
		if err != nil {
			t.Fatalf("%q: %v", src, err)
		}
		if got := Eval(n, Env{}); got != want {
			t.Errorf("%q = %v, want %v", src, got, want)
		}
	}
}

func TestSymbolsAreBoundAndFolded(t *testing.T) {
	n := compile(t, "kv < P or kv <= q", Mask, map[string]float64{"P": 16})
	if got := String(n); got != "kv < 16 or kv <= q" {
		t.Errorf("printed %q", got)
	}
	n = compile(t, "C * tanh(score / C)", Score, map[string]float64{"C": 50})
	if got := String(n); got != "50 * tanh(score / 50)" {
		t.Errorf("printed %q", got)
	}
	// A constant subexpression costs nothing per score.
	n = compile(t, "score + 2 ** 3", Score, nil)
	if got := String(n); got != "score + 8" {
		t.Errorf("printed %q", got)
	}
	if c := Cost(n); c != 1 {
		t.Errorf("cost %v, want 1", c)
	}
}

func TestPrintingKeepsOnlyTheParenthesesItNeeds(t *testing.T) {
	cases := map[string]string{
		"((kv <= q))":                    "kv <= q",
		"(a - b) - c":                    "a - b - c",
		"a - (b - c)":                    "a - (b - c)",
		"(2 ** 3) ** h":                  "8 ** h",
		"2 ** (h ** 2)":                  "2 ** h ** 2",
		"(h ** 2) ** 3":                  "(h ** 2) ** 3",
		"-(q - kv)":                      "-(q - kv)",
		"not (kv > q or kv < 0)":         "not (kv > q or kv < 0)",
		"(kv <= q and q < 4) or kv == 0": "kv <= q and q < 4 or kv == 0",
		"kv <= q and (q < 4 or kv == 0)": "kv <= q and (q < 4 or kv == 0)",
	}
	for src, want := range cases {
		n, err := Parse(src)
		if err != nil {
			t.Fatalf("%q: %v", src, err)
		}
		if got := String(Fold(n)); got != want {
			t.Errorf("%q printed %q, want %q", src, got, want)
		}
		// And it reads back as the same thing.
		again, err := Parse(String(Fold(n)))
		if err != nil || String(Fold(again)) != want {
			t.Errorf("%q does not round-trip: %v", want, err)
		}
	}
}

func TestPythonIsFullyParenthesised(t *testing.T) {
	n := compile(t, "kv <= q and not q - kv >= W", Mask, map[string]float64{"W": 8})
	want := "(kv_idx <= q_idx) & ~(((q_idx - kv_idx) >= 8))"
	if got := Python(n, 12); got != want {
		t.Errorf("got %s\nwant %s", got, want)
	}
	n = compile(t, "score - 2 ** (-8 * (h + 1) / heads) * (q - kv)", Score, nil)
	want = "score - ((2.0 ** (((-8) * (h + 1)) / 12)) * (q_idx - kv_idx))"
	if got := Python(n, 12); got != want {
		t.Errorf("got %s\nwant %s", got, want)
	}
	// A negative constant keeps its parentheses on the left of a power, in
	// both languages, or it becomes the negation of the power.
	n = compile(t, "score * (0 - 2) ** h", Score, nil)
	if got := String(n); got != "score * (-2) ** h" {
		t.Errorf("printed %q", got)
	}
	if got := Eval(compile(t, String(n), Score, nil), Env{Score: 1, H: 2}); got != 4 {
		t.Errorf("reads back as %v", got)
	}
	if got := Python(n, 1); got != "score * ((-2.0) ** h)" {
		t.Errorf("python %q", got)
	}
	n = compile(t, "min(score, 5) + max(1, h) + min(q, kv) + tanh(score)", Score, nil)
	want = "((torch.clamp(score, max=5) + torch.clamp(h, min=1)) + torch.minimum(q_idx, kv_idx)) + torch.tanh(score)"
	if got := Python(n, 1); got != want {
		t.Errorf("got %s\nwant %s", got, want)
	}
}

func TestMistakesAreNamed(t *testing.T) {
	cases := []struct {
		src  string
		kind Kind
		want string
	}{
		{"q - kv", Mask, "a mask is true or false"},
		{"score > 0", Mask, "cannot read the score"},
		{"kv <= q", Score, "a score expression is a number"},
		{"kv <= q and 3", Mask, "joins comparisons"},
		{"kv < q < 3", Mask, "do not chain"},
		{"tanh(score, 2)", Score, "takes 1 argument"},
		{"sigmoid(q) > 0 and kv <= q", Mask, "not a function this knows"},
		{"score + rel()", Score, "needs to be told where"},
		{"score + rel(kv <= q)", Score, "read at numbers"},
		{"score + rel(t5_bucket(kv - q, B, 128, true), h)", Score, `"B" is not`},
		{"score + rel(t5_bucket(kv - q, 32, 128, 1), h)", Score, "true or false"},
		{"where(q, 1, 2)", Score, "first argument is a comparison"},
		{"kv <= Q", Mask, `"Q" is not`},
		{"kv <= (q", Mask, "not closed"},
		{"kv <=", Mask, "ends too soon"},
		{"kv $ q", Mask, "unexpected"},
		{"and kv", Mask, "needs something on each side"},
		{"-(kv <= q)", Mask, "negated with -"},
		{"1 < 2", Mask, "true for every score"},
		{"8 < 4 and kv <= q", Mask, "false for every score"},
		{"q - kv", Score, "throws the attention scores away"},
	}
	for _, c := range cases {
		_, err := Compile(c.src, c.kind, nil)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%q: got %v, want an error containing %q", c.src, err, c.want)
		}
	}
}

func TestConstantComparisonsFoldAway(t *testing.T) {
	syms := map[string]float64{"W": 8, "P": 0}
	n := compile(t, "kv <= q and (W > 4 or q < kv)", Mask, syms)
	if got := String(n); got != "kv <= q" {
		t.Errorf("printed %q", got)
	}
	n = compile(t, "kv <= q or P > 0 and kv < P", Mask, syms)
	if got := String(n); got != "kv <= q" {
		t.Errorf("printed %q", got)
	}
	n = compile(t, "where(W > 4, score, -score)", Score, syms)
	if got := String(n); got != "score" {
		t.Errorf("printed %q", got)
	}
	n = compile(t, "kv <= q and not false", Mask, nil)
	if got := String(n); got != "kv <= q" {
		t.Errorf("printed %q", got)
	}
}

func TestHeadsIsFixedPerBlock(t *testing.T) {
	n := With(compile(t, "kv <= q and (heads > 4 or h == 0)", Mask, nil), "heads", 8)
	if got := String(n); got != "kv <= q" {
		t.Errorf("printed %q", got)
	}
	// A mask that folds away entirely still prints as a tensor.
	n = With(compile(t, "heads > 4 or h == 0", Mask, nil), "heads", 8)
	if got := Python(n, 8); got != "kv_idx >= 0" {
		t.Errorf("python %q", got)
	}
	n = With(compile(t, "where(heads > 4, 0, score)", Score, nil), "heads", 8)
	if got := Python(n, 8); got != "torch.full_like(score, 0)" {
		t.Errorf("python %q", got)
	}
}

func TestModuloIsPythons(t *testing.T) {
	n := compile(t, "(q - kv) % 4 == 1", Mask, nil)
	// -3 % 4 is 1 in Python, and -3 in Go's math.Mod.
	if Eval(n, Env{Q: 0, KV: 3}) != 1 {
		t.Error("(0 - 3) % 4 should be 1")
	}
	n = compile(t, "score + (0 - 3) % 4", Score, nil)
	if got := String(n); got != "score + 1" {
		t.Errorf("printed %q", got)
	}
}

func TestScoreFunctions(t *testing.T) {
	n := compile(t, "where(kv < q, min(score, 1), max(score, -1)) + abs(h) + floor(q / 2)", Score, nil)
	got := Eval(n, Env{Q: 5, KV: 1, H: -2, Score: 3})
	if got != 1+2+2 {
		t.Errorf("got %v", got)
	}
	n = compile(t, "sqrt(exp(log(score)))", Score, nil)
	if got := Eval(n, Env{Score: 4}); math.Abs(got-2) > 1e-12 {
		t.Errorf("got %v", got)
	}
}

// T5's buckets, at values worked through Hugging Face's
// _relative_position_bucket by hand: exact for small distances, logarithmic
// out to 128, two-sided spending half its buckets on each side.
func TestT5Buckets(t *testing.T) {
	cases := []struct {
		relative      float64
		bidirectional bool
		want          float64
	}{
		{0, true, 0}, {-1, true, 1}, {1, true, 17}, {-7, true, 7},
		{-8, true, 8},                 // the first logarithmic bucket
		{-20, true, 10},               // 8 + trunc(log(20/8) / log(128/8) * 8)
		{200, true, 31},               // far on the right: the last bucket of that side
		{-1, false, 1}, {3, false, 0}, // one-sided: the future is bucket 0
		{-20, false, 17}, // 16 + trunc(log(20/16) / log(128/16) * 16)
		{-1000, false, 31},
	}
	for _, c := range cases {
		if got := T5Bucket(c.relative, 32, 128, c.bidirectional); got != c.want {
			t.Errorf("relative %v, two-sided %v: bucket %v, want %v", c.relative, c.bidirectional, got, c.want)
		}
	}
}

// A score can read a tensor wired into the attention by name, and the
// attention learns from Tables what inputs it needs.
func TestAScoreReadsATable(t *testing.T) {
	n := compile(t, "score + rel(t5_bucket(kv - q, 32, 128, true), h)", Score, nil)
	if got := Tables(n); len(got) != 1 || got[0] != "rel" {
		t.Errorf("tables %v", got)
	}
	if got := String(n); got != "score + rel(t5_bucket(kv - q, 32, 128, true), h)" {
		t.Errorf("printed %q", got)
	}
	if got := Python(n, 8); got != "score + rel[t5_bucket(kv_idx - q_idx, 32, 128, True), h]" {
		t.Errorf("python %q", got)
	}
	// A read is one operation; the bucket is a dozen.
	if c := Cost(n); c != 1+1+12+1 {
		t.Errorf("cost %v", c)
	}
	// A table's values are the model's: nothing here can evaluate one.
	if v := Eval(n, Env{Q: 3, KV: 1}); !math.IsNaN(v) {
		t.Errorf("evaluated a table to %v", v)
	}
}
