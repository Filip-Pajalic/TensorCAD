package analysis_test

import (
	"testing"

	"github.com/tensorcad/core/analysis"
)

// The expected values were read off a JavaScript runtime, not derived: the
// point of this table is to be what the other engine actually does, including
// where that is surprising. 62.5 rounding to 63 rather than 62 is the case Go's
// own %.0f gets differently, and it is not rare here, because the memory model
// works in powers of two.
func TestJSToFixedMatchesJavaScript(t *testing.T) {
	cases := []struct {
		x      float64
		digits int
		want   string
	}{
		{0.625, 0, "1"},
		{24.5, 0, "25"},
		{25.5, 0, "26"},
		{-24.5, 0, "-25"},
		{-0.4, 0, "-0"},
		{0.125, 2, "0.13"},
		// 1.005 is really 1.00499999..., so it rounds down in both engines.
		{1.005, 2, "1.00"},
		{8.575, 2, "8.57"},
		{0.615, 2, "0.61"},
		{2.5, 0, "3"},
		{1.5, 0, "2"},
		{0.5, 0, "1"},
		{1234.5678, 2, "1234.57"},
		{12.345, 1, "12.3"},
		{0, 2, "0.00"},
		{1e20, 2, "100000000000000000000.00"},
		// At 1e21 toFixed gives up and returns what String() would.
		{1e21, 2, "1e+21"},
		{99.995, 2, "100.00"},
		{37.5, 0, "38"},
		{62.5, 0, "63"},
	}
	for _, c := range cases {
		if got := analysis.JSToFixed(c.x, c.digits); got != c.want {
			t.Errorf("JSToFixed(%v, %d) = %q, want %q", c.x, c.digits, got, c.want)
		}
	}
}

func TestFormattersReadTheWayTheyShould(t *testing.T) {
	cases := []struct{ got, want string }{
		{analysis.FormatCount(8030000000), "8.03B"},
		{analysis.FormatCount(124400000), "124.4M"},
		{analysis.FormatCount(12900), "12.9K"},
		{analysis.FormatCount(768), "768"},
		{analysis.FormatBytes(320 * 1024), "320.00 KiB"},
		{analysis.FormatBytes(512), "512 B"},
		{analysis.FormatFlops(312e12), "312.00 TFLOP"},
		{analysis.FormatHours(0.5), "30.0 min"},
		{analysis.FormatHours(96), "4.0 days"},
		{analysis.FormatDollars(1.5e6), "$1.50M"},
		{analysis.FormatDollars(4500), "$4.5k"},
		{analysis.FormatDollars(12.5), "$12.50"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("got %q, want %q", c.got, c.want)
		}
	}
}
