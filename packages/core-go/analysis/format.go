package analysis

import (
	"math"
	"math/big"
	"strconv"
	"strings"
)

// JSToFixed formats a number with a fixed number of decimals, the way
// Number.prototype.toFixed does.
//
// Not fmt's %.*f: Go rounds a half to even and JavaScript rounds it away from
// zero, so 0.625 at zero decimals is "63" in a browser and "62" in Go. That is
// not a hypothetical here. These strings go into the messages a person reads,
// and the memory model produces exact halves regularly because its numbers are
// powers of two.
//
// The rounding is done on the double's exact value rather than on a scaled
// float, so it cannot go wrong a second way: x*100 is not always the number
// JavaScript was looking at when it decided which way to round.
func JSToFixed(x float64, digits int) string {
	if math.IsNaN(x) {
		return "NaN"
	}
	// toFixed hands very large numbers and infinities to ToString.
	if math.IsInf(x, 0) || math.Abs(x) >= 1e21 {
		return strconv.FormatFloat(x, 'g', -1, 64)
	}

	// The sign comes off first, as the specification does it: the magnitude is
	// rounded with ties going up, and the sign goes back on afterwards. That is
	// why -24.5 is "-25" and not "-24".
	sign := ""
	if math.Signbit(x) {
		sign = "-"
		x = -x
	}

	scale := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(digits)), nil)
	r := new(big.Rat).SetFloat64(x)
	r.Mul(r, new(big.Rat).SetInt(scale))
	r.Add(r, big.NewRat(1, 2))
	// Denominators are positive, so this floors, which with the half already
	// added is "round to nearest, ties up".
	n := new(big.Int).Div(r.Num(), r.Denom())

	s := n.String()
	if digits > 0 {
		for len(s) <= digits {
			s = "0" + s
		}
		s = s[:len(s)-digits] + "." + s[len(s)-digits:]
	}
	return sign + s
}

// JSNumber prints a number the way JavaScript's String() would, so a message
// or a generated literal written by either engine reads the same.
//
// It is the ECMAScript Number::toString algorithm rather than any of Go's
// formats, because the two languages put the boundary between fixed and
// exponential notation in different places: 1e-5 is "0.00001" in JavaScript and
// "1e-05" under Go's %g. Both spellings end up in generated Python, where they
// would be two different files for the same design.
func JSNumber(v float64) string {
	switch {
	case math.IsInf(v, 1):
		return "Infinity"
	case math.IsInf(v, -1):
		return "-Infinity"
	case math.IsNaN(v):
		return "NaN"
	case v == 0:
		// Including negative zero, which JavaScript prints as "0".
		return "0"
	}
	sign := ""
	if v < 0 {
		sign, v = "-", -v
	}

	// The shortest representation that round-trips, which is the digit string
	// the specification calls s, and the exponent it calls n.
	mantissa, exp := splitExponential(strconv.FormatFloat(v, 'e', -1, 64))
	digits := strings.Replace(mantissa, ".", "", 1)
	k := len(digits)
	n := exp + 1

	switch {
	case k <= n && n <= 21:
		return sign + digits + strings.Repeat("0", n-k)
	case 0 < n && n <= 21:
		return sign + digits[:n] + "." + digits[n:]
	case -6 < n && n <= 0:
		return sign + "0." + strings.Repeat("0", -n) + digits
	}
	// Exponential, with the point after the first digit.
	out := digits[:1]
	if k > 1 {
		out += "." + digits[1:]
	}
	e := n - 1
	if e >= 0 {
		return sign + out + "e+" + strconv.Itoa(e)
	}
	return sign + out + "e-" + strconv.Itoa(-e)
}

// splitExponential takes Go's "1.5e+07" apart into its mantissa and exponent.
func splitExponential(s string) (string, int) {
	i := strings.IndexAny(s, "eE")
	if i < 0 {
		return s, 0
	}
	exp, err := strconv.Atoi(s[i+1:])
	if err != nil {
		return s[:i], 0
	}
	return s[:i], exp
}
