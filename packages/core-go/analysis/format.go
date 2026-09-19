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
// written by either engine reads the same: 4 rather than 4.000000.
func JSNumber(v float64) string {
	if math.IsInf(v, 1) {
		return "Infinity"
	}
	if math.IsInf(v, -1) {
		return "-Infinity"
	}
	if math.IsNaN(v) {
		return "NaN"
	}
	s := strconv.FormatFloat(v, 'g', -1, 64)
	// Go writes an exponent as e+21 and a two-digit e-07; JavaScript writes
	// e+21 and 1e-7, and only leaves the fixed notation outside 1e21 and 1e-7.
	if i := strings.IndexAny(s, "eE"); i >= 0 {
		mantissa, exp := s[:i], s[i+1:]
		signPart := ""
		if exp[0] == '+' || exp[0] == '-' {
			signPart, exp = string(exp[0]), exp[1:]
		}
		exp = strings.TrimLeft(exp, "0")
		if exp == "" {
			exp = "0"
		}
		if signPart == "" {
			signPart = "+"
		}
		s = mantissa + "e" + signPart + exp
	}
	return s
}
