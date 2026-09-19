package codegen

import (
	"math"
	"math/big"
	"strconv"
	"strings"

	"github.com/tensorcad/core/analysis"
)

// pyName turns any identifier into one Python will accept.
func pyName(raw string) string {
	var b strings.Builder
	for _, r := range raw {
		switch {
		case r >= 'A' && r <= 'Z', r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	s := b.String()
	if s != "" && s[0] >= '0' && s[0] <= '9' {
		return "_" + s
	}
	return s
}

// pascal joins the words of a name into a Python class name.
func pascal(base string) string {
	var parts []string
	cur := strings.Builder{}
	for _, r := range base {
		if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			cur.WriteRune(r)
			continue
		}
		if cur.Len() > 0 {
			parts = append(parts, cur.String())
			cur.Reset()
		}
	}
	if cur.Len() > 0 {
		parts = append(parts, cur.String())
	}
	var out strings.Builder
	for _, p := range parts {
		out.WriteString(strings.ToUpper(p[:1]))
		out.WriteString(p[1:])
	}
	return out.String()
}

func pyBool(v any) string {
	if b, ok := v.(bool); ok && b {
		return "True"
	}
	return "False"
}

// pyNum writes a number into generated Python the way the TypeScript's template
// literals do, which is JavaScript's String().
func pyNum(v float64) string { return analysis.JSNumber(v) }

// pyValue writes a resolved parameter into generated Python.
func pyValue(v any) string {
	switch n := v.(type) {
	case float64:
		return pyNum(n)
	case int:
		return strconv.Itoa(n)
	case bool:
		return pyBool(n)
	case string:
		return n
	case nil:
		return "None"
	}
	return "None"
}

// jsToPrecision formats a number with a fixed number of significant digits, the
// way Number.prototype.toPrecision does.
//
// Two things separate it from Go's %.*g. JavaScript keeps trailing zeros, where
// Go strips them, so 0.5 at eight digits is "0.50000000" and not "0.5". And it
// switches to exponential only below 1e-7 or at or above 10^precision, where Go
// switches below 1e-4. Both differences land in generated Python as a literal
// somebody could read, so they are worth getting exactly right.
func jsToPrecision(x float64, precision int) string {
	if math.IsNaN(x) {
		return "NaN"
	}
	if math.IsInf(x, 1) {
		return "Infinity"
	}
	if math.IsInf(x, -1) {
		return "-Infinity"
	}
	sign := ""
	if math.Signbit(x) {
		sign, x = "-", -x
	}
	if x == 0 {
		if precision == 1 {
			return sign + "0"
		}
		return sign + "0." + strings.Repeat("0", precision-1)
	}

	// digits is the number rounded to `precision` significant digits, as an
	// integer, and exp is the power of ten the leading digit sits at.
	r := new(big.Rat).SetFloat64(x)
	exp := 0
	ten := big.NewRat(10, 1)
	one := big.NewRat(1, 1)
	for r.Cmp(ten) >= 0 {
		r.Quo(r, ten)
		exp++
	}
	for r.Cmp(one) < 0 {
		r.Mul(r, ten)
		exp--
	}
	// r is now in [1, 10): shift it to `precision` digits and round, ties up.
	scale := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(precision-1)), nil)
	r.Mul(r, new(big.Rat).SetInt(scale))
	r.Add(r, big.NewRat(1, 2))
	digits := new(big.Int).Div(r.Num(), r.Denom()).String()
	// Rounding 9.99 up to two digits gives 100, one digit too many.
	if len(digits) > precision {
		digits = digits[:precision]
		exp++
	}

	if exp < -6 || exp >= precision {
		mantissa := digits[:1]
		if precision > 1 {
			mantissa += "." + digits[1:]
		}
		e := "+"
		if exp < 0 {
			e = "-"
		}
		return sign + mantissa + "e" + e + strconv.Itoa(abs(exp))
	}
	if exp >= 0 {
		whole := digits[:exp+1]
		frac := digits[exp+1:]
		if frac == "" {
			return sign + whole
		}
		return sign + whole + "." + frac
	}
	return sign + "0." + strings.Repeat("0", -exp-1) + digits
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// localeInt groups a number into thousands with commas, as
// toLocaleString("en-US") does. Up to three fraction digits, none of them
// trailing zeros, which is that locale's default.
func localeInt(v float64) string {
	sign := ""
	if math.Signbit(v) {
		sign, v = "-", -v
	}
	s := analysis.JSToFixed(v, 3)
	whole, frac, _ := strings.Cut(s, ".")
	frac = strings.TrimRight(frac, "0")

	var parts []string
	for len(whole) > 3 {
		parts = append([]string{whole[len(whole)-3:]}, parts...)
		whole = whole[:len(whole)-3]
	}
	parts = append([]string{whole}, parts...)
	out := sign + strings.Join(parts, ",")
	if frac != "" {
		out += "." + frac
	}
	return out
}
