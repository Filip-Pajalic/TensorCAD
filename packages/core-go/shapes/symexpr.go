// Exact symbolic arithmetic for tensor dimensions.
//
// A Sym is a multivariate polynomial with rational coefficients over named
// symbols (B, T, D, H, dh, ...). This is exactly the expressive power tensor
// shapes need: products for reshapes, sums for concatenation, and exact
// divisibility obligations for splits.
//
// Design symbols (D, H, F, ...) always have a concrete numeric value; runtime
// symbols (B, T) stay indeterminate. Two shapes are compatible when their
// difference is the zero polynomial after substituting the concrete values.
package shapes

import (
	"fmt"
	"math"
	"math/big"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// Rational numbers
// ---------------------------------------------------------------------------

var (
	ratOne = new(big.Rat).SetInt64(1)
)

// RatFrom builds a rational from a float.
//
// Decimal literals have to come out as the decimal the author wrote: RatFrom of
// 1.3 is 13/10, not the binary 5854679515581645/4503599627370496 that
// SetFloat64 would give, because 1.3*8/3*D is how a feed-forward width is
// written and 13/10 is what it means. The scaling loop is the TypeScript
// engine's, kept so the two agree term for term.
//
// Numerator and denominator are arbitrary precision here rather than the
// doubles the TypeScript carried, which are exact to 2^53 and silently wrong
// past it. A 671-billion-parameter model is comfortably inside that; its FLOP
// counts are not.
func RatFrom(n float64) *big.Rat { return RatFromPair(n, 1) }

// RatFromPair builds n/d, scaling both until they are whole.
//
// It is the TypeScript engine's `rat()`, kept step for step rather than
// improved. Scaling by ten until both sides are integral is not the same as
// exact conversion — 1/3 comes out as 333333333333/1000000000000 — and making
// it exact here would have the two engines print different shapes for the same
// design. Faithful first; the imprecision can be fixed in both at once.
func RatFromPair(n, d float64) *big.Rat {
	if d == 0 {
		panic("shapes: division by zero")
	}
	if math.IsInf(n, 0) || math.IsNaN(n) || math.IsInf(d, 0) || math.IsNaN(d) {
		panic("shapes: non-finite value")
	}
	if isWhole(n) && isWhole(d) && math.Abs(n) < 1e18 && math.Abs(d) < 1e18 {
		return new(big.Rat).SetFrac64(int64(n), int64(d))
	}
	scale := 1.0
	for (!isWhole(n*scale) || !isWhole(d*scale)) && scale < 1e12 {
		scale *= 10
	}
	return new(big.Rat).SetFrac64(int64(math.Round(n*scale)), int64(math.Round(d*scale)))
}

func isWhole(v float64) bool { return v == math.Trunc(v) && !math.IsInf(v, 0) }

// RatFromDecimal parses a decimal literal exactly, so "1.3" gives 13/10.
func RatFromDecimal(text string) (*big.Rat, error) {
	r, ok := new(big.Rat).SetString(text)
	if !ok {
		return nil, fmt.Errorf("bad number %q", text)
	}
	return r, nil
}

func ratIsZero(a *big.Rat) bool { return a.Sign() == 0 }
func ratIsOne(a *big.Rat) bool  { return a.Cmp(ratOne) == 0 }

func ratString(a *big.Rat) string {
	if a.IsInt() {
		return a.Num().String()
	}
	return a.Num().String() + "/" + a.Denom().String()
}

// ---------------------------------------------------------------------------
// Monomials
// ---------------------------------------------------------------------------

// Mono is the exponent per symbol. An absent symbol means exponent 0.
type Mono map[string]int

func monoKey(m Mono) string {
	names := make([]string, 0, len(m))
	for k, e := range m {
		if e != 0 {
			names = append(names, k)
		}
	}
	if len(names) == 0 {
		return ""
	}
	sort.Strings(names)
	parts := make([]string, len(names))
	for i, k := range names {
		if m[k] == 1 {
			parts[i] = k
		} else {
			parts[i] = fmt.Sprintf("%s^%d", k, m[k])
		}
	}
	return strings.Join(parts, "*")
}

func monoMul(a, b Mono) Mono {
	out := make(Mono, len(a)+len(b))
	for k, e := range a {
		out[k] = e
	}
	for k, e := range b {
		if n := out[k] + e; n == 0 {
			delete(out, k)
		} else {
			out[k] = n
		}
	}
	return out
}

func monoDegree(m Mono) int {
	d := 0
	for _, e := range m {
		d += e
	}
	return d
}

type term struct {
	c *big.Rat
	m Mono
}

// ---------------------------------------------------------------------------
// Sym
// ---------------------------------------------------------------------------

// Sym is a multivariate polynomial in canonical form: one term per distinct
// monomial, never with a zero coefficient. The zero value is the zero
// polynomial, so a declared Sym is usable without construction.
//
// Every method treats the receiver as immutable and returns a fresh value.
type Sym struct {
	terms map[string]term
}

// Zero is the zero polynomial.
func Zero() Sym { return Sym{} }

// Con is a constant.
func Con(v float64) Sym { return ConRat(RatFrom(v)) }

// ConRat is a constant from an exact rational.
func ConRat(r *big.Rat) Sym {
	if ratIsZero(r) {
		return Sym{}
	}
	return Sym{terms: map[string]term{"": {c: new(big.Rat).Set(r), m: Mono{}}}}
}

// V is a single named symbol.
func V(name string) Sym {
	return Sym{terms: map[string]term{name: {c: new(big.Rat).Set(ratOne), m: Mono{name: 1}}}}
}

// Sum adds every part.
func Sum(parts ...Sym) Sym {
	acc := Zero()
	for _, p := range parts {
		acc = acc.Add(p)
	}
	return acc
}

// Product multiplies every part.
func Product(parts ...Sym) Sym {
	acc := Con(1)
	for _, p := range parts {
		acc = acc.Mul(p)
	}
	return acc
}

func (s Sym) clone() map[string]term {
	out := make(map[string]term, len(s.terms))
	for k, t := range s.terms {
		out[k] = t
	}
	return out
}

// Add returns s + other.
func (s Sym) Add(other Sym) Sym {
	out := s.clone()
	for k, t := range other.terms {
		cur, ok := out[k]
		if !ok {
			out[k] = t
			continue
		}
		c := new(big.Rat).Add(cur.c, t.c)
		if ratIsZero(c) {
			delete(out, k)
		} else {
			out[k] = term{c: c, m: cur.m}
		}
	}
	return Sym{terms: out}
}

// Neg returns -s.
func (s Sym) Neg() Sym {
	out := make(map[string]term, len(s.terms))
	for k, t := range s.terms {
		out[k] = term{c: new(big.Rat).Neg(t.c), m: t.m}
	}
	return Sym{terms: out}
}

// Sub returns s - other.
func (s Sym) Sub(other Sym) Sym { return s.Add(other.Neg()) }

// Mul returns s * other.
func (s Sym) Mul(other Sym) Sym {
	out := map[string]term{}
	for _, a := range s.terms {
		for _, b := range other.terms {
			m := monoMul(a.m, b.m)
			k := monoKey(m)
			c := new(big.Rat).Mul(a.c, b.c)
			cur, ok := out[k]
			if !ok {
				out[k] = term{c: c, m: m}
				continue
			}
			nc := new(big.Rat).Add(cur.c, c)
			if ratIsZero(nc) {
				delete(out, k)
			} else {
				out[k] = term{c: nc, m: m}
			}
		}
	}
	return Sym{terms: out}
}

// Pow raises s to a non-negative integer power.
func (s Sym) Pow(k int) (Sym, error) {
	if k < 0 {
		return Zero(), fmt.Errorf("shapes: unsupported exponent %d", k)
	}
	acc := Con(1)
	for i := 0; i < k; i++ {
		acc = acc.Mul(s)
	}
	return acc, nil
}

// DivExact divides exactly.
//
// It succeeds when other is a single term — a constant or a monomial — that
// divides every term of s. It reports ok=false when divisibility cannot be
// proven, which callers turn into a design-rule obligation rather than an
// approximation.
func (s Sym) DivExact(other Sym) (Sym, bool) {
	if other.IsZero() {
		return Zero(), false
	}
	if len(other.terms) != 1 {
		if oc, ok := other.AsConst(); ok {
			return s.DivExact(Con(oc))
		}
		return Zero(), false
	}
	var b term
	for _, t := range other.terms {
		b = t
	}
	out := map[string]term{}
	for _, a := range s.terms {
		m := make(Mono, len(a.m))
		for k, e := range a.m {
			m[k] = e
		}
		for k, e := range b.m {
			n := m[k] - e
			if n < 0 {
				return Zero(), false
			}
			if n == 0 {
				delete(m, k)
			} else {
				m[k] = n
			}
		}
		out[monoKey(m)] = term{c: new(big.Rat).Quo(a.c, b.c), m: m}
	}
	return Sym{terms: out}, true
}

// IsZero reports whether this is the zero polynomial.
func (s Sym) IsZero() bool { return len(s.terms) == 0 }

// AsConst returns the numeric value when s is a constant.
func (s Sym) AsConst() (float64, bool) {
	if len(s.terms) == 0 {
		return 0, true
	}
	if len(s.terms) == 1 {
		if t, ok := s.terms[""]; ok {
			f, _ := t.c.Float64()
			return f, true
		}
	}
	return 0, false
}

// AsExact returns the exact rational value when s is a constant.
func (s Sym) AsExact() (*big.Rat, bool) {
	if len(s.terms) == 0 {
		return new(big.Rat), true
	}
	if len(s.terms) == 1 {
		if t, ok := s.terms[""]; ok {
			return new(big.Rat).Set(t.c), true
		}
	}
	return nil, false
}

// Symbols lists every symbol appearing in s, sorted.
func (s Sym) Symbols() []string {
	seen := map[string]bool{}
	for _, t := range s.terms {
		for k := range t.m {
			seen[k] = true
		}
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// AsAtom returns the name when s is exactly one named symbol with coefficient 1.
func (s Sym) AsAtom() (string, bool) {
	if len(s.terms) != 1 {
		return "", false
	}
	var t term
	for _, v := range s.terms {
		t = v
	}
	if !ratIsOne(t.c) || len(t.m) != 1 {
		return "", false
	}
	for k, e := range t.m {
		if e != 1 {
			return "", false
		}
		return k, true
	}
	return "", false
}

// Subst replaces a subset of symbols with numbers, leaving the rest symbolic.
func (s Sym) Subst(values map[string]float64) Sym {
	acc := Zero()
	for _, t := range s.terms {
		part := ConRat(t.c)
		for k, e := range t.m {
			var base Sym
			if v, ok := values[k]; ok {
				base = Con(v)
			} else {
				base = V(k)
			}
			p, err := base.Pow(e)
			if err != nil {
				continue
			}
			part = part.Mul(p)
		}
		acc = acc.Add(part)
	}
	return acc
}

// ToNumber fully evaluates; ok is false when some symbol has no value.
func (s Sym) ToNumber(values map[string]float64) (float64, bool) {
	return s.Subst(values).AsConst()
}

// Equals is structural equality of the canonical form, with no substitution.
func (s Sym) Equals(other Sym) bool { return s.Sub(other).IsZero() }

// EqualsUnder is equality under a partial environment.
//
// This is what shape compatibility means: design symbols are substituted,
// runtime symbols stay indeterminate, and the difference has to be the zero
// polynomial. A mismatch is therefore a real polynomial difference rather than
// two numbers that happened not to match.
func (s Sym) EqualsUnder(other Sym, values map[string]float64) bool {
	return s.Sub(other).Subst(values).IsZero()
}

func (s Sym) String() string {
	if len(s.terms) == 0 {
		return "0"
	}
	list := make([]term, 0, len(s.terms))
	for _, t := range s.terms {
		list = append(list, t)
	}
	sort.Slice(list, func(i, j int) bool {
		di, dj := monoDegree(list[i].m), monoDegree(list[j].m)
		if di != dj {
			return di > dj
		}
		return monoKey(list[i].m) < monoKey(list[j].m)
	})
	var b strings.Builder
	for i, t := range list {
		negative := t.c.Sign() < 0
		abs := new(big.Rat).Abs(t.c)
		switch {
		case i == 0 && negative:
			b.WriteString("-")
		case i > 0 && negative:
			b.WriteString(" - ")
		case i > 0:
			b.WriteString(" + ")
		}
		key := monoKey(t.m)
		switch {
		case key == "":
			b.WriteString(ratString(abs))
		case ratIsOne(abs):
			b.WriteString(key)
		default:
			b.WriteString(ratString(abs) + "*" + key)
		}
	}
	return b.String()
}
