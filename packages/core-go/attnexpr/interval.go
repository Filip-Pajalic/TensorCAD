package attnexpr

import "math"

// Span is every value a number can take over a box of points: at least Lo
// and at most Hi.
type Span struct{ Lo, Hi float64 }

func point(v float64) Span { return Span{v, v} }

var anything = Span{math.Inf(-1), math.Inf(1)}

// Box is a rectangle of the score matrix: a range of queries against a range
// of keys, for a range of heads and rows.
type Box struct {
	Q, KV, H, B Span
	Heads       float64
	Score       Span
	// Table answers a tensor read over a range of indices: the least and most
	// it can hold there. Without it a read can be anything.
	Table func(name string, at []Span) Span
}

// Possible says whether a mask can be true anywhere in a box, and whether it
// can be false anywhere in it.
//
// It is how the engine decides whether a kernel computes a block of the
// score matrix without evaluating every score in it. The answer is never
// "no" where some point says yes, but it can be "yes" where every point says
// no: a conjunction whose parts are each true somewhere in a box need not be
// true together anywhere. For a mask that keeps documents apart, read against
// a table whose values never fall along a row, it is exact.
func Possible(n Node, box Box) (canTrue, canFalse bool) {
	switch x := n.(type) {
	case Bool:
		return x.V, !x.V
	case Unary:
		if x.Op == "not" {
			t, f := Possible(x.X, box)
			return f, t
		}
	case Binary:
		switch x.Op {
		case "and":
			at, af := Possible(x.X, box)
			bt, bf := Possible(x.Y, box)
			return at && bt, af || bf
		case "or":
			at, af := Possible(x.X, box)
			bt, bf := Possible(x.Y, box)
			return at || bt, af && bf
		}
		a, b := Bounds(x.X, box), Bounds(x.Y, box)
		switch x.Op {
		case "<":
			return a.Lo < b.Hi, a.Hi >= b.Lo
		case "<=":
			return a.Lo <= b.Hi, a.Hi > b.Lo
		case ">":
			return a.Hi > b.Lo, a.Lo <= b.Hi
		case ">=":
			return a.Hi >= b.Lo, a.Lo < b.Hi
		case "==":
			return a.Lo <= b.Hi && b.Lo <= a.Hi, !(a.Lo == a.Hi && b.Lo == b.Hi && a.Lo == b.Lo)
		case "!=":
			return !(a.Lo == a.Hi && b.Lo == b.Hi && a.Lo == b.Lo), a.Lo <= b.Hi && b.Lo <= a.Hi
		}
	}
	// A number where a truth value was wanted is true wherever it is not zero.
	s := Bounds(n, box)
	return s.Lo != 0 || s.Hi != 0, s.Lo <= 0 && s.Hi >= 0
}

// Bounds is every value a numeric expression takes over a box, or a range
// that contains them all.
func Bounds(n Node, box Box) Span {
	switch x := n.(type) {
	case Num:
		return point(x.V)
	case Bool:
		return point(b2f(x.V))
	case Var:
		switch x.Name {
		case "q":
			return box.Q
		case "kv":
			return box.KV
		case "h":
			return box.H
		case "b":
			return box.B
		case "heads":
			return point(box.Heads)
		case "score":
			return box.Score
		}
		return anything
	case Unary:
		if x.Op == "not" {
			t, f := Possible(x.X, box)
			return truthSpan(!f, !t)
		}
		s := Bounds(x.X, box)
		return Span{-s.Hi, -s.Lo}
	case Binary:
		switch x.Op {
		case "and", "or", "<", "<=", ">", ">=", "==", "!=":
			t, f := Possible(x, box)
			return truthSpan(t && !f, f && !t)
		}
		return spanArith(x.Op, Bounds(x.X, box), Bounds(x.Y, box))
	case Call:
		return spanCall(x, box)
	}
	return anything
}

// truthSpan is a truth value as a number: 1, 0, or either.
func truthSpan(alwaysTrue, alwaysFalse bool) Span {
	switch {
	case alwaysTrue:
		return point(1)
	case alwaysFalse:
		return point(0)
	}
	return Span{0, 1}
}

func spanArith(op string, a, b Span) Span {
	switch op {
	case "+":
		return Span{a.Lo + b.Lo, a.Hi + b.Hi}
	case "-":
		return Span{a.Lo - b.Hi, a.Hi - b.Lo}
	case "*":
		return corners(a, b, func(x, y float64) float64 { return x * y })
	case "/":
		if b.Lo <= 0 && b.Hi >= 0 {
			return anything
		}
		return corners(a, b, func(x, y float64) float64 { return x / y })
	case "%":
		// Python's modulo by a positive constant lands in [0, d).
		if b.Lo == b.Hi && b.Lo > 0 {
			return Span{0, b.Lo}
		}
		return anything
	case "**":
		// Monotone in each argument while the base is positive, so the
		// extremes are at the corners.
		if a.Lo > 0 {
			return corners(a, b, math.Pow)
		}
		return anything
	}
	return anything
}

// corners is the range of f over a box of two ranges, for an f whose extremes
// are at its corners. A NaN at a corner, such as zero times infinity, makes
// it anything.
func corners(a, b Span, f func(x, y float64) float64) Span {
	out := Span{math.Inf(1), math.Inf(-1)}
	for _, x := range []float64{a.Lo, a.Hi} {
		for _, y := range []float64{b.Lo, b.Hi} {
			v := f(x, y)
			if math.IsNaN(v) {
				return anything
			}
			out.Lo, out.Hi = math.Min(out.Lo, v), math.Max(out.Hi, v)
		}
	}
	return out
}

func spanCall(x Call, box Box) Span {
	if x.Fn == "where" {
		t, f := Possible(x.Args[0], box)
		switch {
		case t && !f:
			return Bounds(x.Args[1], box)
		case f && !t:
			return Bounds(x.Args[2], box)
		}
		a, b := Bounds(x.Args[1], box), Bounds(x.Args[2], box)
		return Span{math.Min(a.Lo, b.Lo), math.Max(a.Hi, b.Hi)}
	}
	args := make([]Span, len(x.Args))
	for i, a := range x.Args {
		args[i] = Bounds(a, box)
	}
	if IsTable(x.Fn) {
		if box.Table == nil {
			return anything
		}
		return box.Table(x.Fn, args)
	}
	a := args[0]
	switch x.Fn {
	case "tanh", "exp", "floor":
		// Increasing everywhere.
		return Span{call(x.Fn, []float64{a.Lo}), call(x.Fn, []float64{a.Hi})}
	case "log", "sqrt":
		// Increasing where they are defined.
		if a.Lo < 0 {
			return anything
		}
		return Span{call(x.Fn, []float64{a.Lo}), call(x.Fn, []float64{a.Hi})}
	case "abs":
		switch {
		case a.Lo >= 0:
			return a
		case a.Hi <= 0:
			return Span{-a.Hi, -a.Lo}
		}
		return Span{0, math.Max(-a.Lo, a.Hi)}
	case "min":
		b := args[1]
		return Span{math.Min(a.Lo, b.Lo), math.Min(a.Hi, b.Hi)}
	case "max":
		b := args[1]
		return Span{math.Max(a.Lo, b.Lo), math.Max(a.Hi, b.Hi)}
	case "t5_bucket":
		// Not monotone in the distance, and always one of its buckets.
		return Span{0, math.Max(0, args[1].Hi-1)}
	}
	return anything
}
