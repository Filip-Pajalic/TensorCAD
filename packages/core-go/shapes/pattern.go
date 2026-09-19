// Shape patterns, written in einops-flavoured notation.
//
//	"B T D"            three dims
//	"B T (H dh)"       last dim is the product H*dh
//	"... D"            leading batch dims are free, last dim is D
//	"B H T dh"         explicit four dims
//
// Atoms are expressions over parameters and global symbols, so "... 2*D" and
// "B T (H dh)" both work. At most one "..." may appear in a pattern.
//
// A grouped dimension keeps its factors rather than collapsing to a product,
// because code generation needs them to emit the matching view and permute.
package shapes

import (
	"fmt"
	"strings"
	"sync"
)

// AtomKind distinguishes an ellipsis from a dimension.
type AtomKind int

const (
	// AtomEllipsis is "...": however many leading dimensions there are.
	AtomEllipsis AtomKind = iota
	// AtomDims is one dimension. More than one part means their product.
	AtomDims
)

// PatternAtom is one position in a shape pattern.
type PatternAtom struct {
	Kind AtomKind
	// Parts are the factors of this dimension, kept separate rather than
	// multiplied out: codegen needs them to emit the matching view.
	Parts []string
}

// Pattern is a parsed shape pattern.
type Pattern struct {
	Src   string
	Atoms []PatternAtom
}

var patternCache sync.Map // string -> Pattern

// ParsePattern parses a shape pattern, memoising the result.
func ParsePattern(src string) (Pattern, error) {
	if hit, ok := patternCache.Load(src); ok {
		return hit.(Pattern), nil
	}
	p, err := parsePattern(src)
	if err != nil {
		return Pattern{}, err
	}
	patternCache.Store(src, p)
	return p, nil
}

func isSpace(c byte) bool { return c == ' ' || c == '\t' || c == '\n' || c == '\r' }

func parsePattern(src string) (Pattern, error) {
	var atoms []PatternAtom
	s := strings.TrimSpace(src)
	i := 0
	ellipses := 0

	for i < len(s) {
		if isSpace(s[i]) {
			i++
			continue
		}
		if strings.HasPrefix(s[i:], "...") {
			atoms = append(atoms, PatternAtom{Kind: AtomEllipsis})
			ellipses++
			i += 3
			continue
		}
		if s[i] == '(' {
			depth, j := 0, i
			for ; j < len(s); j++ {
				if s[j] == '(' {
					depth++
				} else if s[j] == ')' {
					depth--
					if depth == 0 {
						break
					}
				}
			}
			if depth != 0 {
				return Pattern{}, fmt.Errorf("unbalanced parentheses in pattern %q", src)
			}
			// Inside a group, whitespace separates the factors of one dimension.
			parts := strings.Fields(strings.TrimSpace(s[i+1 : j]))
			if len(parts) == 0 {
				return Pattern{}, fmt.Errorf("empty group in pattern %q", src)
			}
			atoms = append(atoms, PatternAtom{Kind: AtomDims, Parts: parts})
			i = j + 1
			continue
		}
		// A bare atom runs until whitespace.
		j := i
		for j < len(s) && !isSpace(s[j]) {
			j++
		}
		atoms = append(atoms, PatternAtom{Kind: AtomDims, Parts: []string{s[i:j]}})
		i = j
	}

	if ellipses > 1 {
		return Pattern{}, fmt.Errorf("pattern %q has more than one \"...\"", src)
	}
	return Pattern{Src: src, Atoms: atoms}, nil
}

// Shape is a concrete shape: one polynomial per dimension.
type Shape []Sym

// ShapeToString prints a shape the way the canvas labels an edge.
func ShapeToString(shape Shape) string {
	if len(shape) == 0 {
		return "scalar"
	}
	parts := make([]string, len(shape))
	for i, d := range shape {
		parts[i] = d.String()
	}
	return strings.Join(parts, " ")
}

// AtomValue is the value of one dimension: the product of its factors.
func AtomValue(atom PatternAtom, ctx EvalCtx) (Sym, error) {
	acc := Con(1)
	for _, part := range atom.Parts {
		v, err := EvalExpr(part, ctx)
		if err != nil {
			return Zero(), err
		}
		acc = acc.Mul(v)
	}
	return acc, nil
}

// AtomToString prints an atom the way the pattern was written.
func AtomToString(atom PatternAtom) string {
	if atom.Kind == AtomEllipsis {
		return "..."
	}
	if len(atom.Parts) == 1 {
		return atom.Parts[0]
	}
	return "(" + strings.Join(atom.Parts, " ") + ")"
}

// Instantiate turns a pattern into a concrete shape.
//
// batch supplies the dimensions bound to "..." and must be provided when the
// pattern contains an ellipsis.
func Instantiate(pattern Pattern, ctx EvalCtx, batch Shape, hasBatch bool) (Shape, []string) {
	var errs []string
	dims := Shape{}
	for _, a := range pattern.Atoms {
		if a.Kind == AtomEllipsis {
			if !hasBatch {
				return nil, []string{
					fmt.Sprintf("Pattern %q needs batch dims for \"...\" but none were bound", pattern.Src),
				}
			}
			dims = append(dims, batch...)
			continue
		}
		v, err := AtomValue(a, ctx)
		if err != nil {
			return nil, []string{fmt.Sprintf("In pattern %q: %s", pattern.Src, err)}
		}
		dims = append(dims, v)
	}
	return dims, errs
}

// MatchResult is the outcome of checking a shape against a pattern.
type MatchResult struct {
	OK bool
	// Batch is what "..." bound to, empty when the pattern has none.
	Batch  Shape
	Errors []string
}

// MatchPattern checks an actual shape against a pattern, binding "...".
//
// values holds the numeric value of every concrete design symbol; runtime
// symbols are absent and are therefore compared symbolically. That is what
// makes a mismatch a real polynomial difference rather than two numbers that
// happened not to match.
func MatchPattern(actual Shape, pattern Pattern, ctx EvalCtx, values map[string]float64) MatchResult {
	var errs []string

	idx := -1
	for i, a := range pattern.Atoms {
		if a.Kind == AtomEllipsis {
			idx = i
			break
		}
	}

	compare := func(dim Sym, atom PatternAtom, position int) {
		if atom.Kind == AtomEllipsis {
			return
		}
		want, err := AtomValue(atom, ctx)
		if err != nil {
			errs = append(errs, fmt.Sprintf("In pattern %q: %s", pattern.Src, err))
			return
		}
		if !dim.EqualsUnder(want, values) {
			errs = append(errs, fmt.Sprintf(
				"dim %d is %s but the port expects %s = %s",
				position, dim, AtomToString(atom), want,
			))
		}
	}

	if idx == -1 {
		if len(actual) != len(pattern.Atoms) {
			return MatchResult{Errors: []string{fmt.Sprintf(
				"rank %d does not match pattern %q (rank %d)",
				len(actual), pattern.Src, len(pattern.Atoms),
			)}}
		}
		for k, d := range actual {
			compare(d, pattern.Atoms[k], k)
		}
		return MatchResult{OK: len(errs) == 0, Errors: errs}
	}

	prefix := pattern.Atoms[:idx]
	suffix := pattern.Atoms[idx+1:]
	if len(actual) < len(prefix)+len(suffix) {
		return MatchResult{Errors: []string{fmt.Sprintf(
			"rank %d is too small for pattern %q (needs at least %d)",
			len(actual), pattern.Src, len(prefix)+len(suffix),
		)}}
	}
	for k, a := range prefix {
		compare(actual[k], a, k)
	}
	for k, a := range suffix {
		pos := len(actual) - len(suffix) + k
		compare(actual[pos], a, pos)
	}
	batch := append(Shape{}, actual[len(prefix):len(actual)-len(suffix)]...)
	return MatchResult{OK: len(errs) == 0, Batch: batch, Errors: errs}
}
