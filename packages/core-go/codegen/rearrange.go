package codegen

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/shapes"
)

// orderedKeys lists the keys of a value the document wrote as an object.
//
// Sorted, because the value arrives as a Go map and the author's order is gone
// by then. Every boundary node in the catalog and in every preset declares a
// single port, so sorted and declared are the same list; a block that declared
// two would need its order carried through the IR, since the order is the
// signature of the class the emitter builds from it.
func orderedKeys(v any) []string {
	m, ok := v.(map[string]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// dedupKey is the identity two blocks have to share to share a class.
//
// Marshalled rather than compared field by field: what makes two blocks the
// same is their whole resolved parameter set and their whole subgraph, and
// spelling that comparison out by hand would drift from what it is comparing.
func dedupKey(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%#v", v)
	}
	return string(b)
}

// paramSummary is the one-line docstring a generated class carries.
func paramSummary(r *catalog.Resolved) string {
	var parts []string
	for _, key := range r.Keys {
		switch v := r.P[key].(type) {
		case float64:
			parts = append(parts, key+"="+pyNum(v))
		case bool:
			if v {
				parts = append(parts, key+"=true")
			} else {
				parts = append(parts, key+"=false")
			}
		case string:
			parts = append(parts, key+"="+v)
		}
	}
	return strings.Join(parts, ", ")
}

var digitsOnly = regexp.MustCompile(`^[0-9]+$`)

// emitRearrange is the Python for a rearrange, as statements acting on src.
func emitRearrange(r *catalog.Resolved, c *ctx, src, dst, path string) []string {
	evalCtx := infer.EvalCtxFor(r, c.symbols)
	fromSrc, toSrc := r.Str("from"), r.Str("to")
	from, errFrom := shapes.ParsePattern(fromSrc)
	to, errTo := shapes.ParsePattern(toSrc)
	if errFrom != nil || errTo != nil {
		c.warn("%s: could not parse the rearrange patterns.", path)
		return []string{fmt.Sprintf("%s = %s  # TODO: unsupported rearrange", dst, src)}
	}

	if hasEllipsis(from) || hasEllipsis(to) {
		c.warn("%s: cannot generate code for a rearrange that uses \"...\".", path)
		return []string{fmt.Sprintf("%s = %s  # TODO: unsupported rearrange %s -> %s", dst, src, fromSrc, toSrc)}
	}

	// A pattern atom is an expression, not a bare name: composite expansion
	// substitutes parameters in, so the atom may read `((H))` or `2*D`.
	symOf := func(expr string) (shapes.Sym, bool) {
		sym, err := shapes.EvalExpr(expr, evalCtx)
		if err != nil {
			return shapes.Zero(), false
		}
		return sym, true
	}
	// runtimeName is the runtime symbol this atom *is*, or "" when it is a
	// fixed size.
	runtimeName := func(expr string) string {
		sym, ok := symOf(expr)
		if !ok {
			return ""
		}
		atom, isAtom := sym.AsAtom()
		if isAtom && c.symbols.Runtime[atom] {
			return atom
		}
		return ""
	}
	sizeOf := func(expr string) (float64, bool) {
		sym, ok := symOf(expr)
		if !ok {
			return 0, false
		}
		return sym.ToNumber(evalCtx.Values)
	}
	// canonical is the form that makes `((H))` on one side match `(H)` on the
	// other.
	canonical := func(expr string) string {
		if sym, ok := symOf(expr); ok {
			return sym.String()
		}
		return strings.TrimSpace(expr)
	}

	var lines []string

	// Name each incoming dimension; runtime dimensions are read from the tensor.
	groupVars := make([]string, len(from.Atoms))
	for i, a := range from.Atoms {
		groupVars[i] = fmt.Sprintf("_d%d", i)
		if len(a.Parts) == 1 {
			if rt := runtimeName(a.Parts[0]); rt != "" {
				groupVars[i] = pyName(rt)
			}
		}
	}
	lines = append(lines, strings.Join(groupVars, ", ")+" = "+src+".shape")

	// Flatten both sides to atomic axes keyed by their expression.
	type axis struct{ key, code string }
	var fromAxes []axis
	for i, a := range from.Atoms {
		for _, part := range a.Parts {
			key := strings.TrimSpace(part)
			if runtimeName(key) != "" {
				if len(a.Parts) > 1 {
					c.warn("%s: a runtime dimension inside a group is not supported.", path)
				}
				fromAxes = append(fromAxes, axis{key: canonical(key), code: groupVars[i]})
				continue
			}
			n, ok := sizeOf(key)
			if !ok {
				c.warn("%s: could not evaluate %q while generating a rearrange.", path, key)
				return []string{fmt.Sprintf("%s = %s  # TODO: unsupported rearrange", dst, src)}
			}
			fromAxes = append(fromAxes, axis{key: canonical(key), code: pyNum(n)})
		}
	}

	expr := src
	if grouped(from) {
		codes := make([]string, len(fromAxes))
		for i, a := range fromAxes {
			codes[i] = a.code
		}
		expr += ".view(" + strings.Join(codes, ", ") + ")"
	}

	// Permutation, matched by axis expression.
	var toFlat []string
	for _, a := range to.Atoms {
		for _, part := range a.Parts {
			toFlat = append(toFlat, canonical(strings.TrimSpace(part)))
		}
	}
	used := make([]bool, len(fromAxes))
	var perm []int
	permutable := len(toFlat) == len(fromAxes)
	if permutable {
		for _, key := range toFlat {
			idx := -1
			for i, a := range fromAxes {
				if a.key == key && !used[i] {
					idx = i
					break
				}
			}
			if idx < 0 {
				permutable = false
				break
			}
			used[idx] = true
			perm = append(perm, idx)
		}
	}
	if !permutable {
		c.warn("%s: the dimensions of %q and %q do not correspond, so the generated reshape may be wrong.",
			path, fromSrc, toSrc)
	} else if !isIdentity(perm) {
		parts := make([]string, len(perm))
		for i, v := range perm {
			parts[i] = fmt.Sprint(v)
		}
		expr += ".permute(" + strings.Join(parts, ", ") + ")"
	}

	// Regroup into the target shape.
	//
	// Only a grouped target dimension needs a reshape: when every target
	// dimension is a single axis, the view and permute above already produced
	// that shape.
	if grouped(to) {
		sizes := make([]string, len(to.Atoms))
		for i, a := range to.Atoms {
			parts := make([]string, len(a.Parts))
			for j, part := range a.Parts {
				key := strings.TrimSpace(part)
				if rt := runtimeName(key); rt != "" {
					parts[j] = pyName(rt)
					continue
				}
				if n, ok := sizeOf(key); ok {
					parts[j] = pyNum(n)
				} else {
					parts[j] = "-1"
				}
			}
			if len(parts) == 1 {
				sizes[i] = parts[0]
				continue
			}
			// Fold the constant factors so the emitted code stays readable.
			folded := 1.0
			anyNum := false
			var rest []string
			for _, p := range parts {
				if digitsOnly.MatchString(p) {
					anyNum = true
					folded *= mustFloat(p)
					continue
				}
				rest = append(rest, p)
			}
			if anyNum {
				rest = append(rest, pyNum(folded))
			}
			sizes[i] = strings.Join(rest, " * ")
		}
		expr += ".reshape(" + strings.Join(sizes, ", ") + ")"
	}

	return append(lines, dst+" = "+expr)
}

func hasEllipsis(p shapes.Pattern) bool {
	for _, a := range p.Atoms {
		if a.Kind == shapes.AtomEllipsis {
			return true
		}
	}
	return false
}

// grouped reports whether any dimension of a pattern is a product of factors.
func grouped(p shapes.Pattern) bool {
	for _, a := range p.Atoms {
		if a.Kind == shapes.AtomDims && len(a.Parts) > 1 {
			return true
		}
	}
	return false
}

func isIdentity(perm []int) bool {
	for i, v := range perm {
		if v != i {
			return false
		}
	}
	return true
}

func mustFloat(s string) float64 {
	var v float64
	fmt.Sscanf(s, "%g", &v)
	return v
}
