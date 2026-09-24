package catalog

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/tensorcad/core/attnexpr"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/shapes"
)

// ResolveNodeParams turns the raw values a document wrote into numbers,
// booleans and symbolic forms, driven by the block's declared parameter specs.
//
// Type direction matters: without the spec there would be no way to tell the
// enum "silu" from the expression "D".
func ResolveNodeParams(def *BlockDef, raw map[string]any, symbols *ir.SymbolTable) *Resolved {
	ctx := ir.SymbolCtx(symbols)
	out := &Resolved{
		Type:    def.Type,
		P:       map[string]any{},
		S:       map[string]shapes.Sym{},
		Raw:     raw,
		RawFull: map[string]any{},
	}
	if out.Raw == nil {
		out.Raw = map[string]any{}
	}

	// Declared parameters first, in the order the block declares them, then
	// anything else the document wrote. The declared order is what a reader
	// sees; the extras are all errors, so they only need to be in *some* order,
	// and sorted is the one that does not change between runs.
	extra := map[string]bool{}
	for k := range raw {
		if !def.Params.Has(k) {
			extra[k] = true
		}
	}
	keys := make([]string, 0, len(def.Params)+len(extra))
	for _, e := range def.Params {
		keys = append(keys, e.Name)
	}
	keys = append(keys, sortedKeys(extra)...)
	out.Keys = keys

	for _, key := range keys {
		spec, hasSpec := def.Params.Get(key)
		given, hasGiven := raw[key]

		if !hasSpec {
			out.Errors = append(out.Errors,
				fmt.Sprintf("Unknown parameter %q on block %q", key, def.Type))
			out.P[key] = given
			continue
		}

		var value any
		var hasValue bool
		if hasGiven {
			value, hasValue = given, true
		} else if spec.HasDefault {
			value, hasValue = spec.Default, true
		}
		if hasValue {
			out.RawFull[key] = value
		}

		if !hasValue {
			switch spec.Type {
			case ParamBool:
				out.P[key] = false
			case ParamObj:
				out.P[key] = nil
			default:
				out.Errors = append(out.Errors,
					fmt.Sprintf("Missing required parameter %q on block %q", key, def.Type))
			}
			continue
		}

		switch spec.Type {
		case ParamInt, ParamNum:
			resolveNumeric(out, key, spec, value, ctx)
		case ParamBool:
			// A null is kept rather than coerced: some booleans are tri-state,
			// such as an output-projection bias that defaults to whatever the
			// other projections use. Consumers test for true explicitly.
			switch v := value.(type) {
			case bool:
				out.P[key] = v
			case nil:
				out.P[key] = nil
			default:
				out.Errors = append(out.Errors, fmt.Sprintf("Parameter %q should be a boolean", key))
			}
		case ParamEnum:
			switch v := value.(type) {
			case string:
				if contains(spec.Values, v) {
					out.P[key] = v
				} else {
					out.Errors = append(out.Errors, fmt.Sprintf(
						"Parameter %q should be one of %s, got %v", key, join(spec.Values), v))
				}
			case nil:
				out.P[key] = nil
			default:
				out.Errors = append(out.Errors, fmt.Sprintf(
					"Parameter %q should be one of %s, got %v", key, join(spec.Values), value))
			}
		case ParamStr, ParamPattern:
			switch v := value.(type) {
			case string:
				out.P[key] = v
			case nil:
				out.P[key] = nil
			default:
				out.Errors = append(out.Errors, fmt.Sprintf("Parameter %q should be a string", key))
			}
		case ParamObj:
			out.P[key] = value
		case ParamMask, ParamScore:
			resolveExpression(out, key, spec.Type, value, symbols)
		}
	}
	return out
}

// resolveExpression compiles an attention expression against the design's
// symbols and keeps it in the form it was understood in: every symbol replaced
// by its value, every constant folded, and only the parentheses it needs. That
// is what the analysis evaluates and the generated code prints, so it is what
// a reader is shown as the evaluated value.
//
// Nothing is stored for an empty expression, or for one that did not compile:
// a composite passes on what it resolved, and an error reported here would
// otherwise be reported again by every block it expands into.
//
// B and T are left out. They carry a default in the symbol table, but a mask
// that read T would be evaluated at that default rather than at the length it
// runs at, and the analysis measures it at the operating point's.
func resolveExpression(out *Resolved, key string, kind ParamKind, value any, symbols *ir.SymbolTable) {
	text, ok := value.(string)
	if value != nil && !ok {
		out.Errors = append(out.Errors, fmt.Sprintf("Parameter %q should be an expression", key))
		return
	}
	if strings.TrimSpace(text) == "" {
		return
	}
	want := attnexpr.Mask
	if kind == ParamScore {
		want = attnexpr.Score
	}
	values := map[string]float64{}
	if symbols != nil {
		for name, v := range symbols.Values {
			if !symbols.Runtime[name] {
				values[name] = v
			}
		}
	}
	n, err := attnexpr.Compile(text, want, values)
	if err != nil {
		out.Errors = append(out.Errors, fmt.Sprintf("Parameter %q: %s", key, err))
		return
	}
	out.P[key] = attnexpr.String(n)
}

func resolveNumeric(out *Resolved, key string, spec ParamSpec, value any, ctx shapes.EvalCtx) {
	switch v := value.(type) {
	case float64:
		out.P[key] = v
		out.S[key] = shapes.Con(v)
	case int:
		out.P[key] = float64(v)
		out.S[key] = shapes.Con(float64(v))
	case string:
		sym, err := shapes.EvalExpr(v, ctx)
		if err != nil {
			out.Errors = append(out.Errors, fmt.Sprintf("Parameter %q: %s", key, err))
			return
		}
		// Evaluate rather than ToNumber: a symbol that failed carries NaN, and a
		// block written over it has to report that rather than fault. The
		// editor is where this lands, and a faulted engine cannot be restarted
		// without reloading the window.
		n, ok, err := sym.Evaluate(ctx.Values)
		if err != nil {
			out.Errors = append(out.Errors, fmt.Sprintf("Parameter %q: %s", key, err))
			return
		}
		if !ok {
			out.Errors = append(out.Errors, fmt.Sprintf(
				"Parameter %q does not evaluate to a number: %s", key, sym))
			return
		}
		out.P[key] = n
		out.S[key] = sym
	case nil:
		out.P[key] = nil
		return
	default:
		out.Errors = append(out.Errors,
			fmt.Sprintf("Parameter %q should be a number or expression", key))
		return
	}

	n, isNum := out.P[key].(float64)
	if !isNum {
		return
	}
	if spec.Type == ParamInt && n != math.Trunc(n) {
		msg := fmt.Sprintf("Parameter %q must be an integer but evaluates to %s", key, num(n))
		if s, ok := value.(string); ok {
			msg += fmt.Sprintf(" (from %q)", s)
		}
		out.Errors = append(out.Errors, msg)
	}
	if spec.Min != nil && n < *spec.Min {
		out.Errors = append(out.Errors,
			fmt.Sprintf("Parameter %q must be >= %s, got %s", key, num(*spec.Min), num(n)))
	}
	if spec.Max != nil && n > *spec.Max {
		out.Errors = append(out.Errors,
			fmt.Sprintf("Parameter %q must be <= %s, got %s", key, num(*spec.Max), num(n)))
	}
}

// num formats a float the way JavaScript prints one, because these strings end
// up in messages the two engines have to agree on.
func num(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e21 {
		return fmt.Sprintf("%d", int64(v))
	}
	return fmt.Sprintf("%v", v)
}

func contains(list []string, v string) bool {
	for _, s := range list {
		if s == v {
			return true
		}
	}
	return false
}

func join(list []string) string {
	out := ""
	for i, s := range list {
		if i > 0 {
			out += ", "
		}
		out += s
	}
	return out
}

func sortedKeys(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
