package rules

import (
	"fmt"
	"sort"
	"strings"
)

// The element-type check.
//
// This is the one class of wiring error shape inference cannot see. An integer
// tensor arriving at a matmul has a shape that agrees perfectly with what the
// matmul declared — `B T D` is `B T D` whether it holds token ids or
// activations — so the shape checker passes it and the model fails to run.
//
// `PortSpec.Dtype` was added in the seventeenth pass to make this check
// possible and the check was never written; until now exactly one port in the
// catalog declared anything, which is what a declaration with no consumer
// decays to.

// class is what a port carries, coarsely: two things that cannot be wired
// together, rather than five that can.
//
// `int64` and `bool` are both integral — indices and masks — and `fp32`,
// `bf16` and `fp8` are all real. A design that mixes widths is a design that
// needs a cast, which is a decision; a design that feeds indices to a matmul is
// broken.
func class(dtype string) string {
	switch dtype {
	case "int", "bool":
		return "integral"
	case "real", "float", "half", "fp8":
		return "real"
	default:
		return ""
	}
}

// effective resolves what an output port actually carries.
//
// A port that declares something concrete has said it. A port that inherits
// takes its class from what its block was given, which means walking back up
// the graph — and the walk is deliberately timid: if a block's inputs disagree,
// or any of them is unknown, the answer is unknown and nothing is reported.
// A false finding on a design nobody anticipated would be worse than a missed
// one, because the first thing it teaches is to read past the rule.
type dtypeWalk struct {
	ctx  *Ctx
	seen map[string]string
	busy map[string]bool
}

func (w *dtypeWalk) of(pin string) string {
	if got, ok := w.seen[pin]; ok {
		return got
	}
	// A cycle cannot be resolved, and a design can contain one.
	if w.busy[pin] {
		return ""
	}
	w.busy[pin] = true
	defer delete(w.busy, pin)

	path, port, ok := splitPin(pin)
	if !ok {
		return ""
	}
	ports, ok := w.ctx.Infer.Ports[path]
	if !ok {
		return ""
	}
	spec, ok := ports.Out[port]
	if !ok {
		return ""
	}

	if c := class(spec.Dtype); c != "" {
		w.seen[pin] = c
		return c
	}

	// Inherited: whatever came in, when everything that came in agrees.
	answer := ""
	names := make([]string, 0, len(ports.In))
	for name := range ports.In {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		producer, wired := w.ctx.Infer.ProducerOf[path+":"+name]
		if !wired {
			continue
		}
		got := w.of(producer)
		if got == "" {
			// One unknown input makes the whole block unknown: a block that
			// mixes a known real with an unknown is not evidence of anything.
			answer = ""
			break
		}
		if answer == "" {
			answer = got
			continue
		}
		if answer != got {
			// A block legitimately taking both — a gather taking indices and a
			// table — cannot say which one its output resembles.
			answer = ""
			break
		}
	}
	w.seen[pin] = answer
	return answer
}

func splitPin(pin string) (path, port string, ok bool) {
	at := strings.LastIndex(pin, ":")
	if at <= 0 || at == len(pin)-1 {
		return "", "", false
	}
	return pin[:at], pin[at+1:], true
}

var dtypeMismatch = Rule{
	ID:    "dtype",
	Title: "Element types",
	Description: "A port that says what it carries must be given it. Indices and activations have the " +
		"same shape, so this is the one wiring error shape inference cannot see.",
	Run: func(ctx *Ctx) []Finding {
		w := &dtypeWalk{ctx: ctx, seen: map[string]string{}, busy: map[string]bool{}}

		paths := make([]string, 0, len(ctx.Infer.Ports))
		for path := range ctx.Infer.Ports {
			paths = append(paths, path)
		}
		sort.Strings(paths)

		var out []Finding
		for _, path := range paths {
			ports := ctx.Infer.Ports[path]
			names := make([]string, 0, len(ports.In))
			for name := range ports.In {
				names = append(names, name)
			}
			sort.Strings(names)

			for _, name := range names {
				want := class(ports.In[name].Dtype)
				if want == "" {
					continue
				}
				producer, wired := ctx.Infer.ProducerOf[path+":"+name]
				if !wired {
					continue
				}
				got := w.of(producer)
				if got == "" || got == want {
					continue
				}
				out = append(out, Finding{
					Rule:     "dtype",
					Severity: "error",
					Path:     path,
					Port:     name,
					Message: fmt.Sprintf("Port %q takes %s values and is wired to %s, which carries %s.",
						name, wantWord(want), producer, wantWord(got)),
					Hint: "Indices and activations have the same shape, so nothing else catches this. " +
						"Check the wire, or the block that produces it.",
				})
			}
		}
		return out
	},
}

func wantWord(c string) string {
	if c == "integral" {
		return "integer"
	}
	return "floating-point"
}
