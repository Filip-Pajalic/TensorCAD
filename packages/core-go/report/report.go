// Package report shapes the engine's results for a client.
//
// The analysis works in Sym values and Go maps; an editor works in strings and
// JSON. This is where the one becomes the other, in one place rather than once
// per caller, so the desktop service and the WebAssembly build cannot answer
// the same question differently.
package report

import (
	"math"
	"strings"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/rules"
	"github.com/tensorcad/core/shapes"
)

// Shape is one tensor shape, in both the forms an editor shows.
//
// Symbolic is the honest one: `B T D` says the residual stream is D wide
// whatever D is. Numeric substitutes the design symbols and leaves the runtime
// ones alone, which is the quickest way to watch a symbol edit travel through a
// design. Both are computed here because only the engine holds the polynomial.
type Shape struct {
	Symbolic string `json:"symbolic"`
	Numeric  string `json:"numeric"`
}

// Port is one pin, as the canvas draws it.
type Port struct {
	Shape string `json:"shape"`
	Dtype string `json:"dtype"`
	// Anchor is "flow" or "side": which edge of the symbol a wire leaves by.
	Anchor   string `json:"anchor"`
	Optional bool   `json:"optional,omitempty"`
	ShowName bool   `json:"showName,omitempty"`
	Doc      string `json:"doc,omitempty"`
}

// Ports are a block's pins.
type Ports struct {
	In  map[string]Port `json:"in"`
	Out map[string]Port `json:"out"`
}

// Resolved is a node's parameters after evaluation, as the inspector reads
// them: the value, and the symbolic form where the value came from a symbol.
type Resolved struct {
	Type string         `json:"type"`
	P    map[string]any `json:"p"`
	// S is the symbolic form of each numeric parameter, so the canvas can
	// label a width "D" rather than 4096.
	S map[string]string `json:"s"`
}

// Inference is every shape in a design, keyed by "path:port".
type Inference struct {
	Outputs    map[string]Shape    `json:"outputs"`
	Inputs     map[string]Shape    `json:"inputs"`
	ProducerOf map[string]string   `json:"producerOf"`
	Ports      map[string]Ports    `json:"ports"`
	Resolved   map[string]Resolved `json:"resolved"`
	Issues     []infer.Issue       `json:"issues"`
}

// Derived is everything the editor needs for one document at one operating
// point.
//
// One call rather than two: the findings and the shapes come from the same walk
// of the same graph, and asking for them separately would walk it twice on
// every keystroke.
type Derived struct {
	Report *rules.Report `json:"report"`
	// Infer is shape inference with composites expanded, so the interiors of a
	// block can be opened and drawn.
	Infer Inference `json:"infer"`
}

// shapeOf renders a shape in both forms.
func shapeOf(s shapes.Shape, designValues map[string]float64) Shape {
	parts := make([]string, len(s))
	for i, dim := range s {
		// Evaluate rather than ToNumber: a design whose symbols failed carries
		// NaN, and the editor has to draw it rather than fall over on it.
		if n, ok, err := dim.Evaluate(designValues); err == nil && ok && isFinite(n) {
			parts[i] = number(n)
			continue
		}
		parts[i] = dim.String()
	}
	return Shape{Symbolic: shapes.ShapeToString(s), Numeric: strings.Join(parts, " ")}
}

func isFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func number(v float64) string { return analysis.JSNumber(v) }

func portsOf(p catalog.Ports) Ports {
	out := Ports{In: map[string]Port{}, Out: map[string]Port{}}
	for name, spec := range p.In {
		out.In[name] = Port{
			Shape: spec.Shape, Dtype: spec.Dtype, Anchor: spec.Anchor,
			Optional: spec.Optional, ShowName: spec.ShowName, Doc: spec.Doc,
		}
	}
	for name, spec := range p.Out {
		out.Out[name] = Port{
			Shape: spec.Shape, Dtype: spec.Dtype, Anchor: spec.Anchor,
			Optional: spec.Optional, ShowName: spec.ShowName, Doc: spec.Doc,
		}
	}
	return out
}

func resolvedOf(r *catalog.Resolved) Resolved {
	out := Resolved{Type: r.Type, P: map[string]any{}, S: map[string]string{}}
	for k, v := range r.P {
		out.P[k] = v
	}
	for k, sym := range r.S {
		out.S[k] = sym.String()
	}
	return out
}

// InferenceOf renders a shape-inference result for a client.
func InferenceOf(res *infer.Result, symbols *ir.SymbolTable) Inference {
	out := Inference{
		Outputs:    make(map[string]Shape, len(res.Outputs)),
		Inputs:     make(map[string]Shape, len(res.Inputs)),
		ProducerOf: make(map[string]string, len(res.ProducerOf)),
		Ports:      make(map[string]Ports, len(res.Ports)),
		Resolved:   make(map[string]Resolved, len(res.Resolved)),
		Issues:     res.Issues,
	}
	if out.Issues == nil {
		out.Issues = []infer.Issue{}
	}
	for key, shape := range res.Outputs {
		out.Outputs[key] = shapeOf(shape, symbols.DesignValues)
	}
	for key, shape := range res.Inputs {
		out.Inputs[key] = shapeOf(shape, symbols.DesignValues)
	}
	for key, producer := range res.ProducerOf {
		out.ProducerOf[key] = producer
	}
	for path, ports := range res.Ports {
		out.Ports[path] = portsOf(ports)
	}
	for path, r := range res.Resolved {
		out.Resolved[path] = resolvedOf(r)
	}
	return out
}

// Derive runs the design rules and shape inference over one document, and
// returns both.
func Derive(doc *ir.Doc, options analysis.Options) (*Derived, error) {
	report, err := rules.Validate(doc, options)
	if err != nil {
		return nil, err
	}
	expanded := infer.Shapes(doc, report.Analysis.Symbols, infer.Options{ExpandComposites: true})
	return &Derived{
		Report: report,
		Infer:  InferenceOf(expanded, report.Analysis.Symbols),
	}, nil
}

// Infer runs shape inference alone.
//
// The editor uses this to answer "would this wire type-check", which it asks
// for every handle the pointer passes over. Running the whole rule engine for
// that would be the slowest thing in the interaction.
func Infer(doc *ir.Doc, expandComposites bool) (Inference, error) {
	symbols := ir.ResolveSymbols(doc)
	res := infer.Shapes(doc, symbols, infer.Options{ExpandComposites: expandComposites})
	return InferenceOf(res, symbols), nil
}
