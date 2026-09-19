// Package catalog is the block model: what a block declares about itself.
//
// The division that matters is between a primitive and a composite. A primitive
// carries formulas — parameter count, FLOPs, what it must keep alive for the
// backward pass — and a composite carries none, because it expands into
// primitives and the analysis sums what it finds. The test for which one a new
// block should be is whether it has arithmetic that cannot be expressed as an
// arrangement of existing blocks; almost nothing does.
package catalog

import (
	"fmt"

	"github.com/tensorcad/core/shapes"
)

// ParamKind is the declared type of a parameter, which is what lets the
// resolver tell the enum "silu" from the expression "D".
type ParamKind string

const (
	ParamInt     ParamKind = "int"
	ParamNum     ParamKind = "num"
	ParamBool    ParamKind = "bool"
	ParamEnum    ParamKind = "enum"
	ParamStr     ParamKind = "str"
	ParamPattern ParamKind = "pattern"
	ParamObj     ParamKind = "obj"
)

// ParamSpec declares one parameter of a block.
type ParamSpec struct {
	Type ParamKind
	// Default is used when the document does not set the parameter.
	Default    any
	HasDefault bool
	Min        *float64
	Max        *float64
	// Values are the permitted strings of an enum.
	Values []string
	Doc    string
}

// PortSpec is what a pin declares. See the reference in docs/reference/ports.md.
type PortSpec struct {
	Shape string
	// Dtype is what the tensor carries; "inherit" takes it from the producer.
	Dtype string
	// Optional marks a port that may legitimately dangle.
	Optional bool
	// Anchor is "flow" or "side": which edge of the symbol a wire leaves by.
	Anchor   string
	ShowName bool
	Doc      string
}

// ResolvedPort is a PortSpec with every default filled in.
type ResolvedPort = PortSpec

// Ports are a block's pins, by name.
type Ports struct {
	In  map[string]PortSpec
	Out map[string]PortSpec
}

// Port builds a port from a bare shape pattern, which is what most declare.
func Port(shape string) PortSpec {
	return PortSpec{Shape: shape, Dtype: "inherit", Anchor: "flow"}
}

// NormalisePort fills in a port's defaults.
func NormalisePort(p PortSpec) ResolvedPort {
	if p.Dtype == "" {
		p.Dtype = "inherit"
	}
	if p.Anchor == "" {
		p.Anchor = "flow"
	}
	return p
}

// NormalisePorts fills in the defaults of every port on a block.
func NormalisePorts(p Ports) Ports {
	out := Ports{In: map[string]PortSpec{}, Out: map[string]PortSpec{}}
	for k, v := range p.In {
		out.In[k] = NormalisePort(v)
	}
	for k, v := range p.Out {
		out.Out[k] = NormalisePort(v)
	}
	return out
}

// BlockFinding is something a block says about itself: the same shape as every
// other finding in the system, so there is one type rather than two.
type BlockFinding struct {
	ID       string `json:"id"`
	Severity string `json:"severity"`
	Message  string `json:"message"`
	Param    string `json:"param,omitempty"`
	Port     string `json:"port,omitempty"`
	Hint     string `json:"hint,omitempty"`
}

// Resolved is a node's parameters after evaluation.
type Resolved struct {
	Type string
	// P is the concrete value of each parameter.
	P map[string]any
	// S is the symbolic form of each numeric parameter, so a shape keeps
	// showing "D" rather than "4096".
	S map[string]shapes.Sym
	// Raw is what the document wrote, unevaluated.
	Raw map[string]any
	// RawFull is Raw with catalog defaults filled in. Composite expansion uses
	// this so an expression survives into the inner graph.
	RawFull map[string]any
	Errors  []string
}

// Num reads a resolved parameter as a number. Missing or non-numeric is 0.
func (r *Resolved) Num(key string) float64 {
	switch v := r.P[key].(type) {
	case float64:
		return v
	case int:
		return float64(v)
	}
	return 0
}

// Int reads a resolved parameter as an int.
func (r *Resolved) Int(key string) int { return int(r.Num(key)) }

// Str reads a resolved parameter as a string.
func (r *Resolved) Str(key string) string {
	if s, ok := r.P[key].(string); ok {
		return s
	}
	return ""
}

// Bool reads a resolved parameter as a boolean. A tri-state null is false here;
// callers that care about the difference read P directly.
func (r *Resolved) Bool(key string) bool {
	b, ok := r.P[key].(bool)
	return ok && b
}

// FlopsPerToken is what a primitive reports for one token.
type FlopsPerToken struct {
	// Fwd counts only matmuls, a multiply-accumulate as 2.
	Fwd float64
	// Elementwise is norms, activations and adds: memory-bound, and excluded
	// from the headline number because the 6N convention excludes it.
	Elementwise float64
	// FwdSeq is the sequence-dependent part, kept apart so the 2N and 6N rules
	// stay comparable.
	FwdSeq float64
	// FwdSeqUnmasked is the same term counted as a profiler counts it.
	FwdSeqUnmasked float64
}

// AnalysisCtx is the operating point a formula is evaluated under.
type AnalysisCtx struct {
	T     float64
	B     float64
	Bytes float64
	Flash bool
}

// StateBytes is per-token cache and per-sequence state.
type StateBytes struct {
	PerToken    float64
	PerSequence float64
}

// BlockDocs is what a block says for a reader.
type BlockDocs struct {
	Summary string
	Formula string
	Refs    []string
}

// BlockDef is a catalog entry.
//
// One type rather than two, with Kind discriminating: a composite leaves the
// formula fields nil and fills in Expand, and the analysis expands it instead
// of asking it anything.
type BlockDef struct {
	Kind     string // "primitive" | "composite" | "container"
	Type     string
	Category string
	Params   map[string]ParamSpec
	// PortsFn computes ports from resolved parameters; Ports is the fixed form.
	Ports   Ports
	PortsFn func(r *Resolved) Ports

	ParamCount func(r *Resolved) float64
	Flops      func(r *Resolved, ctx AnalysisCtx) FlopsPerToken
	Retains    func(r *Resolved) []string
	// ExtraActivationBytes is what a block keeps per token beyond the tensors on
	// its edges: a fused attention kernel's log-sum-exp statistics, or the
	// logits buffer.
	ExtraActivationBytes func(r *Resolved, ctx AnalysisCtx) float64
	StateBytes           func(r *Resolved, ctx AnalysisCtx) StateBytes
	Constraints          func(r *Resolved) []BlockFinding

	Docs BlockDocs

	// userGraph is the template a block defined by a document carries, instead
	// of the expansion function a built-in composite has.
	userGraph *UserBlockDef
}

// PortsOf is a block's ports, with every default filled in.
//
// Normalising here rather than at each call site is deliberate: a consumer that
// reached for the raw declaration would see a bare shape on most primitives and
// a full spec on the rest, and would get one of the two wrong.
func PortsOf(def *BlockDef, r *Resolved) Ports {
	if def.PortsFn != nil {
		return NormalisePorts(def.PortsFn(r))
	}
	return NormalisePorts(def.Ports)
}

// Catalog is every block type, by name.
type Catalog map[string]*BlockDef

// Get looks a block type up.
func (c Catalog) Get(t string) (*BlockDef, error) {
	def, ok := c[t]
	if !ok {
		return nil, fmt.Errorf("unknown block type %q", t)
	}
	return def, nil
}
