// Package ir is the design document: our own JSON IR.
//
// The editor's node and edge state is a view of this document, never the source
// of truth. Everything downstream — validation, analysis, codegen, MCP — reads
// this type, and it is the wire format between Go and the frontend.
package ir

import (
	"encoding/json"
	"fmt"
	"strings"
)

// DocVersion is the only document version this engine reads.
const DocVersion = 1

// RuntimeSymbols stay indeterminate through analysis: a mismatch involving
// them is a real polynomial difference rather than two numbers that happened
// not to match.
var RuntimeSymbols = []string{"B", "T"}

// ParamValue is a parameter as written in the document: a number, string,
// boolean, null, object or array. It stays untyped because the catalog, not the
// IR, decides what a given block's parameter means.
type ParamValue = any

// SymbolDef is either a literal number, an expression over earlier symbols, or
// a runtime dimension that stays symbolic.
//
// The document spells this four ways — a bare number, a bare string, or an
// object tagged "runtime" or "design" — so it decodes through a custom
// unmarshaller into one shape.
type SymbolDef struct {
	// Kind is "literal", "expr", "runtime" or "design".
	Kind string
	// Number is set for a literal, and is the default for a runtime symbol.
	Number float64
	// Expr is set when the value is an expression.
	Expr string
	Doc  string
	// HasNumber distinguishes an absent value from a zero one.
	HasNumber bool
}

// UnmarshalJSON accepts every spelling the document uses.
func (s *SymbolDef) UnmarshalJSON(b []byte) error {
	var n float64
	if err := json.Unmarshal(b, &n); err == nil {
		*s = SymbolDef{Kind: "literal", Number: n, HasNumber: true}
		return nil
	}
	var str string
	if err := json.Unmarshal(b, &str); err == nil {
		*s = SymbolDef{Kind: "expr", Expr: str}
		return nil
	}
	var obj struct {
		Kind    string          `json:"kind"`
		Default *float64        `json:"default"`
		Value   json.RawMessage `json:"value"`
		Doc     string          `json:"doc"`
	}
	if err := json.Unmarshal(b, &obj); err != nil {
		return fmt.Errorf("unsupported symbol definition: %s", string(b))
	}
	switch obj.Kind {
	case "runtime":
		out := SymbolDef{Kind: "runtime", Doc: obj.Doc}
		if obj.Default != nil {
			out.Number, out.HasNumber = *obj.Default, true
		}
		*s = out
		return nil
	case "design":
		out := SymbolDef{Kind: "design", Doc: obj.Doc}
		var v float64
		if err := json.Unmarshal(obj.Value, &v); err == nil {
			out.Number, out.HasNumber = v, true
		} else {
			var e string
			if err := json.Unmarshal(obj.Value, &e); err != nil {
				return fmt.Errorf("design symbol value must be a number or an expression")
			}
			out.Expr = e
		}
		*s = out
		return nil
	}
	return fmt.Errorf("unsupported symbol definition: %s", string(b))
}

// MarshalJSON writes back the spelling the document uses.
func (s SymbolDef) MarshalJSON() ([]byte, error) {
	switch s.Kind {
	case "literal":
		return json.Marshal(s.Number)
	case "expr":
		return json.Marshal(s.Expr)
	case "runtime":
		return json.Marshal(map[string]any{"kind": "runtime", "default": s.Number, "doc": s.Doc})
	case "design":
		var v any = s.Expr
		if s.HasNumber {
			v = s.Number
		}
		return json.Marshal(map[string]any{"kind": "design", "value": v, "doc": s.Doc})
	}
	return nil, fmt.Errorf("unsupported symbol kind %q", s.Kind)
}

// NodeDef is one block in a graph.
type NodeDef struct {
	ID     string                `json:"id"`
	Type   string                `json:"type"`
	Params map[string]ParamValue `json:"params,omitempty"`
	// Graph is the subgraph of a container node such as repeat.
	Graph *Graph `json:"graph,omitempty"`
	// Variants are named subgraph alternatives, for hybrid repeat patterns.
	Variants map[string]*Graph `json:"variants,omitempty"`
	// Label is free text shown on the canvas instead of the id.
	Label string `json:"label,omitempty"`
}

// Edge is "nodeId:portName" on both ends.
type Edge [2]string

// From is the producing endpoint.
func (e Edge) From() string { return e[0] }

// To is the consuming endpoint.
func (e Edge) To() string { return e[1] }

// Graph is a level of the design.
type Graph struct {
	Nodes []NodeDef `json:"nodes"`
	Edges []Edge    `json:"edges"`
}

// Published are the reference numbers the regression suite checks against.
type Published struct {
	Params          float64 `json:"params,omitempty"`
	ActiveParams    float64 `json:"activeParams,omitempty"`
	KVBytesPerToken float64 `json:"kvBytesPerToken,omitempty"`
	Source          string  `json:"source,omitempty"`
	// Tolerance is the allowed relative difference, defaulting to 0.5%. It is
	// set wider only where the published figure is itself a rounded headline
	// number such as "22B active".
	Tolerance float64 `json:"tolerance,omitempty"`
}

// DocMeta names a design and records what it is meant to reproduce.
type DocMeta struct {
	Name      string     `json:"name"`
	Family    string     `json:"family,omitempty"`
	Notes     string     `json:"notes,omitempty"`
	Published *Published `json:"published,omitempty"`
}

// UiState is the editor's view of the document, carried along with it.
type UiState struct {
	Positions map[string][2]float64 `json:"positions,omitempty"`
	Collapsed []string              `json:"collapsed,omitempty"`
}

// Doc is a design.
type Doc struct {
	Version int                  `json:"version"`
	Meta    DocMeta              `json:"meta"`
	Symbols map[string]SymbolDef `json:"symbols"`
	Graph   Graph                `json:"graph"`
	// Rules is what this design has decided the design rules mean to it,
	// keyed by rule id: "error", "warning", "info" or "off".
	//
	// A rule that is right in general is sometimes wrong here — Gemma cannot
	// use a fused attention kernel and there is no arrangement of the design
	// that changes it — and the alternative to recording that is people
	// learning to read past a warning. It lives in the document rather than in
	// the editor because it is a decision about the design, and it should
	// travel with the file and show up in review.
	Rules map[string]string `json:"rules,omitempty"`
	// Defs are blocks this design defines for itself, keyed by type name.
	//
	// The built-in catalog is fixed, but a design can carry its own composites
	// — a new attention variant, a different block arrangement — the way a
	// KiCad project carries its own symbol library. They resolve exactly like
	// built-in composites, so shape checking, parameter counting and code
	// generation need to know nothing about where a block came from.
	//
	// Typed loosely here because the definition lives in the catalog, which
	// sits above the IR.
	Defs map[string]any `json:"defs,omitempty"`
	// Configurations are named sets of symbol values this design can be built
	// at, keyed by name.
	//
	// Four GPT-2 presets are the same architecture at four sizes, and holding
	// them apart means an architectural change has to be made four times. A
	// configuration overrides symbols and nothing else: a variant that changed
	// the graph would be a different design, and calling it a configuration
	// would be a way of losing track of that.
	Configurations map[string]Configuration `json:"configurations,omitempty"`
	// Active names the configuration in force. Empty means the symbols as
	// written, which is what a design without configurations always has.
	Active string   `json:"active,omitempty"`
	Ui     *UiState `json:"ui,omitempty"`

	// SymbolOrder is the order the document wrote its symbols in, recovered
	// from the raw JSON because neither a Go map nor a JSON object keeps one.
	// The symbol list is something a person wrote and reads back — D before H
	// before dh, not alphabetical.
	SymbolOrder []string `json:"-"`
	// ConfigurationOrder is the same for configurations: the order they were
	// written in, which is the order a picker should offer them in. Small
	// before large, not alphabetical.
	ConfigurationOrder []string `json:"-"`
}

// Configuration is one named set of symbol values.
type Configuration struct {
	Doc string `json:"doc,omitempty"`
	// Symbols override the document's own, by name. A symbol the configuration
	// does not mention keeps whatever the design says, so a configuration says
	// only what is different about it.
	Symbols map[string]SymbolDef `json:"symbols"`
}

// Clone is a deep copy.
//
// Through JSON, so a copy is exactly what the document would be if it had been
// saved and reopened: nothing shared, and nothing carried across that the file
// format does not hold. The symbol order is the one thing JSON cannot carry, so
// it is copied across by hand.
func (d *Doc) Clone() (*Doc, error) {
	raw, err := json.Marshal(d)
	if err != nil {
		return nil, err
	}
	var out Doc
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	out.SymbolOrder = append([]string{}, d.SymbolOrder...)
	return &out, nil
}

// SymbolTable is every symbol resolved to a number.
type SymbolTable struct {
	// Order is the evaluation order actually used, which is dependency order.
	Order []string `json:"order"`
	// Values is the numeric value of every symbol; runtime symbols carry their
	// default.
	Values map[string]float64 `json:"values"`
	// DesignValues is the concrete design symbols only, runtime excluded. This
	// is the environment shapes are compared under.
	DesignValues map[string]float64 `json:"designValues"`
	// Runtime is the set of symbols that stay indeterminate.
	Runtime map[string]bool   `json:"-"`
	Docs    map[string]string `json:"docs"`
	Errors  []string          `json:"errors"`
}

// Endpoint is one end of an edge.
type Endpoint struct {
	Node string
	Port string
}

// SplitEndpoint splits "node:port". The node id may itself contain slashes, so
// the split is at the last colon.
func SplitEndpoint(endpoint string) (Endpoint, error) {
	i := strings.LastIndex(endpoint, ":")
	if i < 0 {
		return Endpoint{}, fmt.Errorf("malformed endpoint %q, expected \"node:port\"", endpoint)
	}
	return Endpoint{Node: endpoint[:i], Port: endpoint[i+1:]}, nil
}

// JoinPath builds the dotted path identifying a node inside nested graphs,
// such as "layers/block/attn".
func JoinPath(prefix, id string) string {
	if prefix == "" {
		return id
	}
	return prefix + "/" + id
}
