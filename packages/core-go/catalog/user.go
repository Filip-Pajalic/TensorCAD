// Blocks defined by a design rather than by this package.
//
// The built-in catalog is code: a primitive carries formulas, so it has to be.
// A composite does not — it is parameters, ports and a subgraph, and the only
// reason gqa_attention is code is that its expansion interpolates parameters
// into the graph it builds. That interpolation is a string substitution, so it
// can be written down instead.
//
// So a design can carry its own block library, the way a KiCad project carries
// its own symbols. A user block is a template subgraph whose node parameters may
// refer to the block's own parameters as $name; expanding it substitutes the
// values the instance was given and hands the result to the same shape
// inference, parameter counting and code generation as everything else.
//
// What this does not give you is a new primitive: anything needing its own
// parameter-count or FLOP formula still belongs in primitives.go.
package catalog

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"

	"github.com/tensorcad/core/ir"
)

// UserBlockDef is a composite written as data.
type UserBlockDef struct {
	Type     string              `json:"type"`
	Category string              `json:"category"`
	Params   map[string]rawParam `json:"params"`
	Ports    rawPorts            `json:"ports"`
	Graph    ir.Graph            `json:"graph"`
	Docs     rawDocs             `json:"docs"`
}

type rawParam struct {
	Type    string   `json:"type"`
	Default any      `json:"default"`
	Min     *float64 `json:"min"`
	Max     *float64 `json:"max"`
	Values  []string `json:"values"`
	Doc     string   `json:"doc"`
}

type rawPorts struct {
	In  map[string]json.RawMessage `json:"in"`
	Out map[string]json.RawMessage `json:"out"`
}

type rawDocs struct {
	Summary string   `json:"summary"`
	Formula string   `json:"formula"`
	Refs    []string `json:"refs"`
}

// $name, not bare name, so a substitution can never be a symbol by accident.
var refRe = regexp.MustCompile(`\$([A-Za-z_][A-Za-z0-9_]*)`)

// fill substitutes the instance's parameters into one template value.
//
// Only strings carry references; a number or a flag in the template is already
// the value it will have. Ex decides the bracketing, so $d_model given the
// expression "H*dh" lands as "(H*dh)" and stays one term inside a larger one.
func fill(value any, raw map[string]any) any {
	s, ok := value.(string)
	if !ok {
		return value
	}
	return refRe.ReplaceAllStringFunc(s, func(m string) string {
		return Ex(raw[m[1:]], "0")
	})
}

func fillNode(n ir.NodeDef, raw map[string]any) ir.NodeDef {
	out := n
	if n.Params != nil {
		p := make(map[string]any, len(n.Params))
		for k, v := range n.Params {
			p[k] = fill(v, raw)
		}
		out.Params = p
	}
	if n.Graph != nil {
		inner := &ir.Graph{Edges: append([]ir.Edge{}, n.Graph.Edges...)}
		for _, child := range n.Graph.Nodes {
			inner.Nodes = append(inner.Nodes, fillNode(child, raw))
		}
		out.Graph = inner
	}
	return out
}

// shapesOf reduces a declared port set to shape patterns.
//
// A boundary node carries shapes and nothing else: the rest of a port's
// declaration is about the outside of the block, and inside its expansion there
// is only a tensor arriving at a shape.
func shapesOf(side map[string]json.RawMessage) map[string]any {
	out := map[string]any{}
	for name, raw := range side {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			out[name] = s
			continue
		}
		var spec PortSpec
		if json.Unmarshal(raw, &spec) == nil {
			out[name] = spec.Shape
		}
	}
	return out
}

func portsOfRaw(side map[string]json.RawMessage) map[string]PortSpec {
	out := map[string]PortSpec{}
	for name, raw := range side {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			out[name] = Port(s)
			continue
		}
		var spec PortSpec
		if json.Unmarshal(raw, &spec) == nil {
			out[name] = NormalisePort(spec)
		}
	}
	return out
}

var paramKinds = map[string]ParamKind{
	"int": ParamInt, "num": ParamNum, "bool": ParamBool,
	"enum": ParamEnum, "str": ParamStr, "pattern": ParamPattern, "obj": ParamObj,
}

// compileUserBlock turns a definition written as data into a catalog entry.
func compileUserBlock(name string, raw any) (*BlockDef, error) {
	encoded, err := json.Marshal(raw)
	if err != nil {
		return nil, err
	}
	var def UserBlockDef
	if err := json.Unmarshal(encoded, &def); err != nil {
		return nil, err
	}
	def.Type = name

	if len(def.Ports.In) == 0 && len(def.Ports.Out) == 0 {
		return nil, fmt.Errorf("block %q declares no ports", name)
	}
	if len(def.Graph.Nodes) == 0 {
		return nil, fmt.Errorf("block %q has an empty graph", name)
	}

	// Sorted, where a built-in block's parameters keep the order they were
	// written in: a definition arrives as JSON, and decoding it into a map has
	// already thrown the author's order away.
	names := make([]string, 0, len(def.Params))
	for key := range def.Params {
		names = append(names, key)
	}
	sort.Strings(names)

	params := make(ParamList, 0, len(names))
	for _, key := range names {
		p := def.Params[key]
		kind, ok := paramKinds[p.Type]
		if !ok {
			return nil, fmt.Errorf("block %q parameter %q has unknown type %q", name, key, p.Type)
		}
		params = append(params, ParamEntry{Name: key, Spec: ParamSpec{
			Type: kind, Default: p.Default, HasDefault: p.Default != nil,
			Min: p.Min, Max: p.Max, Values: p.Values, Doc: p.Doc,
		}})
	}

	return &BlockDef{
		Kind:     "composite",
		Type:     name,
		Category: def.Category,
		Params:   params,
		Ports:    Ports{In: portsOfRaw(def.Ports.In), Out: portsOfRaw(def.Ports.Out)},
		Docs:     BlockDocs{Summary: def.Docs.Summary, Formula: def.Docs.Formula, Refs: def.Docs.Refs},
		// The expansion is data, so it is carried rather than coded.
		userGraph: &def,
	}, nil
}

// ExpandUser builds the subgraph a user-defined block stands for.
//
// The boundary nodes are generated from the declared ports rather than written,
// so a definition cannot declare one interface and wire another.
func ExpandUser(def *BlockDef, raw map[string]any) (Expansion, bool) {
	if def.userGraph == nil {
		return Expansion{}, false
	}
	u := def.userGraph
	nodes := []ir.NodeDef{
		{ID: "_in", Type: "boundary_in", Params: map[string]any{"ports": shapesOf(u.Ports.In)}},
	}
	for _, n := range u.Graph.Nodes {
		nodes = append(nodes, fillNode(n, raw))
	}
	nodes = append(nodes,
		ir.NodeDef{ID: "_out", Type: "boundary_out", Params: map[string]any{"ports": shapesOf(u.Ports.Out)}})

	return Expansion{Nodes: nodes, Edges: append([]ir.Edge{}, u.Graph.Edges...)}, true
}
