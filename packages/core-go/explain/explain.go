// Package explain describes one block: what it is, what it was given, and what
// it costs.
//
// This is the learning surface. Every number the tool reports should be
// traceable to a formula and a source, and every parameter should show both the
// expression somebody wrote and the value it evaluated to.
package explain

import (
	"sort"
	"strings"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
)

// Param is one parameter as written and as evaluated.
type Param struct {
	// Expression is what the document wrote, when it wrote an expression.
	Expression string `json:"expression,omitempty"`
	// Value is what it evaluated to.
	Value any    `json:"value"`
	Doc   string `json:"doc,omitempty"`
}

// Copies is how many of a block exist and how many a token passes through.
type Copies struct {
	Total  float64 `json:"total"`
	Active float64 `json:"active"`
}

// Shapes are the patterns on a block's pins.
type Shapes struct {
	In  map[string]string `json:"in"`
	Out map[string]string `json:"out"`
}

// Contribution is what a block and everything inside it add to the whole model.
type Contribution struct {
	Params                float64 `json:"params"`
	ActiveParams          float64 `json:"activeParams"`
	ShareOfParams         float64 `json:"shareOfParams"`
	FlopsPerToken         float64 `json:"flopsPerToken"`
	ShareOfFlops          float64 `json:"shareOfFlops"`
	ActivationBytes       float64 `json:"activationBytes"`
	CacheBytesPerToken    float64 `json:"cacheBytesPerToken"`
	CacheBytesPerSequence float64 `json:"cacheBytesPerSequence"`
}

// Line is one primitive in the breakdown.
type Line struct {
	Path   string  `json:"path"`
	Type   string  `json:"type"`
	Params float64 `json:"params"`
}

// Explanation is the whole report for one block.
type Explanation struct {
	Path  string            `json:"path"`
	Type  string            `json:"type"`
	Kind  string            `json:"kind"`
	Label string            `json:"label,omitempty"`
	Docs  catalog.BlockDocs `json:"docs"`
	// Copies is how many exist and how many a token passes through.
	Copies Copies           `json:"copies"`
	Params map[string]Param `json:"params"`
	// ParamOrder is the order the block declares its parameters in, which is
	// the order a reader expects them in.
	ParamOrder  []string     `json:"paramOrder"`
	Shapes      Shapes       `json:"shapes"`
	Contributes Contribution `json:"contributes"`
	// Breakdown is the primitives this block expands into, largest first.
	Breakdown []Line `json:"breakdown"`
	NotFound  bool   `json:"notFound,omitempty"`
}

// Inputs are results already computed, so a caller that has them does not pay
// for them twice.
type Inputs struct {
	Symbols *ir.SymbolTable
	// Infer must be shape inference with composites expanded.
	Infer    *infer.Result
	Flat     *analysis.FlatResult
	Analysis *analysis.Result
}

// findNode walks a path down through the document's nested graphs.
func findNode(doc *ir.Doc, path string) *ir.NodeDef {
	graph := &doc.Graph
	var node *ir.NodeDef
	for _, part := range strings.Split(path, "/") {
		if part == "" {
			continue
		}
		if graph == nil {
			return nil
		}
		var hit *ir.NodeDef
		for i := range graph.Nodes {
			if graph.Nodes[i].ID == part {
				hit = &graph.Nodes[i]
				break
			}
		}
		if hit == nil {
			return nil
		}
		node = hit
		graph = hit.Graph
	}
	return node
}

// subtree is the part of the flattened graph rooted at path.
func subtree(flat *analysis.FlatResult, path string) []*analysis.FlatNode {
	prefix := path + "/"
	var out []*analysis.FlatNode
	for i := range flat.Nodes {
		n := &flat.Nodes[i]
		if n.Path == path || strings.HasPrefix(n.Path, prefix) {
			out = append(out, n)
		}
	}
	return out
}

func shapesOnly(side map[string]catalog.PortSpec) map[string]string {
	out := make(map[string]string, len(side))
	for name, port := range side {
		out[name] = port.Shape
	}
	return out
}

// Block explains one block of a design.
func Block(doc *ir.Doc, path string, options analysis.Options, pre Inputs) (*Explanation, error) {
	symbols := pre.Symbols
	if symbols == nil {
		symbols = ir.ResolveSymbols(doc)
	}
	expanded := pre.Infer
	if expanded == nil {
		expanded = infer.Shapes(doc, symbols, infer.Options{ExpandComposites: true})
	}
	flat := pre.Flat
	if flat == nil {
		flat = analysis.Flatten(doc, symbols)
	}
	result := pre.Analysis
	if result == nil {
		var err error
		result, err = analysis.Analyze(doc, options, analysis.Inputs{
			Symbols: symbols, Flat: flat, Expanded: expanded,
		})
		if err != nil {
			return nil, err
		}
	}

	var block *analysis.FlatBlock
	for i := range flat.Blocks {
		if flat.Blocks[i].Path == path {
			block = &flat.Blocks[i]
			break
		}
	}

	if block == nil {
		// The path may name a node inside a composite expansion, which Blocks
		// records only for document nodes. Fall back to the inference result.
		resolved, ok := expanded.Resolved[path]
		ports, hasPorts := expanded.Ports[path]
		if !ok || !hasPorts {
			return notFound(path), nil
		}
		def, known := catalog.Of(doc)[resolved.Type]
		if !known {
			return notFound(path), nil
		}
		return build(path, def, resolved, ports, Copies{Total: 1, Active: 1},
			subtree(flat, path), result, nil), nil
	}

	ports := expanded.Ports[path]
	return build(path, block.Def, block.Resolved, ports,
		Copies{Total: block.Multiplier, Active: block.ActiveMultiplier},
		subtree(flat, path), result, findNode(doc, path)), nil
}

func notFound(path string) *Explanation {
	return &Explanation{
		Path: path, Type: "unknown", Kind: "primitive",
		Docs:     catalog.BlockDocs{Summary: "No block at \"" + path + "\"."},
		Params:   map[string]Param{},
		Shapes:   Shapes{In: map[string]string{}, Out: map[string]string{}},
		NotFound: true,
	}
}

func build(
	path string, def *catalog.BlockDef, resolved *catalog.Resolved,
	ports catalog.Ports, copies Copies, nodes []*analysis.FlatNode,
	result *analysis.Result, docNode *ir.NodeDef,
) *Explanation {
	params := make(map[string]Param, len(resolved.Keys))
	for _, key := range resolved.Keys {
		value := resolved.P[key]
		p := Param{Value: value}
		// The expression only, not the value spelled as one: a width written
		// as 4096 has nothing to show beside the 4096.
		if raw, ok := resolved.RawFull[key].(string); ok && !sameAsValue(raw, value) {
			p.Expression = raw
		}
		if spec, ok := def.Params.Get(key); ok {
			p.Doc = spec.Doc
		}
		params[key] = p
	}

	var paramSum, activeSum, flopSum, activationSum, cacheToken, cacheSeq float64
	breakdown := []Line{}

	for _, node := range nodes {
		p := result.Params.ByPath[node.Path]
		paramSum += p
		if node.Multiplier > 0 {
			activeSum += (p / node.Multiplier) * node.ActiveMultiplier
		}
		flopSum += result.Flops.ByPath[node.Path]
		activationSum += result.Memory.Train.ActivationsByPath[node.Path]
		if node.Def.StateBytes != nil {
			// At two bytes an element: the cache dtype is an operating-point
			// choice, and this report is about the block.
			s := node.Def.StateBytes(node.Resolved, catalog.AnalysisCtx{
				T: result.Options.T, B: result.Options.B, Bytes: 2, Flash: result.Options.Flash,
			})
			cacheToken += s.PerToken * node.Multiplier
			cacheSeq += s.PerSequence * node.Multiplier
		}
		if p > 0 {
			breakdown = append(breakdown, Line{Path: node.Path, Type: node.Type, Params: p})
		}
	}

	// Largest first. Stable, so two blocks contributing the same stay in the
	// order the flattened graph produced them, which is the order they appear
	// in the design.
	sort.SliceStable(breakdown, func(i, j int) bool {
		return breakdown[i].Params > breakdown[j].Params
	})
	if len(breakdown) > 12 {
		breakdown = breakdown[:12]
	}

	out := &Explanation{
		Path: path, Type: def.Type, Kind: def.Kind, Docs: def.Docs,
		Copies: copies, Params: params, ParamOrder: resolved.Keys,
		// Shapes is the report's field and means shapes, so the rest of each
		// port declaration is dropped here rather than leaking into the output.
		Shapes: Shapes{In: shapesOnly(ports.In), Out: shapesOnly(ports.Out)},
		Contributes: Contribution{
			Params:                paramSum,
			ActiveParams:          jsRound(activeSum),
			FlopsPerToken:         flopSum,
			ActivationBytes:       activationSum,
			CacheBytesPerToken:    cacheToken,
			CacheBytesPerSequence: cacheSeq,
		},
		Breakdown: breakdown,
	}
	if result.Params.Total > 0 {
		out.Contributes.ShareOfParams = paramSum / result.Params.Total
	}
	if result.Flops.FwdTotal > 0 {
		out.Contributes.ShareOfFlops = flopSum / result.Flops.FwdTotal
	}
	if docNode != nil {
		out.Label = docNode.Label
	}
	return out
}

// sameAsValue reports whether an expression is just the value written out, in
// which case there is nothing to show beside it.
func sameAsValue(raw string, value any) bool {
	s, ok := value.(string)
	return ok && s == raw
}

// jsRound rounds half towards positive infinity, as Math.round does.
func jsRound(v float64) float64 {
	return float64(int64(v + 0.5))
}

// All explains every block in the document, largest contribution first.
func All(doc *ir.Doc, options analysis.Options) ([]*Explanation, error) {
	symbols := ir.ResolveSymbols(doc)
	expanded := infer.Shapes(doc, symbols, infer.Options{ExpandComposites: true})
	flat := analysis.Flatten(doc, symbols)
	result, err := analysis.Analyze(doc, options, analysis.Inputs{
		Symbols: symbols, Flat: flat, Expanded: expanded,
	})
	if err != nil {
		return nil, err
	}

	pre := Inputs{Symbols: symbols, Infer: expanded, Flat: flat, Analysis: result}
	out := make([]*Explanation, 0, len(flat.Blocks))
	for i := range flat.Blocks {
		e, err := Block(doc, flat.Blocks[i].Path, options, pre)
		if err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].Contributes.Params > out[j].Contributes.Params
	})
	return out, nil
}
