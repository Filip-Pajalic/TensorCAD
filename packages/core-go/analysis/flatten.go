package analysis

import (
	"fmt"
	"math"

	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/ir"
)

// FlatNode is one primitive in the flattened design: the things that carry the
// formulas.
type FlatNode struct {
	// Path is the full path, e.g. "layers/block/attn/q_proj".
	Path     string
	Type     string
	Category string
	Def      *catalog.BlockDef
	Resolved *catalog.Resolved
	// Multiplier is the product of the enclosing container counts: how many
	// copies exist.
	Multiplier float64
	// ActiveMultiplier is the product of the enclosing active counts: how many
	// a single token passes through.
	ActiveMultiplier float64
	// Container is the path of the nearest enclosing repeat container, if any.
	Container string
}

// FlatBlock is a node at any level, composites and containers included.
type FlatBlock struct {
	Path             string
	Type             string
	Kind             string
	Category         string
	Def              *catalog.BlockDef
	Resolved         *catalog.Resolved
	Multiplier       float64
	ActiveMultiplier float64
}

// Repeat is what a container contributes, kept for reporting.
type Repeat struct {
	Path   string
	Type   string
	Count  float64
	Active float64
}

// FlatResult is the design reduced to the nodes that can be counted.
type FlatResult struct {
	// Nodes are primitives only: these carry the formulas.
	Nodes []FlatNode
	// Blocks is every node at every level.
	Blocks  []FlatBlock
	Errors  []string
	Repeats []Repeat
}

// Flatten reduces a design to the primitive nodes that carry the formulas.
//
// Composites are expanded; a repeat container contributes a multiplier instead
// of being unrolled, so a 126-layer model still flattens to a few dozen nodes.
func Flatten(doc *ir.Doc, symbols *ir.SymbolTable) *FlatResult {
	// A design may define blocks of its own, so the catalog is the document's.
	cat := catalog.Of(doc)
	out := &FlatResult{}
	seen := map[string]bool{}

	var walk func(graph *ir.Graph, prefix string, multiplier, activeMultiplier float64, container string, depth int)
	walk = func(graph *ir.Graph, prefix string, multiplier, activeMultiplier float64, container string, depth int) {
		if depth > 32 {
			out.Errors = append(out.Errors, fmt.Sprintf(
				"Graph nesting deeper than 32 levels at %q; is a composite expanding into itself?", prefix))
			return
		}
		for _, node := range graph.Nodes {
			path := ir.JoinPath(prefix, node.ID)
			if seen[path] {
				where := prefix
				if where == "" {
					where = "<root>"
				}
				out.Errors = append(out.Errors, fmt.Sprintf("Duplicate node id %q in %q", node.ID, where))
				continue
			}
			seen[path] = true

			def, ok := cat[node.Type]
			if !ok {
				out.Errors = append(out.Errors, fmt.Sprintf("Unknown block type %q at %q", node.Type, path))
				continue
			}
			resolved := catalog.ResolveNodeParams(def, node.Params, symbols)
			for _, e := range resolved.Errors {
				out.Errors = append(out.Errors, path+": "+e)
			}
			out.Blocks = append(out.Blocks, FlatBlock{
				Path: path, Type: def.Type, Kind: def.Kind, Category: def.Category,
				Def: def, Resolved: resolved,
				Multiplier: multiplier, ActiveMultiplier: activeMultiplier,
			})

			switch {
			case catalog.IsPrimitive(def):
				out.Nodes = append(out.Nodes, FlatNode{
					Path: path, Type: def.Type, Category: def.Category,
					Def: def, Resolved: resolved,
					Multiplier: multiplier, ActiveMultiplier: activeMultiplier,
					Container: container,
				})

			case catalog.IsComposite(def):
				exp, ok := catalog.Expand(def, resolved.RawFull, resolved)
				if !ok {
					out.Errors = append(out.Errors, fmt.Sprintf(
						"%s: expansion failed: %s has no expansion", path, node.Type))
					continue
				}
				walk(&ir.Graph{Nodes: exp.Nodes, Edges: exp.Edges}, path,
					multiplier, activeMultiplier, container, depth+1)

			case catalog.IsContainer(def):
				counts := catalog.MultipliersOf(def, resolved)
				if !finite(counts.Total) || counts.Total < 0 || !finite(counts.Active) || counts.Active < 0 {
					out.Errors = append(out.Errors, path+": container counts must be non-negative numbers")
					continue
				}
				out.Repeats = append(out.Repeats, Repeat{
					Path: path, Type: def.Type, Count: counts.Total, Active: counts.Active,
				})
				if node.Graph == nil {
					out.Errors = append(out.Errors, fmt.Sprintf("%s: container %q has no subgraph", path, def.Type))
					continue
				}
				// Only a layer stack owns its activations for recomputation.
				inner := container
				if def.Type == "repeat" {
					inner = path
				}
				walk(node.Graph, path,
					multiplier*counts.Total, activeMultiplier*counts.Active, inner, depth+1)
			}
		}
	}

	walk(&doc.Graph, "", 1, 1, "", 0)
	return out
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
