package explain

import (
	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/ir"
)

// MaskView is one attention's mask as the kernel sees it, for the inspector to
// draw beside the expression that made it.
type MaskView struct {
	// Path is the block that was asked about.
	Path string `json:"path"`
	// Attention is the sdpa inside it that was drawn: the block itself, or the
	// first attention it expands into.
	Attention string `json:"attention"`
	Found     bool   `json:"found"`
	catalog.MaskGrid
}

// Mask draws the attention mask of the block at path at the operating point's
// sequence length, for one head.
//
// A composite is answered through the first sdpa inside it, which is where its
// mask ends up: a transformer block hands its expression to its attention, and
// that to the kernel.
func Mask(doc *ir.Doc, path string, options analysis.Options, head float64) *MaskView {
	symbols := ir.ResolveSymbols(doc)
	flat := analysis.Flatten(doc, symbols)
	out := &MaskView{Path: path, MaskGrid: catalog.MaskGrid{Kept: []float64{}}}
	for _, n := range subtree(flat, path) {
		if n.Type != "sdpa" {
			continue
		}
		T, B := analysis.Sequence(symbols, options)
		a := catalog.AttentionOf(n.Resolved)
		if head < 0 || head >= a.Heads {
			head = 0
		}
		out.Attention = n.Path
		out.Found = true
		out.MaskGrid = a.Grid(T, B, head)
		return out
	}
	return out
}
