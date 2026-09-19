// Package scale shrinks a design to a parameter budget while keeping its
// proportions.
//
// This is what makes a local test bench possible: take an architecture you care
// about, shrink it until it trains in ten minutes on one GPU, and compare it
// against a baseline shrunk the same way. Because parameters grow roughly as
// layers times width squared, balanced scaling moves both by the cube root of
// the ratio, which keeps the depth-to-width aspect close to the original.
package scale

import (
	"errors"
	"fmt"
	"math"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
)

// Options steer the scaling.
type Options struct {
	// TargetParams is the parameter count to aim for.
	TargetParams float64
	// WidthSymbols scale with the width. An expression over D follows on its
	// own. Empty takes the default set.
	WidthSymbols []string
	// DepthSymbols scale with the depth. Empty takes the default set.
	DepthSymbols []string
	// WidthMultiple keeps the residual width a multiple of this. Nil defaults
	// to the head dimension.
	WidthMultiple *float64
	// Vocab replaces the vocabulary, for a bench with a smaller tokenizer.
	Vocab *float64
	// TargetBasis is whether TargetParams counts the embedding tables. At bench
	// sizes the vocabulary dominates, so "non-embedding" is usually what you
	// mean when you say "a 30M model". Empty means "total".
	TargetBasis string
	// TieHead shares the output projection with the embedding, halving the
	// vocabulary's cost.
	TieHead *bool
	// MinHeads narrows the head dimension when the scaled model would otherwise
	// have very few heads. A two-head model is not a useful bench proxy for a
	// 32-head one. Nil defaults to four.
	MinHeads *float64
	// KeepDepth holds the depth fixed and moves only the width.
	KeepDepth bool
	// MaxIterations bounds the search. Nil defaults to 48.
	MaxIterations *int
}

// Change is one symbol's before and after.
type Change struct {
	From float64 `json:"from"`
	To   float64 `json:"to"`
}

// Result is the scaled design and what it cost to get there.
type Result struct {
	Doc *ir.Doc `json:"doc"`
	// Achieved is the parameter count actually reached.
	Achieved float64           `json:"achieved"`
	Target   float64           `json:"target"`
	Changes  map[string]Change `json:"changes"`
	Notes    []string          `json:"notes"`
}

var (
	defaultWidth = []string{"D", "F", "Fe"}
	defaultDepth = []string{"L"}
)

// literal reads a symbol that is a plain number, which is what can be scaled.
// An expression follows whatever it is written over.
func literal(doc *ir.Doc, name string) (float64, bool) {
	def, ok := doc.Symbols[name]
	if !ok {
		return 0, false
	}
	switch def.Kind {
	case "literal":
		return def.Number, true
	case "design":
		if def.HasNumber {
			return def.Number, true
		}
	}
	return 0, false
}

// setLiteral writes a number, keeping the spelling the document used: a design
// symbol stays a design symbol, with its documentation.
func setLiteral(doc *ir.Doc, name string, value float64) {
	if def, ok := doc.Symbols[name]; ok {
		if def.Kind == "design" {
			def.Number, def.HasNumber, def.Expr = value, true, ""
			doc.Symbols[name] = def
			return
		}
		doc.Symbols[name] = ir.SymbolDef{Kind: "literal", Number: value, HasNumber: true, Doc: def.Doc}
		return
	}
	doc.Symbols[name] = ir.SymbolDef{Kind: "literal", Number: value, HasNumber: true}
	doc.SymbolOrder = append(doc.SymbolOrder, name)
}

// roundTo rounds to the nearest positive multiple of m.
func roundTo(value, m float64) float64 {
	return math.Max(m, jsRound(value/m)*m)
}

// jsRound rounds half towards positive infinity, as Math.round does.
func jsRound(v float64) float64 { return math.Floor(v + 0.5) }

func applyScale(base *ir.Doc, factor float64, opts Options) (*ir.Doc, []string, error) {
	doc, err := base.Clone()
	if err != nil {
		return nil, nil, err
	}
	notes := []string{}

	widthNames := opts.WidthSymbols
	if widthNames == nil {
		widthNames = defaultWidth
	}
	depthNames := opts.DepthSymbols
	if depthNames == nil {
		depthNames = defaultDepth
	}

	// Parameters grow as layers times width squared, so a balanced move takes
	// the cube root in each direction.
	widthFactor, depthFactor := math.Cbrt(factor), math.Cbrt(factor)
	if opts.KeepDepth {
		widthFactor, depthFactor = math.Sqrt(factor), 1
	}

	headDim, hasHead := literal(doc, "dh")
	originalD, hasD := literal(doc, "D")

	// A very narrow model with a wide head dimension ends up with two or three
	// heads, which is a poor proxy for the original. Narrowing the head instead
	// keeps the head count reasonable and the width granularity finer.
	minHeads := 4.0
	if opts.MinHeads != nil {
		minHeads = *opts.MinHeads
	}
	if hasHead && hasD && opts.WidthMultiple == nil {
		if wouldBe := jsRound(originalD * widthFactor / headDim); wouldBe < minHeads && headDim > 64 {
			headDim = 64
			setLiteral(doc, "dh", headDim)
			notes = append(notes, fmt.Sprintf(
				"Narrowed the head dimension to 64 so the scaled design keeps at least %s heads.",
				analysis.JSNumber(minHeads)))
		}
	}

	widthMultiple := 64.0
	switch {
	case opts.WidthMultiple != nil:
		widthMultiple = *opts.WidthMultiple
	case hasHead:
		widthMultiple = headDim
	}

	newD, hasNewD := 0.0, false
	if hasD {
		newD, hasNewD = roundTo(originalD*widthFactor, widthMultiple), true
		setLiteral(doc, "D", newD)
	}

	// Heads follow the width so the head dimension stays kernel-friendly.
	if hasNewD && hasHead {
		heads := math.Max(1, jsRound(newD/headDim))
		setLiteral(doc, "H", heads)
		originalH, hasH := literal(base, "H")
		originalKv, hasKv := literal(base, "Hkv")
		if hasH && hasKv {
			ratio := originalH / originalKv
			kv := math.Max(1, jsRound(heads/ratio))
			for kv > 1 && math.Mod(heads, kv) != 0 {
				kv--
			}
			setLiteral(doc, "Hkv", kv)
			if math.Mod(heads, kv) != 0 {
				notes = append(notes, fmt.Sprintf("Could not keep the %s:1 query-to-key ratio at %s heads.",
					analysis.JSNumber(ratio), analysis.JSNumber(heads)))
			}
		}
	}

	// Other width symbols move with the width, but only when they are
	// literals: an expression over D already follows it.
	widthRatio := widthFactor
	if hasNewD && hasD {
		widthRatio = newD / originalD
	}
	for _, name := range widthNames {
		if name == "D" {
			continue
		}
		if v, ok := literal(doc, name); ok {
			setLiteral(doc, name, roundTo(v*widthRatio, 64))
		}
	}

	for _, name := range depthNames {
		if v, ok := literal(doc, name); ok {
			setLiteral(doc, name, math.Max(1, jsRound(v*depthFactor)))
		}
	}

	// A design with leading dense layers keeps at least one of each kind.
	ld, hasLd := literal(doc, "Ld")
	l, hasL := literal(doc, "L")
	if hasLd && hasL && ld >= l {
		setLiteral(doc, "Ld", math.Max(1, l-1))
		notes = append(notes, "Reduced the leading dense layers so at least one sparse layer remains.")
	}

	if opts.Vocab != nil {
		setLiteral(doc, "V", *opts.Vocab)
	}

	if opts.TieHead != nil {
		for i := range doc.Graph.Nodes {
			if doc.Graph.Nodes[i].Type != "lm_head" {
				continue
			}
			params := map[string]any{}
			for k, v := range doc.Graph.Nodes[i].Params {
				params[k] = v
			}
			params["tied"] = *opts.TieHead
			doc.Graph.Nodes[i].Params = params
			break
		}
	}

	return doc, notes, nil
}

// Design shrinks a design towards a parameter budget.
func Design(base *ir.Doc, opts Options) (*Result, error) {
	if opts.TargetParams <= 0 {
		return nil, errors.New("targetParams must be positive")
	}
	basis := opts.TargetBasis
	if basis == "" {
		basis = "total"
	}
	measure := func(doc *ir.Doc) float64 {
		p := analysis.CountParams(analysis.Flatten(doc, ir.ResolveSymbols(doc)))
		if basis == "total" {
			return p.Total
		}
		return p.NonEmbedding
	}

	startParams := measure(base)

	// Binary search on the scale factor. The relationship is monotone but not
	// smooth, because widths are rounded to kernel-friendly multiples.
	lo, hi := 1e-8, math.Max(4, opts.TargetParams/math.Max(1, startParams)*4)
	bestDoc, bestNotes, err := applyScale(base, 1, opts)
	if err != nil {
		return nil, err
	}
	bestParams := measure(bestDoc)
	bestError := math.Abs(bestParams - opts.TargetParams)

	iterations := 48
	if opts.MaxIterations != nil {
		iterations = *opts.MaxIterations
	}
	for i := 0; i < iterations; i++ {
		mid := math.Sqrt(lo * hi)
		doc, notes, err := applyScale(base, mid, opts)
		if err != nil {
			return nil, err
		}
		params := measure(doc)
		if e := math.Abs(params - opts.TargetParams); e < bestError {
			bestDoc, bestNotes, bestParams, bestError = doc, notes, params, e
		}
		if params > opts.TargetParams {
			hi = mid
		} else {
			lo = mid
		}
		if hi/lo < 1.0001 {
			break
		}
	}

	notes := append([]string{}, bestNotes...)
	if relative := math.Abs(bestParams-opts.TargetParams) / opts.TargetParams; relative > 0.1 {
		notes = append(notes, fmt.Sprintf(
			"The closest reachable size is %s%% from the target. "+
				"Rounding the width to whole heads limits how finely the size can be tuned.",
			analysis.JSToFixed(relative*100, 0)))
	}

	finalCounts := analysis.CountParams(analysis.Flatten(bestDoc, ir.ResolveSymbols(bestDoc)))
	embeddingShare := 0.0
	if finalCounts.Total > 0 {
		embeddingShare = finalCounts.Embedding / finalCounts.Total
	}
	if embeddingShare > 0.4 {
		notes = append(notes, fmt.Sprintf(
			"The embedding table is %s%% of this design's weights. "+
				"Shrink the vocabulary or tie the output projection if you want the comparison "+
				"to be about the transformer.",
			analysis.JSToFixed(embeddingShare*100, 0)))
	}

	changes := map[string]Change{}
	for name := range base.Symbols {
		from, hasFrom := literal(base, name)
		to, hasTo := literal(bestDoc, name)
		if hasFrom && hasTo && from != to {
			changes[name] = Change{From: from, To: to}
		}
	}

	baseName, baseNotes := base.Meta.Name, base.Meta.Notes
	bestDoc.Meta.Name = baseName + "-" + formatShort(finalCounts.Total)
	bestDoc.Meta.Notes = fmt.Sprintf("Scaled down from %s (%s parameters) for a local bench run. %s",
		baseName, formatShort(startParams), baseNotes)
	// The published figure belongs to the original, not to this.
	bestDoc.Meta.Published = nil

	return &Result{
		Doc: bestDoc, Achieved: bestParams, Target: opts.TargetParams,
		Changes: changes, Notes: notes,
	}, nil
}

// formatShort names a size the way a model is named: 1.3b, 124m, 500k.
func formatShort(n float64) string {
	switch {
	case n >= 1e9:
		return analysis.JSToFixed(n/1e9, 1) + "b"
	case n >= 1e6:
		return analysis.JSNumber(jsRound(n/1e6)) + "m"
	case n >= 1e3:
		return analysis.JSNumber(jsRound(n/1e3)) + "k"
	}
	return analysis.JSNumber(n)
}
