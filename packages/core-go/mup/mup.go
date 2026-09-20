// Package mup builds a maximal-update-parametrization ladder.
//
// The point of μP is that a learning rate tuned on a narrow model is still the
// right learning rate on a wide one, provided the initialization and the
// per-parameter learning rates are scaled by width in a particular way. That
// lets a sweep happen at a width which fits on one device and transfer to a
// width which does not, and that is most of what makes a sweep affordable.
//
// The scaling is Table 3 of Tensor Programs V, restated in terms of a width
// multiplier m = width / base_width, which is what a reader has in front of
// them rather than a fan_in they would have to look up:
//
//	                          init variance   Adam LR
//	input weights and biases    1/fan_in          1
//	hidden weights              1/fan_in       1/fan_in
//	output weights              1/fan_in²      1/fan_in
//
// A hidden matrix's fan_in is the width, so against the base its variance goes
// as 1/m and its Adam rate as 1/m. The readout's variance goes as 1/m². An
// embedding's fan_in is the vocabulary, which does not move with the width, so
// nothing about it changes.
//
// What the ladder does not do is decide the base learning rate. That is what
// the sweep is for.
//
// https://arxiv.org/abs/2203.03466
package mup

import (
	"fmt"
	"math"
	"sort"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/scale"
)

// Class is which row of the table a weight belongs to.
//
// Three rather than four: the paper puts every bias in with the input weights,
// because both are the case whose fan_in does not grow with the width.
type Class string

const (
	// Input is a weight whose fan_in is fixed: an embedding, a bias, a norm
	// gain, and anything else that does not widen on both sides.
	Input Class = "input"
	// Hidden is a matrix that widens on both sides.
	Hidden Class = "hidden"
	// Output is a matrix whose fan_out is fixed: the readout, and a router,
	// which is a readout over experts.
	Output Class = "output"
)

// Scaling is what to multiply the base model's settings by, for one class.
type Scaling struct {
	Class Class `json:"class"`
	// InitStd multiplies the base model's initialization standard deviation.
	// The paper gives variances; this is their square root, which is what an
	// initializer is handed.
	InitStd float64 `json:"initStd"`
	// AdamLR multiplies the base model's learning rate.
	AdamLR float64 `json:"adamLr"`
	// Paths are the blocks in this class, so the grouping can be checked
	// against the design rather than taken on trust.
	Paths []string `json:"paths"`
	Why   string   `json:"why"`
}

// Rung is one model in the ladder.
type Rung struct {
	// Width is what the width symbol came out as, which is not always what was
	// asked for: a width is held to a whole number of heads.
	Width float64 `json:"width"`
	// Multiplier is Width over the base width: the m every rule is written in.
	Multiplier float64 `json:"multiplier"`
	Heads      float64 `json:"heads"`
	Params     float64 `json:"params"`
	Doc        *ir.Doc `json:"doc"`
	// Base marks the rung the hyperparameters are tuned at, where m is 1.
	Base    bool      `json:"base"`
	Scaling []Scaling `json:"scaling"`
	Notes   []string  `json:"notes"`
}

// Ladder is the whole set.
type Ladder struct {
	// WidthSymbol is what the ladder moved, normally D.
	WidthSymbol string `json:"widthSymbol"`
	// BaseWidth is the rung m is measured against, after rounding to heads.
	BaseWidth float64 `json:"baseWidth"`
	// HeadDim is what was held fixed while the width moved.
	HeadDim float64  `json:"headDim"`
	Rungs   []Rung   `json:"rungs"`
	Notes   []string `json:"notes"`
}

// Options is which widths to build and which of them is the base.
type Options struct {
	// Widths are the rungs. Empty halves the design's own width down to a
	// width still worth sweeping at.
	Widths []float64 `json:"widths,omitempty"`
	// BaseWidth is the width the sweep happens at. Zero takes the narrowest.
	BaseWidth float64 `json:"baseWidth,omitempty"`
	// WidthSymbols are the symbols that move with the width, beyond D itself.
	// Empty takes the same set `scale` uses, and an expression over D follows
	// on its own either way.
	WidthSymbols []string `json:"widthSymbols,omitempty"`
}

// Build produces the ladder.
func Build(doc *ir.Doc, opts Options) (*Ladder, error) {
	const symbol = "D"
	full, ok := literal(doc, symbol)
	if !ok {
		return nil, fmt.Errorf(
			"this design has no numeric %s, so there is no width to build a ladder over", symbol)
	}
	headDim, ok := literal(doc, "dh")
	if !ok || headDim <= 0 {
		return nil, fmt.Errorf(
			"this design has no numeric head dimension dh; μP grows the number of heads and holds " +
				"the width of one, and without dh there is nothing to hold")
	}

	widths := append([]float64{}, opts.Widths...)
	if len(widths) == 0 {
		widths = defaultLadder(full, headDim)
	}
	for _, w := range widths {
		if !(w > 0) {
			return nil, fmt.Errorf("a width has to be a positive number, not %s", analysis.JSNumber(w))
		}
	}
	sort.Float64s(widths)

	wantBase := opts.BaseWidth
	if wantBase == 0 {
		wantBase = widths[0]
	}
	if !(wantBase > 0) {
		return nil, fmt.Errorf(
			"the base width has to be a positive number, not %s", analysis.JSNumber(wantBase))
	}
	// Round the base to heads before anything is measured against it, so that
	// m comes out exactly 1 at the rung the sweep happens on.
	base := toHeads(wantBase, headDim) * headDim

	out := &Ladder{
		WidthSymbol: symbol, BaseWidth: base, HeadDim: headDim,
		Rungs: []Rung{}, Notes: []string{},
	}
	seen := map[float64]bool{}
	for _, want := range widths {
		rung, err := rungAt(doc, want, base, opts)
		if err != nil {
			return nil, err
		}
		if seen[rung.Width] {
			continue
		}
		seen[rung.Width] = true
		out.Rungs = append(out.Rungs, *rung)
	}
	if !seen[base] {
		out.Notes = append(out.Notes, fmt.Sprintf(
			"No rung is at the base width of %s, so every multiplier is against a model the ladder "+
				"does not contain.", analysis.JSNumber(base)))
	}

	if tiedReadout(doc) {
		out.Notes = append(out.Notes,
			"The readout is tied to the embedding, so one tensor is an input weight and an output "+
				"weight at once. Its initialization is the embedding's and cannot be scaled twice; "+
				"apply the output rule at the readout as a multiplier on its logits instead.")
	}
	out.Notes = append(out.Notes,
		"Sweep at the base rung, then carry the learning rate up the ladder multiplied per class.",
		"Nothing here decides the base learning rate. That is what the sweep is for.",
		"μP also asks attention to divide its logits by the head dimension rather than by its square "+
			"root. That is a property of the design rather than of this ladder.")
	return out, nil
}

// rungAt is the design at one width.
//
// The width moves and the head dimension does not, so the head count is what
// grows. That is the convention μP is stated in for transformers, and it is
// also the one that leaves every head the same shape it had at the base, which
// is what makes the transfer an argument rather than a hope.
func rungAt(base *ir.Doc, want, baseWidth float64, opts Options) (*Rung, error) {
	doc, notes, err := atWidth(base, want, opts)
	if err != nil {
		return nil, err
	}
	width, _ := literal(doc, "D")
	heads, _ := literal(doc, "H")
	if width != want {
		notes = append([]string{fmt.Sprintf(
			"Asked for a width of %s; %s is the nearest whole number of heads.",
			analysis.JSNumber(want), analysis.JSNumber(width))}, notes...)
	}

	flat := analysis.Flatten(doc, ir.ResolveSymbols(doc))
	params := analysis.CountParams(flat)
	if len(params.Errors) > 0 {
		return nil, fmt.Errorf("at a width of %s: %s", analysis.JSNumber(width), params.Errors[0])
	}

	// Classify against the same design twice as wide, rather than against a
	// list of block types: a weight is a hidden one when both of its sides
	// moved, and that is something to measure rather than to assert.
	twice, _, err := atWidth(base, width*2, opts)
	if err != nil {
		return nil, err
	}
	return &Rung{
		Width: width, Multiplier: width / baseWidth, Heads: heads, Params: params.Total,
		Doc: doc, Base: width == baseWidth, Notes: notes,
		Scaling: scalingAt(flat, analysis.Flatten(twice, ir.ResolveSymbols(twice)), width/baseWidth),
	}, nil
}

// atWidth is the design at one width, which `scale` already knows how to
// produce: the heads follow the width, the query-to-key ratio is kept as far as
// divisibility allows, and a feed-forward width moves with the residual one
// whether it is written as a number or as an expression over D.
func atWidth(base *ir.Doc, width float64, opts Options) (*ir.Doc, []string, error) {
	return scale.AtWidth(base, width, scale.Options{WidthSymbols: opts.WidthSymbols})
}

// scalingAt is Table 3, with the design's own blocks sorted into it.
//
// `flat` is the rung and `wider` is the same design at twice the width. A
// weight whose fan_in moved between them has a fan_in that is the width; the
// two answers together are the row.
func scalingAt(flat, wider *analysis.FlatResult, m float64) []Scaling {
	widths := map[string][2]float64{}
	for i := range wider.Nodes {
		node := &wider.Nodes[i]
		if in, out, ok := fans(node); ok {
			widths[node.Path] = [2]float64{in, out}
		}
	}

	paths := map[Class][]string{Input: {}, Hidden: {}, Output: {}}
	for i := range flat.Nodes {
		node := &flat.Nodes[i]
		// A tied readout counts no weights of its own, and it is still where
		// the output rule applies: skipping it for having nothing to count
		// would drop the one row the design most needs to be told about.
		if node.Type == "lm_head" && node.Resolved.Bool("tied") {
			paths[Output] = append(paths[Output], node.Path)
			continue
		}
		if node.Def.ParamCount == nil || node.Def.ParamCount(node.Resolved) == 0 {
			continue
		}
		class := Input
		if in, out, ok := fans(node); ok {
			if grew, seen := widths[node.Path]; seen {
				switch {
				case in != grew[0] && out != grew[1]:
					class = Hidden
				case in != grew[0]:
					class = Output
				}
			}
		}
		paths[class] = append(paths[class], node.Path)
	}
	for _, list := range paths {
		sort.Strings(list)
	}
	return []Scaling{
		{
			Class: Input, InitStd: 1, AdamLR: 1, Paths: paths[Input],
			Why: "Nothing here widens on the way in: an embedding's fan_in is the vocabulary, and a " +
				"bias, a norm gain and a per-head scalar have no fan_in at all.",
		},
		{
			Class: Hidden, InitStd: 1 / math.Sqrt(m), AdamLR: 1 / m, Paths: paths[Hidden],
			Why: "fan_in is the width, so the initialization variance goes as 1/m and the Adam rate as 1/m.",
		},
		{
			Class: Output, InitStd: 1 / m, AdamLR: 1 / m, Paths: paths[Output],
			Why: "A weight that widens on the way in and not on the way out initializes at 1/fan_in² " +
				"rather than 1/fan_in, so its standard deviation goes as 1/m.",
		},
	}
}

// tiedReadout is whether the design shares its readout with its embedding.
//
// It is a fact about the design rather than about any one rung, which is why it
// is asked once here rather than noticed at every width.
func tiedReadout(doc *ir.Doc) bool {
	for _, node := range doc.Graph.Nodes {
		if node.Type != "lm_head" {
			continue
		}
		if tied, ok := node.Params["tied"].(bool); ok && tied {
			return true
		}
	}
	return false
}

// fans is what a block reads and what it writes, for the blocks where both are
// a single number. A norm gain, a per-head scalar and a depthwise convolution
// have no such pair, and they are the input row whatever the width does.
func fans(node *analysis.FlatNode) (in, out float64, ok bool) {
	r := node.Resolved
	switch node.Type {
	case "linear":
		return r.Num("in_features"), r.Num("out_features"), true
	case "lm_head":
		return r.Num("dim"), r.Num("vocab"), true
	case "embedding":
		return r.Num("vocab"), r.Num("dim"), true
	case "pos_embedding":
		return r.Num("max_seq"), r.Num("dim"), true
	case "learned_tokens":
		return r.Num("count"), r.Num("dim"), true
	case "topk_router":
		return r.Num("d_model"), r.Num("experts"), true
	case "conv2d":
		k := r.Num("kernel")
		return r.Num("in_channels") * k * k / math.Max(1, r.Num("groups")), r.Num("out_channels"), true
	}
	return 0, 0, false
}

// defaultLadder halves the design's own width down to something still worth
// sweeping at: four rungs, and nothing narrower than four heads.
func defaultLadder(full, headDim float64) []float64 {
	floor := math.Max(4*headDim, 128)
	out := []float64{}
	for w := full; w >= floor && len(out) < 4; w /= 2 {
		out = append(out, w)
	}
	if len(out) == 0 {
		out = []float64{full}
	}
	return out
}

// toHeads is how many whole heads a width is, and never fewer than one.
func toHeads(width, headDim float64) float64 {
	return math.Max(1, math.Round(width/headDim))
}

func literal(doc *ir.Doc, name string) (float64, bool) {
	s, ok := doc.Symbols[name]
	if !ok || !s.HasNumber || s.Expr != "" {
		return 0, false
	}
	return s.Number, true
}

func setLiteral(doc *ir.Doc, name string, value float64) {
	s, ok := doc.Symbols[name]
	if !ok {
		return
	}
	s.Number, s.HasNumber, s.Expr = value, true, ""
	doc.Symbols[name] = s
}
