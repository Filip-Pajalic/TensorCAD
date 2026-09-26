// Package start makes a new design from a kind of model and a size.
//
// An empty sheet and fifty-five blocks is not a way to start, and neither is a
// preset: that is somebody else's model at somebody else's size. What somebody
// starting out knows is the kind of model they want and roughly how big, or
// what they have to train it on. So a new design is a reference design of that
// kind, scaled to that size with its proportions kept — the same machinery the
// bench uses to shrink one, run in whichever direction the size asks for.
//
// A reference design is a preset, and a preset is held to its published
// figure, so the starting point is always something that was actually built.
package start

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/scale"
)

// Family is a kind of model a design can start as.
type Family struct {
	ID string `json:"id"`
	// Name is what the choice is called.
	Name string `json:"name"`
	// Summary says what the kind is, in a sentence somebody new can read.
	Summary string `json:"summary"`
	// Base is the preset it starts from.
	Base string `json:"base"`
	// Sizes are the sizes worth offering, in parameters, smallest first.
	Sizes []float64 `json:"sizes"`
	// Sequence is the length a run of this kind trains at, which becomes the
	// new design's T. The reference designs carry the length they serve at —
	// Qwen3's is 32,768 — and a first design measured there fits nowhere.
	// Zero keeps the reference's own, for a kind whose T is not a context
	// length: T5's is a 114-token target, a vision transformer's is patches.
	Sequence float64 `json:"sequence,omitempty"`
}

// Families are the kinds of model a new design can be, in the order offered.
var Families = []Family{
	{
		ID:   "dense",
		Name: "Llama-style transformer",
		Summary: "What most language models are now: grouped-query attention, a gated feed-forward, " +
			"rotary positions and RMS norm, in a stack of identical layers.",
		Base:     "llama-3-8b",
		Sizes:    []float64{125e6, 350e6, 1e9, 3e9, 8e9, 70e9},
		Sequence: 4096,
	},
	{
		ID:   "classic",
		Name: "GPT-2-style transformer",
		Summary: "The one every tutorial builds: full multi-head attention, a plain feed-forward, " +
			"learned positions and layer norm. The easiest to read.",
		Base:     "gpt2-small",
		Sizes:    []float64{15e6, 30e6, 125e6, 350e6, 774e6, 1.5e9},
		Sequence: 1024,
	},
	{
		ID:   "moe",
		Name: "Mixture of experts",
		Summary: "The feed-forward is many small experts and each token uses a few of them, so the " +
			"model holds many more parameters than it spends on any one token.",
		Base:     "qwen3-30b-a3b",
		Sizes:    []float64{1e9, 3e9, 8e9, 30e9, 100e9},
		Sequence: 4096,
	},
	{
		ID:   "hybrid",
		Name: "Hybrid: state space and attention",
		Summary: "Mostly Mamba layers, which carry a fixed-size state instead of a cache that grows " +
			"with the text, with attention every eighth layer and experts in every other feed-forward.",
		Base:     "jamba-v0.1",
		Sizes:    []float64{1e9, 4e9, 10e9, 20e9, 52e9},
		Sequence: 4096,
	},
	{
		ID:   "encoder-decoder",
		Name: "Encoder–decoder",
		Summary: "Two stacks: an encoder reads the whole input at once, and a decoder writes the output " +
			"while attending to it. For translation and other text-to-text work.",
		Base:  "t5-small",
		Sizes: []float64{30e6, 60e6, 250e6, 750e6, 3e9},
	},
	{
		ID:   "vision",
		Name: "Vision transformer",
		Summary: "A transformer over patches of an image, with no vocabulary and attention in both " +
			"directions, trained by predicting what is hidden from what is not.",
		Base:  "ijepa-vit-h14",
		Sizes: []float64{40e6, 100e6, 300e6, 630e6, 1.3e9},
	},
}

// FamilyByID looks one up.
func FamilyByID(id string) (*Family, error) {
	for i := range Families {
		if Families[i].ID == id {
			return &Families[i], nil
		}
	}
	ids := make([]string, len(Families))
	for i, f := range Families {
		ids[i] = f.ID
	}
	return nil, fmt.Errorf("there is no kind of model called %q; the kinds are %s", id, strings.Join(ids, ", "))
}

// Request is what the new design should be.
type Request struct {
	// Family is the kind of model, by ID.
	Family string `json:"family"`
	// Params is the size to aim for, counting every parameter. Leave it at zero
	// and set Fit to ask for the largest that trains on one device instead.
	Params float64 `json:"params,omitempty"`
	// Fit is a hardware profile: the design should be the largest of its kind
	// whose training fits on one of these, under the analysis options given.
	Fit string `json:"fit,omitempty"`
	// Name is the new design's name; empty makes one from the kind and size.
	Name string `json:"name,omitempty"`
}

// Result is the new design and what it came to.
type Result struct {
	Doc *ir.Doc `json:"doc"`
	// Family and Base are what it started as.
	Family string `json:"family"`
	Base   string `json:"base"`
	// Params is the size reached, and Target the size asked for — or, for a
	// fit, the size the search settled on.
	Params float64 `json:"params"`
	Target float64 `json:"target"`
	// Symbols are the new design's sizes that moved: width, depth, heads.
	Symbols map[string]float64 `json:"symbols"`
	// Device is the profile the training footprint was measured on.
	Device string `json:"device"`
	// TrainBytes is the training footprint on one device of it.
	TrainBytes float64 `json:"trainBytes"`
	// Budget is what one device allows after headroom.
	Budget float64 `json:"budget"`
	// Fits says whether TrainBytes is within Budget.
	Fits  bool     `json:"fits"`
	Notes []string `json:"notes"`
}

// Headroom is the share of a device's memory left free, as the cluster
// planner leaves it, so that the two cannot disagree about what fits.
const Headroom = 0.1

// The smallest size a search will try. Below it most families are a handful
// of heads of a handful of dimensions, which is not a model of the kind.
const smallest = 1e6

// New makes a design of a kind, at a size.
//
// The options are the conditions the training footprint is measured under,
// the operating point as the editor holds it; the device count is taken to be
// one, because "fits" here means fits on one.
func New(req Request, options analysis.Options) (*Result, error) {
	family, err := FamilyByID(req.Family)
	if err != nil {
		return nil, err
	}
	base, err := presets.Get(family.Base)
	if err != nil {
		return nil, err
	}
	if t, ok := base.Symbols["T"]; ok && family.Sequence > 0 && t.Kind == "runtime" {
		t.Number, t.HasNumber = family.Sequence, true
		base.Symbols["T"] = t
	}
	one := 1.0
	options.GPUs = &one
	options.Parallel = nil
	if req.Fit != "" {
		options.Hardware = req.Fit
	}
	hardware, err := analysis.ResolveHardware(options.Hardware)
	if err != nil {
		return nil, err
	}
	budget := hardware.Memory * (1 - Headroom)

	var made *made
	switch {
	case req.Params > 0:
		made, err = sized(base, req.Params, options)
	case req.Fit != "":
		made, err = fitted(base, budget, options)
	default:
		return nil, errors.New("say how big: a parameter count, or a device the training has to fit on")
	}
	if err != nil {
		return nil, err
	}

	doc := made.doc
	// Named for the size asked for, which is how it will be talked about: a
	// "1b" that came out at 976 million is still the one-billion design.
	doc.Meta.Name = req.Name
	if doc.Meta.Name == "" {
		doc.Meta.Name = family.ID + "-" + size(made.target)
	}
	doc.Meta.Notes = fmt.Sprintf(
		"A new %s of %s parameters, started from %s and scaled with its proportions kept. %s",
		strings.ToLower(family.Name[:1])+family.Name[1:], inWords(made.params), family.Base, family.Summary)
	doc.Meta.Published = nil

	return &Result{
		Doc:        doc,
		Family:     family.ID,
		Base:       family.Base,
		Params:     made.params,
		Target:     made.target,
		Symbols:    moved(base, doc),
		Device:     hardware.ID,
		TrainBytes: made.train,
		Budget:     budget,
		Fits:       made.train <= budget,
		Notes:      made.notes,
	}, nil
}

type made struct {
	doc            *ir.Doc
	params, target float64
	train          float64
	notes          []string
}

// sized scales the base to a parameter count and measures what it takes.
func sized(base *ir.Doc, params float64, options analysis.Options) (*made, error) {
	if params < smallest {
		return nil, fmt.Errorf("%s parameters is smaller than any model of this kind can usefully be; "+
			"the smallest this will make is %s", size(params), size(smallest))
	}
	opts := scale.Options{TargetParams: params}
	result, err := scale.Design(base, opts)
	if err != nil {
		return nil, err
	}
	// A vocabulary of a hundred thousand tokens is a small share of an eight
	// billion parameter model and most of a small one, which scaled that way
	// comes out three layers deep. Small models tie the output projection to
	// the embedding instead — Llama 3.2's one and three billion do, with the
	// same vocabulary as Llama 3's eight — so a design whose tables would be
	// more than a third of it does too.
	if embeddingShare(result.Doc) > 1.0/3 && !tied(result.Doc) {
		yes := true
		opts.TieHead = &yes
		again, err := scale.Design(base, opts)
		if err != nil {
			return nil, err
		}
		if tied(again.Doc) {
			result = again
			result.Notes = append(result.Notes,
				"Tied the output projection to the embedding, as small models do, so the vocabulary "+
					"is not most of the parameters.")
		}
	}
	if doc, count, ok := wholeGroups(base, result.Doc, params); ok {
		result.Doc, result.Achieved = doc, count
	}
	train, err := training(result.Doc, options)
	if err != nil {
		return nil, err
	}
	return &made{
		doc: result.Doc, params: result.Achieved, target: params,
		train: train, notes: result.Notes,
	}, nil
}

// fitted is the largest design of the kind whose training fits the budget.
//
// A search over the size, measuring each candidate's footprint with the full
// analysis: the footprint is weights, gradients and optimizer state, which go
// with the parameter count, plus activations, which go with the width, the
// depth and the sequence — so no formula over the count alone would be right.
func fitted(base *ir.Doc, budget float64, options analysis.Options) (*made, error) {
	low, err := sized(base, smallest, options)
	if err != nil {
		return nil, err
	}
	if low.train > budget {
		return nil, fmt.Errorf(
			"even a %s version of this needs %s to train on one device, and one allows %s; "+
				"a shorter sequence or a smaller batch would bring it down",
			size(smallest), analysis.FormatBytes(low.train), analysis.FormatBytes(budget))
	}
	best := low
	// Nothing trains in less than a byte a parameter, so the budget in bytes
	// is a ceiling on the count.
	lo, hi := smallest, budget
	for i := 0; i < 28 && hi/lo > 1.02; i++ {
		mid := math.Sqrt(lo * hi)
		candidate, err := sized(base, mid, options)
		if err != nil {
			return nil, err
		}
		if candidate.train <= budget {
			if candidate.params > best.params {
				best = candidate
			}
			lo = mid
		} else {
			hi = mid
		}
	}
	best.target = best.params
	return best, nil
}

// wholeGroups keeps query heads in whole groups over the key heads.
//
// The width is rounded to whole heads, and a head count the reference's
// grouping does not divide falls back to fewer key heads: 32 query heads over
// 8 becomes 13 over 1 at a billion parameters, which is a different attention
// rather than a smaller one. So the head count moves to a neighbouring one the
// grouping divides, the width with it, and the depth takes up the difference
// in size. Of the candidates within a twentieth of the size asked for, the one
// that moved the width least wins, since the width is what the scaling chose.
func wholeGroups(base, doc *ir.Doc, target float64) (*ir.Doc, float64, bool) {
	before := ir.ResolveSymbols(base).DesignValues
	after := ir.ResolveSymbols(doc).DesignValues
	h0, kv0 := before["H"], before["Hkv"]
	h, kv, dh, width := after["H"], after["Hkv"], after["dh"], after["D"]
	if h0 <= 0 || kv0 <= 0 || kv0 >= h0 || math.Mod(h0, kv0) != 0 || dh <= 0 {
		return nil, 0, false
	}
	ratio := h0 / kv0
	if kv > 0 && h/kv == ratio {
		return nil, 0, false
	}
	// Only sizes written as numbers: an expression over something else would
	// not follow a value written here.
	number := func(name string) bool {
		def, ok := doc.Symbols[name]
		return ok && (def.Kind == "literal" || def.Kind == "design") && def.HasNumber && def.Expr == ""
	}
	for _, name := range []string{"D", "H", "Hkv"} {
		if !number(name) {
			return nil, 0, false
		}
	}
	depth := ""
	for _, name := range []string{"L", "G"} {
		if number(name) {
			depth = name
			break
		}
	}
	if depth == "" {
		return nil, 0, false
	}
	layers := after[depth]

	set := func(d *ir.Doc, name string, v float64) {
		def := d.Symbols[name]
		def.Number = v
		d.Symbols[name] = def
	}
	count := func(d *ir.Doc) float64 {
		return analysis.CountParams(analysis.Flatten(d, ir.ResolveSymbols(d))).Total
	}

	var best *ir.Doc
	bestCount, bestMove := 0.0, math.Inf(1)
	for _, groups := range []float64{math.Floor(h / ratio), math.Ceil(h / ratio)} {
		if groups < 1 {
			continue
		}
		heads := groups * ratio
		for l := math.Max(1, math.Round(layers/2)); l <= math.Round(layers*2); l++ {
			candidate, err := doc.Clone()
			if err != nil {
				return nil, 0, false
			}
			set(candidate, "D", heads*dh)
			set(candidate, "H", heads)
			set(candidate, "Hkv", groups)
			set(candidate, depth, l)
			n := count(candidate)
			if math.Abs(n-target) > 0.05*target {
				continue
			}
			if move := math.Abs(heads*dh-width) + math.Abs(l-layers)*dh/4; move < bestMove {
				best, bestCount, bestMove = candidate, n, move
			}
		}
	}
	if best == nil {
		return nil, 0, false
	}
	return best, bestCount, true
}

// embeddingShare is how much of a design is its embedding tables.
func embeddingShare(doc *ir.Doc) float64 {
	counts := analysis.CountParams(analysis.Flatten(doc, ir.ResolveSymbols(doc)))
	if counts.Total == 0 {
		return 0
	}
	return counts.Embedding / counts.Total
}

// tied reports whether a design's output projection shares the embedding.
func tied(doc *ir.Doc) bool {
	for _, n := range doc.Graph.Nodes {
		if n.Type == "lm_head" {
			v, _ := n.Params["tied"].(bool)
			return v
		}
	}
	// No output projection to tie, which is as tied as it gets.
	return true
}

// training is the footprint of training a design on one device.
func training(doc *ir.Doc, options analysis.Options) (float64, error) {
	result, err := analysis.Analyze(doc, options, analysis.Inputs{})
	if err != nil {
		return 0, err
	}
	if result.Memory == nil {
		return 0, errors.New("the analysis did not measure memory for this design")
	}
	return result.Memory.Train.PerGpu.Total, nil
}

// sizeSymbols are the symbols that say how big a design is, in the order a
// summary reads them.
var sizeSymbols = []string{"L", "G", "D", "H", "Hkv", "dh", "F", "E", "K", "Fe", "Lp", "Dp"}

// moved is each size symbol's value in the new design, where it has one.
func moved(base, doc *ir.Doc) map[string]float64 {
	before := ir.ResolveSymbols(base).DesignValues
	after := ir.ResolveSymbols(doc).DesignValues
	out := map[string]float64{}
	for _, name := range sizeSymbols {
		if v, ok := after[name]; ok {
			if _, had := before[name]; had {
				out[name] = v
			}
		}
	}
	return out
}

// size names a count the way a model is named: 1.3b, 124m, 500k.
func size(n float64) string {
	switch {
	case n >= 1e9:
		return trim(n/1e9) + "b"
	case n >= 1e6:
		return analysis.JSNumber(math.Round(n/1e6)) + "m"
	case n >= 1e3:
		return analysis.JSNumber(math.Round(n/1e3)) + "k"
	}
	return analysis.JSNumber(n)
}

// inWords is a count as a sentence says it: 1.03 billion, 976 million.
func inWords(n float64) string {
	switch {
	case n >= 1e9:
		return analysis.JSNumber(math.Round(n/1e7)/100) + " billion"
	case n >= 1e6:
		return analysis.JSNumber(math.Round(n/1e6)) + " million"
	}
	return analysis.JSNumber(math.Round(n))
}

// trim keeps one decimal where it says something: 1.3b, but 8b rather than 8.0b.
func trim(v float64) string {
	if v >= 10 || math.Abs(v-math.Round(v)) < 0.05 {
		return analysis.JSNumber(math.Round(v))
	}
	return analysis.JSToFixed(v, 1)
}
