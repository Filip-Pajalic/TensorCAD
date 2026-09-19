package rules

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/ir"
)

// flashHeadDims are the head dimensions fused attention kernels actually
// implement.
var flashHeadDims = []float64{32, 64, 96, 128, 160, 192, 256}

var shapeIssues = Rule{
	ID:          "shape",
	Title:       "Tensor interfaces",
	Description: "Every edge must carry the shape the receiving port declares.",
	Run: func(ctx *Ctx) []Finding {
		out := make([]Finding, 0, len(ctx.Infer.Issues))
		for _, i := range ctx.Infer.Issues {
			out = append(out, Finding{
				Rule: "shape", Severity: i.Severity, Path: i.Path, Port: i.Port, Message: i.Message,
			})
		}
		return out
	},
}

var symbolErrors = Rule{
	ID:          "symbols",
	Title:       "Symbol table",
	Description: "Design symbols must evaluate to numbers and must not depend on each other in a cycle.",
	Run: func(ctx *Ctx) []Finding {
		out := make([]Finding, 0, len(ctx.Symbols.Errors))
		for _, message := range ctx.Symbols.Errors {
			out = append(out, Finding{Rule: "symbols", Severity: "error", Message: message})
		}
		return out
	},
}

var blockConstraints = Rule{
	ID:          "block-constraints",
	Title:       "Block constraints",
	Description: "Per-block checks that hold regardless of which level of the design is on screen.",
	Run: func(ctx *Ctx) []Finding {
		var out []Finding
		for i := range ctx.Flat.Blocks {
			b := &ctx.Flat.Blocks[i]
			if b.Def.Constraints == nil {
				continue
			}
			for _, f := range b.Def.Constraints(b.Resolved) {
				// The block's own id is kept as the rule, so a finding can be
				// excluded or documented individually rather than as a class.
				out = append(out, Finding{
					Rule: f.ID, Severity: f.Severity, Path: b.Path,
					Param: f.Param, Port: f.Port, Message: f.Message, Hint: f.Hint,
				})
			}
		}
		return out
	},
}

var flashHeadDim = Rule{
	ID:          "flash-head-dim",
	Title:       "Kernel-supported head dimension",
	Description: "Fused attention kernels only implement a fixed set of head dimensions.",
	Run: func(ctx *Ctx) []Finding {
		var out []Finding
		seen := map[float64]bool{}
		for i := range ctx.Flat.Nodes {
			n := &ctx.Flat.Nodes[i]
			if n.Type != "sdpa" {
				continue
			}
			dh, ok := asNumber(n.Resolved.P["head_dim"])
			if !ok || seen[dh] {
				continue
			}
			seen[dh] = true
			if contains(flashHeadDims, dh) {
				continue
			}
			nearest := flashHeadDims[0]
			for _, c := range flashHeadDims[1:] {
				if math.Abs(c-dh) < math.Abs(nearest-dh) {
					nearest = c
				}
			}
			out = append(out, Finding{
				Rule: "flash-head-dim", Severity: "warning", Path: n.Path,
				Message: fmt.Sprintf("Head dimension %s is not one a fused attention kernel supports.", jsNum(dh)),
				Hint: fmt.Sprintf(
					"Use %s instead, or expect to fall back to an unfused attention path.", jsNum(nearest)),
			})
		}
		return out
	},
}

var tensorCoreShapes = Rule{
	ID:          "tensor-core-multiples",
	Title:       "Tensor-core friendly widths",
	Description: "Matrix dimensions that are not multiples of 64 leave tensor-core throughput on the table.",
	Run: func(ctx *Ctx) []Finding {
		var out []Finding
		reported := map[string]bool{}
		for i := range ctx.Flat.Nodes {
			n := &ctx.Flat.Nodes[i]
			if n.Type != "linear" {
				continue
			}
			for _, key := range []string{"in_features", "out_features"} {
				v, ok := asNumber(n.Resolved.P[key])
				if !ok || math.Mod(v, 64) == 0 {
					continue
				}
				tag := key + ":" + jsNum(v)
				if reported[tag] {
					continue
				}
				reported[tag] = true
				out = append(out, Finding{
					Rule: "tensor-core-multiples", Severity: "info", Path: n.Path,
					Message: fmt.Sprintf("Projection width %s is not a multiple of 64.", jsNum(v)),
					Hint: fmt.Sprintf(
						"Rounding up to %s usually costs little memory and speeds up the matmul.",
						jsNum(math.Ceil(v/64)*64)),
				})
			}
		}
		return out
	},
}

var vocabPadding = Rule{
	ID:          "vocab-padding",
	Title:       "Vocabulary padding",
	Description: "An unpadded vocabulary makes the largest matmul in the model slower than it needs to be.",
	Run: func(ctx *Ctx) []Finding {
		var out []Finding
		for i := range ctx.Flat.Nodes {
			n := &ctx.Flat.Nodes[i]
			if n.Type != "lm_head" {
				continue
			}
			v, ok := asNumber(n.Resolved.P["vocab"])
			if !ok || math.Mod(v, 128) == 0 {
				continue
			}
			padded := math.Ceil(v/128) * 128
			dim, _ := asNumber(n.Resolved.P["dim"])
			out = append(out, Finding{
				Rule: "vocab-padding", Severity: "info", Path: n.Path,
				Message: fmt.Sprintf("Vocabulary %s is not a multiple of 128.", jsNum(v)),
				Hint: fmt.Sprintf(
					"Padding to %s adds %s parameters and speeds up the output projection.",
					jsNum(padded), analysis.FormatCount((padded-v)*dim)),
			})
		}
		return out
	},
}

var windowVsContext = Rule{
	ID:          "window-vs-context",
	Title:       "Sliding window against context",
	Description: "A sliding window wider than the context does nothing.",
	Run: func(ctx *Ctx) []Finding {
		var out []Finding
		for i := range ctx.Flat.Nodes {
			n := &ctx.Flat.Nodes[i]
			if n.Type != "sdpa" {
				continue
			}
			w, ok := asNumber(n.Resolved.P["window"])
			if !ok || w <= 0 || w < ctx.Analysis.Options.T {
				continue
			}
			out = append(out, Finding{
				Rule: "window-vs-context", Severity: "info", Path: n.Path,
				Message: fmt.Sprintf(
					"The %s-token sliding window is at least as wide as the %s-token context, so it has no effect here.",
					jsNum(w), jsNum(ctx.Analysis.Options.T)),
				Hint: "Analyse at a longer context to see what the window buys.",
			})
		}
		return out
	},
}

var inferenceFits = Rule{
	ID:          "inference-fits",
	Title:       "Serving footprint",
	Description: "Weights plus cache must fit in device memory at the target context and concurrency.",
	Run: func(ctx *Ctx) []Finding {
		memory, options := ctx.Analysis.Memory, ctx.Analysis.Options
		cap := options.Hardware.Memory
		if memory.Infer.Total <= cap {
			return nil
		}
		hint := fmt.Sprintf("Quantize the weights, or split the model across %s devices.",
			jsNum(math.Ceil(memory.Infer.Total/cap)))
		if memory.Infer.Kv > memory.Infer.Weights {
			hint = "The cache dominates. Reduce KV heads, shorten the context, or move some layers to a sliding window."
		}
		return []Finding{{
			Rule: "inference-fits", Severity: "warning",
			Message: fmt.Sprintf(
				"Serving needs about %s but %s has %s. Weights are %s and the cache is %s at %s tokens across %s sequence(s).",
				analysis.FormatBytes(memory.Infer.Total), options.Hardware.Name, analysis.FormatBytes(cap),
				analysis.FormatBytes(memory.Infer.Weights), analysis.FormatBytes(memory.Infer.Kv),
				jsNum(options.T), jsNum(options.Concurrency)),
			Hint: hint,
		}}
	},
}

var trainingFits = Rule{
	ID:          "training-fits",
	Title:       "Training footprint",
	Description: "Weights, gradients, optimizer state and activations must fit on each GPU.",
	Run: func(ctx *Ctx) []Finding {
		memory, options := ctx.Analysis.Memory, ctx.Analysis.Options
		cap := options.Hardware.Memory
		per := memory.Train.PerGpu
		if per.Total <= cap {
			return nil
		}

		dominant := math.Max(math.Max(per.Weights, per.Grads), math.Max(per.Optimizer, per.Activations))
		var hint string
		switch {
		case dominant == per.Activations && options.Recompute == "full":
			hint = "Activations still dominate. Reduce the micro-batch or the context length."
		case dominant == per.Activations:
			hint = "Activations dominate. Turn on activation recomputation, or reduce the micro-batch."
		case options.Parallel.Zero < 3:
			hint = fmt.Sprintf(
				"Model state dominates. Raise the ZeRO stage from %d to 3, or increase tensor parallelism.",
				options.Parallel.Zero)
		default:
			hint = "Model state dominates even when fully sharded. Add more data-parallel replicas or use a smaller optimizer state."
		}

		return []Finding{{
			Rule: "training-fits", Severity: "warning",
			Message: fmt.Sprintf(
				"Training needs about %s per GPU but %s has %s. Weights %s, gradients %s, optimizer %s, activations %s.",
				analysis.FormatBytes(per.Total), options.Hardware.Name, analysis.FormatBytes(cap),
				analysis.FormatBytes(per.Weights), analysis.FormatBytes(per.Grads),
				analysis.FormatBytes(per.Optimizer), analysis.FormatBytes(per.Activations)),
			Hint: hint,
		}}
	},
}

var logitsMemory = Rule{
	ID:          "logits-memory",
	Title:       "Logits buffer",
	Description: "At a large vocabulary the logits tensor can rival the rest of the activations.",
	Run: func(ctx *Ctx) []Finding {
		memory := ctx.Analysis.Memory
		if memory.Train.Activations <= 0 {
			return nil
		}
		share := memory.Train.Logits / memory.Train.Activations
		if share < 0.25 {
			return nil
		}
		return []Finding{{
			Rule: "logits-memory", Severity: "info",
			Message: fmt.Sprintf("The logits buffer is %s, which is %s%% of all activation memory.",
				analysis.FormatBytes(memory.Train.Logits), analysis.JSToFixed(share*100, 0)),
			Hint: "Compute the cross-entropy in chunks over the sequence so the full logits tensor is never materialized.",
		}}
	},
}

var recomputeHint = Rule{
	ID:          "recompute-hint",
	Title:       "Activation recomputation",
	Description: "Flags when recomputation would free a large share of memory.",
	Run: func(ctx *Ctx) []Finding {
		memory, options := ctx.Analysis.Memory, ctx.Analysis.Options
		if options.Recompute != "none" {
			return nil
		}
		per := memory.Train.PerGpu
		if per.Total <= 0 {
			return nil
		}
		share := per.Activations / per.Total
		if share < 0.5 {
			return nil
		}
		return []Finding{{
			Rule: "recompute-hint", Severity: "info",
			Message: fmt.Sprintf("Activations are %s%% of the training footprint at batch %s and %s tokens.",
				analysis.JSToFixed(share*100, 0), jsNum(options.B), jsNum(options.T)),
			Hint: "Full recomputation trades about 33% more compute for most of that memory.",
		}}
	},
}

var attentionShare = Rule{
	ID:          "attention-share",
	Title:       "Attention share of compute",
	Description: "Warns when the sequence-dependent term makes the 6N rule misleading.",
	Run: func(ctx *Ctx) []Finding {
		share := ctx.Analysis.Flops.AttentionShare
		if share < 0.2 {
			return nil
		}
		return []Finding{{
			Rule: "attention-share", Severity: "info",
			Message: fmt.Sprintf("At %s tokens, attention is %s%% of forward FLOPs.",
				jsNum(ctx.Analysis.Options.T), analysis.JSToFixed(share*100, 0)),
			Hint: "The 2N and 6N rules of thumb quietly stop working here. Compare against the reported attention term rather than the parameter count.",
		}}
	},
}

var chinchillaRatio = Rule{
	ID:          "chinchilla",
	Title:       "Token budget",
	Description: "Compares the training budget against the compute-optimal ratio.",
	Run: func(ctx *Ctx) []Finding {
		c := ctx.Analysis.Chinchilla
		if ctx.Analysis.Options.TokensWereDefault {
			return nil
		}
		if c.OverTrainingRatio >= 0.5 && c.OverTrainingRatio <= 20 {
			return nil
		}
		hint := "This is fine when inference cost matters more than training cost, which is usually why it is done."
		if c.OverTrainingRatio < 0.5 {
			hint = "Either train longer or shrink the model; as drawn, most of the parameters will not be paid for."
		}
		return []Finding{{
			Rule: "chinchilla", Severity: "info",
			Message: fmt.Sprintf("%sB tokens is %sx the compute-optimal budget of %sB. %s",
				analysis.JSToFixed(ctx.Analysis.Options.Tokens/1e9, 0),
				analysis.JSToFixed(c.OverTrainingRatio, 1),
				analysis.JSToFixed(c.OptimalTokens/1e9, 0), c.Verdict),
			Hint: hint,
		}}
	},
}

var unusedSymbols = Rule{
	ID:          "unused-symbol",
	Title:       "Unused symbols",
	Description: "A symbol nothing refers to is usually a leftover from an edit.",
	Run: func(ctx *Ctx) []Finding {
		used := map[string]bool{}
		for i := range ctx.Flat.Blocks {
			for _, sym := range ctx.Flat.Blocks[i].Resolved.S {
				for _, name := range sym.Symbols() {
					used[name] = true
				}
			}
		}
		// Shape patterns reference runtime symbols directly.
		for name := range ctx.Symbols.Runtime {
			used[name] = true
		}
		// A symbol used by another symbol's expression counts as used.
		for _, name := range symbolNames(ctx) {
			expr := expressionOf(ctx.Doc.Symbols[name])
			if expr == "" {
				continue
			}
			// Substring, not parse: a symbol mentioned anywhere in another
			// symbol's expression is being used, even in a form this rule has
			// no business evaluating.
			for _, dep := range ctx.Symbols.Order {
				if strings.Contains(expr, dep) {
					used[dep] = true
				}
			}
		}

		var out []Finding
		for _, name := range symbolNames(ctx) {
			if used[name] {
				continue
			}
			out = append(out, Finding{
				Rule: "unused-symbol", Severity: "info",
				Message: fmt.Sprintf("Symbol %q is not referenced by any block.", name),
				Hint:    "Remove it, or wire it into the parameter that should follow it.",
			})
		}
		return out
	},
}

// expressionOf is a symbol's expression, or "" when it is a plain number. The
// document spells an expression two ways: a bare string, or a design symbol
// whose value is a string.
func expressionOf(def ir.SymbolDef) string {
	switch def.Kind {
	case "expr":
		return def.Expr
	case "design":
		if !def.HasNumber {
			return def.Expr
		}
	}
	return ""
}

// symbolNames is the order the document wrote its symbols in, which is the
// order a person reads them back in and therefore the order findings about them
// come out in.
func symbolNames(ctx *Ctx) []string {
	if len(ctx.Doc.SymbolOrder) == len(ctx.Doc.Symbols) {
		return ctx.Doc.SymbolOrder
	}
	names := make([]string, 0, len(ctx.Doc.Symbols))
	for name := range ctx.Doc.Symbols {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

var danglingOutputs = Rule{
	ID:          "dangling-output",
	Title:       "Unused outputs",
	Description: "A computed tensor nobody consumes is dead weight.",
	Run: func(ctx *Ctx) []Finding {
		var out []Finding
		// An output is used when it appears as the source of an edge.
		consumed := map[string]bool{}
		for _, e := range ctx.Doc.Graph.Edges {
			consumed[e.From()] = true
		}
		for _, node := range ctx.Doc.Graph.Nodes {
			if node.Type == "output" || node.Type == "boundary_out" {
				continue
			}
			ports, ok := ctx.Infer.Ports[node.ID]
			if !ok {
				continue
			}
			names := make([]string, 0, len(ports.Out))
			for name := range ports.Out {
				names = append(names, name)
			}
			sort.Strings(names)
			for _, portName := range names {
				if consumed[node.ID+":"+portName] {
					continue
				}
				out = append(out, Finding{
					Rule: "dangling-output", Severity: "warning",
					Path: node.ID, Port: portName,
					Message: fmt.Sprintf("Output %q is not connected to anything.", portName),
					Hint:    "Connect it downstream or delete the block.",
				})
			}
		}
		return out
	},
}

var publishedDrift = Rule{
	ID:          "published-drift",
	Title:       "Drift from the published model",
	Description: "A preset whose parameter count no longer matches its source has drifted.",
	Run: func(ctx *Ctx) []Finding {
		pub := ctx.Doc.Meta.Published
		if pub == nil || pub.Params == 0 {
			return nil
		}
		tolerance := pub.Tolerance
		if tolerance == 0 {
			tolerance = 0.005
		}
		delta := math.Abs(ctx.Analysis.Params.Total-pub.Params) / pub.Params
		if delta < tolerance {
			return nil
		}
		hint := ""
		if pub.Source != "" {
			hint = "Compare against " + pub.Source
		}
		return []Finding{{
			Rule: "published-drift", Severity: "warning",
			Message: fmt.Sprintf(
				"This design computes %s parameters but is recorded as %s (%s%% off, tolerance %s%%).",
				analysis.FormatCount(ctx.Analysis.Params.Total), analysis.FormatCount(pub.Params),
				analysis.JSToFixed(delta*100, 2), analysis.JSToFixed(tolerance*100, 1)),
			Hint: hint,
		}}
	},
}

var activeParamsDrift = Rule{
	ID:          "active-params-drift",
	Title:       "Active parameter drift",
	Description: "A sparse design's active parameter count should match what its authors report.",
	Run: func(ctx *Ctx) []Finding {
		pub := ctx.Doc.Meta.Published
		if pub == nil || pub.ActiveParams == 0 {
			return nil
		}
		tolerance := pub.Tolerance
		if tolerance == 0 {
			tolerance = 0.005
		}
		delta := math.Abs(ctx.Analysis.Params.Active-pub.ActiveParams) / pub.ActiveParams
		if delta < tolerance {
			return nil
		}
		return []Finding{{
			Rule: "active-params-drift", Severity: "warning",
			Message: fmt.Sprintf(
				"This design activates %s parameters per token but is recorded as %s (%s%% off).",
				analysis.FormatCount(ctx.Analysis.Params.Active), analysis.FormatCount(pub.ActiveParams),
				analysis.JSToFixed(delta*100, 2)),
		}}
	},
}

// userBlocks checks the blocks a design defines for itself.
//
// A definition that fails to compile is dropped from the catalog rather than
// thrown, so without this rule the only symptom would be "unknown block type"
// against every instance of it, which points at the wrong thing.
var userBlocks = Rule{
	ID:          "user-blocks",
	Title:       "Block definitions",
	Description: "Blocks a design defines for itself must name real parameters and real ports.",
	Run: func(ctx *Ctx) []Finding {
		if len(ctx.Doc.Defs) == 0 {
			return nil
		}
		builtIn := map[string]bool{}
		for name := range catalog.Builtin {
			builtIn[name] = true
		}
		// Sorted, because a Go map has no order and the panel should not
		// reshuffle between runs.
		types := make([]string, 0, len(ctx.Doc.Defs))
		for t := range ctx.Doc.Defs {
			types = append(types, t)
		}
		sort.Strings(types)

		var out []Finding
		for _, t := range types {
			for _, message := range catalog.ValidateUserBlock(ctx.Doc.Defs[t], t, builtIn) {
				out = append(out, Finding{
					Rule: "user-blocks", Severity: "error",
					Message: fmt.Sprintf("Block %q: %s", t, message),
					Hint:    "Edit the definition, or delete it and rebuild the block.",
				})
			}
		}
		return out
	},
}

// Rules is every check, in the order they run.
var Rules = []Rule{
	userBlocks,
	shapeIssues,
	symbolErrors,
	blockConstraints,
	flashHeadDim,
	tensorCoreShapes,
	vocabPadding,
	windowVsContext,
	inferenceFits,
	trainingFits,
	logitsMemory,
	recomputeHint,
	attentionShare,
	chinchillaRatio,
	unusedSymbols,
	danglingOutputs,
	publishedDrift,
	activeParamsDrift,
}

func asNumber(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	}
	return 0, false
}

func contains(list []float64, v float64) bool {
	for _, x := range list {
		if x == v {
			return true
		}
	}
	return false
}

// jsNum interpolates a number into a message the way JavaScript would.
func jsNum(v float64) string { return analysis.JSNumber(v) }
