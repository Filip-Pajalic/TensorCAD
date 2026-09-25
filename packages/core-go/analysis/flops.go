package analysis

import (
	"math"

	"github.com/tensorcad/core/catalog"
)

// FlopsResult is the arithmetic the design does.
//
// Matmul FLOPs are counted the standard way, a multiply-accumulate as 2, and
// the sequence-dependent attention work is kept apart from the rest because the
// familiar 2N and 6N rules exclude it. Backward is twice forward; full
// activation recomputation adds one more forward pass.
//
// Reference: PaLM appendix B gives the training form 6N + 12*L*H*Q*T.
type FlopsResult struct {
	// FwdDense is matmul FLOPs per token that do not depend on sequence length.
	FwdDense float64 `json:"fwdDense"`
	// FwdAttention is the score and value-product FLOPs per token at this T.
	FwdAttention float64 `json:"fwdAttention"`
	// FwdAttentionUnmasked is the same term counted as if nothing were masked,
	// which is what a profiler reports.
	FwdAttentionUnmasked float64 `json:"fwdAttentionUnmasked"`
	FwdTotal             float64 `json:"fwdTotal"`
	FwdTotalUnmasked     float64 `json:"fwdTotalUnmasked"`
	// Elementwise is norms, activations, RoPE and residual adds: memory-bound,
	// and excluded from the totals above.
	Elementwise float64 `json:"elementwise"`
	// TrainPerToken is forward plus backward, per token.
	TrainPerToken float64 `json:"trainPerToken"`
	// AttentionShare is the fraction of forward FLOPs spent inside attention.
	AttentionShare float64 `json:"attentionShare"`
	// RuleOfThumb2N is the usual inference approximation.
	RuleOfThumb2N float64 `json:"ruleOfThumb2N"`
	// RuleOfThumb6N is the usual training approximation.
	RuleOfThumb6N float64            `json:"ruleOfThumb6N"`
	ByPath        map[string]float64 `json:"byPath"`
	ByCategory    map[string]float64 `json:"byCategory"`
	Errors        []string           `json:"errors"`
	// PerStream is the forward pass per token of each sequence, for a design
	// with two; absent for one, where it would only repeat FwdTotal. With two,
	// every per-token figure above is per target token, the source's share
	// spread over the target's tokens.
	PerStream []StreamFlops `json:"perStream,omitempty"`
	// FwdPerExample is one training example's forward pass, every stream's
	// tokens, for a design with two.
	FwdPerExample float64 `json:"fwdPerExample,omitempty"`
}

// FlopsOptions is the operating point FLOPs are counted at.
type FlopsOptions struct {
	Ctx catalog.AnalysisCtx
	// Recompute is "none", "selective" or "full"; full adds a forward pass.
	Recompute string
	// NonEmbeddingActive is the parameter count the rules of thumb use.
	NonEmbeddingActive float64
	// Streams says which sequence each block runs along; nil is one, T.
	Streams *Streams
}

// CountFlops adds up the arithmetic, by path and category.
func CountFlops(flat *FlatResult, opts FlopsOptions) *FlopsResult {
	res := &FlopsResult{
		RuleOfThumb2N: 2 * opts.NonEmbeddingActive,
		RuleOfThumb6N: 6 * opts.NonEmbeddingActive,
		ByPath:        map[string]float64{},
		ByCategory:    map[string]float64{},
		Errors:        []string{},
	}

	streams := opts.Streams
	if streams == nil {
		streams = &Streams{Target: opts.Ctx.T}
	}
	ownFwd := map[string]float64{}
	for i := range flat.Nodes {
		node := &flat.Nodes[i]
		if node.Def.Flops == nil {
			continue
		}
		// Measured at the length of the sequence it runs along, then spread
		// over the target's tokens.
		ctx := opts.Ctx
		ctx.T = streams.Length(node.Path)
		spread := streams.Spread(node.Path)
		per := node.Def.Flops(node.Resolved, ctx)
		// FLOPs follow the active count: a token passes through top_k experts,
		// not through all of them.
		dense := per.Fwd * node.ActiveMultiplier * spread
		seq := per.FwdSeq * node.ActiveMultiplier * spread
		// A block that does not distinguish the two is counted the same way
		// either way; only attention masks anything.
		unmasked := per.FwdSeqUnmasked
		if unmasked == 0 {
			unmasked = per.FwdSeq
		}
		seqUnmasked := unmasked * node.ActiveMultiplier * spread
		elem := per.Elementwise * node.ActiveMultiplier * spread
		stream := "T"
		if streams.OnSource(node.Path) {
			stream = "S"
		}
		ownFwd[stream] += (per.Fwd + per.FwdSeq) * node.ActiveMultiplier

		res.FwdDense += dense
		res.FwdAttention += seq
		res.FwdAttentionUnmasked += seqUnmasked
		res.Elementwise += elem

		if total := dense + seq; total > 0 {
			res.ByPath[node.Path] = total
			res.ByCategory[node.Category] += total
		}
	}

	res.FwdTotal = res.FwdDense + res.FwdAttention
	res.FwdTotalUnmasked = res.FwdDense + res.FwdAttentionUnmasked
	if streams.Two() {
		res.PerStream = []StreamFlops{
			{Symbol: "S", Length: streams.Source, Fwd: ownFwd["S"]},
			{Symbol: "T", Length: streams.Target, Fwd: ownFwd["T"]},
		}
		res.FwdPerExample = ownFwd["S"]*streams.Source + ownFwd["T"]*streams.Target
	}
	if res.FwdTotal > 0 {
		res.AttentionShare = res.FwdAttention / res.FwdTotal
	}

	// Backward costs two forwards. Recomputing activations adds a third.
	passes := 3.0
	if opts.Recompute == "full" {
		passes = 4
	}
	res.TrainPerToken = res.FwdTotal * passes
	return res
}

// FormatFlops is a FLOP count as a person reads it: 312.00 TFLOP, 1.20 PFLOP.
func FormatFlops(n float64) string {
	units := []struct {
		scale float64
		unit  string
	}{
		{1e18, "EFLOP"}, {1e15, "PFLOP"}, {1e12, "TFLOP"},
		{1e9, "GFLOP"}, {1e6, "MFLOP"}, {1e3, "kFLOP"},
	}
	for _, u := range units {
		if math.Abs(n) >= u.scale {
			return JSToFixed(n/u.scale, 2) + " " + u.unit
		}
	}
	return JSToFixed(n, 0) + " FLOP"
}
