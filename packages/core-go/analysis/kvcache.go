package analysis

import (
	"math"

	"github.com/tensorcad/core/catalog"
)

// KvResult is what serving has to hold on to.
//
// Two kinds of state: what grows with every generated token (the KV cache of a
// full-attention layer) and what is fixed per sequence (a sliding window's
// bounded cache, and the recurrent state of a state-space layer).
type KvResult struct {
	// BytesPerToken is added to the cache for each new token, across all layers.
	BytesPerToken float64 `json:"bytesPerToken"`
	// BytesPerSequenceFixed is held per sequence regardless of length.
	BytesPerSequenceFixed float64            `json:"bytesPerSequenceFixed"`
	ByPath                map[string]float64 `json:"byPath"`
	Errors                []string           `json:"errors"`
}

// CountKvCache adds up the inference state.
func CountKvCache(flat *FlatResult, ctx catalog.AnalysisCtx) *KvResult {
	res := &KvResult{ByPath: map[string]float64{}, Errors: []string{}}

	for i := range flat.Nodes {
		node := &flat.Nodes[i]
		if node.Def.StateBytes == nil {
			continue
		}
		s := node.Def.StateBytes(node.Resolved, ctx)
		perToken := s.PerToken * node.Multiplier
		perSeq := s.PerSequence * node.Multiplier
		res.BytesPerToken += perToken
		res.BytesPerSequenceFixed += perSeq
		if perToken > 0 || perSeq > 0 {
			// Whichever of the two this block has; a block never has both.
			if perToken != 0 {
				res.ByPath[node.Path] = perToken
			} else {
				res.ByPath[node.Path] = perSeq
			}
		}
	}
	return res
}

// KvBytesFor is the cache for `sequences` concurrent sequences of `tokens`
// tokens each.
func KvBytesFor(kv *KvResult, tokens, sequences float64) float64 {
	return (kv.BytesPerToken*tokens + kv.BytesPerSequenceFixed) * sequences
}

// FormatBytes is a byte count as a person reads it: 4.20 GiB, 320.00 KiB.
func FormatBytes(n float64) string {
	units := []struct {
		scale float64
		unit  string
	}{
		{1024 * 1024 * 1024 * 1024 * 1024, "PiB"},
		{1024 * 1024 * 1024 * 1024, "TiB"},
		{1024 * 1024 * 1024, "GiB"},
		{1024 * 1024, "MiB"},
		{1024, "KiB"},
	}
	for _, u := range units {
		if math.Abs(n) >= u.scale {
			return JSToFixed(n/u.scale, 2) + " " + u.unit
		}
	}
	// Math.round, not toFixed(0): they differ only for a negative half, which a
	// byte count never is.
	return JSNumber(math.Round(n)) + " B"
}
