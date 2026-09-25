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
	BytesPerSequenceFixed float64 `json:"bytesPerSequenceFixed"`
	// BytesPerTokenDecompressed is the same cache under an engine that does not
	// absorb the weights latent attention compressed against. It equals
	// BytesPerToken for every design that has no latent attention in it.
	BytesPerTokenDecompressed float64            `json:"bytesPerTokenDecompressed"`
	ByPath                    map[string]float64 `json:"byPath"`
	Errors                    []string           `json:"errors"`
}

// CountKvCache adds up the inference state.
func CountKvCache(flat *FlatResult, ctx catalog.AnalysisCtx, streams *Streams) *KvResult {
	res := &KvResult{ByPath: map[string]float64{}, Errors: []string{}}

	for i := range flat.Nodes {
		node := &flat.Nodes[i]
		if node.Def.StateBytes == nil {
			continue
		}
		own := ctx
		own.T = streams.Length(node.Path)
		s := node.Def.StateBytes(node.Resolved, own)
		// A source is read in full, once, before anything is generated, and
		// what a block along it computed is not needed again: an encoder's own
		// keys and values are discarded. What generation keeps of the source is
		// cross-attention's keys and values, which the decoder's blocks count,
		// once per request.
		if streams.OnSource(node.Path) {
			continue
		}
		perToken := s.PerToken * node.Multiplier
		perSeq := s.PerSequence * node.Multiplier
		res.BytesPerToken += perToken
		res.BytesPerSequenceFixed += perSeq
		// A block with no second way to hold its state contributes the same
		// either way, which is what makes the total meaningful for a design
		// that mixes latent attention with anything else.
		if s.PerTokenDecompressed > 0 {
			res.BytesPerTokenDecompressed += s.PerTokenDecompressed * node.Multiplier
		} else {
			res.BytesPerTokenDecompressed += perToken
		}
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
