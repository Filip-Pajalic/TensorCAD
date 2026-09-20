package analysis

import (
	"math"
)

// ParamsResult is what the design weighs.
//
// Total is every trainable weight; Active is what a single token actually uses,
// which differs from Total only for a mixture of experts; NonEmbedding is the
// number the 2N and 6N FLOPs rules are written against.
type ParamsResult struct {
	Total        float64 `json:"total"`
	Active       float64 `json:"active"`
	Embedding    float64 `json:"embedding"`
	Head         float64 `json:"head"`
	NonEmbedding float64 `json:"nonEmbedding"`
	// NonEmbeddingActive is the active count without the embedding table: the N
	// in the 2N rule.
	NonEmbeddingActive float64 `json:"nonEmbeddingActive"`
	// Expert is the weights that live inside a moe_experts container. They are
	// most of a sparse model and they shard differently from the rest: expert
	// parallelism hands whole experts to whole devices, where tensor
	// parallelism splits every matrix across its group.
	Expert     float64            `json:"expert"`
	ByPath     map[string]float64 `json:"byPath"`
	ByCategory map[string]float64 `json:"byCategory"`
	ByType     map[string]float64 `json:"byType"`
	Errors     []string           `json:"errors"`
}

var embeddingTypes = map[string]bool{"embedding": true, "pos_embedding": true}

// CountParams adds up the weights, by path, category and type.
func CountParams(flat *FlatResult) *ParamsResult {
	res := &ParamsResult{
		ByPath:     map[string]float64{},
		ByCategory: map[string]float64{},
		ByType:     map[string]float64{},
		Errors:     append([]string{}, flat.Errors...),
	}

	for i := range flat.Nodes {
		node := &flat.Nodes[i]
		if node.Def.ParamCount == nil {
			continue
		}
		per := node.Def.ParamCount(node.Resolved)
		if !finite(per) {
			res.Errors = append(res.Errors, node.Path+": parameter count is not a finite number")
			continue
		}
		total := per * node.Multiplier
		active := per * node.ActiveMultiplier
		if total == 0 {
			continue
		}

		res.Total += total
		res.Active += active
		res.ByPath[node.Path] = total
		res.ByCategory[node.Category] += total
		res.ByType[node.Type] += total

		if embeddingTypes[node.Type] {
			res.Embedding += total
		} else if node.Type == "lm_head" {
			res.Head += total
		}
		if node.Expert {
			res.Expert += total
		}
	}

	res.NonEmbedding = res.Total - res.Embedding
	res.NonEmbeddingActive = res.Active - res.Embedding
	return res
}

// FormatCount is a parameter count as a person reads it: 8.03B, 124.4M, 12.9K.
func FormatCount(n float64) string {
	abs := math.Abs(n)
	switch {
	case abs >= 1e12:
		return JSToFixed(n/1e12, 2) + "T"
	case abs >= 1e9:
		return JSToFixed(n/1e9, 2) + "B"
	case abs >= 1e6:
		return JSToFixed(n/1e6, 1) + "M"
	case abs >= 1e3:
		return JSToFixed(n/1e3, 1) + "K"
	}
	return JSNumber(n)
}
