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

// StreamedParams is how many weights a decode step has to read at a given batch
// size: the resident count for a dense model, and something between the active
// count and the resident count for a mixture of experts.
//
// One token passes through `top_k` of `experts`, so it reads that share of them.
// A batch of tokens does not: they route independently, and the expert a token
// skipped is read anyway if any other token in the step wanted it. What a step
// reads is the union rather than one token's share, and the union reaches every
// expert well before the batch reaches the expert count.
//
// A block that exists in `n` copies of which one token uses `a` is read by a
// step of `batch` tokens with probability 1 - (1 - a/n)^batch per copy. For
// everything outside an expert a/n is 1 and the term is the whole block, which
// is why this reduces to the active count at batch 1 and to the resident count
// as the batch grows.
//
// It assumes a token picks its experts independently and uniformly. A real
// router is trained towards balance, which spreads a batch over *more* experts
// than independent sampling would, so this is a floor on what gets read and the
// throughput that follows from it is a ceiling.
func StreamedParams(flat *FlatResult, batch float64) float64 {
	if !(batch > 1) {
		batch = 1
	}
	total := 0.0
	for i := range flat.Nodes {
		node := &flat.Nodes[i]
		if node.Def.ParamCount == nil {
			continue
		}
		per := node.Def.ParamCount(node.Resolved)
		if !finite(per) || per == 0 || node.Multiplier <= 0 {
			continue
		}
		share := node.ActiveMultiplier / node.Multiplier
		if share >= 1 {
			total += per * node.Multiplier
			continue
		}
		total += per * node.Multiplier * (1 - math.Pow(1-share, batch))
	}
	return total
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
