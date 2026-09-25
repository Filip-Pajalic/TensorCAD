package rules

import (
	"fmt"

	"github.com/tensorcad/core/analysis"
)

// What a packing does to a design, and what it does not.

var packingUnused = Rule{
	ID:    "packing-unused",
	Title: "Packing without a document mask",
	Description: "The operating point packs training rows with documents, and nothing in the design keeps " +
		"them apart, so the packing changes nothing.",
	Run: func(ctx *Ctx) []Finding {
		pack := ctx.Analysis.Options.Packing
		if pack == nil || ctx.Analysis.Flops.Packed != nil {
			return nil
		}
		return []Finding{{
			Rule: "packing-unused", Severity: "info",
			Message: fmt.Sprintf("Rows are packed with documents of %s tokens, and every attention here reads "+
				"across them, so the packing changes nothing.", analysis.JSNumber(pack.Mean)),
			Hint: "Keep documents apart with a mask that reads them, doc(b, q) == doc(b, kv), wired from an input " +
				"whose role is documents. Llama 3 found it mattered little in ordinary pretraining and a lot in " +
				"long-context training.",
		}}
	},
}

// A kernel that skips blocks of the score matrix computes the rest whole, and
// a document boundary inside a block makes it compute scores the mask throws
// away. That is noted once it passes a quarter of the scores kept, and warned
// about once it is a real part of what a token costs: a twentieth of the whole
// forward pass. Attention is a small share of a large model's forward pass at
// ordinary lengths, so the first is common and the second is not.
const (
	blocksNoted  = 1.25
	blocksWarned = 0.05
)

var documentBlocks = Rule{
	ID:    "document-blocks",
	Title: "Documents against the kernel's blocks",
	Description: "Documents short against a block-sparse kernel's 128-token blocks make it compute many " +
		"scores the mask throws away, which the count of scores kept does not show.",
	Run: func(ctx *Ctx) []Finding {
		p := ctx.Analysis.Flops.Packed
		pack := ctx.Analysis.Options.Packing
		if p == nil || pack == nil || p.FwdAttention <= 0 {
			return nil
		}
		ratio := p.FwdAttentionBlocks / p.FwdAttention
		excess := (p.FwdAttentionBlocks - p.FwdAttention) / p.FwdTotal
		if ratio <= blocksNoted {
			return nil
		}
		severity := "info"
		if excess > blocksWarned {
			severity = "warning"
		}
		return []Finding{{
			Rule: "document-blocks", Severity: severity,
			Message: fmt.Sprintf("In documents of %s tokens, a block-sparse kernel computes %sx the attention "+
				"scores the mask keeps: %s more a token, %s%% of the forward pass.",
				analysis.JSNumber(pack.Mean), analysis.JSToFixed(ratio, 2),
				analysis.FormatFlops(p.FwdAttentionBlocks-p.FwdAttention), analysis.JSToFixed(excess*100, 1)),
			Hint: "The count is the scores kept, and FlexAttention computes every 128-token block with any of them " +
				"in it whole, so each document boundary through a block costs what the block throws away. " +
				"Longer documents narrow the gap; a kernel that takes each document as its own sequence, as " +
				"FlashAttention's variable-length kernel does, never computes a block across a boundary.",
		}}
	},
}
