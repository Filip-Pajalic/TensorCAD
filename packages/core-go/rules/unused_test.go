package rules_test

import (
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
)

func unused(t *testing.T, doc *ir.Doc) map[string]bool {
	t.Helper()
	rep, err := rules.Validate(doc, analysis.Options{})
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]bool{}
	for _, f := range rep.Findings {
		if f.Rule == "unused-symbol" {
			out[f.Message] = true
		}
	}
	return out
}

// A symbol only an attention expression reads is still used: the block holds
// its value, not its name, so the rule has to read the expression as written.
// T5 names how far its buckets reach, and a prefix-LM names its prefix.
func TestASymbolAnExpressionReadsIsUsed(t *testing.T) {
	for _, name := range []string{"t5-small", "flan-t5-base"} {
		if found := unused(t, presets.MustGet(name)); len(found) > 0 {
			t.Errorf("%s: %v", name, found)
		}
	}

	doc := writtenOut(t, map[string]any{"causal": false, "mask": "kv <= q or kv < P"})
	for _, sym := range []string{"P", "Q"} {
		doc.Symbols[sym] = ir.SymbolDef{Kind: "design", Number: 16, HasNumber: true}
		doc.SymbolOrder = append(doc.SymbolOrder, sym)
	}
	found := unused(t, doc)
	if found[`Symbol "P" is not referenced by any block.`] {
		t.Error("P is read by the mask and was reported unused")
	}
	if !found[`Symbol "Q" is not referenced by any block.`] {
		t.Errorf("Q is read by nothing and was not reported: %v", found)
	}
}
