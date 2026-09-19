// Package rules is the design-rule check: the equivalent of a CAD DRC.
//
// Each rule is independent and returns findings with a severity and, where
// possible, a concrete suggestion. An error means the design cannot be built as
// drawn; a warning means it will build but something is likely wrong or
// wasteful; info is advisory.
package rules

import (
	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
)

// Finding is one thing a rule has to say.
type Finding struct {
	// Rule is the stable identifier, e.g. "flash-head-dim".
	Rule     string `json:"rule"`
	Severity string `json:"severity"`
	// Path is the node the finding is about, when it is about one node.
	Path string `json:"path,omitempty"`
	Port string `json:"port,omitempty"`
	// Param is the parameter that caused it, so the inspector can highlight the
	// field rather than the block.
	Param   string `json:"param,omitempty"`
	Message string `json:"message"`
	// Hint is what to do about it.
	Hint string `json:"hint,omitempty"`
}

// Ctx is everything a rule may look at. It is computed once and shared, so no
// rule pays to re-derive the analysis.
type Ctx struct {
	Doc      *ir.Doc
	Symbols  *ir.SymbolTable
	Infer    *infer.Result
	Flat     *analysis.FlatResult
	Analysis *analysis.Result
}

// Rule is one check.
type Rule struct {
	ID    string
	Title string
	// Description is one line on what the rule protects against, for the docs
	// panel.
	Description string
	Run         func(ctx *Ctx) []Finding
}
