package rules

import (
	"sort"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
)

// Report is what a design-rule check produces.
type Report struct {
	Name     string         `json:"name"`
	Findings []Finding      `json:"findings"`
	Counts   map[string]int `json:"counts"`
	// OK is true when nothing blocks building this design.
	OK bool `json:"ok"`
	// Analysis is the work the rules were run against, returned so a caller
	// that wants both numbers and findings pays for it once.
	Analysis *analysis.Result `json:"analysis"`
}

var severityOrder = map[string]int{"error": 0, "warning": 1, "info": 2}

// Validate runs every rule over a design and returns the findings sorted by
// severity.
func Validate(doc *ir.Doc, options analysis.Options) (*Report, error) {
	symbols := ir.ResolveSymbols(doc)
	shapeInfo := infer.Shapes(doc, symbols, infer.Options{})
	flat := analysis.Flatten(doc, symbols)
	result, err := analysis.Analyze(doc, options, analysis.Inputs{
		Symbols: symbols, Infer: shapeInfo, Flat: flat,
	})
	if err != nil {
		return nil, err
	}

	ctx := &Ctx{Doc: doc, Symbols: symbols, Infer: shapeInfo, Flat: flat, Analysis: result}
	findings := []Finding{}
	for _, rule := range Rules {
		findings = append(findings, rule.Run(ctx)...)
	}
	for _, message := range flat.Errors {
		findings = append(findings, Finding{Rule: "graph", Severity: "error", Message: message})
	}

	// By severity, then node, then pin; stable, so findings that tie keep the
	// order the rules produced them in.
	//
	// It goes down to the pin rather than stopping at the node because this
	// engine has no port order to inherit: a block's pins live in a map, so two
	// findings on the same block would otherwise be left to tie and come out in
	// whichever order the map handed them over.
	sort.SliceStable(findings, func(i, j int) bool {
		a, b := &findings[i], &findings[j]
		if s := severityOrder[a.Severity] - severityOrder[b.Severity]; s != 0 {
			return s < 0
		}
		if a.Path != b.Path {
			return a.Path < b.Path
		}
		return a.Port < b.Port
	})

	counts := map[string]int{"error": 0, "warning": 0, "info": 0}
	for _, f := range findings {
		counts[f.Severity]++
	}

	return &Report{
		Name:     doc.Meta.Name,
		Findings: findings,
		Counts:   counts,
		OK:       counts["error"] == 0,
		Analysis: result,
	}, nil
}
