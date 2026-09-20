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
	// Overridden is what the document's own severities did, so suppression is
	// never silent: a design cannot drop a finding without the report saying so.
	Overridden []Override `json:"overridden"`
}

// Override is one finding whose severity the document changed.
type Override struct {
	Rule string `json:"rule"`
	Path string `json:"path,omitempty"`
	// From is the severity the rule produced, To what the document asked for.
	// "off" means the finding was dropped.
	From string `json:"from"`
	To   string `json:"to"`
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

	// What the document has decided these rules mean to it, before sorting:
	// a downgraded finding sorts where its new severity puts it.
	overridden := []Override{}
	if len(doc.Rules) > 0 {
		kept := findings[:0]
		for _, f := range findings {
			want, ok := doc.Rules[f.Rule]
			if !ok || want == f.Severity {
				kept = append(kept, f)
				continue
			}
			if !validSeverity(want) {
				// An unreadable override is not an override. Say so rather than
				// guessing, and leave the finding as the rule produced it.
				kept = append(kept, f)
				overridden = append(overridden, Override{
					Rule: f.Rule, Path: f.Path, From: f.Severity, To: "?" + want})
				continue
			}
			overridden = append(overridden, Override{
				Rule: f.Rule, Path: f.Path, From: f.Severity, To: want})
			if want == "off" {
				continue
			}
			f.Severity = want
			kept = append(kept, f)
		}
		findings = kept
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
		Name:       doc.Meta.Name,
		Findings:   findings,
		Counts:     counts,
		OK:         counts["error"] == 0,
		Analysis:   result,
		Overridden: overridden,
	}, nil
}

// validSeverity is what a document may ask a rule to be.
func validSeverity(s string) bool {
	switch s {
	case "error", "warning", "info", "off":
		return true
	}
	return false
}
