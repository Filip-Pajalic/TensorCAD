package rules_test

import (
	"encoding/json"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
)

// What a design has decided the rules mean to it.
//
// Every design-rule tool has this, and for the same reason: a rule that is right
// in general is sometimes wrong here, and the alternative to recording that is
// people learning to read past a warning. What must not happen is that the
// record is invisible — a design that can quietly drop its own errors is a
// design nobody can check.

// gemmaWith is Gemma-2-9B, which softcaps its attention scores and therefore
// carries two warnings about the fused kernel it cannot use, plus whatever the
// document says those rules mean.
func gemmaWith(t *testing.T, severities map[string]string) *rules.Report {
	t.Helper()
	doc, err := presets.Get("gemma-2-9b")
	if err != nil {
		t.Fatalf("preset: %v", err)
	}
	doc, err = doc.Clone()
	if err != nil {
		t.Fatalf("clone: %v", err)
	}
	doc.Rules = severities
	seq := 8192.0
	report, err := rules.Validate(doc, analysis.Options{T: &seq})
	if err != nil {
		t.Fatalf("validate: %v", err)
	}
	return report
}

func countOf(r *rules.Report, rule string) int {
	n := 0
	for _, f := range r.Findings {
		if f.Rule == rule {
			n++
		}
	}
	return n
}

func TestADocumentCanLowerARule(t *testing.T) {
	plain := gemmaWith(t, nil)
	if got := countOf(plain, "SDPA-03"); got != 2 {
		t.Fatalf("Gemma has %d softcap findings, expected 2", got)
	}
	if plain.Counts["warning"] < 2 {
		t.Fatalf("they are not warnings: %v", plain.Counts)
	}
	if len(plain.Overridden) != 0 {
		t.Errorf("nothing was overridden, but the report says %+v", plain.Overridden)
	}

	lowered := gemmaWith(t, map[string]string{"SDPA-03": "info"})
	if got := countOf(lowered, "SDPA-03"); got != 2 {
		t.Errorf("lowering a rule dropped %d of its findings", 2-got)
	}
	for _, f := range lowered.Findings {
		if f.Rule == "SDPA-03" && f.Severity != "info" {
			t.Errorf("a lowered finding is still %q", f.Severity)
		}
	}
	if lowered.Counts["warning"] != plain.Counts["warning"]-2 {
		t.Errorf("warnings went from %d to %d, expected %d",
			plain.Counts["warning"], lowered.Counts["warning"], plain.Counts["warning"]-2)
	}
	if len(lowered.Overridden) != 2 {
		t.Errorf("%d overrides recorded, expected 2: %+v", len(lowered.Overridden), lowered.Overridden)
	}
	for _, o := range lowered.Overridden {
		if o.From != "warning" || o.To != "info" || o.Rule != "SDPA-03" {
			t.Errorf("recorded %+v", o)
		}
		if o.Path == "" {
			t.Error("an override with no path; the block it silenced is the point")
		}
	}
}

// Turning a rule off drops its findings and says so in the same breath.
func TestTurningARuleOffIsRecorded(t *testing.T) {
	off := gemmaWith(t, map[string]string{"SDPA-03": "off"})
	if got := countOf(off, "SDPA-03"); got != 0 {
		t.Errorf("%d findings survived being turned off", got)
	}
	if len(off.Overridden) != 2 {
		t.Fatalf("%d overrides recorded, expected 2", len(off.Overridden))
	}
	for _, o := range off.Overridden {
		if o.To != "off" {
			t.Errorf("recorded %+v", o)
		}
	}
}

// A design may also decide a rule matters more than the rule thinks it does.
func TestADocumentCanRaiseARule(t *testing.T) {
	raised := gemmaWith(t, map[string]string{"SDPA-03": "error"})
	if raised.OK {
		t.Error("the design reports ok with two errors in it")
	}
	if raised.Counts["error"] != 2 {
		t.Errorf("errors are %v", raised.Counts)
	}
	// And the findings sort where their new severity puts them, not where the
	// rule's did.
	if len(raised.Findings) == 0 || raised.Findings[0].Severity != "error" {
		t.Errorf("the first finding is %+v", raised.Findings[0])
	}
}

// An override the engine cannot read is not an override. It says so and leaves
// the finding alone, rather than guessing at what was meant.
func TestAnUnreadableSeverityIsRefusedLoudly(t *testing.T) {
	odd := gemmaWith(t, map[string]string{"SDPA-03": "whatever"})
	if got := countOf(odd, "SDPA-03"); got != 2 {
		t.Errorf("%d findings survived an unreadable override, expected 2", got)
	}
	for _, f := range odd.Findings {
		if f.Rule == "SDPA-03" && f.Severity != "warning" {
			t.Errorf("the finding became %q", f.Severity)
		}
	}
	if len(odd.Overridden) != 2 {
		t.Fatalf("%d recorded, expected 2", len(odd.Overridden))
	}
	if odd.Overridden[0].To != "?whatever" {
		t.Errorf("recorded %+v; the unreadable value should be visible", odd.Overridden[0])
	}
}

// Naming a rule that did not fire changes nothing and records nothing.
func TestAnOverrideForASilentRuleIsNotRecorded(t *testing.T) {
	quiet := gemmaWith(t, map[string]string{"no-such-rule": "off", "flash-head-dim": "off"})
	plain := gemmaWith(t, nil)
	if len(quiet.Findings) != len(plain.Findings) {
		t.Errorf("%d findings against %d", len(quiet.Findings), len(plain.Findings))
	}
	if len(quiet.Overridden) != 0 {
		t.Errorf("recorded %+v for rules that never fired", quiet.Overridden)
	}
}

// The severities travel with the design, because they are a decision about it.
func TestSeveritiesSurviveASaveAndReopen(t *testing.T) {
	doc, err := presets.Get("gemma-2-9b")
	if err != nil {
		t.Fatalf("preset: %v", err)
	}
	doc.Rules = map[string]string{"SDPA-03": "off"}
	again, err := doc.Clone()
	if err != nil {
		t.Fatalf("clone: %v", err)
	}
	if got := again.Rules["SDPA-03"]; got != "off" {
		t.Errorf("after a round trip the rule is %q", got)
	}
	// And a document that says nothing carries nothing, rather than an empty
	// object in every file.
	plain, err := presets.Get("gpt2-small")
	if err != nil {
		t.Fatalf("preset: %v", err)
	}
	if plain.Rules != nil {
		t.Errorf("a design that decided nothing carries %v", plain.Rules)
	}
	raw, err := json.Marshal(plain)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if containsRules(string(raw)) {
		t.Error("a design that decided nothing wrote a rules object anyway")
	}
}

func containsRules(s string) bool {
	for i := 0; i+8 <= len(s); i++ {
		if s[i:i+8] == `"rules":` {
			return true
		}
	}
	return false
}
