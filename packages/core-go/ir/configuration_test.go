package ir_test

import (
	"encoding/json"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
)

// One design at four sizes.
//
// The library carries `gpt2-small`, `gpt2-medium`, `gpt2-large` and `gpt2-xl`
// as four files that differ in four numbers. This is the same architecture with
// those numbers as configurations, and it has to reproduce all four exactly —
// which is the whole claim: a configuration is a size, not an approximation of
// one.

func sized(width, layers, heads float64) ir.Configuration {
	return ir.Configuration{Symbols: map[string]ir.SymbolDef{
		"D": {Kind: "design", Number: width, HasNumber: true},
		"L": {Kind: "design", Number: layers, HasNumber: true},
		"H": {Kind: "design", Number: heads, HasNumber: true},
		// GPT-2 has no grouped-query attention, so these move together.
		"Hkv": {Kind: "design", Number: heads, HasNumber: true},
	}}
}

func TestOneDesignAtFourSizes(t *testing.T) {
	// The families, from the presets themselves rather than from memory.
	want := map[string]float64{}
	for _, name := range []string{"gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"} {
		res, err := analysis.Analyze(presets.MustGet(name), analysis.Options{}, analysis.Inputs{})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		want[name] = res.Params.Total
	}

	doc := presets.MustGet("gpt2-small")
	doc.Configurations = map[string]ir.Configuration{
		"gpt2-small":  sized(768, 12, 12),
		"gpt2-medium": sized(1024, 24, 16),
		"gpt2-large":  sized(1280, 36, 20),
		"gpt2-xl":     sized(1600, 48, 25),
	}
	doc.ConfigurationOrder = []string{"gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"}

	for name, expected := range want {
		doc.Active = name
		res, err := analysis.Analyze(doc, analysis.Options{}, analysis.Inputs{})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got := res.Params.Total; got != expected {
			t.Errorf("configuration %q counts %v parameters, want the %v the preset does",
				name, got, expected)
		}
	}
}

// Switching is something you can do twice and end up where you started: a
// configuration is a view of the design, not an edit to it.
func TestSwitchingAConfigurationLeavesTheDesignAlone(t *testing.T) {
	doc := presets.MustGet("gpt2-small")
	before := doc.Symbols["D"].Number
	doc.Configurations = map[string]ir.Configuration{"wide": sized(1600, 48, 25)}

	doc.Active = "wide"
	wide := ir.ResolveSymbols(doc).DesignValues["D"]
	doc.Active = ""
	back := ir.ResolveSymbols(doc).DesignValues["D"]

	if wide != 1600 {
		t.Errorf("the wide configuration resolved D to %v, want 1600", wide)
	}
	if back != before {
		t.Errorf("going back gave D = %v, want the design's own %v", back, before)
	}
	if got := doc.Symbols["D"].Number; got != before {
		t.Errorf("the document's own D is now %v; a configuration edited the design", got)
	}
}

// A configuration says only what is different about it.
func TestAConfigurationOverridesOnlyWhatItNames(t *testing.T) {
	doc := presets.MustGet("gpt2-small")
	doc.Configurations = map[string]ir.Configuration{
		"narrow": {Symbols: map[string]ir.SymbolDef{
			"D": {Kind: "design", Number: 256, HasNumber: true},
		}},
	}
	doc.Active = "narrow"
	table := ir.ResolveSymbols(doc)
	if got := table.DesignValues["D"]; got != 256 {
		t.Errorf("D is %v, want 256", got)
	}
	// Untouched, and still 12 rather than anything the override implied.
	if got := table.DesignValues["L"]; got != 12 {
		t.Errorf("L is %v; the configuration did not mention it", got)
	}
	// And an expression over D follows it, which is the point of leaving the
	// rest alone rather than freezing every number.
	if got := table.DesignValues["F"]; got != 1024 {
		t.Errorf("F is %v, want 4*256; an expression should follow the override", got)
	}
}

// An unknown name is the design as written, not an error and not empty: a
// document that named a configuration somebody deleted still has to open.
func TestAnUnknownConfigurationIsTheDesignItself(t *testing.T) {
	doc := presets.MustGet("gpt2-small")
	doc.Active = "no-such-configuration"
	if got := ir.ResolveSymbols(doc).DesignValues["D"]; got != 768 {
		t.Errorf("D is %v, want the design's own 768", got)
	}
}

// They survive the round trip, in the order they were written.
func TestConfigurationsRoundTrip(t *testing.T) {
	doc := presets.MustGet("gpt2-small")
	doc.Configurations = map[string]ir.Configuration{
		"small": sized(768, 12, 12),
		"xl":    sized(1600, 48, 25),
	}
	doc.ConfigurationOrder = []string{"small", "xl"}
	doc.Active = "xl"

	encoded, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	var back ir.Doc
	if err := json.Unmarshal(encoded, &back); err != nil {
		t.Fatal(err)
	}
	if back.Active != "xl" {
		t.Errorf("the active configuration came back as %q", back.Active)
	}
	if got := ir.ConfigurationNames(&back); len(got) != 2 || got[0] != "small" || got[1] != "xl" {
		t.Errorf("the configurations came back as %v, want them in the order they were written", got)
	}
	if got := ir.ResolveSymbols(&back).DesignValues["D"]; got != 1600 {
		t.Errorf("after the round trip D is %v, want 1600", got)
	}
}

// A design with none of them is unchanged in every way, including on the wire:
// an empty map must not appear in the JSON of a document that never had one.
func TestADesignWithoutConfigurationsIsUntouched(t *testing.T) {
	doc := presets.MustGet("llama-3-8b")
	encoded, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"configurations", "active"} {
		if got := string(encoded); containsKey(got, key) {
			t.Errorf("a design with no configurations wrote %q into its JSON", key)
		}
	}
	if len(ir.ConfigurationNames(doc)) != 0 {
		t.Error("it reports configurations it does not have")
	}
}

func containsKey(s, key string) bool {
	needle := `"` + key + `":`
	for i := 0; i+len(needle) <= len(s); i++ {
		if s[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
