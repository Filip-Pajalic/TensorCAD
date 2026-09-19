package jsonx_test

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/tensorcad/core/jsonx"
)

type inner struct {
	Depth float64 `json:"depth"`
}

type sample struct {
	Name    string             `json:"name"`
	Total   float64            `json:"total"`
	Values  map[string]float64 `json:"values"`
	Notes   []string           `json:"notes"`
	Nested  inner              `json:"nested"`
	Pointer *inner             `json:"pointer"`
	Omitted string             `json:"omitted,omitempty"`
	Hidden  string             `json:"-"`
	Rows    []inner            `json:"rows"`
}

// The fast path has to be encoding/json exactly, because almost every call
// takes it and a difference there would be a difference on every report.
func TestFiniteValuesGoThroughUnchanged(t *testing.T) {
	v := sample{
		Name: "gpt2-small", Total: 124439808,
		Values: map[string]float64{"D": 768, "L": 12},
		Notes:  []string{"one", "two"},
		Nested: inner{Depth: 0.5},
		Rows:   []inner{{Depth: 1}, {Depth: 2}},
		Hidden: "not written",
	}
	got, err := jsonx.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	want, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Errorf("\n got  %s\n want %s", got, want)
	}
}

// And the slow path has to be JavaScript exactly: JSON.stringify writes NaN and
// Infinity as null, and a report full of them is still a report the editor can
// show an error beside.
func TestNonFiniteValuesBecomeNull(t *testing.T) {
	v := sample{
		Name:  "broken",
		Total: math.NaN(),
		Values: map[string]float64{
			"fine":     4096,
			"nan":      math.NaN(),
			"positive": math.Inf(1),
			"negative": math.Inf(-1),
		},
		Notes:   []string{"the symbol did not evaluate"},
		Nested:  inner{Depth: math.NaN()},
		Pointer: &inner{Depth: math.Inf(1)},
		Rows:    []inner{{Depth: math.NaN()}, {Depth: 3}},
		Hidden:  "not written",
	}
	raw, err := jsonx.Marshal(v)
	if err != nil {
		t.Fatalf("a report with a NaN in it could not be sent: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("what came out is not JSON: %v", err)
	}

	if got["total"] != nil {
		t.Errorf("total: got %v, want null", got["total"])
	}
	if got["name"] != "broken" {
		t.Errorf("name: got %v", got["name"])
	}
	values, _ := got["values"].(map[string]any)
	if values["fine"] != 4096.0 {
		t.Errorf("a finite value beside a broken one: got %v", values["fine"])
	}
	for _, key := range []string{"nan", "positive", "negative"} {
		if values[key] != nil {
			t.Errorf("values[%s]: got %v, want null", key, values[key])
		}
	}
	nested, _ := got["nested"].(map[string]any)
	if nested["depth"] != nil {
		t.Errorf("nested.depth: got %v, want null", nested["depth"])
	}
	pointer, _ := got["pointer"].(map[string]any)
	if pointer["depth"] != nil {
		t.Errorf("pointer.depth: got %v, want null", pointer["depth"])
	}
	rows, _ := got["rows"].([]any)
	if len(rows) != 2 {
		t.Fatalf("rows: got %d", len(rows))
	}
	if second, _ := rows[1].(map[string]any); second["depth"] != 3.0 {
		t.Errorf("the finite row was lost: %v", rows[1])
	}

	// The tags still have to be honoured on the slow path, or the editor gets
	// a report with Go's field names on it.
	if _, ok := got["Hidden"]; ok {
		t.Error("a field tagged \"-\" was written")
	}
	if _, ok := got["omitted"]; ok {
		t.Error("an empty omitempty field was written")
	}
	if _, ok := got["Name"]; ok {
		t.Error("a field was written under its Go name")
	}
}

func TestNilSliceAndMapStayNull(t *testing.T) {
	// null is what a missing list means on the wire, and the client turns it
	// into an empty one. What matters is that the slow path does not invent a
	// different answer from the fast one.
	v := sample{Total: math.NaN()}
	raw, err := jsonx.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"values", "notes", "rows", "pointer"} {
		if got[key] != nil {
			t.Errorf("%s: got %v, want null", key, got[key])
		}
	}
}

func TestAnUnmarshallableValueIsStillAnError(t *testing.T) {
	// A channel is not JSON in either engine, and pretending otherwise would
	// hide a real mistake.
	if _, err := jsonx.Marshal(map[string]any{"c": make(chan int)}); err == nil {
		t.Fatal("a channel was accepted")
	}
}
