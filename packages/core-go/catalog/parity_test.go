package catalog_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/ir"
)

// Every primitive, one block at a time, against the goldens.
//
// A preset only exercises the primitives its architecture happens to use, at
// the sizes that architecture happens to pick. This walks the catalog at a
// fixed set of parameters and requires the same answer for each: resolved
// parameters, ports, parameter count, FLOPs, retained inputs, cache state and
// constraint messages. Regenerate with `go run ./cmd/golden`.

type primPorts struct {
	In      map[string]string `json:"in"`
	Out     map[string]string `json:"out"`
	Anchors map[string]string `json:"anchors"`
}

type primCase struct {
	Type                 string             `json:"type"`
	Params               map[string]any     `json:"params"`
	Resolved             map[string]any     `json:"resolved"`
	Errors               []string           `json:"errors"`
	Ports                *primPorts         `json:"ports"`
	PortError            string             `json:"portError"`
	ParamCount           *float64           `json:"paramCount"`
	Flops                map[string]float64 `json:"flops"`
	Retains              []string           `json:"retains"`
	ExtraActivationBytes *float64           `json:"extraActivationBytes"`
	StateBytes           map[string]float64 `json:"stateBytes"`
	Constraints          []string           `json:"constraints"`
}

type primGolden struct {
	Ctx struct {
		T     float64 `json:"T"`
		B     float64 `json:"B"`
		Bytes float64 `json:"bytes"`
		Flash bool    `json:"flash"`
	} `json:"ctx"`
	Cases []primCase `json:"cases"`
}

func loadPrimitives(t *testing.T) primGolden {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "primitives.json"))
	if err != nil {
		t.Fatalf("read primitive golden: %v", err)
	}
	var g primGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("parse primitive golden: %v", err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("primitive golden is empty")
	}
	return g
}

// The symbol environment the golden file was generated under.
func primSymbols() *ir.SymbolTable {
	doc := &ir.Doc{
		Version: ir.DocVersion,
		Symbols: map[string]ir.SymbolDef{
			"B":   {Kind: "literal", Number: 4, HasNumber: true},
			"T":   {Kind: "literal", Number: 2048, HasNumber: true},
			"D":   {Kind: "literal", Number: 4096, HasNumber: true},
			"H":   {Kind: "literal", Number: 32, HasNumber: true},
			"Hkv": {Kind: "literal", Number: 8, HasNumber: true},
			"dh":  {Kind: "literal", Number: 128, HasNumber: true},
			"F":   {Kind: "literal", Number: 14336, HasNumber: true},
			"V":   {Kind: "literal", Number: 128256, HasNumber: true},
			"L":   {Kind: "literal", Number: 32, HasNumber: true},
			"Tc":  {Kind: "literal", Number: 218, HasNumber: true},
			"Dp":  {Kind: "literal", Number: 384, HasNumber: true},
		},
	}
	return ir.ResolveSymbols(doc)
}

func TestPrimitivesMatchTheGoldens(t *testing.T) {
	g := loadPrimitives(t)
	symbols := primSymbols()
	ctx := catalog.AnalysisCtx{T: g.Ctx.T, B: g.Ctx.B, Bytes: g.Ctx.Bytes, Flash: g.Ctx.Flash}

	byType := map[string]*catalog.BlockDef{}
	for _, d := range catalog.Primitives {
		byType[d.Type] = d
	}

	for i, c := range g.Cases {
		t.Run(c.Type, func(t *testing.T) {
			def, ok := byType[c.Type]
			if !ok {
				t.Fatalf("case %d: %q is not in the Go catalog", i, c.Type)
			}
			r := catalog.ResolveNodeParams(def, c.Params, symbols)

			if !sameStringSets(r.Errors, c.Errors) {
				t.Errorf("resolve errors:\n got  %v\n want %v", r.Errors, c.Errors)
			}
			for k, want := range c.Resolved {
				compareParam(t, k, r.P[k], want)
			}

			if c.Ports != nil {
				ports := catalog.PortsOf(def, r)
				compareShapes(t, "in", ports.In, c.Ports.In)
				compareShapes(t, "out", ports.Out, c.Ports.Out)
			}

			if c.ParamCount != nil {
				if def.ParamCount == nil {
					t.Fatalf("no ParamCount, want %g", *c.ParamCount)
				}
				if got := def.ParamCount(r); got != *c.ParamCount {
					t.Errorf("paramCount: got %g, want %g", got, *c.ParamCount)
				}
			}

			if c.Flops != nil && def.Flops != nil {
				f := def.Flops(r, ctx)
				checkFlop(t, "fwd", f.Fwd, c.Flops["fwd"])
				checkFlop(t, "elementwise", f.Elementwise, c.Flops["elementwise"])
				checkFlop(t, "fwdSeq", f.FwdSeq, c.Flops["fwdSeq"])
				checkFlop(t, "fwdSeqUnmasked", f.FwdSeqUnmasked, c.Flops["fwdSeqUnmasked"])
			}

			if c.Retains != nil && def.Retains != nil {
				if !sameStringSets(def.Retains(r), c.Retains) {
					t.Errorf("retains: got %v, want %v", def.Retains(r), c.Retains)
				}
			}

			// A block that keeps something beyond its edges has to keep the same
			// thing in both engines, and a block that keeps nothing has to keep
			// nothing: an unimplemented formula would otherwise read as free.
			if (c.ExtraActivationBytes != nil) != (def.ExtraActivationBytes != nil) {
				t.Errorf("extraActivationBytes: got %v, the golden has %v",
					def.ExtraActivationBytes != nil, c.ExtraActivationBytes != nil)
			} else if c.ExtraActivationBytes != nil {
				checkFlop(t, "extraActivationBytes", def.ExtraActivationBytes(r, ctx), *c.ExtraActivationBytes)
			}

			if c.StateBytes != nil && def.StateBytes != nil {
				s := def.StateBytes(r, ctx)
				checkFlop(t, "perToken", s.PerToken, c.StateBytes["perToken"])
				// The TypeScript calls it perSeq.
				checkFlop(t, "perSeq", s.PerSequence, c.StateBytes["perSeq"])
			}

			var got []string
			if def.Constraints != nil {
				for _, f := range def.Constraints(r) {
					got = append(got, f.ID+": "+f.Message)
				}
			}
			if !sameStringSets(got, c.Constraints) {
				t.Errorf("constraints:\n got  %v\n want %v", got, c.Constraints)
			}
		})
	}
}

// TestEveryPrimitiveIsCovered keeps the golden file honest: a primitive added
// to Go without a case would otherwise pass by not being looked at.
func TestEveryPrimitiveIsCovered(t *testing.T) {
	g := loadPrimitives(t)
	covered := map[string]bool{}
	for _, c := range g.Cases {
		covered[c.Type] = true
	}
	for _, d := range catalog.Primitives {
		if !covered[d.Type] {
			t.Errorf("primitive %q has no golden case", d.Type)
		}
	}
}

func compareParam(t *testing.T, key string, got, want any) {
	t.Helper()
	switch w := want.(type) {
	case float64:
		g, ok := toFloat(got)
		if !ok || g != w {
			t.Errorf("resolved %s: got %v, want %g", key, got, w)
		}
	case string:
		if s, ok := got.(string); !ok || s != w {
			t.Errorf("resolved %s: got %v, want %q", key, got, w)
		}
	case bool:
		if b, ok := got.(bool); !ok || b != w {
			t.Errorf("resolved %s: got %v, want %v", key, got, w)
		}
	}
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	}
	return 0, false
}

func compareShapes(t *testing.T, side string, got map[string]catalog.PortSpec, want map[string]string) {
	t.Helper()
	for name, shape := range want {
		g, ok := got[name]
		if !ok {
			t.Errorf("%s port %q missing, want %q", side, name, shape)
			continue
		}
		if g.Shape != shape {
			t.Errorf("%s port %q: got %q, want %q", side, name, g.Shape, shape)
		}
	}
	for name := range got {
		if _, ok := want[name]; !ok {
			t.Errorf("%s port %q is not in the golden", side, name)
		}
	}
}

func checkFlop(t *testing.T, label string, got, want float64) {
	t.Helper()
	if got == want {
		return
	}
	// These are counts; a last-bit difference means a different formula.
	if math.Abs(got-want) > 1e-6*math.Max(1, math.Abs(want)) {
		t.Errorf("%s: got %g, want %g", label, got, want)
	}
}

func sameStringSets(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	x := append([]string{}, a...)
	y := append([]string{}, b...)
	sort.Strings(x)
	sort.Strings(y)
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}
