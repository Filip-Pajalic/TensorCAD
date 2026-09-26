package catalog_test

import (
	"encoding/json"
	"sort"
	"strings"
	"testing"

	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/ir"
)

// Every parameter of every built-in block is labelled, and the labels are
// ones a person can tell apart.
//
// A block added without a label would show its bare identifier in the
// inspector beside nothing, which is the thing the labels exist to stop, and
// nothing else would notice.
func TestEveryBuiltinParameterHasALabel(t *testing.T) {
	var types []string
	for name := range catalog.Builtin {
		types = append(types, name)
	}
	sort.Strings(types)

	for _, name := range types {
		def := catalog.Builtin[name]
		seen := map[string]string{}
		for _, p := range def.Params {
			spec := p.Spec
			if spec.Label == "" {
				t.Errorf("%s.%s has no label", name, p.Name)
				continue
			}
			if spec.Label != strings.TrimSpace(spec.Label) || strings.ToUpper(spec.Label[:1]) != spec.Label[:1] {
				t.Errorf("%s.%s: label %q should be trimmed and start with a capital", name, p.Name, spec.Label)
			}
			// Two fields on one block that read the same are two fields nobody
			// can tell apart without reading the code names.
			if other, taken := seen[spec.Label]; taken {
				t.Errorf("%s: %s and %s are both labelled %q", name, other, p.Name, spec.Label)
			}
			seen[spec.Label] = p.Name

			if spec.Advanced && !spec.HasDefault {
				t.Errorf("%s.%s is advanced but required: a field a block needs cannot be hidden", name, p.Name)
			}
			for value := range spec.ValueLabels {
				if !contains(spec.Values, value) {
					t.Errorf("%s.%s labels %q, which is not one of its values", name, p.Name, value)
				}
			}
		}
	}
}

// The ones worth hiding are hidden, and the ones a design is made of are not.
func TestAdvancedIsTheRareOnes(t *testing.T) {
	block, _ := catalog.Builtin.Get("transformer_block")
	for _, name := range []string{"sinks", "mask", "score", "written_out", "logit_softcap", "norm_eps"} {
		if spec, _ := block.Params.Get(name); !spec.Advanced {
			t.Errorf("transformer_block.%s should be advanced", name)
		}
	}
	for _, name := range []string{"d_model", "heads", "kv_heads", "head_dim", "ffn_hidden", "attention", "mlp", "norm"} {
		if spec, _ := block.Params.Get(name); spec.Advanced {
			t.Errorf("transformer_block.%s should not be advanced", name)
		}
	}
	norm, _ := catalog.Builtin.Get("rmsnorm")
	if spec, _ := norm.Params.Get("eps"); spec.Advanced {
		t.Error("an RMSNorm's epsilon is one of the two things it has to say")
	}
	attention, _ := block.Params.Get("attention")
	if attention.ValueLabels["mla"] == "" {
		t.Error("the attention kinds should read as words")
	}
}

// A design's own block says what its parameters are called the same way, and
// cannot hide one it needs.
func TestAUserBlockCarriesItsLabels(t *testing.T) {
	var def map[string]any
	if err := json.Unmarshal([]byte(`{
		"category": "mlp",
		"params": {
			"width": { "type": "int", "min": 1, "label": "Hidden width" },
			"gain": { "type": "num", "default": 1, "label": "Gain", "advanced": true },
			"needed": { "type": "int", "advanced": true }
		},
		"ports": { "in": { "x": "B T width" }, "out": { "y": "B T width" } },
		"graph": {
			"nodes": [{ "id": "n", "type": "rmsnorm", "params": { "dim": "$width" } }],
			"edges": [["_in:x", "n:x"], ["n:y", "_out:y"]]
		}
	}`), &def); err != nil {
		t.Fatal(err)
	}
	cat := catalog.Of(&ir.Doc{Defs: map[string]any{"mine": def}})
	block, err := cat.Get("mine")
	if err != nil {
		t.Fatal(err)
	}
	width, _ := block.Params.Get("width")
	gain, _ := block.Params.Get("gain")
	needed, _ := block.Params.Get("needed")
	if width.Label != "Hidden width" || gain.Label != "Gain" || needed.Label != "" {
		t.Errorf("labels: %q %q %q", width.Label, gain.Label, needed.Label)
	}
	if !gain.Advanced || needed.Advanced {
		t.Errorf("advanced: gain %v, needed %v; one with no default cannot be hidden", gain.Advanced, needed.Advanced)
	}
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}
