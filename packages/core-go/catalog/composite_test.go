package catalog_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/ir"
)

// Composite expansions against the goldens, node for node.
//
// A composite is nothing but the subgraph it stands for, so that subgraph is
// the specification. Comparing parameter totals would let a wrong expansion
// pass whenever two wrong numbers happened to cancel; comparing the graph
// cannot. Regenerate with `go run ./cmd/golden`.

type goldenNode struct {
	ID     string         `json:"id"`
	Type   string         `json:"type"`
	Params map[string]any `json:"params"`
	Graph  *struct {
		Nodes []goldenNode `json:"nodes"`
		Edges [][2]string  `json:"edges"`
	} `json:"graph"`
}

type compositeCase struct {
	Type        string         `json:"type"`
	Params      map[string]any `json:"params"`
	Nodes       []goldenNode   `json:"nodes"`
	Edges       [][2]string    `json:"edges"`
	Constraints []string       `json:"constraints"`
}

func loadComposites(t *testing.T) []compositeCase {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "composites.json"))
	if err != nil {
		t.Fatalf("read composite golden: %v", err)
	}
	var g struct {
		Cases []compositeCase `json:"cases"`
	}
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("parse composite golden: %v", err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("composite golden is empty")
	}
	return g.Cases
}

func TestCompositeExpansionsMatchTheGoldens(t *testing.T) {
	cases := loadComposites(t)
	symbols := primSymbols()

	for i, c := range cases {
		t.Run(fmt.Sprintf("%s#%d", c.Type, i), func(t *testing.T) {
			def, err := catalog.Builtin.Get(c.Type)
			if err != nil {
				t.Fatal(err)
			}
			r := catalog.ResolveNodeParams(def, c.Params, symbols)
			exp, ok := catalog.Expand(def, c.Params, r)
			if !ok {
				t.Fatalf("%s did not expand", c.Type)
			}

			compareNodes(t, "", exp.Nodes, c.Nodes)
			compareEdges(t, "", exp.Edges, c.Edges)

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

// TestEveryCompositeIsCovered stops a composite passing by not being looked at.
func TestEveryCompositeIsCovered(t *testing.T) {
	covered := map[string]bool{}
	for _, c := range loadComposites(t) {
		covered[c.Type] = true
	}
	for _, d := range catalog.Composites {
		if !covered[d.Type] {
			t.Errorf("composite %q has no golden case", d.Type)
		}
	}
}

func compareNodes(t *testing.T, prefix string, got []ir.NodeDef, want []goldenNode) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%snode count: got %d, want %d (%s vs %s)",
			prefix, len(got), len(want), nodeIDs(got), goldenIDs(want))
		return
	}
	for i := range got {
		g, w := got[i], want[i]
		at := fmt.Sprintf("%s[%d] %s", prefix, i, w.ID)
		if g.ID != w.ID || g.Type != w.Type {
			t.Errorf("%s: got %s/%s, want %s/%s", at, g.ID, g.Type, w.ID, w.Type)
			continue
		}
		compareNodeParams(t, at, g.Params, w.Params)
		switch {
		case w.Graph == nil && g.Graph != nil:
			t.Errorf("%s: has a subgraph, want none", at)
		case w.Graph != nil && g.Graph == nil:
			t.Errorf("%s: has no subgraph, want one", at)
		case w.Graph != nil && g.Graph != nil:
			compareNodes(t, at+"/", g.Graph.Nodes, w.Graph.Nodes)
			compareEdges(t, at+"/", g.Graph.Edges, w.Graph.Edges)
		}
	}
}

// compareNodeParams compares the parameters a node was given.
//
// The interpolated expressions are the point: "H*dh" landing as "(H*dh)" or as
// "H*dh" is the difference between a readable design and line noise, and the
// two engines have to agree on which.
func compareNodeParams(t *testing.T, at string, got, want map[string]any) {
	t.Helper()
	for k, w := range want {
		g, ok := got[k]
		if !ok {
			t.Errorf("%s: parameter %q missing, want %v", at, k, w)
			continue
		}
		if !sameJSON(g, w) {
			t.Errorf("%s: parameter %q = %#v, want %#v", at, k, g, w)
		}
	}
	for k := range got {
		if _, ok := want[k]; !ok {
			t.Errorf("%s: parameter %q is not in the golden", at, k)
		}
	}
}

func compareEdges(t *testing.T, prefix string, got []ir.Edge, want [][2]string) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%sedge count: got %d, want %d", prefix, len(got), len(want))
		return
	}
	for i := range got {
		if got[i][0] != want[i][0] || got[i][1] != want[i][1] {
			t.Errorf("%sedge %d: got %v, want %v", prefix, i, got[i], want[i])
		}
	}
}

// sameJSON compares two values as the wire would see them, so an int and a
// float that print the same are the same.
func sameJSON(a, b any) bool {
	x, err1 := json.Marshal(a)
	y, err2 := json.Marshal(b)
	if err1 != nil || err2 != nil {
		return false
	}
	return string(x) == string(y)
}

func nodeIDs(nodes []ir.NodeDef) string {
	out := make([]string, len(nodes))
	for i, n := range nodes {
		out[i] = n.ID
	}
	b, _ := json.Marshal(out)
	return string(b)
}

func goldenIDs(nodes []goldenNode) string {
	out := make([]string, len(nodes))
	for i, n := range nodes {
		out[i] = n.ID
	}
	b, _ := json.Marshal(out)
	return string(b)
}
