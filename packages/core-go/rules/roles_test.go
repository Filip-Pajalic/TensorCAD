package rules_test

import (
	"strings"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
)

// llama-3-8b with every layer's attention keeping documents apart, reading
// them from `from` through the stack.
func keepsDocumentsApart(t *testing.T, role string) *ir.Doc {
	t.Helper()
	doc := presets.MustGet("llama-3-8b")
	doc.Graph.Nodes = append(doc.Graph.Nodes, ir.NodeDef{ID: "docs", Type: "input",
		Params: map[string]any{"shape": "B T", "dtype": "int64", "role": role}})
	doc.Graph.Edges = append(doc.Graph.Edges, ir.Edge{"docs:x", "layers:doc"})
	for i := range doc.Graph.Nodes {
		stack := &doc.Graph.Nodes[i]
		if stack.ID != "layers" {
			continue
		}
		for j := range stack.Graph.Nodes {
			n := &stack.Graph.Nodes[j]
			switch n.ID {
			case "_in":
				n.Params["ports"].(map[string]any)["doc"] = "B T"
			case "block":
				n.Params["mask"] = "doc(b, q) == doc(b, kv)"
			}
		}
		stack.Graph.Edges = append(stack.Graph.Edges, ir.Edge{"_in:doc", "block:doc"})
	}
	return doc
}

func documentFindings(t *testing.T, doc *ir.Doc) []rules.Finding {
	t.Helper()
	rep, err := rules.Validate(doc, analysis.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var out []rules.Finding
	for _, f := range rep.Findings {
		if f.Rule == "input-roles" {
			out = append(out, f)
		}
	}
	return out
}

// The documents, followed back through the stack to an input that says it
// holds them, pass; the same wire from an input holding tokens does not.
func TestAMaskIsGivenTheDocuments(t *testing.T) {
	if found := documentFindings(t, keepsDocumentsApart(t, "documents")); len(found) > 0 {
		t.Errorf("wired right, and still: %v", found)
	}
	found := documentFindings(t, keepsDocumentsApart(t, "tokens"))
	if len(found) != 1 || found[0].Path != "layers/block" || !strings.Contains(found[0].Message, "from docs") {
		t.Errorf("tokens where the documents go: %+v", found)
	}
	// Nothing is quiet about it on any preset, none of which masks documents.
	for _, name := range []string{"llama-3-8b", "t5-small", "gpt-oss-20b"} {
		if found := documentFindings(t, presets.MustGet(name)); len(found) > 0 {
			t.Errorf("%s: %v", name, found)
		}
	}
}

// llama-3-8b with every layer's rotation turned by positions read from an
// input with this role.
func restartsPositions(t *testing.T, role string) *ir.Doc {
	t.Helper()
	doc := presets.MustGet("llama-3-8b")
	doc.Graph.Nodes = append(doc.Graph.Nodes, ir.NodeDef{ID: "pos", Type: "input",
		Params: map[string]any{"shape": "B T", "dtype": "int64", "role": role}})
	doc.Graph.Edges = append(doc.Graph.Edges, ir.Edge{"pos:x", "layers:pos"})
	for i := range doc.Graph.Nodes {
		stack := &doc.Graph.Nodes[i]
		if stack.ID != "layers" {
			continue
		}
		for j := range stack.Graph.Nodes {
			n := &stack.Graph.Nodes[j]
			switch n.ID {
			case "_in":
				n.Params["ports"].(map[string]any)["pos"] = "B T"
			case "block":
				n.Params["positions"] = true
			}
		}
		stack.Graph.Edges = append(stack.Graph.Edges, ir.Edge{"_in:pos", "block:pos"})
	}
	return doc
}

// The same for a rotation that restarts at every document: its positions,
// followed back through the stack, start at an input that says it holds them.
func TestARotationIsGivenThePositions(t *testing.T) {
	if found := documentFindings(t, restartsPositions(t, "positions")); len(found) > 0 {
		t.Errorf("wired right, and still: %v", found)
	}
	found := documentFindings(t, restartsPositions(t, "tokens"))
	if len(found) != 1 || found[0].Path != "layers/block" || found[0].Param != "positions" ||
		!strings.Contains(found[0].Message, "not a positions input") {
		t.Errorf("tokens where the positions go: %+v", found)
	}
}
