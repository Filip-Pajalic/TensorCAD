// Package diff says what changed between two designs.
//
// Structure and numbers together, because either alone is misleading: a diff
// that says `F` went from 11008 to 14336 has not told you the model grew by
// 1.3B parameters, and a diff that says it grew by 1.3B has not told you where.
//
// It lives in the engine rather than in a client because it computes numbers
// about designs, which is the line everything else in this repository is on the
// other side of. It was in the command line, where the editor could not reach
// it and the MCP server could not offer it.
package diff

import (
	"encoding/json"
	"math"
	"sort"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/ir"
)

// Change is one named value that moved.
type Change struct {
	Name string          `json:"name"`
	From json.RawMessage `json:"from"`
	To   json.RawMessage `json:"to"`
}

// Block is a block as it stands in one of the two designs.
type Block struct {
	Path   string         `json:"path"`
	Type   string         `json:"type"`
	Label  string         `json:"label,omitempty"`
	Params map[string]any `json:"params"`
}

// ParamChange is one parameter that moved on a block that exists in both.
type ParamChange struct {
	Key  string          `json:"key"`
	From json.RawMessage `json:"from"`
	To   json.RawMessage `json:"to"`
}

// BlockChange is everything that moved on one block.
type BlockChange struct {
	Path string `json:"path"`
	// Type is set when the block became a different kind of block.
	Type   *Change       `json:"type,omitempty"`
	Label  *Change       `json:"label,omitempty"`
	Params []ParamChange `json:"params"`
}

// Edge is one wire, named by the graph it sits in.
type Edge struct {
	Graph string `json:"graph"`
	From  string `json:"from"`
	To    string `json:"to"`
}

// Delta is one number that moved.
type Delta struct {
	Metric string  `json:"metric"`
	A      float64 `json:"a"`
	B      float64 `json:"b"`
	Delta  float64 `json:"delta"`
	// Ratio is nil when A is zero, because the ratio says nothing then.
	Ratio *float64 `json:"ratio"`
}

// Symbols is what happened to the symbol table.
type Symbols struct {
	Added   []Change `json:"added"`
	Removed []Change `json:"removed"`
	Changed []Change `json:"changed"`
}

// Blocks is what happened to the graph's parts.
type Blocks struct {
	Added   []Block       `json:"added"`
	Removed []Block       `json:"removed"`
	Changed []BlockChange `json:"changed"`
}

// Edges is what happened to the wiring.
type Edges struct {
	Added   []Edge `json:"added"`
	Removed []Edge `json:"removed"`
}

// At is the operating point both sides were measured under.
type At struct {
	T        float64 `json:"T"`
	B        float64 `json:"B"`
	Hardware string  `json:"hardware"`
}

// Result is the whole difference.
type Result struct {
	A       string  `json:"a"`
	B       string  `json:"b"`
	Symbols Symbols `json:"symbols"`
	Blocks  Blocks  `json:"blocks"`
	Edges   Edges   `json:"edges"`
	Metrics []Delta `json:"metrics"`
	At      At      `json:"at"`
	// Identical is true when nothing structural moved. The numbers may still
	// differ, because they are measured at an operating point.
	Identical bool `json:"identical"`
}

// Designs compares two documents.
func Designs(a, b *ir.Doc, options analysis.Options) (*Result, error) {
	out := &Result{
		A: a.Meta.Name, B: b.Meta.Name,
		Symbols: Symbols{Added: []Change{}, Removed: []Change{}, Changed: []Change{}},
		Blocks:  Blocks{Added: []Block{}, Removed: []Block{}, Changed: []BlockChange{}},
		Edges:   Edges{Added: []Edge{}, Removed: []Edge{}},
		Metrics: []Delta{},
	}

	// --- symbols ------------------------------------------------------------
	for _, name := range union(keysOf(a.Symbols), keysOf(b.Symbols)) {
		before, inA := a.Symbols[name]
		after, inB := b.Symbols[name]
		switch {
		case !inA:
			out.Symbols.Added = append(out.Symbols.Added, Change{Name: name, To: encode(after)})
		case !inB:
			out.Symbols.Removed = append(out.Symbols.Removed, Change{Name: name, From: encode(before)})
		case !sameJSON(before, after):
			out.Symbols.Changed = append(out.Symbols.Changed,
				Change{Name: name, From: encode(before), To: encode(after)})
		}
	}

	// --- blocks -------------------------------------------------------------
	aBlocks := flattenBlocks(&a.Graph, "", map[string]Block{})
	bBlocks := flattenBlocks(&b.Graph, "", map[string]Block{})
	for _, path := range union(blockPaths(aBlocks), blockPaths(bBlocks)) {
		before, inA := aBlocks[path]
		after, inB := bBlocks[path]
		switch {
		case !inA:
			out.Blocks.Added = append(out.Blocks.Added, after)
		case !inB:
			out.Blocks.Removed = append(out.Blocks.Removed, before)
		default:
			if change, moved := compareBlock(before, after); moved {
				out.Blocks.Changed = append(out.Blocks.Changed, change)
			}
		}
	}

	// --- edges --------------------------------------------------------------
	aEdges := flattenEdges(&a.Graph, "", map[string]Edge{})
	bEdges := flattenEdges(&b.Graph, "", map[string]Edge{})
	for _, k := range union(edgeKeys(aEdges), edgeKeys(bEdges)) {
		if _, ok := aEdges[k]; !ok {
			out.Edges.Added = append(out.Edges.Added, bEdges[k])
		} else if _, ok := bEdges[k]; !ok {
			out.Edges.Removed = append(out.Edges.Removed, aEdges[k])
		}
	}

	// --- numbers ------------------------------------------------------------
	// Both designs are measured at the same sequence length and batch, or the
	// attention terms and the activation memory are not comparable. Without an
	// explicit one that is the longer of the two documents' own defaults.
	shared := options
	if shared.T == nil {
		t := math.Max(runtimeDefault(a, "T", 2048), runtimeDefault(b, "T", 2048))
		shared.T = &t
	}
	if shared.B == nil {
		n := math.Max(runtimeDefault(a, "B", 1), runtimeDefault(b, "B", 1))
		shared.B = &n
	}
	ra, err := analysis.Analyze(a, shared, analysis.Inputs{})
	if err != nil {
		return nil, err
	}
	rb, err := analysis.Analyze(b, shared, analysis.Inputs{})
	if err != nil {
		return nil, err
	}
	out.Metrics = []Delta{
		delta("parameters", ra.Params.Total, rb.Params.Total),
		delta("active parameters", ra.Params.Active, rb.Params.Active),
		delta("non-embedding parameters", ra.Params.NonEmbedding, rb.Params.NonEmbedding),
		delta("training FLOPs/token", ra.Flops.TrainPerToken, rb.Flops.TrainPerToken),
		delta("forward FLOPs/token", ra.Flops.FwdTotal, rb.Flops.FwdTotal),
		delta("KV bytes/token", ra.Kv.BytesPerToken, rb.Kv.BytesPerToken),
		delta("training memory/GPU", ra.Memory.Train.PerGpu.Total, rb.Memory.Train.PerGpu.Total),
		delta("training memory total", ra.Memory.Train.Total, rb.Memory.Train.Total),
		delta("serving memory", ra.Memory.Infer.Total, rb.Memory.Infer.Total),
	}
	out.At = At{T: rb.Options.T, B: rb.Options.B, Hardware: rb.Options.Hardware.ID}

	out.Identical = len(out.Symbols.Added) == 0 && len(out.Symbols.Removed) == 0 &&
		len(out.Symbols.Changed) == 0 && len(out.Blocks.Added) == 0 &&
		len(out.Blocks.Removed) == 0 && len(out.Blocks.Changed) == 0 &&
		len(out.Edges.Added) == 0 && len(out.Edges.Removed) == 0
	return out, nil
}

// compareBlock reports what moved on a block that is in both designs.
func compareBlock(before, after Block) (BlockChange, bool) {
	change := BlockChange{Path: after.Path, Params: []ParamChange{}}
	if before.Type != after.Type {
		change.Type = &Change{Name: "type", From: encode(before.Type), To: encode(after.Type)}
	}
	if before.Label != after.Label {
		change.Label = &Change{Name: "label", From: encode(before.Label), To: encode(after.Label)}
	}
	for _, key := range union(paramKeys(before.Params), paramKeys(after.Params)) {
		if sameJSON(before.Params[key], after.Params[key]) {
			continue
		}
		change.Params = append(change.Params, ParamChange{
			Key: key, From: encode(before.Params[key]), To: encode(after.Params[key]),
		})
	}
	return change, change.Type != nil || change.Label != nil || len(change.Params) > 0
}

// flattenBlocks walks every level, including the variants a repeat declares.
func flattenBlocks(graph *ir.Graph, prefix string, out map[string]Block) map[string]Block {
	for _, node := range graph.Nodes {
		path := ir.JoinPath(prefix, node.ID)
		params := map[string]any{}
		for k, v := range node.Params {
			params[k] = v
		}
		out[path] = Block{Path: path, Type: node.Type, Label: node.Label, Params: params}
		if node.Graph != nil {
			flattenBlocks(node.Graph, path, out)
		}
	}
	return out
}

func flattenEdges(graph *ir.Graph, prefix string, out map[string]Edge) map[string]Edge {
	where := prefix
	if where == "" {
		where = "<root>"
	}
	for _, e := range graph.Edges {
		edge := Edge{Graph: where, From: e.From(), To: e.To()}
		out[where+" "+edge.From+" "+edge.To] = edge
	}
	for _, node := range graph.Nodes {
		if node.Graph != nil {
			flattenEdges(node.Graph, ir.JoinPath(prefix, node.ID), out)
		}
	}
	return out
}

// runtimeDefault is a runtime symbol's declared default, without running the
// whole symbol table: the diff only needs it to pick a sequence length.
func runtimeDefault(doc *ir.Doc, name string, fallback float64) float64 {
	s, ok := doc.Symbols[name]
	if !ok || !s.HasNumber {
		return fallback
	}
	return s.Number
}

func delta(metric string, a, b float64) Delta {
	if math.IsNaN(a) || math.IsInf(a, 0) {
		a = 0
	}
	if math.IsNaN(b) || math.IsInf(b, 0) {
		b = 0
	}
	d := Delta{Metric: metric, A: a, B: b, Delta: b - a}
	if a != 0 {
		r := b / a
		d.Ratio = &r
	}
	return d
}

// --- small helpers ----------------------------------------------------------

// sameJSON compares two parameter values the way the document would store them,
// so 1 and 1.0 are the same and a missing value and a null are too.
func sameJSON(a, b any) bool {
	return string(encode(a)) == string(encode(b))
}

func encode(v any) json.RawMessage {
	raw, err := json.Marshal(v)
	if err != nil || raw == nil {
		return json.RawMessage("null")
	}
	return raw
}

func union(a, b []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, list := range [][]string{a, b} {
		for _, k := range list {
			if !seen[k] {
				seen[k] = true
				out = append(out, k)
			}
		}
	}
	sort.Strings(out)
	return out
}

func keysOf(m map[string]ir.SymbolDef) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func blockPaths(m map[string]Block) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func edgeKeys(m map[string]Edge) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func paramKeys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
