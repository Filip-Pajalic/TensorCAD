// Package infer gives every edge in a design a shape.
//
// The walk is topological: each incoming edge is checked against the consuming
// port's declared pattern, and each producing port's pattern is instantiated.
// Design symbols are substituted before comparison; the runtime symbols B and T
// stay indeterminate, so a mismatch is a genuine polynomial difference rather
// than a coincidence of numbers.
//
// It is its own package because it sits above both of the ones it needs:
// catalog already depends on shapes, so neither of them can hold this.
package infer

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/shapes"
)

// Issue is something wrong with the wiring, addressed to a node.
type Issue struct {
	Path     string `json:"path"`
	Port     string `json:"port,omitempty"`
	Message  string `json:"message"`
	Severity string `json:"severity"`
	// Rule is the stable id when the issue came from a block's own constraints.
	Rule string `json:"rule,omitempty"`
	// Param is the parameter that caused it, so the inspector can point at the
	// field rather than at the block.
	Param string `json:"param,omitempty"`
}

// Result is everything the walk learned.
type Result struct {
	// Outputs maps "path:port" to the shape an output port produces.
	Outputs map[string]shapes.Shape
	// Inputs maps "path:port" to the shape an input port actually received.
	Inputs map[string]shapes.Shape
	// ProducerOf maps a consumer "path:port" to its producer "path:port". It is
	// what lets the memory model count a tensor once even when several blocks
	// read it.
	ProducerOf map[string]string
	// Resolved holds each node's parameters, reused by the rules engine rather
	// than resolved a second time.
	Resolved map[string]*catalog.Resolved
	// Ports holds each node's pins, for the editor.
	Ports map[string]catalog.Ports
	// Expansions is the subgraph each composite stood for, by path.
	//
	// The walk builds these anyway. Handing them back is what lets an editor
	// draw the inside of a block without expanding it a second time, with its
	// own copy of the rules for how a composite unfolds.
	Expansions map[string]*ir.Graph
	Issues     []Issue
}

// Options steer the walk.
type Options struct {
	// ExpandComposites recurses into composite expansions instead of trusting
	// their declared ports.
	ExpandComposites bool
}

// EvalCtxFor is the environment a node's patterns evaluate in: the design's
// symbols, overlaid with the node's own numeric parameters, so a port written
// over "in_features" resolves through the parameter to the symbol behind it.
func EvalCtxFor(r *catalog.Resolved, symbols *ir.SymbolTable) shapes.EvalCtx {
	values := make(map[string]float64, len(symbols.Values)+len(r.P))
	for k, v := range symbols.Values {
		values[k] = v
	}
	for k, v := range r.P {
		if n, ok := asNumber(v); ok && !math.IsInf(n, 0) && !math.IsNaN(n) {
			values[k] = n
		}
	}
	known := make(map[string]bool, len(values))
	for k := range values {
		known[k] = true
	}
	return shapes.EvalCtx{Values: values, Known: known, Substitutions: r.S}
}

func asNumber(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	}
	return 0, false
}

// boundariesOf finds the entry and exit nodes of a subgraph.
func boundariesOf(nodes []ir.NodeDef) (bIn, bOut *ir.NodeDef) {
	for i := range nodes {
		switch nodes[i].Type {
		case "boundary_in":
			if bIn == nil {
				bIn = &nodes[i]
			}
		case "boundary_out":
			if bOut == nil {
				bOut = &nodes[i]
			}
		}
	}
	return bIn, bOut
}

// containerPorts derives a repeat container's ports from its boundary nodes.
//
// A boundary node carries shapes and nothing else, so what comes back is a
// normalised port set built from bare patterns: the rest of a port declaration
// is about the outside of a block, and a container has no outside of its own.
func containerPorts(node ir.NodeDef) catalog.Ports {
	var bIn, bOut *ir.NodeDef
	if node.Graph != nil {
		bIn, bOut = boundariesOf(node.Graph.Nodes)
	}
	return catalog.NormalisePorts(catalog.Ports{
		In:  boundaryShapes(bIn),
		Out: boundaryShapes(bOut),
	})
}

func boundaryShapes(n *ir.NodeDef) map[string]catalog.PortSpec {
	out := map[string]catalog.PortSpec{}
	if n == nil {
		return out
	}
	m, ok := n.Params["ports"].(map[string]any)
	if !ok {
		return out
	}
	for name, v := range m {
		if s, ok := v.(string); ok {
			out[name] = catalog.Port(s)
		}
	}
	return out
}

// portsAsked asks a block for its pins, turning a bad ports function into an
// issue on the node rather than a crash that takes the whole analysis with it.
func portsAsked(def *catalog.BlockDef, r *catalog.Resolved) (p catalog.Ports, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("%v", rec)
		}
	}()
	return catalog.PortsOf(def, r), nil
}

// sortedNames is the order ports are visited in.
//
// The TypeScript walks them in declaration order, which a Go map does not keep.
// The order shows up in two places: which issues come out first, and which of
// two disagreeing ports is named first in the message. The parity test
// therefore compares issues as a set. It would also decide the batch dimensions
// if a block mixed a port whose pattern has an ellipsis with one that has none,
// but no block does, and one that did would be just as ambiguous in the
// TypeScript.
func sortedNames(m map[string]catalog.PortSpec) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// topoOrder puts producers before consumers, reporting a cycle rather than
// looping on it.
func topoOrder(graph *ir.Graph, issues *[]Issue, prefix string) []ir.NodeDef {
	byID := make(map[string]*ir.NodeDef, len(graph.Nodes))
	for i := range graph.Nodes {
		byID[graph.Nodes[i].ID] = &graph.Nodes[i]
	}

	// A slice rather than a set, so dependencies are visited in the order the
	// edges were written and a cycle is always reported by the same path.
	deps := make(map[string][]string, len(graph.Nodes))
	seen := map[string]bool{}
	for _, e := range graph.Edges {
		f, errF := ir.SplitEndpoint(e.From())
		t, errT := ir.SplitEndpoint(e.To())
		if errF != nil || errT != nil {
			*issues = append(*issues, Issue{
				Path:     prefix,
				Message:  fmt.Sprintf("Malformed edge %q -> %q, expected \"node:port\" on both ends", e.From(), e.To()),
				Severity: "error",
			})
			continue
		}
		if _, ok := byID[f.Node]; !ok {
			*issues = append(*issues, Issue{
				Path:     ir.JoinPath(prefix, f.Node),
				Message:  fmt.Sprintf("Edge source %q refers to a missing node", e.From()),
				Severity: "error",
			})
			continue
		}
		if _, ok := byID[t.Node]; !ok {
			*issues = append(*issues, Issue{
				Path:     ir.JoinPath(prefix, t.Node),
				Message:  fmt.Sprintf("Edge target %q refers to a missing node", e.To()),
				Severity: "error",
			})
			continue
		}
		key := t.Node + "\x00" + f.Node
		if seen[key] {
			continue
		}
		seen[key] = true
		deps[t.Node] = append(deps[t.Node], f.Node)
	}

	var order []ir.NodeDef
	const pending, done = 1, 2
	state := map[string]int{}
	var visit func(id string, stack []string)
	visit = func(id string, stack []string) {
		switch state[id] {
		case done:
			return
		case pending:
			path := append(append([]string{}, stack...), id)
			*issues = append(*issues, Issue{
				Path:     ir.JoinPath(prefix, id),
				Message:  "Cycle detected: " + strings.Join(path, " -> ") + ". Use a repeat container for recurrence.",
				Severity: "error",
			})
			return
		}
		state[id] = pending
		for _, d := range deps[id] {
			visit(d, append(append([]string{}, stack...), id))
		}
		state[id] = done
		if n, ok := byID[id]; ok {
			order = append(order, *n)
		}
	}
	for _, n := range graph.Nodes {
		visit(n.ID, nil)
	}
	return order
}

// producerMap is the one edge feeding each input port. A second edge into the
// same port is an error, not a merge.
func producerMap(graph *ir.Graph, issues *[]Issue, prefix string) map[string]string {
	m := map[string]string{}
	for _, e := range graph.Edges {
		to := e.To()
		if _, taken := m[to]; taken {
			t, err := ir.SplitEndpoint(to)
			if err != nil {
				continue
			}
			*issues = append(*issues, Issue{
				Path:     ir.JoinPath(prefix, t.Node),
				Port:     t.Port,
				Message:  fmt.Sprintf("Input port %q has more than one incoming edge", t.Port),
				Severity: "error",
			})
			continue
		}
		m[to] = e.From()
	}
	return m
}

type walker struct {
	doc     *ir.Doc
	cat     catalog.Catalog
	symbols *ir.SymbolTable
	opts    Options
	result  *Result
}

// Shapes walks a document and reports the shape on every port.
func Shapes(doc *ir.Doc, symbols *ir.SymbolTable, opts Options) *Result {
	result := &Result{
		Issues:     []Issue{},
		Outputs:    map[string]shapes.Shape{},
		Inputs:     map[string]shapes.Shape{},
		ProducerOf: map[string]string{},
		Resolved:   map[string]*catalog.Resolved{},
		Ports:      map[string]catalog.Ports{},
		Expansions: map[string]*ir.Graph{},
	}
	w := &walker{doc: doc, cat: catalog.Of(doc), symbols: symbols, opts: opts, result: result}
	w.graph(&doc.Graph, "", map[string]shapes.Shape{})
	return result
}

func (w *walker) issue(i Issue) { w.result.Issues = append(w.result.Issues, i) }

func (w *walker) graph(graph *ir.Graph, prefix string, seeds map[string]shapes.Shape) {
	order := topoOrder(graph, &w.result.Issues, prefix)
	producers := producerMap(graph, &w.result.Issues, prefix)

	for _, node := range order {
		path := ir.JoinPath(prefix, node.ID)
		def, ok := w.cat[node.Type]
		if !ok {
			w.issue(Issue{Path: path, Message: fmt.Sprintf("Unknown block type %q", node.Type), Severity: "error"})
			continue
		}

		resolved := catalog.ResolveNodeParams(def, node.Params, w.symbols)
		w.result.Resolved[path] = resolved
		for _, e := range resolved.Errors {
			w.issue(Issue{Path: path, Message: e, Severity: "error"})
		}

		var nodePorts catalog.Ports
		if catalog.IsContainer(def) {
			nodePorts = containerPorts(node)
		} else {
			p, err := portsAsked(def, resolved)
			if err != nil {
				w.issue(Issue{Path: path, Message: "Could not determine ports: " + err.Error(), Severity: "error"})
				continue
			}
			nodePorts = p
		}
		w.result.Ports[path] = nodePorts

		// A block's own constraints, surfaced here too so the canvas has them
		// before the rule engine has run. They keep their id and severity.
		if def.Constraints != nil {
			for _, c := range def.Constraints(resolved) {
				// Shape inference reports errors and warnings; an informational
				// constraint is still worth seeing, so it arrives as a warning
				// here and keeps its own severity in the rule engine.
				sev := "warning"
				if c.Severity == "error" {
					sev = "error"
				}
				w.issue(Issue{Path: path, Message: c.Message, Severity: sev, Rule: c.ID, Param: c.Param})
			}
		}

		ctx := EvalCtxFor(resolved, w.symbols)
		ctx = w.alongSource(ctx, node, prefix, nodePorts, producers)
		batch := w.checkInputs(node, path, prefix, nodePorts, producers, ctx)

		if catalog.IsContainer(def) && node.Graph != nil {
			w.recurseContainer(node, path, nodePorts)
			continue
		}

		if catalog.IsComposite(def) && w.opts.ExpandComposites {
			exp, ok := catalog.Expand(def, resolved.RawFull, resolved)
			if !ok {
				w.issue(Issue{
					Path:     path,
					Message:  fmt.Sprintf("Expansion failed: %s has no expansion", node.Type),
					Severity: "error",
				})
				continue
			}
			inner := &ir.Graph{Nodes: exp.Nodes, Edges: exp.Edges}
			w.result.Expansions[path] = inner
			w.recurseComposite(inner, path, nodePorts)
			continue
		}

		w.instantiateOutputs(node, path, nodePorts, ctx, batch, seeds)
	}
}

// alongSource lets a block run along the source.
//
// A block declares its pins over T, the sequence: "B heads T head_dim", or a
// rearrange from "B T (H dh)". In a declaration T means *this block's
// sequence*, and a block that everything arrives at S long is running along
// the source, an encoder's, so there T is S — on every pin, in and out. A
// block that receives both S and T is left as it is, so the S one is reported
// against the T its pattern asks for: two sequences meeting where a block only
// takes one is a real mismatch, which is what the check is for. A design with
// no S never reaches the substitution.
func (w *walker) alongSource(
	ctx shapes.EvalCtx, node ir.NodeDef, prefix string,
	nodePorts catalog.Ports, producers map[string]string,
) shapes.EvalCtx {
	if !w.symbols.Runtime[ir.SourceSymbol] {
		return ctx
	}
	seen := map[string]bool{}
	for portName := range nodePorts.In {
		from, ok := producers[node.ID+":"+portName]
		if !ok {
			continue
		}
		parsed, err := ir.SplitEndpoint(from)
		if err != nil {
			continue
		}
		for _, dim := range w.result.Outputs[ir.JoinPath(prefix, parsed.Node)+":"+parsed.Port] {
			for _, name := range dim.Symbols() {
				seen[name] = true
			}
		}
	}
	if !seen[ir.SourceSymbol] || seen["T"] {
		return ctx
	}
	subs := make(map[string]shapes.Sym, len(ctx.Substitutions)+1)
	for k, v := range ctx.Substitutions {
		subs[k] = v
	}
	subs["T"] = shapes.V(ir.SourceSymbol)
	ctx.Substitutions = subs
	return ctx
}

// checkInputs compares what arrived at each input port with what the port says
// it takes, and binds the batch dimensions the outputs will be built over.
func (w *walker) checkInputs(
	node ir.NodeDef, path, prefix string,
	nodePorts catalog.Ports, producers map[string]string, ctx shapes.EvalCtx,
) shapes.Shape {
	var batch shapes.Shape
	// Distinct from len(batch) == 0: a port whose pattern has no ellipsis binds
	// an empty batch, and that still counts as having bound one.
	bound := false

	for _, portName := range sortedNames(nodePorts.In) {
		port := nodePorts.In[portName]
		from, connected := producers[node.ID+":"+portName]
		if !connected {
			w.issue(Issue{
				Path: path, Port: portName,
				Message:  fmt.Sprintf("Input port %q is not connected", portName),
				Severity: "error",
			})
			continue
		}
		fromParsed, err := ir.SplitEndpoint(from)
		if err != nil {
			w.issue(Issue{Path: path, Port: portName, Message: err.Error(), Severity: "error"})
			continue
		}
		producerEndpoint := ir.JoinPath(prefix, fromParsed.Node) + ":" + fromParsed.Port
		w.result.ProducerOf[path+":"+portName] = producerEndpoint
		actual, known := w.result.Outputs[producerEndpoint]
		if !known {
			w.issue(Issue{
				Path: path, Port: portName,
				Message:  fmt.Sprintf("Upstream shape for %q is unavailable", from),
				Severity: "warning",
			})
			continue
		}
		w.result.Inputs[path+":"+portName] = actual

		pattern, perr := shapes.ParsePattern(port.Shape)
		if perr != nil {
			w.issue(Issue{Path: path, Port: portName, Message: perr.Error(), Severity: "error"})
			continue
		}
		m := shapes.MatchPattern(actual, pattern, ctx, w.symbols.DesignValues)
		for _, e := range m.Errors {
			w.issue(Issue{
				Path: path, Port: portName,
				Message:  fmt.Sprintf("Port %q received %s: %s", portName, shapes.ShapeToString(actual), e),
				Severity: "error",
			})
		}
		if !m.OK {
			continue
		}
		if !bound {
			batch, bound = m.Batch, true
			continue
		}
		if !sameShape(batch, m.Batch, w.symbols.DesignValues) {
			w.issue(Issue{
				Path: path, Port: portName,
				Message: fmt.Sprintf("Batch dimensions disagree between ports: %s vs %s",
					shapes.ShapeToString(batch), shapes.ShapeToString(m.Batch)),
				Severity: "error",
			})
		}
	}
	return batch
}

// recurseContainer walks a repeat container's subgraph, seeded with what
// arrived at the container itself.
func (w *walker) recurseContainer(node ir.NodeDef, path string, nodePorts catalog.Ports) {
	bIn, bOut := boundariesOf(node.Graph.Nodes)

	seeds := map[string]shapes.Shape{}
	if bIn != nil {
		for _, portName := range sortedNames(nodePorts.In) {
			if s, ok := w.result.Inputs[path+":"+portName]; ok {
				seeds[bIn.ID+":"+portName] = s
			}
		}
	}
	w.graph(node.Graph, path, seeds)

	if bIn == nil || bOut == nil {
		return
	}
	// The stack applies its subgraph repeatedly, so the residual stream must
	// come out the same shape it went in.
	for _, portName := range sortedNames(nodePorts.Out) {
		outShape, hasOut := w.result.Inputs[ir.JoinPath(path, bOut.ID)+":"+portName]
		inShape, hasIn := w.result.Inputs[path+":"+portName]
		if hasOut {
			w.result.Outputs[path+":"+portName] = outShape
		}
		if hasOut && hasIn && !sameShape(inShape, outShape, w.symbols.DesignValues) {
			w.issue(Issue{
				Path: path, Port: portName,
				Message: fmt.Sprintf(
					"A repeat container must preserve its shape, but port %q goes in as %s and comes out as %s",
					portName, shapes.ShapeToString(inShape), shapes.ShapeToString(outShape)),
				Severity: "error",
			})
		}
	}
}

// recurseComposite walks the subgraph a composite stands for, so the shapes
// inside it are the ones its primitives actually see.
func (w *walker) recurseComposite(inner *ir.Graph, path string, nodePorts catalog.Ports) {
	bIn, bOut := boundariesOf(inner.Nodes)

	seeds := map[string]shapes.Shape{}
	if bIn != nil {
		for _, portName := range sortedNames(nodePorts.In) {
			if s, ok := w.result.Inputs[path+":"+portName]; ok {
				seeds[bIn.ID+":"+portName] = s
			}
		}
	}
	w.graph(inner, path, seeds)
	if bOut == nil {
		return
	}
	for _, portName := range sortedNames(nodePorts.Out) {
		if s, ok := w.result.Inputs[ir.JoinPath(path, bOut.ID)+":"+portName]; ok {
			w.result.Outputs[path+":"+portName] = s
		}
	}
}

// instantiateOutputs turns each output pattern into a concrete shape over the
// batch dimensions the inputs bound.
func (w *walker) instantiateOutputs(
	node ir.NodeDef, path string, nodePorts catalog.Ports,
	ctx shapes.EvalCtx, batch shapes.Shape, seeds map[string]shapes.Shape,
) {
	for _, portName := range sortedNames(nodePorts.Out) {
		port := nodePorts.Out[portName]
		pattern, perr := shapes.ParsePattern(port.Shape)
		if perr != nil {
			w.issue(Issue{Path: path, Port: portName, Message: perr.Error(), Severity: "error"})
			continue
		}
		// A boundary node takes its shape from the graph outside it rather than
		// from its own pattern, which is how a container's input reaches in.
		if seed, ok := seeds[node.ID+":"+portName]; ok {
			w.result.Outputs[path+":"+portName] = seed
			continue
		}
		// Always with a batch, even an empty one. A pattern's ellipsis binds
		// whatever the inputs bound, and when they bound nothing it contributes
		// no dimensions — that is a rank-1 output, not a failure. Instantiate's
		// "no batch at all" case is for a caller working outside a graph, where
		// there is nothing an ellipsis could stand for.
		shape, errs := shapes.Instantiate(pattern, ctx, batch, true)
		for _, e := range errs {
			w.issue(Issue{Path: path, Port: portName, Message: e, Severity: "error"})
		}
		if shape != nil {
			w.result.Outputs[path+":"+portName] = shape
		}
	}
}

func sameShape(a, b shapes.Shape, values map[string]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !a[i].EqualsUnder(b[i], values) {
			return false
		}
	}
	return true
}
