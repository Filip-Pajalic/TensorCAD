package golden

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/codegen"
	"github.com/tensorcad/core/explain"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/jsonx"
	"github.com/tensorcad/core/mup"
	"github.com/tensorcad/core/plan"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/rules"
	"github.com/tensorcad/core/scale"
	"github.com/tensorcad/core/shapes"
)

// Write regenerates every golden file under dir.
//
// Deliberately, and never as part of a test: a test that rewrote its own
// expectations would pass whatever the engine did.
func Write(dir string) (Summary, error) {
	var s Summary
	w := writer{dir: dir}

	names, err := presets.Names()
	if err != nil {
		return s, err
	}
	s.Presets = len(names)

	w.expressions()
	w.patterns()
	s.Primitives = w.primitives()
	s.Composites = w.composites()
	s.Blocks = w.catalogDocs()
	for _, name := range names {
		w.preset(name)
		w.analysis(name)
		w.rules(name)
		w.codegen(name)
	}
	s.OperatingPoints = len(names) * len(OperatingPoints)
	s.CodegenFiles = len(names) * len(CodegenVariants)
	w.explain(names)
	s.ScaleCases = w.scale()
	s.MupCases = w.mup()
	s.Plans = w.plans()
	s.BrokenFindings, err = w.broken()
	if err != nil {
		return s, err
	}
	return s, w.err
}

// Summary is what was written, for the command to print.
type Summary struct {
	Presets         int
	Primitives      int
	Composites      int
	Blocks          int
	OperatingPoints int
	CodegenFiles    int
	ScaleCases      int
	MupCases        int
	Plans           int
	BrokenFindings  int
}

type writer struct {
	dir string
	err error
}

// write encodes one file.
//
// Through an Encoder with HTML escaping off, because a shape pattern contains
// `<` nowhere but a source URL and a formula contain `&`, and `&` in a
// file a person reads is noise. Two-space indent, one trailing newline: the
// files are reviewed as diffs.
func (w *writer) write(name string, value any) {
	if w.err != nil {
		return
	}
	path := filepath.Join(w.dir, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		w.err = err
		return
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(value); err != nil {
		w.err = fmt.Errorf("%s: %w", name, err)
		return
	}
	w.err = os.WriteFile(path, buf.Bytes(), 0o644)
}

// read loads a file the generator both reads and rewrites.
func (w *writer) read(name string, into any) error {
	raw, err := os.ReadFile(filepath.Join(w.dir, filepath.FromSlash(name)))
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, into)
}

func evalCtx() shapes.EvalCtx {
	known := make(map[string]bool, len(Env))
	for k := range Env {
		known[k] = true
	}
	return shapes.EvalCtx{Values: Env, Known: known}
}

// ---------------------------------------------------------------------------
// The algebra
// ---------------------------------------------------------------------------

type exprOK struct {
	Src     string   `json:"src"`
	Text    string   `json:"text"`
	Value   *float64 `json:"value"`
	Symbols []string `json:"symbols"`
}

type exprErr struct {
	Src     string `json:"src"`
	Message string `json:"message"`
	Text    string `json:"text"`
}

func (w *writer) expressions() {
	ctx := evalCtx()
	out := struct {
		Env    map[string]float64 `json:"env"`
		OK     []exprOK           `json:"ok"`
		Errors []exprErr          `json:"errors"`
	}{Env: Env, OK: []exprOK{}, Errors: []exprErr{}}

	for _, src := range Expressions {
		sym, err := shapes.EvalExpr(src, ctx)
		if err != nil {
			w.err = fmt.Errorf("expression %q: %w", src, err)
			return
		}
		entry := exprOK{Src: src, Text: sym.String(), Symbols: sym.Symbols()}
		if entry.Symbols == nil {
			entry.Symbols = []string{}
		}
		if v, ok := sym.ToNumber(Env); ok {
			entry.Value = &v
		}
		out.OK = append(out.OK, entry)
	}

	for _, src := range ExpressionErrors {
		sym, err := shapes.EvalExpr(src, ctx)
		if err != nil {
			out.Errors = append(out.Errors, exprErr{Src: src, Message: err.Error()})
			continue
		}
		// Not every one of these fails; `B/T` is a legitimate polynomial that
		// simply has no number, and recording that is the point.
		out.Errors = append(out.Errors, exprErr{Src: src, Text: sym.String()})
	}
	w.write("expressions.json", out)
}

type patternParse struct {
	Src          string   `json:"src"`
	Atoms        []string `json:"atoms"`
	Instantiated *string  `json:"instantiated"`
	Errors       []string `json:"errors"`
}

type patternErr struct {
	Src     string `json:"src"`
	Message string `json:"message"`
}

type patternMatch struct {
	Actual  string   `json:"actual"`
	Pattern string   `json:"pattern"`
	OK      bool     `json:"ok"`
	Batch   string   `json:"batch"`
	Errors  []string `json:"errors"`
}

func (w *writer) patterns() {
	ctx := evalCtx()
	out := struct {
		Parse       []patternParse `json:"parse"`
		ParseErrors []patternErr   `json:"parseErrors"`
		Matches     []patternMatch `json:"matches"`
	}{Parse: []patternParse{}, ParseErrors: []patternErr{}, Matches: []patternMatch{}}

	for _, src := range Patterns {
		p, err := shapes.ParsePattern(src)
		if err != nil {
			w.err = fmt.Errorf("pattern %q: %w", src, err)
			return
		}
		entry := patternParse{Src: src, Atoms: []string{}, Errors: []string{}}
		for _, a := range p.Atoms {
			// The parts, not AtomToString: a group is written out flat here,
			// because what is being recorded is how the atom was parsed rather
			// than how it would be printed back.
			if a.Kind == shapes.AtomEllipsis {
				entry.Atoms = append(entry.Atoms, "...")
				continue
			}
			entry.Atoms = append(entry.Atoms, strings.Join(a.Parts, " "))
		}
		shape, errs := shapes.Instantiate(p, ctx, shapes.Shape{shapes.V("B"), shapes.V("T")}, true)
		if shape != nil {
			text := shapes.ShapeToString(shape)
			entry.Instantiated = &text
		}
		entry.Errors = append(entry.Errors, errs...)
		out.Parse = append(out.Parse, entry)
	}

	for _, src := range PatternErrors {
		_, err := shapes.ParsePattern(src)
		entry := patternErr{Src: src}
		if err != nil {
			entry.Message = err.Error()
		}
		out.ParseErrors = append(out.ParseErrors, entry)
	}

	for _, m := range Matches {
		actualPattern, err := shapes.ParsePattern(m.Actual)
		if err != nil {
			w.err = fmt.Errorf("match actual %q: %w", m.Actual, err)
			return
		}
		want, err := shapes.ParsePattern(m.Pattern)
		if err != nil {
			w.err = fmt.Errorf("match pattern %q: %w", m.Pattern, err)
			return
		}
		actual, _ := shapes.Instantiate(actualPattern, ctx, shapes.Shape{}, true)
		r := shapes.MatchPattern(actual, want, ctx, Env)
		entry := patternMatch{
			Actual: m.Actual, Pattern: m.Pattern, OK: r.OK,
			Batch: shapes.ShapeToString(r.Batch), Errors: []string{},
		}
		entry.Errors = append(entry.Errors, r.Errors...)
		out.Matches = append(out.Matches, entry)
	}
	w.write("patterns.json", out)
}

// ---------------------------------------------------------------------------
// The catalog
// ---------------------------------------------------------------------------

type goldenPorts struct {
	In      map[string]string `json:"in"`
	Out     map[string]string `json:"out"`
	Anchors map[string]string `json:"anchors"`
}

type primitiveCase struct {
	Type                 string             `json:"type"`
	Params               map[string]any     `json:"params"`
	Resolved             map[string]any     `json:"resolved"`
	Errors               []string           `json:"errors"`
	Ports                *goldenPorts       `json:"ports"`
	PortError            string             `json:"portError"`
	ParamCount           *float64           `json:"paramCount"`
	Flops                map[string]float64 `json:"flops"`
	Retains              []string           `json:"retains"`
	ExtraActivationBytes *float64           `json:"extraActivationBytes"`
	StateBytes           map[string]float64 `json:"stateBytes"`
	Constraints          []string           `json:"constraints"`
}

func portsGolden(p catalog.Ports) *goldenPorts {
	out := &goldenPorts{In: map[string]string{}, Out: map[string]string{}, Anchors: map[string]string{}}
	for name, spec := range p.In {
		out.In[name] = spec.Shape
		if spec.Anchor != "flow" || spec.Dtype != "inherit" {
			out.Anchors[name] = spec.Anchor + "/" + spec.Dtype
		}
	}
	for name, spec := range p.Out {
		out.Out[name] = spec.Shape
		if spec.Anchor != "flow" || spec.Dtype != "inherit" {
			out.Anchors[name] = spec.Anchor + "/" + spec.Dtype
		}
	}
	return out
}

func (w *writer) primitives() int {
	symbols := PrimitiveSymbols()
	cases := make([]primitiveCase, 0, len(Primitives))
	for _, c := range Primitives {
		def, err := catalog.Builtin.Get(c.Type)
		if err != nil {
			w.err = err
			return 0
		}
		r := catalog.ResolveNodeParams(def, c.Params, symbols)
		entry := primitiveCase{
			Type: c.Type, Params: c.Params, Resolved: r.P,
			Errors: append([]string{}, r.Errors...), Constraints: []string{},
			Ports: portsGolden(catalog.PortsOf(def, r)),
		}
		if def.ParamCount != nil {
			n := def.ParamCount(r)
			entry.ParamCount = &n
		}
		if def.Flops != nil {
			f := def.Flops(r, PrimitiveCtx)
			entry.Flops = map[string]float64{
				"fwd": f.Fwd, "elementwise": f.Elementwise,
				"fwdSeq": f.FwdSeq, "fwdSeqUnmasked": f.FwdSeqUnmasked,
			}
		}
		if def.Retains != nil {
			entry.Retains = def.Retains(r)
			if entry.Retains == nil {
				entry.Retains = []string{}
			}
		}
		if def.ExtraActivationBytes != nil {
			n := def.ExtraActivationBytes(r, PrimitiveCtx)
			entry.ExtraActivationBytes = &n
		}
		if def.StateBytes != nil {
			sb := def.StateBytes(r, PrimitiveCtx)
			entry.StateBytes = map[string]float64{"perToken": sb.PerToken, "perSeq": sb.PerSequence}
		}
		if def.Constraints != nil {
			for _, f := range def.Constraints(r) {
				entry.Constraints = append(entry.Constraints, f.ID+": "+f.Message)
			}
		}
		cases = append(cases, entry)
	}
	w.write("primitives.json", struct {
		Ctx struct {
			T     float64 `json:"T"`
			B     float64 `json:"B"`
			Bytes float64 `json:"bytes"`
			Flash bool    `json:"flash"`
		} `json:"ctx"`
		Cases []primitiveCase `json:"cases"`
	}{
		Ctx: struct {
			T     float64 `json:"T"`
			B     float64 `json:"B"`
			Bytes float64 `json:"bytes"`
			Flash bool    `json:"flash"`
		}{PrimitiveCtx.T, PrimitiveCtx.B, PrimitiveCtx.Bytes, PrimitiveCtx.Flash},
		Cases: cases,
	})
	return len(cases)
}

type goldenNode struct {
	ID     string         `json:"id"`
	Type   string         `json:"type"`
	Params map[string]any `json:"params"`
	Graph  *goldenGraph   `json:"graph,omitempty"`
}

type goldenGraph struct {
	Nodes []goldenNode `json:"nodes"`
	Edges [][2]string  `json:"edges"`
}

func flatNodes(nodes []ir.NodeDef) []goldenNode {
	out := make([]goldenNode, 0, len(nodes))
	for _, n := range nodes {
		entry := goldenNode{ID: n.ID, Type: n.Type, Params: n.Params}
		if entry.Params == nil {
			entry.Params = map[string]any{}
		}
		if n.Graph != nil {
			entry.Graph = &goldenGraph{Nodes: flatNodes(n.Graph.Nodes), Edges: flatEdges(n.Graph.Edges)}
		}
		out = append(out, entry)
	}
	return out
}

func flatEdges(edges []ir.Edge) [][2]string {
	out := make([][2]string, 0, len(edges))
	for _, e := range edges {
		out = append(out, [2]string{e[0], e[1]})
	}
	return out
}

type compositeCase struct {
	Type        string         `json:"type"`
	Params      map[string]any `json:"params"`
	Nodes       []goldenNode   `json:"nodes"`
	Edges       [][2]string    `json:"edges"`
	Constraints []string       `json:"constraints"`
}

func (w *writer) composites() int {
	symbols := PrimitiveSymbols()
	cases := make([]compositeCase, 0, len(Composites))
	for _, c := range Composites {
		def, err := catalog.Builtin.Get(c.Type)
		if err != nil {
			w.err = err
			return 0
		}
		r := catalog.ResolveNodeParams(def, c.Params, symbols)
		exp, ok := catalog.Expand(def, c.Params, r)
		if !ok {
			w.err = fmt.Errorf("%s did not expand", c.Type)
			return 0
		}
		entry := compositeCase{
			Type: c.Type, Params: c.Params,
			Nodes: flatNodes(exp.Nodes), Edges: flatEdges(exp.Edges),
			Constraints: []string{},
		}
		if def.Constraints != nil {
			for _, f := range def.Constraints(r) {
				entry.Constraints = append(entry.Constraints, f.ID+": "+f.Message)
			}
		}
		cases = append(cases, entry)
	}
	w.write("composites.json", struct {
		Cases []compositeCase `json:"cases"`
	}{cases})
	return len(cases)
}

// goldenParamDoc is [name, type, doc, values], which keeps the file readable
// beside the block it describes.
type goldenParamDoc [4]any

type blockDocs struct {
	Type     string           `json:"type"`
	Kind     string           `json:"kind"`
	Category string           `json:"category"`
	Name     string           `json:"name"`
	Summary  string           `json:"summary"`
	Formula  string           `json:"formula"`
	Refs     []string         `json:"refs"`
	Params   []goldenParamDoc `json:"params"`
}

// catalogDocs pins the prose: every block's summary, formula and sources, and
// every parameter's one-line documentation.
//
// This is what the inspector shows and what explain reads out, so it is part of
// the engine's output rather than decoration around it — and it is the easiest
// thing to lose, because nothing computes with it.
func (w *writer) catalogDocs() int {
	types := make([]string, 0, len(catalog.Builtin))
	for name := range catalog.Builtin {
		types = append(types, name)
	}
	sort.Strings(types)

	blocks := make([]blockDocs, 0, len(types))
	for _, name := range types {
		def := catalog.Builtin[name]
		entry := blockDocs{
			Type: name, Kind: def.Kind, Category: def.Category,
			Name: def.Docs.Name, Summary: def.Docs.Summary, Formula: def.Docs.Formula,
			Refs: def.Docs.Refs, Params: []goldenParamDoc{},
		}
		if entry.Refs == nil {
			entry.Refs = []string{}
		}
		for _, p := range def.Params {
			var values any
			if p.Spec.Values != nil {
				values = p.Spec.Values
			}
			entry.Params = append(entry.Params,
				goldenParamDoc{p.Name, string(p.Spec.Type), p.Spec.Doc, values})
		}
		blocks = append(blocks, entry)
	}
	w.write("catalog-docs.json", struct {
		Blocks []blockDocs `json:"blocks"`
	}{blocks})
	return len(blocks)
}

// ---------------------------------------------------------------------------
// Per preset
// ---------------------------------------------------------------------------

type pair [2]string

func sortedPairs(m map[string]string) []pair {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]pair, 0, len(keys))
	for _, k := range keys {
		out = append(out, pair{k, m[k]})
	}
	return out
}

type numPair [2]any

func sortedNumbers(m map[string]float64) []numPair {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]numPair, 0, len(keys))
	for _, k := range keys {
		out = append(out, numPair{k, m[k]})
	}
	return out
}

type goldenIssue struct {
	Path     string `json:"path"`
	Port     string `json:"port,omitempty"`
	Message  string `json:"message"`
	Severity string `json:"severity"`
	Rule     string `json:"rule,omitempty"`
	Param    string `json:"param,omitempty"`
}

type inferGolden struct {
	Outputs    []pair        `json:"outputs"`
	Inputs     []pair        `json:"inputs"`
	ProducerOf []pair        `json:"producerOf"`
	Issues     []goldenIssue `json:"issues"`
}

// inferenceOf records what shape inference decided.
//
// Shapes are compared as text: a shape is a polynomial, and its printed form is
// what the editor puts on a wire, so agreeing on the string is agreeing on both
// the value and how it reads.
//
// Issues are sorted rather than in report order. A block's ports live in a map,
// so the sequence is not the contract; the set is.
func inferenceOf(res *infer.Result) inferGolden {
	out := inferGolden{Issues: []goldenIssue{}}
	shapesText := map[string]string{}
	for k, v := range res.Outputs {
		shapesText[k] = shapes.ShapeToString(v)
	}
	out.Outputs = sortedPairs(shapesText)

	shapesText = map[string]string{}
	for k, v := range res.Inputs {
		shapesText[k] = shapes.ShapeToString(v)
	}
	out.Inputs = sortedPairs(shapesText)
	out.ProducerOf = sortedPairs(res.ProducerOf)

	for _, i := range res.Issues {
		out.Issues = append(out.Issues, goldenIssue{
			Path: i.Path, Port: i.Port, Message: i.Message,
			Severity: i.Severity, Rule: i.Rule, Param: i.Param,
		})
	}
	sort.Slice(out.Issues, func(a, b int) bool {
		x, _ := json.Marshal(out.Issues[a])
		y, _ := json.Marshal(out.Issues[b])
		return string(x) < string(y)
	})
	return out
}

func (w *writer) preset(name string) {
	doc := presets.MustGet(name)
	table := ir.ResolveSymbols(doc)

	runtime := make([]string, 0, len(table.Runtime))
	for k := range table.Runtime {
		runtime = append(runtime, k)
	}
	sort.Strings(runtime)

	for symbol, value := range table.Values {
		if value != value {
			w.err = fmt.Errorf("%s: symbol %s is NaN", name, symbol)
			return
		}
	}

	w.write("golden/"+name+".json", struct {
		Preset  string `json:"preset"`
		Symbols struct {
			Order        []string           `json:"order"`
			Values       map[string]float64 `json:"values"`
			DesignValues map[string]float64 `json:"designValues"`
			Runtime      []string           `json:"runtime"`
			Docs         map[string]string  `json:"docs"`
			Errors       []string           `json:"errors"`
		} `json:"symbols"`
		Infer         inferGolden `json:"infer"`
		InferExpanded inferGolden `json:"inferExpanded"`
	}{
		Preset: name,
		Symbols: struct {
			Order        []string           `json:"order"`
			Values       map[string]float64 `json:"values"`
			DesignValues map[string]float64 `json:"designValues"`
			Runtime      []string           `json:"runtime"`
			Docs         map[string]string  `json:"docs"`
			Errors       []string           `json:"errors"`
		}{table.Order, table.Values, table.DesignValues, runtime, table.Docs, table.Errors},
		// Both modes, because both are used: the rule engine and the canvas
		// read the graph a person drew, the analysis reads it expanded.
		Infer:         inferenceOf(infer.Shapes(doc, table, infer.Options{})),
		InferExpanded: inferenceOf(infer.Shapes(doc, table, infer.Options{ExpandComposites: true})),
	})
}

func (w *writer) analysis(name string) {
	type resolvedOptions struct {
		T                   float64               `json:"T"`
		B                   float64               `json:"B"`
		Dtype               string                `json:"dtype"`
		InferenceDtype      string                `json:"inferenceDtype"`
		KvDtype             string                `json:"kvDtype"`
		Hardware            string                `json:"hardware"`
		GPUs                float64               `json:"gpus"`
		Parallel            analysis.ParallelPlan `json:"parallel"`
		Optimizer           string                `json:"optimizer"`
		Recompute           string                `json:"recompute"`
		Flash               bool                  `json:"flash"`
		Tokens              float64               `json:"tokens"`
		TokensWereDefaulted bool                  `json:"tokensWereDefaulted"`
		MFU                 float64               `json:"mfu"`
		DecodeEfficiency    float64               `json:"decodeEfficiency"`
		Concurrency         float64               `json:"concurrency"`
	}
	type repeat struct {
		Path   string  `json:"path"`
		Type   string  `json:"type"`
		Count  float64 `json:"count"`
		Active float64 `json:"active"`
	}
	type oneCase struct {
		Label   string          `json:"label"`
		Options resolvedOptions `json:"options"`
		Params  struct {
			Total              float64   `json:"total"`
			Active             float64   `json:"active"`
			Embedding          float64   `json:"embedding"`
			Head               float64   `json:"head"`
			NonEmbedding       float64   `json:"nonEmbedding"`
			NonEmbeddingActive float64   `json:"nonEmbeddingActive"`
			ByPath             []numPair `json:"byPath"`
			ByCategory         []numPair `json:"byCategory"`
			ByType             []numPair `json:"byType"`
		} `json:"params"`
		Flops struct {
			FwdDense             float64   `json:"fwdDense"`
			FwdAttention         float64   `json:"fwdAttention"`
			FwdAttentionUnmasked float64   `json:"fwdAttentionUnmasked"`
			FwdTotal             float64   `json:"fwdTotal"`
			FwdTotalUnmasked     float64   `json:"fwdTotalUnmasked"`
			Elementwise          float64   `json:"elementwise"`
			TrainPerToken        float64   `json:"trainPerToken"`
			AttentionShare       float64   `json:"attentionShare"`
			RuleOfThumb2N        float64   `json:"ruleOfThumb2N"`
			RuleOfThumb6N        float64   `json:"ruleOfThumb6N"`
			ByPath               []numPair `json:"byPath"`
			ByCategory           []numPair `json:"byCategory"`
		} `json:"flops"`
		Kv struct {
			BytesPerToken         float64   `json:"bytesPerToken"`
			BytesPerSequenceFixed float64   `json:"bytesPerSequenceFixed"`
			ByPath                []numPair `json:"byPath"`
		} `json:"kv"`
		Memory struct {
			WeightsBytes float64 `json:"weightsBytes"`
			Train        struct {
				Weights             float64              `json:"weights"`
				Grads               float64              `json:"grads"`
				Optimizer           float64              `json:"optimizer"`
				Activations         float64              `json:"activations"`
				Logits              float64              `json:"logits"`
				Total               float64              `json:"total"`
				PerGpu              analysis.TrainPerGpu `json:"perGpu"`
				ActivationsByPath   []numPair            `json:"activationsByPath"`
				ActivationsByTensor []numPair            `json:"activationsByTensor"`
			} `json:"train"`
			Infer          analysis.InferMemory `json:"infer"`
			OptimizerLabel string               `json:"optimizerLabel"`
			Notes          []string             `json:"notes"`
		} `json:"memory"`
		Throughput analysis.ThroughputResult `json:"throughput"`
		Cost       analysis.CostResult       `json:"cost"`
		Chinchilla struct {
			OptimalTokens        float64   `json:"optimalTokens"`
			TokensPerParam       float64   `json:"tokensPerParam"`
			TokensPerActiveParam float64   `json:"tokensPerActiveParam"`
			OverTrainingRatio    float64   `json:"overTrainingRatio"`
			PredictedLoss        []numPair `json:"predictedLoss"`
			Verdict              string    `json:"verdict"`
		} `json:"chinchilla"`
		Errors []string `json:"errors"`
		Flat   struct {
			Nodes   int      `json:"nodes"`
			Blocks  int      `json:"blocks"`
			Repeats []repeat `json:"repeats"`
			Errors  []string `json:"errors"`
		} `json:"flat"`
	}

	doc := presets.MustGet(name)
	cases := make([]oneCase, 0, len(OperatingPoints))
	for _, point := range OperatingPoints {
		a, err := analysis.Analyze(doc, point.Options, analysis.Inputs{})
		if err != nil {
			w.err = fmt.Errorf("%s/%s: %w", name, point.Label, err)
			return
		}
		var c oneCase
		c.Label = point.Label
		c.Options = resolvedOptions{
			T: a.Options.T, B: a.Options.B, Dtype: a.Options.Dtype,
			InferenceDtype: a.Options.InferenceDtype, KvDtype: a.Options.KvDtype,
			Hardware: a.Options.Hardware.ID, GPUs: a.Options.GPUs,
			Parallel: a.Options.Parallel, Optimizer: a.Options.Optimizer,
			Recompute: a.Options.Recompute, Flash: a.Options.Flash,
			Tokens: a.Options.Tokens, TokensWereDefaulted: a.Options.TokensWereDefault,
			MFU: a.Options.MFU, DecodeEfficiency: a.Options.DecodeEfficiency,
			Concurrency: a.Options.Concurrency,
		}
		c.Params.Total, c.Params.Active = a.Params.Total, a.Params.Active
		c.Params.Embedding, c.Params.Head = a.Params.Embedding, a.Params.Head
		c.Params.NonEmbedding, c.Params.NonEmbeddingActive = a.Params.NonEmbedding, a.Params.NonEmbeddingActive
		c.Params.ByPath = sortedNumbers(a.Params.ByPath)
		c.Params.ByCategory = sortedNumbers(a.Params.ByCategory)
		c.Params.ByType = sortedNumbers(a.Params.ByType)

		c.Flops.FwdDense, c.Flops.FwdAttention = a.Flops.FwdDense, a.Flops.FwdAttention
		c.Flops.FwdAttentionUnmasked = a.Flops.FwdAttentionUnmasked
		c.Flops.FwdTotal, c.Flops.FwdTotalUnmasked = a.Flops.FwdTotal, a.Flops.FwdTotalUnmasked
		c.Flops.Elementwise, c.Flops.TrainPerToken = a.Flops.Elementwise, a.Flops.TrainPerToken
		c.Flops.AttentionShare = a.Flops.AttentionShare
		c.Flops.RuleOfThumb2N, c.Flops.RuleOfThumb6N = a.Flops.RuleOfThumb2N, a.Flops.RuleOfThumb6N
		c.Flops.ByPath = sortedNumbers(a.Flops.ByPath)
		c.Flops.ByCategory = sortedNumbers(a.Flops.ByCategory)

		c.Kv.BytesPerToken, c.Kv.BytesPerSequenceFixed = a.Kv.BytesPerToken, a.Kv.BytesPerSequenceFixed
		c.Kv.ByPath = sortedNumbers(a.Kv.ByPath)

		c.Memory.WeightsBytes = a.Memory.WeightsBytes
		c.Memory.Train.Weights, c.Memory.Train.Grads = a.Memory.Train.Weights, a.Memory.Train.Grads
		c.Memory.Train.Optimizer, c.Memory.Train.Activations = a.Memory.Train.Optimizer, a.Memory.Train.Activations
		c.Memory.Train.Logits, c.Memory.Train.Total = a.Memory.Train.Logits, a.Memory.Train.Total
		c.Memory.Train.PerGpu = a.Memory.Train.PerGpu
		c.Memory.Train.ActivationsByPath = sortedNumbers(a.Memory.Train.ActivationsByPath)
		c.Memory.Train.ActivationsByTensor = sortedNumbers(a.Memory.Train.ActivationsByTensor)
		c.Memory.Infer = a.Memory.Infer
		c.Memory.OptimizerLabel, c.Memory.Notes = a.Memory.OptimizerLabel, a.Memory.Notes

		c.Throughput, c.Cost = *a.Throughput, *a.Cost
		c.Chinchilla.OptimalTokens = a.Chinchilla.OptimalTokens
		c.Chinchilla.TokensPerParam = a.Chinchilla.TokensPerParam
		c.Chinchilla.TokensPerActiveParam = a.Chinchilla.TokensPerActiveParam
		c.Chinchilla.OverTrainingRatio = a.Chinchilla.OverTrainingRatio
		c.Chinchilla.PredictedLoss = sortedNumbers(a.Chinchilla.PredictedLoss)
		c.Chinchilla.Verdict = a.Chinchilla.Verdict
		c.Errors = a.Errors

		c.Flat.Nodes, c.Flat.Blocks = len(a.Flat.Nodes), len(a.Flat.Blocks)
		c.Flat.Errors = a.Flat.Errors
		c.Flat.Repeats = []repeat{}
		for _, r := range a.Flat.Repeats {
			c.Flat.Repeats = append(c.Flat.Repeats, repeat{r.Path, r.Type, r.Count, r.Active})
		}
		cases = append(cases, c)
	}
	w.write("analysis/"+name+".json", struct {
		Preset string    `json:"preset"`
		Cases  []oneCase `json:"cases"`
	}{name, cases})
}

type goldenFinding struct {
	Rule     string `json:"rule"`
	Severity string `json:"severity"`
	Path     string `json:"path,omitempty"`
	Port     string `json:"port,omitempty"`
	Param    string `json:"param,omitempty"`
	Message  string `json:"message"`
	Hint     string `json:"hint,omitempty"`
}

func findingsOf(report *rules.Report) []goldenFinding {
	out := make([]goldenFinding, 0, len(report.Findings))
	for _, f := range report.Findings {
		out = append(out, goldenFinding{
			Rule: f.Rule, Severity: f.Severity, Path: f.Path, Port: f.Port,
			Param: f.Param, Message: f.Message, Hint: f.Hint,
		})
	}
	return out
}

// rules records the whole finding, message and hint included.
//
// A rule is a sentence a person reads and acts on: "the cache dominates" and
// "quantize the weights" send someone to different parts of their design.
func (w *writer) rules(name string) {
	type oneCase struct {
		Label    string          `json:"label"`
		OK       bool            `json:"ok"`
		Counts   map[string]int  `json:"counts"`
		Findings []goldenFinding `json:"findings"`
	}
	doc := presets.MustGet(name)
	cases := make([]oneCase, 0, len(OperatingPoints))
	for _, point := range OperatingPoints {
		report, err := rules.Validate(doc, point.Options)
		if err != nil {
			w.err = fmt.Errorf("%s/%s: %w", name, point.Label, err)
			return
		}
		cases = append(cases, oneCase{
			Label: point.Label, OK: report.OK, Counts: report.Counts,
			Findings: findingsOf(report),
		})
	}
	w.write("rules/"+name+".json", struct {
		Preset string    `json:"preset"`
		Cases  []oneCase `json:"cases"`
	}{name, cases})
}

// codegen records the generated model.py whole.
//
// The strongest file in the set: a model.py is the engine's output as one
// artifact, and a file that differs by one character was generated differently.
func (w *writer) codegen(name string) {
	// The options travel with the answer. A label alone would leave every
	// reader of the file — the Go test, the test that runs the compiled
	// module — restating what "dense" meant, which is a second place to get
	// it wrong.
	type oneCase struct {
		Label    string              `json:"label"`
		Options  codegen.WireOptions `json:"options"`
		Warnings []string            `json:"warnings"`
		Model    string              `json:"model"`
	}
	doc := presets.MustGet(name)
	cases := make([]oneCase, 0, len(CodegenVariants))
	for _, variant := range CodegenVariants {
		g := codegen.GenerateTorch(doc, variant.Options)
		model := ""
		for _, file := range g.Files {
			if file.Path == "model.py" {
				model = file.Contents
			}
		}
		cases = append(cases, oneCase{
			Label: variant.Label, Options: codegen.Wire(variant.Options),
			Warnings: g.Warnings, Model: model,
		})
	}
	w.write("codegen/"+name+".json", struct {
		Preset string    `json:"preset"`
		Cases  []oneCase `json:"cases"`
	}{name, cases})
}

// explain records the top eight blocks of each design.
//
// A whole design's worth for twenty presets would be a megabyte saying the same
// thing twenty times; the eight that contribute most are where a mistake shows.
func (w *writer) explain(names []string) {
	type explainedParam [4]any
	type line struct {
		Path   string  `json:"path"`
		Type   string  `json:"type"`
		Params float64 `json:"params"`
	}
	type block struct {
		Path        string               `json:"path"`
		Type        string               `json:"type"`
		Kind        string               `json:"kind"`
		Docs        catalog.BlockDocs    `json:"docs"`
		Copies      explain.Copies       `json:"copies"`
		Params      []explainedParam     `json:"params"`
		Shapes      explain.Shapes       `json:"shapes"`
		Contributes explain.Contribution `json:"contributes"`
		Breakdown   []line               `json:"breakdown"`
	}
	type oneCase struct {
		Preset string  `json:"preset"`
		Blocks []block `json:"blocks"`
	}

	cases := make([]oneCase, 0, len(names))
	for _, name := range names {
		all, err := explain.All(presets.MustGet(name), analysis.Options{})
		if err != nil {
			w.err = fmt.Errorf("explain %s: %w", name, err)
			return
		}
		if len(all) > 8 {
			all = all[:8]
		}
		blocks := make([]block, 0, len(all))
		for _, e := range all {
			b := block{
				Path: e.Path, Type: e.Type, Kind: e.Kind, Docs: e.Docs,
				Copies: e.Copies, Shapes: e.Shapes, Contributes: e.Contributes,
				Params: []explainedParam{}, Breakdown: []line{},
			}
			for _, key := range e.ParamOrder {
				p := e.Params[key]
				var expression, doc any
				if p.Expression != "" {
					expression = p.Expression
				}
				if p.Doc != "" {
					doc = p.Doc
				}
				b.Params = append(b.Params, explainedParam{key, expression, p.Value, doc})
			}
			for _, l := range e.Breakdown {
				b.Breakdown = append(b.Breakdown, line{l.Path, l.Type, l.Params})
			}
			blocks = append(blocks, b)
		}
		cases = append(cases, oneCase{Preset: name, Blocks: blocks})
	}
	w.write("explain.json", struct {
		Cases []oneCase `json:"cases"`
	}{cases})
}

// scale records what a search landed on.
//
// A search is where two implementations drift most easily: the same binary
// search over the same rounding has to reach the same widths, not merely near
// them. The comparison is on the symbols the scaled document ends up with.
// plans records what the planner offers for a handful of clusters.
//
// The whole result, because the ranking is as much the answer as the numbers
// are: a planner that starts recommending a pipeline where it used to recommend
// sharding the optimizer has changed its advice, and that should be reviewed
// rather than discovered.
func (w *writer) plans() int {
	type oneCase struct {
		Label      string           `json:"label"`
		Preset     string           `json:"preset"`
		Seq        float64          `json:"seq"`
		Cluster    plan.Request     `json:"cluster"`
		Budget     float64          `json:"budget"`
		Considered int              `json:"considered"`
		Fits       []plan.Candidate `json:"fits"`
		Closest    *plan.Candidate  `json:"closest"`
		Notes      []string         `json:"notes"`
	}
	cases := make([]oneCase, 0, len(PlanCases))
	for _, c := range PlanCases {
		seq := c.Seq
		res, err := plan.Search(presets.MustGet(c.Preset),
			analysis.Options{T: &seq, Hardware: "h100-sxm"}, c.Cluster)
		if err != nil {
			w.err = fmt.Errorf("plan %s: %w", c.Label, err)
			return 0
		}
		cases = append(cases, oneCase{
			Label: c.Label, Preset: c.Preset, Seq: c.Seq, Cluster: c.Cluster,
			Budget: res.Budget, Considered: res.Considered,
			Fits: res.Fits, Closest: res.Closest, Notes: res.Notes,
		})
	}
	w.write("plans.json", struct {
		Cases []oneCase `json:"cases"`
	}{cases})
	return len(cases)
}

func (w *writer) scale() int {
	type change [3]any
	type oneCase struct {
		Label      string        `json:"label"`
		Preset     string        `json:"preset"`
		Options    scale.Options `json:"options"`
		Achieved   float64       `json:"achieved"`
		Target     float64       `json:"target"`
		Changes    []change      `json:"changes"`
		Notes      []string      `json:"notes"`
		Name       string        `json:"name"`
		NotesOnDoc string        `json:"notesOnDoc"`
		Published  *ir.Published `json:"published"`
		Symbols    []numPair     `json:"symbols"`
	}
	cases := make([]oneCase, 0, len(ScaleCases))
	for _, c := range ScaleCases {
		r, err := scale.Design(presets.MustGet(c.Preset), c.Options)
		if err != nil {
			w.err = fmt.Errorf("scale %s: %w", c.Label, err)
			return 0
		}
		entry := oneCase{
			Label: c.Label, Preset: c.Preset, Options: c.Options,
			Achieved: r.Achieved, Target: r.Target,
			Notes: r.Notes, Name: r.Doc.Meta.Name, NotesOnDoc: r.Doc.Meta.Notes,
			Published: r.Doc.Meta.Published, Changes: []change{},
			Symbols: sortedNumbers(ir.ResolveSymbols(r.Doc).DesignValues),
		}
		keys := make([]string, 0, len(r.Changes))
		for k := range r.Changes {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			entry.Changes = append(entry.Changes, change{k, r.Changes[k].From, r.Changes[k].To})
		}
		cases = append(cases, entry)
	}
	w.write("scale.json", struct {
		Cases []oneCase `json:"cases"`
	}{cases})
	return len(cases)
}

// mup records the ladder: the widths, what each one costs, and what to multiply
// the initialization and the learning rate by at each.
//
// The rungs' documents are left out and their symbols recorded instead. A
// document per rung would be most of the file and none of the answer, and the
// symbols are where a width that failed to follow would show.
func (w *writer) mup() int {
	type oneScaling struct {
		Class   string   `json:"class"`
		InitStd string   `json:"initStd"`
		AdamLR  string   `json:"adamLr"`
		Paths   []string `json:"paths"`
	}
	type oneRung struct {
		Width      float64      `json:"width"`
		Multiplier string       `json:"multiplier"`
		Heads      float64      `json:"heads"`
		Params     float64      `json:"params"`
		Base       bool         `json:"base"`
		Scaling    []oneScaling `json:"scaling"`
		Notes      []string     `json:"notes"`
		Symbols    []numPair    `json:"symbols"`
	}
	type oneCase struct {
		Label       string      `json:"label"`
		Preset      string      `json:"preset"`
		Options     mup.Options `json:"options"`
		WidthSymbol string      `json:"widthSymbol"`
		BaseWidth   float64     `json:"baseWidth"`
		HeadDim     float64     `json:"headDim"`
		Rungs       []oneRung   `json:"rungs"`
		Notes       []string    `json:"notes"`
	}
	cases := make([]oneCase, 0, len(MupCases))
	for _, c := range MupCases {
		ladder, err := mup.Build(presets.MustGet(c.Preset), c.Opts)
		if err != nil {
			w.err = fmt.Errorf("mup %s: %w", c.Label, err)
			return 0
		}
		entry := oneCase{
			Label: c.Label, Preset: c.Preset, Options: c.Opts,
			WidthSymbol: ladder.WidthSymbol, BaseWidth: ladder.BaseWidth,
			HeadDim: ladder.HeadDim, Notes: ladder.Notes, Rungs: []oneRung{},
		}
		for _, r := range ladder.Rungs {
			rung := oneRung{
				Width: r.Width, Multiplier: analysis.JSNumber(r.Multiplier), Heads: r.Heads,
				Params: r.Params, Base: r.Base, Notes: r.Notes, Scaling: []oneScaling{},
				Symbols: sortedNumbers(ir.ResolveSymbols(r.Doc).DesignValues),
			}
			for _, s := range r.Scaling {
				// The multipliers are irrational as often as not, so they are
				// recorded the way every other number that crosses the boundary
				// is printed rather than as a raw double.
				rung.Scaling = append(rung.Scaling, oneScaling{
					Class: string(s.Class), InitStd: analysis.JSNumber(s.InitStd),
					AdamLR: analysis.JSNumber(s.AdamLR), Paths: s.Paths,
				})
			}
			entry.Rungs = append(entry.Rungs, rung)
		}
		cases = append(cases, entry)
	}
	w.write("mup.json", struct {
		Cases []oneCase `json:"cases"`
	}{cases})
	return len(cases)
}

// broken rewrites the findings for the designs that are wrong on purpose.
//
// The documents are inputs and stay as they are; only the answers below them
// are recomputed. Eight of the eighteen rules fire only on a mistake, and their
// messages are what a person sees when their design is broken, which makes them
// the most important sentences the engine writes.
func (w *writer) broken() (int, error) {
	var file struct {
		Cases []struct {
			Name     string          `json:"name"`
			Note     string          `json:"note"`
			Doc      json.RawMessage `json:"doc"`
			OK       bool            `json:"ok"`
			Counts   map[string]int  `json:"counts"`
			Findings []goldenFinding `json:"findings"`
		} `json:"cases"`
	}
	if err := w.read("broken.json", &file); err != nil {
		return 0, fmt.Errorf("broken.json: %w", err)
	}

	total := 0
	for i := range file.Cases {
		var doc ir.Doc
		if err := json.Unmarshal(file.Cases[i].Doc, &doc); err != nil {
			return 0, fmt.Errorf("broken %s: %w", file.Cases[i].Name, err)
		}
		report, err := rules.Validate(&doc, analysis.Options{})
		if err != nil {
			return 0, fmt.Errorf("broken %s: %w", file.Cases[i].Name, err)
		}
		file.Cases[i].OK = report.OK
		file.Cases[i].Counts = report.Counts
		file.Cases[i].Findings = findingsOf(report)
		total += len(file.Cases[i].Findings)
	}
	w.write("broken.json", file)
	return total, nil
}

// unusedJSONX keeps the import honest: the writer uses encoding/json directly
// because nothing it writes can be non-finite — a preset whose symbols do not
// evaluate fails the generator rather than being recorded.
var _ = jsonx.Marshal
