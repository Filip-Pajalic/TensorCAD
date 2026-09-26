//go:build js && wasm

// The engine, compiled for the browser.
//
// One engine everywhere: the editor runs this in the window, in a plain browser
// tab and inside the desktop shell, so there is no second implementation of the
// analysis to keep in step with the first.
//
// Everything crosses as JSON text, the same boundary the desktop service uses.
// The alternative — a JavaScript object per result — would mean building a
// second description of every report in syscall/js calls, and the reports are
// exactly what changes most often.
//
// The calls are synchronous. They run on the same thread the editor draws on,
// which is where the TypeScript engine ran too, and the slowest design in the
// library analyses in a couple of milliseconds.
package main

import (
	"encoding/json"
	"fmt"
	"strconv"
	"syscall/js"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/codegen"
	"github.com/tensorcad/core/diff"
	"github.com/tensorcad/core/explain"
	"github.com/tensorcad/core/hf"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/jsonx"
	"github.com/tensorcad/core/mup"
	"github.com/tensorcad/core/plan"
	"github.com/tensorcad/core/presets"
	"github.com/tensorcad/core/report"
	"github.com/tensorcad/core/rules"
	"github.com/tensorcad/core/scale"
)

// call is one exported entry point: JSON strings in, a JSON string out.
type call func(args []string) (string, error)

func main() {
	api := map[string]any{
		"version":  wrap(0, version),
		"analyze":  wrap(2, analyze),
		"validate": wrap(2, validate),
		// One call for the editor: the findings and the shapes come from the
		// same walk, and asking separately would walk the graph twice per
		// keystroke.
		"derive":  wrap(2, derive),
		"infer":   wrap(2, inferShapes),
		"explain": wrap(3, explainOne),
		// The whole model at once, for the panel that lists every block.
		"explainAll": wrap(2, explainAll),
		// One attention's mask, drawn as the kernel's block mask sees it, at
		// the operating point's sequence length.
		"attentionMask": wrap(4, attentionMask),
		"generateTorch": wrap(2, generateTorch),
		"scale":         wrap(2, scaleDesign),
		// The same design at several widths, with what to multiply the
		// initialization and the learning rate by at each one.
		"mup": wrap(2, mupLadder),
		// Every way of splitting the work across a cluster, and which of them
		// fit. Three arguments rather than two: the design, the operating point
		// it is measured at, and the cluster it is being fitted to.
		"plan": wrap(3, planCluster),
		// What changed between two designs: structure and numbers together,
		// because either alone is misleading.
		"diff":           wrap(3, diffDesigns),
		"presets":        wrap(0, presetNames),
		"preset":         wrap(1, preset),
		"importHf":       wrap(2, importHf),
		"catalog":        wrap(0, blockCatalog),
		"rules":          wrap(0, designRules),
		"checkUserBlock": wrap(2, checkUserBlock),
		"hardware":       wrap(0, hardware),
	}
	js.Global().Set("__tensorcad", js.ValueOf(api))

	// The program has to stay alive to answer calls; returning from main would
	// tear the runtime down under the first one.
	select {}
}

// wrap turns a call into a JavaScript function.
//
// A failure comes back as {error} rather than as a JavaScript exception,
// because a panic crossing the boundary takes the whole runtime with it and
// there is no way to start it again without reloading the page.
func wrap(arity int, fn call) js.Func {
	return js.FuncOf(func(_ js.Value, args []js.Value) any {
		defer func() {
			// Nothing here should panic. If something does, the caller gets a
			// message instead of a dead engine.
			_ = recover()
		}()
		if len(args) < arity {
			return fail(fmt.Errorf("expected %d argument(s), got %d", arity, len(args)))
		}
		strs := make([]string, len(args))
		for i, a := range args {
			if a.Type() == js.TypeString {
				strs[i] = a.String()
			}
		}
		out, err := safely(fn, strs)
		if err != nil {
			return fail(err)
		}
		return out
	})
}

// safely turns a panic inside the engine into an error. The engine does not
// panic on any input the tests cover, but a design is arbitrary data and a
// crashed runtime cannot be restarted without reloading the window.
func safely(fn call, args []string) (out string, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("the engine could not finish: %v", r)
		}
	}()
	return fn(args)
}

func fail(err error) string {
	raw, marshalErr := json.Marshal(map[string]string{"error": err.Error()})
	if marshalErr != nil {
		return `{"error":"the engine could not report what went wrong"}`
	}
	return string(raw)
}

// encode writes a report for the other side of the boundary.
//
// Through jsonx rather than encoding/json, because a design whose symbols fail
// to evaluate produces NaN, and a report that cannot be sent leaves the editor
// blank where it should be showing the error.
func encode(v any) (string, error) {
	raw, err := jsonx.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("could not write the result: %w", err)
	}
	return string(raw), nil
}

func decodeDoc(text string) (*ir.Doc, error) {
	var doc ir.Doc
	if err := json.Unmarshal([]byte(text), &doc); err != nil {
		return nil, fmt.Errorf("could not read the design: %w", err)
	}
	if doc.Version == 0 {
		return nil, fmt.Errorf("the design has no version")
	}
	return &doc, nil
}

// operatingPoint is the point the editor measures a design at. Every field is
// a pointer, because "batch of zero" and "no batch given" are different
// questions.
type operatingPoint struct {
	T                *float64 `json:"T"`
	B                *float64 `json:"B"`
	S                *float64 `json:"S"`
	Dtype            string   `json:"dtype"`
	InferenceDtype   string   `json:"inferenceDtype"`
	KvDtype          string   `json:"kvDtype"`
	Hardware         string   `json:"hardware"`
	GPUs             *float64 `json:"gpus"`
	Optimizer        string   `json:"optimizer"`
	Recompute        string   `json:"recompute"`
	Flash            *bool    `json:"flash"`
	Tokens           *float64 `json:"tokens"`
	MFU              *float64 `json:"mfu"`
	DecodeEfficiency *float64 `json:"decodeEfficiency"`
	Concurrency      *float64 `json:"concurrency"`
	Packing          *struct {
		Mean   *float64 `json:"mean"`
		Spread *float64 `json:"spread"`
	} `json:"packing"`
	Parallel *struct {
		DP               *float64 `json:"dp"`
		TP               *float64 `json:"tp"`
		PP               *float64 `json:"pp"`
		EP               *float64 `json:"ep"`
		Zero             *int     `json:"zero"`
		SequenceParallel *bool    `json:"sequenceParallel"`
	} `json:"parallel"`
}

func decodeOptions(text string) (analysis.Options, error) {
	if text == "" {
		return analysis.Options{}, nil
	}
	var o operatingPoint
	if err := json.Unmarshal([]byte(text), &o); err != nil {
		return analysis.Options{}, fmt.Errorf("could not read the operating point: %w", err)
	}
	out := analysis.Options{
		T: o.T, B: o.B, S: o.S,
		Dtype: o.Dtype, InferenceDtype: o.InferenceDtype, KvDtype: o.KvDtype,
		Hardware: o.Hardware, GPUs: o.GPUs,
		Optimizer: o.Optimizer, Recompute: o.Recompute, Flash: o.Flash,
		Tokens: o.Tokens, MFU: o.MFU,
		DecodeEfficiency: o.DecodeEfficiency, Concurrency: o.Concurrency,
	}
	if o.Packing != nil {
		if o.Packing.Mean == nil {
			return analysis.Options{}, fmt.Errorf("a packing needs a mean document length, packing.mean")
		}
		out.Packing = &catalog.Packing{Mean: *o.Packing.Mean}
		if o.Packing.Spread != nil {
			out.Packing.Spread = *o.Packing.Spread
		}
	}
	if o.Parallel != nil {
		out.Parallel = &analysis.PartialParallel{
			DP: o.Parallel.DP, TP: o.Parallel.TP, PP: o.Parallel.PP, EP: o.Parallel.EP,
			Zero: o.Parallel.Zero, SequenceParallel: o.Parallel.SequenceParallel,
		}
	}
	return out, nil
}

func derive(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	opts, err := decodeOptions(args[1])
	if err != nil {
		return "", err
	}
	out, err := report.Derive(doc, opts)
	if err != nil {
		return "", err
	}
	return encode(out)
}

func inferShapes(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	out, err := report.Infer(doc, args[1] == "expanded")
	if err != nil {
		return "", err
	}
	return encode(out)
}

// ruleList is a design rule as the rules panel lists it.
type ruleList struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
}

func designRules([]string) (string, error) {
	out := make([]ruleList, 0, len(rules.Rules))
	for _, r := range rules.Rules {
		out = append(out, ruleList{ID: r.ID, Title: r.Title, Description: r.Description})
	}
	return encode(out)
}

// checkUserBlock reports what is wrong with a block a design defines for
// itself, so the block editor can say so while it is being written rather than
// after it is saved.
func checkUserBlock(args []string) (string, error) {
	var def any
	if err := json.Unmarshal([]byte(args[0]), &def); err != nil {
		return "", fmt.Errorf("could not read the block definition: %w", err)
	}
	builtIn := map[string]bool{}
	for name := range catalog.Builtin {
		builtIn[name] = true
	}
	problems := catalog.ValidateUserBlock(def, args[1], builtIn)
	if problems == nil {
		problems = []string{}
	}
	return encode(problems)
}

func version([]string) (string, error) {
	return encode(map[string]string{"engine": "go", "target": "wasm"})
}

func analyze(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	opts, err := decodeOptions(args[1])
	if err != nil {
		return "", err
	}
	result, err := analysis.Analyze(doc, opts, analysis.Inputs{})
	if err != nil {
		return "", err
	}
	return encode(result)
}

func validate(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	opts, err := decodeOptions(args[1])
	if err != nil {
		return "", err
	}
	report, err := rules.Validate(doc, opts)
	if err != nil {
		return "", err
	}
	return encode(report)
}

func explainOne(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	opts, err := decodeOptions(args[2])
	if err != nil {
		return "", err
	}
	e, err := explain.Block(doc, args[1], opts, explain.Inputs{})
	if err != nil {
		return "", err
	}
	return encode(e)
}

func attentionMask(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	opts, err := decodeOptions(args[2])
	if err != nil {
		return "", err
	}
	head, err := strconv.ParseFloat(args[3], 64)
	if err != nil {
		head = 0
	}
	return encode(explain.Mask(doc, args[1], opts, head))
}

func explainAll(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	opts, err := decodeOptions(args[1])
	if err != nil {
		return "", err
	}
	all, err := explain.All(doc, opts)
	if err != nil {
		return "", err
	}
	return encode(all)
}

func generateTorch(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	var o codegen.WireOptions
	if args[1] != "" {
		if err := json.Unmarshal([]byte(args[1]), &o); err != nil {
			return "", fmt.Errorf("could not read the generation settings: %w", err)
		}
	}
	return encode(codegen.GenerateTorch(doc, o.Options()))
}

func scaleDesign(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	if args[1] == "" {
		return "", fmt.Errorf("scaling needs a target parameter count")
	}
	// Straight into scale.Options: its json tags are the wire contract, so a
	// copy here would only be a second place for a field to go missing.
	var o scale.Options
	if err := json.Unmarshal([]byte(args[1]), &o); err != nil {
		return "", fmt.Errorf("could not read the scaling settings: %w", err)
	}
	result, err := scale.Design(doc, o)
	if err != nil {
		return "", err
	}
	return encode(result)
}

func mupLadder(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	// Straight into mup.Options, for the same reason scaling does: its json
	// tags are the wire contract.
	var o mup.Options
	if args[1] != "" {
		if err := json.Unmarshal([]byte(args[1]), &o); err != nil {
			return "", fmt.Errorf("could not read the ladder settings: %w", err)
		}
	}
	ladder, err := mup.Build(doc, o)
	if err != nil {
		return "", err
	}
	return encode(ladder)
}

func planCluster(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	options, err := decodeOptions(args[1])
	if err != nil {
		return "", err
	}
	if args[2] == "" {
		return "", fmt.Errorf("planning needs to know how many GPUs there are")
	}
	var req plan.Request
	if err := json.Unmarshal([]byte(args[2]), &req); err != nil {
		return "", fmt.Errorf("could not read the cluster: %w", err)
	}
	result, err := plan.Search(doc, options, req)
	if err != nil {
		return "", err
	}
	return encode(result)
}

func diffDesigns(args []string) (string, error) {
	a, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	b, err := decodeDoc(args[1])
	if err != nil {
		return "", err
	}
	options, err := decodeOptions(args[2])
	if err != nil {
		return "", err
	}
	result, err := diff.Designs(a, b, options)
	if err != nil {
		return "", err
	}
	return encode(result)
}

func presetNames([]string) (string, error) {
	names, err := presets.Names()
	if err != nil {
		return "", err
	}
	return encode(names)
}

func preset(args []string) (string, error) {
	doc, err := presets.Get(args[0])
	if err != nil {
		return "", err
	}
	return encode(doc)
}

func importHf(args []string) (string, error) {
	result, err := hf.ImportJSON(args[0], args[1])
	if err != nil {
		return "", err
	}
	return encode(result)
}

// catalogEntry is one block as the palette and the inspector need it.
//
// Parameters by name with the order beside them, rather than a list of named
// parameters: a panel looks one up far more often than it walks them all, and
// the order is what the inspector lays its fields out in.
type catalogEntry struct {
	Type     string                  `json:"type"`
	Kind     string                  `json:"kind"`
	Category string                  `json:"category"`
	Docs     catalog.BlockDocs       `json:"docs"`
	Params   map[string]catalogParam `json:"params"`
	// ParamOrder is the order the block declares them in.
	ParamOrder []string     `json:"paramOrder"`
	Ports      catalogPorts `json:"ports"`
}

type catalogParam struct {
	Type    string   `json:"type"`
	Doc     string   `json:"doc,omitempty"`
	Default any      `json:"default,omitempty"`
	Min     *float64 `json:"min,omitempty"`
	Max     *float64 `json:"max,omitempty"`
	Values  []string `json:"values,omitempty"`
	// Group is the heading the inspector lays this field out under.
	Group string `json:"group,omitempty"`
	// When is the condition under which the field means anything.
	When *catalog.ParamWhen `json:"when,omitempty"`
	// Label is what the inspector leads with; the name is beside it.
	Label string `json:"label,omitempty"`
	// Advanced puts the field under a heading that starts closed.
	Advanced bool `json:"advanced,omitempty"`
	// ValueLabels are what an enum's values are called, by value.
	ValueLabels map[string]string `json:"valueLabels,omitempty"`
}

// catalogPorts are the pins a block declares before any parameter is known, so
// a block whose pins depend on its parameters reports none here.
type catalogPorts struct {
	In  map[string]string `json:"in"`
	Out map[string]string `json:"out"`
}

func blockCatalog([]string) (string, error) {
	defs := make([]*catalog.BlockDef, 0, len(catalog.Builtin))
	defs = append(defs, catalog.Primitives...)
	defs = append(defs, catalog.Composites...)
	defs = append(defs, catalog.Containers...)

	out := make([]catalogEntry, 0, len(defs))
	for _, def := range defs {
		entry := catalogEntry{
			Type: def.Type, Kind: def.Kind, Category: def.Category, Docs: def.Docs,
			Params:     make(map[string]catalogParam, len(def.Params)),
			ParamOrder: make([]string, 0, len(def.Params)),
			Ports:      catalogPorts{In: map[string]string{}, Out: map[string]string{}},
		}
		for _, p := range def.Params {
			cp := catalogParam{
				Type: string(p.Spec.Type), Doc: p.Spec.Doc,
				Min: p.Spec.Min, Max: p.Spec.Max, Values: p.Spec.Values,
				Group: p.Spec.Group, When: p.Spec.When,
				Label: p.Spec.Label, Advanced: p.Spec.Advanced, ValueLabels: p.Spec.ValueLabels,
			}
			if p.Spec.HasDefault {
				cp.Default = p.Spec.Default
			}
			entry.Params[p.Name] = cp
			entry.ParamOrder = append(entry.ParamOrder, p.Name)
		}
		for name, port := range def.Ports.In {
			entry.Ports.In[name] = port.Shape
		}
		for name, port := range def.Ports.Out {
			entry.Ports.Out[name] = port.Shape
		}
		out = append(out, entry)
	}
	return encode(out)
}

func hardware([]string) (string, error) { return encode(analysis.Hardware) }
