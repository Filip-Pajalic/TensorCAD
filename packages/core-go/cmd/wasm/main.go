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
	"syscall/js"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/codegen"
	"github.com/tensorcad/core/explain"
	"github.com/tensorcad/core/hf"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/jsonx"
	"github.com/tensorcad/core/presets"
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
		"explain":  wrap(3, explainOne),
		// The whole model at once, for the panel that lists every block.
		"explainAll":    wrap(2, explainAll),
		"generateTorch": wrap(2, generateTorch),
		"scale":         wrap(2, scaleDesign),
		"presets":       wrap(0, presetNames),
		"preset":        wrap(1, preset),
		"importHf":      wrap(2, importHf),
		"catalog":       wrap(0, blockCatalog),
		"hardware":      wrap(0, hardware),
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
	Parallel         *struct {
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
		T: o.T, B: o.B,
		Dtype: o.Dtype, InferenceDtype: o.InferenceDtype, KvDtype: o.KvDtype,
		Hardware: o.Hardware, GPUs: o.GPUs,
		Optimizer: o.Optimizer, Recompute: o.Recompute, Flash: o.Flash,
		Tokens: o.Tokens, MFU: o.MFU,
		DecodeEfficiency: o.DecodeEfficiency, Concurrency: o.Concurrency,
	}
	if o.Parallel != nil {
		out.Parallel = &analysis.PartialParallel{
			DP: o.Parallel.DP, TP: o.Parallel.TP, PP: o.Parallel.PP, EP: o.Parallel.EP,
			Zero: o.Parallel.Zero, SequenceParallel: o.Parallel.SequenceParallel,
		}
	}
	return out, nil
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
	var o struct {
		ClassName        string   `json:"className"`
		IncludeSmokeTest *bool    `json:"includeSmokeTest"`
		MoeDispatch      string   `json:"moeDispatch"`
		InitStd          *float64 `json:"initStd"`
	}
	if args[1] != "" {
		if err := json.Unmarshal([]byte(args[1]), &o); err != nil {
			return "", fmt.Errorf("could not read the generation settings: %w", err)
		}
	}
	opts := codegen.Options{ClassName: o.ClassName, MoeDispatch: o.MoeDispatch, InitStd: o.InitStd}
	if o.IncludeSmokeTest != nil && !*o.IncludeSmokeTest {
		opts.NoSmokeTest = true
	}
	return encode(codegen.GenerateTorch(doc, opts))
}

func scaleDesign(args []string) (string, error) {
	doc, err := decodeDoc(args[0])
	if err != nil {
		return "", err
	}
	if args[1] == "" {
		return "", fmt.Errorf("scaling needs a target parameter count")
	}
	var o struct {
		TargetParams  float64  `json:"targetParams"`
		WidthSymbols  []string `json:"widthSymbols"`
		DepthSymbols  []string `json:"depthSymbols"`
		WidthMultiple *float64 `json:"widthMultiple"`
		Vocab         *float64 `json:"vocab"`
		TargetBasis   string   `json:"targetBasis"`
		TieHead       *bool    `json:"tieHead"`
		MinHeads      *float64 `json:"minHeads"`
		KeepDepth     bool     `json:"keepDepth"`
		MaxIterations *int     `json:"maxIterations"`
	}
	if err := json.Unmarshal([]byte(args[1]), &o); err != nil {
		return "", fmt.Errorf("could not read the scaling settings: %w", err)
	}
	result, err := scale.Design(doc, scale.Options{
		TargetParams: o.TargetParams,
		WidthSymbols: o.WidthSymbols, DepthSymbols: o.DepthSymbols,
		WidthMultiple: o.WidthMultiple, Vocab: o.Vocab,
		TargetBasis: o.TargetBasis, TieHead: o.TieHead, MinHeads: o.MinHeads,
		KeepDepth: o.KeepDepth, MaxIterations: o.MaxIterations,
	})
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
type catalogEntry struct {
	Type     string         `json:"type"`
	Kind     string         `json:"kind"`
	Category string         `json:"category"`
	Summary  string         `json:"summary"`
	Formula  string         `json:"formula,omitempty"`
	Refs     []string       `json:"refs,omitempty"`
	Params   []catalogParam `json:"params"`
	Ports    catalogPorts   `json:"ports"`
}

type catalogParam struct {
	Name    string   `json:"name"`
	Type    string   `json:"type"`
	Doc     string   `json:"doc,omitempty"`
	Default any      `json:"default,omitempty"`
	Min     *float64 `json:"min,omitempty"`
	Max     *float64 `json:"max,omitempty"`
	Values  []string `json:"values,omitempty"`
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
			Type: def.Type, Kind: def.Kind, Category: def.Category,
			Summary: def.Docs.Summary, Formula: def.Docs.Formula, Refs: def.Docs.Refs,
			Params: make([]catalogParam, 0, len(def.Params)),
			Ports:  catalogPorts{In: map[string]string{}, Out: map[string]string{}},
		}
		for _, p := range def.Params {
			cp := catalogParam{
				Name: p.Name, Type: string(p.Spec.Type), Doc: p.Spec.Doc,
				Min: p.Spec.Min, Max: p.Spec.Max, Values: p.Spec.Values,
			}
			if p.Spec.HasDefault {
				cp.Default = p.Spec.Default
			}
			entry.Params = append(entry.Params, cp)
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
