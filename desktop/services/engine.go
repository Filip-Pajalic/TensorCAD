package services

import (
	"encoding/json"
	"fmt"

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

// EngineService is the analysis engine, exposed to the window.
//
// Everything crosses as JSON text rather than as generated types. That is
// deliberate: the design document *is* JSON, the reports are JSON, and the
// frontend already has TypeScript types for both. Binding the Go structs
// instead would put a second, generated definition of the same shapes beside
// the hand-written ones, and the two would drift.
//
// The service is stateless. A document arrives with every call, because the
// editor owns the document and this owns none of it.
type EngineService struct{}

// NewEngineService builds the service. There is nothing to wire: it holds no
// state and touches no files.
func NewEngineService() *EngineService { return &EngineService{} }

// ServiceName is what Wails calls this service in logs.
func (s *EngineService) ServiceName() string { return "Engine" }

// Version identifies the engine behind this window, so a frontend can tell
// whether it is talking to the Go engine or running its own.
func (s *EngineService) Version() string { return "go" }

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

// options is the operating point as the editor writes it.
//
// Every field is a pointer, because the difference between "batch of zero" and
// "no batch given" is the difference between an answer and a default.
type options struct {
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
	var o options
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

// Analyze reports every number for a design at one operating point.
func (s *EngineService) Analyze(document string, operatingPoint string) (string, error) {
	doc, err := decodeDoc(document)
	if err != nil {
		return "", err
	}
	opts, err := decodeOptions(operatingPoint)
	if err != nil {
		return "", err
	}
	result, err := analysis.Analyze(doc, opts, analysis.Inputs{})
	if err != nil {
		return "", err
	}
	return encode(result)
}

// Validate runs the design rules and returns the findings with the analysis
// they were drawn from, so a caller that wants both pays for the work once.
func (s *EngineService) Validate(document string, operatingPoint string) (string, error) {
	doc, err := decodeDoc(document)
	if err != nil {
		return "", err
	}
	opts, err := decodeOptions(operatingPoint)
	if err != nil {
		return "", err
	}
	report, err := rules.Validate(doc, opts)
	if err != nil {
		return "", err
	}
	return encode(report)
}

// Explain describes one block: its parameters as written and as evaluated, its
// shapes, its share of the model, and its documentation.
func (s *EngineService) Explain(document string, path string, operatingPoint string) (string, error) {
	doc, err := decodeDoc(document)
	if err != nil {
		return "", err
	}
	opts, err := decodeOptions(operatingPoint)
	if err != nil {
		return "", err
	}
	e, err := explain.Block(doc, path, opts, explain.Inputs{})
	if err != nil {
		return "", err
	}
	return encode(e)
}

// ExplainAll describes every block, largest contribution first.
func (s *EngineService) ExplainAll(document string, operatingPoint string) (string, error) {
	doc, err := decodeDoc(document)
	if err != nil {
		return "", err
	}
	opts, err := decodeOptions(operatingPoint)
	if err != nil {
		return "", err
	}
	all, err := explain.All(doc, opts)
	if err != nil {
		return "", err
	}
	return encode(all)
}

// GenerateTorch emits PyTorch for a design.
func (s *EngineService) GenerateTorch(document string, settings string) (string, error) {
	doc, err := decodeDoc(document)
	if err != nil {
		return "", err
	}
	var o struct {
		ClassName        string   `json:"className"`
		IncludeSmokeTest *bool    `json:"includeSmokeTest"`
		MoeDispatch      string   `json:"moeDispatch"`
		InitStd          *float64 `json:"initStd"`
	}
	if settings != "" {
		if err := json.Unmarshal([]byte(settings), &o); err != nil {
			return "", fmt.Errorf("could not read the generation settings: %w", err)
		}
	}
	opts := codegen.Options{
		ClassName: o.ClassName, MoeDispatch: o.MoeDispatch, InitStd: o.InitStd,
	}
	if o.IncludeSmokeTest != nil && !*o.IncludeSmokeTest {
		opts.NoSmokeTest = true
	}
	return encode(codegen.GenerateTorch(doc, opts))
}

// Scale shrinks a design towards a parameter budget, keeping its proportions.
func (s *EngineService) Scale(document string, settings string) (string, error) {
	doc, err := decodeDoc(document)
	if err != nil {
		return "", err
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
	if settings == "" {
		return "", fmt.Errorf("scaling needs a target parameter count")
	}
	if err := json.Unmarshal([]byte(settings), &o); err != nil {
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

// Presets lists the design library.
func (s *EngineService) Presets() ([]string, error) { return presets.Names() }

// Preset returns one preset's document.
func (s *EngineService) Preset(name string) (string, error) {
	doc, err := presets.Get(name)
	if err != nil {
		return "", err
	}
	return encode(doc)
}

// ImportHuggingFace reads a config.json into a design.
func (s *EngineService) ImportHuggingFace(text string, name string) (string, error) {
	result, err := hf.ImportJSON(text, name)
	if err != nil {
		return "", err
	}
	return encode(result)
}

// CatalogEntry is one block as the palette and the inspector need it.
type CatalogEntry struct {
	Type     string         `json:"type"`
	Kind     string         `json:"kind"`
	Category string         `json:"category"`
	Summary  string         `json:"summary"`
	Formula  string         `json:"formula,omitempty"`
	Refs     []string       `json:"refs,omitempty"`
	Params   []CatalogParam `json:"params"`
}

// CatalogParam is one declared parameter, in the order the block declares it.
type CatalogParam struct {
	Name    string   `json:"name"`
	Type    string   `json:"type"`
	Doc     string   `json:"doc,omitempty"`
	Default any      `json:"default,omitempty"`
	Min     *float64 `json:"min,omitempty"`
	Max     *float64 `json:"max,omitempty"`
	Values  []string `json:"values,omitempty"`
}

// Catalog lists every block the engine knows, for the palette.
func (s *EngineService) Catalog() (string, error) {
	out := make([]CatalogEntry, 0, len(catalog.Builtin))
	for _, def := range allBlocks() {
		entry := CatalogEntry{
			Type: def.Type, Kind: def.Kind, Category: def.Category,
			Summary: def.Docs.Summary, Formula: def.Docs.Formula, Refs: def.Docs.Refs,
			Params: make([]CatalogParam, 0, len(def.Params)),
		}
		for _, p := range def.Params {
			cp := CatalogParam{
				Name: p.Name, Type: string(p.Spec.Type), Doc: p.Spec.Doc,
				Min: p.Spec.Min, Max: p.Spec.Max, Values: p.Spec.Values,
			}
			if p.Spec.HasDefault {
				cp.Default = p.Spec.Default
			}
			entry.Params = append(entry.Params, cp)
		}
		out = append(out, entry)
	}
	return encode(out)
}

// allBlocks lists the catalog in a fixed order: primitives, then composites,
// then containers, each as the engine declares them. A palette that reshuffled
// between launches would be unusable.
func allBlocks() []*catalog.BlockDef {
	out := make([]*catalog.BlockDef, 0, len(catalog.Builtin))
	out = append(out, catalog.Primitives...)
	out = append(out, catalog.Composites...)
	out = append(out, catalog.Containers...)
	return out
}

// Hardware lists the accelerator profiles the analysis can be measured against.
func (s *EngineService) Hardware() (string, error) { return encode(analysis.Hardware) }
