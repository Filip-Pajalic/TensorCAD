package analysis

import (
	"fmt"
	"sort"

	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
)

// Options is the operating point an analysis is measured under.
//
// A nil pointer means "not set", which is the difference between asking for a
// batch of zero and not asking for one at all. The operating point is not part
// of the document: batch, sequence length, dtype, device and sharding are
// conditions the design is measured under, not properties of it.
type Options struct {
	// T is the sequence length; it defaults to the document's runtime T.
	T *float64
	// B is the micro-batch size; it defaults to the document's runtime B.
	B *float64
	// S is the source length of a design with a second sequence, an
	// encoder-decoder's; it defaults to the document's runtime S, and means
	// nothing to a design that does not declare one.
	S *float64
	// Dtype is the training dtype for weights and activations.
	Dtype string
	// InferenceDtype is the serving dtype, often smaller than the training one.
	InferenceDtype string
	// KvDtype is the cache dtype; it defaults to the serving dtype.
	KvDtype string
	// Hardware is a profile id; empty means the default profile.
	Hardware string
	// GPUs is the total number of devices used for training.
	GPUs     *float64
	Parallel *PartialParallel
	// Optimizer names an entry in Optimizers.
	Optimizer string
	// Recompute is "none", "selective" or "full".
	Recompute string
	// Flash assumes a memory-efficient attention kernel.
	Flash *bool
	// Tokens is the training budget; it defaults to the Chinchilla-optimal one.
	Tokens *float64
	// MFU defaults to the midpoint of the device's band.
	MFU *float64
	// DecodeEfficiency is the fraction of peak reached by decode's skinny
	// matmuls.
	DecodeEfficiency *float64
	// Concurrency is the number of concurrent sequences when serving.
	Concurrency *float64
	// Packing is how training rows are filled, for a design whose mask keeps
	// documents apart. Nil is one document a row, which is what serving is.
	Packing *catalog.Packing
}

// PartialParallel overrides part of the default parallel plan.
type PartialParallel struct {
	DP               *float64
	TP               *float64
	PP               *float64
	EP               *float64
	Zero             *int
	SequenceParallel *bool
}

func (p *PartialParallel) apply(plan ParallelPlan) ParallelPlan {
	if p == nil {
		return plan
	}
	if p.DP != nil {
		plan.DP = *p.DP
	}
	if p.TP != nil {
		plan.TP = *p.TP
	}
	if p.PP != nil {
		plan.PP = *p.PP
	}
	if p.EP != nil {
		plan.EP = *p.EP
	}
	if p.Zero != nil {
		plan.Zero = *p.Zero
	}
	if p.SequenceParallel != nil {
		plan.SequenceParallel = *p.SequenceParallel
	}
	return plan
}

// ResolvedOptions is the operating point with every default filled in, so a
// report can say what it assumed.
type ResolvedOptions struct {
	T float64 `json:"T"`
	B float64 `json:"B"`
	// S is the source length, present only for a design with a second
	// sequence.
	S                 float64          `json:"S,omitempty"`
	Dtype             string           `json:"dtype"`
	InferenceDtype    string           `json:"inferenceDtype"`
	KvDtype           string           `json:"kvDtype"`
	Hardware          *HardwareProfile `json:"hardware"`
	GPUs              float64          `json:"gpus"`
	Parallel          ParallelPlan     `json:"parallel"`
	Optimizer         string           `json:"optimizer"`
	Recompute         string           `json:"recompute"`
	Flash             bool             `json:"flash"`
	Tokens            float64          `json:"tokens"`
	TokensWereDefault bool             `json:"tokensWereDefaulted"`
	MFU               float64          `json:"mfu"`
	DecodeEfficiency  float64          `json:"decodeEfficiency"`
	Concurrency       float64          `json:"concurrency"`
	// Packing is present only when the operating point gave one.
	Packing *catalog.Packing `json:"packing,omitempty"`
}

// Result is every number the editor, the CLI and the MCP server report.
type Result struct {
	Name       string            `json:"name"`
	Options    ResolvedOptions   `json:"options"`
	Symbols    *ir.SymbolTable   `json:"symbols"`
	Infer      *infer.Result     `json:"-"`
	Expanded   *infer.Result     `json:"-"`
	Flat       *FlatResult       `json:"-"`
	Params     *ParamsResult     `json:"params"`
	Flops      *FlopsResult      `json:"flops"`
	Kv         *KvResult         `json:"kv"`
	Memory     *MemoryResult     `json:"memory"`
	Throughput *ThroughputResult `json:"throughput"`
	Cost       *CostResult       `json:"cost"`
	Chinchilla *ChinchillaResult `json:"chinchilla"`
	Errors     []string          `json:"errors"`
}

// Inputs are results already computed, so a caller that has them does not pay
// for them twice.
type Inputs struct {
	Symbols *ir.SymbolTable
	Infer   *infer.Result
	Flat    *FlatResult
	// Expanded is shape inference with composites expanded.
	Expanded *infer.Result
}

// streamWidths is the per-token activation width of each repeat container's
// stream, needed to model full activation recomputation. It comes from the
// container's input shape with the runtime dimensions set to one.
func streamWidths(in *infer.Result, symbols *ir.SymbolTable, flat *FlatResult) map[string]float64 {
	out := map[string]float64{}
	env := make(map[string]float64, len(symbols.DesignValues)+len(symbols.Runtime))
	for k, v := range symbols.DesignValues {
		env[k] = v
	}
	for name := range symbols.Runtime {
		env[name] = 1
	}

	for _, rep := range flat.Repeats {
		ports, ok := in.Ports[rep.Path]
		if !ok {
			continue
		}
		names := make([]string, 0, len(ports.In))
		for name := range ports.In {
			names = append(names, name)
		}
		sort.Strings(names)

		width := 0.0
		for _, portName := range names {
			shape, known := in.Inputs[rep.Path+":"+portName]
			if !known {
				continue
			}
			if product, sized := elementsPerToken(shape, env); sized {
				width += product
			}
		}
		if width > 0 {
			out[rep.Path] = width
		}
	}
	return out
}

func orDefault(v *float64, fallback float64) float64 {
	if v == nil {
		return fallback
	}
	return *v
}

func orString(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}

// Analyze produces every number at once.
//
// It is pure and fast enough to re-run on every keystroke: a repeat container
// contributes a multiplier instead of being unrolled, so the 405B preset
// analyses in a couple of milliseconds.
func Analyze(doc *ir.Doc, options Options, pre Inputs) (*Result, error) {
	hardware, err := ResolveHardware(options.Hardware)
	if err != nil {
		return nil, err
	}
	if p := options.Packing; p != nil {
		if !(p.Mean >= 1) {
			return nil, fmt.Errorf("a packing's documents have to be at least one token long on average, not %s",
				JSNumber(p.Mean))
		}
		if !(p.Spread >= 0) {
			return nil, fmt.Errorf("a packing's spread is a coefficient of variation, zero or more, not %s",
				JSNumber(p.Spread))
		}
	}

	symbols := pre.Symbols
	if symbols == nil {
		symbols = ir.ResolveSymbols(doc)
	}
	shapeInfo := pre.Infer
	if shapeInfo == nil {
		shapeInfo = infer.Shapes(doc, symbols, infer.Options{})
	}
	flat := pre.Flat
	if flat == nil {
		flat = Flatten(doc, symbols)
	}
	// Activation memory needs the shape of every tensor inside each composite,
	// so it runs against the fully expanded graph rather than the collapsed one.
	expanded := pre.Expanded
	if expanded == nil {
		expanded = infer.Shapes(doc, symbols, infer.Options{ExpandComposites: true})
	}

	dtype := orString(options.Dtype, "bf16")
	inferenceDtype := orString(options.InferenceDtype, dtype)
	kvDtype := orString(options.KvDtype, inferenceDtype)

	t, b := Sequence(symbols, options)

	parallel := options.Parallel.apply(DefaultParallel)
	recompute := orString(options.Recompute, "none")
	flash := true
	if options.Flash != nil {
		flash = *options.Flash
	}
	optimizer := orString(options.Optimizer, "adamw")
	concurrency := orDefault(options.Concurrency, 1)

	params := CountParams(flat)

	trainCtx := catalog.AnalysisCtx{
		T: t, B: b, Bytes: DtypeBytes[dtype],
		Flash: flash || recompute == "selective",
	}
	cacheCtx := trainCtx
	cacheCtx.Bytes = DtypeBytes[kvDtype]

	source := 0.0
	if symbols.Runtime["S"] {
		source = orDefault(options.S, symbols.Values["S"])
	}
	streams := newStreams(expanded, symbols, t, source)
	// Every block is told the source's length, which is what cross-attention
	// reads all of.
	trainCtx.S, cacheCtx.S = streams.Source, streams.Source

	flops := CountFlops(flat, FlopsOptions{
		Ctx:                trainCtx,
		Recompute:          recompute,
		NonEmbeddingActive: params.NonEmbeddingActive,
		Streams:            streams,
	})
	kv := CountKvCache(flat, cacheCtx, streams)

	// Training under the packing, when there is one and a mask it changes:
	// the same count with every row drawn from it. Everything else stays one
	// document a row, which is what a request is.
	trainFlops := flops.TrainPerToken
	if options.Packing != nil && keepsDocumentsApart(flat) {
		packedCtx := trainCtx
		packedCtx.Packing = options.Packing
		p := CountFlops(flat, FlopsOptions{
			Ctx:                packedCtx,
			Recompute:          recompute,
			NonEmbeddingActive: params.NonEmbeddingActive,
			Streams:            streams,
		})
		flops.Packed = &PackedFlops{
			FwdAttention:       p.FwdAttention,
			FwdTotal:           p.FwdTotal,
			TrainPerToken:      p.TrainPerToken,
			AttentionShare:     p.AttentionShare,
			FwdAttentionBlocks: p.FwdAttentionBlocks,
		}
		trainFlops = p.TrainPerToken
	}

	memory := AnalyzeMemory(flat, MemoryOptions{
		Ctx:                 trainCtx,
		Optimizer:           optimizer,
		Recompute:           recompute,
		Parallel:            parallel,
		InferenceDtypeBytes: DtypeBytes[inferenceDtype],
		Concurrency:         concurrency,
		StreamWidths:        streamWidths(shapeInfo, symbols, flat),
		Expanded:            expanded,
		Symbols:             symbols,
		Params:              params,
		Kv:                  kv,
		Streams:             streams,
	})

	peak := PeakFlops(hardware, dtype)
	mfu := orDefault(options.MFU, (hardware.MFUHint[0]+hardware.MFUHint[1])/2)
	decodeEfficiency := orDefault(options.DecodeEfficiency, 0.3)

	tokensWereDefault := options.Tokens == nil
	tokens := orDefault(options.Tokens, 20*params.NonEmbeddingActive)

	throughput := AnalyzeThroughput(ThroughputOptions{
		Hardware:         hardware,
		Peak:             peak,
		MFU:              mfu,
		DecodeEfficiency: decodeEfficiency,
		Batch:            concurrency,
		Seq:              t,
		// What a step at this batch reads, which for a mixture of experts is
		// more than one token's share and less than every weight.
		DecodeWeightBytes:   StreamedParams(flat, concurrency) * DtypeBytes[inferenceDtype],
		ResidentWeightBytes: params.Total * DtypeBytes[inferenceDtype],
		Kv:                  kv,
		Flops:               flops,
	})

	cost := AnalyzeCost(CostOptions{
		TrainFlopsPerToken: trainFlops,
		Tokens:             tokens,
		GPUs:               orDefault(options.GPUs, 1),
		Peak:               peak,
		MFU:                mfu,
		PricePerHour:       hardware.PricePerHour,
	})

	chinchilla := AnalyzeChinchilla(ChinchillaInputs{
		Total:        params.Total,
		Active:       params.Active,
		NonEmbedding: params.NonEmbeddingActive,
		Tokens:       tokens,
	})

	errors := make([]string, 0,
		len(symbols.Errors)+len(params.Errors)+len(flops.Errors)+len(kv.Errors)+len(memory.Errors))
	errors = append(errors, symbols.Errors...)
	errors = append(errors, params.Errors...)
	errors = append(errors, flops.Errors...)
	errors = append(errors, kv.Errors...)
	errors = append(errors, memory.Errors...)

	return &Result{
		Name: doc.Meta.Name,
		Options: ResolvedOptions{
			T: t, B: b, S: source,
			Dtype: dtype, InferenceDtype: inferenceDtype, KvDtype: kvDtype,
			Hardware: hardware, GPUs: orDefault(options.GPUs, 1),
			Parallel: parallel, Optimizer: optimizer, Recompute: recompute,
			Flash: flash, Tokens: tokens, TokensWereDefault: tokensWereDefault,
			MFU: mfu, DecodeEfficiency: decodeEfficiency, Concurrency: concurrency,
			Packing: options.Packing,
		},
		Symbols:    symbols,
		Infer:      shapeInfo,
		Expanded:   expanded,
		Flat:       flat,
		Params:     params,
		Flops:      flops,
		Kv:         kv,
		Memory:     memory,
		Throughput: throughput,
		Cost:       cost,
		Chinchilla: chinchilla,
		Errors:     errors,
	}, nil
}

// keepsDocumentsApart is whether any attention's mask reads the documents,
// which is the one thing a packing changes the cost of.
func keepsDocumentsApart(flat *FlatResult) bool {
	for i := range flat.Nodes {
		node := &flat.Nodes[i]
		if node.Type == "sdpa" && catalog.AttentionOf(node.Resolved).Documents {
			return true
		}
	}
	return false
}

// Sequence is the sequence length and batch a design is measured at: what the
// operating point says, or the defaults of the design's own T and B.
func Sequence(symbols *ir.SymbolTable, options Options) (T, B float64) {
	T = 2048.0
	if v, ok := symbols.Values["T"]; ok {
		T = v
	}
	T = orDefault(options.T, T)
	B = 1.0
	if v, ok := symbols.Values["B"]; ok {
		B = v
	}
	B = orDefault(options.B, B)
	return T, B
}
