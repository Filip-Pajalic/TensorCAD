package analysis

import (
	"fmt"
	"math"
)

// ThroughputResult is how fast the design runs.
//
// The model is a roofline: a decode step moves the active weights and the cache
// through memory and does a little arithmetic on them, so it is memory-bound
// until the batch exceeds the device's ridge point.
//
// Reference: kipply's transformer inference arithmetic.
type ThroughputResult struct {
	// RidgePoint is the FLOP per byte above which the device is compute-bound.
	RidgePoint float64 `json:"ridgePoint"`
	// DecodeBytesPerStep is the bytes read per decode step.
	DecodeBytesPerStep float64 `json:"decodeBytesPerStep"`
	// DecodeWeightBytes is the weights of that, which for a mixture of experts
	// is neither the active count nor the resident one: a batch reads the union
	// of what its tokens routed to.
	DecodeWeightBytes float64 `json:"decodeWeightBytes"`
	// ResidentWeightBytes is every weight the device has to hold, whether or
	// not a given step reads it.
	ResidentWeightBytes float64 `json:"residentWeightBytes"`
	DecodeFlopsPerStep  float64 `json:"decodeFlopsPerStep"`
	// DecodeSecondsPerStep is the larger of the memory and compute times.
	DecodeSecondsPerStep  float64 `json:"decodeSecondsPerStep"`
	DecodeTokensPerSecond float64 `json:"decodeTokensPerSecond"`
	// MemoryBound is true when bandwidth, not arithmetic, sets the pace.
	MemoryBound bool `json:"memoryBound"`
	// PrefillSeconds is the time to prefill one full context for the batch.
	PrefillSeconds float64  `json:"prefillSeconds"`
	Notes          []string `json:"notes"`
}

// ThroughputOptions is what the roofline needs.
type ThroughputOptions struct {
	Hardware *HardwareProfile
	Peak     float64
	// MFU is the fraction of peak achieved during prefill.
	MFU float64
	// DecodeEfficiency is the fraction of peak reached by decode's small
	// matmuls.
	DecodeEfficiency float64
	Batch            float64
	Seq              float64
	// DecodeWeightBytes is what a step at this batch reads, from
	// `StreamedParams`. For a dense model it is every weight.
	DecodeWeightBytes float64
	// ResidentWeightBytes is every weight, which is what has to be held.
	ResidentWeightBytes float64
	Kv                  *KvResult
	Flops               *FlopsResult
}

// AnalyzeThroughput runs the roofline.
func AnalyzeThroughput(o ThroughputOptions) *ThroughputResult {
	notes := []string{}
	ridgePoint := o.Peak / o.Hardware.Bandwidth

	kvBytes := KvBytesFor(o.Kv, o.Seq, o.Batch)
	decodeBytesPerStep := o.DecodeWeightBytes + kvBytes
	decodeFlopsPerStep := o.Batch * o.Flops.FwdTotal

	tMem := decodeBytesPerStep / o.Hardware.Bandwidth
	tCompute := decodeFlopsPerStep / (o.Peak * o.DecodeEfficiency)
	step := math.Max(tMem, tCompute)
	memoryBound := tMem >= tCompute

	if memoryBound {
		notes = append(notes, fmt.Sprintf(
			"Decoding is memory-bound at batch %s. It stays that way until the batch passes this device's ridge point of about %s FLOP per byte.",
			JSNumber(o.Batch), JSNumber(math.Round(ridgePoint))))
	} else {
		notes = append(notes, fmt.Sprintf("Decoding is compute-bound at batch %s.", JSNumber(o.Batch)))
	}

	// A sparse model reads more than one token's share and less than all of it,
	// and which end it is near is the difference between a plausible decode
	// figure and a fanciful one.
	if o.ResidentWeightBytes > o.DecodeWeightBytes {
		if o.Batch <= 1 {
			notes = append(notes, fmt.Sprintf(
				"One token routes to %s of the %s of weights held, so that is what a step reads at this "+
					"batch. A larger batch reads the union of its tokens' choices, which reaches nearly "+
					"every expert long before the batch reaches the expert count.",
				FormatBytes(o.DecodeWeightBytes), FormatBytes(o.ResidentWeightBytes)))
		} else {
			notes = append(notes, fmt.Sprintf(
				"A step at batch %s reads %s of the %s of weights held, because its tokens between them "+
					"route to that much: a batch reads the union of their choices rather than one token's "+
					"share. Routing is taken to be independent and uniform, and a router trained towards "+
					"balance spreads a batch wider still, so this is a floor and the rate above it a ceiling.",
				JSNumber(o.Batch), FormatBytes(o.DecodeWeightBytes), FormatBytes(o.ResidentWeightBytes)))
		}
	}

	prefillSeconds := (o.Batch * o.Seq * o.Flops.FwdTotal) / (o.Peak * o.MFU)

	tokensPerSecond := 0.0
	if step > 0 {
		tokensPerSecond = o.Batch / step
	}

	return &ThroughputResult{
		RidgePoint:            ridgePoint,
		DecodeBytesPerStep:    decodeBytesPerStep,
		DecodeWeightBytes:     o.DecodeWeightBytes,
		ResidentWeightBytes:   o.ResidentWeightBytes,
		DecodeFlopsPerStep:    decodeFlopsPerStep,
		DecodeSecondsPerStep:  step,
		DecodeTokensPerSecond: tokensPerSecond,
		MemoryBound:           memoryBound,
		PrefillSeconds:        prefillSeconds,
		Notes:                 notes,
	}
}

// CostOptions is what a training run is priced from.
type CostOptions struct {
	TrainFlopsPerToken float64
	Tokens             float64
	GPUs               float64
	Peak               float64
	MFU                float64
	PricePerHour       float64
}

// CostResult is what a training run would cost.
type CostResult struct {
	TotalFlops     float64 `json:"totalFlops"`
	GpuHours       float64 `json:"gpuHours"`
	WallClockHours float64 `json:"wallClockHours"`
	Dollars        float64 `json:"dollars"`
	Tokens         float64 `json:"tokens"`
	MFU            float64 `json:"mfu"`
}

// AnalyzeCost prices a training run.
func AnalyzeCost(o CostOptions) *CostResult {
	totalFlops := o.TrainFlopsPerToken * o.Tokens
	gpuHours := totalFlops / (o.Peak * o.MFU * 3600)
	return &CostResult{
		TotalFlops:     totalFlops,
		GpuHours:       gpuHours,
		WallClockHours: gpuHours / math.Max(1, o.GPUs),
		Dollars:        gpuHours * o.PricePerHour,
		Tokens:         o.Tokens,
		MFU:            o.MFU,
	}
}

// ---------------------------------------------------------------------------
// Scaling laws
// ---------------------------------------------------------------------------

// ScalingLawFit is one published fit of loss against parameters and tokens.
type ScalingLawFit struct {
	Name   string  `json:"name"`
	E      float64 `json:"E"`
	A      float64 `json:"A"`
	B      float64 `json:"B"`
	Alpha  float64 `json:"alpha"`
	Beta   float64 `json:"beta"`
	Source string  `json:"source"`
}

// ChinchillaFits are the fits the analysis predicts loss under.
var ChinchillaFits = map[string]ScalingLawFit{
	"hoffmann": {
		Name: "Hoffmann et al. 2022",
		E:    1.69, A: 406.4, B: 410.7, Alpha: 0.34, Beta: 0.28,
		Source: "https://arxiv.org/abs/2203.15556",
	},
	"epoch": {
		Name: "Epoch AI replication",
		E:    1.8172, A: 482.01, B: 2085.43, Alpha: 0.3478, Beta: 0.3658,
		Source: "https://epoch.ai/blog/chinchilla-scaling-a-replication-attempt",
	},
}

// ChinchillaResult is how the token budget compares to compute-optimal.
type ChinchillaResult struct {
	// OptimalTokens is the compute-optimal budget, about 20 per parameter.
	OptimalTokens float64 `json:"optimalTokens"`
	// TokensPerParam is the ratio for the budget actually chosen.
	TokensPerParam float64 `json:"tokensPerParam"`
	// TokensPerActiveParam is the meaningful ratio for a sparse model.
	TokensPerActiveParam float64 `json:"tokensPerActiveParam"`
	// OverTrainingRatio is how far the chosen budget is from compute-optimal.
	OverTrainingRatio float64 `json:"overTrainingRatio"`
	// PredictedLoss is the loss under each fit, when a token budget was given.
	PredictedLoss map[string]float64 `json:"predictedLoss"`
	Verdict       string             `json:"verdict"`
}

// ChinchillaInputs is the design as the scaling law sees it.
type ChinchillaInputs struct {
	Total        float64
	Active       float64
	NonEmbedding float64
	Tokens       float64
}

// AnalyzeChinchilla compares the token budget against the scaling laws.
//
// References: Hoffmann et al. (Chinchilla) and Epoch's refit.
func AnalyzeChinchilla(p ChinchillaInputs) *ChinchillaResult {
	n := p.NonEmbedding
	optimalTokens := 20 * n

	res := &ChinchillaResult{
		OptimalTokens: optimalTokens,
		PredictedLoss: map[string]float64{},
	}
	if p.Total > 0 {
		res.TokensPerParam = p.Tokens / p.Total
	}
	if p.Active > 0 {
		res.TokensPerActiveParam = p.Tokens / p.Active
	}
	if optimalTokens > 0 {
		res.OverTrainingRatio = p.Tokens / optimalTokens
	}

	if p.Tokens > 0 && n > 0 {
		for key, f := range ChinchillaFits {
			res.PredictedLoss[key] = f.E + f.A/math.Pow(n, f.Alpha) + f.B/math.Pow(p.Tokens, f.Beta)
		}
	}

	switch {
	case p.Tokens <= 0:
		res.Verdict = "No token budget given."
	case res.OverTrainingRatio < 0.5:
		res.Verdict = "Under-trained relative to Chinchilla: the model is larger than the data supports."
	case res.OverTrainingRatio <= 2:
		res.Verdict = "Close to compute-optimal."
	case res.OverTrainingRatio <= 20:
		res.Verdict = "Over-trained, which is the norm for models meant to be served."
	default:
		res.Verdict = "Heavily over-trained, in the range of small models trained on very large corpora."
	}
	return res
}

// FormatHours is a duration as a person reads it: 45.0 min, 12.5 h, 3.2 days.
func FormatHours(h float64) string {
	switch {
	case h < 1:
		return JSToFixed(h*60, 1) + " min"
	case h < 48:
		return JSToFixed(h, 1) + " h"
	}
	return JSToFixed(h/24, 1) + " days"
}

// FormatDollars is a price as a person reads it: $1.20M, $4.5k, $12.34.
func FormatDollars(d float64) string {
	switch {
	case d >= 1e6:
		return "$" + JSToFixed(d/1e6, 2) + "M"
	case d >= 1e3:
		return "$" + JSToFixed(d/1e3, 1) + "k"
	}
	return "$" + JSToFixed(d, 2)
}
