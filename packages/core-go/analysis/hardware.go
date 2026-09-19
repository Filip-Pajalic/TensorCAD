// Package analysis turns a design into every number the editor, the CLI and
// the MCP server report: parameters, FLOPs, memory, cache, throughput and cost.
//
// It is pure and fast enough to re-run on every keystroke, because a repeat
// container contributes a multiplier rather than being unrolled: a 126-layer
// model flattens to a few dozen nodes.
package analysis

import "fmt"

// HardwareProfile is one accelerator.
//
// Peak numbers are dense (no structured sparsity) tensor-core throughput from
// vendor spec sheets, and prices are rough market rates. Both are approximate
// and meant to be overridden: a starting point, not a measurement. MFUHint is a
// realistic model-FLOPs-utilization band for training on that part, drawn from
// published runs where one exists.
type HardwareProfile struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	// PeakBf16 is dense BF16/FP16 tensor throughput, FLOP/s.
	PeakBf16 float64 `json:"peakBf16"`
	// PeakFp8 is dense FP8 tensor throughput, FLOP/s; zero when unsupported.
	PeakFp8 float64 `json:"peakFp8"`
	// Memory is device memory in bytes.
	Memory float64 `json:"memory"`
	// Bandwidth is device memory bandwidth in bytes/s.
	Bandwidth float64 `json:"bandwidth"`
	// PricePerHour is an indicative rental price in US dollars per GPU-hour.
	PricePerHour float64    `json:"pricePerHour"`
	MFUHint      [2]float64 `json:"mfuHint"`
	Notes        string     `json:"notes,omitempty"`
}

const (
	tera = 1e12
	gib  = 1024 * 1024 * 1024
	gbs  = 1e9
)

// Hardware is every profile the engine ships with, in menu order.
var Hardware = []HardwareProfile{
	{
		ID:       "rtx5080",
		Name:     "GeForce RTX 5080 (16 GB)",
		PeakBf16: 225 * tera, PeakFp8: 450 * tera,
		Memory: 16 * gib, Bandwidth: 960 * gbs,
		PricePerHour: 0.2, MFUHint: [2]float64{0.2, 0.35},
		Notes: "Blackwell consumer part. Approximate spec-sheet values; no NVLink, so multi-GPU scaling is poor.",
	},
	{
		ID:       "rtx4090",
		Name:     "GeForce RTX 4090 (24 GB)",
		PeakBf16: 165 * tera, PeakFp8: 330 * tera,
		Memory: 24 * gib, Bandwidth: 1008 * gbs,
		PricePerHour: 0.35, MFUHint: [2]float64{0.2, 0.35},
	},
	{
		ID:       "a100-80",
		Name:     "A100 SXM (80 GB)",
		PeakBf16: 312 * tera, PeakFp8: 0,
		Memory: 80 * gib, Bandwidth: 2039 * gbs,
		PricePerHour: 1.6, MFUHint: [2]float64{0.3, 0.45},
	},
	{
		ID:       "h100-sxm",
		Name:     "H100 SXM (80 GB)",
		PeakBf16: 989 * tera, PeakFp8: 1979 * tera,
		Memory: 80 * gib, Bandwidth: 3350 * gbs,
		PricePerHour: 2.5, MFUHint: [2]float64{0.35, 0.45},
		Notes: "Llama 3 405B reported 38-43% BF16 MFU on H100 clusters.",
	},
	{
		ID:       "h200-sxm",
		Name:     "H200 SXM (141 GB)",
		PeakBf16: 989 * tera, PeakFp8: 1979 * tera,
		Memory: 141 * gib, Bandwidth: 4800 * gbs,
		PricePerHour: 3.2, MFUHint: [2]float64{0.35, 0.45},
	},
	{
		ID:       "b200",
		Name:     "B200 SXM (192 GB)",
		PeakBf16: 2250 * tera, PeakFp8: 4500 * tera,
		Memory: 192 * gib, Bandwidth: 8000 * gbs,
		PricePerHour: 5.5, MFUHint: [2]float64{0.3, 0.45},
		Notes: "Approximate; dense Blackwell datacenter throughput.",
	},
}

// HardwareByID indexes Hardware by its id.
var HardwareByID = func() map[string]*HardwareProfile {
	m := make(map[string]*HardwareProfile, len(Hardware))
	for i := range Hardware {
		m[Hardware[i].ID] = &Hardware[i]
	}
	return m
}()

// DefaultHardware is what an analysis assumes when nothing says otherwise.
const DefaultHardware = "h100-sxm"

// ResolveHardware looks a profile up by id, defaulting when none is named.
func ResolveHardware(id string) (*HardwareProfile, error) {
	if id == "" {
		return HardwareByID[DefaultHardware], nil
	}
	hit, ok := HardwareByID[id]
	if !ok {
		known := ""
		for i, h := range Hardware {
			if i > 0 {
				known += ", "
			}
			known += h.ID
		}
		return nil, fmt.Errorf("unknown hardware profile %q. Known: %s", id, known)
	}
	return hit, nil
}

// DtypeBytes is the width of each supported element type.
var DtypeBytes = map[string]float64{
	"fp32": 4,
	"bf16": 2,
	"fp16": 2,
	"fp8":  1,
}

// PeakFlops is the throughput for a dtype, falling back to BF16 where FP8 is
// unsupported.
func PeakFlops(hw *HardwareProfile, dtype string) float64 {
	if dtype == "fp8" && hw.PeakFp8 > 0 {
		return hw.PeakFp8
	}
	return hw.PeakBf16
}

// OptimizerBytes is what the training state holds per parameter, split so each
// part can be sharded independently. Mixed-precision AdamW is the classic
// 2 + 2 + 4 + 4 + 4 = 16 bytes per parameter.
type OptimizerBytes struct {
	Weights   float64
	Grads     float64
	Optimizer float64
	Label     string
}

// Optimizers is every optimizer the memory model knows.
var Optimizers = map[string]OptimizerBytes{
	"adamw":        {2, 2, 12, "AdamW, mixed precision (16 B/param)"},
	"adamw8bit":    {2, 2, 6, "8-bit AdamW (10 B/param)"},
	"muon":         {2, 2, 8, "Muon with fp32 master weights (12 B/param)"},
	"sgd_momentum": {2, 2, 8, "SGD with momentum (12 B/param)"},
	"sgd":          {2, 2, 4, "SGD (8 B/param)"},
	"bf16_adam":    {2, 2, 4, "Pure bf16 Adam, no master weights (8 B/param)"},
}
