// Package plan searches for a way to train a design on a cluster.
//
// The analysis prices one parallelism plan: give it a design, a device and a
// way of splitting the work, and it says how many bytes land on each GPU.
// "Will this train on sixty-four H100s, and how" is the question people
// actually have, and answering it by hand means trying the combinations one at
// a time. This tries them all.
//
// What it claims is memory, which is arithmetic, and it reports the plans that
// fit in the order of how little they ask of you. What it does not claim is
// which is fastest: that turns on interconnect topology, kernel
// implementations and the shape of the communication schedule, none of which a
// parameter count knows. Each plan instead carries a note saying what it costs
// to run, so the choice among the ones that fit stays with the person making
// it.
package plan

import (
	"fmt"
	"math"
	"sort"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
)

// Request is the cluster, and how much of it may be spent.
type Request struct {
	// GPUs is how many devices there are. Required.
	GPUs float64 `json:"gpus"`
	// GPUsPerNode bounds the tensor-parallel degree, because splitting a matrix
	// across a slower link than NVLink is rarely worth it. Zero means eight.
	GPUsPerNode float64 `json:"gpusPerNode,omitempty"`
	// Headroom is the fraction of device memory left free for fragmentation,
	// the allocator and the communication buffers. Zero means a tenth.
	Headroom float64 `json:"headroom,omitempty"`
	// MicroBatch is the micro-batch sizes to try. Empty means the one the
	// analysis options give.
	MicroBatch []float64 `json:"microBatch,omitempty"`
	// Recompute is the settings to try. Empty means all three.
	Recompute []string `json:"recompute,omitempty"`
	// Limit is how many plans to return. Zero means eight.
	Limit int `json:"limit,omitempty"`
}

// Candidate is one way of splitting the work, and what it costs to hold.
type Candidate struct {
	Parallel   analysis.ParallelPlan `json:"parallel"`
	Recompute  string                `json:"recompute"`
	MicroBatch float64               `json:"microBatch"`
	// PerGpu is the training footprint on one device.
	PerGpu analysis.TrainPerGpu `json:"perGpu"`
	// Used is the fraction of the budget this plan takes; over 1 does not fit.
	Used float64 `json:"used"`
	// Summary reads the way a person would say it out loud.
	Summary string `json:"summary"`
	// Notes are what this plan asks of whoever runs it.
	Notes []string `json:"notes"`
}

// Result is every plan that fits, and the nearest miss when none do.
type Result struct {
	// Fits are the plans that fit, least demanding first.
	Fits []Candidate `json:"fits"`
	// Closest is the plan that came nearest when nothing fit, so the gap can be
	// read rather than guessed at.
	Closest *Candidate `json:"closest,omitempty"`
	// Considered is how many combinations were priced.
	Considered int `json:"considered"`
	// Budget is the bytes each device may use, after headroom.
	Budget float64 `json:"budget"`
	// Memory is the device's own memory, before headroom.
	Memory   float64  `json:"memory"`
	Hardware string   `json:"hardware"`
	Notes    []string `json:"notes"`
}

// Search prices every plan the cluster admits and returns those that fit.
func Search(doc *ir.Doc, options analysis.Options, req Request) (*Result, error) {
	if req.GPUs < 1 || math.Trunc(req.GPUs) != req.GPUs {
		return nil, fmt.Errorf("a cluster needs a whole number of GPUs, not %v", req.GPUs)
	}
	perNode := req.GPUsPerNode
	if perNode <= 0 {
		perNode = 8
	}
	headroom := req.Headroom
	if headroom <= 0 {
		headroom = 0.1
	}
	if headroom >= 1 {
		return nil, fmt.Errorf("headroom is a fraction of device memory, not %v", headroom)
	}
	limit := req.Limit
	if limit <= 0 {
		limit = 8
	}
	recomputes := req.Recompute
	if len(recomputes) == 0 {
		recomputes = []string{"none", "selective", "full"}
	}

	hardware, err := analysis.ResolveHardware(options.Hardware)
	if err != nil {
		return nil, err
	}
	out := &Result{
		Fits: []Candidate{}, Notes: []string{},
		Memory: hardware.Memory, Budget: hardware.Memory * (1 - headroom),
		Hardware: hardware.Name,
	}

	// The design is flattened once. Only the memory model depends on the plan,
	// and walking the graph a few thousand times to learn the same thing is
	// the whole reason this was slow enough to do by hand.
	symbols := ir.ResolveSymbols(doc)
	flat := analysis.Flatten(doc, symbols)
	params := analysis.CountParams(flat)
	shapes := infer.Shapes(doc, symbols, infer.Options{})
	expanded := infer.Shapes(doc, symbols, infer.Options{ExpandComposites: true})
	pre := analysis.Inputs{Symbols: symbols, Flat: flat, Infer: shapes, Expanded: expanded}

	batches := req.MicroBatch
	if len(batches) == 0 {
		batches = []float64{0} // zero means "whatever the options say"
	}

	gpus := req.GPUs
	tensor := divisorsUpTo(gpus, perNode)
	pipelines := divisorsUpTo(gpus, maxStages(flat))
	experts := []float64{1}
	if params.Expert > 0 {
		experts = divisorsUpTo(gpus, gpus)
	} else {
		out.Notes = append(out.Notes,
			"This design has no experts, so expert parallelism is not among the plans.")
	}

	for _, batch := range batches {
		for _, tp := range tensor {
			for _, pp := range pipelines {
				for _, ep := range experts {
					rest := gpus / (tp * pp * ep)
					if rest < 1 || math.Trunc(rest) != rest {
						continue
					}
					dp := rest
					for _, zero := range zeroStages(dp) {
						for _, sp := range sequenceParallel(tp) {
							for _, recompute := range recomputes {
								out.Considered++
								c, err := price(doc, options, pre, candidateOf(
									dp, tp, pp, ep, zero, sp, recompute, batch), out.Budget)
								if err != nil {
									return nil, err
								}
								if c.Used <= 1 {
									out.Fits = append(out.Fits, *c)
								} else if out.Closest == nil || c.Used < out.Closest.Used {
									out.Closest = c
								}
							}
						}
					}
				}
			}
		}
	}

	sort.SliceStable(out.Fits, func(i, j int) bool {
		a, b := cost(out.Fits[i]), cost(out.Fits[j])
		if a != b {
			return a < b
		}
		// Between two equally demanding plans, the roomier one.
		return out.Fits[i].Used < out.Fits[j].Used
	})
	out.Fits = dedupe(out.Fits)
	if len(out.Fits) > limit {
		out.Fits = out.Fits[:limit]
	}
	if len(out.Fits) == 0 && out.Closest != nil {
		out.Notes = append(out.Notes, fmt.Sprintf(
			"Nothing fits. The nearest plan needs %s per device against a budget of %s; "+
				"a bigger device, more of them, or a smaller design is the way out.",
			analysis.FormatBytes(out.Closest.PerGpu.Total), analysis.FormatBytes(out.Budget)))
	}
	return out, nil
}

func candidateOf(dp, tp, pp, ep float64, zero int, sp bool, recompute string, batch float64) Candidate {
	return Candidate{
		Parallel: analysis.ParallelPlan{
			DP: dp, TP: tp, PP: pp, EP: ep, Zero: zero, SequenceParallel: sp,
		},
		Recompute: recompute, MicroBatch: batch,
	}
}

// price runs the memory model for one plan.
func price(doc *ir.Doc, options analysis.Options, pre analysis.Inputs, c Candidate, budget float64) (*Candidate, error) {
	o := options
	o.Recompute = c.Recompute
	gpus := c.Parallel.DP * c.Parallel.TP * c.Parallel.PP * c.Parallel.EP
	o.GPUs = &gpus
	dp, tp, pp, ep := c.Parallel.DP, c.Parallel.TP, c.Parallel.PP, c.Parallel.EP
	zero, sp := c.Parallel.Zero, c.Parallel.SequenceParallel
	o.Parallel = &analysis.PartialParallel{
		DP: &dp, TP: &tp, PP: &pp, EP: &ep, Zero: &zero, SequenceParallel: &sp,
	}
	if c.MicroBatch != 0 {
		b := c.MicroBatch
		o.B = &b
	}
	// The flattening is reused, but the activations depend on the micro-batch
	// and the recompute setting, so the rest of the analysis runs each time.
	res, err := analysis.Analyze(doc, o, pre)
	if err != nil {
		return nil, err
	}
	c.PerGpu = res.Memory.Train.PerGpu
	c.MicroBatch = res.Options.B
	c.Used = res.Memory.Train.PerGpu.Total / budget
	c.Summary = summarize(c)
	c.Notes = notesFor(c)
	return &c, nil
}

// cost ranks a plan by how much it asks of whoever has to run it.
//
// Not by speed: that turns on the interconnect and the kernels. This is the
// order in which a person would reach for these things — recomputation before
// sharding the optimizer, sharding the optimizer before splitting a matrix,
// and a pipeline last, because a pipeline changes the training loop.
func cost(c Candidate) float64 {
	n := 0.0
	switch c.Recompute {
	case "selective":
		n += 2
	case "full":
		n += 5
	}
	n += float64(c.Parallel.Zero)
	n += 2 * log2(c.Parallel.TP)
	n += 1.5 * log2(c.Parallel.EP)
	n += 4 * log2(c.Parallel.PP)
	if c.Parallel.SequenceParallel {
		n += 0.5
	}
	// A plan that fills the budget is a plan that fails on the day someone
	// lengthens a sequence. Past the point where the headroom stops being
	// headroom, tightness costs as much as a sharding decision.
	if c.Used > tight {
		n += 3 * (c.Used - tight) / (1 - tight)
	}
	return n
}

// tight is where a fit stops being comfortable, as a fraction of the budget —
// which already has the headroom taken out of it.
const tight = 0.85

// dedupe drops plans that hold the same memory as one already listed under the
// same split. Recomputation that frees nothing is not a different plan; it is
// the same plan with extra work in it, and the list is sorted so the cheaper
// one comes first.
func dedupe(cs []Candidate) []Candidate {
	out := cs[:0]
	seen := map[string]bool{}
	for _, c := range cs {
		key := fmt.Sprintf("%v|%.0f", c.Parallel, c.PerGpu.Total)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, c)
	}
	return out
}

func summarize(c Candidate) string {
	parts := []string{}
	if c.Parallel.DP > 1 {
		parts = append(parts, fmt.Sprintf("DP %s", num(c.Parallel.DP)))
	}
	if c.Parallel.TP > 1 {
		parts = append(parts, fmt.Sprintf("TP %s", num(c.Parallel.TP)))
	}
	if c.Parallel.PP > 1 {
		parts = append(parts, fmt.Sprintf("PP %s", num(c.Parallel.PP)))
	}
	if c.Parallel.EP > 1 {
		parts = append(parts, fmt.Sprintf("EP %s", num(c.Parallel.EP)))
	}
	if len(parts) == 0 {
		parts = append(parts, "one device")
	}
	s := join(parts, " x ")
	if c.Parallel.Zero > 0 {
		s += fmt.Sprintf(", ZeRO-%d", c.Parallel.Zero)
	}
	if c.Parallel.SequenceParallel {
		s += ", sequence parallel"
	}
	switch c.Recompute {
	case "selective":
		s += ", selective recompute"
	case "full":
		s += ", full recompute"
	}
	return s
}

// notesFor says what a plan asks for, in the words someone setting it up needs.
func notesFor(c Candidate) []string {
	out := []string{}
	if c.Parallel.TP > 1 {
		out = append(out, fmt.Sprintf(
			"Tensor parallelism all-reduces twice per layer, so keep the %s ranks inside one node.",
			num(c.Parallel.TP)))
	}
	if c.Parallel.PP > 1 {
		out = append(out, fmt.Sprintf(
			"A %s-stage pipeline needs enough micro-batches to fill it; the bubble is (stages-1)/micro-batches of the step.",
			num(c.Parallel.PP)))
	}
	if c.Parallel.EP > 1 {
		out = append(out, "Expert parallelism puts an all-to-all before and after every sparse layer.")
	}
	if c.Parallel.Zero == 3 {
		out = append(out, "ZeRO-3 gathers each layer's weights as it is reached, which is an all-gather per layer in both directions.")
	}
	switch c.Recompute {
	case "selective":
		out = append(out, "Selective recomputation redoes the attention, which is the cheap part to recompute and the expensive part to keep.")
	case "full":
		out = append(out, "Full recomputation costs about a third more compute for every activation it drops.")
	}
	if c.Used > tight {
		out = append(out, fmt.Sprintf(
			"This fills %.0f%% of the budget, which is little room for a longer sequence or a larger micro-batch.",
			c.Used*100))
	}
	return out
}

// --- the search space -------------------------------------------------------

// divisorsUpTo is the divisors of n that are at most cap, ascending.
func divisorsUpTo(n, cap float64) []float64 {
	var out []float64
	for d := 1.0; d <= n && d <= cap; d++ {
		if math.Mod(n, d) == 0 {
			out = append(out, d)
		}
	}
	if len(out) == 0 {
		out = []float64{1}
	}
	return out
}

// zeroStages are the stages worth trying. Sharding the optimizer across one
// replica shards it across nothing.
func zeroStages(dp float64) []int {
	if dp <= 1 {
		return []int{0}
	}
	return []int{0, 1, 2, 3}
}

// sequenceParallel is only a choice when there is a tensor-parallel group to
// shard the sequence across.
func sequenceParallel(tp float64) []bool {
	if tp <= 1 {
		return []bool{false}
	}
	return []bool{false, true}
}

// maxStages is how many pipeline stages the design could be cut into: one per
// layer of its deepest stack, and no more, because a stage holds whole layers.
func maxStages(flat *analysis.FlatResult) float64 {
	deepest := 1.0
	for _, r := range flat.Repeats {
		if r.Count > deepest {
			deepest = r.Count
		}
	}
	return deepest
}

func log2(v float64) float64 {
	if v <= 1 {
		return 0
	}
	return math.Log2(v)
}

func num(v float64) string { return analysis.JSNumber(v) }

func join(parts []string, sep string) string {
	s := ""
	for i, p := range parts {
		if i > 0 {
			s += sep
		}
		s += p
	}
	return s
}
