package analysis

import (
	"fmt"
	"math"
	"strings"

	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/shapes"
)

// ParallelPlan is how the model is spread over the devices.
type ParallelPlan struct {
	// DP is data-parallel replicas.
	DP float64 `json:"dp"`
	// TP is the tensor-parallel degree.
	TP float64 `json:"tp"`
	// PP is the number of pipeline stages.
	PP float64 `json:"pp"`
	// EP is the expert-parallel degree.
	EP float64 `json:"ep"`
	// Zero is the ZeRO/FSDP stage, 0 to 3.
	Zero int `json:"zero"`
	// SequenceParallel shards activations along the sequence dimension across
	// the tensor-parallel group.
	SequenceParallel bool `json:"sequenceParallel"`
}

// DefaultParallel is one device, nothing sharded.
var DefaultParallel = ParallelPlan{DP: 1, TP: 1, PP: 1, EP: 1, Zero: 0, SequenceParallel: false}

// MemoryOptions is everything the memory model needs beyond the flattened
// design.
type MemoryOptions struct {
	Ctx       catalog.AnalysisCtx
	Optimizer string
	// Recompute is "none", "selective" or "full".
	Recompute string
	Parallel  ParallelPlan
	// InferenceDtypeBytes is bytes per weight when serving.
	InferenceDtypeBytes float64
	// Concurrency is the number of concurrent sequences when serving.
	Concurrency float64
	// StreamWidths maps a repeat container's path to the per-token activation
	// width of its stream.
	StreamWidths map[string]float64
	// Expanded is shape inference with composites expanded, used to size the
	// retained tensors.
	Expanded *infer.Result
	Symbols  *ir.SymbolTable
	Params   *ParamsResult
	Kv       *KvResult
}

// TrainPerGpu is the training footprint on one device.
type TrainPerGpu struct {
	Weights     float64 `json:"weights"`
	Grads       float64 `json:"grads"`
	Optimizer   float64 `json:"optimizer"`
	Activations float64 `json:"activations"`
	Total       float64 `json:"total"`
}

// TrainMemory is the training footprint.
type TrainMemory struct {
	Weights     float64 `json:"weights"`
	Grads       float64 `json:"grads"`
	Optimizer   float64 `json:"optimizer"`
	Activations float64 `json:"activations"`
	// Logits is the part of Activations attributable to the vocabulary logits.
	Logits            float64            `json:"logits"`
	Total             float64            `json:"total"`
	PerGpu            TrainPerGpu        `json:"perGpu"`
	ActivationsByPath map[string]float64 `json:"activationsByPath"`
	// ActivationsByTensor is the same bytes charged to the tensor rather than
	// to the block that produced it, keyed "path:port".
	//
	// The two are not a reformatting of each other. A block with several
	// outputs is one row in ActivationsByPath and several here, which is the
	// difference between "this block costs 4 MiB" and "*this* wire is the
	// expensive one" — and the second is the question somebody pointing at a
	// net is asking.
	ActivationsByTensor map[string]float64 `json:"activationsByTensor"`
}

// InferMemory is the serving footprint.
type InferMemory struct {
	Weights  float64 `json:"weights"`
	Kv       float64 `json:"kv"`
	Overhead float64 `json:"overhead"`
	Total    float64 `json:"total"`
}

// MemoryResult is what the design costs to hold.
type MemoryResult struct {
	// WeightsBytes is the weights at the training dtype, unsharded.
	WeightsBytes   float64     `json:"weightsBytes"`
	Train          TrainMemory `json:"train"`
	Infer          InferMemory `json:"infer"`
	OptimizerLabel string      `json:"optimizerLabel"`
	Notes          []string    `json:"notes"`
	Errors         []string    `json:"errors"`
}

// elementsPerToken is the product of a tensor's dimensions with the runtime
// dimensions set to one.
func elementsPerToken(shape shapes.Shape, env map[string]float64) (float64, bool) {
	product := 1.0
	for _, dim := range shape {
		v, ok := dim.ToNumber(env)
		if !ok {
			return 0, false
		}
		product *= v
	}
	return product, true
}

// AnalyzeMemory accounts for training and inference memory.
//
// Training memory is four things: weights, gradients, optimizer state and
// activations, plus the logits buffer, which is large enough at modern
// vocabularies to deserve its own line. Sharding follows ZeRO: stage 1 shards
// the optimizer, stage 2 adds gradients, stage 3 adds weights. Tensor
// parallelism divides the weights, and divides activations only when sequence
// parallelism is on.
//
// References: ZeRO (arXiv 1910.02054) and Korthikanti et al. on activation
// recomputation (arXiv 2205.05198).
func AnalyzeMemory(flat *FlatResult, opts MemoryOptions) *MemoryResult {
	// Empty rather than nil: these cross to the editor as JSON, where a nil
	// slice is null and every reader would have to defend against it.
	notes, errs := []string{}, []string{}
	ctx, par := opts.Ctx, opts.Parallel
	tokens := ctx.B * ctx.T

	// --- activations --------------------------------------------------------
	// Memory is attributed to tensors rather than to blocks. A tensor several
	// blocks read (the residual stream feeding the query, key and value
	// projections) is kept alive once, not once per reader.
	activationsByPath := map[string]float64{}
	activationsByTensor := map[string]float64{}
	activations := 0.0
	logits := 0.0

	fullRecompute := opts.Recompute == "full"
	env := make(map[string]float64, len(opts.Symbols.DesignValues)+len(opts.Symbols.Runtime))
	for k, v := range opts.Symbols.DesignValues {
		env[k] = v
	}
	for name := range opts.Symbols.Runtime {
		env[name] = 1
	}

	counted := map[string]bool{}
	// `tensor` is the producing "path:port" when there is one; the extras that
	// belong to a block rather than to a wire pass their own path, which is
	// what they are.
	add := func(path, tensor string, bytes float64) {
		if bytes == 0 {
			return
		}
		activations += bytes
		activationsByPath[path] += bytes
		activationsByTensor[tensor] += bytes
	}

	for i := range flat.Nodes {
		node := &flat.Nodes[i]
		if fullRecompute && node.Container != "" {
			continue
		}

		var retained []string
		if node.Def.Retains != nil {
			retained = node.Def.Retains(node.Resolved)
		}
		for _, port := range retained {
			consumerKey := node.Path + ":" + port
			producer, ok := opts.Expanded.ProducerOf[consumerKey]
			if !ok {
				producer = consumerKey
			}
			if counted[producer] {
				continue
			}
			counted[producer] = true

			shape, known := opts.Expanded.Inputs[consumerKey]
			if !known {
				errs = append(errs, fmt.Sprintf(
					"%s: no shape for retained input %q, so its activation memory is missing", node.Path, port))
				continue
			}
			elements, sized := elementsPerToken(shape, env)
			if !sized {
				errs = append(errs, fmt.Sprintf("%s: could not size the tensor on input %q", node.Path, port))
				continue
			}
			// The tensor belongs to whoever produced it, not to whoever is
			// reading it here.
			owner := node.Path
			if at := strings.LastIndex(producer, ":"); at > 0 {
				owner = producer[:at]
			}
			add(owner, producer, elements*ctx.Bytes*node.ActiveMultiplier*tokens)
		}

		if node.Def.ExtraActivationBytes != nil {
			extra := node.Def.ExtraActivationBytes(node.Resolved, ctx)
			total := extra * node.ActiveMultiplier * tokens
			add(node.Path, node.Path, total)
			if node.Type == "lm_head" {
				logits += total
			}
		}
	}

	if fullRecompute {
		for _, rep := range flat.Repeats {
			if rep.Type != "repeat" {
				continue
			}
			width, ok := opts.StreamWidths[rep.Path]
			if !ok {
				errs = append(errs, fmt.Sprintf(
					"Could not determine the stream width of %q for full recomputation", rep.Path))
				continue
			}
			total := width * ctx.Bytes * rep.Count * tokens
			activations += total
			activationsByPath[rep.Path] = total
			activationsByTensor[rep.Path] = total
		}
		notes = append(notes,
			"Full recomputation keeps only each layer's input, at the cost of one extra forward pass (8N instead of 6N).")
	} else if opts.Recompute == "selective" {
		notes = append(notes,
			"Selective recomputation drops the attention score matrix. A memory-efficient attention kernel already does this, so the two coincide here.")
	}

	// --- weights, gradients, optimizer --------------------------------------
	bytesPer, known := Optimizers[opts.Optimizer]
	if !known {
		errs = append(errs, fmt.Sprintf("Unknown optimizer %q", opts.Optimizer))
		bytesPer = Optimizers["adamw"]
	}

	total := opts.Params.Total
	weights := total * bytesPer.Weights
	grads := total * bytesPer.Grads
	optimizer := total * bytesPer.Optimizer

	shardModel := math.Max(1, par.TP) * math.Max(1, par.PP)
	dp := math.Max(1, par.DP)

	// Two pools, because they shard by different degrees. The experts go to
	// whole devices; everything else is split across the tensor-parallel group.
	// A design with no experts has an empty second pool and the arithmetic is
	// what it always was.
	ep := math.Max(1, par.EP)
	expert := math.Min(opts.Params.Expert, total)
	dense := total - expert
	perGpuParams := dense/shardModel + expert/(shardModel*ep)
	if ep > 1 && expert == 0 {
		notes = append(notes,
			"Expert parallelism has nothing to divide: this design has no experts.")
	}
	if ep > 1 && expert > 0 {
		notes = append(notes, fmt.Sprintf(
			"Expert parallelism holds %s of expert weights per device instead of %s, and puts an all-to-all around every sparse layer.",
			FormatBytes(expert*bytesPer.Weights/(shardModel*ep)),
			FormatBytes(expert*bytesPer.Weights/shardModel)))
	}

	wGpu := perGpuParams * bytesPer.Weights
	gGpu := perGpuParams * bytesPer.Grads
	oGpu := perGpuParams * bytesPer.Optimizer
	if par.Zero >= 1 {
		oGpu /= dp
	}
	if par.Zero >= 2 {
		gGpu /= dp
	}
	if par.Zero >= 3 {
		wGpu /= dp
	}

	actGpu := activations
	if par.SequenceParallel {
		actGpu = activations / math.Max(1, par.TP)
	}

	if par.PP > 1 {
		notes = append(notes,
			"Pipeline parallelism divides the weights but not the activations of the first stage: under 1F1B it holds one micro-batch worth of the whole model.")
	}
	if par.TP > 1 && !par.SequenceParallel {
		notes = append(notes,
			"Without sequence parallelism, tensor parallelism leaves the norm and dropout activations replicated.")
	}
	if opts.Optimizer == "bf16_adam" {
		notes = append(notes,
			"Pure bf16 Adam without master weights saves 8 bytes per parameter but is prone to divergence.")
	}

	// --- inference ----------------------------------------------------------
	inferWeights := total * opts.InferenceDtypeBytes
	inferKv := KvBytesFor(opts.Kv, ctx.T, opts.Concurrency)
	overhead := 0.2 * (inferWeights + inferKv)

	// Latent attention's cache is small only under a kernel that scores in the
	// latent space. An engine that materializes keys and values per head holds
	// the whole multi-head cache, which for DeepSeek-V3 is 57 times as much, so
	// which figure is being quoted is worth saying rather than assuming.
	if opts.Kv.BytesPerTokenDecompressed > opts.Kv.BytesPerToken {
		decompressed := (opts.Kv.BytesPerTokenDecompressed*ctx.T + opts.Kv.BytesPerSequenceFixed) *
			opts.Concurrency
		notes = append(notes, fmt.Sprintf(
			"The %s of cache assumes a kernel that absorbs the up-projections and scores in the "+
				"compressed space. An engine that materializes keys and values per head holds %s "+
				"instead, which is %sx as much.",
			FormatBytes(inferKv), FormatBytes(decompressed),
			JSToFixed(opts.Kv.BytesPerTokenDecompressed/opts.Kv.BytesPerToken, 0)))
	}

	return &MemoryResult{
		WeightsBytes: weights,
		Train: TrainMemory{
			Weights: weights, Grads: grads, Optimizer: optimizer,
			Activations: activations, Logits: logits,
			Total: weights + grads + optimizer + activations,
			PerGpu: TrainPerGpu{
				Weights: wGpu, Grads: gGpu, Optimizer: oGpu, Activations: actGpu,
				Total: wGpu + gGpu + oGpu + actGpu,
			},
			ActivationsByPath:   activationsByPath,
			ActivationsByTensor: activationsByTensor,
		},
		Infer: InferMemory{
			Weights: inferWeights, Kv: inferKv, Overhead: overhead,
			Total: inferWeights + inferKv + overhead,
		},
		OptimizerLabel: bytesPer.Label,
		Notes:          notes,
		Errors:         errs,
	}
}
