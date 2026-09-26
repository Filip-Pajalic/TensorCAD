package analysis

import (
	"strings"

	"github.com/tensorcad/core/infer"
)

// Autocast is the precision recipe plain PyTorch trains with.
//
// The analysis's default is mixed precision the way the large frameworks do
// it: weights, activations and gradients in bf16 over an fp32 master copy.
// `torch.autocast` gets to the same sixteen bytes a parameter by another
// route — the weights themselves are fp32 and each matrix multiply casts to
// bf16 on the way in — and saves more for the backward pass, three ways:
//
//   - Some tensors stay fp32. An embedding looks up fp32 weights; a sum with
//     an fp32 operand is fp32, so the residual stream is; a norm and a softmax
//     run in fp32 by autocast's own list.
//   - A matrix multiply that reads an fp32 tensor saves its own bf16 cast of
//     it. Three projections reading one norm output save three copies.
//   - Every weight a matrix multiply casts is held as a bf16 copy until the
//     backward pass has used it.
//
// Measured with `tensorcad-runtime measure --recipe amp`, which is what the
// tests hold this to.
const Autocast = "autocast"

// castsToHalf are the blocks autocast runs in bf16: matrix multiplies and the
// attention kernel. They cast what they read, weights included.
var castsToHalf = map[string]bool{
	"linear": true, "lm_head": true, "conv1d": true, "conv2d": true,
	"sdpa": true, "attn_scores": true, "attn_values": true,
}

// staysFull are the blocks whose output is fp32 whatever they read: lookups of
// fp32 tables, and what autocast lists to run in fp32.
var staysFull = map[string]bool{
	"embedding": true, "pos_embedding": true, "learned_tokens": true, "position_bias": true,
	"rmsnorm": true, "layernorm": true, "attn_softmax": true, "topk_router": true,
	// A rotation multiplies by fp32 cosines and sines.
	"rope": true,
	// The scans are written in plain operations, exp among them, which
	// autocast runs in fp32.
	"selective_scan": true, "ssd_scan": true, "gated_delta_scan": true,
}

// precisions follows each tensor's dtype under autocast: 4 bytes an element
// for fp32, 2 for bf16.
type precisions struct {
	expanded *infer.Result
	types    map[string]string
	kinds    map[string]string
	memo     map[string]float64
	visiting map[string]bool
	// inputs are what each node reads, by path.
	inputs map[string][]string
}

func newPrecisions(flat *FlatResult, expanded *infer.Result) *precisions {
	p := &precisions{
		expanded: expanded,
		types:    map[string]string{},
		kinds:    map[string]string{},
		memo:     map[string]float64{},
		visiting: map[string]bool{},
		inputs:   map[string][]string{},
	}
	for consumer, from := range expanded.ProducerOf {
		if at := strings.LastIndex(consumer, ":"); at > 0 {
			p.inputs[consumer[:at]] = append(p.inputs[consumer[:at]], from)
		}
	}
	for _, b := range flat.Blocks {
		p.types[b.Path] = b.Type
		p.kinds[b.Path] = b.Kind
	}
	for _, n := range flat.Nodes {
		p.types[n.Path] = n.Type
		p.kinds[n.Path] = "primitive"
	}
	return p
}

// bytesOf is the width of an element of the tensor at a producing
// "path:port", following it across the boundaries it was handed through.
func (p *precisions) bytesOf(endpoint string) float64 {
	if v, ok := p.memo[endpoint]; ok {
		return v
	}
	// A stack reads its own output; the loop resolves to whatever else feeds
	// it, and fp32 anywhere in it is fp32 everywhere, so half is the answer
	// that cannot be wrong for long.
	if p.visiting[endpoint] {
		return 2
	}
	p.visiting[endpoint] = true
	v := p.compute(endpoint)
	delete(p.visiting, endpoint)
	p.memo[endpoint] = v
	return v
}

func (p *precisions) compute(endpoint string) float64 {
	at := strings.LastIndex(endpoint, ":")
	if at <= 0 {
		return 2
	}
	path, port := endpoint[:at], endpoint[at+1:]
	switch p.types[path] {
	case "input":
		// Token ids. What is looked up with them is fp32; this is never
		// saved at its own width by anything that matters here.
		return 2
	case "boundary_in":
		// Whatever arrived at the port of the block this is the inside of, and
		// for a stack also what its last copy handed back.
		owner := parentOf(path)
		width := 2.0
		if from, ok := p.expanded.ProducerOf[owner+":"+port]; ok {
			width = max(width, p.bytesOf(from))
		}
		if p.types[owner] == "repeat" {
			if from, ok := p.expanded.ProducerOf[owner+"/_out:"+port]; ok {
				width = max(width, p.bytesOf(from))
			}
		}
		return width
	}
	if kind := p.kinds[path]; kind == "composite" || kind == "container" {
		// A composite's output is what feeds its inside's boundary.
		if from, ok := p.expanded.ProducerOf[path+"/_out:"+port]; ok {
			return p.bytesOf(from)
		}
		return 2
	}
	kind := p.types[path]
	if castsToHalf[kind] {
		return 2
	}
	if staysFull[kind] {
		return 4
	}
	// Everything else runs at the widest of what it reads: an add with the
	// fp32 residual stream is fp32.
	width := 2.0
	for _, from := range p.inputs[path] {
		width = max(width, p.bytesOf(from))
	}
	return width
}

// precisionName is what the resolved options say: autocast, or nothing for
// the default.
func precisionName(autocast bool) string {
	if autocast {
		return Autocast
	}
	return ""
}

func parentOf(path string) string {
	if at := strings.LastIndex(path, "/"); at > 0 {
		return path[:at]
	}
	return ""
}
