package analysis_test

import (
	"encoding/json"
	"testing"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/shapes"
)

// An encoder-decoder, small: an encoder stack over the source, a decoder stack
// over the target whose every layer reads the encoder's output, and nothing
// T5-specific yet. What M11's second phase claims: the decoder attends to all
// S source positions, its view of the encoder is one tensor every layer reads,
// and generation caches cross-attention's keys and values once per request.

const encoderDecoder = `{
  "version": 1,
  "meta": {"name": "tiny-seq2seq"},
  "symbols": {
    "B": {"kind": "runtime", "default": 1},
    "T": {"kind": "runtime", "default": 32},
    "S": {"kind": "runtime", "default": 48},
    "D": {"kind": "design", "value": 64},
    "H": {"kind": "design", "value": 4},
    "dh": {"kind": "design", "value": 16},
    "V": {"kind": "design", "value": 100},
    "L": {"kind": "design", "value": 2}
  },
  "graph": {
    "nodes": [
      {"id": "src", "type": "input", "params": {"shape": "B S", "dtype": "int64"}},
      {"id": "src_embed", "type": "embedding", "params": {"vocab": "V", "dim": "D"}},
      {"id": "encoder", "type": "repeat", "params": {"count": "L"}, "graph": {
        "nodes": [
          {"id": "_in", "type": "boundary_in", "params": {"ports": {"x": "B S D"}}},
          {"id": "block", "type": "transformer_block", "params": {"d_model": "D", "heads": "H", "kv_heads": "H",
            "head_dim": "dh", "ffn_hidden": "4*D", "mlp": "dense", "act": "relu", "causal": false}},
          {"id": "_out", "type": "boundary_out", "params": {"ports": {"x": "B S D"}}}
        ],
        "edges": [["_in:x", "block:x"], ["block:y", "_out:x"]]
      }},
      {"id": "enc_norm", "type": "rmsnorm", "params": {"dim": "D"}},
      {"id": "tgt", "type": "input", "params": {"shape": "B T", "dtype": "int64"}},
      {"id": "tgt_embed", "type": "embedding", "params": {"vocab": "V", "dim": "D"}},
      {"id": "decoder", "type": "repeat", "params": {"count": "L"}, "graph": {
        "nodes": [
          {"id": "_in", "type": "boundary_in", "params": {"ports": {"x": "B T D", "memory": "B S D"}}},
          {"id": "block", "type": "transformer_block", "params": {"d_model": "D", "heads": "H", "kv_heads": "H",
            "head_dim": "dh", "ffn_hidden": "4*D", "mlp": "dense", "act": "relu", "causal": true,
            "cross_attention": true}},
          {"id": "_out", "type": "boundary_out", "params": {"ports": {"x": "B T D"}}}
        ],
        "edges": [["_in:x", "block:x"], ["_in:memory", "block:memory"], ["block:y", "_out:x"]]
      }},
      {"id": "final_norm", "type": "rmsnorm", "params": {"dim": "D"}},
      {"id": "head", "type": "lm_head", "params": {"vocab": "V", "dim": "D", "tied": false}},
      {"id": "logits", "type": "output"}
    ],
    "edges": [
      ["src:x", "src_embed:ids"], ["src_embed:y", "encoder:x"], ["encoder:x", "enc_norm:x"],
      ["tgt:x", "tgt_embed:ids"], ["tgt_embed:y", "decoder:x"], ["enc_norm:y", "decoder:memory"],
      ["decoder:x", "final_norm:x"], ["final_norm:y", "head:x"], ["head:y", "logits:x"]
    ]
  }
}`

func seq2seq(t *testing.T) *ir.Doc {
	t.Helper()
	var doc ir.Doc
	if err := json.Unmarshal([]byte(encoderDecoder), &doc); err != nil {
		t.Fatal(err)
	}
	return &doc
}

// The whole thing checks: the encoder runs along S, the decoder along T, and
// cross-attention is where the two meet on purpose.
func TestAnEncoderDecoderChecks(t *testing.T) {
	doc := seq2seq(t)
	res := infer.Shapes(doc, ir.ResolveSymbols(doc), infer.Options{ExpandComposites: true})
	for _, issue := range res.Issues {
		t.Errorf("%s: %s", issue.Path, issue.Message)
	}
	want := map[string]string{
		"decoder/block/cross/attn:y":    "B H T dh",
		"decoder/block/cross/k_heads:y": "B H S dh",
		"decoder/block/attn/attn:y":     "B H T dh",
		"encoder/block/attn/attn:y":     "B H S dh",
	}
	for key, shape := range want {
		if got := shapes.ShapeToString(res.Outputs[key]); got != shape {
			t.Errorf("%s is %s, want %s", key, got, shape)
		}
	}
}

// Cross-attention reads every source position for every target token:
// 2·heads·(head_dim + head_dim)·S a layer, and nothing masked.
func TestCrossAttentionReadsTheWholeSource(t *testing.T) {
	doc := seq2seq(t)
	short, long := at(t, doc, 32, 48), at(t, doc, 32, 96)
	const H, dh, L = 4.0, 16.0, 2.0
	cross := func(r *analysis.Result) float64 { return r.Flops.ByPath["decoder/block/cross/attn"] }
	if got := cross(short); got != 2*H*(dh+dh)*48*L {
		t.Errorf("cross-attention is %v FLOPs a target token, want %v", got, 2*H*(dh+dh)*48*L)
	}
	if cross(long) != 2*cross(short) {
		t.Errorf("doubling S took cross-attention from %v to %v", cross(short), cross(long))
	}
	// The decoder's own attention does not care how long the source is.
	self := func(r *analysis.Result) float64 { return r.Flops.ByPath["decoder/block/attn/attn"] }
	if self(long) != self(short) {
		t.Errorf("the decoder's self-attention moved with S: %v, %v", self(short), self(long))
	}
}

// Generation caches the decoder's own keys and values a token at a time, and
// cross-attention's once per request from the encoder: S positions, every
// layer. The encoder itself caches nothing.
func TestGenerationCachesCrossAttentionPerRequest(t *testing.T) {
	res := at(t, seq2seq(t), 32, 48)
	const H, dh, L, bytes, S = 4.0, 16.0, 2.0, 2.0, 48.0
	if want := H * 2 * dh * bytes * L; res.Kv.BytesPerToken != want {
		t.Errorf("per target token %v, want %v", res.Kv.BytesPerToken, want)
	}
	if want := H * 2 * dh * bytes * S * L; res.Kv.BytesPerSequenceFixed != want {
		t.Errorf("per request %v, want %v", res.Kv.BytesPerSequenceFixed, want)
	}
}

// Every decoder layer reads the encoder's output, and it is one tensor: kept
// once for the backward pass, not once per layer.
func TestTheEncoderOutputIsKeptOnce(t *testing.T) {
	res := at(t, seq2seq(t), 32, 48)
	const D, S, bytes = 64.0, 48.0, 2.0
	var memory float64
	for key, b := range res.Memory.Train.ActivationsByTensor {
		if key == "enc_norm:y" {
			memory = b
		}
	}
	if memory != D*S*bytes {
		t.Errorf("the encoder's output is kept as %v bytes, want one %v", memory, D*S*bytes)
	}
}
