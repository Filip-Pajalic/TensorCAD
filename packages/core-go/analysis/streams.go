package analysis

import (
	"strings"

	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/shapes"
)

// Streams is which sequence each block runs along.
//
// A decoder-only design has one, T, and everything the analysis reports is per
// token of it. An encoder-decoder has a second, the source, S tokens long: the
// encoder runs along it and the decoder along T. A block's cost per token is
// only meaningful per token of its own stream, so each block is measured at
// its own stream's length and then spread over the target's tokens, which is
// what "per token" goes on meaning — per target token. A design that never
// declares S has one stream and is measured exactly as it always was.
type Streams struct {
	// Target is T and Source is S; Source is zero with no second sequence.
	Target, Source float64
	source         map[string]bool
}

// StreamFlops is one stream's forward pass per token of that stream.
type StreamFlops struct {
	// Symbol is "T" for the target and "S" for the source.
	Symbol string  `json:"symbol"`
	Length float64 `json:"length"`
	// Fwd is matmul FLOPs per token of this stream, attention included.
	Fwd float64 `json:"fwd"`
}

// sequenceSymbols are the runtime symbols a tensor can be as long as.
var sequenceSymbols = []string{"T", "S"}

// newStreams reads which blocks run along the source from the shapes on their
// pins: a block every one of whose sequence axes is S is the source's, and
// anything that touches T is the target's.
func newStreams(expanded *infer.Result, symbols *ir.SymbolTable, T, S float64) *Streams {
	st := &Streams{Target: T}
	if expanded == nil || symbols == nil || !symbols.Runtime["S"] {
		return st
	}
	st.Source = S
	st.source = map[string]bool{}
	touches := map[string]map[string]bool{}
	note := func(key string, shape shapes.Shape) {
		at := strings.LastIndex(key, ":")
		if at < 0 {
			return
		}
		path := key[:at]
		if touches[path] == nil {
			touches[path] = map[string]bool{}
		}
		for _, dim := range shape {
			for _, name := range dim.Symbols() {
				if name == "T" || name == "S" {
					touches[path][name] = true
				}
			}
		}
	}
	for key, shape := range expanded.Inputs {
		note(key, shape)
	}
	for key, shape := range expanded.Outputs {
		note(key, shape)
	}
	for path, seen := range touches {
		if seen["S"] && !seen["T"] {
			st.source[path] = true
		}
	}
	return st
}

// OnSource reports whether a block runs along the source.
func (s *Streams) OnSource(path string) bool { return s != nil && s.source[path] }

// Length is the length of the stream a block runs along.
func (s *Streams) Length(path string) float64 {
	if s.OnSource(path) {
		return s.Source
	}
	return s.Target
}

// Spread is what one token of a block's stream is worth per target token: S/T
// for the source, one for the target.
func (s *Streams) Spread(path string) float64 {
	if s.OnSource(path) && s.Target > 0 {
		return s.Source / s.Target
	}
	return 1
}

// Two reports whether the design has a second sequence at all.
func (s *Streams) Two() bool { return s != nil && s.Source > 0 }

// tensorTotal is how many elements a tensor holds across a batch, given its
// size with every runtime symbol at one.
//
// Each sequence axis contributes its own length: B S D is S long, B heads T T
// is T² long, and a cross-attention's scores, B heads T S, are T·S. A tensor
// with no sequence axis at all is charged per target token, as it always was.
func tensorTotal(shape shapes.Shape, perToken float64, env map[string]float64, B float64, lengths map[string]float64) float64 {
	scale := B
	any := false
	for _, name := range sequenceSymbols {
		length, ok := lengths[name]
		if !ok || length <= 0 {
			continue
		}
		power := sequencePower(shape, env, name)
		for i := 0; i < power; i++ {
			scale *= length
			any = true
		}
	}
	if !any {
		return perToken * B * lengths["T"]
	}
	return perToken * scale
}

// sequencePower is how many times a sequence symbol multiplies a tensor's size:
// the size at that symbol two, over its size at one, is two to that power.
func sequencePower(shape shapes.Shape, env map[string]float64, name string) int {
	one, okOne := elementsPerToken(shape, env)
	doubled := make(map[string]float64, len(env))
	for k, v := range env {
		doubled[k] = v
	}
	doubled[name] = 2
	two, okTwo := elementsPerToken(shape, doubled)
	if !okOne || !okTwo || one == 0 {
		return 0
	}
	power := 0
	for ratio := two / one; ratio >= 1.5; ratio /= 2 {
		power++
	}
	return power
}
