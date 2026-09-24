package catalog

import (
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/tensorcad/core/attnexpr"
)

// Attention is what one sdpa does to its scores: which it keeps and what each
// becomes before the softmax.
//
// `causal`, `window` and `logit_softcap` are the three variants common enough
// to have had switches before a design could write the rest as expressions,
// and they stay switches — but they mean exactly the masks and the score they
// are, so a design that writes them out as expressions is the same design.
// That is what Mask and Score return: every part of it, as one expression.
type Attention struct {
	Causal bool
	Window float64
	Cap    float64
	Heads  float64
	// MaskExpr and ScoreExpr are what the design wrote, nil where it wrote
	// nothing.
	MaskExpr  attnexpr.Node
	ScoreExpr attnexpr.Node
}

// AttentionOf reads an sdpa's resolved parameters. `heads` is a constant
// once the expression is on a block, so it is replaced by the block's own.
func AttentionOf(r *Resolved) Attention {
	a := Attention{
		Causal: r.Bool("causal"),
		Window: math.Max(0, r.Num("window")),
		Cap:    r.Num("logit_softcap"),
		Heads:  r.Num("heads"),
	}
	if m := compiled(r.Str("mask"), attnexpr.Mask); m != nil {
		a.MaskExpr = attnexpr.With(m, "heads", a.Heads)
		// A mask that is always true once heads is known masks nothing.
		if b, ok := a.MaskExpr.(attnexpr.Bool); ok && b.V {
			a.MaskExpr = nil
		}
	}
	if s := compiled(r.Str("score"), attnexpr.Score); s != nil {
		a.ScoreExpr = attnexpr.With(s, "heads", a.Heads)
	}
	return a
}

// Expressions reports whether the design wrote either expression, which is
// what decides the kernel: FlexAttention for any expression, FlashAttention
// for the switches alone.
func (a Attention) Expressions() bool { return a.MaskExpr != nil || a.ScoreExpr != nil }

var (
	causalMask = attnexpr.Binary{Op: "<=", X: attnexpr.Var{Name: "kv"}, Y: attnexpr.Var{Name: "q"}}
	scoreVar   = attnexpr.Var{Name: "score"}
)

// Mask is every condition a score has to meet to count, or nil when every
// score counts.
func (a Attention) Mask() attnexpr.Node {
	var m attnexpr.Node
	if a.Causal {
		m = causalMask
	}
	if a.Window > 0 {
		m = attnexpr.And(m, attnexpr.Binary{Op: "<",
			X: attnexpr.Binary{Op: "-", X: attnexpr.Var{Name: "q"}, Y: attnexpr.Var{Name: "kv"}},
			Y: attnexpr.Num{V: a.Window}})
	}
	m = attnexpr.And(m, a.MaskExpr)
	if m != nil {
		m = attnexpr.Fold(m)
	}
	return m
}

// Score is what a score becomes, or nil when it is left alone. The design's
// expression comes first and the cap after it, so a capped score stays within
// the cap whatever the expression added.
func (a Attention) Score() attnexpr.Node {
	s := a.ScoreExpr
	if a.Cap != 0 {
		inner := s
		if inner == nil {
			inner = scoreVar
		}
		c := attnexpr.Num{V: a.Cap}
		s = attnexpr.Binary{Op: "*", X: c, Y: attnexpr.Call{Fn: "tanh",
			Args: []attnexpr.Node{attnexpr.Binary{Op: "/", X: inner, Y: c}}}}
	}
	return s
}

// KeysPerQuery is how many keys the average query attends to at sequence
// length T: the matmul work a kernel that skips what the mask removes does.
//
// Causal and window are counted in closed form. A causal window of W keeps
// min(q + 1, W) keys at position q, which averages W - W²/2T once the
// sequence is longer than the window — not W/2, which is what halving the
// window for being causal gave, and which undercounted a long sequence's
// windowed layers by up to half. The design's own mask is then the share of
// what those two leave that it keeps too, measured by evaluating it.
func (a Attention) KeysPerQuery(T, B float64) float64 {
	keys := a.structuredKeys(T)
	if a.MaskExpr != nil {
		keys *= a.keptShare(T, B)
	}
	return keys
}

func (a Attention) structuredKeys(T float64) float64 {
	w := a.Window
	switch {
	case a.Causal && w > 0 && w < T:
		return w - w*w/(2*T)
	case a.Causal:
		return T / 2
	case w > 0:
		return math.Min(T, w)
	}
	return T
}

// span is the keys causal and window leave for the query at q, inclusive.
func (a Attention) span(q, T float64) (lo, hi float64) {
	lo, hi = 0, T-1
	if a.Causal {
		hi = q
	}
	if a.Window > 0 {
		lo = math.Max(0, q-a.Window+1)
	}
	return lo, hi
}

// How much of the score matrix is evaluated to measure a mask. A few tens of
// thousands of points, which is milliseconds, and the answer is kept until the
// mask or the sequence length changes.
const (
	sampleRows    = 128
	sampleColumns = 512
	sampleHeads   = 8
	sampleBatch   = 4
)

type shareKey struct {
	mask         string
	causal       bool
	window, T, B float64
	heads        float64
	usesH, usesB bool
	probe        int
}

var (
	cacheMu    sync.Mutex
	shareCache = map[shareKey]float64{}
	probeCache = map[shareKey]probeResult{}
	exprCache  = map[string]attnexpr.Node{}
)

// remember keeps a cache from growing without bound. A session that edits a
// mask one character at a time leaves every intermediate behind; clearing is
// cheaper than tracking which are stale.
func remember[K comparable, V any](m map[K]V, k K, v V) {
	if len(m) > 512 {
		clear(m)
	}
	m[k] = v
}

// compiled reads back an expression the resolver already checked, which is
// why the symbols are gone and an error cannot happen: what is stored is the
// resolver's own printing of it.
func compiled(text string, kind attnexpr.Kind) attnexpr.Node {
	if text == "" {
		return nil
	}
	cacheMu.Lock()
	defer cacheMu.Unlock()
	if n, ok := exprCache[text]; ok {
		return n
	}
	n, err := attnexpr.Compile(text, kind, nil)
	if err != nil {
		return nil
	}
	remember(exprCache, text, n)
	return n
}

// keptShare is the fraction of the scores causal and window leave that the
// design's mask also keeps, at sequence length T.
//
// Measured, not derived: a mask is an arbitrary expression, and the only
// general way to know what it keeps is to ask it. Rows are drawn one from each
// of up to 128 equal strata of the sequence, and within a row the keys causal
// and window leave are either all evaluated or, past 512 of them, drawn one
// from each of 512 strata. A point drawn at random within its stratum rather
// than at its centre is what keeps a periodic mask — every 64th key — from
// landing on the same phase every time. The draws are seeded, so an answer
// never changes between two runs of the same design.
func (a Attention) keptShare(T, B float64) float64 {
	if a.MaskExpr == nil || !(T >= 1) {
		return 1
	}
	key := shareKey{
		mask: attnexpr.String(a.MaskExpr), causal: a.Causal, window: a.Window, T: T,
		usesH: attnexpr.Uses(a.MaskExpr, "h"), usesB: attnexpr.Uses(a.MaskExpr, "b"),
	}
	if key.usesH {
		key.heads = a.Heads
	}
	if key.usesB {
		key.B = B
	}
	cacheMu.Lock()
	if v, ok := shareCache[key]; ok {
		cacheMu.Unlock()
		return v
	}
	cacheMu.Unlock()

	rows, columns := sampleRows, sampleColumns
	if key.usesH {
		// Eight heads at a time, so a quarter of the points each.
		rows, columns = sampleRows/2, sampleColumns/4
	}
	g := seed(key.mask)
	heads := []float64{0}
	if key.usesH {
		heads = spread(a.Heads, sampleHeads, &g)
	}
	batch := []float64{0}
	if key.usesB {
		batch = spread(B, sampleBatch, &g)
	}
	var kept, total float64
	env := attnexpr.Env{Heads: a.Heads}
	for _, b := range batch {
		env.B = b
		for _, h := range heads {
			env.H = h
			for _, q := range spread(T, rows, &g) {
				env.Q = q
				lo, hi := a.span(q, T)
				length := hi - lo + 1
				if length <= 0 {
					continue
				}
				total += length
				if length <= float64(columns) {
					for kv := lo; kv <= hi; kv++ {
						env.KV = kv
						if attnexpr.Eval(a.MaskExpr, env) != 0 {
							kept++
						}
					}
					continue
				}
				hits := 0.0
				step := length / float64(columns)
				for j := 0; j < columns; j++ {
					env.KV = lo + math.Floor((float64(j)+g.next())*step)
					if attnexpr.Eval(a.MaskExpr, env) != 0 {
						hits++
					}
				}
				kept += hits * step
			}
		}
	}
	share := 0.0
	if total > 0 {
		share = kept / total
	}
	cacheMu.Lock()
	remember(shareCache, key, share)
	cacheMu.Unlock()
	return share
}

// probeLength is how much of the sequence the design rules check a mask over.
// The rules run without an operating point, so they cannot ask about the
// length a design will be trained at; the first positions are the ones every
// sequence has.
const probeLength = 128

type probeResult struct {
	kept  float64
	empty []int
}

// probe evaluates the whole mask, causal and window included, over the first
// 128 positions exactly: how many scores it keeps, and which queries it
// leaves with nothing to attend to.
func (a Attention) probe() probeResult {
	m := a.Mask()
	if m == nil {
		return probeResult{kept: 1}
	}
	key := shareKey{mask: attnexpr.String(m), probe: probeLength, usesH: attnexpr.Uses(m, "h")}
	if key.usesH {
		key.heads = a.Heads
	}
	cacheMu.Lock()
	if v, ok := probeCache[key]; ok {
		cacheMu.Unlock()
		return v
	}
	cacheMu.Unlock()

	g := seed(key.mask)
	heads := []float64{0}
	if key.usesH {
		heads = spread(a.Heads, sampleHeads, &g)
	}
	var out probeResult
	env := attnexpr.Env{Heads: a.Heads}
	for q := 0; q < probeLength; q++ {
		env.Q = float64(q)
		empty := false
		for _, h := range heads {
			env.H = h
			row := 0.0
			for kv := 0; kv < probeLength; kv++ {
				env.KV = float64(kv)
				row += attnexpr.Eval(m, env)
			}
			out.kept += row
			if row == 0 {
				empty = true
			}
		}
		if empty {
			out.empty = append(out.empty, q)
		}
	}
	cacheMu.Lock()
	remember(probeCache, key, out)
	cacheMu.Unlock()
	return out
}

// MaskGrid is a mask drawn the way FlexAttention's block mask sees it: the
// sequence cut into blocks along both sides, and for each block the share of
// its scores the mask keeps. A block at zero is skipped outright; a full one is
// computed without the mask; anything between is computed and then masked.
type MaskGrid struct {
	// T is the sequence length it was drawn at.
	T float64 `json:"T"`
	// Cells is how many blocks along each side.
	Cells int `json:"cells"`
	// Span is how many positions one block covers along each side.
	Span float64 `json:"span"`
	// Head is which head was drawn, for a mask that differs between them.
	Head    float64 `json:"head"`
	PerHead bool    `json:"perHead"`
	// Kept is row-major, a query block per row and a key block per column.
	Kept []float64 `json:"kept"`
	// Density is the share of the whole score matrix kept, as the analysis
	// counts it.
	Density float64 `json:"density"`
	// Mask is every condition a score has to meet, causal and window
	// included, in the design's own language; empty when every score counts.
	Mask string `json:"mask"`
	// Score is what each score becomes, the cap included; empty when nothing.
	Score string `json:"score"`
}

// Grid draws the mask at sequence length T, for one head.
func (a Attention) Grid(T, B float64, head float64) MaskGrid {
	cells := 32
	if T < float64(cells) {
		cells = int(math.Max(1, T))
	}
	span := T / float64(cells)
	out := MaskGrid{
		T: T, Cells: cells, Span: span, Head: head,
		Kept:    make([]float64, cells*cells),
		Density: a.KeysPerQuery(T, B) / T,
	}
	m := a.Mask()
	if s := a.Score(); s != nil {
		out.Score = attnexpr.String(s)
	}
	if m == nil {
		for i := range out.Kept {
			out.Kept[i] = 1
		}
		return out
	}
	out.Mask = attnexpr.String(m)
	out.PerHead = attnexpr.Uses(m, "h")

	// Eight by eight points a block, each drawn within its own sub-block.
	per := int(math.Min(8, math.Max(1, math.Floor(span))))
	g := seed(out.Mask)
	env := attnexpr.Env{H: head, Heads: a.Heads}
	for i := 0; i < cells; i++ {
		for j := 0; j < cells; j++ {
			kept := 0.0
			for u := 0; u < per; u++ {
				for v := 0; v < per; v++ {
					env.Q = math.Floor((float64(i) + (float64(u)+g.next())/float64(per)) * span)
					env.KV = math.Floor((float64(j) + (float64(v)+g.next())/float64(per)) * span)
					kept += attnexpr.Eval(m, env)
				}
			}
			out.Kept[i*cells+j] = kept / float64(per*per)
		}
	}
	return out
}

// constraints are what the rules say about an sdpa's expressions.
func (a Attention) constraints(r *Resolved) []BlockFinding {
	if !a.Expressions() {
		return nil
	}
	var out []BlockFinding
	if a.MaskExpr != nil {
		p := a.probe()
		switch {
		case p.kept == 0:
			out = append(out, BlockFinding{
				ID: "SDPA-04", Severity: "error", Param: "mask",
				Message: fmt.Sprintf("The mask keeps no score in the first %d positions: nothing is attended to.",
					probeLength),
				Hint: "A score counts only when causal, window and the mask all allow it, so a mask " +
					"that contradicts either of the other two keeps nothing.",
			})
		case len(p.empty) > 0:
			out = append(out, BlockFinding{
				ID: "SDPA-05", Severity: "warning", Param: "mask",
				Message: fmt.Sprintf("The mask leaves %s with no key to attend to.", positions(p.empty)),
				Hint: "A softmax over nothing has no answer. FlexAttention returns zeros for such a " +
					"query, and so does the generated code, but nothing flows back through it. " +
					"Keeping the diagonal, kv == q, is the usual way out.",
			})
		}
	}
	if r.Bool("flash") {
		out = append(out, BlockFinding{
			ID: "SDPA-06", Severity: "info", Param: expressionParam(a),
			Message: "The mask and score expressions are counted as FlexAttention runs them: one fused " +
				"kernel that skips the blocks the mask removes and changes each score inside, so the " +
				"score matrix is never kept for the backward pass.",
			Hint: "The generated model compiles flex_attention on CUDA. Anywhere else, or where it " +
				"cannot compile, it applies the same expressions to the whole score matrix, which is " +
				"exact but unfused, and is what a CPU verifies and profiles.",
		})
	}
	return out
}

func expressionParam(a Attention) string {
	if a.MaskExpr != nil {
		return "mask"
	}
	return "score"
}

// positions names a set of query positions as ranges: "queries 0 to 15 and 32".
func positions(qs []int) string {
	var parts []string
	for i := 0; i < len(qs); {
		j := i
		for j+1 < len(qs) && qs[j+1] == qs[j]+1 {
			j++
		}
		if j == i {
			parts = append(parts, fmt.Sprint(qs[i]))
		} else {
			parts = append(parts, fmt.Sprintf("%d to %d", qs[i], qs[j]))
		}
		i = j + 1
		if len(parts) == 4 && i < len(qs) {
			parts = append(parts, "others")
			break
		}
	}
	noun := "the query at position "
	if len(qs) > 1 {
		noun = "the queries at positions "
	}
	if len(parts) == 1 {
		return noun + parts[0]
	}
	return noun + strings.Join(parts[:len(parts)-1], ", ") + " and " + parts[len(parts)-1]
}

// splitmix is a small seeded generator: the same design draws the same points
// in every engine, so a measured density is as reproducible as a formula.
type splitmix uint64

func seed(text string) splitmix {
	h := splitmix(1469598103934665603)
	for i := 0; i < len(text); i++ {
		h = (h ^ splitmix(text[i])) * 1099511628211
	}
	return h
}

func (s *splitmix) next() float64 {
	*s += 0x9e3779b97f4a7c15
	z := uint64(*s)
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	return float64(z>>11) / (1 << 53)
}

// spread picks k of the integers 0 to n-1, one drawn from each of k equal
// strata, or every one of them when there are no more than k.
func spread(n float64, k int, g *splitmix) []float64 {
	count := int(n)
	if count <= k {
		out := make([]float64, count)
		for i := range out {
			out[i] = float64(i)
		}
		return out
	}
	out := make([]float64, k)
	step := n / float64(k)
	for i := range out {
		out[i] = math.Min(n-1, math.Floor((float64(i)+g.next())*step))
	}
	return out
}
