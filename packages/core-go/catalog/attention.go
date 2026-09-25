package catalog

import (
	"fmt"
	"math"
	"sort"
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
	// Sinks is a learned score per head in the softmax's denominator.
	Sinks bool
	// Cross is attention to the source: every position of it, unmasked.
	Cross bool
	// MaskExpr and ScoreExpr are what the design wrote, nil where it wrote
	// nothing.
	MaskExpr  attnexpr.Node
	ScoreExpr attnexpr.Node
	// Documents is a mask that reads each position's document: what it keeps
	// depends on how the batch was packed, not only on positions.
	Documents bool
}

// Packing is how a training batch fills its rows: documents laid end to end
// and cut into rows T long, their lengths drawn from a gamma distribution of
// mean Mean tokens and coefficient of variation Spread. A spread of zero is
// every document the same length, one is exponential.
type Packing struct {
	Mean   float64 `json:"mean"`
	Spread float64 `json:"spread"`
}

// AttentionOf reads an sdpa's resolved parameters. `heads` is a constant
// once the expression is on a block, so it is replaced by the block's own.
func AttentionOf(r *Resolved) Attention {
	a := Attention{
		Causal: r.Bool("causal"),
		Window: math.Max(0, r.Num("window")),
		Cap:    r.Num("logit_softcap"),
		Heads:  r.Num("heads"),
		Sinks:  r.Bool("sinks"),
		Cross:  r.Bool("cross"),
	}
	if m := compiled(r.Str("mask"), attnexpr.Mask); m != nil {
		a.MaskExpr = attnexpr.With(m, "heads", a.Heads)
		// A mask that is always true once heads is known masks nothing.
		if b, ok := a.MaskExpr.(attnexpr.Bool); ok && b.V {
			a.MaskExpr = nil
		}
		a.Documents = len(attnexpr.Tables(a.MaskExpr)) > 0
	}
	if s := compiled(r.Str("score"), attnexpr.Score); s != nil {
		a.ScoreExpr = attnexpr.With(s, "heads", a.Heads)
	}
	return a
}

// TablesOf are the tensors a block's expressions read: each one an input of
// that name, which any block that carries the expressions down to its
// attention also has, and passes on. The mask's come first.
func TablesOf(r *Resolved) []string {
	out := DocumentsOf(r)
	for _, name := range attnexpr.Tables(compiled(r.Str("score"), attnexpr.Score)) {
		if !contains(out, name) {
			out = append(out, name)
		}
	}
	return out
}

// DocumentsOf are the tensors a block's mask reads, which can only be a
// documents input: each position's document, B T.
func DocumentsOf(r *Resolved) []string {
	return attnexpr.Tables(compiled(r.Str("mask"), attnexpr.Mask))
}

// tablePorts adds an input for each tensor an expression reads. What a score
// reads is any shape, because what it holds is between the design and the
// expression: a [buckets, heads] table for T5. What a mask reads is the
// documents, a document index for each position of each row.
func tablePorts(r *Resolved, in map[string]PortSpec) {
	documents := DocumentsOf(r)
	for _, name := range TablesOf(r) {
		if _, taken := in[name]; taken {
			continue
		}
		if contains(documents, name) {
			in[name] = PortSpec{Shape: "B T", Anchor: "side", Dtype: "int",
				Doc: "Each position's document, which the mask reads"}
			continue
		}
		in[name] = PortSpec{Shape: "*", Anchor: "side", Dtype: "inherit",
			Doc: "A tensor the score expression reads"}
	}
}

// Expressions reports whether the design wrote either expression, which is
// what decides the kernel: FlexAttention for any expression, FlashAttention
// for the switches alone.
func (a Attention) Expressions() bool { return a.MaskExpr != nil || a.ScoreExpr != nil }

// Flex reports whether the layer is FlexAttention's to run: an expression,
// or sinks, which FlashAttention 2 has no way to take and FlexAttention takes
// through the log-sum-exp it returns.
func (a Attention) Flex() bool { return a.Expressions() || a.Sinks }

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
//
// A mask that reads the documents keeps what it keeps of the batch it was
// given, so it is measured over rows drawn from the packing: nil is a row
// that is one document, which is what a request is.
func (a Attention) KeysPerQuery(T, B float64, pack *Packing) float64 {
	keys := a.structuredKeys(T)
	if a.MaskExpr != nil {
		keys *= a.keptShare(T, B, pack)
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
	// A packed batch is a draw of documents as well as of points, and what
	// varies most is which queries land in long documents, so it is measured
	// over many rows and few points in each: 1,024 rows cut from a stream of
	// at least as many documents, sixteen queries a row and sixteen keys a
	// query. Against a brute-force stream of twenty thousand rows that is
	// within a percent, where sixty-four rows of thirty-two queries each —
	// the same work — was off by five.
	packedRows    = 1024
	packedQueries = 16
	packedColumns = 16
	packedStream  = 1024
)

type shareKey struct {
	mask         string
	causal       bool
	window, T, B float64
	heads        float64
	usesH, usesB bool
	probe        int
	pack         Packing
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
//
// A mask that reads the documents is measured over rows drawn from the
// packing instead of over the batch: what a step costs on average is an
// average over the packings it will be given, whatever its batch size.
func (a Attention) keptShare(T, B float64, pack *Packing) float64 {
	if a.MaskExpr == nil || !(T >= 1) {
		return 1
	}
	packed := a.Documents && pack != nil
	key := shareKey{
		mask: attnexpr.String(a.MaskExpr), causal: a.Causal, window: a.Window, T: T,
		usesH: attnexpr.Uses(a.MaskExpr, "h"), usesB: attnexpr.Uses(a.MaskExpr, "b"),
	}
	if key.usesH {
		key.heads = a.Heads
	}
	if key.usesB && !packed {
		key.B = B
	}
	if packed {
		key.pack = *pack
	}
	cacheMu.Lock()
	if v, ok := shareCache[key]; ok {
		cacheMu.Unlock()
		return v
	}
	cacheMu.Unlock()

	rows, columns := sampleRows, sampleColumns
	if packed {
		rows, columns = packedQueries, packedColumns
	}
	switch {
	case key.usesH && packed:
		rows, columns = rows/2, columns/2
	case key.usesH:
		// Eight heads at a time, so a quarter of the points each.
		rows, columns = rows/2, columns/4
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
	env := attnexpr.Env{Heads: a.Heads}
	switch {
	case packed:
		p := drawPacking(T, *pack, &g)
		batch = make([]float64, len(p.starts))
		for i := range batch {
			batch[i] = float64(i)
		}
		env.Table = p.document
	case a.Documents:
		env.Table = oneDocument
	}
	var kept, total float64
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
				if packed {
					kept += bandedKeys(a.MaskExpr, env, lo, hi, columns, &g)
					continue
				}
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

// bandedKeys estimates how many of the keys lo to hi a mask keeps, from a few
// points in each of a run of bands that widen by four going back from the
// query: the nearest four keys, the twelve before them, the forty-eight
// before those, and so on to lo.
//
// Unbiased for any mask, like the even spread, and far steadier for one that
// keeps documents apart: what such a mask keeps runs back from the query to
// the start of its document, and the even spread finds a sixty-four-token
// document with one point in several hundred or with none. Banded, the band
// the document starts in is never more than three times as wide as what it
// keeps.
func bandedKeys(m attnexpr.Node, env attnexpr.Env, lo, hi float64, points int, g *splitmix) float64 {
	length := hi - lo + 1
	edges := []float64{0}
	for d := 4.0; d < length; d *= 4 {
		edges = append(edges, d)
	}
	edges = append(edges, length)
	per := max(1, points/(len(edges)-1))
	kept := 0.0
	for i := 1; i < len(edges); i++ {
		near, width := edges[i-1], edges[i]-edges[i-1]
		if width <= float64(per) {
			for d := near; d < near+width; d++ {
				env.KV = hi - d
				kept += attnexpr.Eval(m, env)
			}
			continue
		}
		step := width / float64(per)
		hits := 0.0
		for j := 0; j < per; j++ {
			env.KV = hi - near - math.Floor((float64(j)+g.next())*step)
			hits += attnexpr.Eval(m, env)
		}
		kept += hits * step
	}
	return kept
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
	// Checked as one document, the one case every packing has: a mask that
	// keeps nothing even then keeps nothing ever.
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
	env := attnexpr.Env{Heads: a.Heads, Table: oneDocument}
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
		Density: a.KeysPerQuery(T, B, nil) / T,
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
	// Drawn as one document for now: a packing drawn here is the editor's
	// next step, and a mask read as NaN everywhere would draw nothing at all.
	env := attnexpr.Env{H: head, Heads: a.Heads, Table: oneDocument}
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
	if a.Cross {
		// Everything that decides which scores count, or changes them, is
		// about positions in one sequence; across two it means nothing yet.
		var lost []string
		if a.Causal {
			lost = append(lost, "causal")
		}
		if a.Window > 0 {
			lost = append(lost, "window")
		}
		if a.Cap != 0 {
			lost = append(lost, "logit_softcap")
		}
		if a.MaskExpr != nil {
			lost = append(lost, "mask")
		}
		if a.ScoreExpr != nil {
			lost = append(lost, "score")
		}
		if a.Sinks {
			lost = append(lost, "sinks")
		}
		if len(lost) == 0 {
			return nil
		}
		return []BlockFinding{{
			ID: "SDPA-07", Severity: "error", Param: lost[0],
			Message: "Cross-attention sees every source position, so " + strings.Join(lost, ", ") +
				" would mean nothing here.",
			Hint: "Those belong to self-attention, where a query and a key are positions in one sequence.",
		}}
	}
	if !a.Flex() {
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
		message := "The mask and score expressions are counted as FlexAttention runs them: one " +
			"fused kernel that skips the blocks the mask removes and changes each score inside"
		if a.Sinks {
			if a.Expressions() {
				message = "The mask and score expressions and the sinks are counted as FlexAttention " +
					"runs them: one fused kernel that skips the blocks the mask removes and changes each " +
					"score inside, with the sinks applied to the log-sum-exp it returns"
			} else {
				message = "The sinks are counted as FlexAttention runs them: one fused kernel that skips " +
					"the blocks the mask removes, with the sinks applied to the log-sum-exp it returns"
			}
		}
		out = append(out, BlockFinding{
			ID: "SDPA-06", Severity: "info", Param: expressionParam(a),
			Message: message + ", so the score matrix is never kept for the backward pass.",
			Hint: "The generated model compiles flex_attention on CUDA. Anywhere else, or where it " +
				"cannot compile, it computes the same attention over the whole score matrix, which is " +
				"exact but unfused, and is what a CPU verifies and profiles.",
		})
	}
	return out
}

func expressionParam(a Attention) string {
	switch {
	case a.MaskExpr != nil:
		return "mask"
	case a.ScoreExpr != nil:
		return "score"
	}
	return "sinks"
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

// oneDocument answers a documents read for a row that holds one document:
// every position is document zero.
func oneDocument(_ string, at []float64) float64 {
	if len(at) != 2 || at[1] < 0 {
		return math.NaN()
	}
	return 0
}

// packing is rows cut from a stream of documents: for each row, the positions
// at which a new document begins, in order. The first position is never one:
// whatever a row begins with is its first document, whole or not.
type packing struct {
	T      float64
	starts [][]float64
}

// document is the index, within its row, of the document at a position.
func (p packing) document(_ string, at []float64) float64 {
	if len(at) != 2 {
		return math.NaN()
	}
	row, pos := at[0], at[1]
	if row < 0 || int(row) >= len(p.starts) || row != math.Floor(row) ||
		pos < 0 || pos >= p.T || pos != math.Floor(pos) {
		return math.NaN()
	}
	s := p.starts[int(row)]
	return float64(sort.SearchFloat64s(s, pos+0.5))
}

// drawPacking lays documents end to end in a circle and cuts rows out of it.
//
// The lengths are one from each of several hundred equal strata of the
// distribution, shuffled, so the stream holds the distribution rather than
// whatever one draw of it happened to be: the cost of a packing turns on the
// long documents, and a random handful of those is the noisiest thing about
// it. Each row starts at a point drawn from its own stratum of the circle, so
// a row begins inside a document as often as a real stream's would, and a
// document is as likely to be the one a row begins in as its length makes it.
// That is what makes long documents count for more than their number: a
// token is more often in a long document than a short one.
func drawPacking(T float64, pack Packing, g *splitmix) packing {
	docs := int(math.Max(packedStream, math.Ceil(4*T/pack.Mean)))
	lengths := make([]float64, docs)
	for i := range lengths {
		u := (float64(i) + g.next()) / float64(docs)
		lengths[i] = math.Max(1, math.Round(gammaQuantile(u, pack.Mean, pack.Spread)))
	}
	for i := docs - 1; i > 0; i-- {
		j := int(g.next() * float64(i+1))
		lengths[i], lengths[j] = lengths[j], lengths[i]
	}
	begins := make([]float64, docs)
	total := 0.0
	for i, l := range lengths {
		begins[i] = total
		total += l
	}
	out := packing{T: T, starts: make([][]float64, packedRows)}
	for r := range out.starts {
		start := math.Floor((float64(r) + g.next()) / packedRows * total)
		i := sort.SearchFloat64s(begins, start+1)
		var s []float64
		for k := 0; k < docs*int(2+T/total); k++ {
			at := begins[(i+k)%docs] + total*float64((i+k)/docs) - start
			if at >= T {
				break
			}
			s = append(s, at)
		}
		out.starts[r] = s
	}
	return out
}

// gammaQuantile is the u-th quantile of a gamma distribution of the given mean
// and coefficient of variation: the length below which a share u of documents
// fall.
func gammaQuantile(u, mean, cv float64) float64 {
	if cv <= 0 {
		return mean
	}
	shape, scale := 1/(cv*cv), mean*cv*cv
	if shape == 1 {
		return -scale * math.Log(1-u)
	}
	lo, hi := 0.0, mean
	for lowerGamma(shape, hi/scale) < u {
		hi *= 2
	}
	for i := 0; i < 64; i++ {
		mid := (lo + hi) / 2
		if lowerGamma(shape, mid/scale) < u {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// lowerGamma is the regularized lower incomplete gamma function P(a, x): the
// share of a gamma distribution of shape a and unit scale that lies below x.
// A series below a+1 and a continued fraction above, as Numerical Recipes has
// them.
func lowerGamma(a, x float64) float64 {
	if x <= 0 {
		return 0
	}
	lg, _ := math.Lgamma(a)
	front := math.Exp(-x + a*math.Log(x) - lg)
	if x < a+1 {
		sum, term := 1/a, 1/a
		for n := 1; n < 500; n++ {
			term *= x / (a + float64(n))
			sum += term
			if math.Abs(term) < math.Abs(sum)*1e-15 {
				break
			}
		}
		return sum * front
	}
	const tiny = 1e-300
	b := x + 1 - a
	c, d := 1/tiny, 1/b
	h := d
	for n := 1; n < 500; n++ {
		an := -float64(n) * (float64(n) - a)
		b += 2
		d = an*d + b
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = b + an/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		step := d * c
		h *= step
		if math.Abs(step-1) < 1e-15 {
			break
		}
	}
	return 1 - front*h
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
