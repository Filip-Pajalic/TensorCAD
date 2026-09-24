package catalog

import (
	"math"
	"strings"
	"testing"

	"github.com/tensorcad/core/ir"
)

// sdpaWith resolves an sdpa with these parameters over a small symbol table.
func sdpaWith(t *testing.T, params map[string]any) (*BlockDef, *Resolved) {
	t.Helper()
	def := Builtin["sdpa"]
	raw := map[string]any{"heads": 8.0, "kv_heads": 8.0, "head_dim": 64.0}
	for k, v := range params {
		raw[k] = v
	}
	doc := &ir.Doc{Symbols: map[string]ir.SymbolDef{
		"P": {Kind: "literal", Number: 256, HasNumber: true},
		"W": {Kind: "literal", Number: 1024, HasNumber: true},
		"C": {Kind: "literal", Number: 512, HasNumber: true},
	}}
	r := ResolveNodeParams(def, raw, ir.ResolveSymbols(doc))
	if len(r.Errors) > 0 {
		t.Fatalf("%v: %v", params, r.Errors)
	}
	return def, r
}

func keysOf(t *testing.T, params map[string]any, T float64) float64 {
	t.Helper()
	def, r := sdpaWith(t, params)
	f := def.Flops(r, AnalysisCtx{T: T, B: 1, Bytes: 2, Flash: true})
	return f.FwdSeq / (4 * 8 * 64)
}

func near(t *testing.T, what string, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance*math.Abs(want) {
		t.Errorf("%s: %v, want %v (within %v%%)", what, got, want, tolerance*100)
	}
}

// Causal attention keeps half the keys, exactly as it always has.
func TestCausalIsHalf(t *testing.T) {
	if got := keysOf(t, map[string]any{"causal": true}, 8192); got != 4096 {
		t.Errorf("causal keys per query at 8k: %v", got)
	}
	if got := keysOf(t, map[string]any{"causal": false}, 8192); got != 8192 {
		t.Errorf("bidirectional keys per query at 8k: %v", got)
	}
}

// A causal window of W keeps min(q + 1, W) keys at q: W - W²/2T on average once
// the sequence outgrows it. It used to be counted at W/2, as though the window
// were halved again for being causal — Gemma 2's local layers at 8k were
// counted at 2,048 keys a query where they attend to 3,072.
func TestACausalWindowIsItsWidth(t *testing.T) {
	cases := []struct{ T, W, want float64 }{
		{8192, 4096, 4096 - 4096.0*4096/(2*8192)},
		{131072, 1024, 1024 - 1024.0*1024/(2*131072)},
		// A window at least as long as the sequence is plain causal.
		{4096, 4096, 2048},
		{2048, 4096, 1024},
	}
	for _, c := range cases {
		if got := keysOf(t, map[string]any{"causal": true, "window": c.W}, c.T); got != c.want {
			t.Errorf("T=%v W=%v: %v keys, want %v", c.T, c.W, got, c.want)
		}
	}
	// And the profiler's count ignores the window, as it ignores the causal
	// half: the operator's shape does not depend on either.
	def, r := sdpaWith(t, map[string]any{"causal": true, "window": 1024.0})
	if f := def.Flops(r, AnalysisCtx{T: 8192, B: 1}); f.FwdSeqUnmasked != 4*8192*8*64 {
		t.Errorf("unmasked %v", f.FwdSeqUnmasked)
	}
}

// Writing causal and the window out as a mask is the same design, give or take
// the diagonal: the switches are counted in the continuous limit, and the
// expression is counted by evaluating it.
func TestTheSwitchesAreTheMasksTheyAre(t *testing.T) {
	for _, T := range []float64{2048, 8192, 32768} {
		sugar := keysOf(t, map[string]any{"causal": true, "window": "W"}, T)
		spelled := keysOf(t, map[string]any{"causal": false, "mask": "kv <= q and q - kv < W"}, T)
		near(t, "a causal window written as a mask", spelled, sugar, 0.01)
		near(t, "causal written as a mask",
			keysOf(t, map[string]any{"causal": false, "mask": "kv <= q"}, T),
			keysOf(t, map[string]any{"causal": true}, T), 0.002)
	}
}

// Masks with a known answer, measured.
func TestAMaskIsCountedAtWhatItKeeps(t *testing.T) {
	T := 8192.0
	// Prefix-LM: the first P positions bidirectional, causal after. The
	// prefix square is P², and every later query's prefix is already inside
	// its causal half, so T²/2 + P²/2, give or take the diagonal.
	prefix := keysOf(t, map[string]any{"causal": false, "mask": "kv <= q or kv < P"}, T)
	near(t, "prefix-LM", prefix, (T*T/2+256*256/2)/T, 0.002)

	// Chunked attention, causal within chunks of C: C/2 on average.
	chunked := keysOf(t, map[string]any{"causal": true, "mask": "floor(q / C) == floor(kv / C)"}, T)
	near(t, "chunked", chunked, 512.0/2, 0.02)

	// Every fourth key, causal: a periodic mask the sampler must not alias.
	strided := keysOf(t, map[string]any{"causal": true, "mask": "(q - kv) % 4 == 0"}, T)
	near(t, "strided", strided, T/2/4, 0.02)

	// Half the heads see everything, half only themselves.
	perHead := keysOf(t, map[string]any{"causal": true, "mask": "h < heads / 2 or kv == q"}, T)
	near(t, "per head", perHead, (T/2+1)/2, 0.02)
}

// The same design measures the same every time, in every engine: the points
// are drawn from a generator seeded by the mask.
func TestAMeasuredMaskIsReproducible(t *testing.T) {
	params := map[string]any{"causal": true, "mask": "(q - kv) % 3 != 1"}
	first := keysOf(t, params, 100000)
	clear(shareCache)
	if again := keysOf(t, params, 100000); again != first {
		t.Errorf("%v then %v", first, again)
	}
}

// A score expression costs its arithmetic over the scores the mask keeps, and a
// cap written as an expression costs what the parameter does.
func TestAScoreCostsItsArithmetic(t *testing.T) {
	ctx := AnalysisCtx{T: 4096, B: 1, Bytes: 2, Flash: true}
	def, capped := sdpaWith(t, map[string]any{"causal": true, "logit_softcap": 50.0})
	_, written := sdpaWith(t, map[string]any{"causal": true, "score": "50 * tanh(score / 50)"})
	if a, b := def.Flops(capped, ctx).Elementwise, def.Flops(written, ctx).Elementwise; a != b || a != softcapCost*2048*8 {
		t.Errorf("a cap costs %v, written out %v", a, b)
	}
	_, alibi := sdpaWith(t, map[string]any{"causal": true, "score": "score - (q - kv) / 8"})
	// A subtract, a subtract and a divide.
	if got := def.Flops(alibi, ctx).Elementwise; got != 3*2048*8 {
		t.Errorf("a linear bias costs %v", got)
	}
}

func findingIDs(fs []BlockFinding) string {
	var ids []string
	for _, f := range fs {
		ids = append(ids, f.ID+":"+f.Severity)
	}
	return strings.Join(ids, " ")
}

func TestTheRulesReadTheMask(t *testing.T) {
	cases := []struct {
		params map[string]any
		want   string
	}{
		// Contradicts causal: nothing left.
		{map[string]any{"causal": true, "mask": "kv > q"}, "SDPA-04:error SDPA-06:info"},
		// Strictly causal: the first query has nothing.
		{map[string]any{"causal": false, "mask": "kv < q"}, "SDPA-05:warning SDPA-06:info"},
		{map[string]any{"causal": true, "mask": "kv < P or q - kv < W"}, "SDPA-06:info"},
		// Eager on purpose: counted as that, so nothing to say about kernels.
		{map[string]any{"causal": true, "score": "score / 2", "flash": false}, ""},
		// The cap alone is still FlashAttention's.
		{map[string]any{"causal": true, "logit_softcap": 30.0}, "SDPA-03:info"},
		{map[string]any{"causal": true, "logit_softcap": 30.0, "score": "score / 2"}, "SDPA-06:info"},
	}
	for _, c := range cases {
		def, r := sdpaWith(t, c.params)
		if got := findingIDs(def.Constraints(r)); got != c.want {
			t.Errorf("%v: %q, want %q", c.params, got, c.want)
		}
	}
	def, r := sdpaWith(t, map[string]any{"causal": false, "mask": "kv < q"})
	if msg := def.Constraints(r)[0].Message; !strings.Contains(msg, "the query at position 0 ") {
		t.Errorf("names %q", msg)
	}
}

func TestAnExpressionIsResolvedAgainstTheDesign(t *testing.T) {
	_, r := sdpaWith(t, map[string]any{"mask": "kv < P or kv <= q"})
	if got := r.Str("mask"); got != "kv < 256 or kv <= q" {
		t.Errorf("resolved to %q", got)
	}
	def := Builtin["sdpa"]
	for text, want := range map[string]string{
		"kv < T":    `"T" is not`,
		"kv <= q +": "ends too soon",
		"score":     "a mask is true or false",
	} {
		raw := map[string]any{"heads": 8.0, "kv_heads": 8.0, "head_dim": 64.0, "mask": text}
		r := ResolveNodeParams(def, raw, ir.ResolveSymbols(&ir.Doc{Symbols: map[string]ir.SymbolDef{}}))
		if len(r.Errors) != 1 || !strings.Contains(r.Errors[0], want) {
			t.Errorf("%q: %v", text, r.Errors)
		}
		if _, set := r.P["mask"]; set {
			t.Errorf("%q: a mask that did not compile was kept", text)
		}
	}
}

func TestTheGridIsTheBlockMask(t *testing.T) {
	_, r := sdpaWith(t, map[string]any{"causal": true, "window": 1024.0})
	g := AttentionOf(r).Grid(8192, 1, 0)
	if g.Cells != 32 || g.Span != 256 || len(g.Kept) != 32*32 {
		t.Fatalf("%d cells of %v", g.Cells, g.Span)
	}
	at := func(i, j int) float64 { return g.Kept[i*g.Cells+j] }
	// Above the diagonal nothing, on it about half, well inside the window
	// everything, past it nothing.
	if at(0, 5) != 0 || at(10, 10) < 0.3 || at(10, 10) > 0.7 || at(10, 8) != 1 || at(20, 5) != 0 {
		t.Errorf("row 10: %v", g.Kept[10*32:11*32])
	}
	if g.Mask != "kv <= q and q - kv < 1024" {
		t.Errorf("mask %q", g.Mask)
	}
	near(t, "density", g.Density, (1024-1024.0*1024/(2*8192))/8192, 1e-12)

	// A sequence shorter than 32 is drawn a position a block.
	g = AttentionOf(r).Grid(11, 1, 0)
	if g.Cells != 11 || g.Span != 1 || g.Kept[0*11+1] != 0 || g.Kept[5*11+5] != 1 {
		t.Errorf("%+v", g)
	}
}

// Expressions survive into the kernel through both composites, as the
// resolver understood them.
func TestCompositesPassTheExpressionsDown(t *testing.T) {
	doc := &ir.Doc{Symbols: map[string]ir.SymbolDef{"P": {Kind: "literal", Number: 16, HasNumber: true}}}
	symbols := ir.ResolveSymbols(doc)
	block := Builtin["transformer_block"]
	r := ResolveNodeParams(block, map[string]any{
		"d_model": 64.0, "heads": 4.0, "kv_heads": 4.0, "head_dim": 16.0, "ffn_hidden": 128.0,
		"mask": "kv < P or kv <= q", "score": "score * 2",
	}, symbols)
	exp, ok := Expand(block, r.RawFull, r)
	if !ok {
		t.Fatal("no expansion")
	}
	var attn *ir.NodeDef
	for i := range exp.Nodes {
		if exp.Nodes[i].ID == "attn" {
			attn = &exp.Nodes[i]
		}
	}
	if attn.Params["mask"] != "kv < 16 or kv <= q" || attn.Params["score"] != "score * 2" {
		t.Errorf("gqa_attention given %v, %v", attn.Params["mask"], attn.Params["score"])
	}
	gqa := Builtin["gqa_attention"]
	rg := ResolveNodeParams(gqa, attn.Params, symbols)
	inner, _ := Expand(gqa, rg.RawFull, rg)
	for _, n := range inner.Nodes {
		if n.Type == "sdpa" && n.Params["mask"] != "kv < 16 or kv <= q" {
			t.Errorf("sdpa given %v", n.Params["mask"])
		}
	}
	// A mask that did not compile is reported once, by the block it was
	// written on, and goes no further.
	bad := ResolveNodeParams(gqa, map[string]any{
		"d_model": 64.0, "heads": 4.0, "kv_heads": 4.0, "head_dim": 16.0, "mask": "kv <",
	}, symbols)
	inner, _ = Expand(gqa, bad.RawFull, bad)
	for _, n := range inner.Nodes {
		if _, has := n.Params["mask"]; n.Type == "sdpa" && has {
			t.Errorf("a broken mask was passed down: %v", n.Params["mask"])
		}
	}
}
