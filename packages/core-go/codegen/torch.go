// Package codegen emits PyTorch from a design.
//
// One generic emitter walks a graph and produces an nn.Module. Composites and
// repeat containers become their own classes, named after the block, so the
// output reads like code somebody wrote rather than a flattened trace. Because
// there is a single code path, generated code cannot drift from the analysis:
// both consume the same expansion.
//
// The emitted file has no dependency beyond PyTorch itself.
package codegen

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
	"github.com/tensorcad/core/infer"
	"github.com/tensorcad/core/ir"
	"github.com/tensorcad/core/shapes"
)

// Options steer the emitter.
type Options struct {
	// ClassName is the name of the top-level module; empty derives it from the
	// design's own name.
	ClassName string
	// NoSmokeTest leaves out the __main__ block that instantiates the model and
	// checks its size. The block is emitted by default.
	NoSmokeTest bool
	// MoeDispatch is how a mixture-of-experts layer routes tokens.
	//
	// "sparse", the default, gathers the rows each expert was given, which is
	// what you want for speed but uses nonzero, whose output shape depends on
	// the data. That makes it untraceable by torch.export.
	//
	// "dense" runs every expert over every token and weights the results by
	// whether the expert was selected. It computes the same thing, costs
	// experts/top_k times as much, and traces cleanly. Use it to verify a
	// design, not to train one.
	MoeDispatch string
	// InitStd is the standard deviation for the weight initialization. Nil
	// takes the default of 0.02, and zero leaves PyTorch's own defaults alone.
	//
	// This matters more than it looks. nn.Embedding defaults to a unit normal,
	// which starts a language model at a cross-entropy of a few hundred instead
	// of ln(vocab), and it takes a long time to recover.
	InitStd *float64
}

// WireOptions is Options as a client sends it, and the only spelling of these
// settings that a client should know.
//
// It differs from Options in one way, deliberately: the smoke test is named for
// what a caller decides ("include it") rather than for what the emitter then
// skips. Everything is optional, so an absent field and a zero one have to be
// distinguishable, which is what the pointers are for.
type WireOptions struct {
	ClassName        string   `json:"className,omitempty"`
	IncludeSmokeTest *bool    `json:"includeSmokeTest,omitempty"`
	MoeDispatch      string   `json:"moeDispatch,omitempty"`
	InitStd          *float64 `json:"initStd,omitempty"`
}

// Options is what the emitter reads.
func (w WireOptions) Options() Options {
	o := Options{ClassName: w.ClassName, MoeDispatch: w.MoeDispatch, InitStd: w.InitStd}
	if w.IncludeSmokeTest != nil && !*w.IncludeSmokeTest {
		o.NoSmokeTest = true
	}
	return o
}

// Wire is the inverse, for writing down the settings an answer was produced
// under in the words a client would have used.
func Wire(o Options) WireOptions {
	w := WireOptions{ClassName: o.ClassName, MoeDispatch: o.MoeDispatch, InitStd: o.InitStd}
	if o.NoSmokeTest {
		no := false
		w.IncludeSmokeTest = &no
	}
	return w
}

// File is one emitted file.
type File struct {
	Path     string `json:"path"`
	Contents string `json:"contents"`
}

// Generated is everything the emitter produced.
type Generated struct {
	Files []File `json:"files"`
	// Warnings is never nil: an empty list means the design generated cleanly,
	// and null would mean the same thing in a way every reader has to handle.
	Warnings []string `json:"warnings"`
}

// ---------------------------------------------------------------------------
// Helpers emitted only when the design uses them
// ---------------------------------------------------------------------------

const helperRope = `class RotaryEmbedding(nn.Module):
    """Rotary position embedding over the last dimension of (B, H, T, head_dim)."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[-2]
        pos = torch.arange(seq, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(x.device))
        cos = freqs.cos().to(x.dtype)[None, None, :, :]
        sin = freqs.sin().to(x.dtype)[None, None, :, :]
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
`

const helperWindow = "def sliding_window_mask(seq: int, window: int, device, dtype) -> torch.Tensor:\n" +
	"    \"\"\"Causal mask that also forbids attending further back than `window` tokens.\"\"\"\n" +
	"    i = torch.arange(seq, device=device)\n" +
	"    allowed = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)\n" +
	"    mask = torch.zeros(seq, seq, device=device, dtype=dtype)\n" +
	"    return mask.masked_fill(~allowed, float(\"-inf\"))\n"

const helperSoftcap = "def softcap_attention(q, k, v, cap, mask=None, scale=None, enable_gqa=False):\n" +
	"    \"\"\"Attention whose scores are bounded to +/-cap by a tanh.\n" +
	"\n" +
	"    Not `F.scaled_dot_product_attention`: the cap applies to the score matrix,\n" +
	"    and a fused kernel never materializes one. This is the eager form, and it\n" +
	"    costs the memory the fused kernel would have saved.\n" +
	"    \"\"\"\n" +
	"    if enable_gqa and k.shape[-3] != q.shape[-3]:\n" +
	"        k = k.repeat_interleave(q.shape[-3] // k.shape[-3], dim=-3)\n" +
	"        v = v.repeat_interleave(q.shape[-3] // v.shape[-3], dim=-3)\n" +
	"    if scale is None:\n" +
	"        scale = q.shape[-1] ** -0.5\n" +
	"    scores = torch.tanh((q @ k.transpose(-2, -1)) * scale / cap) * cap\n" +
	"    if mask is not None:\n" +
	"        scores = scores + mask\n" +
	"    return torch.softmax(scores, dim=-1).to(v.dtype) @ v\n"

const helperCausal = "def causal_mask(seq: int, device, dtype) -> torch.Tensor:\n" +
	"    \"\"\"Additive mask forbidding a token from attending to anything after it.\"\"\"\n" +
	"    i = torch.arange(seq, device=device)\n" +
	"    mask = torch.zeros(seq, seq, device=device, dtype=dtype)\n" +
	"    return mask.masked_fill(i[:, None] < i[None, :], float(\"-inf\"))\n"

const helperShift = "def shift_sequence(x: torch.Tensor, by: int) -> torch.Tensor:\n" +
	"    \"\"\"Move a sequence `by` positions earlier, zero-filling the end.\n" +
	"\n" +
	"    What a multi-token predictor reads: at depth k it wants the embedding of\n" +
	"    the token k ahead, which over a whole training sequence is this. A slice\n" +
	"    and a concatenation rather than a roll, so nothing wraps around from the\n" +
	"    end of the sequence to the start of it.\n" +
	"    \"\"\"\n" +
	"    if by <= 0:\n" +
	"        return x\n" +
	"    pad = torch.zeros_like(x[..., :by, :])\n" +
	"    return torch.cat([x[..., by:, :], pad], dim=-2)\n"

const helperSsd = "class SSDScan(nn.Module):\n" +
	"    \"\"\"Mamba-2 state-space scan.\n" +
	"\n" +
	"    A readable sequential reference so the generated file runs unmodified. For a\n" +
	"    real training run, swap this for the fused kernel in `mamba_ssm`; it is the\n" +
	"    same recurrence but orders of magnitude faster.\n" +
	"    \"\"\"\n" +
	"\n" +
	"    def __init__(self, heads: int, head_dim: int, state: int, groups: int):\n" +
	"        super().__init__()\n" +
	"        self.heads, self.head_dim, self.state, self.groups = heads, head_dim, state, groups\n" +
	"        self.A_log = nn.Parameter(torch.zeros(heads))\n" +
	"        self.D = nn.Parameter(torch.ones(heads))\n" +
	"        self.dt_bias = nn.Parameter(torch.zeros(heads))\n" +
	"\n" +
	"    def forward(self, xbc: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:\n" +
	"        b, t, _ = xbc.shape\n" +
	"        d_inner = self.heads * self.head_dim\n" +
	"        gs = self.groups * self.state\n" +
	"        x, bs, cs = torch.split(xbc, [d_inner, gs, gs], dim=-1)\n" +
	"        x = x.view(b, t, self.heads, self.head_dim)\n" +
	"        rep = self.heads // self.groups\n" +
	"        bs = bs.view(b, t, self.groups, self.state).repeat_interleave(rep, dim=2)\n" +
	"        cs = cs.view(b, t, self.groups, self.state).repeat_interleave(rep, dim=2)\n" +
	"        dt = F.softplus(dt + self.dt_bias)\n" +
	"        a = -torch.exp(self.A_log)\n" +
	"        h = torch.zeros(b, self.heads, self.head_dim, self.state, device=xbc.device, dtype=torch.float32)\n" +
	"        out = []\n" +
	"        for i in range(t):\n" +
	"            decay = torch.exp(dt[:, i] * a)[:, :, None, None].float()\n" +
	"            h = h * decay + (dt[:, i][:, :, None, None] * x[:, i][..., None] * bs[:, i][:, :, None, :]).float()\n" +
	"            y = (h * cs[:, i][:, :, None, :].float()).sum(-1).to(x.dtype) + self.D[None, :, None] * x[:, i]\n" +
	"            out.append(y)\n" +
	"        return torch.stack(out, dim=1).reshape(b, t, d_inner)\n"

// activationCall is how each nonlinearity is written in Python.
var activationCall = map[string]string{
	"silu":      "F.silu(%s)",
	"swish":     "F.silu(%s)",
	"gelu":      "F.gelu(%s)",
	"gelu_tanh": `F.gelu(%s, approximate="tanh")`,
	"relu":      "F.relu(%s)",
	"relu2":     "F.relu(%s).square()",
	"tanh":      "torch.tanh(%s)",
	"sigmoid":   "torch.sigmoid(%s)",
	"identity":  "%s",
}

func activation(kind, value string) string {
	tmpl, ok := activationCall[kind]
	if !ok {
		tmpl = activationCall["identity"]
	}
	return fmt.Sprintf(tmpl, value)
}

func activationOr(kind, fallback, value string) string {
	if _, ok := activationCall[kind]; !ok {
		kind = fallback
	}
	return activation(kind, value)
}

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

type emitted struct {
	name string
	code string
}

type ctx struct {
	doc *ir.Doc
	// cat is the document's catalog, so blocks it defines itself generate too.
	cat      catalog.Catalog
	symbols  *ir.SymbolTable
	warnings []string
	// classes are the deduplicated classes, in dependency order.
	classes      []emitted
	byKey        map[string]string
	usedNames    map[string]bool
	needsRope    bool
	needsWindow  bool
	needsSoftcap bool
	needsShift   bool
	needsCausal  bool
	needsSsd     bool
	moeDispatch  string
}

func (c *ctx) warn(format string, args ...any) {
	c.warnings = append(c.warnings, fmt.Sprintf(format, args...))
}

func (c *ctx) className(base string) string {
	p := pascal(base)
	name := p
	if name == "" {
		name = "Block"
	}
	for n := 2; c.usedNames[name]; n++ {
		name = fmt.Sprintf("%s%d", p, n)
	}
	c.usedNames[name] = true
	return name
}

// order is a topological order of a graph's nodes.
func order(graph *ir.Graph) []ir.NodeDef {
	byID := make(map[string]*ir.NodeDef, len(graph.Nodes))
	for i := range graph.Nodes {
		byID[graph.Nodes[i].ID] = &graph.Nodes[i]
	}
	deps := make(map[string][]string, len(graph.Nodes))
	seen := map[string]bool{}
	for _, e := range graph.Edges {
		f, errF := ir.SplitEndpoint(e.From())
		t, errT := ir.SplitEndpoint(e.To())
		if errF != nil || errT != nil {
			continue
		}
		_, okF := byID[f.Node]
		_, okT := byID[t.Node]
		if !okF || !okT {
			continue
		}
		key := t.Node + "\x00" + f.Node
		if seen[key] {
			continue
		}
		seen[key] = true
		deps[t.Node] = append(deps[t.Node], f.Node)
	}

	var out []ir.NodeDef
	state := map[string]bool{}
	var visit func(id string)
	visit = func(id string) {
		if state[id] {
			return
		}
		state[id] = true
		for _, d := range deps[id] {
			visit(d)
		}
		if n, ok := byID[id]; ok {
			out = append(out, *n)
		}
	}
	for _, n := range graph.Nodes {
		visit(n.ID)
	}
	return out
}

type graphEmit struct {
	init    []string
	forward []string
	outputs map[string]string
	// outputOrder is the top-level output nodes in the order the graph lists
	// them, which is the order a model returns them in.
	outputOrder []string
}

type classInfo struct {
	name    string
	inputs  []string
	outputs []string
}

// portNames lists a boundary node's ports in the order the document wrote them.
//
// A Go map has no order, and here the order is the signature of the generated
// class: the arguments of its forward and the tuple it returns. It is recovered
// from the raw JSON the same way the symbol table's order is.
func portNames(node *ir.NodeDef, fallback string) []string {
	if node == nil {
		return []string{fallback}
	}
	names := orderedKeys(node.Params["ports"])
	if len(names) == 0 {
		return []string{fallback}
	}
	return names
}

// emitGraph emits the body of one graph. inputs maps the boundary port names to
// the variable names they arrive in.
func emitGraph(graph *ir.Graph, prefix string, inputs map[string]string, c *ctx) graphEmit {
	out := graphEmit{outputs: map[string]string{}}

	producers := map[string]string{}
	for _, e := range graph.Edges {
		if _, taken := producers[e.To()]; !taken {
			producers[e.To()] = e.From()
		}
	}

	// vars holds the variable carrying the value on a given output endpoint.
	vars := map[string]string{}
	valueOf := func(nodeID, port string) string {
		if v, ok := vars[nodeID+":"+port]; ok {
			return v
		}
		c.warn("%s: no value for port %q.", ir.JoinPath(prefix, nodeID), port)
		return "None"
	}
	inputVar := func(nodeID, port string) string {
		from, ok := producers[nodeID+":"+port]
		if !ok {
			c.warn("%s: input %q is not connected.", ir.JoinPath(prefix, nodeID), port)
			return "None"
		}
		f, err := ir.SplitEndpoint(from)
		if err != nil {
			return "None"
		}
		return valueOf(f.Node, f.Port)
	}

	for _, node := range order(graph) {
		path := ir.JoinPath(prefix, node.ID)
		def, known := c.cat[node.Type]
		if !known {
			c.warn("%s: unknown block type %q.", path, node.Type)
			continue
		}
		r := catalog.ResolveNodeParams(def, node.Params, c.symbols)
		attr := pyName(node.ID)
		outName := func(port string) string { return pyName(node.ID + "_" + port) }
		set := func(port, value string) { vars[node.ID+":"+port] = value }
		p := r.P

		switch node.Type {
		case "input":
			set("x", "ids")

		case "output":
			// Keyed by the node, not by the port: a design may have several
			// output nodes and they all call their port "x". Multi-token
			// prediction is the first thing that does — the model's own logits
			// and one more set per prediction depth — and keying by port meant
			// the last one silently won.
			out.outputs[node.ID] = inputVar(node.ID, "x")
			out.outputOrder = append(out.outputOrder, node.ID)

		case "boundary_in":
			for _, port := range orderedKeys(node.Params["ports"]) {
				v, ok := inputs[port]
				if !ok {
					v = "None"
				}
				set(port, v)
			}

		case "boundary_out":
			for _, port := range orderedKeys(node.Params["ports"]) {
				out.outputs[port] = inputVar(node.ID, port)
			}

		case "embedding":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Embedding(%s, %s)",
				attr, pyValue(p["vocab"]), pyValue(p["dim"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "ids")))
			set("y", outName("y"))

		case "pos_embedding":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Embedding(%s, %s)",
				attr, pyValue(p["max_seq"]), pyValue(p["dim"])))
			src := inputVar(node.ID, "x")
			out.forward = append(out.forward, fmt.Sprintf(
				"%s = %s + self.%s(torch.arange(%s.shape[1], device=%s.device))",
				outName("y"), src, attr, src, src))
			set("y", outName("y"))

		case "conv2d":
			groups := ""
			if r.Num("groups") > 1 {
				groups = ", groups=" + pyValue(p["groups"])
			}
			out.init = append(out.init, fmt.Sprintf(
				"self.%s = nn.Conv2d(%s, %s, kernel_size=%s, stride=%s, padding=%s%s, bias=%s)",
				attr, pyValue(p["in_channels"]), pyValue(p["out_channels"]),
				pyValue(p["kernel"]), pyValue(p["stride"]), pyValue(p["padding"]),
				groups, pyBool(p["bias"])))
			src := inputVar(node.ID, "x")
			out.forward = append(out.forward, fmt.Sprintf("%s = %s",
				outName("y"), activationOr(r.Str("act"), "identity", fmt.Sprintf("self.%s(%s)", attr, src))))
			set("y", outName("y"))

		case "maxpool2d":
			out.init = append(out.init, fmt.Sprintf(
				"self.%s = nn.MaxPool2d(kernel_size=%s, stride=%s, padding=%s)",
				attr, pyValue(p["kernel"]), pyValue(p["stride"]), pyValue(p["padding"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "shift":
			by := r.Num("by")
			if by == 0 {
				set("y", inputVar(node.ID, "x"))
				break
			}
			c.needsShift = true
			out.forward = append(out.forward, fmt.Sprintf("%s = shift_sequence(%s, %s)",
				outName("y"), inputVar(node.ID, "x"), pyNum(by)))
			set("y", outName("y"))

		case "flatten2d":
			out.forward = append(out.forward, fmt.Sprintf("%s = torch.flatten(%s, 1)",
				outName("y"), inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "learned_tokens":
			// A parameter, not a module: registered so it trains and moves with
			// the model, expanded to the batch at use. expand rather than repeat
			// keeps it a view, which is what the reference relies on.
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Parameter(torch.zeros(1, %s, %s))",
				attr, pyValue(p["count"]), pyValue(p["dim"])))
			// Expanded against the model's own input, which is the only tensor a
			// block with no inputs can learn the batch size from. `ids` is the
			// forward's parameter, whatever the design feeds it.
			tokens := "-1"
			if v, ok := p["tokens"]; ok && truthy(v) {
				tokens = pyValue(v)
			}
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s.expand(ids.shape[0], %s, -1)",
				outName("y"), attr, tokens))
			set("y", outName("y"))

		case "linear":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Linear(%s, %s, bias=%s)",
				attr, pyValue(p["in_features"]), pyValue(p["out_features"]), pyBool(p["bias"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "lm_head":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.Linear(%s, %s, bias=%s)",
				attr, pyValue(p["dim"]), pyValue(p["vocab"]), pyBool(p["bias"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			if cap := r.Num("softcap"); cap != 0 {
				// Gemma's bound on the logits. It changes the loss, not the
				// shapes, so it is one line after the projection.
				out.forward = append(out.forward, fmt.Sprintf(
					"%s = torch.tanh(%s / %s) * %s",
					outName("y"), outName("y"), pyNum(cap), pyNum(cap)))
			}
			set("y", outName("y"))

		case "rmsnorm":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.RMSNorm(%s, eps=%s)",
				attr, pyValue(p["dim"]), pyValue(p["eps"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "layernorm":
			out.init = append(out.init, fmt.Sprintf("self.%s = nn.LayerNorm(%s, eps=%s, bias=%s)",
				attr, pyValue(p["dim"]), pyValue(p["eps"]), pyBool(p["bias"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "activation":
			out.forward = append(out.forward, fmt.Sprintf("%s = %s",
				outName("y"), activationOr(r.Str("kind"), "silu", inputVar(node.ID, "x"))))
			set("y", outName("y"))

		case "add":
			out.forward = append(out.forward, fmt.Sprintf("%s = %s + %s",
				outName("y"), inputVar(node.ID, "a"), inputVar(node.ID, "b")))
			set("y", outName("y"))

		case "mul":
			out.forward = append(out.forward, fmt.Sprintf("%s = %s * %s",
				outName("y"), inputVar(node.ID, "a"), inputVar(node.ID, "b")))
			set("y", outName("y"))

		case "rearrange":
			out.forward = append(out.forward,
				emitRearrange(r, c, inputVar(node.ID, "x"), outName("y"), path)...)
			set("y", outName("y"))

		case "rope":
			c.needsRope = true
			out.init = append(out.init, fmt.Sprintf("self.%s = RotaryEmbedding(%s, theta=%s)",
				attr, pyValue(p["head_dim"]), pyValue(p["theta"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
				outName("y"), attr, inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "conv1d":
			src := inputVar(node.ID, "x")
			channels := pyValue(p["channels"])
			out.init = append(out.init, fmt.Sprintf(
				"self.%s = nn.Conv1d(%s, %s, %s, groups=%s, padding=%s, bias=%s)",
				attr, channels, channels, pyValue(p["kernel"]), channels,
				pyNum(r.Num("kernel")-1), pyBool(p["bias"])))
			// Convolve over time, then drop the padding the causal kernel added.
			out.forward = append(out.forward, fmt.Sprintf(
				"%s = self.%s(%s.transpose(1, 2))[..., : %s.shape[1]].transpose(1, 2)",
				outName("y"), attr, src, src))
			set("y", outName("y"))

		case "ssd_scan":
			c.needsSsd = true
			out.init = append(out.init, fmt.Sprintf("self.%s = SSDScan(%s, %s, %s, %s)",
				attr, pyValue(p["heads"]), pyValue(p["head_dim"]),
				pyValue(p["state"]), pyValue(p["groups"])))
			out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s, %s)",
				outName("y"), attr, inputVar(node.ID, "xbc"), inputVar(node.ID, "dt")))
			set("y", outName("y"))

		case "kv_latent_cache":
			// Analysis only: it marks what an inference engine would cache. In a
			// forward pass the compressed vector simply flows through.
			out.forward = append(out.forward, fmt.Sprintf("%s = %s  # cached latent",
				outName("y"), inputVar(node.ID, "x")))
			set("y", outName("y"))

		case "split":
			// Sizes are expressions over the design symbols, not plain numbers.
			splitCtx := infer.EvalCtxFor(r, c.symbols)
			raw, _ := p["sizes"].([]any)
			sizes := make([]string, len(raw))
			bad := false
			for i, v := range raw {
				n, ok := evalSize(v, splitCtx)
				if !ok {
					bad = true
					sizes[i] = "NaN"
					continue
				}
				sizes[i] = pyNum(n)
			}
			if bad {
				c.warn("%s: could not evaluate the split widths.", path)
			}
			names := make([]string, len(sizes))
			for i := range sizes {
				names[i] = outName(fmt.Sprintf("y%d", i))
			}
			out.forward = append(out.forward, fmt.Sprintf("%s = torch.split(%s, [%s], dim=-1)",
				strings.Join(names, ", "), inputVar(node.ID, "x"), strings.Join(sizes, ", ")))
			for i := range sizes {
				set(fmt.Sprintf("y%d", i), names[i])
			}

		case "concat":
			raw, _ := p["sizes"].([]any)
			parts := make([]string, len(raw))
			for i := range raw {
				parts[i] = inputVar(node.ID, fmt.Sprintf("y%d", i))
			}
			axis := "-1"
			if v, ok := asNumber(p["axis"]); ok {
				axis = pyNum(v)
			}
			out.forward = append(out.forward, fmt.Sprintf("%s = torch.cat([%s], dim=%s)",
				outName("y"), strings.Join(parts, ", "), axis))
			set("y", outName("y"))

		case "expand_heads":
			src := inputVar(node.ID, "x")
			out.forward = append(out.forward, fmt.Sprintf("%s = %s.unsqueeze(1).expand(-1, %s, -1, -1)",
				outName("y"), src, pyValue(p["heads"])))
			set("y", outName("y"))

		case "sdpa":
			q, k, v := inputVar(node.ID, "q"), inputVar(node.ID, "k"), inputVar(node.ID, "v")
			gqa := ""
			if r.Num("heads") != r.Num("kv_heads") {
				gqa = ", enable_gqa=True"
			}
			// A narrower value head means the kernel cannot infer the scale from
			// the query width, so it is passed explicitly.
			scale := ""
			if vd := r.Num("v_head_dim"); vd != 0 && vd != r.Num("head_dim") {
				scale = ", scale=" + jsToPrecision(1/math.Sqrt(r.Num("head_dim")), 12)
			}
			w := r.Num("window")
			cap := r.Num("logit_softcap")
			switch {
			case cap != 0:
				// A cap on the scores rules the fused kernel out, so the mask
				// has to be built here rather than left to `is_causal`.
				c.needsSoftcap = true
				mask := "None"
				switch {
				case w > 0:
					c.needsWindow = true
					out.forward = append(out.forward, fmt.Sprintf(
						"%s_mask = sliding_window_mask(%s.shape[-2], %s, %s.device, %s.dtype)",
						outName("y"), q, pyNum(w), q, q))
					mask = outName("y") + "_mask"
				case r.Bool("causal"):
					c.needsCausal = true
					out.forward = append(out.forward, fmt.Sprintf(
						"%s_mask = causal_mask(%s.shape[-2], %s.device, %s.dtype)",
						outName("y"), q, q, q))
					mask = outName("y") + "_mask"
				}
				capScale := "None"
				if scale != "" {
					capScale = strings.TrimPrefix(scale, ", scale=")
				}
				out.forward = append(out.forward, fmt.Sprintf(
					"%s = softcap_attention(%s, %s, %s, %s, mask=%s, scale=%s, enable_gqa=%s)",
					outName("y"), q, k, v, pyNum(cap), mask, capScale,
					pyBool(gqa != "")))
			case w > 0:
				c.needsWindow = true
				out.forward = append(out.forward, fmt.Sprintf(
					"%s_mask = sliding_window_mask(%s.shape[-2], %s, %s.device, %s.dtype)",
					outName("y"), q, pyNum(w), q, q))
				out.forward = append(out.forward, fmt.Sprintf(
					"%s = F.scaled_dot_product_attention(%s, %s, %s, attn_mask=%s_mask%s%s)",
					outName("y"), q, k, v, outName("y"), gqa, scale))
			default:
				out.forward = append(out.forward, fmt.Sprintf(
					"%s = F.scaled_dot_product_attention(%s, %s, %s, is_causal=%s%s%s)",
					outName("y"), q, k, v, pyBool(p["causal"]), gqa, scale))
			}
			set("y", outName("y"))

		default:
			switch {
			case catalog.IsContainer(def) && node.Graph != nil:
				inner := emitClassForGraph(node.Graph, path, "Layer", c, nil)
				out.init = append(out.init, fmt.Sprintf("self.%s = nn.ModuleList([%s() for _ in range(%s)])",
					attr, inner.name, pyValue(p["count"])))
				seed := make([]string, len(inner.inputs))
				loopVars := make([]string, len(inner.inputs))
				for i, port := range inner.inputs {
					seed[i] = inputVar(node.ID, port)
					loopVars[i] = outName(port)
				}
				out.forward = append(out.forward,
					strings.Join(loopVars, ", ")+" = "+strings.Join(seed, ", "),
					"for layer in self."+attr+":",
					"    "+strings.Join(loopVars, ", ")+" = layer("+strings.Join(loopVars, ", ")+")")
				for _, port := range inner.outputs {
					set(port, outName(port))
				}

			case catalog.IsComposite(def) && def.Type == "moe_layer":
				inner := emitMoeClass(r, path, c)
				out.init = append(out.init, fmt.Sprintf("self.%s = %s()", attr, inner.name))
				out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
					outName("y"), attr, inputVar(node.ID, "x")))
				set("y", outName("y"))

			case catalog.IsComposite(def):
				exp, ok := catalog.Expand(def, r.RawFull, r)
				if !ok {
					c.warn("%s: expansion failed: %s has no expansion.", path, node.Type)
					break
				}
				inner := emitClassForGraph(&ir.Graph{Nodes: exp.Nodes, Edges: exp.Edges}, path, def.Type, c, r)
				args := make([]string, len(inner.inputs))
				for i, port := range inner.inputs {
					args[i] = inputVar(node.ID, port)
				}
				rets := make([]string, len(inner.outputs))
				for i, port := range inner.outputs {
					rets[i] = outName(port)
				}
				out.init = append(out.init, fmt.Sprintf("self.%s = %s()", attr, inner.name))
				out.forward = append(out.forward, fmt.Sprintf("%s = self.%s(%s)",
					strings.Join(rets, ", "), attr, strings.Join(args, ", ")))
				for _, port := range inner.outputs {
					set(port, outName(port))
				}

			default:
				c.warn("%s: no code generator for block type %q.", path, node.Type)
				ports := catalog.Ports{}
				if !catalog.IsContainer(def) {
					ports = catalog.PortsOf(def, r)
				}
				for port := range ports.Out {
					set(port, "None")
				}
			}
		}
	}

	return out
}

// emitClassForGraph generates, or reuses, a class for a graph and returns its
// name and signature.
func emitClassForGraph(graph *ir.Graph, path, baseName string, c *ctx, resolved *catalog.Resolved) classInfo {
	var bIn, bOut *ir.NodeDef
	for i := range graph.Nodes {
		switch graph.Nodes[i].Type {
		case "boundary_in":
			if bIn == nil {
				bIn = &graph.Nodes[i]
			}
		case "boundary_out":
			if bOut == nil {
				bOut = &graph.Nodes[i]
			}
		}
	}
	inputs := portNames(bIn, "x")
	outputs := portNames(bOut, "y")

	// Two blocks of the same type with the same numbers share a class.
	var params any
	if resolved != nil {
		params = resolved.P
	}
	key := dedupKey(map[string]any{"baseName": baseName, "params": params, "graph": graph})
	if existing, ok := c.byKey[key]; ok {
		return classInfo{name: existing, inputs: inputs, outputs: outputs}
	}

	name := c.className(baseName)
	c.byKey[key] = name

	argMap := make(map[string]string, len(inputs))
	for _, port := range inputs {
		argMap[port] = pyName(port)
	}
	body := emitGraph(graph, path, argMap, c)

	returned := make([]string, len(outputs))
	args := make([]string, len(inputs))
	for i, port := range outputs {
		if v, ok := body.outputs[port]; ok {
			returned[i] = v
		} else {
			returned[i] = "None"
		}
	}
	for i, port := range inputs {
		args[i] = pyName(port)
	}

	lines := []string{fmt.Sprintf("class %s(nn.Module):", name)}
	if resolved != nil {
		if summary := paramSummary(resolved); summary != "" {
			lines = append(lines, fmt.Sprintf(`    """%s: %s"""`, baseName, summary))
		}
	}
	lines = append(lines, "", "    def __init__(self):", "        super().__init__()")
	for _, l := range body.init {
		lines = append(lines, "        "+l)
	}
	if len(body.init) == 0 {
		lines = append(lines, "        pass")
	}
	lines = append(lines, "", fmt.Sprintf("    def forward(self, %s):", strings.Join(args, ", ")))
	for _, l := range body.forward {
		lines = append(lines, "        "+l)
	}
	lines = append(lines, "        return "+strings.Join(returned, ", "), "")

	c.classes = append(c.classes, emitted{name: name, code: strings.Join(lines, "\n")})
	return classInfo{name: name, inputs: inputs, outputs: outputs}
}

// emitMoeClass is the mixture-of-experts dispatch.
//
// This is the one block whose runtime behaviour is not a dataflow graph of our
// primitives: routing sends different tokens to different modules. The expert
// body still comes from the generic emitter, so only the dispatch is bespoke.
func emitMoeClass(r *catalog.Resolved, path string, c *ctx) classInfo {
	raw := r.RawFull
	d := catalog.Ex(raw["d_model"], "0")
	fe := catalog.Ex(raw["expert_hidden"], "0")
	shared := r.Num("shared_experts")

	mlpDef := catalog.Builtin["gated_mlp"]
	expertParams := map[string]any{
		"d_model": d, "hidden": fe, "act": r.P["act"], "bias": r.P["bias"] == true,
	}
	expertResolved := catalog.ResolveNodeParams(mlpDef, expertParams, c.symbols)
	expertGraph, _ := catalog.Expand(mlpDef, expertResolved.RawFull, expertResolved)
	expert := emitClassForGraph(
		&ir.Graph{Nodes: expertGraph.Nodes, Edges: expertGraph.Edges},
		path+"/expert", "Expert", c, expertResolved)

	sharedClass := ""
	if shared > 0 {
		sharedParams := map[string]any{
			"d_model": d, "hidden": fmt.Sprintf("%s*%s", pyNum(shared), fe),
			"act": r.P["act"], "bias": r.P["bias"] == true,
		}
		sharedResolved := catalog.ResolveNodeParams(mlpDef, sharedParams, c.symbols)
		sharedGraph, _ := catalog.Expand(mlpDef, sharedResolved.RawFull, sharedResolved)
		sharedClass = emitClassForGraph(
			&ir.Graph{Nodes: sharedGraph.Nodes, Edges: sharedGraph.Edges},
			path+"/shared", "SharedExpert", c, sharedResolved).name
	}

	var sharedKey any
	if sharedClass != "" {
		sharedKey = sharedClass
	}
	key := dedupKey(map[string]any{"moe": r.P, "expert": expert.name, "sharedClass": sharedKey})
	if existing, ok := c.byKey[key]; ok {
		return classInfo{name: existing, inputs: []string{"x"}, outputs: []string{"y"}}
	}

	name := c.className("MoeLayer")
	c.byKey[key] = name

	experts, topK := r.Num("experts"), r.Num("top_k")
	doc := fmt.Sprintf("%s experts, top %s per token", pyNum(experts), pyNum(topK))
	if shared > 0 {
		doc += fmt.Sprintf(", plus %s shared expert(s)", pyNum(shared))
	}
	doc += "."
	if c.moeDispatch == "dense" {
		doc += fmt.Sprintf(" Dense dispatch: traceable, but %sx the work.",
			pyNum(jsRound(experts/topK)))
	}

	lines := []string{
		fmt.Sprintf("class %s(nn.Module):", name),
		fmt.Sprintf(`    """%s"""`, doc),
		"",
		"    def __init__(self):",
		"        super().__init__()",
		fmt.Sprintf("        self.top_k = %s", pyNum(topK)),
		fmt.Sprintf("        self.router = nn.Linear(%s, %s, bias=%s)",
			pyValue(r.P["d_model"]), pyNum(experts), pyBool(r.P["router_bias"])),
		fmt.Sprintf("        self.experts = nn.ModuleList([%s() for _ in range(%s)])",
			expert.name, pyNum(experts)),
	}
	if sharedClass != "" {
		lines = append(lines, fmt.Sprintf("        self.shared = %s()", sharedClass))
	}
	lines = append(lines,
		"",
		"    def forward(self, x):",
		"        shape = x.shape",
		fmt.Sprintf("        flat = x.reshape(-1, %s)", pyValue(r.P["d_model"])),
		"        scores = F.softmax(self.router(flat).float(), dim=-1)",
		"        weight, index = torch.topk(scores, self.top_k, dim=-1)")
	if r.P["normalize"] != false {
		lines = append(lines, "        weight = weight / weight.sum(dim=-1, keepdim=True)")
	}
	lines = append(lines, "        weight = weight.to(x.dtype)")
	if c.moeDispatch == "dense" {
		// Every expert sees every token, weighted by whether it was chosen. Same
		// result, no data-dependent shapes, so torch.export can trace it.
		lines = append(lines,
			fmt.Sprintf("        gate = torch.zeros(flat.shape[0], %s, device=x.device, dtype=x.dtype)",
				pyNum(experts)),
			"        gate = gate.scatter(1, index, weight)",
			"        out = torch.zeros_like(flat)",
			"        for e, expert in enumerate(self.experts):",
			"            out = out + expert(flat) * gate[:, e : e + 1]",
			"        y = out.reshape(shape)")
	} else {
		lines = append(lines,
			"        out = torch.zeros_like(flat)",
			"        for e, expert in enumerate(self.experts):",
			"            rows, slot = (index == e).nonzero(as_tuple=True)",
			"            if rows.numel() == 0:",
			"                continue",
			"            out.index_add_(0, rows, expert(flat[rows]) * weight[rows, slot, None])",
			"        y = out.reshape(shape)")
	}
	if sharedClass != "" {
		lines = append(lines, "        y = y + self.shared(x)")
	}
	lines = append(lines, "        return y", "")

	c.classes = append(c.classes, emitted{name: name, code: strings.Join(lines, "\n")})
	return classInfo{name: name, inputs: []string{"x"}, outputs: []string{"y"}}
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

// totalLayers is the number of stacked layers, used to scale the residual
// initialization by depth.
func totalLayers(flat *analysis.FlatResult) float64 {
	n := 0.0
	for _, rep := range flat.Repeats {
		if rep.Type == "repeat" {
			n += rep.Count
		}
	}
	if n == 0 {
		return 1
	}
	return n
}

// GenerateTorch emits a PyTorch module for a design.
func GenerateTorch(doc *ir.Doc, options Options) *Generated {
	symbols := ir.ResolveSymbols(doc)
	dispatch := options.MoeDispatch
	if dispatch == "" {
		dispatch = "sparse"
	}
	c := &ctx{
		doc:         doc,
		cat:         catalog.Of(doc),
		symbols:     symbols,
		warnings:    append([]string{}, symbols.Errors...),
		byKey:       map[string]string{},
		usedNames:   map[string]bool{},
		moeDispatch: dispatch,
	}

	modelName := options.ClassName
	if modelName == "" {
		base := doc.Meta.Name
		if base == "" {
			base = "Model"
		}
		modelName = c.className(base)
	}

	body := emitGraph(&doc.Graph, "", map[string]string{}, c)

	// Weight tying, which the graph expresses as a flag rather than an edge.
	//
	// Every tied head, not the first one: a design with multi-token prediction
	// has one output head per prediction depth and they all share the
	// embedding. Counting them as shared and then emitting them untied is a
	// model that does not have the parameters the analysis said it does, which
	// is exactly what `verify` is for and exactly what it caught.
	var tie []string
	var embed *ir.NodeDef
	for i := range doc.Graph.Nodes {
		if doc.Graph.Nodes[i].Type == "embedding" && embed == nil {
			embed = &doc.Graph.Nodes[i]
		}
	}
	if embed != nil {
		for i := range doc.Graph.Nodes {
			head := &doc.Graph.Nodes[i]
			if head.Type != "lm_head" {
				continue
			}
			r := catalog.ResolveNodeParams(c.cat[head.Type], head.Params, symbols)
			if r.P["tied"] == true {
				tie = append(tie, fmt.Sprintf("self.%s.weight = self.%s.weight",
					pyName(head.ID), pyName(embed.ID)))
			}
		}
	}

	flatResult := analysis.Flatten(doc, symbols)
	params := analysis.CountParams(flatResult)

	header := []string{
		fmt.Sprintf(`"""%s — generated by TensorCAD.`, doc.Meta.Name),
		"",
		"Do not edit by hand: change the design and regenerate.",
		"",
		"Symbols:",
	}
	for _, n := range symbols.Order {
		if symbols.Runtime[n] {
			continue
		}
		header = append(header, fmt.Sprintf("    %s = %s", n, analysis.JSNumber(symbols.Values[n])))
	}
	header = append(header,
		"",
		fmt.Sprintf("Parameters: %s", localeInt(params.Total)),
		`"""`,
		"",
		"import torch",
		"import torch.nn as nn",
		"import torch.nn.functional as F",
		"",
		"")

	var helpers []string
	if c.needsRope {
		helpers = append(helpers, helperRope, "")
	}
	if c.needsWindow {
		helpers = append(helpers, helperWindow, "")
	}
	if c.needsCausal {
		helpers = append(helpers, helperCausal, "")
	}
	if c.needsSoftcap {
		helpers = append(helpers, helperSoftcap, "")
	}
	if c.needsShift {
		helpers = append(helpers, helperShift, "")
	}
	if c.needsSsd {
		helpers = append(helpers, helperSsd, "")
	}

	modelLines := []string{
		fmt.Sprintf("class %s(nn.Module):", modelName),
		"",
		"    def __init__(self):",
		"        super().__init__()",
	}
	for _, l := range body.init {
		modelLines = append(modelLines, "        "+l)
	}
	for _, l := range tie {
		modelLines = append(modelLines, "        "+l)
	}

	// Weight initialization. Residual projections are scaled down by the depth
	// so the residual stream does not grow as layers are added, which is what
	// GPT-2 does and what every model since has kept.
	initStd := 0.02
	if options.InitStd != nil {
		initStd = *options.InitStd
	}
	var initLines []string
	// The blank line belongs to the forward that follows, whether or not there
	// is an init_weights between them.
	modelLines = append(modelLines, "")
	if initStd > 0 {
		residualStd := initStd / math.Sqrt(2*math.Max(1, totalLayers(flatResult)))
		std := analysis.JSNumber(initStd)
		modelLines = append(modelLines,
			"    @torch.no_grad()",
			"    def init_weights(self):",
			fmt.Sprintf(`        """Normal(0, %s), with residual projections scaled by depth.`, std),
			"",
			"        PyTorch's defaults leave nn.Embedding at a unit normal, which starts",
			"        a language model near a cross-entropy of a few hundred rather than",
			"        ln(vocab). Call this after moving the model to its device.",
			`        """`,
			"        for module in self.modules():",
			"            if isinstance(module, (nn.Linear, nn.Embedding)):",
			fmt.Sprintf("                nn.init.normal_(module.weight, mean=0.0, std=%s)", std),
			`                bias = getattr(module, "bias", None)`,
			"                if bias is not None:",
			"                    nn.init.zeros_(bias)",
			"        for name, param in self.named_parameters():",
			"            if name.endswith(RESIDUAL_PROJECTIONS):",
			fmt.Sprintf("                nn.init.normal_(param, mean=0.0, std=%s)", jsToPrecision(residualStd, 8)),
			"        return self",
			"")
		initLines = append(initLines,
			"",
			"# Projections that write back into the residual stream. Their initial scale",
			"# is divided by sqrt(2 * layers) so depth does not inflate the stream.",
			`RESIDUAL_PROJECTIONS = ("o_proj.weight", "down.weight", "out_proj.weight")`,
			"")
	}

	modelLines = append(modelLines, "    def forward(self, ids):")
	for _, l := range body.forward {
		modelLines = append(modelLines, "        "+l)
	}
	// One output returns a tensor; several return a tuple, in the order the
	// graph lists them.
	returned := "None"
	switch len(body.outputOrder) {
	case 0:
	case 1:
		returned = body.outputs[body.outputOrder[0]]
	default:
		parts := make([]string, len(body.outputOrder))
		for i, id := range body.outputOrder {
			parts[i] = body.outputs[id]
		}
		returned = strings.Join(parts, ", ")
	}
	modelLines = append(modelLines, "        return "+returned, "")

	var smoke []string
	if !options.NoSmokeTest {
		smoke = append(smoke,
			"",
			`if __name__ == "__main__":`,
			fmt.Sprintf("    expected = %s", analysis.JSNumber(params.Total)),
			`    with torch.device("meta"):`,
			fmt.Sprintf("        model = %s()", modelName),
			"    actual = sum(p.numel() for p in model.parameters())",
			`    print(f"parameters: {actual:,} (design says {expected:,})")`,
			`    assert actual == expected, f"parameter count differs: {actual} vs {expected}"`,
			"")
	}

	classCode := make([]string, len(c.classes))
	for i, cl := range c.classes {
		classCode[i] = cl.code
	}

	sections := []string{
		strings.Join(header, "\n"),
		strings.Join(initLines, "\n"),
		strings.Join(helpers, "\n"),
		strings.Join(classCode, "\n"),
		strings.Join(modelLines, "\n"),
		strings.Join(smoke, "\n"),
	}
	var kept []string
	for _, s := range sections {
		if strings.TrimSpace(s) != "" {
			kept = append(kept, s)
		}
	}
	contents := strings.Join(kept, "\n")
	if !strings.HasSuffix(contents, "\n") {
		contents += "\n"
	}

	design, _ := json.MarshalIndent(doc, "", "  ")

	return &Generated{
		Files: []File{
			{Path: "model.py", Contents: contents},
			{Path: "design.tensorcad.json", Contents: string(design) + "\n"},
		},
		Warnings: c.warnings,
	}
}

func asNumber(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	}
	return 0, false
}

// truthy is JavaScript's notion of it, which the emitter leans on where the
// TypeScript writes `x ? x : -1`.
func truthy(v any) bool {
	switch n := v.(type) {
	case nil:
		return false
	case bool:
		return n
	case float64:
		return n != 0 && !math.IsNaN(n)
	case int:
		return n != 0
	case string:
		return n != ""
	}
	return true
}

// jsRound rounds half towards positive infinity, as Math.round does.
func jsRound(v float64) float64 { return math.Floor(v + 0.5) }

func evalSize(v any, ctx shapes.EvalCtx) (float64, bool) {
	if n, ok := asNumber(v); ok {
		return n, true
	}
	s, ok := v.(string)
	if !ok {
		return 0, false
	}
	sym, err := shapes.EvalExpr(s, ctx)
	if err != nil {
		return 0, false
	}
	n, ok := sym.ToNumber(ctx.Values)
	return n, ok
}
