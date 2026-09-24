// Package attnexpr is the language a design uses to say what attention does:
// which scores count, and what happens to each one before the softmax.
//
// It is FlexAttention's shape, not its syntax. FlexAttention takes two Python
// functions — mask_mod(b, h, q_idx, kv_idx) -> bool and score_mod(score, b, h,
// q_idx, kv_idx) -> score — and compiles them into one fused kernel. A design
// cannot carry Python, and the engine could not read it if it did: it has to
// evaluate a mask to count what it keeps, cost a score expression per score,
// check both, and print them as Python for the generated model. So they are
// written in a small language the engine owns, and printed into FlexAttention's
// form.
//
// A mask is a boolean expression over q (the query's position), kv (the key's),
// h (the head), b (the sequence in the batch) and heads (how many there are):
//
//	kv <= q and q - kv < 1024        causal, within a window
//	kv <= q or kv < 16               prefix-LM, the first 16 positions bidirectional
//
// A score expression is a number, over the same names and score itself:
//
//	score - 2 ** (-8 * (h + 1) / heads) * (q - kv)    ALiBi
//	50 * tanh(score / 50)                              Gemma 2's cap
//
// Any other name is a design symbol, replaced by its value when the design is
// resolved, so a prefix can be `kv < P` and follow P.
package attnexpr

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Kind is what an expression must evaluate to.
type Kind int

const (
	// Mask is a boolean: whether a score counts.
	Mask Kind = iota
	// Score is a number: what a score becomes.
	Score
)

// Vars are the names an expression reads at run time, as opposed to design
// symbols, which are fixed when the design is resolved.
var Vars = map[string]bool{"q": true, "kv": true, "h": true, "b": true, "heads": true, "score": true}

// Node is one node of a parsed expression.
type Node interface {
	isNode()
}

// Num is a constant.
type Num struct{ V float64 }

// Bool is a comparison that folded to a constant.
type Bool struct{ V bool }

// Var is a name read at run time: q, kv, h, b, heads or score.
type Var struct{ Name string }

// Unary is -x or not x.
type Unary struct {
	Op string
	X  Node
}

// Binary is x op y.
type Binary struct {
	Op   string
	X, Y Node
}

// Call is a function applied to arguments.
type Call struct {
	Fn   string
	Args []Node
}

func (Num) isNode()    {}
func (Bool) isNode()   {}
func (Var) isNode()    {}
func (Unary) isNode()  {}
func (Binary) isNode() {}
func (Call) isNode()   {}

// funcs are the functions an expression may call, with how many arguments each
// takes and roughly what one costs in FLOPs, for the elementwise count. A tanh
// is six, the figure the logit softcap has always been counted at, so a cap
// written as an expression costs what the parameter does.
var funcs = map[string]struct {
	arity int
	cost  float64
}{
	"tanh":  {1, 6},
	"exp":   {1, 4},
	"log":   {1, 4},
	"sqrt":  {1, 2},
	"abs":   {1, 1},
	"floor": {1, 1},
	"min":   {2, 1},
	"max":   {2, 1},
	"where": {3, 1},
}

// --- lexing ------------------------------------------------------------------

type token struct {
	kind string // num, name, op, eof
	text string
	pos  int
}

func lex(src string) ([]token, error) {
	var out []token
	i := 0
	for i < len(src) {
		c := src[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n' || c == '\r':
			i++
		case c >= '0' && c <= '9' || c == '.':
			j := i
			for j < len(src) && (src[j] >= '0' && src[j] <= '9' || src[j] == '.') {
				j++
			}
			// An exponent: 1e-3.
			if j < len(src) && (src[j] == 'e' || src[j] == 'E') {
				k := j + 1
				if k < len(src) && (src[k] == '-' || src[k] == '+') {
					k++
				}
				if k < len(src) && src[k] >= '0' && src[k] <= '9' {
					j = k
					for j < len(src) && src[j] >= '0' && src[j] <= '9' {
						j++
					}
				}
			}
			out = append(out, token{"num", src[i:j], i})
			i = j
		case c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z':
			j := i
			for j < len(src) && (src[j] == '_' || src[j] >= 'a' && src[j] <= 'z' || src[j] >= 'A' && src[j] <= 'Z' || src[j] >= '0' && src[j] <= '9') {
				j++
			}
			out = append(out, token{"name", src[i:j], i})
			i = j
		default:
			two := ""
			if i+1 < len(src) {
				two = src[i : i+2]
			}
			switch two {
			case "**", "<=", ">=", "==", "!=", "&&", "||":
				out = append(out, token{"op", two, i})
				i += 2
				continue
			}
			switch c {
			case '+', '-', '*', '/', '%', '<', '>', '(', ')', ',', '!':
				out = append(out, token{"op", string(c), i})
				i++
			default:
				return nil, fmt.Errorf("unexpected %q at %d", string(c), i+1)
			}
		}
	}
	return append(out, token{"eof", "", len(src)}), nil
}

// --- parsing -----------------------------------------------------------------

type parser struct {
	toks []token
	i    int
}

func (p *parser) peek() token { return p.toks[p.i] }
func (p *parser) next() token {
	t := p.toks[p.i]
	if p.i < len(p.toks)-1 {
		p.i++
	}
	return t
}

func (p *parser) isOp(ops ...string) (string, bool) {
	t := p.peek()
	if t.kind == "name" {
		// The word forms, which read better in a mask than the symbols do.
		for _, op := range ops {
			if t.text == op {
				return op, true
			}
		}
		return "", false
	}
	if t.kind != "op" {
		return "", false
	}
	for _, op := range ops {
		if t.text == op {
			return op, true
		}
	}
	return "", false
}

// Parse reads an expression. It does not check names or kinds; Check does.
func Parse(src string) (Node, error) {
	toks, err := lex(src)
	if err != nil {
		return nil, err
	}
	p := &parser{toks: toks}
	n, err := p.or()
	if err != nil {
		return nil, err
	}
	if t := p.peek(); t.kind != "eof" {
		return nil, fmt.Errorf("unexpected %q at %d", t.text, t.pos+1)
	}
	return n, nil
}

func (p *parser) or() (Node, error) {
	x, err := p.and()
	if err != nil {
		return nil, err
	}
	for {
		if _, ok := p.isOp("or", "||"); !ok {
			return x, nil
		}
		p.next()
		y, err := p.and()
		if err != nil {
			return nil, err
		}
		x = Binary{"or", x, y}
	}
}

func (p *parser) and() (Node, error) {
	x, err := p.not()
	if err != nil {
		return nil, err
	}
	for {
		if _, ok := p.isOp("and", "&&"); !ok {
			return x, nil
		}
		p.next()
		y, err := p.not()
		if err != nil {
			return nil, err
		}
		x = Binary{"and", x, y}
	}
}

func (p *parser) not() (Node, error) {
	if _, ok := p.isOp("not", "!"); ok {
		p.next()
		x, err := p.not()
		if err != nil {
			return nil, err
		}
		return Unary{"not", x}, nil
	}
	return p.compare()
}

func (p *parser) compare() (Node, error) {
	x, err := p.sum()
	if err != nil {
		return nil, err
	}
	if op, ok := p.isOp("<", "<=", ">", ">=", "==", "!="); ok {
		p.next()
		y, err := p.sum()
		if err != nil {
			return nil, err
		}
		if _, again := p.isOp("<", "<=", ">", ">=", "==", "!="); again {
			return nil, fmt.Errorf("comparisons do not chain: write a < b and b < c")
		}
		return Binary{op, x, y}, nil
	}
	return x, nil
}

func (p *parser) sum() (Node, error) {
	x, err := p.product()
	if err != nil {
		return nil, err
	}
	for {
		op, ok := p.isOp("+", "-")
		if !ok {
			return x, nil
		}
		p.next()
		y, err := p.product()
		if err != nil {
			return nil, err
		}
		x = Binary{op, x, y}
	}
}

func (p *parser) product() (Node, error) {
	x, err := p.unary()
	if err != nil {
		return nil, err
	}
	for {
		op, ok := p.isOp("*", "/", "%")
		if !ok {
			return x, nil
		}
		p.next()
		y, err := p.unary()
		if err != nil {
			return nil, err
		}
		x = Binary{op, x, y}
	}
}

func (p *parser) unary() (Node, error) {
	if _, ok := p.isOp("-"); ok {
		p.next()
		x, err := p.unary()
		if err != nil {
			return nil, err
		}
		return Unary{"-", x}, nil
	}
	if _, ok := p.isOp("+"); ok {
		p.next()
		return p.unary()
	}
	return p.power()
}

func (p *parser) power() (Node, error) {
	x, err := p.atom()
	if err != nil {
		return nil, err
	}
	if _, ok := p.isOp("**"); ok {
		p.next()
		// Right-associative, and binding tighter than a unary minus on its
		// left: -2 ** 2 is -(2 ** 2), as in Python.
		y, err := p.unary()
		if err != nil {
			return nil, err
		}
		return Binary{"**", x, y}, nil
	}
	return x, nil
}

func (p *parser) atom() (Node, error) {
	t := p.next()
	switch t.kind {
	case "num":
		v, err := strconv.ParseFloat(t.text, 64)
		if err != nil {
			return nil, fmt.Errorf("%q is not a number", t.text)
		}
		return Num{v}, nil
	case "name":
		switch t.text {
		case "and", "or", "not":
			return nil, fmt.Errorf("%q needs something on each side, at %d", t.text, t.pos+1)
		case "true", "false":
			return Bool{t.text == "true"}, nil
		}
		if _, ok := p.isOp("("); ok {
			p.next()
			var args []Node
			if _, closed := p.isOp(")"); !closed {
				for {
					a, err := p.or()
					if err != nil {
						return nil, err
					}
					args = append(args, a)
					if _, more := p.isOp(","); more {
						p.next()
						continue
					}
					break
				}
			}
			if _, ok := p.isOp(")"); !ok {
				return nil, fmt.Errorf("%s( is not closed", t.text)
			}
			p.next()
			return Call{t.text, args}, nil
		}
		return Var{t.text}, nil
	case "op":
		if t.text == "(" {
			x, err := p.or()
			if err != nil {
				return nil, err
			}
			if _, ok := p.isOp(")"); !ok {
				return nil, fmt.Errorf("( at %d is not closed", t.pos+1)
			}
			p.next()
			return x, nil
		}
	case "eof":
		return nil, fmt.Errorf("the expression ends too soon")
	}
	return nil, fmt.Errorf("unexpected %q at %d", t.text, t.pos+1)
}

// --- binding, checking, folding ---------------------------------------------

// Bind replaces every name that is not a run-time variable with its value from
// symbols, and refuses a name it has no value for.
func Bind(n Node, symbols map[string]float64) (Node, error) {
	switch x := n.(type) {
	case Num, Bool:
		return x, nil
	case Var:
		if Vars[x.Name] {
			return x, nil
		}
		if v, ok := symbols[x.Name]; ok && !math.IsNaN(v) {
			return Num{v}, nil
		}
		return nil, fmt.Errorf("%q is not q, kv, h, b, heads, score or a symbol of the design", x.Name)
	case Unary:
		inner, err := Bind(x.X, symbols)
		if err != nil {
			return nil, err
		}
		return Unary{x.Op, inner}, nil
	case Binary:
		l, err := Bind(x.X, symbols)
		if err != nil {
			return nil, err
		}
		r, err := Bind(x.Y, symbols)
		if err != nil {
			return nil, err
		}
		return Binary{x.Op, l, r}, nil
	case Call:
		args := make([]Node, len(x.Args))
		for i, a := range x.Args {
			b, err := Bind(a, symbols)
			if err != nil {
				return nil, err
			}
			args[i] = b
		}
		return Call{x.Fn, args}, nil
	}
	return nil, fmt.Errorf("unknown node %T", n)
}

// Check says whether an expression is a well-formed mask or score.
func Check(n Node, kind Kind) error {
	isBool, err := typeOf(n)
	if err != nil {
		return err
	}
	switch kind {
	case Mask:
		if !isBool {
			return fmt.Errorf("a mask is true or false for each score: compare something, as in kv <= q")
		}
		if Uses(n, "score") {
			return fmt.Errorf("a mask cannot read the score: it decides which scores are computed at all")
		}
	case Score:
		if isBool {
			return fmt.Errorf("a score expression is a number, not a comparison; a comparison belongs in the mask")
		}
	}
	return nil
}

func typeOf(n Node) (isBool bool, err error) {
	switch x := n.(type) {
	case Num, Var:
		return false, nil
	case Bool:
		return true, nil
	case Unary:
		inner, err := typeOf(x.X)
		if err != nil {
			return false, err
		}
		if x.Op == "not" {
			if !inner {
				return false, fmt.Errorf("not needs a comparison, not a number")
			}
			return true, nil
		}
		if inner {
			return false, fmt.Errorf("a comparison cannot be negated with -; use not")
		}
		return false, nil
	case Binary:
		l, err := typeOf(x.X)
		if err != nil {
			return false, err
		}
		r, err := typeOf(x.Y)
		if err != nil {
			return false, err
		}
		switch x.Op {
		case "and", "or":
			if !l || !r {
				return false, fmt.Errorf("%s joins comparisons, not numbers", x.Op)
			}
			return true, nil
		case "<", "<=", ">", ">=", "==", "!=":
			if l || r {
				return false, fmt.Errorf("%s compares numbers, not comparisons", x.Op)
			}
			return true, nil
		default:
			if l || r {
				return false, fmt.Errorf("%s works on numbers, not comparisons", x.Op)
			}
			return false, nil
		}
	case Call:
		f, ok := funcs[x.Fn]
		if !ok {
			return false, fmt.Errorf("%s is not a function this knows: %s", x.Fn, strings.Join(funcNames(), ", "))
		}
		if len(x.Args) != f.arity {
			return false, fmt.Errorf("%s takes %d argument(s), not %d", x.Fn, f.arity, len(x.Args))
		}
		for i, a := range x.Args {
			isB, err := typeOf(a)
			if err != nil {
				return false, err
			}
			wantBool := x.Fn == "where" && i == 0
			if isB != wantBool {
				if wantBool {
					return false, fmt.Errorf("where's first argument is a comparison")
				}
				return false, fmt.Errorf("%s takes numbers", x.Fn)
			}
		}
		return false, nil
	}
	return false, fmt.Errorf("unknown node %T", n)
}

func funcNames() []string {
	names := make([]string, 0, len(funcs))
	for _, n := range []string{"tanh", "exp", "log", "sqrt", "abs", "floor", "min", "max", "where"} {
		if _, ok := funcs[n]; ok {
			names = append(names, n)
		}
	}
	return names
}

// Uses says whether an expression reads a run-time variable.
func Uses(n Node, name string) bool {
	switch x := n.(type) {
	case Var:
		return x.Name == name
	case Unary:
		return Uses(x.X, name)
	case Binary:
		return Uses(x.X, name) || Uses(x.Y, name)
	case Call:
		for _, a := range x.Args {
			if Uses(a, name) {
				return true
			}
		}
	}
	return false
}

// Fold evaluates whatever does not depend on a run-time variable, so that what
// is printed and what is costed is the expression's real work: 2 ** 3 is 8,
// and costs nothing per score.
func Fold(n Node) Node {
	switch x := n.(type) {
	case Unary:
		inner := Fold(x.X)
		if v, ok := inner.(Num); ok && x.Op == "-" {
			return Num{-v.V}
		}
		if v, ok := inner.(Bool); ok && x.Op == "not" {
			return Bool{!v.V}
		}
		return Unary{x.Op, inner}
	case Binary:
		l, r := Fold(x.X), Fold(x.Y)
		// A constant side of an and or an or decides it or drops out, so what
		// is left never mixes a Python bool into a tensor expression, where
		// ~True is -2.
		lb, lbok := l.(Bool)
		rb, rbok := r.(Bool)
		switch x.Op {
		case "and":
			switch {
			case lbok && !lb.V, rbok && !rb.V:
				return Bool{false}
			case lbok:
				return r
			case rbok:
				return l
			}
		case "or":
			switch {
			case lbok && lb.V, rbok && rb.V:
				return Bool{true}
			case lbok:
				return r
			case rbok:
				return l
			}
		}
		lv, lok := l.(Num)
		rv, rok := r.(Num)
		if lok && rok {
			if v, ok := arith(x.Op, lv.V, rv.V); ok {
				return Num{v}
			}
			if v, ok := compare(x.Op, lv.V, rv.V); ok {
				return Bool{v}
			}
		}
		return Binary{x.Op, l, r}
	case Call:
		args := make([]Node, len(x.Args))
		all := true
		vals := make([]float64, len(x.Args))
		for i, a := range x.Args {
			args[i] = Fold(a)
			if v, ok := args[i].(Num); ok {
				vals[i] = v.V
			} else {
				all = false
			}
		}
		if all && x.Fn != "where" {
			return Num{call(x.Fn, vals)}
		}
		if cond, ok := args[0].(Bool); ok && x.Fn == "where" {
			if cond.V {
				return args[1]
			}
			return args[2]
		}
		return Call{x.Fn, args}
	}
	return n
}

func arith(op string, a, b float64) (float64, bool) {
	switch op {
	case "+":
		return a + b, true
	case "-":
		return a - b, true
	case "*":
		return a * b, true
	case "/":
		return a / b, true
	case "%":
		return floorMod(a, b), true
	case "**":
		return math.Pow(a, b), true
	}
	return 0, false
}

// floorMod is Python's %, which takes the sign of the divisor: (q - kv) % 64
// is never negative there, where Go's math.Mod would follow q - kv below zero.
// The generated code is Python, so this is the one that has to agree with it.
func floorMod(a, b float64) float64 {
	m := math.Mod(a, b)
	if m != 0 && (m < 0) != (b < 0) {
		m += b
	}
	return m
}

func compare(op string, a, b float64) (bool, bool) {
	switch op {
	case "<":
		return a < b, true
	case "<=":
		return a <= b, true
	case ">":
		return a > b, true
	case ">=":
		return a >= b, true
	case "==":
		return a == b, true
	case "!=":
		return a != b, true
	}
	return false, false
}

func call(fn string, v []float64) float64 {
	switch fn {
	case "tanh":
		return math.Tanh(v[0])
	case "exp":
		return math.Exp(v[0])
	case "log":
		return math.Log(v[0])
	case "sqrt":
		return math.Sqrt(v[0])
	case "abs":
		return math.Abs(v[0])
	case "floor":
		return math.Floor(v[0])
	case "min":
		return math.Min(v[0], v[1])
	case "max":
		return math.Max(v[0], v[1])
	}
	return math.NaN()
}

// Compile parses, binds the design's symbols, checks and folds an expression.
func Compile(src string, kind Kind, symbols map[string]float64) (Node, error) {
	n, err := Parse(src)
	if err != nil {
		return nil, err
	}
	n, err = Bind(n, symbols)
	if err != nil {
		return nil, err
	}
	if err := Check(n, kind); err != nil {
		return nil, err
	}
	n = Fold(n)
	switch kind {
	case Mask:
		if b, ok := n.(Bool); ok {
			if b.V {
				return nil, fmt.Errorf("the mask is true for every score, so it masks nothing: it has to read q, kv, h or b")
			}
			return nil, fmt.Errorf("the mask is false for every score, so nothing is attended to: it has to read q, kv, h or b")
		}
	case Score:
		if !Uses(n, "score") {
			return nil, fmt.Errorf("a score expression that does not read score throws the attention scores away")
		}
	}
	return n, nil
}

// --- evaluating --------------------------------------------------------------

// Env is one point an expression is evaluated at.
type Env struct {
	Q, KV, H, B, Heads, Score float64
}

// Eval evaluates a checked expression. A mask gives 1 for true and 0 for false.
func Eval(n Node, e Env) float64 {
	switch x := n.(type) {
	case Num:
		return x.V
	case Bool:
		return b2f(x.V)
	case Var:
		switch x.Name {
		case "q":
			return e.Q
		case "kv":
			return e.KV
		case "h":
			return e.H
		case "b":
			return e.B
		case "heads":
			return e.Heads
		case "score":
			return e.Score
		}
	case Unary:
		v := Eval(x.X, e)
		if x.Op == "not" {
			return b2f(v == 0)
		}
		return -v
	case Binary:
		switch x.Op {
		case "and":
			return b2f(Eval(x.X, e) != 0 && Eval(x.Y, e) != 0)
		case "or":
			return b2f(Eval(x.X, e) != 0 || Eval(x.Y, e) != 0)
		}
		a, b := Eval(x.X, e), Eval(x.Y, e)
		if v, ok := compare(x.Op, a, b); ok {
			return b2f(v)
		}
		v, _ := arith(x.Op, a, b)
		return v
	case Call:
		if x.Fn == "where" {
			if Eval(x.Args[0], e) != 0 {
				return Eval(x.Args[1], e)
			}
			return Eval(x.Args[2], e)
		}
		vals := make([]float64, len(x.Args))
		for i, a := range x.Args {
			vals[i] = Eval(a, e)
		}
		return call(x.Fn, vals)
	}
	return math.NaN()
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// Cost is roughly what one evaluation costs, in FLOPs: an operation each, a
// transcendental function a few.
func Cost(n Node) float64 {
	switch x := n.(type) {
	case Unary:
		return 1 + Cost(x.X)
	case Binary:
		return 1 + Cost(x.X) + Cost(x.Y)
	case Call:
		c := funcs[x.Fn].cost
		for _, a := range x.Args {
			c += Cost(a)
		}
		return c
	}
	return 0
}

// --- printing ----------------------------------------------------------------

// String prints an expression in this language, with parentheses only where
// they are needed, which is what a design shows after its symbols are bound.
func String(n Node) string { return str(n, 0) }

var prec = map[string]int{
	"or": 1, "and": 2, "not": 3,
	"<": 4, "<=": 4, ">": 4, ">=": 4, "==": 4, "!=": 4,
	"+": 5, "-": 5, "*": 6, "/": 6, "%": 6, "neg": 7, "**": 8,
}

func str(n Node, outer int) string {
	wrap := func(s string, p int) string {
		if p < outer {
			return "(" + s + ")"
		}
		return s
	}
	switch x := n.(type) {
	case Num:
		// A negative constant is a unary minus as far as a reader is
		// concerned: -2 ** h is -(2 ** h), so a folded -2 on the left of a
		// power needs its parentheses back.
		if x.V < 0 {
			return wrap(num(x.V), prec["neg"])
		}
		return num(x.V)
	case Bool:
		if x.V {
			return "true"
		}
		return "false"
	case Var:
		return x.Name
	case Unary:
		if x.Op == "not" {
			return wrap("not "+str(x.X, prec["not"]), prec["not"])
		}
		return wrap("-"+str(x.X, prec["neg"]), prec["neg"])
	case Binary:
		p := prec[x.Op]
		l, r := p, p+1
		if x.Op == "**" {
			l, r = p+1, p
		}
		return wrap(str(x.X, l)+" "+x.Op+" "+str(x.Y, r), p)
	case Call:
		args := make([]string, len(x.Args))
		for i, a := range x.Args {
			args[i] = str(a, 0)
		}
		return x.Fn + "(" + strings.Join(args, ", ") + ")"
	}
	return "?"
}

func num(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatFloat(v, 'f', -1, 64)
	}
	return strconv.FormatFloat(v, 'g', 12, 64)
}

// Python prints an expression as the body of a FlexAttention mask_mod or
// score_mod: over tensors named q_idx, kv_idx, h, b and score, and a heads
// constant. Fully parenthesised below the top, because Python's & and | bind
// tighter than its comparisons, and a precedence table that disagreed with
// Python's would be a mask that silently means something else.
//
// Every variable is a tensor in both of the places the function is called —
// FlexAttention hands it zero-dimensional ones, the eager form broadcast
// grids — and folding leaves no operation between two constants, so every
// operation has a tensor on one side. The one exception is an expression that
// folded to a constant outright, which is given a tensor's shape here, because
// ~True is -2 in Python rather than False.
func Python(n Node, heads float64) string {
	switch x := n.(type) {
	case Bool:
		if x.V {
			return "kv_idx >= 0"
		}
		return "kv_idx < 0"
	case Num:
		return "torch.full_like(score, " + num(x.V) + ")"
	}
	s := python(n, heads)
	if _, ok := n.(Binary); ok {
		return s[1 : len(s)-1]
	}
	return s
}

// With replaces a run-time variable by a value and folds again: heads is fixed
// once the expression is on a block.
func With(n Node, name string, v float64) Node {
	var sub func(Node) Node
	sub = func(n Node) Node {
		switch x := n.(type) {
		case Var:
			if x.Name == name {
				return Num{v}
			}
		case Unary:
			return Unary{x.Op, sub(x.X)}
		case Binary:
			return Binary{x.Op, sub(x.X), sub(x.Y)}
		case Call:
			args := make([]Node, len(x.Args))
			for i, a := range x.Args {
				args[i] = sub(a)
			}
			return Call{x.Fn, args}
		}
		return n
	}
	return Fold(sub(n))
}

// float is a constant written so Python computes with it in floating point: an
// integer tensor raised to a negative integer power is an error in PyTorch.
func pyFloat(n Node, heads float64) string {
	if x, ok := n.(Num); ok && x.V == math.Trunc(x.V) && math.Abs(x.V) < 1e15 {
		s := num(x.V) + ".0"
		if x.V < 0 {
			return "(" + s + ")"
		}
		return s
	}
	return python(n, heads)
}

func python(n Node, heads float64) string {
	switch x := n.(type) {
	case Num:
		if x.V < 0 {
			return "(" + num(x.V) + ")"
		}
		return num(x.V)
	case Bool:
		if x.V {
			return "True"
		}
		return "False"
	case Var:
		switch x.Name {
		case "q":
			return "q_idx"
		case "kv":
			return "kv_idx"
		case "heads":
			return num(heads)
		}
		return x.Name
	case Unary:
		if x.Op == "not" {
			return "~(" + python(x.X, heads) + ")"
		}
		return "-(" + python(x.X, heads) + ")"
	case Binary:
		op := x.Op
		switch op {
		case "and":
			op = "&"
		case "or":
			op = "|"
		}
		if op == "**" {
			return "(" + pyFloat(x.X, heads) + " ** " + pyFloat(x.Y, heads) + ")"
		}
		return "(" + python(x.X, heads) + " " + op + " " + python(x.Y, heads) + ")"
	case Call:
		// An argument is bracketed by the call already.
		args := make([]string, len(x.Args))
		for i, a := range x.Args {
			args[i] = python(a, heads)
			if _, ok := a.(Binary); ok {
				args[i] = args[i][1 : len(args[i])-1]
			}
		}
		switch x.Fn {
		case "min", "max":
			// A bound against a constant is a clamp; torch.minimum wants two
			// tensors on one device, and a constant is neither.
			bound := "max="
			if x.Fn == "max" {
				bound = "min="
			}
			if _, ok := x.Args[1].(Num); ok {
				return "torch.clamp(" + args[0] + ", " + bound + args[1] + ")"
			}
			if _, ok := x.Args[0].(Num); ok {
				return "torch.clamp(" + args[1] + ", " + bound + args[0] + ")"
			}
			return "torch." + x.Fn + "imum(" + args[0] + ", " + args[1] + ")"
		case "where":
			return "torch.where(" + args[0] + ", " + args[1] + ", " + args[2] + ")"
		}
		return "torch." + x.Fn + "(" + args[0] + ")"
	}
	return "None"
}

// And is x and y, dropping a side that is absent.
func And(x, y Node) Node {
	if x == nil {
		return y
	}
	if y == nil {
		return x
	}
	return Binary{"and", x, y}
}
