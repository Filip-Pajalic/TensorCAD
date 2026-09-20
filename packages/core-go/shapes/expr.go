// Expression parser for parameter values and shape-pattern atoms.
//
// Grammar:
//
//	expr    := term (('+' | '-') term)*
//	term    := unary (('*' | '/') unary)*
//	unary   := '-' unary | power
//	power   := primary ('^' unary)?
//	primary := NUMBER | IDENT | IDENT '(' args ')' | '(' expr ')'
//
// Symbol references evaluate to symbolic atoms so that a design keeps showing
// "B T D" rather than "B T 4096". Function calls are folded eagerly: their
// arguments must be numerically determined, which is always true for design
// parameters.
package shapes

import (
	"fmt"
	"math"
	"math/big"
	"strings"
	"sync"
)

type tokKind int

const (
	tokNum tokKind = iota
	tokIdent
	tokOp
	tokEOF
)

type tok struct {
	kind tokKind
	// text is the literal source for a number, so "1.3" can become 13/10
	// exactly rather than going through a float.
	text string
	pos  int
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }
func isIdentStart(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}
func isIdentPart(c byte) bool { return isIdentStart(c) || isDigit(c) }

func tokenize(src string) ([]tok, error) {
	var out []tok
	i := 0
	for i < len(src) {
		c := src[i]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			i++
			continue
		}
		if isDigit(c) || (c == '.' && i+1 < len(src) && isDigit(src[i+1])) {
			j := i
			for j < len(src) && (isDigit(src[j]) || src[j] == '.' || src[j] == '_') {
				j++
			}
			if j < len(src) && (src[j] == 'e' || src[j] == 'E') {
				k := j + 1
				if k < len(src) && (src[k] == '+' || src[k] == '-') {
					k++
				}
				if k < len(src) && isDigit(src[k]) {
					j = k
					for j < len(src) && isDigit(src[j]) {
						j++
					}
				}
			}
			text := strings.ReplaceAll(src[i:j], "_", "")
			out = append(out, tok{kind: tokNum, text: text, pos: i})
			i = j
			continue
		}
		if isIdentStart(c) {
			j := i
			for j < len(src) && isIdentPart(src[j]) {
				j++
			}
			out = append(out, tok{kind: tokIdent, text: src[i:j], pos: i})
			i = j
			continue
		}
		if strings.IndexByte("+-*/^(),", c) >= 0 {
			out = append(out, tok{kind: tokOp, text: string(c), pos: i})
			i++
			continue
		}
		return nil, fmt.Errorf("Unexpected character %q at %d in %q", string(c), i, src)
	}
	out = append(out, tok{kind: tokEOF, pos: len(src)})
	return out, nil
}

// AstKind distinguishes the node types of a parsed expression.
type AstKind int

const (
	AstNum AstKind = iota
	AstSym
	AstCall
	AstBin
	AstNeg
)

// Ast is a parsed expression.
type Ast struct {
	Kind AstKind
	// Num holds the exact literal value.
	Num *big.Rat
	// Name is the symbol or function name.
	Name string
	// Op is one of + - * / ^ for a binary node.
	Op   byte
	L, R *Ast
	Args []*Ast
}

type parser struct {
	toks []tok
	i    int
	src  string
}

func (p *parser) peek() tok { return p.toks[p.i] }

func (p *parser) eat(op string) bool {
	t := p.peek()
	if t.kind == tokOp && t.text == op {
		p.i++
		return true
	}
	return false
}

func (p *parser) expect(op string) error {
	if !p.eat(op) {
		return fmt.Errorf("Expected %q at %d in %q", op, p.peek().pos, p.src)
	}
	return nil
}

func (p *parser) parse() (*Ast, error) {
	e, err := p.expr()
	if err != nil {
		return nil, err
	}
	if p.peek().kind != tokEOF {
		return nil, fmt.Errorf("Trailing input at %d in %q", p.peek().pos, p.src)
	}
	return e, nil
}

func (p *parser) expr() (*Ast, error) {
	l, err := p.term()
	if err != nil {
		return nil, err
	}
	for {
		switch {
		case p.eat("+"):
			r, err := p.term()
			if err != nil {
				return nil, err
			}
			l = &Ast{Kind: AstBin, Op: '+', L: l, R: r}
		case p.eat("-"):
			r, err := p.term()
			if err != nil {
				return nil, err
			}
			l = &Ast{Kind: AstBin, Op: '-', L: l, R: r}
		default:
			return l, nil
		}
	}
}

func (p *parser) term() (*Ast, error) {
	l, err := p.unary()
	if err != nil {
		return nil, err
	}
	for {
		switch {
		case p.eat("*"):
			r, err := p.unary()
			if err != nil {
				return nil, err
			}
			l = &Ast{Kind: AstBin, Op: '*', L: l, R: r}
		case p.eat("/"):
			r, err := p.unary()
			if err != nil {
				return nil, err
			}
			l = &Ast{Kind: AstBin, Op: '/', L: l, R: r}
		default:
			return l, nil
		}
	}
}

func (p *parser) unary() (*Ast, error) {
	if p.eat("-") {
		e, err := p.unary()
		if err != nil {
			return nil, err
		}
		return &Ast{Kind: AstNeg, L: e}, nil
	}
	return p.power()
}

func (p *parser) power() (*Ast, error) {
	base, err := p.primary()
	if err != nil {
		return nil, err
	}
	if p.eat("^") {
		e, err := p.unary()
		if err != nil {
			return nil, err
		}
		return &Ast{Kind: AstBin, Op: '^', L: base, R: e}, nil
	}
	return base, nil
}

func (p *parser) primary() (*Ast, error) {
	t := p.peek()
	switch {
	case t.kind == tokNum:
		p.i++
		r, err := RatFromDecimal(t.text)
		if err != nil {
			return nil, fmt.Errorf("%w at %d in %q", err, t.pos, p.src)
		}
		return &Ast{Kind: AstNum, Num: r}, nil

	case t.kind == tokIdent:
		p.i++
		if p.eat("(") {
			var args []*Ast
			if !p.eat(")") {
				for {
					a, err := p.expr()
					if err != nil {
						return nil, err
					}
					args = append(args, a)
					if p.eat(",") {
						continue
					}
					if err := p.expect(")"); err != nil {
						return nil, err
					}
					break
				}
			}
			return &Ast{Kind: AstCall, Name: t.text, Args: args}, nil
		}
		return &Ast{Kind: AstSym, Name: t.text}, nil

	case t.kind == tokOp && t.text == "(":
		p.i++
		e, err := p.expr()
		if err != nil {
			return nil, err
		}
		if err := p.expect(")"); err != nil {
			return nil, err
		}
		return e, nil
	}
	return nil, fmt.Errorf("Unexpected token at %d in %q", t.pos, p.src)
}

// Parsed expressions are cached because the same handful of strings are
// re-evaluated on every keystroke.
var astCache sync.Map // string -> *Ast

// ParseExpr parses an expression, memoising the result.
func ParseExpr(src string) (*Ast, error) {
	if hit, ok := astCache.Load(src); ok {
		return hit.(*Ast), nil
	}
	toks, err := tokenize(src)
	if err != nil {
		return nil, err
	}
	ast, err := (&parser{toks: toks, src: src}).parse()
	if err != nil {
		return nil, err
	}
	astCache.Store(src, ast)
	return ast, nil
}

// ExprFunction is a numeric function available inside parameter expressions.
type ExprFunction struct {
	Arity int
	Fn    func(a ...float64) float64
	Doc   string
}

// ExprFunctions is every function a parameter expression may call. They are
// folded eagerly, so their arguments have to be numerically determined.
var ExprFunctions = map[string]ExprFunction{
	"floor":      {1, func(a ...float64) float64 { return math.Floor(a[0]) }, "floor(x)"},
	"ceil":       {1, func(a ...float64) float64 { return math.Ceil(a[0]) }, "ceil(x)"},
	"round":      {1, func(a ...float64) float64 { return jsRound(a[0]) }, "round(x) to nearest integer"},
	"abs":        {1, func(a ...float64) float64 { return math.Abs(a[0]) }, "abs(x)"},
	"sqrt":       {1, func(a ...float64) float64 { return math.Sqrt(a[0]) }, "sqrt(x)"},
	"log2":       {1, func(a ...float64) float64 { return math.Log2(a[0]) }, "log2(x)"},
	"min":        {2, func(a ...float64) float64 { return math.Min(a[0], a[1]) }, "min(a, b)"},
	"max":        {2, func(a ...float64) float64 { return math.Max(a[0], a[1]) }, "max(a, b)"},
	"ceil_mult":  {2, func(a ...float64) float64 { return math.Ceil(a[0]/a[1]) * a[1] }, "ceil_mult(x, k): smallest multiple of k >= x"},
	"floor_mult": {2, func(a ...float64) float64 { return math.Floor(a[0]/a[1]) * a[1] }, "floor_mult(x, k): largest multiple of k <= x"},
	"round_mult": {2, func(a ...float64) float64 { return jsRound(a[0]/a[1]) * a[1] }, "round_mult(x, k): nearest multiple of k"},
}

// jsRound rounds half away from... upward, the way JavaScript's Math.round
// does: round(-0.5) is -0, not -1. Go's math.Round rounds half away from zero,
// which disagrees on negative halves, and a design width is never negative —
// but the engines have to agree everywhere, not only where it matters.
func jsRound(x float64) float64 { return math.Floor(x + 0.5) }

// EvalCtx is the environment a parameter expression evaluates in.
type EvalCtx struct {
	// Values is every known symbol with its numeric value; runtime symbols
	// carry their default.
	Values map[string]float64
	// Known is the set of symbols allowed to appear. An unknown symbol is an
	// error rather than a fresh indeterminate.
	Known map[string]bool
	// Substitutions are names that expand to an expression rather than to an
	// atom. A port pattern written over parameter names ("... in_features")
	// uses this to display the symbol the parameter was bound to (D) instead
	// of the parameter's own name.
	Substitutions map[string]Sym
}

// EvalAst evaluates a parsed expression.
func EvalAst(ast *Ast, ctx EvalCtx) (result Sym, err error) {
	defer catchFault(&err)
	return evalAst(ast, ctx)
}

func evalAst(ast *Ast, ctx EvalCtx) (Sym, error) {
	switch ast.Kind {
	case AstNum:
		return ConRat(ast.Num), nil

	case AstSym:
		if sub, ok := ctx.Substitutions[ast.Name]; ok {
			return sub, nil
		}
		if !ctx.Known[ast.Name] {
			return Zero(), fmt.Errorf("Unknown symbol %q", ast.Name)
		}
		return V(ast.Name), nil

	case AstNeg:
		e, err := EvalAst(ast.L, ctx)
		if err != nil {
			return Zero(), err
		}
		return e.Neg(), nil

	case AstCall:
		f, ok := ExprFunctions[ast.Name]
		if !ok {
			return Zero(), fmt.Errorf("Unknown function %q", ast.Name)
		}
		if len(ast.Args) != f.Arity {
			return Zero(), fmt.Errorf("Function %q expects %d argument(s), got %d", ast.Name, f.Arity, len(ast.Args))
		}
		nums := make([]float64, len(ast.Args))
		for i, a := range ast.Args {
			s, err := EvalAst(a, ctx)
			if err != nil {
				return Zero(), err
			}
			n, ok := s.ToNumber(ctx.Values)
			if !ok {
				return Zero(), fmt.Errorf("Argument of %q is not numeric: %s", ast.Name, s)
			}
			nums[i] = n
		}
		return Con(f.Fn(nums...)), nil

	case AstBin:
		l, err := EvalAst(ast.L, ctx)
		if err != nil {
			return Zero(), err
		}
		r, err := EvalAst(ast.R, ctx)
		if err != nil {
			return Zero(), err
		}
		switch ast.Op {
		case '+':
			return l.Add(r), nil
		case '-':
			return l.Sub(r), nil
		case '*':
			return l.Mul(r), nil
		case '/':
			// Dividing by a constant goes through the float the way the
			// TypeScript does — `rat(1, rc)` — rather than inverting the exact
			// rational. The two differ once a coefficient is already
			// inexact, and a shape label that reads differently in the two
			// engines is a migration bug that would be found by eye.
			if rc, ok := r.AsConst(); ok {
				if rc == 0 {
					return Zero(), fmt.Errorf("Division by zero")
				}
				return l.Mul(ConRat(RatFromPair(1, rc))), nil
			}
			if q, ok := l.DivExact(r); ok {
				return q, nil
			}
			// Both sides determined: fall back to numeric division.
			ln, lok := l.ToNumber(ctx.Values)
			rn, rok := r.ToNumber(ctx.Values)
			if lok && rok && rn != 0 {
				return Con(ln / rn), nil
			}
			return Zero(), fmt.Errorf("Cannot divide %s by %s exactly", l, r)
		case '^':
			e, ok := r.AsConst()
			if !ok || e != math.Trunc(e) || e < 0 {
				return Zero(), fmt.Errorf("Exponent must be a non-negative integer, got %s", r)
			}
			return l.Pow(int(e))
		}
	}
	return Zero(), fmt.Errorf("Unhandled expression node")
}

// EvalExpr parses and evaluates an expression.
func EvalExpr(src string, ctx EvalCtx) (Sym, error) {
	ast, err := ParseExpr(src)
	if err != nil {
		return Zero(), err
	}
	return EvalAst(ast, ctx)
}

// ExprSymbols lists the symbols an expression references, before evaluation.
func ExprSymbols(src string) ([]string, error) {
	ast, err := ParseExpr(src)
	if err != nil {
		return nil, err
	}
	var out []string
	seen := map[string]bool{}
	var walk func(a *Ast)
	walk = func(a *Ast) {
		if a == nil {
			return
		}
		switch a.Kind {
		case AstSym:
			if !seen[a.Name] {
				seen[a.Name] = true
				out = append(out, a.Name)
			}
		case AstCall:
			for _, x := range a.Args {
				walk(x)
			}
		case AstBin:
			walk(a.L)
			walk(a.R)
		case AstNeg:
			walk(a.L)
		}
	}
	walk(ast)
	return out, nil
}
