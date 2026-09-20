package ir

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"

	"github.com/tensorcad/core/shapes"
)

// UnmarshalJSON decodes a document and remembers the order its symbols were
// written in.
//
// Go's maps do not keep insertion order and JSON objects have none to begin
// with, but the symbol list is something a person wrote and reads back: D
// before H before dh, not alphabetical. So the raw bytes are scanned once for
// the key order and everything downstream works from that.
func (d *Doc) UnmarshalJSON(b []byte) error {
	type plain Doc
	var p plain
	if err := json.Unmarshal(b, &p); err != nil {
		return err
	}
	*d = Doc(p)
	d.SymbolOrder = objectKeyOrder(b, "symbols")
	d.ConfigurationOrder = objectKeyOrder(b, "configurations")
	return nil
}

// ConfigurationNames is the configurations in the order they were written,
// which is the order a picker should offer them in: small before large, not
// alphabetical.
func ConfigurationNames(doc *Doc) []string {
	out := make([]string, 0, len(doc.Configurations))
	seen := map[string]bool{}
	for _, name := range doc.ConfigurationOrder {
		if _, ok := doc.Configurations[name]; ok && !seen[name] {
			seen[name] = true
			out = append(out, name)
		}
	}
	rest := make([]string, 0, len(doc.Configurations))
	for name := range doc.Configurations {
		if !seen[name] {
			rest = append(rest, name)
		}
	}
	sort.Strings(rest)
	return append(out, rest...)
}

// ActiveSymbols is the design's symbols with the active configuration applied.
//
// The document's own symbols are left alone: a configuration is a view of the
// design rather than an edit to it, and switching between two of them has to be
// something you can do twice and end up where you started.
func ActiveSymbols(doc *Doc) map[string]SymbolDef {
	config, ok := doc.Configurations[doc.Active]
	if !ok || len(config.Symbols) == 0 {
		return doc.Symbols
	}
	out := make(map[string]SymbolDef, len(doc.Symbols)+len(config.Symbols))
	for name, def := range doc.Symbols {
		out[name] = def
	}
	for name, def := range config.Symbols {
		out[name] = def
	}
	return out
}

// MarshalJSON writes the symbols back in the order they were written in.
//
// The other half of UnmarshalJSON, and the half whose absence is invisible
// until a document makes a round trip: the engine hands a design to a client,
// the client hands it back, and a symbol list a person wrote as D, H, dh comes
// back alphabetized, because Go sorts a map's keys on the way out. Everything
// downstream that reads the order — the generated file's header, the order
// findings about symbols come out in, the panel the user reads — quietly moves.
//
// Only the symbols object is rewritten; every other field keeps the encoding
// and the position the struct gives it.
func (d Doc) MarshalJSON() ([]byte, error) {
	type plain Doc
	b, err := marshalNoEscape(plain(d))
	if err != nil {
		return nil, err
	}
	if len(d.Symbols) < 2 || len(d.SymbolOrder) != len(d.Symbols) {
		return b, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(b, &fields); err != nil {
		return b, nil
	}
	if _, ok := fields["symbols"]; !ok {
		return b, nil
	}
	var ordered bytes.Buffer
	ordered.WriteByte('{')
	for i, name := range d.SymbolOrder {
		sym, ok := d.Symbols[name]
		if !ok {
			// The order disagrees with the map, so it is not an order for this
			// document. Leave what the encoder produced.
			return b, nil
		}
		if i > 0 {
			ordered.WriteByte(',')
		}
		key, err := marshalNoEscape(name)
		if err != nil {
			return b, nil
		}
		value, err := marshalNoEscape(sym)
		if err != nil {
			return b, nil
		}
		ordered.Write(key)
		ordered.WriteByte(':')
		ordered.Write(value)
	}
	ordered.WriteByte('}')
	fields["symbols"] = ordered.Bytes()

	var out bytes.Buffer
	out.WriteByte('{')
	for i, key := range objectKeys(b) {
		if i > 0 {
			out.WriteByte(',')
		}
		k, err := marshalNoEscape(key)
		if err != nil {
			return b, nil
		}
		out.Write(k)
		out.WriteByte(':')
		out.Write(fields[key])
	}
	out.WriteByte('}')
	return out.Bytes(), nil
}

// marshalNoEscape encodes without turning `<`, `>` and `&` into escapes.
//
// A MarshalJSON result is compacted by whichever encoder asked for it, and that
// is where escaping is decided; anything escaped here would be escaped whatever
// the caller wanted.
func marshalNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// objectKeys is the top-level keys of a JSON object, in order.
func objectKeys(b []byte) []string {
	dec := json.NewDecoder(bytes.NewReader(b))
	if t, err := dec.Token(); err != nil || t != json.Delim('{') {
		return nil
	}
	var keys []string
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil
		}
		key, _ := kt.(string)
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			return nil
		}
		keys = append(keys, key)
	}
	return keys
}

func objectKeyOrder(b []byte, field string) []string {
	dec := json.NewDecoder(bytes.NewReader(b))
	t, err := dec.Token()
	if err != nil || t != json.Delim('{') {
		return nil
	}
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil
		}
		key, _ := kt.(string)
		if key != field {
			var skip json.RawMessage
			if err := dec.Decode(&skip); err != nil {
				return nil
			}
			continue
		}
		if t, err := dec.Token(); err != nil || t != json.Delim('{') {
			return nil
		}
		var keys []string
		for dec.More() {
			kk, err := dec.Token()
			if err != nil {
				return nil
			}
			s, _ := kk.(string)
			keys = append(keys, s)
			var skip json.RawMessage
			if err := dec.Decode(&skip); err != nil {
				return nil
			}
		}
		return keys
	}
	return nil
}

var identRe = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

type normalized struct {
	name    string
	runtime bool
	literal float64
	hasLit  bool
	expr    string
	doc     string
}

func normalize(name string, def SymbolDef) (normalized, error) {
	switch def.Kind {
	case "literal":
		return normalized{name: name, literal: def.Number, hasLit: true}, nil
	case "expr":
		return normalized{name: name, expr: def.Expr}, nil
	case "runtime":
		out := normalized{name: name, runtime: true, doc: def.Doc}
		if def.HasNumber {
			out.literal, out.hasLit = def.Number, true
		}
		return out, nil
	case "design":
		out := normalized{name: name, doc: def.Doc}
		if def.HasNumber {
			out.literal, out.hasLit = def.Number, true
		} else {
			out.expr = def.Expr
		}
		return out, nil
	}
	return normalized{}, fmt.Errorf("symbol %q has an unsupported definition", name)
}

// ResolveSymbols evaluates the symbol table.
//
// Symbols may depend on each other — F = ceil_mult(1.3*8/3*D, 1024) — so they
// are evaluated in dependency order, with cycles reported rather than hung on.
func ResolveSymbols(doc *Doc) *SymbolTable {
	table := &SymbolTable{
		// Empty rather than nil throughout: this crosses to the editor as JSON,
		// where a nil slice is null and a panel asking for its length finds
		// nothing to ask.
		Order:        []string{},
		Values:       map[string]float64{},
		DesignValues: map[string]float64{},
		Runtime:      map[string]bool{},
		Docs:         map[string]string{},
		Errors:       []string{},
	}

	defs := map[string]normalized{}
	var declOrder []string
	add := func(name string, n normalized) {
		if _, seen := defs[name]; !seen {
			declOrder = append(declOrder, name)
		}
		defs[name] = n
	}

	// Through the active configuration, so every number downstream is the one
	// the design is currently built at. The document's own symbols are left as
	// written: a configuration is a view of the design, not an edit to it.
	symbols := ActiveSymbols(doc)
	for _, name := range symbolNames(doc) {
		def := symbols[name]
		if !identRe.MatchString(name) {
			table.Errors = append(table.Errors, fmt.Sprintf("Symbol name %q is not a valid identifier", name))
			continue
		}
		n, err := normalize(name, def)
		if err != nil {
			table.Errors = append(table.Errors, err.Error())
			continue
		}
		add(name, n)
	}

	// Runtime symbols are always available, even if the document forgot them.
	for _, name := range RuntimeSymbols {
		if _, ok := defs[name]; ok {
			continue
		}
		lit := 2048.0
		if name == "B" {
			lit = 1
		}
		add(name, normalized{name: name, runtime: true, literal: lit, hasLit: true})
	}

	known := map[string]bool{}
	for name := range defs {
		known[name] = true
	}

	const (
		pending = 1
		done    = 2
	)
	state := map[string]int{}

	var visit func(name string, stack []string)
	visit = func(name string, stack []string) {
		switch state[name] {
		case done:
			return
		case pending:
			cycle := append(append([]string{}, stack...), name)
			table.Errors = append(table.Errors, "Symbol cycle: "+joinArrow(cycle))
			table.Values[name] = math.NaN()
			state[name] = done
			return
		}
		def, ok := defs[name]
		if !ok {
			table.Errors = append(table.Errors, fmt.Sprintf("Unknown symbol %q", name))
			return
		}
		state[name] = pending

		switch {
		case def.runtime:
			table.Runtime[name] = true
			if def.hasLit {
				table.Values[name] = def.literal
			} else {
				table.Values[name] = 1
			}

		case def.hasLit:
			table.Values[name] = def.literal
			table.DesignValues[name] = def.literal

		case def.expr != "":
			deps, err := shapes.ExprSymbols(def.expr)
			if err != nil {
				table.Errors = append(table.Errors, fmt.Sprintf("Symbol %q: %s", name, err))
			}
			for _, d := range deps {
				if !known[d] {
					table.Errors = append(table.Errors, fmt.Sprintf("Symbol %q references unknown symbol %q", name, d))
					continue
				}
				visit(d, append(stack, name))
			}
			sym, err := shapes.EvalExpr(def.expr, shapes.EvalCtx{Values: table.Values, Known: known})
			if err != nil {
				table.Errors = append(table.Errors, fmt.Sprintf("Symbol %q: %s", name, err))
				table.Values[name] = math.NaN()
				break
			}
			// Evaluate rather than ToNumber: the table can hold a NaN for a
			// symbol that already failed, and a symbol that mentions it has to
			// report that rather than fault.
			v, ok, err := sym.Evaluate(table.Values)
			switch {
			case err != nil:
				table.Errors = append(table.Errors, fmt.Sprintf("Symbol %q: %s", name, err))
				table.Values[name] = math.NaN()
			case !ok:
				table.Errors = append(table.Errors,
					fmt.Sprintf("Symbol %q does not evaluate to a number (got %s)", name, sym))
				table.Values[name] = math.NaN()
			default:
				table.Values[name] = v
				table.DesignValues[name] = v
			}
		}

		table.Docs[name] = def.doc
		table.Order = append(table.Order, name)
		state[name] = done
	}

	for _, name := range declOrder {
		visit(name, nil)
	}
	return table
}

// symbolNames returns the document's symbols in the order they were written,
// falling back to whatever the map gives when the order was not recorded — a
// document built in memory rather than decoded.
func symbolNames(doc *Doc) []string {
	if len(doc.SymbolOrder) > 0 {
		out := make([]string, 0, len(doc.Symbols))
		seen := map[string]bool{}
		for _, name := range doc.SymbolOrder {
			if _, ok := doc.Symbols[name]; ok && !seen[name] {
				seen[name] = true
				out = append(out, name)
			}
		}
		for name := range doc.Symbols {
			if !seen[name] {
				out = append(out, name)
			}
		}
		return out
	}
	out := make([]string, 0, len(doc.Symbols))
	for name := range doc.Symbols {
		out = append(out, name)
	}
	return out
}

func joinArrow(names []string) string {
	var b bytes.Buffer
	for i, n := range names {
		if i > 0 {
			b.WriteString(" -> ")
		}
		b.WriteString(n)
	}
	return b.String()
}

// SymbolCtx is the evaluation context for parameter expressions in a document.
func SymbolCtx(table *SymbolTable) shapes.EvalCtx {
	known := make(map[string]bool, len(table.Values))
	for name := range table.Values {
		known[name] = true
	}
	return shapes.EvalCtx{Values: table.Values, Known: known}
}
