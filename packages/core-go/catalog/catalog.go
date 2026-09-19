package catalog

import "github.com/tensorcad/core/ir"

// Builtin is every block type the engine ships with.
var Builtin = func() Catalog {
	c := Catalog{}
	for _, d := range Primitives {
		c[d.Type] = d
	}
	for _, d := range Composites {
		c[d.Type] = d
	}
	for _, d := range Containers {
		c[d.Type] = d
	}
	return c
}()

// Of returns the catalog a document resolves against.
//
// The built-in set is fixed, but a design carries its own composites in
// `defs` — a new attention variant, a different block arrangement — the way a
// KiCad project carries its own symbol library. Resolve through this and never
// through Builtin, or a design's own blocks become "unknown block type".
//
// A definition never shadows a built-in, and one that fails to compile is
// dropped rather than thrown: the `user-blocks` design rule reports it.
func Of(doc *ir.Doc) Catalog {
	if doc == nil || len(doc.Defs) == 0 {
		return Builtin
	}
	out := Catalog{}
	for k, v := range Builtin {
		out[k] = v
	}
	for name, raw := range doc.Defs {
		if _, taken := out[name]; taken {
			continue
		}
		if def, err := compileUserBlock(name, raw); err == nil {
			out[name] = def
		}
	}
	return out
}

// IsPrimitive reports whether a block carries its own formulas.
func IsPrimitive(d *BlockDef) bool { return d.Kind == "primitive" }

// IsComposite reports whether a block is an arrangement to be expanded.
func IsComposite(d *BlockDef) bool { return d.Kind == "composite" }

// IsContainer reports whether a block holds a subgraph and carries multipliers.
func IsContainer(d *BlockDef) bool { return d.Kind == "container" }
