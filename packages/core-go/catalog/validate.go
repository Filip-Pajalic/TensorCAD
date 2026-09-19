package catalog

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"

	"github.com/tensorcad/core/ir"
)

// BoundaryIn and BoundaryOut are the generated entry and exit nodes of an
// expansion. A definition may not write them itself, or it could declare one
// interface and wire another.
const (
	BoundaryIn  = "_in"
	BoundaryOut = "_out"
)

var typeNameRe = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

// ValidateUserBlock reports what is wrong with a block a design defines for
// itself.
//
// A definition that fails to compile is dropped from the catalog rather than
// thrown, so without this the only symptom would be "unknown block type"
// against every instance of it, which points at the wrong thing.
func ValidateUserBlock(raw any, name string, taken map[string]bool) []string {
	var errs []string

	encoded, err := json.Marshal(raw)
	if err != nil {
		return []string{fmt.Sprintf("%q could not be read: %s", name, err)}
	}
	var def UserBlockDef
	if err := json.Unmarshal(encoded, &def); err != nil {
		return []string{fmt.Sprintf("%q could not be read: %s", name, err)}
	}
	def.Type = name

	if !typeNameRe.MatchString(def.Type) {
		errs = append(errs, fmt.Sprintf(
			"%q is not a usable type name: lower case, digits and underscores.", def.Type))
	}
	if taken[def.Type] {
		errs = append(errs, fmt.Sprintf("%q is already a built-in block.", def.Type))
	}
	if len(def.Ports.In) == 0 {
		errs = append(errs, "A block needs at least one input port.")
	}
	if len(def.Ports.Out) == 0 {
		errs = append(errs, "A block needs at least one output port.")
	}
	if len(def.Graph.Nodes) == 0 {
		errs = append(errs, "A block needs at least one node.")
	}

	for _, n := range def.Graph.Nodes {
		if n.ID == BoundaryIn || n.ID == BoundaryOut {
			errs = append(errs, fmt.Sprintf(
				"%q and %q are generated; do not define them.", BoundaryIn, BoundaryOut))
			break
		}
	}

	// Every $ref has to name a parameter, or the expansion produces a "0" that
	// silently counts as a valid expression.
	known := map[string]bool{}
	for k := range def.Params {
		known[k] = true
	}
	missing := map[string]bool{}
	var scan func(nodes []ir.NodeDef)
	scan = func(nodes []ir.NodeDef) {
		for _, node := range nodes {
			for _, value := range node.Params {
				s, ok := value.(string)
				if !ok {
					continue
				}
				for _, m := range refRe.FindAllStringSubmatch(s, -1) {
					if !known[m[1]] {
						missing[m[1]] = true
					}
				}
			}
			if node.Graph != nil {
				scan(node.Graph.Nodes)
			}
		}
	}
	scan(def.Graph.Nodes)

	// Sorted, where the TypeScript reports them in the order it met them: a Go
	// map has no order, and a stable list beats an arbitrary one.
	names := make([]string, 0, len(missing))
	for name := range missing {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		errs = append(errs, fmt.Sprintf("$%s is not a parameter of this block.", name))
	}
	return errs
}
