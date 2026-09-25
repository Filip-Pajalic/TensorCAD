package rules

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
)

// The input-roles check.
//
// A mask that keeps documents apart reads a B T integer tensor, and so does a
// rotation that restarts at every document, and so does every token id in the
// design: the shape checker and the element-type check both pass the tokens
// wired where either belongs. The analysis would then count the mask as
// though it had been given documents, and the model would keep apart whatever
// tokens happened to be equal, or turn each token by its id. So the wire is
// followed back to where it starts, through every stack it was handed into,
// and what it starts at has to say it holds what is read.
var rolesWired = Rule{
	ID:    "input-roles",
	Title: "Input roles",
	Description: "A port that reads an input for what it holds — the documents, for a mask that keeps " +
		"them apart; each token's place in its document, for a rotation that restarts — is given an input " +
		"whose role says so, and not the tokens, which have the same shape and element type.",
	Run: func(ctx *Ctx) []Finding {
		blocks := make(map[string]*analysis.FlatBlock, len(ctx.Flat.Blocks))
		for i := range ctx.Flat.Blocks {
			blocks[ctx.Flat.Blocks[i].Path] = &ctx.Flat.Blocks[i]
		}
		// source follows a pin up through the stacks it was handed into:
		// a stack's input boundary passes on whatever reached the stack.
		source := func(pin string) (string, bool) {
			for hops := 0; hops < 64; hops++ {
				path, port, ok := splitPin(pin)
				if !ok {
					return "", false
				}
				b := blocks[path]
				at := strings.LastIndex(path, "/")
				if b == nil || b.Type != "boundary_in" || at < 0 {
					return path, true
				}
				next, wired := ctx.Infer.ProducerOf[path[:at]+":"+port]
				if !wired {
					return "", false
				}
				pin = next
			}
			return "", false
		}

		paths := make([]string, 0, len(ctx.Infer.Ports))
		for path := range ctx.Infer.Ports {
			paths = append(paths, path)
		}
		sort.Strings(paths)

		type reads struct{ port, role, param, reader string }
		var out []Finding
		for _, path := range paths {
			b := blocks[path]
			if b == nil || b.Resolved == nil {
				continue
			}
			var wants []reads
			for _, name := range catalog.DocumentsOf(b.Resolved) {
				wants = append(wants, reads{name, "documents", "mask", "The mask"})
			}
			if _, has := ctx.Infer.Ports[path].In["pos"]; has && b.Resolved.Bool("positions") {
				reader := "The rotation"
				if b.Type == "pos_embedding" {
					reader = "The position embedding"
				}
				wants = append(wants, reads{"pos", "positions", "positions", reader})
			}
			for _, w := range wants {
				producer, wired := ctx.Infer.ProducerOf[path+":"+w.port]
				if !wired {
					out = append(out, Finding{
						Rule: "input-roles", Severity: "error", Path: path, Port: w.port, Param: w.param,
						Message: fmt.Sprintf("%s reads %s, and nothing is wired to it.", w.reader, w.port),
						Hint:    fmt.Sprintf("Wire an input whose role is %s to it: B T integers.", w.role),
					})
					continue
				}
				from, ok := source(producer)
				if !ok {
					continue
				}
				if s := blocks[from]; s != nil && s.Type == "input" && s.Resolved != nil &&
					s.Resolved.Str("role") == w.role {
					continue
				}
				out = append(out, Finding{
					Rule: "input-roles", Severity: "error", Path: path, Port: w.port, Param: w.param,
					Message: fmt.Sprintf("%s reads %s from %s, which is not a %s input.", w.reader, w.port, from, w.role),
					Hint: fmt.Sprintf("Wire an input whose role is %s. Anything else has the same shape and "+
						"element type, so nothing else notices, and the model would compute with whatever it "+
						"was given as though it were %s.", w.role, w.role),
				})
			}
		}
		return out
	},
}
