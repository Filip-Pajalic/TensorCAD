package rules

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tensorcad/core/analysis"
	"github.com/tensorcad/core/catalog"
)

// The documents check.
//
// A mask that keeps documents apart reads a B T integer tensor, and so does
// every token id in the design: the shape checker and the element-type check
// both pass the tokens wired where the documents belong. The analysis would
// then count the mask as though it had been given documents, and the model
// would keep apart whatever tokens happened to be equal. So the wire is
// followed back to where it starts, through every stack it was handed into,
// and what it starts at has to say it holds documents.
var documentsWired = Rule{
	ID:    "documents",
	Title: "Documents",
	Description: "A mask that keeps documents apart has to be given the documents: an input whose role " +
		"says so, and not the tokens, which have the same shape and the same element type.",
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

		var out []Finding
		for _, path := range paths {
			b := blocks[path]
			if b == nil || b.Resolved == nil {
				continue
			}
			for _, name := range catalog.DocumentsOf(b.Resolved) {
				producer, wired := ctx.Infer.ProducerOf[path+":"+name]
				if !wired {
					out = append(out, Finding{
						Rule: "documents", Severity: "error", Path: path, Port: name, Param: "mask",
						Message: fmt.Sprintf("The mask reads %s, and nothing is wired to it.", name),
						Hint:    "Wire an input with role documents to it: each position's document, B T integers.",
					})
					continue
				}
				from, ok := source(producer)
				if !ok {
					continue
				}
				if s := blocks[from]; s != nil && s.Type == "input" && s.Resolved != nil &&
					s.Resolved.Str("role") == "documents" {
					continue
				}
				out = append(out, Finding{
					Rule: "documents", Severity: "error", Path: path, Port: name, Param: "mask",
					Message: fmt.Sprintf("The mask reads %s from %s, which is not a documents input.", name, from),
					Hint: "A mask can read only each position's document: an input whose role is documents. " +
						"Anything else is counted as though it were one, and kept apart wherever two of its " +
						"values differ.",
				})
			}
		}
		return out
	},
}
