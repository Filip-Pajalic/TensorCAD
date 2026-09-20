// Regenerate the golden files.
//
//	go run ./cmd/golden          # from packages/core-go
//
// Deliberately. A diff in testdata is either a bug being fixed or a behaviour
// being changed, and both want to be seen in review.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/tensorcad/core/golden"
)

func main() {
	dir := flag.String("dir", "testdata", "where the golden files live")
	flag.Parse()

	summary, err := golden.Write(*dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "golden: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf(
		"wrote %d presets: symbol tables and inferred shapes, %d analyses and rule checks,\n"+
			"  %d generated PyTorch files, %d primitive cases, %d composite expansions,\n"+
			"  the prose of %d blocks, %d scaled designs and %d findings on designs that are wrong\n",
		summary.Presets, summary.OperatingPoints, summary.CodegenFiles,
		summary.Primitives, summary.Composites, summary.Blocks,
		summary.ScaleCases, summary.BrokenFindings)
}
