// Package presets is the design library the engine ships with.
//
// Every preset carries meta.published: the parameter count its authors
// reported, which the tests assert the analysis reproduces. They are the
// regression suite, not examples — a change that breaks Llama or DeepSeek
// breaks a test before it breaks a design.
//
// They are documents rather than code. A preset is a design, a design is the
// IR, and the IR is JSON; building them in a host language would make the
// library something only that language could read, and something the editor
// could not round-trip.
package presets

import (
	"embed"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/tensorcad/core/ir"
)

//go:embed data/*.json
var files embed.FS

var (
	once  sync.Once
	names []string
	index map[string]bool
	load  error
)

func read() {
	once.Do(func() {
		raw, err := files.ReadFile("data/index.json")
		if err != nil {
			load = fmt.Errorf("preset index: %w", err)
			return
		}
		if err := json.Unmarshal(raw, &names); err != nil {
			load = fmt.Errorf("preset index: %w", err)
			return
		}
		index = make(map[string]bool, len(names))
		for _, n := range names {
			index[n] = true
		}
	})
}

// Names lists every preset, in the order the library presents them.
func Names() ([]string, error) {
	read()
	if load != nil {
		return nil, load
	}
	return append([]string{}, names...), nil
}

// Get returns a preset's document.
//
// A fresh copy each time: a caller that edits what it is given must not change
// what the next caller sees.
func Get(name string) (*ir.Doc, error) {
	read()
	if load != nil {
		return nil, load
	}
	if !index[name] {
		return nil, fmt.Errorf("unknown preset %q", name)
	}
	raw, err := files.ReadFile("data/" + name + ".json")
	if err != nil {
		return nil, fmt.Errorf("preset %q: %w", name, err)
	}
	var doc ir.Doc
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("preset %q: %w", name, err)
	}
	return &doc, nil
}

// MustGet is Get for a name the caller knows exists, such as a test fixture.
func MustGet(name string) *ir.Doc {
	doc, err := Get(name)
	if err != nil {
		panic(err)
	}
	return doc
}
