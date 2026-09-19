# Define a block inside a document

A design can carry its own composites, the way a KiCad project carries its own
symbol library. No TypeScript, no rebuild — the definition lives in the document
JSON and resolves exactly like a built-in.

This is how you add an architecture the catalog does not have.

## Where it goes

`doc.defs`, keyed by type name:

```json
{
  "defs": {
    "my_gated_ffn": {
      "category": "mlp",
      "params": {
        "d_model": { "type": "int", "min": 1 },
        "hidden":  { "type": "int", "min": 1 }
      },
      "ports": {
        "in":  { "x": "... d_model" },
        "out": { "y": "... d_model" }
      },
      "graph": {
        "nodes": [
          { "id": "up",   "type": "linear",     "params": { "in_features": "$d_model", "out_features": "$hidden" } },
          { "id": "gate", "type": "linear",     "params": { "in_features": "$d_model", "out_features": "$hidden" } },
          { "id": "act",  "type": "activation", "params": { "kind": "silu", "dim": "$hidden" } },
          { "id": "mul",  "type": "mul",        "params": { "dim": "$hidden" } },
          { "id": "down", "type": "linear",     "params": { "in_features": "$hidden", "out_features": "$d_model" } }
        ],
        "edges": [
          ["_in:x", "up:x"], ["_in:x", "gate:x"],
          ["gate:y", "act:x"], ["act:y", "mul:a"], ["up:y", "mul:b"],
          ["mul:y", "down:x"], ["down:y", "_out:x"]
        ]
      },
      "docs": { "summary": "A gated feed-forward." }
    }
  }
}
```

## The rules

**`$name` refers to the block's own parameters.** Substituted when the definition
expands, so an instance with `hidden: 14336` gets that number everywhere `$hidden`
appears.

**Boundary nodes are generated, not written.** `_in` and `_out` come from the
declared `ports`, so a definition cannot declare one interface and wire another.
Refer to them by those ids in `edges`.

**A definition never shadows a built-in.** If the name is taken, yours is ignored.

**A broken definition is reported, not thrown.** It is dropped from the catalog
and the `userBlocks` design rule says why.

## From the editor

**Blocks ▸ Make a block from this level** lifts what is on screen into a
definition, promoting the symbols it uses into parameters. A block extracted from
a 4,096-wide design therefore works at 2,048 without being rebuilt.

**Blocks ▸ Import / Export** move definitions between documents as JSON.

## What you cannot do this way

Add a new *formula*. If your block's parameter count or FLOPs cannot be derived
from the primitives it contains, it needs to be a primitive — see
[Add a block to the catalog](add-a-block.md). In practice this is rare: almost
every architecture is a new arrangement of existing operations.
