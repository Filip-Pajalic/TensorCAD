# The document format

The design is JSON. It is the source of truth: editor state is a view of it, and
everything downstream — validation, analysis, code generation, MCP — reads this
and nothing else.

```json
{
  "version": 1,
  "meta": { "name": "my-model", "family": "…", "notes": "…",
            "published": { "params": 8030261248, "source": "https://…" } },
  "symbols": { "D": 4096, "H": 32, "F": "ceil_mult(1.3*8/3*D, 1024)" },
  "graph": { "nodes": [...], "edges": [...] },
  "defs": { "my_block": { ... } },
  "ui": { "positions": { "attn": [120, 40] } }
}
```

## `symbols`

The design's free variables. Four spellings:

| Written | Means |
|---|---|
| `4096` | A literal |
| `"4*D"` | An expression over other symbols |
| `{ "kind": "runtime", "default": 2048 }` | Stays indeterminate through shape checking |
| `{ "kind": "design", "value": …, "doc": "…" }` | The long form, with documentation |

Symbols may depend on each other and are evaluated in dependency order; a cycle
is reported rather than hung on.

**`B` and `T` are reserved runtime symbols** and are always available even if the
document omits them. They stay symbolic so a shape mismatch involving them is a
real polynomial difference. See [the shape algebra](../explanation/shape-algebra.md).

## `graph`

```json
{
  "nodes": [
    { "id": "attn", "type": "gqa_attention",
      "params": { "d_model": "D", "heads": "H" }, "label": "Attention" },
    { "id": "layers", "type": "repeat", "params": { "count": "L" },
      "graph": { "nodes": [...], "edges": [...] } }
  ],
  "edges": [["embed:y", "attn:x"]]
}
```

An edge is `["<node>:<port>", "<node>:<port>"]`, producer first. Node ids are
local to their graph; a path through nesting is slash-separated,
`"layers/block/attn"`.

A parameter value may be a number, a string expression over the symbol table, a
boolean, or an object. **Expressions are stored as written**, not as the number
they evaluate to — that is what makes `scaleDesign` possible.

A container node carries a `graph`, entered through `boundary_in` and
`boundary_out` nodes that declare its ports.

## `defs`

Composites the design defines for itself, resolving exactly like built-ins. See
[Define a block inside a document](../how-to/define-a-block-in-a-document.md).

## `meta.published`

What the design claims to reproduce. The regression suite asserts it, and the
`published-drift` rule fires when a design stops matching its own claim.

`tolerance` widens the check, and belongs only where the published figure is
itself rounded.

## `ui`

Node positions and collapsed state. The only part of the document that is about
the drawing rather than the design, and the only part nothing downstream reads.
