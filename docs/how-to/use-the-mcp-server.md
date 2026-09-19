# Drive TensorCAD from an agent

The engine is exposed over [MCP](https://modelcontextprotocol.io/), so an
assistant can design, check, analyse and generate without a human driving the
editor.

## Point a client at it

The repository ships `.mcp.json`:

```json
{
  "mcpServers": {
    "tensorcad": {
      "type": "stdio",
      "command": "bun",
      "args": ["packages/mcp/src/index.ts"],
      "env": { "TENSORCAD_ROOT": "." }
    }
  }
}
```

`TENSORCAD_ROOT` bounds where designs may be read and written.

## The shape of a session

Hold a `design_id` from `tensorcad_new_design` or `tensorcad_open_design` and
pass it to everything else.

**Read cheaply.** `tensorcad_get_design` with `format: "outline"` carries the
whole structure, the symbol table and the shape on every edge for a fraction of
the tokens of the full document.

**Edit atomically.** `tensorcad_apply_ops` takes a batch that is all-or-nothing.
Passing `expected_revision` turns a concurrent edit into a clear error instead of
a silent overwrite.

**Experiment safely.** `tensorcad_checkpoint` before, `tensorcad_restore` to put
it back — or to undo the last batch when you name no checkpoint.

## Prefer symbols to blocks

Most designs are parameterised by symbols: `L` layers, `D` width, `H` heads,
`Hkv` key/value heads, `dh` head dimension, `F` feed-forward width, `V` vocabulary.

A `set_symbol` operation is usually the right edit. Reaching into individual
blocks to change widths is how a design stops being coherent.

## The tools

| | |
|---|---|
| `new_design` `open_design` `save_design` `list_designs` | lifecycle |
| `get_design` `get_block` `search_catalog` | reading |
| `apply_ops` `checkpoint` `restore` | editing |
| `validate` `analyze` | checking and costing |
| `generate_code` | PyTorch out |

`get_block` reports each port's declared shape, and its `dtype` and `optional`
flags where they are not the default — see [Ports](../reference/ports.md).
