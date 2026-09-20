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

## Watch it work

By default the server is headless: it reads and writes `.tensorcad.json` files
and never touches a window. Start it with `TENSORCAD_BRIDGE=1` and a running
editor attaches to it instead.

```bash
TENSORCAD_BRIDGE=1 npx @tensor-cad/mcp
```

Open the editor and the rightmost cell of the status bar says `agent`. What
happens from there:

- The editor **publishes the design on screen**, so the agent works on that
  rather than on a file that resembles it. It appears in `list_designs`.
- Every `apply_ops` **lands on the canvas** as it is made, and goes onto the
  editor's undo stack — an agent's edit is one a person watching can take back.
- What the person does **comes back the other way**, and the client is told
  through `notifications/resources/updated` that the design's resources moved.

There is one document, not two. An editor's edit goes through the same `apply`
a tool call does, so the revision check that stops two agents overwriting each
other is the same one that stops an agent overwriting a human.

!!! note "It only listens to this machine"

    The bridge binds `127.0.0.1`, refuses any connection whose `Origin` is not
    a localhost one — the same-origin policy does *not* stop a page you visited
    opening a socket to your own machine — and wants a token from
    `~/.tensorcad/session.json`. Without `TENSORCAD_BRIDGE=1` it opens no port
    at all, which is what keeps the server usable in CI.

    `TENSORCAD_BRIDGE_PORT` moves it off 7357. The editor probes that port and
    the three above it.

The hosted editor at [app.tensorcad.dev](https://app.tensorcad.dev/) cannot
attach: a browser will not open a plain socket from an https page, and there is
no agent on the far side of the internet to attach to. Use the dev server or
the desktop build.
