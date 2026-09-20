# @tensor-cad/mcp

An MCP server that lets an assistant design transformer language models and find out what they would cost, before anyone writes a training script.

It wraps [`@tensor-cad/engine`](../engine), the analysis compiled from Go to WebAssembly: a typed block graph, symbolic shape inference, a design-rule check, the parameter/FLOPs/memory/cost model, and a PyTorch emitter. The server is headless — it works on `.tensorcad.json` files and the built-in reference architectures, with no editor running.

Built on the official TypeScript SDK v2 (`@modelcontextprotocol/server` 2.0.0), served over stdio.

## Install

### Claude Code

```bash
# from a checkout, during development
claude mcp add --transport stdio tensorcad -- bun packages/mcp/src/stdio.ts

# published
claude mcp add --transport stdio tensorcad -- npx -y @tensor-cad/mcp
```

Add `--scope project` to write a committed `.mcp.json` for everyone on the repo. Resources then surface as `@tensorcad:tensorcad://...` mentions and prompts as `/mcp__tensorcad__design_model`.

### Cursor

`.cursor/mcp.json` in the project, or `~/.cursor/mcp.json` for every project:

```json
{
  "mcpServers": {
    "tensorcad": {
      "command": "npx",
      "args": ["-y", "@tensor-cad/mcp"]
    }
  }
}
```

Cursor keeps a limited number of tools active across all servers, which is why this one ships eighteen rather than forty.

### Claude Desktop, in one click

Download `tensorcad.mcpb` from a release and open it. Claude Desktop installs it
as an extension and asks once for a folder to keep designs in.

The bundle is the server built for Node — which Claude Desktop ships and this
repository does not use — with the WebAssembly engine beside it. `bun run
build:mcpb` builds one and starts it under Node before it is done, because the
two things that have broken it were both differences between Bun and Node rather
than anything in the server.

### Claude Desktop, by hand

`claude_desktop_config.json` (Settings, Developer, Edit Config):

```json
{
  "mcpServers": {
    "tensorcad": {
      "command": "npx",
      "args": ["-y", "@tensor-cad/mcp"],
      "env": { "TENSORCAD_ROOT": "/absolute/path/to/your/designs" }
    }
  }
}
```

### Any client

[`mcp.example.json`](./mcp.example.json) is the repo-relative development form: `bun packages/mcp/src/stdio.ts`, run from the repository root.

`TENSORCAD_ROOT` sets the directory `tensorcad_list_designs` scans for `.tensorcad.json` files and that relative paths resolve against. It defaults to the process working directory.

## Tools

Eighteen, all namespaced `tensorcad_`. Every one declares an `inputSchema` and an `outputSchema` and returns `structuredContent` alongside a readable text mirror; reads are annotated `readOnlyHint` and `idempotentHint`, and the two that overwrite something are annotated `destructiveHint`.

| Tool | What it does |
|---|---|
| `tensorcad_list_designs` | Open designs, the built-in reference architectures, and `.tensorcad.json` files on disk |
| `tensorcad_new_design` | Start from a preset or from nothing; returns a `design_id` |
| `tensorcad_open_design` | Load a `.tensorcad.json`; opening the same path twice returns the same handle |
| `tensorcad_save_design` | Write a design back to disk |
| `tensorcad_get_design` | `format: "outline"` (default) or `"full"` |
| `tensorcad_get_block` | One block: parameters as written and as resolved, port shapes, wiring, parameter count |
| `tensorcad_search_catalog` | The block catalog with parameter schemas, port patterns and docs |
| `tensorcad_apply_ops` | Batched edits with optimistic concurrency |
| `tensorcad_validate` | Every design rule, each finding with a fix hint |
| `tensorcad_analyze` | Parameters, FLOPs, KV cache, memory, throughput, cost, Chinchilla |
| `tensorcad_generate_code` | PyTorch module and config, inline or written to a directory |
| `tensorcad_checkpoint` | Named snapshot |
| `tensorcad_restore` | Back to a checkpoint, or undo the last batch |
| `tensorcad_explain` | One block: what it contributes, and why it is the size it is |
| `tensorcad_scale` | Shrink a design to a parameter budget, keeping its proportions |
| `tensorcad_plan` | Every way to split the training across a cluster, and which of them fit |
| `tensorcad_diff` | What changed between two designs, structurally and numerically |
| `tensorcad_import_hf` | Read a Hugging Face `config.json` into a design |

### Five that answer questions the others cannot

`analyze` says what a design costs; it does not say *why*. `explain` takes one
path and answers that — the parameters as written and as evaluated, the shape on
every port, the share of the model's weights and compute — without the agent
reading the whole document to work it out.

`diff` is the other half of editing. An agent that has just applied four
operations can ask what they moved, structurally and numerically, against the
design it started from. `plan` answers the question that decides whether any of
it matters: would this train on the machines available, and how would it have to
be split.

`scale` and `import_hf` both produce a design rather than a report, and both save
it in the session, so the result can be analysed, diffed and generated from like
any other. That is why the store gained `adopt`: `create` builds a design from a
preset or from nothing, and these arrive whole.

### Handles, not sessions

`tensorcad_new_design` and `tensorcad_open_design` mint a `design_id`; everything else takes it as an argument. That is what the 2026-07-28 revision asks for — cross-call state travels as an explicit handle rather than as connection state — and it means a client can restart the server mid-conversation without losing the thread of *which* design is meant.

Each design carries a `revision` that increments on every change. Pass `expected_revision` to `tensorcad_apply_ops` and a concurrent edit becomes a clear error naming the current revision, instead of a silent overwrite.

### One patch tool, not thirty setters

`tensorcad_apply_ops` takes a list of operations and applies them in order to a copy. The first rejected operation aborts the batch and the design is left exactly as it was, so a half-applied edit is never observable.

| Op | Fields |
|---|---|
| `add_node` | `parent?`, `id`, `type`, `params?`, `label?` |
| `remove_node` | `path` (its edges go with it) |
| `set_param` | `path`, `key`, `value` |
| `connect` | `graph?`, `from`, `to` |
| `disconnect` | `graph?`, `from`, `to` |
| `set_symbol` | `name`, `value` (number, expression, or `null` to delete), `doc?`, `runtime?` |
| `rename` | `path`, `id` (edges are rewritten) |
| `set_label` | `path`, `label` |

A `path` is slash-separated (`layers/block`), and endpoints are `blockId:port` local to their own graph. Most designs are parameterised by symbols — `L` layers, `D` width, `H` heads, `Hkv` key/value heads, `dh` head dimension, `F` feed-forward width, `V` vocabulary — so `set_symbol` is usually the right edit rather than touching individual blocks.

## Resources

| URI | Contents |
|---|---|
| `tensorcad://designs/{id}` | The document plus its outline |
| `tensorcad://designs/{id}/validation` | Every finding |
| `tensorcad://designs/{id}/analysis` | Every number, at the document's own defaults |
| `tensorcad://catalog` | Every block type with parameter schemas and port patterns |
| `tensorcad://catalog/{type}` | One block type |
| `tensorcad://schema/design` | JSON Schema for the `.tensorcad.json` format |

The templated ones support argument completion, and the design and catalog templates enumerate their instances, so a client can offer them without guessing an id.

## Prompts

`design_model` (target parameter count, context length, family), `review_design`, `scale_design`, `explain_costs`.

## A typical session

```
tensorcad_new_design { preset: "llama-3-8b" }        -> dsn_1, revision 1
tensorcad_get_design { design_id: "dsn_1" }          -> the outline: symbols, blocks, edge shapes
tensorcad_checkpoint { design_id: "dsn_1" }          -> ckpt_1
tensorcad_apply_ops  { design_id: "dsn_1", expected_revision: 1,
                    ops: [{ op: "set_symbol", name: "D", value: 5120 },
                          { op: "set_symbol", name: "L", value: 40 }] }
                                                  -> revision 2, parameters and a validation summary
tensorcad_validate   { design_id: "dsn_1" }          -> findings with fix hints
tensorcad_analyze    { design_id: "dsn_1", T: 8192, hardware: "h100-sxm", gpus: 8, zero: 3 }
tensorcad_generate_code { design_id: "dsn_1", out_dir: "out/my-model" }
tensorcad_restore    { design_id: "dsn_1", checkpoint_id: "ckpt_1" }   # if it did not work out
```

## The live editor bridge

Start the server with `TENSORCAD_BRIDGE=1` and a running TensorCAD editor attaches to it. The agent's edits appear on the canvas as it makes them, and the human's edits come back the other way.

```bash
TENSORCAD_BRIDGE=1 bun run packages/mcp/src/stdio.ts
```

Then open the editor (`bun run --cwd packages/ui dev`, or the desktop build). The status bar's rightmost cell says `agent`, and pressing it detaches.

What happens, in order: the editor finds the bridge, publishes the design on screen — so the agent works on *that*, not on a file that resembles it — and the agent sees it in `tensorcad_list_designs`. From then on `tensorcad_apply_ops` lands on the canvas, and what the human does lands in the agent's store, at which point the client is told through `notifications/resources/updated` that the design's resources moved. An edit that arrives from the agent goes onto the editor's undo stack, so the person watching can take it back.

There is no second document. An earlier design had a `LiveStore` beside the `FileStore`, both behind `DocumentStore`; that gives a design two homes and no rule for which one is right when they differ. The bridge observes the one store the tools already write to, and an editor's edit goes through the same `apply` a tool call does — same revision check, same undo log.

**Off unless asked for.** Without the environment variable the server opens no port and is exactly as headless as it was, which is what keeps it usable in CI.

**Who can connect.** It binds 127.0.0.1, so nothing off the machine reaches it. It refuses an upgrade whose `Origin` is not a localhost one, because the same-origin policy does *not* stop a page on the internet opening a WebSocket to your loopback address. And it requires a token, written to `~/.tensorcad/session.json` with owner-only permissions and served to loopback callers of `GET /session` — a browser cannot read the file, which is the only reason that endpoint exists. Anything already running on this machine as you can read the file anyway; the token is not a defence against that and is not meant to be.

`TENSORCAD_BRIDGE_PORT` moves it off 7357. The editor probes that port and the three above it, which is also the range the bridge falls back through when one is taken.

Also deliberately absent: `tensorcad_render_preview` (a canvas image needs the editor) and `tensorcad_run_script` (a scripting escape hatch, which wants a sandbox and an opt-in environment variable before it is worth shipping). Both are in the research notes as future tools; leaving them out keeps the count where clients are happy.

## Publishing

`server.json` beside this README is the [MCP registry][registry] manifest:
the namespace (`io.github.filip-pajalic/tensorcad`, which matches `mcpName` in
`package.json`), the npm package it points at, and the one environment variable
the server reads. It is checked against the registry's own schema when it is
written and against `package.json` on every test run, because what drifts is not
the schema but the version in two files.

```bash
mcp-publisher login github            # the namespace is verified as the GitHub user
mcp-publisher publish --file server.json
```

[registry]: https://github.com/modelcontextprotocol/registry

## Development

```bash
bun run packages/mcp/src/stdio.ts     # serve on stdio
bun test packages/mcp/test            # contract tests: spin the server up and call every tool
```

The contract tests connect with the SDK's own client over a real stdio pipe and validate every tool's `structuredContent` against the `outputSchema` that tool advertised, so the declared contract and the actual answer cannot drift apart.
