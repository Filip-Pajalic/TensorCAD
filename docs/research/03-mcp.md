# Research 03 — MCP: spec state, SDKs, and how CAD/design tools expose themselves

Researched 2026-09-18.

## 1. MCP current state (Sept 2026)

### 1.1 Spec revision 2026-07-28 (previous: 2025-11-25, 2025-06-18)

Biggest revision since launch ([changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog), [RC post](https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/)). What matters for a server author:

| Area | 2026-07-28 behavior |
|---|---|
| Statelessness | `initialize` handshake and `Mcp-Session-Id` removed. Every request carries `_meta.io.modelcontextprotocol/protocolVersion` + `clientCapabilities`; servers implement `server/discover`. Cross-call state = **explicit server-minted handles passed as tool args** (spec's `create_basket → basket_id` example) ([tools spec](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)). |
| Transports | **stdio**: newline-delimited JSON-RPC on stdout, logs to stderr, nothing else on stdout; client restarts crashed servers. **Streamable HTTP**: single POST endpoint, per-request JSON or SSE response, required `MCP-Protocol-Version`/`Mcp-Method`/`Mcp-Name` headers, no GET stream, must validate `Origin` and bind to 127.0.0.1 locally. HTTP+SSE deprecated. |
| Server→client (MRTR) | Servers no longer send requests; they return `resultType: "input_required"` with `inputRequests` (elicitation etc.) and the client retries with `inputResponses` + `requestState`. |
| Notifications | `resources/subscribe` etc. replaced by one `subscriptions/listen` request whose filter opts into `toolsListChanged`, `promptsListChanged`, `resourcesListChanged`, `resourceSubscriptions: [uri...]` ([subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)). Works over stdio. |
| Tools | `inputSchema`/`outputSchema` any JSON Schema 2020-12; `structuredContent` any JSON value, mirrored as text; `annotations` (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) drive client confirmation UX; tools/list should be deterministically ordered (prompt-cache hits). Two error channels: protocol errors vs `isError: true` results the model can self-correct from. |
| Resources / prompts | `resources/list`, `resources/read`, `resources/templates/list` (RFC 6570 `uriTemplate`), completion for template args, `annotations {audience, priority, lastModified}`, `resource_link` and embedded resources in tool results. |
| Elicitation | Form mode (flat object of primitives/enums) and URL mode; delivered via MRTR; accept/decline/cancel. |
| Deprecated | Roots, Sampling, Logging. Pass paths as tool args; call LLM APIs directly; log to stderr. Don't build on these. |
| Extensions | `capabilities.extensions`. Official: **MCP Apps** (`ui://` HTML in sandboxed iframe), **Tasks** (`resultType:"task"` + `tasks/get`/`tasks/cancel`), **Skills over MCP** (`skills/list`, `skills/get`, SKILL.md via resources), auth ([overview](https://modelcontextprotocol.io/extensions/overview), [tasks](https://modelcontextprotocol.io/extensions/tasks/overview), [skills](https://modelcontextprotocol.io/extensions/skills/overview)). |

Client support for extensions is uneven: MCP Apps is supported by Claude web/Desktop, VS Code Copilot, Cursor, ChatGPT and others, but **not Claude Code** (issue [#95149](https://github.com/anthropics/claude-code/issues/95149)); Tasks does not appear in the [client matrix](https://modelcontextprotocol.io/extensions/client-matrix) yet. Treat both as progressive enhancements.

### 1.2 Official SDKs

**TypeScript v2**: split packages `@modelcontextprotocol/server` (2.0.0, deps `@modelcontextprotocol/core` + `zod ^4`, Node ≥20), `/client`, `/node`, `/express`, `/fastify`, `/hono`, `/codemod`. `@modelcontextprotocol/sdk` 1.30.0 is the maintenance line ([repo](https://github.com/modelcontextprotocol/typescript-sdk), [v2 docs](https://ts.sdk.modelcontextprotocol.io/v2/), [tools guide](https://ts.sdk.modelcontextprotocol.io/v2/servers/tools.html)). Canonical pattern:

```ts
import { McpServer } from '@modelcontextprotocol/server';
import { serveStdio } from '@modelcontextprotocol/server/stdio';
import * as z from 'zod/v4';
serveStdio(() => {
  const server = new McpServer({ name: 'tensorcad', version: '0.1.0' });
  server.registerTool('analyze_design',
    { description: '...', inputSchema: z.object({ design: z.string() }),
      outputSchema: z.object({ params: z.number(), flops: z.number() }),
      annotations: { readOnlyHint: true } },
    async ({ design }) => ({ content: [{ type: 'text', text: '...' }], structuredContent: {...} }));
  return server;
});
```

**Python v2**: `mcp` 2.2.0 (2026-09-07); `FastMCP` renamed `MCPServer` (`from mcp.server import MCPServer`), same `@mcp.tool()/@mcp.resource()/@mcp.prompt()` decorators, type hints → schemas, `structured_output=True` opt-in, stdio and `--transport streamable-http` ([releases](https://github.com/modelcontextprotocol/python-sdk/releases)). Standalone PrefectHQ FastMCP 4.x reportedly still bundles `mcp` 1.x (unverified).

### 1.3 Distribution and discovery

- **Official registry** ([registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io), metadata-only, preview). Flow: publish to npm/PyPI → add `"mcpName": "io.github.<user>/<server>"` to `package.json` → `mcp-publisher init` → `server.json` → `mcp-publisher login github` → `mcp-publisher publish` ([quickstart](https://modelcontextprotocol.io/registry/quickstart)).
- **Claude Code**: `claude mcp add --transport stdio tensorcad -- npx -y @tensorcad/mcp`; project scope writes committed `.mcp.json`. Resources surface as `@server:uri` mentions, prompts as `/mcp__server__prompt`. Tool Search loads tools on demand; tool calls over 2 min auto-background ([docs](https://code.claude.com/docs/en/mcp)).
- **Claude Desktop**: `claude_desktop_config.json` `mcpServers` block, or an **MCPB bundle** (`npm i -g @anthropic-ai/mcpb; mcpb init; mcpb pack`; `manifest.json` with `server.type: node`, `user_config`, `tools[]`; directory submission requires tool annotations) ([MCPB guide](https://claude.com/docs/connectors/building/mcpb), [manifest](https://github.com/modelcontextprotocol/mcpb/blob/main/MANIFEST.md)).
- **Cursor**: `.cursor/mcp.json`, one-click deeplink install ([docs](https://cursor.com/docs/context/mcp)). Reported hard cap of ~40 active tools across all servers, silently dropped beyond ([third-party](https://mcpverdict.com/mcp/clients/cursor/)). Keep the tool count small.

## 2. How design/CAD tools expose themselves via MCP

| Tool | Bridge to the app | Tool surface | Notable |
|---|---|---|---|
| [Blender MCP](https://github.com/ahujasid/blender-mcp) | Python MCP server ↔ Blender addon TCP socket `localhost:9876`, JSON `{type, params}`; addon marshals onto main thread via `bpy.app.timers` | ~31 tools: `get_scene_info`, `get_object_info`, `get_viewport_screenshot`, `execute_blender_code`, `describe_node_type`, asset search/download, `poll_*_job_status`; 1 prompt | Snapshot tool + screenshot + code escape hatch (safe mode env var) |
| [Figma official Dev Mode MCP](https://developers.figma.com/docs/figma-mcp-server/tools-and-prompts/) | Remote `mcp.figma.com/mcp` or desktop-local `127.0.0.1:3845/mcp` | Read: `get_design_context`, `get_metadata` (sparse outline), `get_screenshot`, `get_variable_defs`; Write: `use_figma` (coarse, skill-gated), `generate_figma_design`, `generate_diagram` | Layered read: cheap outline first, heavy code/image second |
| [cursor-talk-to-figma](https://github.com/grab/cursor-talk-to-figma-mcp) | MCP ↔ WebSocket relay ↔ Figma plugin | ~45 granular tools incl. batch variants (`set_multiple_text_contents`, `delete_multiple_nodes`) | "read overview → verify selection → batch" workflow |
| [Fusion 360](https://github.com/frankhommers/autodesk-fusion-mcp) | MCP server inside Fusion add-in over Streamable HTTP; work-queue to main thread | 13: `call_autodesk_api` (generic), `execute_python`, `capture_viewport`, `get_active_selection`, `fetch_api_documentation`, `save/load/list_script` | Few focused tools + generic API + persisted script library |
| [FreeCAD](https://github.com/neka-nat/freecad-mcp) | XML-RPC addon on `127.0.0.1:9875`, `QTimer` queue | 11: `create_document`, `create/edit/delete_object`, `get_objects`, `execute_code`, `get_view` (PNG), `run_fem_analysis` | `--only-text-feedback` toggle; analysis as a tool |
| [KiCad](https://github.com/Seeed-Studio/kicad-mcp-server) | No live app: parses files, `kicad-cli` for ERC/DRC | `list_schematic_components`, `trace_netlist_connection`, `run_erc`, `run_drc`, `add_component_from_library`, `export_gerber` | File-based, CI-friendly; **validation tools first-class** |
| Unity ([CoplayDev](https://github.com/CoplayDev/unity-mcp), [IvanMurzak](https://github.com/IvanMurzak/Unity-MCP)) | In-editor plugin ↔ server; main-thread dispatch | `manage_scene/gameobject/script/asset` (+`action`), `read_console`, `run_tests`, `batch_execute`; reflection escape hatch | Tool groups toggled per client to stay under limits |
| [Godot](https://github.com/Coding-Solo/godot-mcp) | Headless `godot --headless --script ... <op> <json>` per call | `run_project`, `get_debug_output`, `create_scene`, `add_node`, `save_scene` | Pure headless approach; simplest to ship |
| Excalidraw ([yctimlin](https://github.com/yctimlin/mcp_excalidraw); [official](https://github.com/excalidraw/excalidraw-mcp)) | stdio MCP ↔ Express canvas server + WebSocket sync, auto-spawned; official uses MCP Apps | 26: `create/get/update/delete_element`, `batch_create_elements`, `apply` ({create,update,delete} patch), `describe_scene`, `get_canvas_screenshot`, `snapshot_scene/restore_snapshot`, `export/import_scene`, `get_resource` (schema) | **Closest analogue**: JSON doc + live canvas; deterministic IDs and byte-stable export; verify-by-describe loop |
| [tldraw](https://github.com/AndresMuelas2004/tldraw-mcp-server) | Headless in-memory store ↔ `.tldr` files | 30: `new/load/save_document`, create shapes, `bind_arrow`, `list_shapes`, `get_shape` | Pure file model; document = load/save handle |
| draw.io ([official](https://github.com/jgraph/drawio-mcp); [lgazo](https://github.com/lgazo/drawio-mcp-server)) | Official: remote, stateless, full XML in/out + MCP Apps preview; lgazo: stdio ↔ bridge ↔ browser extension | Official: `create_diagram`, `search_shapes`, `open_drawio_xml`; lgazo: granular cell ops, `import-mermaid` | Two ends: stateless whole-document vs live-attached granular |
| OpenSCAD/CadQuery ([petrijr](https://github.com/petrijr/openscad-mcp), [cadquery](https://github.com/rishigundakaram/cadquery-mcp-server)) | Headless CLI; "code in, STL/PNG out" | validate/render/export | Code-as-document works when the app *is* a language |

### Lessons that generalize

1. Live-app integrations are "MCP server ⇄ local IPC ⇄ in-app agent on the UI thread". Headless file/library servers (KiCad, tldraw, Godot) are simpler and CI-friendly; the best (Excalidraw) auto-spawn the UI and sync via WebSocket.
2. A cheap **snapshot/overview tool** (`get_scene_info`, `describe_scene`, `get_metadata`) is what the model calls first; a **screenshot** tool is the second most common verification aid.
3. Mature servers converge on **few, coarse, well-described tools** plus explicit **batch/patch** variants and one **scripting escape hatch** behind a safe mode. Anthropic's guidance: consolidate, namespace, return semantic IDs, offer `response_format: concise|detailed`, paginate ([writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)).
4. **Validation as a first-class tool** (ERC/DRC) and **checkpoint/restore** let the model iterate safely.
5. Almost nobody exposes the document as an MCP **resource**. That is a gap to exploit (resources are `@`-mentionable in Claude Code and cache-friendly).
6. Stateless whole-document in/out is trivially correct but wastes tokens; handle-based state with explicit `design_id` is what the 2026 spec recommends.

## 3. Recommended architecture

**Shared core, three thin heads.**

```
packages/
  core/   @tensorcad/core  — zod schema of the graph doc (versioned), block catalog,
                          validate(doc) → report, analyze(doc) → {shapes, params, flops, memory},
                          codegen(doc, target) → files, apply(doc, ops) → doc'
                          (pure, deterministic, no I/O; also the UI's model layer)
  ui/     React + React Flow; imports core; optional "agent bridge" WebSocket server
          (127.0.0.1, random port + token written to ~/.tensorcad/session.json)
  cli/    tensorcad validate|analyze|codegen|serve-mcp  (thin wrapper over core)
  mcp/    @tensorcad/mcp — McpServer over stdio (default) + Streamable HTTP (opt-in);
          DocumentStore interface: FileStore (headless) and LiveStore (proxies to running UI over WS)
```

Design principles:

- **Explicit handles, not sessions.** `open_design`/`new_design` return `design_id` (+ `revision`); mutating tools take `design_id` and optional `expected_revision` for optimistic concurrency with the human editing the same canvas. Keep an op log for undo/`restore_checkpoint`.
- **One patch tool, not thirty setters.** `apply_ops(design_id, ops[])` where `ops` is the same discriminated union the UI's undo stack uses (`add_block`, `connect`, `set_param`, `remove`, `move`, `group`). Keeps under Cursor's tool cap; matches Excalidraw's `apply`.
- **Every read tool declares `outputSchema`** and returns `structuredContent` (+ text mirror).
- **Annotations everywhere**: reads `readOnlyHint: true, idempotentHint: true`; `apply_ops` `idempotentHint: false`; delete/overwrite `destructiveHint: true`.
- **Live UI optional**: no session file → pure file mode; session file present → attach, and the UI receives the same op stream.

### Initial tool list (15 tools)

| Tool | Purpose |
|---|---|
| `tensorcad_list_designs` | Enumerate open/recent designs (headless: files under cwd) |
| `tensorcad_new_design` | Create design from template (`gpt2-small`, `llama-style`, `empty`) → `design_id` |
| `tensorcad_open_design` | Load `.tensorcad.json` (or attach to UI's active doc) → `design_id`, revision |
| `tensorcad_get_design` | Full JSON, or `format: "outline"` for a compact block/edge summary |
| `tensorcad_get_block` | One block with params, inferred in/out shapes, param count |
| `tensorcad_search_catalog` | Query block catalog with param schemas |
| `tensorcad_apply_ops` | Batched graph edits; returns new revision + validation summary |
| `tensorcad_validate` | Shape/type/cycle checks → structured report with fix hints |
| `tensorcad_analyze` | Params, FLOPs/token, memory for given batch/seq/dtype; per-block breakdown |
| `tensorcad_generate_code` | PyTorch module + config; returns files as embedded resources or writes to path |
| `tensorcad_render_preview` | PNG of canvas (via UI if attached, else headless SVG) |
| `tensorcad_checkpoint` / `tensorcad_restore` | Named snapshots for safe experimentation |
| `tensorcad_save_design` | Persist (deterministic key order, stable IDs) |
| `tensorcad_run_script` | Escape hatch: sandboxed JS against the core API; gated by `TENSORCAD_ALLOW_SCRIPTS` |

### Resources

- `tensorcad://designs/{design_id}` — current JSON, `lastModified`; subscribable so the client learns when the human edits the canvas.
- `tensorcad://designs/{design_id}/validation` and `/analysis` — latest reports.
- `tensorcad://catalog` and `tensorcad://catalog/{block_type}` — block catalog with param schemas.
- `tensorcad://schema/design` — JSON Schema of the document format.
- `skill://tensorcad-design/SKILL.md` via Skills extension — design guide loaded on demand.

### Prompts

`design_model(target_params, context_len, family)`, `review_design(design_id)`, `scale_design(design_id, factor)`, `explain_costs(design_id, batch, seq)`.

### Spec features to lean on

- `structuredContent` + `outputSchema`; `resource_link`s to generated code instead of dumping files into context.
- `subscriptions/listen` with `resourceSubscriptions` on the design URI.
- Elicitation (form mode) for confirmations like "overwrite existing file?".
- Tasks extension for long runs (smoke-train of generated code); Claude Code auto-backgrounds >2-minute calls anyway.
- MCP Apps: canvas preview as `ui://` HTML for Claude Desktop/Cursor, text fallback for Claude Code.
- Packaging: npm publish → `mcpName` + `server.json`; `.mcpb` for Claude Desktop; `.mcp.json` + Cursor deeplink in README; bind HTTP to 127.0.0.1 with Origin validation.

## Unverified

- Exact TS v2 patch versions beyond 2.0.0; some release-page wording still says "beta".
- Cursor's 40-tool cap is from a third-party site.
- Tasks extension has no host in the official client matrix yet.
