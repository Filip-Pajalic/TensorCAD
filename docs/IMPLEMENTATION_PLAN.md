# TensorCAD — Implementation plan

Companion to [../ROADMAP.md](../ROADMAP.md) and [FEATURES.md](FEATURES.md). This document fixes the technical decisions and breaks M0–M3 into tasks.

**This plan has been implemented through M2.** Sections 3 and 4 below were updated to describe the code as built; the rest describes the intent and remains accurate. Three decisions changed during implementation and are recorded in section 16.

## 1. Decisions

| # | Decision | Choice | Why | Alternatives considered |
|---|---|---|---|---|
| D1 | Node editor | `@xyflow/react` 12.x + `elkjs` (worker) | Typed handles map to shape checks; sub-flows give the ×N container; MIT; React 19; largest ecosystem | Rete.js (more assembly), Comfy LiteGraph fork (API churn, GPL monorepo) |
| D2 | Document format | Own JSON IR, versioned; UI positions in a separate `ui` map | Git-friendly, editor-independent, MCP-friendly (ComfyUI lesson) | Persisting React Flow state directly |
| D3 | Where the math lives | Pure TypeScript `@tensorcad/core` | One engine for browser (offline), CLI, MCP; instant re-analysis on edit | Python core with a server (slower feedback, harder to ship) |
| D4 | Analysis granularity | Formulas on primitives; composites are subgraphs | New blocks need no new math; explain tab falls out naturally | Per-block hand formulas (faster PoC, dead end) |
| D5 | Symbolic shapes | Small in-house polynomial canonicalizer over named symbols | ~300 lines; no CAS dependency | Algebrite/math.js (heavy), concrete-only (loses "why") |
| D6 | Codegen | Composite-first templates emitting idiomatic PyTorch, primitive fallback | Readable output people will actually train | torch.fx graph codegen (unreadable), HF modular (later) |
| D7 | Verification/training | Python package `tensorcad-runtime` (torch), driven over subprocess with JSON | Only Python can instantiate/export/train; keep it thin | Pyodide (no CUDA) |
| D8 | MCP | TypeScript SDK v2 (`@modelcontextprotocol/server`), stdio first, file-mode store, live bridge later | Same language as core; spec 2026-07-28 handle-based state | Python MCP (would duplicate core) |
| D9 | Package manager / runtime | Bun workspaces for dev; Node 22-compatible output for published packages | Bun is installed, fast; users run `npx @tensorcad/mcp` on Node | pnpm (not installed), npm workspaces (slower) |
| D10 | Python env | `uv` + venv, Python 3.13, PyTorch CUDA 12.8+ wheels (RTX 5080 is Blackwell sm_120) | Reproducible; Blackwell needs recent wheels | conda |

## 2. Repository layout

```
TensorCAD/
  package.json               # bun workspaces: packages/*
  tsconfig.base.json
  packages/
    core/                    # @tensorcad/core — pure, no DOM, no I/O
      src/
        ir/                  # schema.ts (zod), types.ts, migrate.ts, ops.ts (edit ops + apply)
        shapes/              # symexpr.ts, unify.ts, infer.ts
        catalog/             # primitives/*.ts, composites/*.ts, index.ts, presets/*.json
        rules/               # engine.ts, rules/*.ts
        analysis/            # params.ts, flops.ts, kvcache.ts, memory.ts, cost.ts, hardware.ts, index.ts
        codegen/             # torch/{emit.ts, templates/*.ts}, litgpt.ts, hf.ts
        import/              # hf-config.ts, litgpt.ts
        diff.ts
      test/                  # vitest: regression table, golden codegen, property tests
    ui/                      # Vite + React 19 + @xyflow/react + zustand + Tailwind
      src/
        app/                 # layout, routes
        canvas/              # ReactFlow wrapper, node components, edge components, layout worker
        panels/              # Palette, Inspector, Symbols, Rules, Analysis, Verify, Runs
        state/               # doc store (IR + undo), view mapping IR<->ReactFlow, selection
        bridge/              # WebSocket server for MCP live mode (dev server only)
    cli/                     # tensorcad <cmd> — commander, calls core + runtime
    mcp/                     # @tensorcad/mcp — McpServer, DocumentStore (file | live), tools, resources, prompts
  python/
    tensorcad_runtime/          # pyproject (uv); verify.py, smoke_train.py, data/, cli.py
  docs/
  examples/                  # *.tensorcad.json presets are also copied into core/catalog/presets
```

## 3. Core IR

### 3.1 Document

```jsonc
{
  "version": 1,
  "meta": {
    "name": "llama-3-8b",
    "family": "llama",
    "published": { "params": 8030261248, "source": "https://huggingface.co/..." }
  },
  "symbols": {
    "B": { "kind": "runtime", "default": 1 },
    "T": { "kind": "runtime", "default": 8192 },
    "L": { "kind": "design", "value": 32 },
    "D": { "kind": "design", "value": 4096 },
    "H": { "kind": "design", "value": 32 },
    "Hkv": { "kind": "design", "value": 8 },
    "dh": { "kind": "design", "value": 128 },
    "F": { "kind": "design", "value": "ceil_mult(1.3*8/3*D, 1024)" },
    "V": { "kind": "design", "value": 128256 }
  },
  "graph": {
    "nodes": [
      { "id": "tokens", "type": "input", "params": { "shape": "B T", "dtype": "int64" } },
      { "id": "embed",  "type": "embedding", "params": { "vocab": "V", "dim": "D" } },
      { "id": "layers", "type": "repeat", "label": "Transformer block x32",
        "params": { "count": "L" },
        "graph": {
          "nodes": [
            { "id": "_in",  "type": "boundary_in",  "params": { "ports": { "x": "B T D" } } },
            { "id": "block", "type": "transformer_block", "params": {
                "d_model": "D", "heads": "H", "kv_heads": "Hkv", "head_dim": "dh",
                "ffn_hidden": "F", "norm": "rmsnorm", "mlp": "gated", "act": "silu",
                "rope": { "theta": 500000 } } },
            { "id": "_out", "type": "boundary_out", "params": { "ports": { "x": "B T D" } } }
          ],
          "edges": [["_in:x", "block:x"], ["block:y", "_out:x"]]
        } },
      { "id": "final_norm", "type": "rmsnorm", "params": { "dim": "D" } },
      { "id": "head",   "type": "lm_head", "params": { "vocab": "V", "dim": "D", "tied": false } },
      { "id": "logits", "type": "output" }
    ],
    "edges": [
      ["tokens:x", "embed:ids"], ["embed:y", "layers:x"], ["layers:x", "final_norm:x"],
      ["final_norm:y", "head:x"], ["head:y", "logits:x"]
    ]
  },
  "ui": { "positions": {} }
}
```

Conventions as built:
- Nodes and edges live under `graph`, not at the top level, so a subgraph and the document share one type.
- Edges are `"node:port"` string pairs, local to the graph they appear in. Nested nodes are addressed by path (`layers/block/attn/q_proj`) in analysis results.
- Any numeric parameter may be a number or an expression over symbols (`"8/3*D"`, `"ceil_mult(1.3*8/3*D, 1024)"`). Symbols may themselves be expressions and are evaluated in dependency order with cycle detection.
- `B` and `T` are reserved runtime symbols. They stay indeterminate through shape checking, so a mismatch is a real polynomial difference rather than a coincidence of numbers.
- Heterogeneous stacks are expressed by putting several blocks in one repeat's subgraph, or by chaining repeats. There is no separate pattern language.
- Stable ids; deterministic key order on save; `ui` is the only section the semantic diff ignores.

### 3.2 Types (core/src/ir/types.ts, sketch)

```ts
type SymValue = number | { kind: "runtime"; default?: number };
type ParamValue = number | string | boolean | Record<string, unknown> | null;
interface Node { id: string; type: string; params?: Record<string, ParamValue>; graph?: Graph; variants?: Record<string, Graph>; }
interface Graph { nodes: Node[]; edges: [string, string][]; }
interface Doc { version: 1; meta: Meta; symbols: Record<string, SymValue>; nodes: Node[]; edges: [string, string][]; ui?: UiState; }

type Op =
  | { op: "add_node"; parent?: string; node: Node }
  | { op: "remove_node"; path: string }
  | { op: "set_param"; path: string; key: string; value: ParamValue }
  | { op: "connect"; from: string; to: string } | { op: "disconnect"; from: string; to: string }
  | { op: "set_symbol"; name: string; value: SymValue }
  | { op: "rename"; path: string; id: string }
  | { op: "move"; path: string; xy: [number, number] };
function apply(doc: Doc, ops: Op[]): { doc: Doc; inverse: Op[] };   // used by UI undo and MCP apply_ops
```

## 4. Catalog

### 4.1 Primitive definition

```ts
interface PrimitiveDef {
  type: string;
  params: Record<string, ParamSpec>;                 // { in: "int", bias: { type: "bool", default: false } }
  ports: { in: Record<string, string>; out: Record<string, string> };  // einops-style patterns: "B T D", "B H T dh", "... in"
  constraints?: (p: Resolved) => Constraint[];       // e.g. H % Hkv == 0
  params_count?: (p: Resolved) => number;
  flops_per_token?: (p: Resolved, ctx: Ctx) => { fwd: number; bwd: number };
  activation_bytes?: (p: Resolved, ctx: Ctx) => number;   // saved for backward, per token
  state_bytes?: (p: Resolved, ctx: Ctx) => { per_token: number; per_seq: number };  // KV cache / SSM state
  codegen: { torch: TorchTemplate };
  docs: { summary: string; formula: string; refs: string[] };
}
```

Examples of the formulas each primitive carries (from [research/02-analysis-math.md](research/02-analysis-math.md)):

| Primitive | params | flops/token fwd | activation bytes/token | state |
|---|---|---|---|---|
| `linear(in,out,bias)` | `in·out (+out)` | `2·in·out` | `in·bytes` (input saved) | – |
| `embedding(V,D)` | `V·D` | 0 (gather) | ids only | – |
| `rmsnorm(D)` | `D` | ~`4·D` (memory-bound, reported separately) | `D·bytes` | – |
| `activation` | 0 | ~`k·D` | `D·bytes` | – |
| `add`/`mul` | 0 | `D` | 0 / `2·D·bytes` | – |
| `rope` | 0 | ~`6·H·dh` | 0 | – |
| `sdpa(H,Hkv,dh,causal,window,flash)` | 0 | `4·T_eff·H·dh` (`T_eff = min(T,W)`, ×½ if causal skip) | retains q,k,v; keeps `o + lse` itself | KV: `2·Hkv·dh·bytes`/token (or `W` fixed) |
| `topk_router(D,E,k,bias)` | `D·E (+E)` | `2·D·E` | small | – |
| `ssd_scan(d_inner,N_s,P,G)` | `3·nheads + norm` | ≈`6·d_inner·N_s + 4·d_inner·Q` | ~`d_inner·bytes·c` | per seq: `nheads·P·N_s·bytes + conv state` |
| `lm_head(V,D,tie)` | `V·D` or 0 | `2·V·D` | logits `V·(2+4)` bytes | – |

### 4.2 Composite definition

A composite is a `Graph` template plus parameter mapping and an optional idiomatic codegen template:

```ts
interface CompositeDef {
  type: "gqa_attention";
  params: { d_model: "int"; heads: "int"; kv_heads: "int"; head_dim: "int"; rope: RopeSpec | null; causal: bool; bias: bool; qk_norm: bool };
  ports: { in: { x: "B T d_model" }; out: { y: "B T d_model" } };
  expand: (p) => Graph;       // q/k/v linear -> rearrange -> rope -> sdpa -> rearrange -> o linear
  codegen?: { torch: TorchTemplate };   // class GQAAttention(nn.Module) ...; falls back to expand() if absent
}
```

Analysis always runs on `expand(p)`; the UI shows the composite collapsed with the sum. Golden test: `params(expand(gqa))` equals the closed-form `D·H·dh + 2·D·Hkv·dh + H·dh·D`.

### 4.3 Presets

`core/src/catalog/presets/*.tensorcad.json`, one per reference model, with `meta.published = { params: 8.03e9, source: url }`. The test suite asserts `|calc − published| / published < 0.5%` (exact for models whose config is fully specified).

## 5. Symbolic shape engine (core/src/shapes)

- `SymExpr`: map from monomial (sorted symbol×power list) to rational coefficient; ops `add`, `mul`, `divExact` (returns obligation `a % b == 0` when not provable), `eq`, `toString`.
- `parsePattern("B T (H dh)") → Dim[]`, with `...` for leading batch dims.
- `unify(producer: Dim[], consumerPattern: Dim[], env) → { ok, bindings, obligations, diff }`.
- `inferShapes(doc) → Map<edgeId, Shape> & errors`: topological walk; `repeat` checks `boundary_in` ≡ `boundary_out` shapes (residual stream invariant) and iterates variants for patterns.
- Runtime symbols (`B`, `T`) stay symbolic in labels and get concrete values only in analysis.

Tests: property tests (random polynomials round-trip through `toString`/parse), unification fixtures, full-preset inference snapshots.

## 6. Rule engine (core/src/rules)

```ts
interface Rule { id: string; severity: "error" | "warning" | "info"; run(ctx: RuleCtx): Finding[]; }
interface Finding { rule: string; severity; path?: string; message: string; fix?: Op[]; }
```

Initial rules (M1): `dangling-port`, `cycle`, `unreachable`, `shape-mismatch` (from inference), `residual-dim-constant`, `heads-divide` (`D = H·dh`, `H % Hkv`), `rope-even-head-dim`, `flash-head-dim` (∈ {64,128,256}), `tensor-core-multiples` (D, F, V), `vocab-padding`, `window-le-context`, `kv-fits-gpu`, `train-memory-fits`, `activation-suggest-recompute`, `chinchilla-ratio`, `moe-topk-le-experts`, `tied-head-dims`, `unused-symbol`.

Rules receive the inferred shapes, the analysis result, and the hardware profile, so capacity rules are cheap.

## 7. Analysis engine (core/src/analysis)

```ts
interface AnalysisInput { doc: Doc; T: number; B: number; dtype: "bf16" | "fp8" | "fp32"; hardware: HardwareProfile;
  parallel: { dp: number; fsdp: boolean; zero: 0|1|2|3; tp: number; pp: number; ep: number };
  optimizer: "adamw" | "adamw8bit" | "muon" | "sgd"; recompute: "none" | "selective" | "full"; flash: boolean;
  tokens?: number; mfu?: number; concurrency?: number }
interface AnalysisResult { params: {...}; flops: {...}; kv: {...}; memory: { train: {...}; infer: {...} };
  throughput: {...}; cost: {...}; chinchilla: {...}; perBlock: Record<path, BlockStats>; formulas: Record<path, string> }
```

Implementation order: `params` → `flops` → `kv` → `memory` → `cost/throughput` → `chinchilla`. Each module is a pure function over the expanded primitive graph and returns the formula strings used, so the UI can show them. Hardware profiles live in `hardware.ts` as data (peak bf16/fp8 TFLOPS, HBM GB, bandwidth GB/s, $/h); RTX 5080 numbers must be filled from the spec sheet and marked approximate.

## 8. Code generation (core/src/codegen/torch)

- Emitter walks the top-level graph, emits one `nn.Module` class per composite type used (deduplicated), a `Block` class per `repeat` variant, and a `Model` class with `forward(ids)`. Residual wiring comes from the graph, not hard-coded.
- Templates are TypeScript tagged-template functions (no runtime template engine); output is run through a formatter-free but consistent indenter; golden files in `core/test/golden/*.py`.
- Attention: `F.scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)`; sliding window via `flex_attention` block mask when enabled.
- SSM blocks: `from fla.layers import GatedDeltaNet` / `from mamba_ssm.modules.mamba2 import Mamba2` behind try/except with a clear error.
- Also emits `config.json` (our IR) next to the model and a `train.py` (nanoGPT-style) when requested.
- `litgpt.ts` and `hf.ts` map a subset of designs to those configs and refuse (with reasons) otherwise.

## 9. Python runtime (python/tensorcad_runtime)

- `tensorcad-runtime verify <model.py> --B 2 --T 128` → JSON `{ params, param_table, shapes, flops, warnings }` using `torch.device("meta")`, `torch.export.export` with `Dim("B")`, `Dim("T")`, and `FlopCounterMode` on fake tensors.
- `tensorcad-runtime smoke-train <dir> --tokens 100M --minutes 10` → writes `runs/<hash>.jsonl` (step, loss, tok/s, mem) using a fixed FineWeb-Edu shard; bf16, AdamW, cosine schedule, `torch.compile` optional.
- `tensorcad-runtime data prepare --dataset fineweb-edu-sample --tokens 100M` → cached `.bin` tokens.
- Install: `uv venv && uv pip install -e .[cuda]`; PyTorch from the CUDA 12.8+ index for Blackwell (verify exact index at install time).
- The CLI/UI call it via `child_process` with JSON on stdout and progress on stderr.

## 10. UI (packages/ui)

- State: zustand store holding the `Doc`, an undo stack of `Op[]`, selection, and derived caches (`shapes`, `analysis`, `findings`) recomputed with a debounced worker.
- View mapping: `docToFlow(doc, ui) → { nodes, edges }`; React Flow node types: `primitive`, `composite`, `repeat` (group with breadcrumb), `boundary`. Parents precede children in the array (React Flow requirement).
- Node component: title, param summary, shape badges on handles, error/warning badge, "expand" toggle for composites, param count chip.
- Panels: Palette (search, drag), Inspector (params with expression input + symbol autocomplete, docs/explain tabs), Symbols, Rules (findings list, click-to-focus, fix-it button), Analysis (tables + per-block treemap), Verify (M2), Runs (M4).
- Layout: `elkjs` in a web worker with `layered` algorithm; manual positions win once a node has been dragged.
- Dev bridge (M3): Vite dev server starts a WebSocket server on `127.0.0.1:<random>` and writes `~/.tensorcad/session.json` with port + token.

## 11. MCP server (packages/mcp)

- `McpServer` from `@modelcontextprotocol/server`, `serveStdio`; optional Streamable HTTP via `/hono` bound to 127.0.0.1 with Origin validation.
- `DocumentStore` interface: `list()`, `open(path|id)`, `get(id)`, `apply(id, ops, expectedRevision?)`, `save(id)`, `checkpoint/restore`; `FileStore` (headless) and `LiveStore` (proxies to UI over WS, falls back to FileStore).
- Tools (namespaced `tensorcad_*`), each with zod `inputSchema`/`outputSchema`, `structuredContent` + text mirror, annotations (`readOnlyHint`, `idempotentHint`, `destructiveHint`). Tool list is deterministic. See [research/03-mcp.md §3](research/03-mcp.md).
- Resources with templates (`tensorcad://designs/{id}`, `/validation`, `/analysis`, `tensorcad://catalog/{type}`, `tensorcad://schema/design`); `subscriptions/listen` support for design updates in live mode.
- Prompts: `design_model`, `review_design`, `scale_design`, `explain_costs`.
- Ship `.mcp.json` at repo root: `{"mcpServers":{"tensorcad":{"type":"stdio","command":"bun","args":["packages/mcp/src/index.ts"]}}}` (dev) and the `npx -y @tensorcad/mcp` form in README.

## 12. Testing strategy

- **Regression table**: one test per reference model asserting params (and active params) against published numbers; extended with KV/FLOPs assertions where a published figure exists (DeepSeek-V3 70 KB/token, Llama-3 MFU-derived FLOPs).
- **Property tests** for `SymExpr` and `unify`.
- **Golden codegen** files per preset; CI runs `python -c "import model"` plus `verify` on CPU meta device (no GPU needed).
- **Rule fixtures**: broken designs with expected findings.
- **MCP contract tests**: spin up the server over stdio with the SDK client, call each tool, validate `structuredContent` against `outputSchema`.
- **UI**: component tests for node/inspector; Playwright smoke (open preset, edit symbol, see count change) later.

## 13. Task breakdown

### M0 — Sketch

| # | Task | Notes | Est. |
|---|---|---|---|
| 0.1 | Bootstrap monorepo: bun workspaces, TS strict, vitest, eslint/prettier, `packages/core`, `packages/ui` (Vite React TS) | Node-compatible output (`tsup`) for core | 0.5 d |
| 0.2 | IR schema + types + `apply(ops)` with inverse ops; JSON round-trip with deterministic key order | zod 4 | 1 d |
| 0.3 | `SymExpr`, pattern parser, `unify`, `inferShapes` incl. `repeat` invariant | property tests | 1.5 d |
| 0.4 | Catalog: primitives (input, output, embedding, linear, rmsnorm, layernorm, activation, add, mul, rearrange, rope, sdpa, lm_head, boundary_in/out) with ports + `params_count` | docs strings from research | 1 d |
| 0.5 | Composites: `gqa_attention`, `gated_mlp`, `dense_mlp`, `transformer_block_prenorm` with `expand()` | closed-form golden tests | 1 d |
| 0.6 | `analysis/params.ts` over expanded graph; presets GPT-2 small, Llama-3-8B, Mistral-7B; regression tests | must match table | 0.5 d |
| 0.7 | UI shell: store, `docToFlow`, node components with shape badges, typed handles + `isValidConnection` from `unify` | React Flow 12 | 2 d |
| 0.8 | Palette (drag to add), Inspector (params/expressions), Symbols panel, live param count in header | | 1.5 d |
| 0.9 | `repeat` as sub-flow + breadcrumb; composite expand/collapse; elkjs auto-layout in worker | parents-before-children | 1.5 d |
| 0.10 | Save/load, undo/redo, keyboard shortcuts; preset menu | | 0.5 d |

Total ≈ 11 working days.

### M1 — Check

| # | Task | Est. |
|---|---|---|
| 1.1 | `flops.ts` (per primitive, attention term, causal skip, train multiplier, recompute) + tests vs cookbook | 1 d |
| 1.2 | `kvcache.ts` (GQA, window, later MLA/SSM hooks) | 0.5 d |
| 1.3 | `memory.ts` (weights by dtype, optimizer bytes/param, activations w/ Megatron + SwiGLU formulas, logits, ZeRO/FSDP/TP/PP) | 1.5 d |
| 1.4 | `cost.ts`, `throughput.ts` (roofline), `chinchilla.ts`, `hardware.ts` profiles | 1 d |
| 1.5 | Rule engine + initial rules + fix-its | 1.5 d |
| 1.6 | Analysis panel (tables, treemap, formula hover), Rules panel, what-if sliders | 2 d |
| 1.7 | `packages/cli`: `validate`, `analyze` (JSON/markdown output), CI workflow over presets | 0.5 d |

### M2 — Manufacture

| # | Task | Est. |
|---|---|---|
| 2.1 | Torch emitter + templates for M0 composites; golden files | 2 d |
| 2.2 | `python/tensorcad_runtime` `verify` (meta device, export, FlopCounterMode) + subprocess protocol | 1.5 d |
| 2.3 | Verify panel with diff table | 1 d |
| 2.4 | HF `config.json` importer (Llama/Mistral/Qwen/Gemma) + tests; litgpt export | 1.5 d |
| 2.5 | `train.py` + data prep template emission | 0.5 d |

### M3 — Agent

| # | Task | Est. |
|---|---|---|
| 3.1 | `packages/mcp`: server, FileStore, 15 tools with schemas/annotations, resources, prompts; contract tests | 2 d |
| 3.2 | `.mcp.json`, README install snippets (Claude Code, Cursor, Claude Desktop), `mcpName` | 0.5 d |
| 3.3 | UI bridge (WS server in dev, session file) + `LiveStore`; resource update notifications | 1.5 d |

## 14. First commands

```bash
cd C:/Git/TensorCAD
git init
bun init -y
mkdir -p packages/core packages/ui packages/cli packages/mcp python/tensorcad_runtime
```

```bash
cd packages/ui && bun create vite . --template react-ts && bun add @xyflow/react zustand elkjs
```

```bash
cd packages/core && bun add zod && bun add -d vitest tsup typescript
```

```bash
pip install uv
```

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| React Flow slows down with many DOM nodes | Always collapse `repeat` to one node; expand composites lazily; virtualize palette |
| Activation-memory formulas drift from real kernels | Label as estimates; verify against a profiler run in M4; keep Megatron formula as the documented baseline |
| Codegen becomes a second source of truth | Composites' `expand()` is the truth; templates are tested to match `expand()` param counts and torch verification |
| Python deps (fla, mamba_ssm) hard to install on Windows | Optional extras; CPU meta-device verification never needs them; document WSL2 path |
| Tool sprawl in MCP (Cursor tool cap) | One `apply_ops` patch tool; outline mode on `get_design`; keep ≤ 15 tools |
| Scope creep into a general DL editor | Non-goals in FEATURES.md; catalog PRs must include formulas, docs, and a preset or test |

## 16. Decisions that changed during implementation

**Activation memory is attributed to tensors, not to blocks.** The first model asked each block how many bytes it saved, which counted the residual stream three times because the query, key and value projections all read it. A block now declares which of its *input ports* must stay alive (`retains`) plus any buffer of its own (`extraActivationBytes`), and the analysis counts each producing tensor once. For Llama-3-8B this moved the estimate from 200 KB to 176 KB per token per layer. The remaining gap to the 147 KB in the research note is the two extra intermediate tensors an unfused SwiGLU keeps; a fused kernel reaches the lower figure.

**Containers carry two multipliers, not one.** A `repeat` stacks its subgraph, and the count drives both the parameter total and the per-token cost. A `moe_experts` container holds `experts` copies but a token passes through `top_k` of them. Making that a property of the container means total and active parameters, FLOPs and activation memory all fall out of one mechanism, and mixture-of-experts needed no special case in the analysis.

**No schema-validation dependency.** The core has zero runtime dependencies, so `bun test` runs with nothing installed. Parameter validation is type-directed from the catalog's own specs, which it had to be anyway: without the spec there is no way to tell the enum `"silu"` from the expression `"D"`. A schema library is still the right choice for the MCP server's tool definitions.
