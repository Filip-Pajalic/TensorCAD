# Add a block to the catalog

Before you start, check that it needs to be one. The test is in
`catalog/types.ts`: **a block is a primitive only if it carries a formula that
cannot be expressed as an arrangement of existing primitives.**

Most things people want — MQA, QK-norm, shared experts, NoPE, attention sinks —
are an arrangement or a parameter, and belong in a composite or in
[a document's own definitions](define-a-block-in-a-document.md). That is the
system working, not a limitation.

## A composite

A subgraph of primitives, expanded by the analysis. It needs no formulas: the
primitives underneath already have them.

```ts
{
  kind: "composite",
  type: "my_attention",
  category: "attention",
  params: { d_model: { type: "int", min: 1 }, heads: { type: "int", min: 1 } },
  ports: { in: { x: "... d_model" }, out: { y: "... d_model" } },
  expand: (raw, r) => ({ nodes: [...], edges: [...] }),
  docs: { summary: "…", formula: "…", refs: ["…"] },
}
```

If you find yourself writing new arithmetic for a composite, stop — that is the
sign a primitive underneath is missing.

## A primitive

Four formulas, all of them per-instance and per-token:

```ts
paramCount: (r) => …,                      // trainable parameters
flops:      (r, ctx) => ({ fwd, elementwise }),  // forward, per token
retains:    (r) => ["x"],                  // inputs kept alive for backward
stateBytes: (r, ctx) => …,                 // optimiser/cache state, if any
```

`flops.fwd` counts multiply-accumulates as 2 and includes only matmuls.
Elementwise work goes in `elementwise`, separately, because the 6N convention
excludes it and because those operations are memory-bound.

`retains` names the **input ports** whose tensors must survive to the backward
pass. Memory is attributed to the tensor, not to the consumer, so a tensor read
by three blocks is counted once.

## Ports

See [the ports reference](../reference/ports.md). Declare `dtype` when the port
produces something other than activations, and `anchor: "side"` when the pin
means something by leaving sideways.

## Documentation is not optional

```ts
docs: {
  summary: "One sentence saying what it is.",
  formula: "params = …; FLOPs/token = …",
  refs: ["https://…"],
}
```

Someone will want to know where the arithmetic came from, and that someone is
usually you in six months.

## Code generation

Add a `case` to the switch in `packages/core/src/codegen/torch.ts`. If the block
cannot be emitted, say so in a warning rather than emitting something wrong.

## Prove it

Either a preset that uses it with a published figure, or a test pinning the
arithmetic. Preferably both.

```bash
bun run scripts/report.ts     # before and after; the table must not move
bun test packages
```
