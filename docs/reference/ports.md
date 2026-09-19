# Ports

A port is a pin on a block: one end of a wire, with a declared shape and a
handful of facts about what may be attached to it.

Declared in a catalog entry's `ports`, as either a bare shape pattern or an
object. The two are equivalent; a block with nothing more to say about a port
writes the string.

```ts
ports: {
  in:  { x: "... d_model" },                        // shape only
  out: { y: { shape: "... d_model", doc: "…" } },   // with more to say
}
```

## Fields

### `shape` — required

The shape pattern, as space-separated atoms. `B` and `T` are reserved runtime
symbols and stay indeterminate; `...` matches any leading dimensions; parentheses
group a product, so `(H dh)` is one axis of size `H × dh`. Parameter names of the
owning block resolve against its resolved parameters, so `"... d_model"` means
whatever that instance's `d_model` evaluates to.

### `dtype`

One of `float`, `half`, `fp8`, `int`, `bool`, `inherit`. Defaults to `inherit`,
meaning the tensor's type comes from whatever produced it.

Declare it when the port produces or demands something specific. It is what lets
an integer tensor arriving at a float matmul be refused: **shape inference cannot
catch that**, because the shapes agree perfectly. It also decides how the wire is
drawn — an index net is dotted.

### `optional`

`true` when the port may legitimately be left unwired. Defaults to `false`.

An unwired required port is a warning and draws the hollow circle eeschema uses
for a dangling end. An unwired optional port is neither.

### `whenUnconnected`

What a consumer should assume when nothing is attached: `"zero"`, `"identity"`,
`"causal"`, or `{ tensor: "<name>" }`. Only meaningful on an `optional` port.

### `anchor`

`"flow"` (default) or `"side"`.

`flow` follows the reading direction: a target below its source leaves the bottom
and arrives at the top; blocks placed beside each other connect across. `side`
always leaves sideways, whatever the geometry.

Three ports declare `side`, and they mean something by it:

| Port | Why |
|---|---|
| `add:b` | The residual bypass. Entering from above it would read as the main path rather than the one that skips it. |
| `mul:b` | The gate in a gated feed-forward — the same argument. |
| `rope:y` | An accessory hanging off the line, not a stage on it. |

This used to be a lookup table in the renderer keyed on `"type:port"` strings.
Two of its five entries matched no catalog type at all, which is the failure mode
a table like that has: nothing checks it. Being a field on the port, it is now
checked by the same thing that checks everything else.

### `showName`

Draw the port's name beside the pin. Off by default; a block with one input and
one output does not need to label them.

### `doc`

One line, for the inspector and the MCP catalog.

## How it resolves

Consumers never see the raw declaration. `portsOf(def.ports, resolved)` returns
`ResolvedPorts`, with every default filled in:

```ts
interface ResolvedPort {
  shape: string;
  dtype: "float" | "half" | "fp8" | "int" | "bool" | "inherit";
  optional: boolean;
  anchor: "flow" | "side";
  whenUnconnected?: …;
  showName?: boolean;
  doc?: string;
}
```

Normalising at the single resolver rather than at each call site is deliberate:
a consumer reaching for the raw declaration would find a bare string on most
primitives and an object on the rest, and would get one of the two wrong.

`derived.infer.ports` carries the same resolved form for every node in a
document, which is what the canvas reads to decide a wire's side and kind.

## Boundary nodes

A container's ports come from the `boundary_in` / `boundary_out` nodes inside its
subgraph, which carry **shapes only**. The rest of a port's declaration is about
the outside of a block; inside its expansion there is only a tensor arriving at a
shape.

A block that a document defines for itself generates its boundary nodes from its
declared ports, so the two cannot disagree. See
[Define a block inside a document](../how-to/define-a-block-in-a-document.md).
