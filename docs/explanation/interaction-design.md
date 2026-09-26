# UI critique (self-assessment before redesign)

Observed directly in the browser at 1600x950 on the llama-3-8b and deepseek-v3 presets.

## What is wrong

1. **Illegible at working zoom.** Nodes are about 120 px wide with 6-7 px type. The default
   view fits the whole graph, so nothing can be read without zooming. A schematic symbol
   should be legible at the zoom you actually work at.
2. **Bezier wires.** Curved noodles read as a dataflow toy. Schematic capture uses
   orthogonal segments with junction dots, which is how an engineer reads connectivity.
3. **No engineering grid and no snap.** There is a faint dot texture, not a grid with a
   stated pitch that positions snap to.
4. **No model tree.** Every CAD application has a hierarchy panel: FreeCAD's model tree,
   KiCad's hierarchy navigator, Fusion's browser. The breadcrumb alone is not navigation.
5. **No status bar.** No cursor position, grid pitch, zoom level, selection summary or
   live rule-violation count.
6. **Toolbar is a row of text buttons.** CAD toolbars are grouped icon bands with tooltips.
7. **No locking.** Nothing can be pinned in place; the user asked for this specifically.
8. **Ports are not typed visually.** Every handle looks the same regardless of what flows
   through it.
9. **Panels are tabs, so you cannot see the analysis and the inspector at once.** CAD tools
   dock several panels simultaneously.
10. **No cross-probing.** Clicking a rule violation should select and flash the offending
    item; the issues list is nearly inert.

## What is right and must survive the rework

- The drill-down model with a breadcrumb, and read-only composite interiors with a banner.
- Shape labels on wires, and the symbolic/numeric toggle.
- Live recomputation: editing a symbol updates every number immediately.
- The analysis panel's published-versus-computed comparison.
- The connection checker that refuses shape-incompatible wires.


## Research behind the rework

Read first-hand from KiCad's Eeschema manual and source, the FreeCAD wiki, Blender's
node-editor docs and source, Horizon EDA, LibreCAD and KiCanvas. The findings that
changed the design:

- **KiCad marks a locked item with a coloured shadow, not a padlock badge.** The exact
  value is `#FF26E2` at 50%, dilated from the item's own outline. The padlock glyph lives
  in menus and the confirmation dialog, never permanently on the canvas.
- **A lock must stop automatic edits, not just dragging.** KiCad drops locked items from
  the drag collector and asks before overriding; FreeCAD's Frozen flag excludes an object
  from recompute. A lock that only stops the mouse is a hint, not a lock.
- **Blender encodes the data type in the socket colour and the arity in the socket shape.**
  Circle for a single value, diamond for a field. The two are separable, which is also
  KiCad's rule: pin electrical type and pin graphic style are independent.
- **An unconnected pin is drawn hollow** and fills when wired. KiCad uses an open circle
  for an unconnected pin and a square for an unconnected wire end.
- **Silence is the wrong failure mode for a refused connection.** Blender turns the link
  red and names the reason at the link midpoint.
- **Dark is the right choice here.** The convention is not "CAD is light": KiCad's
  schematic canvas is `#F5F4EF` but its PCB canvas is `#001023`, Blender and Horizon are
  dark, and KiCanvas, KiCad's own web renderer, ships dark by default. Symbolic paper
  drawings are light; dense technical canvases are dark.
- **Severity should be three shapes, not three colours**, so it survives colour blindness
  and a greyscale screenshot. KiCad's pin-conflict matrix cycles green square, amber
  triangle, red circle.

## What was changed

| Defect | Fix |
|---|---|
| Curved wires | True right angles, using the step edge router |
| No grid | Minor and major grid at 16 and 80 units, with snap |
| Illegible blocks | Redrawn as schematic parts: designator, type, value row, parameter chip, category stripe |
| No model tree | Left dock listing the design with lock column, parameter counts, severity marks, click to cross-probe |
| No locking | Magenta halo on canvas, padlock in the tree, inherits into subgraphs, and auto-layout skips locked blocks and says how many |
| No status bar | Cursor position, grid pitch, zoom, level, selection, parameter count, live rule-check state |
| Untyped ports | Colour by element type, diamond when the size depends on batch or sequence length, hollow until connected |
| Silent refusals | The in-flight wire turns alert red and a chip names the reason |

## Still outstanding

- Junction dots at branch points, at KiCad's ratio of six times the stroke width.
- Cross-probing markers drawn on the canvas pointing at the offending port.
- A rule-severity settings pane with per-rule Error/Warning/Ignore, and persisted exclusions.
- Locks respected by `scaleDesign` and by edits arriving over MCP, which is where a lock
  matters most.
- Frames and groups, and a highlight-tensor mode that dims everything else.


## Second pass: the drawing-sheet rework

The first pass kept a dark canvas, on the argument that a dense technical canvas
is conventionally dark. That was the wrong call. Every published architecture
figure, and every one the user pointed at, is a *light drawing*: pastel fills,
thin dark outlines, black labels, and annotations on leader lines. My own
research had already recorded that KiCad's schematic canvas is `#F5F4EF` paper
and only its PCB canvas is dark; I read that and chose dark anyway.

### Changed

- **Light drawing sheet.** Canvas `#f7f6f2` with a two-level grid, neutral grey
  chrome around it so the drawing is the brightest thing on screen.
- **Parts, not cards.** A pastel body with a thin outline of the same hue, the
  name centred, the type beneath, and the defining numbers on a rule below that.
  The palette is taken from the published figures: green embeddings, salmon
  attention, yellow feed-forward, purple norms, blue containers.
- **Callouts on leader lines.** Derived from the design, so they cannot go
  stale: "Supported context length of 8,192 tokens", "Embedding dimension of
  4,096", "32 query heads, 8 key/value heads", "Intermediate projection size:
  4,096 = 14,336". This is the single feature that makes the canvas read like
  the figures it is imitating, and it is worth more than any amount of chrome.
- **`32x` written beside a stacked block**, as the figures mark a repeat.
- **A title block**, bottom-right, carrying the drawing name, parameter count,
  active parameters, layers, `d_model`, scale, check state, and the comparison
  against the published figure. This is the convention that most says
  "engineering drawing" and it costs almost nothing.
- **Port labels only where they disambiguate.** A part with one input and one
  output needs none, and the published figures label no pins at all.
- **Command bar grouped and captioned** by role, the way Fusion and AutoCAD
  group a ribbon: Drawing, File, Edit, View. View-state toggles read as pressed.

### Still outstanding

- Junction dots at branch points, at KiCad's ratio of six times the stroke width.
- Canvas markers pointing at an offending port, with cross-probing from the
  issues list.
- A rule-severity pane with per-rule Error/Warning/Ignore and persisted exclusions.
- Locks respected by `scaleDesign` and by edits arriving over MCP.
- Frames and groups, and a highlight-tensor mode that dims everything else.
- An export of the sheet to SVG, which is most of the way there now that the
  drawing is made of flat shapes on a light ground.

## Third pass: the engine was not plugged in

The two earlier passes were about how the drawing looked. This one is about what
the application was for. The core computes parameters, FLOPs, KV cache, training
and serving memory under a sharding plan, decode throughput against a roofline,
a Chinchilla token budget and a cost estimate, and it runs seventeen design
rules over all of it. The editor showed the parameter count and a list of shape
errors. Everything else was computed and thrown away: `derive()` called
`countParams` and `inferShapes` and never called `analyze` or `validate` at all.

That, not the colour scheme, is why the thing still felt unclear. It looked like
a diagram editor because that is all it was exposing.

### Research this pass

- **KiCad's DRC** is a docked list, sorted by severity, and every entry
  cross-probes: selecting it selects the offending item in the other editor and
  centres the view on it. The list is the interface to the check, not a report
  you generate and read once.
- **Tabs hide comparisons.** The Nielsen Norman guidance and the general
  dashboard advice agree: tabs are for alternative views of *one* thing, and
  dense readouts do better as expandable sections than as tabs, because a tab
  costs you the ability to see two things at once. Analysis behind a tab from
  the Inspector meant you could change a parameter or see what it did, never
  both.
- **Dark themes need desaturated accents and a measured contrast floor.** Pure
  black is avoided for eye strain and OLED smearing; saturated hues fail 4.5:1
  against dark surfaces and shimmer. The fix is a token set per theme and an
  actual measurement, not an eyeball.

### Changed

- **The whole engine is wired in.** `derive()` now calls `validate()`, which
  returns both the full analysis and every rule finding for one traversal. The
  heaviest preset costs ten milliseconds and the rest cost one, so it still runs
  on every edit.
- **An operating point.** Batch, sequence length, training and serving dtype,
  device, GPU count, optimizer, recomputation, ZeRO stage, tensor and pipeline
  parallelism, concurrent streams and token budget, all editable in place at the
  top of the readout. Every number below it depends on these, and leaving them
  as invisible defaults is how a readout becomes untrustworthy. Folded, the
  header still states them: `B1 · T8,192 · bf16 · H100 SXM · 8×`.
- **The right column splits instead of tabbing.** The readout is permanently on
  screen above a draggable divider; Inspector, Symbols, Rules, Cluster and
  Ladder share the pane below it. The readout answers "what does this design
  cost under the operating point I set"; the tabs are the things you change one
  at a time, including the two that answer the questions which come before it —
  what operating point to set, and what width to sweep at.
- **The readout says everything.** Parameters against the published figure, then
  compute against the 2N and 6N rules of thumb and against what a profiler would
  report, then a training memory budget drawn as a stacked bar with the device's
  capacity marked on it, then serving footprint, KV cache and decode throughput
  with what it is bound by, then the token budget and its cost. Each section
  folds and remembers.
- **A real DRC panel.** All seventeen rules, grouped by severity, filterable,
  each finding carrying its rule id, its hint and a click that cross-probes to
  the offending block. A rule book lists every rule and whether it fired, so the
  panel teaches the checks rather than only reporting them.
- **The palette folds away.** It is authoring furniture; the model tree is what
  you read. The tree now gets the left column and the palette opens when asked.
- **Both themes measured.** Every hard-coded colour in the stylesheet became a
  token, which is what actually broke dark mode: the drawing switched but the
  badges, chips, banners and title block never did. A contrast audit over all
  280 text elements now floors at 5.3:1 in dark and 4.5:1 in light, with the
  brand mark the one deliberate exception at 4.2:1, which clears the 3:1 that
  large text needs.

### Still outstanding

- Junction dots at branch points, at KiCad's ratio of six times the stroke width.
- Canvas markers pointing at an offending port, so a rule finding is visible on
  the drawing and not only in the list.
- Per-rule Error/Warning/Ignore with persisted exclusions. The severity filter
  is per-session and applies to all rules at once.
- Locks respected by `scaleDesign` and by edits arriving over MCP.
- Frames and groups, and a highlight-tensor mode that dims everything else.
- An SVG export of the sheet.
- The operating point is editor state rather than document state, so it is
  remembered per browser and not carried with a design that was tuned for a
  particular machine.

## Fourth pass: unfolding

Told the UX was still bad and to look at the references, I did — properly, at
the actual images rather than from memory. Raschka's architecture comparison
draws six models side by side, and the grammar is consistent across all of them:

- **Nothing is hidden behind a drill-down.** The repeated block is a rounded
  rectangle with the sublayers drawn inside it, and `16 ×` written on its edge
  with a bracket. You see the whole architecture at once.
- **The residual spine is the shape of the drawing.** Norm, attention, a circled
  plus, norm, feed-forward, a circled plus, with the bypass running up the side.
- **Almost every box is white.** One tint on the enclosing block, and exactly
  one box filled dark: the attention. That is what the eye lands on, and it is
  the part that distinguishes one architecture from another.
- **Annotations sit outside on leader lines**, spread down both margins.

Against that, what this editor drew was six boxes in a column with
`transformer_block` as an opaque rectangle. The graph already had the right
structure — the catalog expands a block to exactly `norm1 → attn → ⊕ → norm2 →
mlp → ⊕` with both bypasses — but the canvas showed one flat level, so none of
it was ever on screen.

### Changed

- **Unfolding** (`state/unfold.ts`). A container is drawn as a frame around its
  contents rather than a box you open. Boundary nodes are short-circuited away
  so a wire runs from the part that produces a tensor to the part that consumes
  it, across however many container walls lie between. ELK lays the tree out in
  one pass and sizes every frame around what it holds.
- **A detail control** in the toolbar: flat, or one to five levels open. Two is
  the default, which is exactly the published figure — the stack, and inside it
  the six parts. Three opens the attention and the feed-forward, which is the
  side panel those figures draw. Flat is the old editable view, unchanged.
- **Trivial nesting is merged.** `repeat` around a single `transformer_block` is
  two rectangles in the document and one in every figure, so the outer frame
  keeps its count and takes the inner one's identity.
- **Residual sums are circled glyphs**, not labelled boxes, and the bypass wire
  is dashed and thinner than the line it rejoins.
- **One dark box.** The palette was fifteen competing pastels; now the hue lives
  in the outline, fills barely lift off the sheet, and attention is the single
  filled block — inverted in dark mode, where it becomes the light one, so it is
  still what you see first.
- **Callouts were being drawn in the wrong place.** A part inside a frame is
  positioned relative to that frame, so every annotation below the top level was
  landing on top of the ones above it. They now resolve absolute positions, push
  each other apart down the margin, drop duplicates, and route their leaders as
  an elbow to wherever they ended up.

### Two bugs worth remembering

Merging frames before resolving edges cut every wire that crossed a container
wall, because the set of open frames is exactly what the edge tracer walks
through to find the parts on either side. Merging is now the last step.

A near-black outline on the attention category made every *frame* around an
attention block read as a dark slab. A frame tints from its outline, so the
outline cannot be the thing that makes a leaf stand out — the fill is.

### Still outstanding

- Bottom-to-top flow. Every reference draws the input at the bottom; this draws
  it at the top. It is one ELK option and a handle swap, and it should be a view
  toggle rather than a silent change.
- Comparing two designs side by side, which is what that reference figure is
  *for* and what this cannot do at all.
- Junction dots at branch points, at KiCad's ratio of six times the stroke width.
- Canvas markers pointing at an offending port.
- Per-rule Error/Warning/Ignore with persisted exclusions.
- Locks respected by `scaleDesign` and by edits arriving over MCP.
- An SVG export of the sheet.

## Fifth pass: tooling, wiring and a design system

A blunt list this time, so a blunt reply. Each item, and what happened.

**No tooling.** There was no tool strip, no menu, no
settings, no shortcuts — only a row of buttons. Now: a modal tool strip down the
left of the sheet (select, pan, wire, then fit/zoom/arrange, then lock/delete),
an application menu carrying every command with its key beside it, a settings
dialog, and a generated shortcut sheet.

**"use shadcn with baseui with no corner radius."** Tailwind v4 plus Base UI,
with shadcn's component API and token names mapped onto the ones this project
already had, so `theme.css` stays the only place a colour is decided. Every
radius is zero — asserted in a base layer, not merely configured, because one
library default undoes the look. The three things still round are the residual
glyph and the pin dots, which are marks rather than containers.

**"drawing lines across components should work in all sides not just up down."**
Every port now carries a handle on all four sides. Which side a wire uses is
chosen from where the two boxes actually are: down the sheet by default,
sideways when a wire has to climb back up or when the blocks sit beside each
other rather than above.

**Anchoring was arbitrary; wires wanted line types and exit nodes
per object.** Two parts. Anchors: a block can declare that a port is always
sideways, which `add:b` and `mul:b` do — the residual bypass and the gate are
the lines a figure draws *alongside* the main path, and entering from above made
them read as the main path. Line types: three, the way a schematic has a net, a
bus and a no-connect. Solid for the main path, dashed for a line that skips a
stage, dotted and thinner for one carrying indices rather than activations.

**"keybindings non existant, delete, undo."** `state/commands.ts` is one list
feeding three consumers — the keyboard, the menu and the shortcut sheet — so a
key cannot be documented wrong. Delete, Backspace, undo, redo, duplicate, lock,
fit, zoom, arrange, detail in and out, the panel tabs, theme, settings. Tools
are V, H and W. Shift is only named in a chord when the character does not
already spell it, which is why `?` works.

**"zooming out causes glitches."** The floor was 0.1. Below about a third the
labels are sub-pixel and the browser drops them unevenly, which reads as
flicker; the grid shimmers for the same reason. The floor is now 0.3, and the
grid is a setting.

**"no menu to change settings."** Theme, grid, snap, minimap, title block,
detail, annotations and wire labels, all persisted. The analysis conditions
stay in the operating point, where the numbers that depend on them are.

### Two bugs worth remembering

Base UI's `Menu.GroupLabel` requires a surrounding `Menu.Group` and throws
without one. These captions head a run of items rather than a labelled group, so
they are a plain element.

A Base UI `Switch.Root` renders as `display: inline`, which silently ignores the
height it is given. It needs `inline-flex`.

### Still outstanding

- Bottom-to-top flow, which every reference uses and this does not.
- Comparing two designs side by side.
- A command palette, now that there is a command list to search.
- Rubber-band select across frames; selection is still one block at a time.
- Junction dots at branch points.
- Per-rule Error/Warning/Ignore with persisted exclusions.
- An SVG export of the sheet.

## Sixth pass: the gibberish, and a volume view

### The gibberish

`o_merge` was showing `B ((H)) T ((dh)) · B T ((H)) ((dh…`, which is two einops
patterns run together with the parentheses multiplying. Two separate faults.

**`ex()` compounded parentheses.** The helper that turns a parameter into an
expression wrapped every string in brackets so it could be interpolated safely.
Composites pass their parameters into the composites they expand into, so three
levels of nesting turned `H` into `(((H)))`. It parses the same and counts the
same — every preset still matches to the parameter — but it is what every shape
and every summary was rendering. `ex()` now adds brackets only when they are
needed: a bare symbol, a number, or something already wrapped is returned as it
is. `o_merge` now reads `B H T dh`.

**A reshape was summarised as two values.** The block summary joined parameters
with a middot, which for `rearrange` meant printing two patterns side by side
with no indication that one becomes the other. It is now one string with an
arrow: `B H T dh → B T (H dh)`.

### The empty frame

A container drawn one level open came out as a three-pixel box with its child
floating outside it. ELK was *throwing*, and the catch turned it into a status
message that the next status overwrote, so the failure looked like a drawing bug
rather than a failure.

The cause: `elk.layered.considerModelOrder.strategy` and
`hierarchyHandling: INCLUDE_CHILDREN` cannot be used together. The combination
throws `cannot read properties of undefined (reading 'a')` from inside the
minified worker, which says nothing. Bisecting the option set in the browser
found it in one pass. Model order is worth having on a flat graph and is now
dropped when the graph has any nesting, and a layout failure logs as well as
posting a status.

### The volume view

Brendan Bycroft's LLM visualisation draws a transformer as a tower you fly
through, with every tensor at its real shape. What makes it work is that size
means something: you do not read that the feed-forward is most of the model, you
see it.

`three/model3d.ts` builds the same thing from whatever design is on the canvas.
Every weight tensor becomes a box with its actual two dimensions, layers stack
down the tower, and the embedding and output head sit above and below it in flow
order. Press `3`, or the cube in the tool strip.

Three decisions worth recording, because the first two attempts were wrong:

- **Scale against `d_model`, not against the largest dimension.** Normalising to
  the largest made the vocabulary fill the view and collapsed every projection
  to a sliver. The residual width is the one dimension every tensor in a
  transformer shares, so it is the unit.
- **Colour by role, not by block type.** Nearly every weight is a `linear`, so
  colouring by type painted the whole tower one shade and threw away the only
  comparison anyone wants. The role comes off the path: inside `attn` is
  attention, inside `mlp` is feed-forward.
- **Look down on the stack.** The tensors are plates lying flat; a camera level
  with them sees nothing but their edges. The first render was thirty-two rows
  of streaks.

Clicking a tensor selects the block it came from, so the tower cross-probes into
the inspector the same way the rules list does.

### Still outstanding

- Activations, not just weights. The reference animates a token moving through;
  this draws the static parameters. The attention score matrix and the KV cache
  are the two that would most repay being drawn at size.
- Labels in the scene. Names are in the hover readout only, so a still of the
  tower does not say which slab is which.
- Instancing. Thirty-two identical layers are thirty-two times the geometry;
  one `InstancedMesh` per role would make a 100-layer design cheap.
- Bottom-to-top flow, side-by-side comparison, and the rest of the fifth pass's
  list.

## Seventh pass: finding the 3D view, and collapsing the docks

**"How do I actually view 3D?"** — which is the answer to whether a cube icon at
the top of a tool strip is a view switch. It is not. The mode now has a labelled
segmented control in the toolbar, `Sheet | 3D`, under a `VIEW` caption, which is
the one control that has to name both states.

The key was wrong too. `3` had been given to the volume view, colliding with the
panel keys where `1` and `2` open the Inspector and Symbols — so the third in an
obvious run of three did something unrelated. Panels have `1 2 3` back, and the
volume view is `Shift+V`.

**Collapsing docks, as Fusion does it.** Fusion collapses its browser to a strip
with a double arrow, and the strip still says what is behind it. That last part
is the bit worth copying: a dock that disappears entirely leaves you hunting
through a menu, while a rail keeps the affordance on screen for twenty-six
pixels. The right rail keeps the parameter count visible, because that is the
one number worth having even when the readout is closed.

Three ways to do it, because a dock toggle people cannot find is the problem
this pass started with: the two panel buttons in the toolbar, `Ctrl+B` and
`Ctrl+Shift+B` (with `Ctrl+Alt+B` for both at once), and a double-click on the
divider, which is what the divider is for everywhere else.

### The bug

Collapsing a dock removed its divider from the DOM, and the body is a five
column CSS grid. Dropping a grid item shifts every later one into the wrong
column, so the canvas ended up in a zero-width slot and the right rail got the
whole window. All five items are now always rendered; a collapsed dock swaps its
contents for a rail and its divider for a zero-width spacer.

### Still outstanding

The volume view still draws weights only, has no in-scene labels and no
instancing; and the rest of the fifth pass's list — bottom-to-top flow,
side-by-side comparison, a command palette, rubber-band selection, junction
dots, per-rule severities, SVG export.

## Eighth pass: a selection that is one box, and the layout actually ported

### The double selection

A selected block drew two rectangles, twelve pixels apart. `.part` was a
hard-coded `228px` while the React Flow node it sits in was `NODE_WIDTH`, 216 —
so the symbol overhung its own node and the selection ring sat around one while
the body sat around the other. The part now fills its node, and selection is a
single ring on the symbol with a hairline of sheet between it and the part's own
outline so the two do not merge into one thick border.

### The volume view, ported rather than guessed

The previous pass built a tower from first principles and it did not look like
the reference, which is a fair complaint: the reference's arrangement *is* the
idea. llm-viz is MIT licensed, so the layout is now ported and credited, in the
source header and in the README.

What was actually missing:

- **A residual pathway.** Their layout hangs everything off a spine at `x = 0`,
  one plate per stage, and the previous version had no spine at all — just bands
  of boxes with nothing joining them.
- **Sides that mean something.** Norms go to the left of the pathway, attention
  and feed-forward expand rightward. The previous version fanned every tensor in
  a layer into one row, which threw away the distinction.
- **Sublayers sequenced downward.** A block is four rows — norm, attention,
  norm, feed-forward — not one band.
- **Column wrapping.** They wrap at twelve blocks. That is the answer to the
  thing that made the last attempt unreadable: thirty-two layers at true spacing
  is a spike a thousand units tall where each layer is three per cent of the
  height. Llama-3-8B is now three columns of twelve.

One deliberate departure: their `cell` is a fixed size per element, which is
right for a forty-eight-channel demo and impossible for a 128k vocabulary. Sizes
here are normalised against the residual width instead.

### Framing, twice wrong

First a bounding sphere, which no rotation can escape but which for a stack that
is wide, deep and thin has a diagonal far larger than anything you ever see — so
the model sat at half the size of the view. Then per-axis maxima added together,
which over-counts because the widest corner is rarely the nearest one. It now
computes, per corner, the distance that just contains it, and only reframes on a
rebuild or a resize so the zoom never jumps mid-orbit.

### Still outstanding

Activations rather than weights; labels in the scene; instancing; and the
standing list from the fifth pass.

## Ninth pass: porting the layout rather than its description

The last pass read a summary of llm-viz and rebuilt from that. This one reads
`src/llm/GptModelLayout.ts` and ports it. The difference is not small.

### What the summary hid

**`cy` is the vertical extent, and the residual is `cx: T, cy: C, cz: B`.** The
residual pathway is a *tall standing plate* — T wide, C high, B deep — and the
stack advances downward by `C * cell` per stage. Every previous attempt drew
tensors as flat tiles lying in the x-z plane, which is why they read as a floor
of slabs rather than as a tower. This single fact is most of the visual gap.

**Attention heads fan out along z**, at `headWidth = 3*B*cell + qkvMargin*2`,
each head's Q, K and V offset from its own centre. That depth is what gives the
reference its layered look.

**The column wrap is `columnWidth = (C * 14) * cell + margin * 2`**, and it
works by shifting `leftX`, `rightX` and `lnLeftX` right and resetting `y` to the
top — not by adding an origin offset. Without it a 32-block design is a sliver
twenty-six times taller than it is wide, which is exactly what the previous
attempt produced.

**Activations are drawn, not just weights.** `t: 'w' | 'i' | 'a'` — weights,
intermediate values, and the layer-norm and softmax aggregates. Drawing only the
weights throws away the residual pathway, the Q/K/V vectors, the attention
matrix and the MLP activation, which between them are most of what the picture
is *of*.

**`small` blocks are dropped on large models.** Biases and the layer-norm
aggregates are flagged in the source for exactly this reason.

### Departures, and why

Their `cell` is a constant 1.5, which is right for a 48-channel demo and
impossible for a 128k vocabulary, so `cell` here is chosen to give the residual
column a fixed height whatever the model. And heads are capped at six per block,
since 32 blocks × 32 heads is a thousand head groups.

### The look

Faces are shaded by a fragment shader that draws the block's own cell grid, with
the density clamped by `fwidth` so a 128k-wide plate stops asking for lines once
a cell falls below a pixel, plus a faint per-cell speckle so a tensor reads as
data. Weights and activations are separate materials, which is the distinction
the view exists to make.

One bug worth remembering: `THREE.Color` converts a hex from sRGB into the
linear working space, and a raw `ShaderMaterial` writes straight to the
framebuffer with no conversion back, so every colour rendered about two stops
too dark until the shader ended with `#include <colorspace_fragment>`. And a
backtick inside a GLSL comment closes the template literal the shader lives in.

### Still outstanding

Real values rather than a speckle (the design has shapes, not weights); labels
in the scene; arrows between blocks; instancing; and the standing list.

## Tenth pass: the detail

Zoomed in, every plate was flat colour. Two things were missing and one was
actively wrong.

### The grid gave up instead of adapting

The shader drew one line per cell and bailed out once `fwidth` said a cell was
under a pixel. That guard is right in principle and useless in practice: a real
tensor here has thousands of cells per axis, so *every* face fell through to the
flat branch. The reference gets away with per-cell lines because its demo model
has six tokens and forty-eight channels.

The face is now ruled into as many divisions as can actually be seen — halving
down from the true cell count until they are at least eight pixels apart. Zoom
in far enough on a small tensor and the divisions *are* the cells, and the
per-cell speckle appears with them; zoom out and you get a readable eight or
sixteen. Either way a face says "matrix" rather than "painted box". A hairline
inside each outer edge keeps two adjacent plates reading as two plates.

### Nothing had a name

The reference labels every tensor — "Q Weights", "Attention Matrix", "MLP
Result" — and that is most of what makes it a diagram rather than a texture.
Labels are now an HTML layer, projected after each render: a block's screen size
is computed from its extent at its own depth, anything under forty-six pixels is
dropped, and the largest four dozen win. A wall of forty-eight identical
"Layer Norm" tags is noise; the one on the block you are looking at is not.

### Still outstanding

Real values rather than a speckle — the design carries shapes, not weights, so
this would need a loaded checkpoint. Flow arrows between blocks. Instancing:
about fifteen hundred meshes each carry their own material, which an RTX 5080
does not notice and a laptop might.

## Eleventh pass: read the label code instead of inventing one

The last pass added a label layer that drew every name it could fit. Zoomed in
that is a wall of overlapping text, because there are several thousand blocks
and no amount of size-culling fixes it.

The repository answers this directly. `mkLabel` creates
`IBlkLabel { visible: number, cubes: IBlkDef[] }` with `visible` at zero, and
`Interaction.ts` raises it in exactly one place:

```ts
for (let label of state.layout.labels) {
    for (let c of label.cubes) {
        if (c === main) {
            label.visible = 1.0;
        }
    }
}
```

`main` is the block under the mouse. Labels are never all on. And they are
*grouped*: `qLabel = [qWeightBlock, qBiasBlock, qBlock]`,
`mtxLabel = [attnMtx, aggs, attnMtxSm]`, `mlpLabel = [...]` — hovering any
member names the whole group, because those are one step rather than three
boxes.

That is now what happens here. Blocks carry a group key, a hover lights the
group and names only its members, and nothing is labelled otherwise. Hovering
the MLP weights of block 7 gives six labels — MLP Weights, MLP, MLP Activation,
MLP Projection Weights, MLP Result, MLP Residual — and nothing else on screen.

The lesson is the obvious one, and it took being told twice: when the source is
available, read the part that does the thing rather than the part that
describes it. Three of the last four passes were spent rebuilding behaviour that
was thirty lines away.

### Still outstanding

Real values rather than a speckle; flow arrows; dimension rules along a hovered
block's edges (`blockDimension` in `Annotations.ts`, not yet ported); instancing.

## Twelfth pass: the right button, and blocks as data

### Inspect should not be on the menu

Right-clicking showed the browser's menu — Back, Refresh, Save as, Inspect. The
sheet now has its own, built on `Menu` with a virtual anchor at the pointer
rather than on Base UI's `ContextMenu`, whose trigger never received the event
on this canvas. Right-clicking a block selects it first, because a menu offering
"Delete" for whatever was selected a minute ago is worse than no menu.

One embarrassment worth recording: the component was written, imported and
type-checked, and never rendered — an earlier patch's JSX anchor had been
reformatted by prettier, so the `replace` silently matched nothing. TypeScript
does not complain about an import that is only ever imported. Three rounds of
debugging Base UI went into something that was not on screen.

### Blocks a design defines for itself

The catalog was three arrays compiled into the package: adding a block meant
editing TypeScript and rebuilding. A *primitive* has to be code, because it
carries the parameter-count and FLOP formulas. A *composite* does not — it is
parameters, ports and a subgraph, and the only reason `gqa_attention` is a
function is that its expansion interpolates parameters into the graph it builds.
That interpolation is string substitution, so it can be written down.

So a design carries its own library in `doc.defs`, the way a KiCad project
carries its own symbols:

- `catalogOf(doc)` returns the built-ins plus whatever the document defines,
  cached on the document object. Every resolution in the core now goes through
  it — shape inference, flattening, code generation, `explain`.
- A definition is parameters, ports and a template subgraph whose node
  parameters refer to the block's own parameters as `$name`. Boundary nodes are
  generated from the declared ports, so a definition cannot declare one set and
  wire another.
- A user block never shadows a built-in, and one that fails to compile is
  dropped from the catalog and reported by a design rule rather than thrown.
- "Make a block from this level" lifts what is on screen into a definition,
  turning the symbols it uses into parameters — so a block extracted from a
  4,096-wide design works at 2,048 without being rebuilt. Libraries import and
  export as JSON.

Verified end to end: a gated feed-forward written purely as data resolves with
no shape errors, counts 57,344 parameters against a hand-computed 57,344,
passes the rules, and generates the three `nn.Linear` layers it should.

What this does not give you is a new primitive. Anything needing its own
parameter-count or FLOP formula still belongs in `primitives.ts` — but a new
architecture is almost always a new arrangement of existing primitives, which is
what this covers.

### Still outstanding

A visual block editor: definitions are made from a level or hand-written JSON,
and there is no way to edit a definition's parameters or ports in the UI. A
shared library location, rather than per-document plus import. And the standing
list from earlier passes.

## Thirteenth pass: menus that fit, and controls where the thing is

### The menu was a list

Every command in one flat column. It ran past the bottom of the window, and
reaching for the end of it closed it — a menu you have to scroll is a menu that
does not work. Fusion hides the same quantity of commands behind a dozen short
ribbon dropdowns.

`menu-tree.tsx` declares the structure once as data and renders it for both the
application menu and the right-click menu, so a new command has one place to be
added and the two cannot drift. The application menu is now seven entries — File,
Edit, View, Blocks, Panels, and the two help items — and the right-click menu is
nine at most, with the block actions on top only when a block was clicked.
Nothing scrolls.

### Collapse controls belong on the panel

They were in the command bar under a "Docks" caption, which is the wrong place
twice over: it is not where you are looking when you want a panel gone, and it
puts a view-state toggle among the drawing tools. Fusion puts the arrow on the
browser itself. So the model tree's header carries a `«` and the readout's
carries a `»`, the "Docks" group is gone, and the keys and the Panels submenu
still do the same thing for anyone who prefers them.

The right dock also gained the header it never had — the readout started
straight in on the operating point, with nothing saying what the column was.

### Still outstanding

A visual block editor; a shared library location rather than per-document; and
the standing list from earlier passes.

## Fourteenth pass: thickness, flow, and a camera that stays where it is put

### The parameter count sat under a pin

`.part` was padded top and bottom only. Once pins moved onto all four sides, a
body that ran to the left and right walls put the count — `6.98B` — underneath
the right-hand pin. The fix is one line, `padding: 12px`, and the check is not a
screenshot: every text node in every part is measured against every handle, and
nothing intersects.

### The volume view had no flow and no sides

Two things were missing from the port, and they are the two that make the
reference read as a computation rather than a pile of plates.

**Thickness.** The reference's `cell` is a constant 1.5 against a 48-channel
model, so a one-cell-deep weight plate is two per cent of its own height and
plainly visible. Scaling `cell` to give the residual column a fixed height —
which is what makes a 4,096-wide model fit on screen — shrinks that to two
hundredths of a per cent, and every plate renders as a sheet with no sides at
all. A floor of `COLUMN_HEIGHT * 0.022` on each extent puts them back. The thin
biases and layer-norm aggregates are drawn on large models too, now: they are
the grey posts between the plates, and dropping them is what left the earlier
render looking like loose tiles.

**Flow.** `components/Arrow.ts`, ported: a ribbon leaves the middle of one
block's edge and arrives at the middle of another's, padded off the face,
dog-legged where the two edges do not face each other, with a triangular head
twice the ribbon's width. Blue out of a weight, green out of a value. The
reference's own first line about it is the part that was easy to miss — "a flat,
rectangular, ribbon-like pathway **with lines down the edges**" — and a
translucent quad with no border is a smear, not a path. So the fill is the
block's colour at three tenths and the border is the same at eight, which is
what its `ribbonColor` and `borderColor` are.

Widths are the one place the port had to reason rather than copy. The reference
draws 6 units against a residual plate 9 wide, and passes an explicit 2 where an
arrow feeds a one-cell aggregate: it sizes an arrow to the tensor it carries.
So the width here is two thirds of the narrower of the two faces, bounded so a
ribbon into a `γ` column is still drawn and one between two residual plates does
not swallow them. Padding is a sixth of the margin, which is the reference's
ratio exactly.

The two colours are theme tokens, not literals. On paper they are the
reference's `#3333aa` and `#33aa33`; against the dark sheet those sink into the
background, so the dark theme carries lifted versions of the same two hues.

### A resize threw the view away

The pane refits the camera whenever it is resized, which is right until someone
has zoomed in — after which a status line growing a word, or a dock opening,
silently returns them to the whole model. A resize now reframes only a camera
nobody has touched. Which leaves the obvious gap: there was no way back. The
volume view now registers the same viewport API the sheet does, so `F` fits,
the zoom commands work, and turning the model round is no longer a one-way trip.

### Still outstanding

A visual block editor; a shared library location rather than per-document; real
values in the cells rather than the speckle; instancing, since the view is still
one mesh and one material per plate; and the standing list from earlier passes.

## Fifteenth pass: the arrows curve, and the boxes have sides

The previous pass put ribbons in but drew them as flat polylines in the x-y
plane. That is not what the reference does, and the difference is the whole
point of the view.

### The frame, and why it matters

`drawArrow` builds an orthonormal basis from the run's own direction — `side`,
`dir`, `normal` — flattens both endpoints into it, lays the ribbon out in that
frame's x-y plane, and transforms the result back. The basis is a reflection, so
the same matrix does both directions. Everything downstream can then think in
two dimensions while the ribbon sits wherever in space it needs to.

That is what makes a curve expressible. When the two ends are at different
depths, or the ribbon has to arrive side-on, the run is swept as a cubic bezier
whose tangents are `dir` at the start and the arrival direction at the end. Six
attention heads funnelling back into one output is exactly that case, and
`drawArrowBotToSide` — Q and K leaving the bottom of their vectors and landing
on the *face* of an attention matrix that sits at a different depth — is the
other. A dogleg is two runs, the second carrying a rounded corner and the head.
For llama-3-8b that is 366 runs, 126 of them curved.

### Q, K and V are three blocks, always

This drawing used to merge them into one plate on any model with more than
twelve layers, on the grounds that three per head was noise. The reference does
not: it keeps them at three depths and only drops the *margin* between heads
once there are a lot of them. The depth between Q, K and V is what every arrow
into the attention matrix curves through, so merging them removed the thing
worth looking at and left the head reading as one flat sandwich.

### Ribbon width is the one thing that cannot be copied

The reference's 6 units sits inside a 12-unit margin and against a residual
plate 9 wide, so both "half the margin" and "two thirds of the face" describe
it. Its demo has six tokens. A real design has four thousand, so its plates are
six times wider than its margins and the two rules disagree by that factor. Two
thirds of a face wins on appearance until you notice the dog-legs turning inside
out — a ribbon wider than the gap it runs through puts its own corner above the
edge it left. So the width is two thirds of the narrower face, capped at the
margin: the widest ribbon that still fits between two stacked blocks.

### The boxes had no sides

Faces were outlined by darkening a hairline inside each edge, which against a
dark sheet is the same as not outlining them. The reference draws its cube edges
in white, on top of everything, and that is what makes a plate read as a solid
rather than a painted rectangle — particularly where the side is only a few
pixels across. `--vol-edge` is that outline, light on dark and dark on paper,
and the minimum thickness went from 2.2% of the column height to 3.5%.

### Still outstanding

Arrows stop after three blocks, as the reference's do. The view is now ~3,000
meshes each with its own material, so instancing has gone from nice to needed.
Plus the standing list: a visual block editor, a shared library location, real
values in the cells rather than the speckle.

## Sixteenth pass: picking a wire

Wire selection, dragging and connecting were the worst part of the tool, which
for a CAD is the wrong part to be worst. Four separate causes, only one of them
about wires.

### The frames were eating every click

React Flow renders edges in a layer beneath nodes, and a frame is a node: a
full-size div with `pointer-events: all` set by React Flow's own stylesheet. In
an unfolded drawing nearly every wire runs inside a container, so nearly every
wire was underneath an invisible sheet of glass. Clicking one selected the frame.

The fix is the rule every vector editor uses for a shape with no fill: **it is
picked by its outline, not its interior.** A frame is a container drawn around
what it holds, and its interior belongs to the parts and wires inside it. The
wrapper is `pointer-events: none` — set inline, because React Flow's stylesheet
is imported after ours and specificity alone does not settle it — and the
outline band and the caption re-enable themselves. Measured afterwards: forty
sample points along the visible wires, none of them landing on a frame.

### The pick aperture shrank as you zoomed out

React Flow draws an invisible band around each edge for hit testing, twenty
units wide. Those are *flow* units, inside the viewport transform, so at the
minimum zoom of 0.3 the band is six screen pixels. That is the zoom a real
design is worked at.

Every drafting program keeps its pick aperture constant in screen space — you
aim with the cursor, not with the drawing — so `interactionWidth` is now divided
by the zoom. Measured at 0.30: sixteen screen pixels. Measured at 0.60: sixteen
screen pixels.

Pins had the same problem and could not be fixed the same way, because React
Flow measures the handle element to decide where a wire begins: growing it moves
the wire. A pseudo-element does not change the measured rectangle but does count
for hit testing, so the target grows to a fingertip while the wire still starts
on the pin.

### A wire could not be re-pointed

`edgesReconnectable` defaults to true and does nothing without an `onReconnect`
handler, which there was not one of. Changing where a wire went meant deleting
it and drawing it again — something no schematic editor has ever asked anyone to
do. Now both ends drag, with an eighteen-pixel grab radius, and the move is a
single commit so one undo puts it back. A drop into empty space leaves the wire
alone rather than deleting it: an edge with one end attached is not something
this IR can hold, and deleting on a cancelled drag is a destructive surprise.

### Nothing said what the cursor was over

eeschema highlights before you commit, and highlights the whole **net** rather
than the segment under the cursor, because the net is the thing that means
something — a wire that branches is still one signal. Hovering now lights every
wire leaving the same output pin, which is exactly the set the analysis treats
as one tensor, and raises them over the blocks they run between. The ends of a
lit or selected wire show as hollow squares, eeschema's mark for a wire end,
which is also the only thing that says the ends can be dragged.

### Three smaller things, all from the same reading

- `paneClickDistance` and `nodeClickDistance` were zero, meaning one pixel of
  travel between press and release turned a click into a drag and the click was
  lost. On a trackpad that is most clicks. Four pixels, the slop every desktop
  toolkit allows.
- `Delete` deleted nothing. React Flow's default is Backspace alone; the app's
  own command owned Delete and only knew about blocks. Both keys now go through
  React Flow, which handles wires as well as blocks. The two paths cannot
  double-fire because `commit` returns early when the document is unchanged.
- `connectionRadius` was React Flow's default of twenty, which assumes a mouse
  on a small graph. Thirty-four, for a dense sheet worked at low zoom.

### Still outstanding, from the same research

KiCad has three more things worth taking. A **box selection whose direction
changes its meaning** — left-to-right selects only what is fully enclosed,
right-to-left selects anything it touches — which is a decades-old CAD
convention React Flow has as a fixed mode rather than a gesture. **Clarify
selection**, a pop-up listing what is under the cursor when several things
overlap, reached by a long click or Alt. And **select connection**, which grows
a selection along a net one junction at a time.

### Postscript: the circled plus

A part you can open carried a `⊕` in its bottom-right corner, meaning
"double-click to open". On a transformer sheet a circled plus is a residual sum
— this drawing sets `add` in exactly that glyph, deliberately, because that is
how the figures draw it. So the mark for "this block has an inside" was the mark
for "add these two tensors", placed on the border where it also fouled the
parameter count.

It is now the flowchart standard's predefined process: a box with a second rule
inboard of each side, which is the established way of saying "defined
elsewhere". It borrows no symbol that already means something, and being
structural rather than a twelve-pixel badge it survives being zoomed out.

## Seventeenth pass: the port becomes an object

Phase E1 of the editor handoff. Not a visible change — a change that makes four
of the visible ones possible.

### Nine facts in one string

A port was `name → shape pattern`. Everything else about it lived somewhere
downstream, and two things lived in the renderer:

- **Which side a wire leaves by** was a lookup table keyed on `"type:port"`
  strings. `SIDEWAYS_IN` held `add:b` and `mul:b`; `SIDEWAYS_OUT` held `rope:y`,
  `causal_mask:mask` and `router:weights`.
- **What a tensor carried** was decided by asking whether a dtype name started
  with `"int"`.

Two of those five table entries matched no catalog type at all. `causal_mask` and
`router` are not block types — the real ones are `sdpa` and `topk_router` — so
those rows had never done anything and nothing had noticed. That is the failure
mode of a table keyed on strings assembled at a distance from what they name.

A port now declares its own facts:

```ts
interface PortSpec {
  shape: string;
  dtype?: "float" | "half" | "fp8" | "int" | "bool" | "inherit";
  optional?: boolean;
  whenUnconnected?: "zero" | "identity" | "causal" | { tensor: string };
  anchor?: "flow" | "side";
  showName?: boolean;
  doc?: string;
}
```

A bare string is still legal and means `{ shape }`, so the twenty-two primitives
with nothing more to say did not have to say it.

### Normalising in one place, deliberately

`portsOf()` returns ports with every default filled in. The alternative — a
`normalisePort` helper each consumer calls — was rejected: a consumer reaching
for the raw declaration would find a string on most primitives and an object on
the rest, and would get one of the two wrong. Making the resolver the only door
means there is no raw form to reach for.

This rippled further than expected. `infer.ports` now carries resolved ports
rather than pattern strings, which is what the canvas reads; `explain` had to
stop passing ports straight into a field called `shapes`; and the MCP's
`get_block` gained `dtype` and `optional` on every port, which is strictly more
than it could say before.

### What it unblocks

The three ports that mean something by leaving sideways — `add:b`, `mul:b`,
`rope:y` — now say so. A block a *document* defines can say it too, which the
table could never have allowed: there was no way to add a row to a set compiled
into the renderer. That was the real cost of the old design, and it only becomes
visible once the definition editor exists.

`dtype` is the other half. An integer tensor arriving at a float matmul is a bug
shape inference cannot catch, because the shapes agree perfectly. The declaration
is what makes the check possible; the check itself is E1's remaining item.

### Still outstanding

E2 through E7 of the handoff, in dependency order: findings with stable ids drawn
on the canvas (the highest-value single item — seventeen rules run on every edit
and none of them is visible where the work happens), the command surface and
palette, the inspector, the definition editor and library, multi-selection and
the bottom dock, and then operations, configurations and tensors as first-class
objects.

## Twelfth pass: the probe goes both ways, and the cluster gets a panel

### A marker you could see and not follow

Findings had reached the drawing: a part with something wrong carries a coloured
disc at its corner, and a port a finding names carries a mark on the pin. What
the disc could not do was anything. Hovering gave a tooltip, which is the whole
finding crammed into a `title` attribute, and reading it properly still meant
crossing to the panel, finding the row, and matching the path by eye.

The list had gone the other way since it was built — click a finding and the
editor opens the level that owns the block, selects it and centres the viewport.
Pressing the marker now does the reverse: it opens the rules list, scrolls to
that block's findings and outlines them, with a bar at the top naming the block
and a way to clear it. The two directions together are what a DRC list is for.
A highlight that outlives what it pointed at would be worse than none, so
selecting anything else clears it.

The marker had to become a `button` to be pressable, which is also what makes it
reachable by keyboard and announceable — it was a `div` with a tooltip. It
carries `nodrag` so pressing it does not drag the part, and stops the click so it
does not also land on the canvas as a selection: it is a control on the part
rather than part of it.

### The cluster

The readout answers what a design costs under the operating point you set. The
question that comes first — what the operating point should *be* — had no
answer anywhere in the editor, and the analysis had everything needed to give
one. `Cluster` is a fourth tab: every way of splitting the training that the
cluster admits, priced, with the ones that fit listed least demanding first and
a bar for how much of the device each fills. Pressing one applies it.

That last part is why it belongs in the editor rather than only in the command
line. The parallelism is editor state, so a plan is something a panel can hand
to the operating point and watch every other panel follow.

It is also the one panel that does not read `derive()`. That runs on every
keystroke; this is a few hundred analyses, and the answer only moves when the
cluster or the design does — so it keys on the operating point *minus* the
parallelism and the recompute setting. Applying a plan must not send the panel
looking for another one.

Making it applyable found two things the operating point could not say. Sequence
parallelism the analysis had modelled all along and the editor had no way to ask
for, so every plan offered with it would have applied as something else and
contradicted its own number; it is a checkbox now, disabled when there is no
tensor-parallel group to shard across. Expert parallelism had just become real
in the engine and had no control at all.

### What a design has decided a rule means

The last piece of E2. Every design-rule tool has this and for the same reason: a
rule that is right in general is sometimes wrong here — Gemma cannot use a fused
attention kernel and no arrangement of the design changes that — and the
alternative to recording the decision is people learning to read past a warning
until the warnings stop meaning anything.

`doc.rules` is a map from rule id to `error`, `warning`, `info` or `off`. It is
in the document rather than in the editor because it is a decision about the
design: it travels with the file and shows up in review. It can raise as well as
lower, because "in this project a padded vocabulary is an error" is as real a
decision as accepting a warning.

Suppression is never silent. The report carries `overridden` — which rule, on
which block, from what to what — and the panel shows a line saying how many
findings the design dropped and how many it re-graded. A design that could
quietly hide its own errors would make every report unreadable, including the
ones the preset tests rely on.

The control went in twice, and the second time was the one that mattered. The
rule book lists the eighteen design rules, so putting it only there left out
exactly the findings most worth accepting: a block's own constraint — `SDPA-03`,
`ATTN-01` — is a finding with a rule id and no row in that book. It is on the
finding itself as well, which is also simply where the decision gets made.

### Comparing, in a dialog rather than a tab

`View > Compare` against where this design started, or against any preset. Both
halves, because either alone misleads: that `F` went from 11008 to 14336 does
not tell you the model grew by 1.3B parameters, and that it grew by 1.3B does
not tell you where. A changed block's path is a link into the drawing.

A dialog rather than a fifth tab. Comparing is something you open, read and
close; the four tabs are used constantly, and a fifth would cost them width for
something that is not. The rule for the readout column is that everything in it
is worth having on screen while you work, and a diff is not.

The baseline is the design as it was when it was opened, which `setDoc` records
and an edit does not touch — that is the whole distinction. "Compare against
this from now on" moves it, for when the interesting question becomes what has
changed since a particular point rather than since the file was loaded.

### The command surface

E3. The menu groups thirty-six commands behind six dropdowns, which is the right
way to *browse* them and the wrong way to reach one whose name you already know.
`Ctrl+K` lists them all, filtered as you type.

`state/commands.ts` was already the single list behind the keyboard, the menu,
the shortcut sheet and the native menu, so the palette is one component over it
and cannot fall out of step with any of them. It shows each command's group and
its shortcut, so using it teaches the shortcut — which is the point of a palette
in a tool people use every day.

Two decisions that took a second pass:

- **Matching is a subsequence, weighted towards word starts.** "mb" should find
  "Make a block from this level" because that is what typing initials means. The
  first version scored only on how tightly the letters sat, which ranked
  "Vi[e]w [B]oth docks" above "Export blocks" for "eb" — arithmetically true and
  not what was asked for.
- **A command that is unavailable sorts on its match, not below everything.**
  Available-first buried a command named exactly under one that was not, which
  is the same failure as hiding it, arrived at politely. The row is greyed and
  cannot be pressed; that is what says it is unavailable. "Export blocks is
  greyed out" is an answer, and a command that vanishes when a design has no
  blocks of its own is one you conclude does not exist.

### The inspector, which was twenty-nine fields in a row

E4. `transformer_block` declares twenty-nine parameters and about ten of them
mean nothing at any given moment: a dense block has no `expert_hidden`,
grouped-query attention has no `kv_lora`, an RMSNorm has no bias. They were all
shown alike, so reading the panel meant already knowing the architecture well
enough not to need it.

A parameter can now say two things about itself. `group` is the heading it
belongs under — Shape, Attention, Feed-forward, Normalization — and `when` is the
condition under which it means anything, as one other parameter's value being
one of a set. One level, deliberately: every case in the catalog is "this field
matters when that enum is one of these", and a condition language would be a
second thing to learn for no case that exists.

Greyed and labelled "unused", not hidden. A field that disappears when you change
`mlp` is a field you go looking for, and the value is still in the document
either way — a block may legitimately be set up before the switch that turns it
on. The heading carries the count, so "11 not in use" is visible before you have
read a single row.

The declared order is left alone. It is also the order a generated class
documents its parameters in, and reordering it to suit one panel would have
rewritten sixty `model.py` docstrings for a layout decision; the grouping is the
panel's job.

## Thirteenth pass: the library, and what a definition cannot yet be

A design carries its own composites in `doc.defs`, and you could already make
one out of a level, import a file of them and export one — but not look at what
you had. `Blocks > Blocks this design defines` is the list: what each one
declares, where it is used, and rename and delete.

Delete is refused while a block is in use, and the count that matters is not
just the instances on the canvas. An instance inside *another* definition is a
use too, and it is one nobody can navigate to, so it is counted separately and
said separately. Rename moves every instance with the definition, at any depth
and inside other definitions, because a rename that moved the definition alone
would leave the design reporting "unknown block type" with no clue that it used
to be known.

### What "show" does, and what it does not

It goes to an instance, not to the definition. That is not a shortcut: a
template is written in terms of its parameters — `$D`, not 4096 — and only an
instance says what those are. There is nothing to draw until something binds
them, and once something does, the composite machinery already draws it.

Editing the template in place is the piece that is not here. It could be:
`graphAtPath` is the single door every operation goes through, so teaching it
one prefix would make every tool write into `defs` instead of the graph. What
stops it is not plumbing but a question with no obvious answer — a `linear`
dragged into a parameterised template with `in_features` of 4096 either means
4096 or means `$D`, and the editor would have to decide which without being
told. Guessing wrong writes a definition that silently stops scaling, which is
exactly the failure the parameterisation exists to prevent. The prototype that
reached this point was reverted rather than shipped half-answered.

This is also where the first test of the editor's own logic went. `state/`
is pure — a document in, a document out — and which type a rename left behind is
the sort of thing that type-checking cannot see. Writing those tests immediately
found that `freeTypeName` captured the catalog at module-load time, so it knew
nothing was taken. It works in the app only because `main.tsx` loads the engine
before it imports anything, which is an ordering rule nothing enforces; it is
looked up on demand now.

## Fourteenth pass: selecting more than one

React Flow has always drawn a rubber band and reported what it enclosed. The
canvas threw the report away — `onNodesChange` filtered `select` out on the
grounds that selection is driven by the document — so the gesture did nothing
and ctrl-clicking replaced rather than added.

Selection is still driven by the document. What changed is that the document
can now hold more than one: `selection` stays the *primary* and `also` carries
the rest. Keeping them apart rather than making `selection` an array is what
let the change be additive — the inspector, the volume view and a finding's
highlight each answer exactly one block, and asking them to mean "one of
several" would have made every one of them worse. Only what can act on several
— delete, duplicate, lock — had to learn there are several.

Which one is primary follows the gesture. A modified click names one block, so
that block becomes primary and the inspector follows it. A rubber band names no
one block, so whatever was primary stays, and the inspector does not jump to an
arbitrary corner of the box. Delete works deepest-path-first, or removing one
would shift the path of the next.

### The key that was not there

Writing the first test of the command list found that `Ctrl+D` was claimed by
both Duplicate and Compare. `handleKey` looks a chord up in a Map, so the last
one written won and **Duplicate's shortcut had silently stopped working** while
the menu went on printing it. The single list was supposed to make exactly this
impossible; what it prevents is a key *documented* wrong, not two commands
claiming one. A test now says no two may, with the deliberate aliases — the
second key for Redo, Delete and Zoom In — named rather than excused. Compare
moved to `Ctrl+Shift+C`.

## Fifteenth pass: the checks come out of the tab

The first complaint on the original list was that panels are tabs, so you cannot
see the analysis and the inspector at once, and the tenth was that the issues
list is nearly inert. The readout came out of the tabs three passes ago. The
checks had not: eighteen rules ran on every edit and their answer lived behind a
tab, so finding out what was wrong meant leaving the inspector, and fixing it
meant leaving the findings.

They are a dock along the bottom now, which is where a PCB tool puts its DRC and
for the same reason — the checks are about the drawing, so they belong under it
rather than beside it. Collapsed, the dock is still a strip carrying the counts,
because a design that is broken should say so somewhere on screen whatever else
is being looked at. Pressing a marker on the canvas opens it filtered to that
block, which is the cross-probe the list was supposed to have.

Under the whole body rather than inside a column: a dock that lived in one
column would be as easy to lose as the tab it replaced.

### A panel that contradicted itself

Notes are muted by default, so a design whose only finding is a note showed both
"1 info" in the badge row and "Nothing to report. Every rule is satisfied."
below it. The empty state asked `derived.ok`, which only knows about errors,
where the question is whether anything was found at all. It now distinguishes
the two and says how many findings are hidden.

## Sixteenth pass: editing the definition itself

A block the document defines can now be opened and edited directly, rather than
looked at through an instance. `graphAtPath` is the single door every operation
goes through, so teaching it one prefix — `@def/<type>` — made every tool that
already exists write into `defs`: add a block, wire it, move it, rename it,
delete it, all of them, with undo working because they are the same operations.

### What a literal means

The question that stopped this last time: a `linear` dragged into a
parameterised template with `in_features` of 4096 either means 4096 or means
`$D`, and the editor cannot ask.

It does not have to. The rule is that **inside a definition, a bare identifier
naming a declared parameter is that parameter** — which is exactly how the
built-in composites are written, where `gqa_attention`'s expansion names
`d_model` directly. So the canvas is handed a *preview*: the template with every
`$name` rewritten to `name`, over a symbol table made of the block's own
parameters at their declared defaults. Shapes resolve, weights count, and the
inspector shows `D` where the template holds `$D`. On the way back, a value
naming a parameter is stored as a reference to it and everything else is stored
exactly as written. 4096 means 4096 at every width; `D` means the width.

What you see is what is kept, and the rewrite is total in both directions, which
is the property that makes it safe. It is tested as a round trip rather than in
one direction.

### One seam, not a fork

`useDerived` is the design's numbers and `useLevel` is the level's. They were the
same until now. Inside a definition the template is analysed as a design of its
own — only then do its parameters have values — and the answers are re-keyed
under the definition's path, so the canvas and the inspector go on asking the one
way they always have. The readout keeps showing the design, because a definition
being edited does not change what the design weighs.

### Still outstanding

E7: operations as first-class objects.

## Eighteenth pass: the timeline

The store had kept a hundred states since the first pass, and undo walked them
one press at a time. Making that list *readable* was the obvious half: give
every state the sentence the toolbar showed when it was made and let a row be
pressed. That took an afternoon and is genuinely useful.

It is also not what E7 asked for, and the gap is the interesting part. A list
of states can say what the design looked like before. It cannot say what the
design would look like **without the third edit**, because by the time an edit
is a document the operation that made it is gone.

### An edit as a value

`commit` was always the single door every document change went through. It took
a closure — `(d) => ops.setParam(d, path, key, value)` — and a closure can be
called and nothing else. It now takes a value:

```
{ kind: "setParam", path, key, value }
```

and `applyEdit` is the only thing that knows how to perform one. The history is
a base document and a list of these, and the drawing is the fold of the list
over the base. Seventeen call sites, one dispatcher, and every argument a value
rather than a reference into the document as it was — an edit that read state
outside itself would replay differently depending on what came before it, which
is exactly the property a timeline cannot have.

Three gestures, and they are now different things rather than three names for
undo. Pressing a row moves the mark. Suppressing takes a step out of the middle
and leaves it in the list, struck through. Removing takes it out for good.

### The fold has to be cheap

Replaying from the base on every keystroke is a hundred document clones. So
`cache[i]` is the document after `i` steps and the replay starts at the first
step whose meaning changed — which for the ordinary edit, appended at the end,
is one apply. Suppressing a step in the middle is the case that actually
replays, and it replays only the tail.

### When a step cannot replay

Suppress the edit that added a block and the edit that wired it has nothing to
wire. That is not a bug to prevent; it is what taking a step out of the middle
means, and every parametric CAD tool has the condition. The step is marked, the
fold carries on past it, and it says what it could not find — *"there is no
extra"* — because that is what tells you which earlier row to put back.

The alternative, refusing to suppress anything another step might depend on,
would refuse almost everything.

### What this found in `ops.ts`

Failure is detected by an edit returning the document it was given, and the
module was documented as working that way. Three functions did not. `removeNode`
filtered a node that was not there and returned a clone; so did `disconnect` for
an absent edge, and `moveNodes` wrote positions for paths that had gone. Each
reported success for work it had not done — invisible while the result was only
ever thrown away, and wrong the moment something read it.

### Where the agent's edits go

The live bridge hands over documents, not operations: what the agent did is
expressible as operations, but what arrives over the wire is the result. So it
is a `replaceDoc` step — a real row in the timeline, labelled, jumpable, and
suppressible. An agent's edit can be taken back out of the middle of your work
like any other.

## Nineteenth pass: handing over the drawing

A schematic tool that cannot give you the drawing is missing something obvious.
A figure goes in a paper, a slide, a pull request, and a screenshot is a picture
of the drawing at one zoom on one screen.

`File > Export the sheet as SVG` writes the sheet as a vector. It reads what is
rendered rather than re-deriving it, and the wires are why: `wiring.ts` has
already chosen which of a port's four sides each one leaves by and routed it
around what is in the way, as `<path d="…">`. A second router here would be a
second opinion, and the two would disagree. Only the blocks are drawn again,
because they are HTML and an SVG cannot hold an HTML div without
`foreignObject`, which is a screenshot with extra steps and does not open in
Illustrator.

### Two things it got wrong first

Every text run was centred on its block. A row of three inline runs — *vocab
128256 · D 4096 · 525.3M* — collapsed into one illegible pile. Each run now
goes where the browser put it.

And every block came out with no fill, because "the block" is not one element:
the fill and border are on `.part__body`, the pins are their own marks, and a
container is a `.frame`. Reading the node gave a transparent background, and
nothing *looked* wrong until a block with dark text on a light fill turned up —
attention, invisible against the sheet. It now looks for anything with a
background or a visible border rather than naming classes, which would go stale
the first time the canvas is restyled.

### Somewhere to check it

`bun run scripts/export-svg.ts` drives the same exporter from the same headless
Chrome the screenshots use, and refuses to write a file with no wires in it. A
schematic with no nets is not a schematic, and writing one quietly would be
worse than not writing it.

## Twentieth pass: saying which slab is which

The volume view labelled nothing until you hovered. That was the right call for
the *tensor* names — there are several thousand cubes and drawing all their
names is a wall of text, which is why the reference raises a label only for the
group under the pointer.

It is the wrong call for the stages. A still of the tower said nothing at all
about what any of it was, and hovering is not something a screenshot does.

So the half dozen top-level stages — `embed`, `layers`, `final_norm`, `head` —
are always named, in the left margin at the height of the stage they belong to.
The distinction is the count: six against several thousand, so the reason the
tensor labels are hover-only simply does not apply to these.

### In the margin, not in the model

The first version pinned each name to a point off to the left of the model,
which looked right and then swung across the picture the moment the camera
orbited. A name now goes in the margin at the projected height of its stage,
the way a schematic puts its row names down the side rather than beside
whatever happens to stick out furthest on that row.

Two stages that project to the same height would print over each other, so the
second is dropped: two names in one place is worse than one name. And the
margin is not the whole height — the title sits across the top and the legend
across the bottom left, and a stage name printed over the legend is the same
mistake. No room, no label, which orbiting a little fixes.

### The two items left on the volume view's list, measured

The eighth pass left three things outstanding here. Two of them are no longer
what they were.

**"Activations, not just weights"** is done and has been for a while: the
attention score matrix is drawn at size, the legend has three entries, and
`kind: "i"` blocks are most of what is on screen. The item outlived the work.

**"Instancing — thirty-two identical layers are thirty-two times the
geometry"** rests on a premise the cap removed. `MAX_BLOCKS_DRAWN` is 32 and
`MAX_HEADS_DRAWN` is 6, so the model built for Llama-3-8B, DeepSeek-V3 and
Llama-3.1-405B is the *same* 3,051 blocks and 3,904 arrows — a 100-layer design
is not a hundred layers of geometry, it is thirty-two. Building it takes a
millisecond.

What that measurement does not cover is the draw calls: three thousand meshes
is three thousand of them per frame, and a headless Chrome with the GPU
disabled cannot say what that costs on a real one. So this is left as what it
is — an optimisation whose stated reason has gone, with an unmeasured one that
might replace it, and nobody has reported the view being slow. Instancing a
render path that works, in a file whose conventions are a port of somebody
else's and easy to get subtly wrong, is not a trade worth making on a guess.

## Twenty-first pass: the drawing says what it is

The first three phases of [legibility.md](legibility.md), which is the roadmap
this pass is working through rather than a record of it.

### The middle row was an identifier

Every part on the sheet printed `data.type`: `gqa_attention`, `rmsnorm`,
`gated_mlp`, `lm_head`, `boundary_in`. That is the string the engine dispatches
on and a path is written with, and it was doing duty as the label on a drawing.
Meanwhile the volume view — being a port of somebody else's visualisation —
called the same parts *Token Embed* and *Attention Matrix*. The vocabulary was
already in the repository and one view had it.

`BlockDocs` now carries a `Name`: a short noun phrase in the words a published
figure would use, one per catalog entry, forty-five of them. The sheet prints
the name and the inspector leads with it and carries the identifier beside it,
because the inspector is where somebody who found a block by its drawing goes to
learn what to type. A test fails if any block lacks one — the failure is
otherwise invisible, since the fallback is the identifier and the identifier is
what was there before.

The type row also stopped being set in the monospace face. A name in a monospace
face still reads as code, which was most of what was wrong with it.

### The catalog's prose had nowhere to go

Forty-odd summaries, written for M6, crossing the boundary on every catalog
entry, and the only way to read one was to select the block and look at the
inspector. Hovering a part now raises a card with its name, its identifier, what
it is and the formula it counts by.

It went in twice. The first version wrapped each part in the house `Tooltip`,
which is Base UI underneath, and that could not be verified: Base UI's hover
detection does not fire under synthetic pointer events, so the automation could
not tell a working tooltip from a broken one — and the toolbar's existing
tooltips did not open either, which is what proved it was the harness. It is now
one card owned by the canvas, positioned from React Flow's own
`onNodeMouseEnter`. That is better for three other reasons: a dense sheet is a
few hundred parts and this is one floating element rather than several hundred,
it cannot interfere with dragging, and it clears itself on a drag or a pan
instead of hanging over a drawing that has moved underneath it.

### The key

`B T (H dh)` is not gibberish to somebody who has been told what the four
letters are. Nothing had ever told anyone.

A key sits on the sheet, open by default and shut to a tab, remembered like the
grid and the title block. Two of its five sections are the design describing
itself — the symbol table with its documentation strings, which `derive()`
already returns — and the marks are drawn from the same custom properties the
canvas draws with, for the reason the callouts are derived: a legend that
restates the stylesheet in its own numbers goes quietly wrong the first time the
canvas is restyled. Only the sentence about the shape grammar is authored.

### What the design is

`meta.notes` — a paragraph per preset, saying that nanoGPT's vocabulary padding
is a speed decision rather than a modelling one and costs 36,096 parameters —
was in every file and rendered nowhere. It is now under the name in the title
block, clamped to two lines, and in full in a new library dialog: twenty-three
designs grouped by family with their published counts and their sources, so
choosing between `qwen3-30b-a3b` and `qwen3-next-80b-a3b` no longer means
loading both and looking.

Six presets had no notes at all. They have them now. One draft of them was
wrong — it said Llama 3 70B writes its feed-forward width out because the 8B's
rounding rule does not produce it, and `ceil_mult(1.3 * 8/3 * 8192, 1024)` is
exactly 28,672. The rule breaks at 405B, not at 70B, so the fact moved to the
preset it is true of. Both cache figures in that note were checked against the
engine rather than worked out by hand.

### Four bugs the work walked into

All four are the same bug, and it is invariant 1: resolve a block through
`catalogOf(doc)`, never the bare `CATALOG`. The built-in catalog is filled once
at load, so against it a design's *own* block does not exist — and every one of
these then failed silently, returning a value the same shape as a real answer.

- **`newNodeFor` returned `null`.** That is how the drop handler is told there
  is no such block, so a block the palette listed could be dragged onto the
  sheet and simply not appear.
- **`isDrillable` returned false**, so a design's own composite could not be
  opened — which is the kind of block somebody most wants to open.
- **`kindOf` returned `"unknown"`**, so it drew as an unknown category.
- **`flatItems`, the inspector and the tensor panel** all resolved against the
  built-ins, so the same block had no parameter summary and no documentation.

`kindOf` and `isDrillable` now take the definition rather than a type to look
up, which is the fix rather than a patch: looking a type up means choosing a
catalog, and the whole failure was choosing the wrong one at a distance from
where the block had already been resolved. Every caller had one in hand.
`newNodeFor` genuinely has only a name — it is answering a click in the palette
— so it takes the document. `packages/ui/test/user-block.test.ts` pins all of
it; three of its five assertions failed before the change.

A fifth, found the same way: a frame that had merged with the one inside it
showed the inner block's *type* with the outer block's *name*. Three parallel
`merged*` fields were being copied by hand at two hops, and the third one added
was missed at one of them. They are one `mergedDef` now.

### Still outstanding

L4 and L6 through L8 of [legibility.md](legibility.md): shapes written in
English, a model small enough to see every number of, the plumbing out of the
drawing, and the walkthrough.

## Twenty-second pass: the rest of M7

L4 and L6 through L8 of [legibility.md](legibility.md), which finishes it.

### A model small enough to see

The editor opened on Llama 3 8B. Opening a CAD tool on the hardest thing it can
draw costs a reader who has never seen it everything and costs somebody who
loads a preset in the first five seconds nothing, so the default is now
`nano-sort`: three layers of width 48, three heads of 16, a vocabulary of three.
It is the model in Karpathy's minGPT sorting demo and in the reference
visualisation, and its structure is GPT-2's exactly — so nothing about it is a
special case, it is just small enough that every weight fits on the screen.

Its published figure is 85,728, which is a number this repository computed and
then checked against PyTorch rather than one a vendor stated. A first attempt at
it by hand gave 64,992: the down-projection was worked out at the wrong width.
The engine was right and the arithmetic in my head was not, which is the whole
argument for having the engine.

The last preset loaded is remembered, so only a first visit lands on the toy.
That is a name in `localStorage`, not a document — an edited design is the
storage provider's job (invariant 9) and this is one line.

**And a bug in the verifier.** `tensorcad-runtime verify` runs its forward pass
at a fixed sequence of 128. A design whose context is 11 has a position table
with 11 rows, so the pass raised `index out of range in self` — which reads as a
fault in the generated model rather than as the default being longer than the
model. The export phase had consulted `Tmax` all along; the forward pass now
does too, and says when it shortened.

### Shapes in English

A third setting beside symbolic and numeric. `B T D` becomes
`1 batch × 8,192 tokens × 4,096`, and `B T H*dh` becomes
`1 batch × 8,192 tokens × 4,096 (32 heads × 128)`.

What an axis *counts* is a fact about the design, not about the shape: `T` is
tokens in a language model, patches in a vision transformer and one image in a
convnet, and every design already says so in its own symbol table. So the noun
comes from the symbol's documentation — and getting that right took three
attempts, each one found by rendering all twenty-four presets and reading them.

- **A width is not a count of the thing it is a width of.** `dh` is documented
  "Head dimension", so the first version rendered it `128 heads`. Not merely
  unhelpful — false, on every attention block of every design. Anything whose
  documentation says *width*, *dimension*, *resolution* or *rank* gets no noun.
- **The earliest word in the sentence wins, not the first row of the table.**
  Gemma 3's `G` is "Groups of eight layers" and its `Ltail` is "Windowed layers
  past the last whole group". A fixed table order had to call one of them wrong;
  which word the sentence leads with tells them apart.
- **A count of what is inside one of a thing is not a count of the thing.**
  I-JEPA's `P` is "Values in one patch": 588 values, one patch.

One preset's prose was the problem rather than the rule — `nano-sort`'s
vocabulary said "the three tokens it sorts" and so read as a count of tokens. It
says "three symbols, A, B and C" now, which is better in the symbols panel too.

A reshape gets a sentence rather than an einops pattern: `B T (H dh) → B H T dh`
becomes *split 4,096 into 32 heads of 128*. Only the two reshapes this catalog
actually produces are named; anything else prints its pattern, because a reshape
nobody can name is better as notation than as a sentence that might be
describing something else.

### The plumbing out of the drawing

Figure mode, on `g`. A reshape is real, necessary, and the thing that makes
multi-head attention multi-head — and no published figure draws one, because it
moves no data and costs no parameters. So the mode leaves them out and traces
the wire straight through: `q_proj → q_heads → rope_q` becomes
`q_proj → rope_q`.

The risk was named before it was built: a view that hides a block is a view that
can lose a design-rule finding. `unfold` returns what each hidden block was
absorbed into, and the canvas shows an absorbed block's findings on whatever
absorbed it, with the hidden block's name in the message so the marker is
honestly pointing at a consequence rather than a cause.

The set of what counts as plumbing is named rather than inferred, and pinned by
a test that every name in it is a real catalog type of the category it claims.
That is the failure `SIDEWAYS_IN` had in the seventeenth pass: two of its five
rows matched no block type at all and had never done anything.

### The walkthrough

The reference has ten hand-written phases against one model. Twenty-four presets
cannot each have ten and do not need to: what differs between them is which
*kinds* of stage they have. An embedding is an embedding in a 85,728-parameter
sorter and in a 671B mixture of experts, and one sentence explains both once its
numbers are filled in.

So the steps are derived from the blocks the design actually contains, in flow
order, and the prose is authored once per kind. `alexnet` gets a convolution
step and no attention step; `nemotron-h-8b` gets a state-space one; a dense
model is told its feed-forward widens the vector and a sparse one is told a
router picks two of eight experts. A step exists because the design has the
block it is about, and if it does not, the step is not there to be wrong.

That is also the thing a recorded explanation cannot do. Change `D` from 768 to
1536 and the walkthrough changes with it, because every number in it was read
out of the design rather than typed — which a test asserts directly.

It is a column beside the drawing rather than a dialog over it, and the step
lights the blocks it names and dims everything else. A modal would cover exactly
what it was describing. The key shuts to its tab while a walkthrough runs,
without touching the stored preference, because two panels over one sheet is one
too many and a walkthrough is the guided version of the same job.

### What the tests hold

Four new files, and each one is about a way this could be quietly wrong rather
than about a happy path: that a block a design defines for itself can be placed,
drawn and opened; that every shape of every preset renders without a `NaN` or a
leftover symbol; that figure mode strands no wire and loses no finding; and that
the walkthrough names no block that does not exist, prints no empty paragraph,
and says something different when the design changes.

### Still outstanding

The volume view still names its stages from a port of somebody else's layout
rather than from the catalog's names, so the same block is *Token Embed* there
and *token embedding* on the sheet. Now that a name is a fact the catalog
carries, the port should read it.

## Twenty-third pass: five silent no-ops

M7 shipped and then went looking for what it had missed. Every one of these is
the same shape as the bugs the twenty-first pass found, which is the shape worth
learning: something reports success and puts nothing on screen.

- **A walkthrough started below the top level dimmed the whole sheet.** The
  steps name blocks of the whole design; drilled into `layers`, not one of them
  matched anything drawn, so every part was dimmed and none was lit — a drawing
  saying the step was about nothing. Opening one now goes back to the top,
  which is where the thing it narrates is.
- **A walkthrough of an empty design did nothing and said nothing.** No blocks,
  no steps, a panel that renders null, and a command reporting itself as on. It
  says why now.
- **The breadcrumb still printed the identifier.** `repeat`, where the sheet,
  the tree and the inspector all said "stack". The twenty-first pass reached
  every other surface and missed this one.
- **The volume view labelled its stages with node ids** — `embed`, `layers`,
  `final_norm` — beside a sheet that called the same blocks "token embedding",
  "stack" and "layer norm". One design, two vocabularies. It reads the catalog
  now; the id stays in `path`, which is what cross-probing uses.
- **The MCP never learned the name.** An agent knew `gqa_attention` while the
  human's screen said "grouped-query attention", which is half of what M7 was
  for: the point of naming a block is that everybody uses the same word for it.
  `catalogEntry` carries it, `catalogText` prints it, and `search_catalog`
  matches on it.

And one thing that was simply wasteful: `resolveLevel` called `catalogOf(doc)`
once per path segment. Once now.

### Still outstanding

`PortSpec.dtype` has been declared since the seventeenth pass and nothing checks
it. An integer tensor arriving at a float matmul is a real design error that
shape inference cannot catch, because the shapes agree perfectly — the
declaration was added to make the check possible and the check was never
written. It is a design rule and it belongs beside the other eighteen.

## Twenty-fourth pass: the element types, and picking things

Two lists, both of them old, both of them finished here.

### The check the declaration was added for

`PortSpec.Dtype` arrived in the seventeenth pass with a stated purpose — "an
integer tensor arriving at a float matmul is a bug shape inference cannot catch,
because the shapes agree perfectly" — and the check was never written. By the
time it was looked at again, **exactly one port in the whole catalog declared
anything**, and the renderer was still deciding what a wire carried from a
two-row table keyed on `node.type` strings. That is what a declaration with no
consumer decays to: not wrong, just inert, and quietly replaced by the thing it
was supposed to replace.

So both halves went in together.

**A port declares a kind, not a width.** "real" for anything a matmul can
multiply, "int" for an index, "bool" for a mask. Which real type — fp32, bf16,
fp8 — is a condition of the run and belongs to the operating point (invariant
8); a block has no business declaring it. The blocks that multiply say they take
numbers to multiply, and `embedding` is the one that changes class — an index
in, a vector out — so both ends of it say so. Without that, `int` would
propagate through the entire model.

**The rule walks back through `inherit`** to find what a pin actually carries,
and is deliberately timid: one input it cannot resolve makes the whole block
unknown rather than a guess. A rule that fires on a shipped design is a rule
that teaches people to read past it, so a test holds it silent on all
twenty-four presets. It found a real one on the first run, in a broken-design
fixture that had been wiring an `int64` input into a linear since it was
written.

**And the renderer reads the declaration.** Twenty pins carry a colour on the
default sheet where two did, and `declared.startsWith("int")` — the sniffing the
seventeenth pass added the declaration to be rid of — is gone.

### Three things from KiCad

The sixteenth pass listed them and did none of them.

**A box selection whose direction changes its meaning.** Left-to-right takes
only what it fully encloses; right-to-left takes anything it touches. React Flow
has this as a fixed prop rather than a gesture, and the prop is read on every
move while the box is open — so tracking which way the pointer has gone since it
went down is enough. The box is drawn differently for each: solid and accented
for enclosing, dashed and amber for touching. A mode you cannot see is a mode
you cannot trust, and this one changes what the gesture means. It resets when
the drag ends, or the sheet goes on claiming a mode with nothing being dragged.

**Clarify selection.** Alt and a press, and what is under the cursor is listed
rather than guessed at — on a dense sheet a click is ambiguous far more often
than a tool admits, and the only other way to reach the thing underneath is to
move the drawing. The hit test is `elementsFromPoint` rather than geometry: the
browser already knows what is stacked at a point, and a second opinion computed
from node boxes would disagree with what the eye sees. Frames are skipped, for
the reason the sixteenth pass gave — a frame is picked by its outline and its
interior belongs to what it holds.

Deliberately not a long press. A long press on a block is how a touch device
starts a drag, and a gesture that means two things on two devices means neither.

**Select connection**, on `Shift+G`: add everything directly wired to what is
selected, and press again to go further. One junction at a time rather than the
whole net, because the common case is "this block and what feeds it" and
swallowing the net in one press makes that unreachable. The primary selection
stays primary, so the inspector does not jump to some other block because the
selection grew around it.

### What is left, and what was already done

Several entries on the standing lists had outlived their work. E7 — operations
as first-class objects — was delivered by the eighteenth pass, which said so at
the time. A visual block editor is the definition editor, thirteenth and
sixteenth passes.

Two are real and neither is a defect. **Instancing** in the volume view: the
twentieth pass measured it, found the premise gone — the block cap means a
100-layer design is thirty-two layers of geometry — and left it with an
unmeasured reason that might replace it. That reasoning still holds, and nobody
has reported the view being slow. **Real values in the cells** rather than the
speckle needs trained weights, which a design tool does not have; it is a
feature about loading checkpoints, not a gap in this one.

## Twenty-fifth pass: one bar, and a way to share

The first thing a deployment with accounts showed was two top bars. Its own
strip said the product's name and Sign in; the editor's toolbar under it said
the product's name again. The editor had nowhere for an account to go, so the
deployment built a place above it, and every page paid a row of height and a
duplicated name for it.

And sharing, which is the first thing anybody asks for once they have drawn
something, was four steps into a tab: sign in, open Designs, save, press share
on the row. In a plain checkout — the public editor — it was not there at all.

### The account goes in the editor's corner

The storage seam already carried who is signed in, for the Designs panel. It
now carries two more things a provider may offer: `signIn()`, which the editor
calls and the deployment answers with whatever signing in looks like there,
and `signOut()`. An account can also carry a `problem`, a line for its menu.
The toolbar's right-hand corner renders from that: Sign in when nobody is and
the provider can ask, a name and a menu when somebody is, and nothing at all
when there is no provider. The deployment's bar goes. The seam still names no
vendor and no host: the corner shows what the provider reports and cannot tell
where it came from.

### Share, where it is looked for

A Share button beside the account, one press. Signed in to a store that shares,
it saves the design and copies the store's link, short and view-only. Anywhere
else — the public editor, a plain checkout, somebody not signed in — the design
goes in the link: serialized, deflated by the browser, and written into the
fragment as `#design=…`. A fragment never leaves the browser, so nothing is
uploaded and no server is involved, and the editor reads it back when the page
loads, opens it and takes it off the address bar. Nemotron-H's fifty-two layers
written out, the longest preset, is under sixteen thousand characters; a
typical design is two or three thousand.

The dialog says which kind of link it made, because they behave differently in
the recipient's hands: a store's link shows the saved design, and a link that
carries the design gives them a copy of their own to change.

### What the tests hold

Every preset's link comes back byte for byte, deflated and plain; a link this
editor did not make, or one a browser without deflate cannot read, is refused
with a reason; opening one works with no provider registered and clears the
address bar; and Share picks the store only when somebody is signed in to one
that shares, saving the design first. In the browser, a link made by Share in
one tab opened the same design in a fresh one.

## Twenty-sixth pass: plain names

Select a transformer block and the inspector listed forty-two fields, each
headed by its code name and its type: `ffn_hidden int`, `kv_lora int`,
`shared_expert_gate bool`. Twelve of them meant nothing for the block as it
was set — `experts` on a dense one — and were greyed with "unused" beside them.
Another eighteen were things almost no design touches: attention sinks, a score
cap, an expression mask. Somebody opening the editor for the first time met a
forty-two-field form for what is a fifteen-field decision, in a vocabulary they
had to learn first.

### A label, and the name beside it

Every built-in parameter now has a label, and the inspector leads with it:
**Feed-forward width**, with `ffn_hidden` small beside it. The name stays on
screen because it is still what a document, a path and an MCP call write, and
the person who needs to type it should not have to go looking. The type is gone
from the heading; the control already says it.

The labels are the engine's, not the editor's, in `catalog/labels.go`. They are
keyed by the parameter's name, because a name means the same thing on nearly
every block that has it — `heads` is heads on seventeen of the eighteen blocks
that have one — with an override
for the few names that do not: `count` is **Repeats** on a stack and **Vectors**
on learned tokens, `by` is **Multiply by** on a scale and **Tokens ahead** on a
multi-token head. Keeping them in the engine is what lets `explain` carry them,
so an agent reading a block gets the same words a person does. A test fails for
a built-in parameter without a label, and for two fields on one block that read
alike.

Enums read as words too, where the words are the ones a paper uses: the
attention kinds are *grouped-query*, *latent (MLA)* and *differential*, the
nonlinearities *SiLU*, *GELU* and *squared ReLU*. A switch says *on* and *off*,
and a switch whose default belongs to another field says *default*.

### Three piles

The fields a block is made of are shown. The rare ones are under **Advanced**,
which starts closed and opens by itself on a block that has changed one of them,
so a design with sinks shows its sinks without a click; only a field with a
default can be advanced, since a field a block cannot do without is not one to
hide. The ones that do not apply are no longer greyed in place but hidden behind
a count at the bottom — *9 more that don't apply* — which puts them back,
greyed, where they were, and says in the other field's words when they would:
*only used when feed-forward is mixture of experts*. The earlier pass kept them on screen because a field
that vanishes when `mlp` changes is one you go looking for; the count is what
says where it went.

A transformer block now opens on fifteen fields.

### The palette says the same

Its headings were the engine's category keys, `ssm`, `moe`, `io`, in the order
the catalog happened to list them. They are words now — *State space*,
*Mixture of experts*, *Inputs and outputs* — in the order a design is built: whole
layers first, then what a layer is made of, then the plumbing. A search also
matches what a block's parameters are called, so *experts per token* finds the
blocks that have one.

### What is left

Clicking a block drawn inside an unfolded frame selects it, and the inspector
then says it is not on this level. That is true, and it is a poor answer to
somebody who clicked the thing they wanted to know about. It belongs to the
next pass, with the first screen.

## Twenty-seventh pass: a calm first screen

What somebody saw on a first visit, at 1440 by 900: the key open over the left
half of the drawing, the title block over the bottom right, the minimap in the
corner, sixteen controls for the operating point, and six tabs. At 1280 by 800,
a thirteen-inch laptop, the drawing was almost entirely covered and the toolbar
ran off the right edge, taking the Share button with it. Every one of those
panels had been added for a reason, and the reasons were good; what nobody had
done was open the editor for the first time after all of them.

### Nothing over the drawing

The key starts shut to its tab, which stays on the sheet: the reader who does
not know what a dotted line means still has a button that says Key. The title
block and the minimap start off, since every figure on the title block is in the
readout beside the sheet and the drawing fits the window when it opens. All three
are one keypress or one menu item away, and remember being turned on.

### Four fields, and More

Batch, sequence, device and GPUs are what a first question about a design turns
on. Precision, the optimizer, recomputation, ZeRO, the three parallel degrees,
the serving streams and the three switches are how a training run is set up,
and they are under **More**. Shut, its line says what is inside it —
*precision, optimizer, parallelism* — until something there differs from the
default, and then it says that instead, in the accent colour: *ZeRO 3 · TP 2*.
Pressing a cluster plan sets several of them at once, and tidying them away must
not hide that the numbers now assume it.

### Four tabs

Cluster, the width ladder and Runs are one tab, **Training**, with the three as
a switch inside it. They answer one question — what happens when this is
trained — and six tabs across a column four hundred pixels wide read as six
things to learn before starting. The tab opens on whichever of the three was
open last, and each still has its own command and shortcut.

### Start here

With nothing selected the inspector used to say "Select a block on the canvas to
edit its parameters", which is true and is the whole of what it said. It now
says what there is to do — click any block to see what it is — and offers the
walkthrough with a **Start here** button. The same command is the first item
on the Help menu.

### A block where it is

The last pass left this open. Clicking a block drawn inside an unfolded frame
selected it, and the inspector said it was not on this level. It now finds the
level the block lives on and shows it there. Inside a stack that level is
editable, so the block can be edited in place. Inside a built-in block's own
expansion it is read-only, and the inspector says so: *part of block, a built-in
transformer block, so it is read-only*. An **Open its level** link goes to where
it lives, with the block still selected.

### The toolbar at 1280

The product name and the per-token FLOPs cell both appeared at Tailwind's `xl`,
which is exactly 1280, and together they pushed Share off a laptop's screen. They
now appear at 1360 and 1520. The FLOPs figure is in the readout either way.

### What the tests hold

The browser test opens a fresh profile and checks the first visit:
- nothing covers the sheet;
- the operating point is four fields;
- the tabs are the four;
- Start here opens the walkthrough;
- at 1280 wide, Share ends inside the window.

It then checks two interactions:
- More names a ZeRO stage set under it;
- a grouped-query attention clicked inside an unfolded transformer block is
  inspected in place, read-only.

The README screenshots are regenerated. They now open Llama-3-8B from a
`#design=` link rather than by driving the preset menu.

## Twenty-eighth pass: a design of your own

File ▸ New design made an empty sheet. For somebody who knows what a
transformer is, and has not yet learned which of fifty-five blocks to drag and
how to wire them, that is where the tool stops. The other way in was a preset,
which is somebody else's model at somebody else's size: Llama 3 at eight
billion, when what they have is one 24 GB card.

### Two questions

What somebody starting out knows is the kind of model they want and roughly how
big. So those are the two questions. There are six kinds, each named and
described in a sentence, and each size is a chip. You can also type a size of
your own, or ask for *as large as trains on one* GPU.

The answer is the kind's reference design scaled to the size. It is a preset,
so it is something that was actually built and is held to its published count.
Nothing is created until the dialog has said what the design came to: the
parameters, the layers and width and heads, and whether training fits on one
device, measured by the same analysis the readout will use a moment later.

### What scaling a real model involves

Scaling by the cube root is the bench's rule, and it was not enough for a
design somebody will keep:

- **Llama 3's four query heads per key head became thirteen over one** at a
  billion parameters, because the width was rounded to whole heads. The head
  count now moves to a neighbour the grouping divides, and the depth takes up
  the size.
- **A 125M Llama-style design came out three layers deep**, because 128,000
  vocabulary entries each way were most of it. It now ties its head, as Llama
  3.2's small models do.
- **A mixture of experts did not fit on any single card at any size.** It had
  been measured at Qwen3's 32,768-token serving context. It now trains at 4,096.

### And the label on the frame

The stack's frame on the sheet is labelled the way a figure labels it,
*Transformer block x32*. The label is text, so a design scaled to seven layers
went on saying thirty-two. The bench scaler had the same fault. `scale` now
rewrites a label that ends in the count.

## Twenty-ninth pass: a drawing you can read when it opens

A sheet opened fitted to its window with a floor of 60% zoom, so that it would
never open "at a zoom where the symbols cannot be read". At 60% a part's name is
eight pixels and its type and summary six. The floor was named for legibility
and set below it. And the summary, where it could be made out, said
`D 48 · dh 16 · ffn 192`: the design's code names, which the inspector had
stopped leading with a pass earlier.

### A floor that is readable

A part's type line is 10.5 pixels at full size and its summary 10, and below
about nine on screen they stop being words. That puts the floor at 0.85. A
drawing that fits at 0.85 or better opens whole and centred. One that doesn't
opens at 0.85 from its top, and from its left if it is wider than the pane.
That is where a model is read from, and its input at a legible size beats its
middle as a thumbnail. Pressing `f` still fits the whole drawing, however small
that makes it, because somebody pressing it wants the whole.

The arithmetic is `standFor` in `canvas/viewport.ts`, with a test. It was the
easy part. The first attempt opened on an empty corner: the fit ran on the
render the layout arrived in, and read every part still stacked at the origin.
The canvas copies laid-out nodes into its state a render later, and React Flow
adopts that array a render after that. React Flow's own fit queues itself
until then, which is why the old one never showed it. This one now waits the
same way, until the nodes React Flow holds carry the layout's positions.

The margins kept for the title block only apply when the title block is shown,
which since the twenty-seventh pass is not by default.

### A phrase, not code names

Each common block now says what it is the way a figure annotates one:
- *128,256 × 4,096* on an embedding;
- *4,096 → 128,256 · tied* on an output projection;
- *32 heads × 128 · 8 kv* on an attention;
- *14,336 wide · SiLU* on a feed-forward;
- *256 experts · top 8* on a mixture of experts.

Enums are in the catalog's words. A block with no phrase of its own is described
by its parameters' labels.

The line is 10-pixel monospace, about six pixels a character, so it holds about
twenty-one characters beside a parameter count and twenty-eight without one.
Measured on DeepSeek-V3 and Jamba, five summaries ran into an ellipsis. So a
phrase now lists its parts in order of importance and drops the ones that don't
fit, never cutting one in half. *32 heads × 128 · 8 kv · causal* on a part
with parameters loses *causal*, not *8 kv*.

## Thirtieth pass: what a change did

The editor has kept the design as it was opened since the Compare dialog was
written. The dialog diffs against it, structure and numbers together, measured
at one operating point. It sits under View, two clicks deep and behind a
choice of what to compare with, and nobody trying "what if this had one key
head" goes there. They change the value and look at the parameter count. The
parameter count moves, and says nothing about the cache, which is the reason
to make that change.

### A strip under the number

Once the design differs from where it started, a strip appears under the
parameter count. It says what moved, and what that did to the four numbers a
change is usually made for:

> **Since you opened it** · key/value heads 3 → 1
> parameters −11% · compute −11% · cache −67% · memory/GPU −11%

What moved is said the way the rest of the editor says it:
- a symbol by its own description, when the design gives it a short one, so
  *key/value heads* rather than `Hkv`;
- a block's parameter by the catalog's label, with an enum in the catalog's
  words, so *block: activation GELU → ReLU*;
- a block added or removed by what it is.

The first two changes are listed, and the rest are counted.

The percentages are signed and not coloured good or bad. A model made larger
on purpose is not a warning, and the sign is all the direction there is.

**Compare…** opens the full diff. **Compare from here** makes the design as it
stands the new baseline, and the strip goes until something moves again.
Opening a design, loading a preset or making one with New design sets the
baseline too, which the store already did. Undoing back to the start makes the
diff identical, and the strip goes with it.

### Measured when you stop

A diff analyses both designs. The readout has already analysed one of them,
and a third analysis on every keystroke would be felt on a large design. So the
strip is measured 200 milliseconds after the last edit, and dims while it
waits rather than showing numbers that belong to the previous one.
