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
  screen above a draggable divider; Inspector, Symbols and Rules share the pane
  below it. Tabs now cover only the three things you edit one at a time.
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
