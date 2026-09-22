# Legibility: making the ported models readable

A roadmap, written before the work rather than after it, so it does not belong
with the twenty passes in [interaction-design.md](interaction-design.md). Those
record what was changed and why. This records what is wrong and what to do about
it, and it will be folded into that file a pass at a time as each phase lands.

The complaint that started it: Brendan Bycroft's LLM visualisation is clean and
easy to read, and this is mathematical gibberish. That is fair, and the rest of
this document is an attempt to say precisely where it is true, which parts of the
fix we already own and are not using, and what order to do the work in.

## What the tool actually says

Read out of the running editor, not from memory. Opening the default document —
which is `llama-3-8b`, an eight-billion-parameter model, because `store.ts:351`
says `getPreset("llama-3-8b")` — at the default detail of two levels, every part
on the sheet reads:

```
tokens                 input               B T · int64
embed                  embedding           vocab 128256 · D 4096      525.3M
Transformer block x32   transformer_block                              6.98B   32×
norm1                  rmsnorm             D 4096                     131.1K
attn                   gqa_attention       heads 32 · kv 8 · dh 128     1.34B
norm2                  rmsnorm             D 4096                     131.1K
mlp                    gated_mlp           ffn 14336 · silu             5.64B
final_norm             rmsnorm             D 4096                       4.1K
head                   lm_head             vocab 128256 · D 4096      525.3M
logits                 output
```

Open it to five and the attention block becomes:

```
q_proj     linear       in 4096 · out 4096            536.9M
k_proj     linear       in 4096 · out 1024            134.2M
v_proj     linear       in 4096 · out 1024            134.2M
q_heads    rearrange    B T (H dh) → B H T dh
k_heads    rearrange    B T (Hkv dh) → B Hkv T dh
rope_q     rope         heads 32 · dh 128 · theta 500000
attn       sdpa         heads 32 · kv 8 · dh 128
o_merge    rearrange    B H T dh → B T (H dh)
o_proj     linear       in 4096 · out 4096            536.9M
```

Three separate things are wrong with that, and they need separate fixes.

**The middle row is a machine identifier.** `BlockNode.tsx` renders
`data.type` — `gqa_attention`, `rmsnorm`, `gated_mlp`, `lm_head`,
`boundary_in` — because `BlockDef` in `catalog/types.go` has a `Type` and a
`Docs.Summary` and nothing between them. There is no short human name for a
block anywhere in the engine. The identifier is the right thing to type into an
MCP call and the wrong thing to print on a drawing.

**The top row is a variable name.** `q_proj`, `o_merge`, `rope_k`, `norm1`.
These come from the composite expansions and the preset files, and they read
like the inside of a `state_dict`, which is what they were taken from.

**The value row is shorthand with no key.** `dh`, `kv`, `ffn`, `B T (H dh)`.
Every one of these is defined somewhere — `doc.symbols` carries a `doc` string
for each, and the Symbols panel shows them — but nothing on the sheet says so,
and `B T (H dh) → B H T dh` on a box labelled `o_merge` is the single densest
thing the tool draws.

Set against that, the same model in the volume view names its parts **Token
Embed**, **Position Embed**, **Attention Matrix**, **Projection Weights**,
**Attention Output**, **LN Agg: μ, σ**. Those are Bycroft's names, carried over
in the port in `three/model3d.ts`. So the tool already contains a good
vocabulary for a transformer; it lives in one view, hard-coded, and the sheet —
the surface almost all the work happens on — cannot see it.

## What Bycroft is actually doing

Worth being precise, because "it looks cleaner" is not a plan.

1. **A toy model, not a real one.** The default is nano-gpt: three layers,
   48-wide, a vocabulary of three letters, trained to sort six of them. His own
   notes say it is "several orders of magnitude smaller than serious LLMs, but a
   lot easier to digest," and that its "structure and function are identical to
   that of GPT-2." Every number on screen is a number you can see all of.
2. **Narration, in ten ordered phases.** `Walkthrough00_Intro` through
   `Walkthrough09_Output`: intro, preliminaries, embedding, layer norm,
   self-attention, softmax, projection, MLP, transformer, output. Each is prose
   written in a small DSL where `c_blockRef` and `c_dimRef` tie a phrase in the
   text to a specific object in the 3D scene, and `dimHighlightBlocks` dims
   everything the current sentence is not about.
3. **Inference only, stated up front.** The scope is narrowed deliberately and
   the narrowing is announced.
4. **English names on every part.** "multi-head, causal self-attention", not
   `sdpa`.

The third and fourth are cheap. The first is a file. The second is the one that
actually produces the feeling of "clean and easy to read", and it is the most
work.

## Four things we own and do not use

Before adding anything, this is what is already built and not reaching a reader.

**`docs.summary` on every catalog block.** Forty-odd sentences, already written
and already crossing the WebAssembly boundary — "Grouped-query attention.
kv_heads = heads gives multi-head attention, kv_heads = 1 gives multi-query.",
"Gated feed-forward network (SwiGLU when act is silu, GeGLU when gelu).",
"Marks the compressed vector that latent attention caches instead of keys and
values." Only the Inspector renders them, and only when a block is selected. The
part body's `title` attribute says `double-click to open this block` and nothing
else.

**`meta.notes` on every preset.** One paragraph per ported model, in the file,
rendered nowhere — `grep` for `.notes` in `packages/ui` finds the analysis,
memory, throughput, cluster and ladder notes and not this one. What is being
thrown away, for `nanogpt`:

> Karpathy's nanoGPT at its defaults: GPT-2 small, with the vocabulary padded
> from 50,257 up to 50,304 so it divides by 64. That padding is a speed decision
> rather than a modelling one and it costs 36,096 parameters, which is the whole
> point of having it beside gpt2-small.

That is exactly what a reader wants and it has never been on screen.

**`explain()`.** Its package comment calls itself "the learning surface": every
parameter as written and as evaluated, the shapes on the pins, the block's share
of the model's parameters and FLOPs, the breakdown into primitives largest
first, and the docs. The MCP server and the CLI use it. The editor does not —
the Inspector re-derives a subset.

**The callouts.** `canvas/callouts.ts` already writes the right register:
"Embedding dimension of 4096", "32 query heads, 8 key/value heads",
"Intermediate size: 4096 = 14336", "only the latent is cached". Twelve block
types get one. This is the proof that the house style for explaining a part in a
sentence exists; it is simply not applied to the parts themselves.

## The roadmap

Ordered by value per unit of work, not by dependency, though the dependencies
happen to agree. Estimates follow the convention in `ROADMAP.md`: one developer
with an AI assistant, part-time, ranges rather than commitments.

### L1 — A name for every block · **done**

Add `Name` to `BlockDocs`: a lower-case noun phrase, one line per catalog entry.
`gqa_attention` → "grouped-query attention". `rmsnorm` → "RMS norm".
`gated_mlp` → "gated feed-forward". `lm_head` → "output projection". `sdpa` →
"attention". `rope` → "rotary positions". `rearrange` → "reshape".
`boundary_in` → "in". Roughly fifty of them, and each is a judgement call worth
about thirty seconds.

The sheet's middle row shows the name. The identifier does not disappear — it
moves to the Inspector header and the status bar, because it is what a path
contains and what an MCP call names, and a reader who finds a block by its
drawing and then wants to talk to an agent about it needs the translation to be
one glance away.

Everything downstream follows for free: the model tree, the palette, the rule
findings, the compare dialog and `explain()` all print the type today.

**Done.** The depth-two `llama-3-8b` sheet reads *token embedding*, *RMS
norm*, *grouped-query attention*, *gated feed-forward*, *output projection*;
the depth-five one reads *scaled dot-product attention*, *rotary positions* and
*reshape* where it read `sdpa`, `rope` and `rearrange`. Forty-five names, a test
that fails if a block lacks one, and the identifier kept in the inspector and on
hover.

### L2 — The summary reaches the drawing · **done**

Hovering a part shows its `docs.summary`. Not in a `title` attribute — a
sentence of forty words in a native tooltip appears after a second and wraps
badly. A hover card, the same one the Inspector uses, with the summary, the
formula and the source link.

**Done**, with one change of approach. The card is owned by the canvas and
positioned from React Flow's own `onNodeMouseEnter`, rather than a floating
element per part: one instance instead of several hundred, nothing in the way of
a drag, and it clears itself when the drawing moves. The sources stayed in the
inspector — a card that closes when the pointer leaves it cannot hold a link.

### L3 — A notation key · **done**

A legend, dismissible, folded by default, docked to the sheet. Three parts, and
two of them are derived rather than authored:

- **The symbols**, from `doc.symbols` — each name, its value under the active
  configuration, and its `doc` string. This is already computed for the Symbols
  panel; it belongs on the drawing, where `D 4096` is written.
- **The line types and the marks**, from `canvas/wiring.ts` — solid for the main
  path, dashed for a line that skips a stage, dotted for indices rather than
  activations; a filled junction dot, a hollow unwired pin, the pin colours by
  element type; `⊕` and `⊗`.
- **The shape grammar**, authored once: that a shape is written
  `B T D`, that brackets mean one axis folded out of several, that `B` and `T`
  stay symbolic because they are conditions of a run rather than properties of
  the design.

This is the cheapest fix for the densest complaint. `B T (H dh)` is not gibberish
to someone who has been told what the four letters are; it is gibberish to
everyone else, and the tool has never told anyone.

**Done.** Open by default, shut to a tab, remembered, and on `k`. Five
sections: the symbols with their documentation and values, the shape grammar,
the three line types, the marks, and the pin colours. Only the grammar is
authored.

### L4 — Shapes in English · **done**

A third setting beside the existing symbolic/numeric toggle. `B T (H dh)`
becomes `batch × 8192 tokens × 4096 (32 heads of 128)`. `B H T dh → B T (H dh)`
becomes `fold the 32 heads back into the stream`.

The second of those is the interesting one and it is not a general
pretty-printer: a reshape is only explicable because we know what the axes
*mean*, and the honest scope is the dozen patterns the catalog actually
produces, with a fall-back to the einops form for anything else. A reshape
nobody can name is better printed as notation than as a wrong sentence.

**Done.** What an axis counts comes from the symbol's own documentation,
because that is a fact about the design — `T` is tokens, patches or one image
depending on which design is open. Three rules were needed and each was found by
rendering all twenty-four presets and reading them: a width is not a count of
the thing it is a width of, the earliest word in the sentence wins, and a count
of what is inside *one* of a thing is not a count of the thing. The two reshapes
this catalog produces are named; anything else prints its einops pattern.

### L5 — The model says what it is · **done**

Render `meta.notes`. Two places: the title block, under the design name, where
it is the one-line answer to "what am I looking at"; and the preset picker,
*before* loading, where it is the answer to "which of these twenty-three do I
want".

That second one turns the picker from a flat list of twenty-three slugs into a
library: grouped by family, each entry carrying its notes line, its published
parameter count and the link to the source it was checked against. The
information is all in `presets/data/index.json` and the files already.

**Done.** `File > Reference architectures…` lists all twenty-three grouped by
family with their notes, published counts and sources, and loads one on a click;
the title block carries the open design's own sentence under its name. Six
presets had no notes and now have them.

### L6 — A model small enough to see · **done**

There is no toy in the library. `nanogpt` is Karpathy's config, which is GPT-2
small at 124,475,904 parameters. Bycroft's central move is that his default
model is 85,000 parameters and every number on screen is one you can actually
look at.

Add a preset at that scale — three layers, 48-wide, three heads, a vocabulary of
a handful of tokens — and make it the default document instead of an
eight-billion-parameter Llama. `scale()` already shrinks a design while keeping
its proportions, so the work is mostly deciding the target and writing the
`meta.notes` that says why it exists.

The judgement is whether the default document should change. Opening a CAD tool
on the hardest thing it can draw is a real cost for a new reader and no cost at
all for someone who uses it daily and loads a preset in the first five seconds.
The compromise — toy by default, remember the last design — is probably right
and should be argued for explicitly rather than slipped in.

**Done.** `nano-sort`, and it is the default document. 85,728 parameters,
computed here and then checked against PyTorch; the last preset loaded is
remembered, so only a first visit lands on the toy. It also found a bug in the
verifier, whose forward pass ran at a fixed sequence of 128 and so indexed off
the end of an 11-row position table.

### L7 — Plumbing out of the way · **done**

`_in boundary_in` and `_out boundary_out` are in the model tree of every preset.
They are how a container subgraph is wired and they mean nothing to a reader.
`rearrange` is in the same category: real, necessary, and not a stage of a
transformer.

A **figure** detail mode, beside flat and one-to-five: draws what a paper draws.
Boundary nodes gone, reshapes folded into the block on either side, the residual
spine straight down the sheet. Not a separate renderer — a filter over the
unfold in `state/unfold.ts`, which already short-circuits boundary nodes out of
the *wiring* and could as easily drop them from the drawing.

The risk worth naming: a view that hides blocks is a view where a design-rule
finding can point at something not on screen. Findings on a hidden block must
surface on whatever absorbed it, or the mode is a way to lose an error.

**Done**, on `g`. The wire traces straight through a dropped reshape, and
`unfold` returns what each hidden block was absorbed into so its findings appear
on whatever took its place, named. What counts as plumbing is a declared set
with a test that every name in it is a real catalog type — the failure
`SIDEWAYS_IN` had.

### L8 — The walkthrough · **done**

The big one, and the one that actually answers the original complaint.

Bycroft's ten phases are written against one model. Twenty-three presets cannot
each have ten hand-written phases, and they do not need to: the *kinds* are few.
A dense pre-norm transformer, a mixture-of-experts, a state-space hybrid, a
vision transformer, a convolutional classifier — five scripts, and the numbers
in each are read from whatever design is open. "Each token becomes a vector of
`D` numbers" is one sentence that works for all twenty-three, and the sentence
that follows it differs by kind rather than by model.

A step names:
- a set of block paths to light, with everything else dimmed;
- a detail level and a viewport, or a camera in the volume view;
- a paragraph, with phrases bound to those paths the way `c_blockRef` binds
  them, so pointing at a word points at the drawing.

Three things make this harder than it looks, and they should be decided before
any of it is written:

- **It must run on the open design, including an edited one.** A walkthrough
  that only works on presets is a video. One that survives the user changing `D`
  is the thing this tool can do that a video cannot — and it means every
  sentence with a number in it has to be derived, not typed.
- **A step that cannot find its blocks must say so, not break.** The same
  condition the feature timeline already handles when a suppressed step's target
  is gone, and the same answer: name what is missing and carry on.
- **Where does it live.** A panel beside the sheet is the obvious answer and
  probably wrong: the whole point is that the text and the drawing are one thing.
  Bycroft puts the narration in a column and highlights into the scene. That is
  worth copying rather than improving on.

**Done**, on `w`. Eleven steps for the default design, derived from the blocks
it contains rather than recorded: `alexnet` gets a convolution step and no
attention step, a sparse model is told about its router and a dense one is not.
A test asserts the last part directly — change `D` from 768 to 1536 and the
embedding step says 1,536.

## The other meaning of the word

"Accessibility" also means WCAG, and it is worth saying where that stands so it
does not get silently folded into the above.

The chrome is in reasonable shape. The running app has `lang="en"`, a heading
hierarchy from the panel titles down, landmark elements, two `aria-live`
regions, no unlabelled buttons and no images without alternative text. The third
pass audited contrast across every text element and floors at 5.3:1 in dark and
4.5:1 in light.

The canvas is the gap, and it is the usual one: a React Flow graph is a pile of
absolutely-positioned divs, reachable by tab in DOM order, which is not the order
the drawing reads in, and announced as nothing in particular. The honest thing to
say is that the sheet has a text alternative already — the model tree is the same
design as a labelled hierarchy, and it cross-probes — and that making the tree
the *declared* alternative, with the canvas marked as presentational and a
documented keyboard path through it, is a smaller and more truthful job than
trying to make a node graph screen-readable. It is worth doing, and it is not
what this roadmap is about.

## Order, and what to do first

All eight are done. The twenty-first and twenty-second passes in
[interaction-design.md](interaction-design.md) record what happened and what it
found — including five bugs that were invisible because each returned a value
the same shape as a real answer: a block a design defines for itself could be
listed in the palette, dragged onto the sheet and simply not appear.

What is left is one thing this roadmap did not name. The volume view still takes
its stage names from a port of somebody else's layout, so the same block is
*Token Embed* there and *token embedding* on the sheet. Now that a name is a
fact the catalog carries, the port should read it.

L8 is the one that earns the comparison to Bycroft, and it should not be started
until L1, L3 and L6 are done: narration over parts named `sdpa`, shapes with no
key and a model too big to see would be narration about gibberish.
