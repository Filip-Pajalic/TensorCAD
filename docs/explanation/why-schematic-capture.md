# Why schematic capture

TensorCAD treats a neural network the way an EDA tool treats a circuit. That is a
deliberate choice over the two obvious alternatives, and this is the argument.

## The alternatives

**A diagram tool** — draw.io, Figma, a figure in a paper. The drawing is a
picture *of* a model. Nothing checks it, nothing counts it, and it drifts from the
code the moment either changes.

**Code** — which is what everyone actually does. Accurate by construction, and
completely opaque about the things you most want to know before you commit: what
does this cost, will it fit, where did the parameters go, what happens if the
width doubles. Those questions are answerable from the source, but only by
running it or by working it out on paper.

## What a schematic gives you that neither does

**Connectivity is data.** A netlist is checkable. An interface that does not
match is caught by the drawing rather than by a stack trace.

**Rules run continuously.** Electrical rule checking has been table stakes in EDA
since the 1980s: a design is checked while you work, not after you build. Head
divisibility, vocabulary padding, whether the thing fits your GPUs — these are
the same kind of question and deserve the same treatment.

**Symbols carry their own maths.** A resistor knows its power dissipation. A
block knows its parameter count and FLOPs. Analysis is not a separate model of
the design that can disagree with it; it is a property of the symbols you placed.

**The drawing is the source.** Not a picture: generate the PyTorch from it, and
the picture cannot be stale.

## What it borrows, specifically

From **KiCad's eeschema**: the interaction model. Pick apertures generous enough
to hit, net highlighting rather than segment highlighting, junction dots at a 6×
ratio, hollow circles on unconnected pins, wires reconnectable by dragging an
end. Those conventions are decades old and they work.

From **published architecture figures**: the visual language. One filled box for
attention and pale outlines for everything else, because that is what makes those
figures readable at a glance. A repeated stack drawn once with a `32×` bracket.

From **chip design**: the idea that committing a design to manufacture is an
event — the tape-out — and that everything before it is checkable.

## What it is not

Not a training framework. Not a profiler. Not a replacement for reading the
paper. It answers the questions you have *before* you write the training script,
and it answers them exactly rather than approximately.
