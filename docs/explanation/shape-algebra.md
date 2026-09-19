# The shape algebra

Every tensor dimension in TensorCAD is a **multivariate polynomial with exact
rational coefficients** over named symbols. Not an integer. This page argues why
that is worth the machinery.

## The problem with integers

Resolve `D` to 4096 and `H` to 32 early, and a shape is a tuple of numbers. Then
two things go wrong.

A mismatch becomes uninformative. `4096 ≠ 4032` tells you two numbers differ. It
does not tell you that one path computes `H × dh` and the other computes `D`, and
that they diverged because `dh` was set independently.

And a shape can no longer *mean* anything at runtime. Batch size and sequence
length are not known when you are drawing. Substitute a placeholder and every
downstream check is about the placeholder.

## What a polynomial buys

`B` and `T` stay indeterminate all the way through. The design symbols — `D`,
`H`, `dh`, `F` — are substituted. Two shapes are compatible when their difference
is the **zero polynomial** under that partial environment.

So `B T (H dh)` and `B T D` are the same shape when `H × dh = D`, and when they
are not, the difference is `H·dh − D` — an expression naming exactly what
disagrees. That is what the canvas shows you.

## Division is an obligation, not a rounding

`D / H` succeeds only when `H` divides every term of `D` exactly. When it cannot
be proven, the algebra refuses rather than rounding, and the refusal becomes a
design-rule finding.

This is the check that catches `D = 1024, H = 7` at the keystroke instead of at
`model.to(device)`.

## Rationals, exactly

Coefficients are rationals, so `1.3 * 8/3 * D` — how Llama writes a feed-forward
width — stays `52/15 · D` rather than drifting through binary floating point.
`ceil_mult(1.3*8/3*D, 1024)` then folds to a number, because rounding functions
need their arguments determined and design parameters always are.

One deliberate wart: `rat()` scales by ten until numerator and denominator are
whole, so `1/3` becomes `333333333333/1000000000000`. It is not exact. The Go
port reproduces it exactly rather than fixing it, because a shape label that read
differently in the two engines would be a migration bug you could only find by
eye. It gets fixed in both at once, or not at all.

## Why this makes `scaleDesign` possible

Because a parameter is stored as the expression the author wrote — not the number
it evaluated to — halving `D` moves everything derived from it. The feed-forward
width is still `ceil_mult(1.3*8/3*D, 1024)`; it just evaluates to something else.

This is also why **a numeric parameter field must never become a stepper or a
slider**. Doing so overwrites `ceil_mult(1.3*8/3*D, 1024)` with `14336` and
destroys the design intent. Parameters are edited as text with the evaluated
number shown beside them.

## Where it lives

`packages/core-go/shapes/symexpr.go` is the polynomial; `expr.go` parses the
expressions; `pattern.ts` parses shape patterns; `infer.ts` walks a graph
propagating shapes and collecting the disagreements.
