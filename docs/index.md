# TensorCAD documentation

Organised by [Diátaxis](https://diataxis.fr/): four kinds of document for four
different needs. Knowing which one you are in saves everybody time — a tutorial
that stops to explain itself is a bad tutorial, and a reference that tries to
teach is a bad reference.

|  | **Practical steps** | **Theoretical knowledge** |
|---|---|---|
| **Study** | [Tutorials](#tutorials) — learning by doing | [Explanation](#explanation) — understanding why |
| **Work** | [How-to guides](#how-to-guides) — solving a problem | [Reference](#reference) — looking something up |

## Tutorials

Start here if you are new. These are lessons: follow them start to finish and
you will have built something.

- [Your first design](tutorials/first-design.md) — build a small transformer
  from an empty sheet, check it, and generate the PyTorch.

## How-to guides

Recipes for a specific job, assuming you already know your way around.

- [Add a block to the catalog](how-to/add-a-block.md)
- [Add a preset](how-to/add-a-preset.md)
- [Define a block inside a document](how-to/define-a-block-in-a-document.md) —
  the KiCad-style project library, no TypeScript required
- [Verify a design against PyTorch](how-to/verify-against-pytorch.md)
- [Drive TensorCAD from an agent](how-to/use-the-mcp-server.md)

## Reference

Descriptions of the machinery. Dry on purpose.

- [Ports](reference/ports.md) — what a pin declares and what each field does
- [The document format](reference/document.md) — the IR, which is the source of
  truth for everything else
- [Design rules](reference/design-rules.md) — every check, what it fires on
- [Analysis outputs](reference/analysis.md) — every number the engine reports
- [The analysis maths](reference/analysis-math.md) — the formulas, with sources
- [Command line](reference/cli.md)

## Explanation

Background and argument. Read when you want to know why something is the way it
is, or why it is not something else.

- [Why schematic capture](explanation/why-schematic-capture.md) — the case for
  treating a model like a circuit
- [The shape algebra](explanation/shape-algebra.md) — why dimensions are exact
  polynomials rather than integers
- [Interaction design](explanation/interaction-design.md) — sixteen passes of
  editor work, each with what was wrong and what replaced it
- [Landscape](explanation/landscape.md) — the tools that already exist and what
  they do not do

---

Contributor-facing material lives at the root: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)
for the workflow, [`../CLAUDE.md`](../CLAUDE.md) for the invariants and the
agent entry point, [`../ROADMAP.md`](../ROADMAP.md) for what is planned.
