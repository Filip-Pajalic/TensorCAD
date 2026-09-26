# Start a design of your own

**File ▸ New design…** (Ctrl+N) asks two things, the kind of model and how big,
and makes a design of that kind at that size. You don't start from an empty
sheet or from somebody else's model at somebody else's size.

## Choose a kind

| Kind | Starts from | What it is |
|---|---|---|
| Llama-style transformer | `llama-3-8b` | Grouped-query attention, a gated feed-forward, rotary positions, RMS norm |
| GPT-2-style transformer | `gpt2-small` | Full multi-head attention, a plain feed-forward, learned positions, layer norm |
| Mixture of experts | `qwen3-30b-a3b` | Many small experts, a few per token |
| Hybrid: state space and attention | `jamba-v0.1` | Mostly Mamba layers, attention every eighth, experts every other |
| Encoder–decoder | `t5-small` | An encoder over the input and a decoder attending to it |
| Vision transformer | `ijepa-vit-h14` | Patches of an image, attention in both directions |

Each kind starts from a reference design, which is a preset held to its
published parameter count. The new design is that reference scaled to the size
you ask for, with its proportions kept.

## Choose a size

Pick one of the sizes offered, or type your own (`2.5B`, `500M`, `3e9`).

Or choose **As large as trains on one…** and a GPU. That finds the largest
design of the kind whose training fits on one of those GPUs, leaving the same
10% of memory free that the Cluster view leaves.

## Read what it came to

Before anything is created, the dialog shows:

- the parameter count reached, and the reference it was scaled from;
- the design's shape: layers, width, heads, and the feed-forward or experts;
- what training it takes on one device, measured the way the readout measures
  it afterwards. That device is the operating point's, or the one you chose to
  fit. The operating point supplies everything else too: batch, precision,
  optimizer, recomputation.

**Create** opens the design, and by default starts the walkthrough on it.
**Blank sheet** is still there if you want one.

## What scaling does

Width and depth move together, by the cube root of the size ratio, the way the
bench shrinks a design. Three things follow from what real models at that size
do:

- **Heads stay in whole groups.** Llama 3 has four query heads for every
  key/value head. A billion-parameter version keeps that ratio, 12 over 3,
  rather than 13 over 1. When the width moves to keep the groups whole, the
  depth takes up the difference in size.
- **A small model ties its output projection to its embedding.** This happens
  when the vocabulary would otherwise be more than a third of the parameters,
  as with Llama 3.2's 1B and 3B.
- **A new design trains at a sequence length that suits pretraining.** For the
  Llama-style, mixture-of-experts and hybrid kinds that is 4,096 tokens, and
  1,024 for GPT-2. It is not the reference's serving context (Qwen3's is
  32,768). T5 and the vision transformer keep their own lengths, because theirs
  are a target length and a patch grid.

The design says what it started from in its notes. It carries no published
figure, because nobody published this one.

From an agent or a script, the engine's `newDesign({ family, params })` or
`newDesign({ family, fit })` returns the same design, and `families()` lists
the kinds.
