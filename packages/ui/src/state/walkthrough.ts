/**
 * The walkthrough: the design explaining itself, a stage at a time.
 *
 * Brendan Bycroft's LLM visualisation has ten hand-written phases against one
 * model. Twenty-four presets cannot each have ten, and do not need to: what
 * differs between them is which *kinds* of stage they have, not what a stage
 * is. An embedding is an embedding in a 124k-parameter sorter and in a 671B
 * mixture of experts, and the sentence that explains one explains the other
 * once its numbers are filled in.
 *
 * So the steps are derived from the blocks the design actually contains, in
 * flow order, and the prose is authored once per kind. That is also the thing
 * a recorded explanation cannot do: change `D` and the walkthrough changes
 * with it, because every number in it was read out of the design rather than
 * typed.
 *
 * Nothing here is allowed to invent. A step exists because a block of that
 * category is in the graph, and if the design has no attention there is no
 * attention step.
 */

import { catalogOf } from "../engine.js";
import { formatBytes, formatCount, formatFlops, joinPath } from "@tensor-cad/engine";
import type { Doc, NodeDef, Resolved } from "@tensor-cad/engine";
import type { Derived } from "./derive.js";
import type { Trace } from "../three/trace.js";

export interface Step {
  id: string;
  title: string;
  /** Paragraphs. Written as plain strings: this is prose, not markup. */
  body: string[];
  /**
   * Blocks this step is about. The canvas lights these and dims the rest, which
   * is the whole reason the narration is attached to the drawing rather than
   * printed beside it.
   */
  paths: string[];
  /** How far to open the containers, so the step's blocks are on screen. */
  detail: number;
}

/** One block found in the design, wherever it was found. */
interface Found {
  path: string;
  node: NodeDef;
  type: string;
  category: string;
  resolved: Resolved | undefined;
  /** How deep in the container tree, which is the detail a step needs. */
  depth: number;
}

const n = (v: unknown, fallback = 0): number =>
  typeof v === "number" && Number.isFinite(v) ? v : fallback;

const count = (v: number): string => v.toLocaleString("en-US");

/**
 * Every block in the design, including the ones inside composites.
 *
 * A composite's interior is what the analysis expanded it into, which is the
 * same thing the canvas draws when the detail control opens it — so a step that
 * names a path from here is naming a block that can be shown.
 */
function findAll(doc: Doc, derived: Derived): Found[] {
  const cat = catalogOf(doc);
  const out: Found[] = [];

  const walk = (nodes: NodeDef[], prefix: string, depth: number): void => {
    for (const node of nodes) {
      const path = joinPath(prefix, node.id);
      const def = cat[node.type];
      out.push({
        path,
        node,
        type: node.type,
        category: def?.category ?? "unknown",
        resolved: derived.infer.resolved.get(path),
        depth,
      });
      const interior = node.graph ?? derived.infer.expansions.get(path);
      if (interior) walk(interior.nodes, path, depth + 1);
    }
  };

  walk(doc.graph.nodes, "", 0);
  return out;
}

const firstOf = (all: Found[], test: (f: Found) => boolean): Found | undefined => all.find(test);
const allOf = (all: Found[], test: (f: Found) => boolean): Found[] => all.filter(test);

/** A number in prose: three significant figures and a real minus sign. */
const num = (v: number): string => v.toPrecision(3).replace(/^-/, "\u2212");
const pct = (v: number): string => `${Math.round(v * 100)}%`;

/**
 * What the traced run did, in sentences for the steps that it illustrates.
 *
 * Only ever added to a step, never a step of its own: the steps are what the
 * design has, and a trace changes what can be said about them, not which of
 * them there are. The canvas lights a step by its index, and that must not
 * move because a file finished loading.
 */
function traced(trace: Trace, embedPath: string | undefined, attnPath: string | undefined, layers: number) {
  const f = trace.file;
  const letters = trace.letters;
  const input = f.input.map((t) => f.task.symbols[t]).join(" ");
  const answer = f.answer.map((t) => f.task.symbols[t]).join(" ");
  const out: Partial<Record<"input" | "embed" | "attention" | "output", string>> = {};

  out.input =
    `In the run the volume view shows, the input is ${input}: ${f.task.length} symbols to sort, followed by the model's own answer so far, ${letters.length} positions in all. ` +
    `It was trained on nothing else for ${count(f.training.steps)} steps, and ` +
    (f.training.held_out_accuracy === 1
      ? "sorts every held-out input it was tested on."
      : `sorts ${pct(f.training.held_out_accuracy)} of the held-out inputs it was tested on.`);

  const table = embedPath ? trace.tensor(embedPath, -1, "weight") : null;
  if (table && table.shape.length === 2) {
    const dim = table.shape[1]!;
    const first = f.sequence[0]!;
    const row = [...table.data.subarray(first * dim, first * dim + 4)].map(num).join(", ");
    out.embed = `Its first symbol, ${letters[0]}, is row ${first} of the table: ${row}, and ${dim - 4} more. In the volume view that is column ${first} of the token embedding; the input embedding beside it is that plus the first position's own row.`;
  }

  // The sharpest look any head takes while the model is writing its answer,
  // in the last block, where attention has had the most to work with.
  const probs = attnPath ? trace.tensor(attnPath, layers - 1, "probs") : null;
  if (probs && probs.shape.length === 3) {
    const [H, T] = [probs.shape[0]!, probs.shape[1]!];
    let best = { h: 0, q: 0, k: 0, p: -1 };
    for (let h = 0; h < H; h++) {
      for (let q = f.task.length - 1; q < T; q++) {
        for (let k = 0; k <= q; k++) {
          const p = probs.data[h * T * T + q * T + k]!;
          if (p > best.p) best = { h, q, k, p };
        }
      }
    }
    const writing = best.q - (f.task.length - 1);
    out.attention =
      `In the run the volume view shows, the sharpest look is in the last block: when the model is about to write its ${ordinal(writing + 1)} answer symbol, head ${best.h + 1} puts ${pct(best.p)} of its attention on position ${best.k}, the ${letters[best.k]}. ` +
      "Each row of an attention matrix in that view is one position deciding what to look at.";
  }

  out.output = `Reading ${input}, the traced model wrote ${answer}, one symbol at a time, each one the highest of its ${f.task.vocab} scores.`;
  return out;
}

const ordinal = (n: number): string => ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"][n - 1] ?? `${n}th`;

/**
 * The steps, in the order a token passes through them.
 *
 * Built by asking the design what it has rather than by assuming a transformer:
 * `alexnet` has convolutions and no attention, `nemotron-h-8b` has state-space
 * layers, and each gets the steps its own blocks earn.
 *
 * With a trace — a run of this exact design, which only `nano-sort` has — the
 * steps it illustrates also quote what the run actually computed.
 */
export function buildWalkthrough(doc: Doc, derived: Derived, trace: Trace | null = null): Step[] {
  const all = findAll(doc, derived);
  const steps: Step[] = [];
  const sym = derived.symbols.values;
  const a = derived.analysis;

  const run = trace
    ? traced(
        trace,
        firstOf(all, (f) => f.type === "embedding")?.path,
        firstOf(all, (f) => f.category === "attention")?.path,
        sym.L ?? 1,
      )
    : {};

  const push = (
    id: string,
    title: string,
    body: (string | null)[],
    paths: (string | undefined)[],
    detail: number,
  ): void => {
    const real = paths.filter((p): p is string => !!p);
    const text = body.filter((s): s is string => !!s && s.length > 0);
    if (real.length === 0 || text.length === 0) return;
    steps.push({ id, title, body: text, paths: real, detail });
  };

  // ---------------------------------------------------------------- the whole
  const layers = sym.L ?? sym.Lm ?? null;
  const dense = derived.params.active === derived.params.total;
  push(
    "whole",
    `What ${doc.meta.name} is`,
    [
      // A convnet has no `L` and no `D`, so it gets counted the only way it
      // can be: by the stages it is written out as. Without this the opening
      // line of AlexNet's walkthrough was four words long.
      `${formatCount(derived.params.total)} parameters` +
        (layers
          ? `, in ${count(layers)} layers` + (sym.D ? ` of width ${count(sym.D)}` : "")
          : `, across ${count(doc.graph.nodes.length)} stages drawn end to end`) +
        ".",
      dense
        ? "Every parameter is used on every token: this is a dense model."
        : `Only ${formatCount(derived.params.active)} of them are used on any one token — the rest belong to experts that token was not routed to.`,
      doc.meta.notes ?? null,
      "Each step below lights the parts it is about and dims the rest. The numbers are read out of the design, so changing one changes them.",
    ],
    doc.graph.nodes.map((node) => node.id),
    1,
  );

  // ---------------------------------------------------------------- the input
  const input = firstOf(all, (f) => f.type === "input");
  const embed = firstOf(all, (f) => f.type === "embedding");
  push(
    "input",
    "What goes in",
    [
      embed
        ? `A sequence of integers, one per token, each one an index into a vocabulary of ${count(n(embed.resolved?.p.vocab))}.`
        : "A batch of inputs; this design takes tensors rather than token ids.",
      "Batch and sequence length stay symbolic all the way through the design, because they are conditions of a run rather than properties of the model.",
      run.input ?? null,
    ],
    [input?.path],
    1,
  );

  // ------------------------------------------------------------ the embedding
  if (embed) {
    const dim = n(embed.resolved?.p.dim);
    const vocab = n(embed.resolved?.p.vocab);
    push(
      "embed",
      "Every token becomes a vector",
      [
        `A table with ${count(vocab)} rows and ${count(dim)} columns. Looking a token up is a row lookup, not a matrix multiply, which is why this block costs ${formatCount(vocab * dim)} parameters and almost no arithmetic.`,
        `Those ${count(dim)} numbers are the residual stream. Every block from here to the output reads a vector of that width and writes one back.`,
        run.embed ?? null,
      ],
      [embed.path],
      1,
    );
  }

  // -------------------------------------------------------------- positions
  const pos = firstOf(all, (f) => f.type === "pos_embedding");
  const rope = firstOf(all, (f) => f.type === "rope");
  if (pos) {
    push(
      "positions",
      "And where it sits",
      [
        `Attention sees a set, not a sequence, so the position has to be put in by hand. This design adds a learned vector per position, ${count(n(pos.resolved?.p.max_seq))} of them — which is also the longest sequence it can ever take.`,
      ],
      [pos.path],
      1,
    );
  } else if (rope) {
    push(
      "positions",
      "And where it sits",
      [
        "Attention sees a set, not a sequence. Rather than adding a position vector, this design rotates each query and key by an angle that depends on the position, so what attention sees is the distance between two tokens rather than where either one is.",
        "Nothing is stored for it: a rotation has no parameters, which is why this design can run at sequence lengths it never trained at.",
      ],
      [rope.path],
      3,
    );
  }

  // ------------------------------------------------------------- the stack
  const stack = firstOf(all, (f) => f.category === "container");
  if (stack && layers) {
    push(
      "stack",
      "The same block, over and over",
      [
        `${count(layers)} identical blocks in series, each one reading the residual stream and adding to it. They hold ${formatCount(derived.paramsByPath.get(stack.path) ?? 0)} of the model — almost all of it.`,
        "The next few steps are what happens inside one of them.",
      ],
      [stack.path],
      1,
    );
  }

  const inStack = (f: Found): boolean => !stack || f.path.startsWith(`${stack.path}/`);

  // ------------------------------------------------------------------ norms
  const norms = allOf(all, (f) => f.category === "norm" && inStack(f));
  if (norms.length > 0) {
    push(
      "norm",
      "Steady the numbers first",
      [
        `Before each of the two heavy stages the stream is normalised: rescaled so its numbers have a consistent size whatever the block before it did to them. It costs ${formatCount(norms.reduce((t, f) => t + (derived.paramsByPath.get(f.path) ?? 0), 0))} across the block and it is what keeps a deep stack from drifting apart as it trains.`,
      ],
      norms.map((f) => f.path),
      Math.max(...norms.map((f) => f.depth)),
    );
  }

  // -------------------------------------------------------------- attention
  const attn = firstOf(all, (f) => f.category === "attention" && inStack(f));
  if (attn) {
    const p = attn.resolved?.p ?? {};
    const heads = n(p.heads, n(sym.H));
    const kv = n(p.kv_heads, heads);
    const dh = n(p.head_dim, n(sym.dh));
    push(
      "attention",
      "Every token looks at the others",
      [
        `Each token asks a question, every earlier token offers an answer, and the token takes a weighted average of what it is offered. ${count(heads)} heads do this at once, each over its own ${count(dh)} numbers, so the block can attend to several things at a time.`,
        kv === heads
          ? `All ${count(heads)} heads keep their own keys and values.`
          : `${count(heads)} heads ask, but only ${count(kv)} sets of keys and values are kept and shared between them. That is what makes the cache affordable: ${formatBytes(a.kv.bytesPerToken)} per token rather than ${formatBytes((a.kv.bytesPerToken * heads) / Math.max(kv, 1))}.`,
        `Attention is the one stage whose cost grows with the sequence: ${formatFlops(a.flops.fwdAttention)} per token at ${count(a.options.T)} tokens, against ${formatFlops(a.flops.fwdDense)} for everything else.`,
        run.attention ?? null,
      ],
      [attn.path],
      attn.depth,
    );
  }

  // ------------------------------------------------------- state-space layers
  const ssm = firstOf(all, (f) => f.category === "ssm" && inStack(f));
  if (ssm) {
    push(
      "ssm",
      "Or carry a state forward instead",
      [
        "This design does not use attention everywhere. A state-space layer walks the sequence once, carrying a fixed-size state from one token to the next, so its cost is linear in the sequence rather than quadratic.",
        "It also caches nothing that grows with the context: the state is the same size at token ten as at token ten thousand, which is the whole argument for a hybrid.",
      ],
      [ssm.path],
      ssm.depth,
    );
  }

  // ------------------------------------------------------------- feed-forward
  const mlp = firstOf(all, (f) => (f.category === "mlp" || f.category === "moe") && inStack(f));
  if (mlp) {
    const p = mlp.resolved?.p ?? {};
    const experts = n(p.experts);
    const share = derived.params.total
      ? ((derived.paramsByPath.get(mlp.path) ?? 0) / derived.params.total) * 100
      : 0;
    push(
      "mlp",
      experts ? "Then one token at a time, to a chosen few" : "Then each token thinks on its own",
      [
        experts
          ? `Attention is how tokens talk to each other; this is what each one does alone afterwards. Instead of one feed-forward there are ${count(experts)}, and a router sends each token to ${count(n(p.top_k))} of them — so the model can be large without every token paying for all of it.`
          : `Attention is how tokens talk to each other; this is what each one does alone afterwards. The vector is widened from ${count(n(sym.D))} to ${count(n(p.hidden, n(p.expert_hidden, n(sym.F))))}, put through a nonlinearity, and brought back.`,
        `It is the largest single thing in the block: ${formatCount(derived.paramsByPath.get(mlp.path) ?? 0)}, ${share.toFixed(0)}% of the whole model.`,
      ],
      [mlp.path],
      mlp.depth,
    );
  }

  // -------------------------------------------------------------- the residual
  const adds = allOf(all, (f) => f.type === "add" && inStack(f));
  if (adds.length > 0) {
    push(
      "residual",
      "Nothing replaces the stream, everything adds to it",
      [
        "Each stage's output is added back rather than substituted. The circled plus is where that happens, and the line running past the block is the stream going by untouched.",
        "That is why a hundred-layer model trains at all: there is a path from the output back to the input that passes through no matmul, so a gradient can reach the bottom without being multiplied away.",
      ],
      adds.map((f) => f.path),
      Math.max(...adds.map((f) => f.depth)),
    );
  }

  // ---------------------------------------------------------- convolutions
  const conv = allOf(all, (f) => f.type === "conv2d");
  if (conv.length > 0) {
    push(
      "conv",
      "A window slid over the image",
      [
        `${count(conv.length)} convolutions, each one sweeping a small window over the whole image and reusing the same weights at every position. That reuse is why a convnet has so few parameters for the work it does.`,
        "There is no sequence here: a token is one image, so every per-token figure in the readout reads as per-image.",
      ],
      conv.map((f) => f.path),
      Math.max(...conv.map((f) => f.depth)),
    );
  }

  // ------------------------------------------------------------- the output
  const head = firstOf(all, (f) => f.category === "head");
  const finalNorm = allOf(all, (f) => f.category === "norm" && !inStack(f)).at(-1);
  if (head) {
    const p = head.resolved?.p ?? {};
    const vocab = n(p.vocab);
    push(
      "output",
      "And out the other end",
      [
        finalNorm ? "One last normalisation, and then the stream is turned back into a score per token in the vocabulary." : "The stream is turned back into a score per token in the vocabulary.",
        p.tied === true
          ? `The projection reuses the embedding table rather than learning its own, which costs nothing and is why this block reports no parameters of its own.`
          : `${count(vocab)} scores per position, from its own ${formatCount(vocab * n(p.dim, n(sym.D)))} parameters.`,
        "The highest score is the next token. Feed it back in at the end and you have the loop the whole thing exists for.",
        run.output ?? null,
      ],
      [finalNorm?.path, head.path],
      1,
    );
  }

  // --------------------------------------------------------- what it costs
  push(
    "cost",
    "What it costs to run",
    [
      `At the operating point set above — ${count(a.options.T)} tokens, batch ${count(a.options.B)}, ${a.options.inferenceDtype} — one token costs ${formatFlops(a.flops.fwdTotal)} to predict.`,
      a.kv.bytesPerToken > 0
        ? `Serving it means keeping ${formatBytes(a.kv.bytesPerToken)} per token of context, so a full ${count(a.options.T)}-token conversation holds ${formatBytes(a.kv.bytesPerSequenceFixed)} before a single reply is generated.`
        : null,
      "Every one of those numbers is in the readout on the right, under the same operating point, and every one of them moves when you change the design.",
    ],
    doc.graph.nodes.map((node) => node.id),
    1,
  );

  return steps;
}
