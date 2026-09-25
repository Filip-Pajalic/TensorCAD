/**
 * The walkthrough: the design explaining itself, a stage at a time.
 *
 * Brendan Bycroft's LLM visualisation has ten hand-written phases against one
 * model. Twenty-eight presets cannot each have ten, and do not need to: what
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
function traced(
  trace: Trace,
  embedPath: string | undefined,
  attnPath: string | undefined,
  headPath: string | undefined,
  layers: number,
) {
  const f = trace.file;
  const letters = trace.letters;
  const out: Partial<Record<"input" | "embed" | "attention" | "output", string>> = {};
  const n = letters.length;

  if (trace.untrained) {
    out.input =
      `In the run the volume view shows, this design has never been trained: the numbers are its weights exactly as it initialises them (seed ${f.training.seed}), ` +
      `run on ${count(n)} random token ids. They are what it computes on its first step, and nothing it has learnt.`;
  } else {
    const input = f.input.map((t) => f.task.symbols[t]).join(" ");
    const accuracy = f.training.held_out_accuracy ?? 0;
    out.input =
      `In the run the volume view shows, the input is ${input}: ${f.task.length} symbols to sort, followed by the model's own answer so far, ${n} positions in all. ` +
      `It was trained on nothing else for ${count(f.training.steps)} steps, and ` +
      (accuracy === 1
        ? "sorts every held-out input it was tested on."
        : `sorts ${pct(accuracy)} of the held-out inputs it was tested on.`);
  }

  const table = embedPath ? trace.tensor(embedPath, -1, "weight") : null;
  if (table && table.shape.length === 2) {
    const dim = table.shape[1]!;
    const first = f.sequence[0]!;
    const row = [...table.data.subarray(first * dim, first * dim + 4)].map(num).join(", ");
    const which = trace.untrained ? `Its first token, id ${first},` : `Its first symbol, ${letters[0]},`;
    out.embed = `${which} is row ${first} of the table: ${row}, and ${dim - 4} more. In the volume view that is column ${first} of the token embedding; the input embedding beside it is that plus the first position's own row.`;
  }

  const probs = attnPath ? trace.tensor(attnPath, layers - 1, "probs") : null;
  if (probs && probs.shape.length === 3) {
    const [H, T] = [probs.shape[0]!, probs.shape[1]!];
    if (trace.untrained) {
      // Untrained attention is close to even, and saying so is the point: the
      // last position, which can see all of them, and how far its largest
      // share is from an exactly even one.
      const q = T - 1;
      let largest = 0;
      for (let h = 0; h < H; h++) {
        for (let k = 0; k <= q; k++) largest = Math.max(largest, probs.data[h * T * T + q * T + k]!);
      }
      out.attention =
        `Untrained, attention has nothing to look for yet. In the last block the last position can see all ${count(T)}, and the largest share any head gives one of them is ${pct(largest)} — against ${pct(1 / T)} if it were exactly even. ` +
        "Each row of an attention matrix in the volume view is one position deciding what to look at.";
    } else {
      // The sharpest look any head takes while the model is writing its
      // answer, in the last block, where attention has had the most to work with.
      const from = (f.task.length ?? 1) - 1;
      let best = { h: 0, q: 0, k: 0, p: -1 };
      for (let h = 0; h < H; h++) {
        for (let q = from; q < T; q++) {
          for (let k = 0; k <= q; k++) {
            const p = probs.data[h * T * T + q * T + k]!;
            if (p > best.p) best = { h, q, k, p };
          }
        }
      }
      out.attention =
        `In the run the volume view shows, the sharpest look is in the last block: when the model is about to write its ${ordinal(best.q - from + 1)} answer symbol, head ${best.h + 1} puts ${pct(best.p)} of its attention on position ${best.k}, the ${letters[best.k]}. ` +
        "Each row of an attention matrix in that view is one position deciding what to look at.";
    }
  }

  if (trace.untrained) {
    // How close to a guess its first answer is: the highest probability at the
    // last position, against an even spread over the vocabulary.
    const logits = headPath ? trace.resolve({ path: headPath, layer: -1, role: "softmax", across: 0 }) : null;
    if (logits && logits.shape.length === 2) {
      const [T, V] = [logits.shape[0]!, logits.shape[1]!];
      let top = 0;
      for (let v = 0; v < V; v++) top = Math.max(top, logits.data[(T - 1) * V + v]!);
      out.output = `Untrained, its scores say almost nothing: after the last position the likeliest next token gets ${pct(top)}, against ${pct(1 / V)} for an even spread over all ${count(V)}.`;
    }
  } else if (f.answer) {
    const input = f.input.map((t) => f.task.symbols[t]).join(" ");
    const answer = f.answer.map((t) => f.task.symbols[t]).join(" ");
    out.output = `Reading ${input}, the traced model wrote ${answer}, one symbol at a time, each one the highest of its ${f.task.vocab} scores.`;
  }
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
        firstOf(all, (f) => f.category === "head")?.path,
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

  // ------------------------------------------------------------ two sequences
  // An encoder-decoder has a stack over each sequence. The decoder is the one
  // whose blocks attend to the other's output, which is how it is found here
  // rather than by its name.
  const S = a.flops.perStream?.find((s) => s.symbol === "S")?.length;
  const crosses = allOf(all, (f) => f.type === "cross_attention");
  const stacks = allOf(all, (f) => f.category === "container" && f.depth === 0);
  const decoder = crosses[0] ? stacks.find((s) => crosses[0]!.path.startsWith(`${s.path}/`)) : undefined;
  const encoder = decoder ? stacks.find((s) => s !== decoder) : undefined;
  const seq2seq = S !== undefined && !!encoder && !!decoder;
  const depthOf = (f: Found | undefined): number => n(f?.resolved?.p.count, 1);
  const causalIn = (stack: Found | undefined): boolean =>
    firstOf(all, (f) => f.category === "attention" && !!stack && f.path.startsWith(`${stack.path}/`))?.resolved?.p
      .causal !== false;

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
        (seq2seq
          ? `, in an encoder of ${count(depthOf(encoder))} layers and a decoder of ${count(depthOf(decoder))}` +
            (sym.D ? `, of width ${count(sym.D)}` : "")
          : layers
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
  const inputs = allOf(all, (f) => f.type === "input");
  const embed = firstOf(all, (f) => f.type === "embedding");
  push(
    "input",
    "What goes in",
    [
      seq2seq && embed
        ? `Two sequences of integers, each token an index into a vocabulary of ${count(n(embed.resolved?.p.vocab))}: the source, ${count(S!)} tokens that the encoder reads whole, and the target, which the decoder writes one token at a time.`
        : embed
          ? `A sequence of integers, one per token, each one an index into a vocabulary of ${count(n(embed.resolved?.p.vocab))}.`
          : "A batch of inputs; this design takes tensors rather than token ids.",
      "Batch and sequence length stay symbolic all the way through the design, because they are conditions of a run rather than properties of the model.",
      run.input ?? null,
    ],
    seq2seq ? inputs.map((f) => f.path) : [inputs[0]?.path],
    1,
  );

  // ------------------------------------------------------------ the embedding
  if (embed) {
    const dim = n(embed.resolved?.p.dim);
    const vocab = n(embed.resolved?.p.vocab);
    const sharing = allOf(all, (f) => f.type === "embedding" && f.resolved?.p.tied === true);
    push(
      "embed",
      "Every token becomes a vector",
      [
        `A table with ${count(vocab)} rows and ${count(dim)} columns. Looking a token up is a row lookup, not a matrix multiply, which is why this block costs ${formatCount(vocab * dim)} parameters and almost no arithmetic.`,
        `Those ${count(dim)} numbers are the residual stream. Every block from here to the output reads a vector of that width and writes one back.`,
        sharing.length > 0
          ? "The decoder's tokens are looked up in the same table: its embedding is tied to this one, so the vocabulary is learned once for both."
          : null,
        run.embed ?? null,
      ],
      [embed.path, ...sharing.map((f) => f.path)],
      1,
    );
  }

  // -------------------------------------------------------------- positions
  const pos = firstOf(all, (f) => f.type === "pos_embedding");
  const rope = firstOf(all, (f) => f.type === "rope");
  // A score expression that reads both positions is a bias on distance —
  // ALiBi, most often — which is where such a design keeps its sense of order.
  const distance = firstOf(all, (f) => {
    const score = f.type === "sdpa" ? f.resolved?.p.score : undefined;
    return typeof score === "string" && /\bq\b/.test(score) && /\bkv\b/.test(score);
  });
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
  } else if (distance && allOf(all, (f) => f.type === "position_bias").length > 0) {
    const tables = allOf(all, (f) => f.type === "position_bias");
    const t = tables[0]!.resolved?.p ?? {};
    const score = String(distance.resolved?.p.score);
    push(
      "positions",
      "And where it sits",
      [
        "Attention sees a set, not a sequence. This design adds nothing to the vectors: instead each attention score gets a learned bias, one per head, for how far its key is from its query." +
          (score.includes("t5_bucket(")
            ? " Distances are sorted into buckets — one bucket per distance close by, wider ones further out, and one for everything past the furthest — so the table stays small whatever the length."
            : ""),
        `${tables.length === 1 ? "The table is" : `Each of the ${count(tables.length)} tables is`} ${count(n(t.buckets))} buckets by ${count(n(t.heads))} heads, ${formatCount(tables.reduce((s, f) => s + (derived.paramsByPath.get(f.path) ?? 0), 0))} parameters in all, and every layer it is wired to reads the same one rather than learning its own.`,
        `The score expression says exactly how: ${score}.`,
      ],
      tables.map((f) => f.path),
      1,
    );
  } else if (distance) {
    push(
      "positions",
      "And where it sits",
      [
        "Attention sees a set, not a sequence. This design adds nothing to the vectors at all: instead each attention score is docked in proportion to how far back its key is, so a token attends less to what is further away.",
        `The attention's score expression says exactly how: ${String(distance.resolved?.p.score)}. Nothing is stored for it, and nothing caps the length, which is why such a design degrades gracefully past the sequences it trained on rather than failing.`,
      ],
      [distance.path],
      distance.depth,
    );
  }

  // ------------------------------------------------------------- the stack
  const stack = seq2seq ? encoder : firstOf(all, (f) => f.category === "container");
  if (seq2seq) {
    push(
      "stack",
      "Two stacks, over two sequences",
      [
        `An encoder of ${count(depthOf(encoder))} blocks reads the source, ${causalIn(encoder) ? "each token seeing those before it" : "every token seeing every other"}. A decoder of ${count(depthOf(decoder))} blocks writes the target, ${causalIn(decoder) ? "each token seeing only those before it" : "every token seeing every other"}. They hold ${formatCount(derived.paramsByPath.get(encoder!.path) ?? 0)} and ${formatCount(derived.paramsByPath.get(decoder!.path) ?? 0)} of the model.`,
        "The next few steps are what happens inside an encoder block. A decoder block is the same with one more attention, between its own and its feed-forward, which comes after them.",
      ],
      [encoder!.path, decoder!.path],
      1,
    );
  } else if (stack && layers) {
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
        `Each token asks a question, ${p.causal === false ? "every other token" : "every earlier token"} offers an answer, and the token takes a weighted average of what it is offered. ${count(heads)} heads do this at once, each over its own ${count(dh)} numbers, so the block can attend to several things at a time.`,
        kv === heads
          ? `All ${count(heads)} heads keep their own keys and values.`
          : `${count(heads)} heads ask, but only ${count(kv)} sets of keys and values are kept and shared between them. That is what makes the cache affordable: ${formatBytes(a.kv.bytesPerToken)} per token rather than ${formatBytes((a.kv.bytesPerToken * heads) / Math.max(kv, 1))}.`,
        seq2seq
          ? `Attention is the one stage whose cost grows with the sequence: ${formatFlops(a.flops.fwdAttention)} per target token at ${count(S!)} source tokens and ${count(a.options.T)} target tokens, against ${formatFlops(a.flops.fwdDense)} for everything else.`
          : `Attention is the one stage whose cost grows with the sequence: ${formatFlops(a.flops.fwdAttention)} per token at ${count(a.options.T)} tokens, against ${formatFlops(a.flops.fwdDense)} for everything else.`,
        attn.type === "diff_attention"
          ? "Each of these heads is two attention maps over the same values, one taken away from the other, scaled by a learned lambda. Whatever both maps put on tokens that do not matter cancels, which is the point: less attention wasted on context that is only there."
          : null,
        p.written_out === true || p.talking_heads === true
          ? `This attention is written out rather than fused: each head's whole matrix of scores is computed, kept and passed along like any other tensor${p.talking_heads === true ? ", because talking heads mixes every head's matrix into every other's, which needs them all at once" : ""}. That is the memory a fused kernel exists to save, and the design rules say how much it is here.`
          : null,
        p.sinks === true
          ? `Each head also has a sink: one learned score that sits beside the keys in every softmax, so a token that finds nothing worth attending to can put its attention there instead of spreading it thin. It is ${count(heads)} parameters, and it is why a head can stay quiet.`
          : null,
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

  // ------------------------------------------------------- cross-attention
  if (seq2seq) {
    const cached = Object.entries(a.kv.byPath)
      .filter(([path]) => crosses.some((c) => path.startsWith(`${c.path}/`)))
      .reduce((t, [, bytes]) => t + bytes, 0);
    push(
      "cross",
      "The decoder reads the encoder",
      [
        `Each of the decoder's ${count(depthOf(decoder))} blocks has a second attention, between its own and its feed-forward. Its queries come from the target, its keys and values from the encoder's output, and nothing is masked: every target token can look at every source token.`,
        `That is the only way the source reaches the output. Those keys and values are computed once per request, from the encoder, and kept while the target is written` +
          (cached > 0 ? `: ${formatBytes(cached)} for ${count(S!)} source tokens.` : "."),
      ],
      crosses.map((f) => f.path),
      Math.max(...crosses.map((f) => f.depth)),
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
        seq2seq
          ? "The highest score is the target's next token. Feed it back into the decoder and you have the loop; the encoder ran once, and does not run again."
          : "The highest score is the next token. Feed it back in at the end and you have the loop the whole thing exists for.",
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
      seq2seq
        ? `At the operating point set above — ${count(S!)} source tokens and ${count(a.options.T)} target tokens, batch ${count(a.options.B)}, ${a.options.inferenceDtype} — one target token costs ${formatFlops(a.flops.fwdTotal)} to predict, the encoder's share spread over the target, and one whole example ${formatFlops(a.flops.fwdPerExample ?? 0)}.`
        : `At the operating point set above — ${count(a.options.T)} tokens, batch ${count(a.options.B)}, ${a.options.inferenceDtype} — one token costs ${formatFlops(a.flops.fwdTotal)} to predict.`,
      // What a context holds is every token's share and whatever is held per
      // sequence regardless: a sliding window's bounded cache, a state-space
      // layer's state, the source's keys and values.
      a.kv.bytesPerToken > 0
        ? seq2seq
          ? `Serving it means keeping ${formatBytes(a.kv.bytesPerToken)} per target token and ${formatBytes(a.kv.bytesPerSequenceFixed)} per request for the source, so a ${count(a.options.T)}-token reply holds ${formatBytes(a.kv.bytesPerToken * a.options.T + a.kv.bytesPerSequenceFixed)} by its last token.`
          : `Serving it means keeping ${formatBytes(a.kv.bytesPerToken)} per token of context, so a full ${count(a.options.T)}-token conversation holds ${formatBytes(a.kv.bytesPerToken * a.options.T + a.kv.bytesPerSequenceFixed)} before a single reply is generated.`
        : null,
      "Every one of those numbers is in the readout on the right, under the same operating point, and every one of them moves when you change the design.",
    ],
    doc.graph.nodes.map((node) => node.id),
    1,
  );

  return steps;
}
