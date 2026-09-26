/**
 * The operating point, at the top of the readout.
 *
 * A parameter count is a property of the design. Everything else on screen —
 * FLOPs, memory, tokens per second, cost — is a property of the design *under
 * conditions*. Leaving those conditions as invisible defaults is how a readout
 * becomes untrustworthy, so they sit above the numbers they produce and every
 * one of them is editable in place.
 */

import { useState } from "react";
import { useEditor } from "../state/store.js";
import { useDerived } from "../state/hooks.js";
import { DEFAULT_OPERATING, type OperatingPoint, type Recompute } from "../state/operating.js";
import type { Dtype, OptimizerKind } from "@tensor-cad/engine";
import { formatCount } from "@tensor-cad/engine";
import { HARDWARE, HARDWARE_BY_ID } from "../engine.js";

const DTYPES: Dtype[] = ["fp32", "bf16", "fp16", "fp8"];
const OPTIMIZERS: { id: OptimizerKind; label: string }[] = [
  { id: "adamw", label: "AdamW" },
  { id: "adamw8bit", label: "AdamW 8-bit" },
  { id: "muon", label: "Muon" },
  { id: "sgd_momentum", label: "SGD + momentum" },
  { id: "sgd", label: "SGD" },
  { id: "bf16_adam", label: "bf16 Adam" },
];
const RECOMPUTE: Recompute[] = ["none", "selective", "full"];
const POWERS = [1, 2, 4, 8, 16, 32, 64, 128];

function Num({
  label,
  value,
  onChange,
  min = 1,
  title,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min?: number;
  title?: string;
}): React.ReactElement {
  return (
    <label className="op__field" title={title}>
      <span className="op__label">{label}</span>
      <input
        className="field field--num"
        type="number"
        min={min}
        value={value}
        spellCheck={false}
        onKeyDown={(e) => e.stopPropagation()}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n) && n >= min) onChange(Math.floor(n));
        }}
      />
    </label>
  );
}

function Pick<T extends string | number>({
  label,
  value,
  options,
  onChange,
  title,
}: {
  label: string;
  value: T;
  options: { id: T; label: string }[];
  onChange: (v: T) => void;
  title?: string;
}): React.ReactElement {
  return (
    <label className="op__field" title={title}>
      <span className="op__label">{label}</span>
      <select
        className="field"
        value={String(value)}
        onKeyDown={(e) => e.stopPropagation()}
        onChange={(e) => {
          const hit = options.find((o) => String(o.id) === e.target.value);
          if (hit) onChange(hit.id);
        }}
      >
        {options.map((o) => (
          <option key={String(o.id)} value={String(o.id)}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

/**
 * Whether a design has documents to pack: an input whose role says it holds
 * them, which is what a mask that keeps them apart reads.
 */
function readsDocuments(doc: { graph: { nodes: { type: string; params?: Record<string, unknown> }[] } }): boolean {
  return doc.graph.nodes.some((n) => n.type === "input" && n.params?.role === "documents");
}

/** A panel's open or shut, remembered per browser. */
function useRemembered(name: string, fallback: boolean): [boolean, () => void] {
  const [open, setOpen] = useState(() => {
    try {
      const raw = localStorage.getItem(`tensorcad.${name}`);
      return raw === null ? fallback : raw === "1";
    } catch {
      return fallback;
    }
  });
  const toggle = (): void =>
    setOpen((was) => {
      try {
        localStorage.setItem(`tensorcad.${name}`, was ? "0" : "1");
      } catch {
        // Not remembering it is harmless.
      }
      return !was;
    });
  return [open, toggle];
}

/**
 * What under More is not at its default, in a few words.
 *
 * The heading says it while More is shut, because a setting that moves every
 * number on screen must not be hidden by the act of tidying it away: pressing
 * a cluster plan sets TP and ZeRO, and the heading then says so.
 */
export function changedUnderMore(o: OperatingPoint): string[] {
  const d = DEFAULT_OPERATING;
  const out: string[] = [];
  if (o.dtype !== d.dtype) out.push(`train ${o.dtype}`);
  if (o.precision !== d.precision) out.push(o.precision);
  if (o.inferenceDtype !== d.inferenceDtype) out.push(`serve ${o.inferenceDtype}`);
  if (o.optimizer !== d.optimizer) out.push(OPTIMIZERS.find((x) => x.id === o.optimizer)?.label ?? o.optimizer);
  if (o.recompute !== d.recompute) out.push(`recompute ${o.recompute}`);
  if (o.zero !== d.zero) out.push(`ZeRO ${o.zero}`);
  if (o.tp !== d.tp) out.push(`TP ${o.tp}`);
  if (o.pp !== d.pp) out.push(`PP ${o.pp}`);
  if (o.ep !== d.ep) out.push(`EP ${o.ep}`);
  if (o.tp > 1 && o.sequenceParallel) out.push("sequence parallel");
  if (o.concurrency !== d.concurrency) out.push(`${o.concurrency} streams`);
  if (o.flash !== d.flash) out.push("unfused attention");
  if (o.tokens !== null) out.push(`${formatCount(o.tokens)} tokens`);
  return out;
}

export default function Operating(): React.ReactElement {
  const o = useEditor((s) => s.operating);
  const doc = useEditor((s) => s.doc);
  const set = useEditor((s) => s.setOperating);
  const reset = useEditor((s) => s.resetOperating);
  // With no override the analysis falls back to the design's own T; showing it
  // as the placeholder is how the field says what blank means.
  const options = useDerived().analysis.options;
  const effectiveT = options.T;
  // Only a design with a second sequence has a source length to set.
  const effectiveS = options.S;
  const dp = Math.max(1, Math.floor(o.gpus / Math.max(1, o.tp * o.pp)));
  const [open, toggle] = useRemembered("op.open", true);
  // Shut on a first visit: batch, sequence, device and GPUs are the four
  // things a first question about a design turns on, and the other twelve
  // are how a training run is set up — which matters, and is not where
  // anybody starts.
  const [more, toggleMore] = useRemembered("op.more", false);
  const changed = changedUnderMore(o);

  // Folded, the header still has to say what the numbers below it mean.
  const summary = [
    `B${o.B}`,
    `T${effectiveT.toLocaleString("en-US")}`,
    ...(effectiveS !== undefined ? [`S${effectiveS.toLocaleString("en-US")}`] : []),
    ...(o.packing ? [`packed ${o.packing.mean.toLocaleString("en-US")}`] : []),
    o.dtype,
    HARDWARE_BY_ID[o.hardware]?.name.replace(/ \(.*\)$/, "") ?? o.hardware,
    `${o.gpus}×`,
  ].join(" · ");

  const header = (
    <div className="op__head">
      <button className="op__toggle" onClick={toggle} aria-expanded={open}>
        <span className="fold__caret" aria-hidden>
          {open ? "▾" : "▸"}
        </span>
        <h3>Operating point</h3>
      </button>
      {open ? (
        <button className="linkish" onClick={reset} title="Back to the defaults">
          reset
        </button>
      ) : (
        <span className="op__summary mono">{summary}</span>
      )}
    </div>
  );

  if (!open) return <div className="op">{header}</div>;

  return (
    <div className="op is-open">
      {header}
      <div className="op__grid">
        <Num label="batch" value={o.B} onChange={(B) => set({ B })} title="Micro-batch per GPU." />
        <label className="op__field" title="Sequence length. Blank follows the design's own T.">
          <span className="op__label">sequence</span>
          <input
            className="field field--num"
            type="number"
            min={1}
            placeholder={String(effectiveT)}
            value={o.T ?? ""}
            spellCheck={false}
            onKeyDown={(e) => e.stopPropagation()}
            onChange={(e) => {
              const raw = e.target.value.trim();
              if (raw === "") return set({ T: null });
              const n = Number(raw);
              if (Number.isFinite(n) && n >= 1) set({ T: Math.floor(n) });
            }}
          />
        </label>
        {effectiveS !== undefined && (
          <label
            className="op__field"
            title="Source length: the second sequence, which the encoder runs along. Blank follows the design's own S."
          >
            <span className="op__label">source</span>
            <input
              className="field field--num"
              type="number"
              min={1}
              placeholder={String(effectiveS)}
              value={o.S ?? ""}
              spellCheck={false}
              aria-label="source length"
              onKeyDown={(e) => e.stopPropagation()}
              onChange={(e) => {
                const raw = e.target.value.trim();
                if (raw === "") return set({ S: null });
                const n = Number(raw);
                if (Number.isFinite(n) && n >= 1) set({ S: Math.floor(n) });
              }}
            />
          </label>
        )}
      </div>

      {/* Shown for a design with documents to pack, and whenever a packing is set,
          so one carried over from another design can always be taken off. */}
      {(o.packing !== null || readsDocuments(doc)) && (
        <div className="op__grid" data-testid="packing">
          <label
            className="op__field"
            title="Training rows packed with documents of this mean length, which the mask keeps apart. Blank is one document a row, which is what serving is."
          >
            <span className="op__label">docs</span>
            <input
              className="field field--num"
              type="number"
              min={1}
              placeholder="off"
              value={o.packing?.mean ?? ""}
              spellCheck={false}
              aria-label="mean document length"
              onKeyDown={(e) => e.stopPropagation()}
              onChange={(e) => {
                const raw = e.target.value.trim();
                if (raw === "") return set({ packing: null });
                const n = Number(raw);
                if (Number.isFinite(n) && n >= 1) set({ packing: { mean: Math.floor(n), spread: o.packing?.spread ?? 1 } });
              }}
            />
          </label>
          {o.packing !== null && (
            <label
              className="op__field"
              title="How much the document lengths vary: the coefficient of variation. 0 is every document the same length, 1 is exponential; real corpora are more."
            >
              <span className="op__label">spread</span>
              <input
                className="field field--num"
                type="number"
                min={0}
                step={0.1}
                value={o.packing.spread}
                spellCheck={false}
                aria-label="spread of document lengths"
                onKeyDown={(e) => e.stopPropagation()}
                onChange={(e) => {
                  const n = Number(e.target.value);
                  if (Number.isFinite(n) && n >= 0 && o.packing) set({ packing: { mean: o.packing.mean, spread: n } });
                }}
              />
            </label>
          )}
        </div>
      )}

      <div className="op__grid op__grid--wide">
        <Pick
          label="device"
          value={o.hardware}
          options={HARDWARE.map((h) => ({ id: h.id, label: h.name }))}
          onChange={(hardware) => set({ hardware })}
        />
        <Num label="GPUs" value={o.gpus} onChange={(gpus) => set({ gpus })} />
      </div>

      <button
        type="button"
        className="op__more"
        data-testid="operating-more"
        aria-expanded={more}
        onClick={toggleMore}
      >
        <span className="fold__caret" aria-hidden>
          {more ? "▾" : "▸"}
        </span>
        More
        <span className={"op__changed" + (changed.length ? " is-changed" : "")}>
          {changed.length ? changed.join(" · ") : "precision, optimizer, parallelism"}
        </span>
      </button>

      {more && (
        <>
          <div className="op__grid">
            <Pick
              label="train"
              value={o.dtype}
              options={DTYPES.map((d) => ({ id: d, label: d }))}
              onChange={(dtype) => set({ dtype })}
              title="Precision of weights and activations while training."
            />
            <Pick
              label="recipe"
              value={o.precision}
              options={[
                { id: "mixed", label: "mixed" },
                { id: "autocast", label: "autocast" },
              ]}
              onChange={(precision) => set({ precision })}
              title="How bf16 training is done. Mixed: bf16 weights and activations over an fp32 master copy, as Megatron does it. Autocast: plain PyTorch's torch.autocast, which keeps the residual stream and the norms in fp32 and a bf16 copy of every weight, and so saves more for the backward pass."
            />
            <Pick
              label="serve"
              value={o.inferenceDtype}
              options={DTYPES.map((d) => ({ id: d, label: d }))}
              onChange={(inferenceDtype) => set({ inferenceDtype })}
              title="Precision of the served weights and the cache."
            />
            <Pick
              label="optimizer"
              value={o.optimizer}
              options={OPTIMIZERS}
              onChange={(optimizer) => set({ optimizer })}
            />
            <Pick
              label="recompute"
              value={o.recompute}
              options={RECOMPUTE.map((r) => ({ id: r, label: r }))}
              onChange={(recompute) => set({ recompute })}
              title="Activation checkpointing: trade a second forward pass for activation memory."
            />
            <Pick
              label="ZeRO"
              value={o.zero}
              options={[0, 1, 2, 3].map((z) => ({
                id: z as 0 | 1 | 2 | 3,
                label: `stage ${z}`,
              }))}
              onChange={(zero) => set({ zero })}
              title="What the data-parallel group shards: nothing, optimizer state, then gradients, then weights."
            />
            <Pick
              label="TP"
              value={o.tp}
              options={POWERS.map((n) => ({ id: n, label: `${n}×` }))}
              onChange={(tp) => set({ tp })}
              title="Tensor-parallel degree."
            />
            <Pick
              label="PP"
              value={o.pp}
              options={POWERS.map((n) => ({ id: n, label: `${n}×` }))}
              onChange={(pp) => set({ pp })}
              title="Pipeline stages."
            />
            <Pick
              label="EP"
              value={o.ep}
              options={POWERS.map((n) => ({ id: n, label: `${n}×` }))}
              onChange={(ep) => set({ ep })}
              title="Expert-parallel degree. A design with no experts has nothing to divide."
            />
            <Num
              label="streams"
              value={o.concurrency}
              onChange={(concurrency) => set({ concurrency })}
              title="Concurrent sequences held in the cache while serving."
            />
          </div>

          <div className="op__foot">
            <label className="op__check" title="Assume a memory-efficient attention kernel.">
              <input
                type="checkbox"
                checked={o.flash}
                onChange={(e) => set({ flash: e.target.checked })}
              />
              fused attention
            </label>
            <label
              className="op__check"
              title={
                o.tp > 1
                  ? "Shard the activations along the sequence across the tensor-parallel group."
                  : "Needs tensor parallelism: there is no group to shard the sequence across."
              }
            >
              <input
                type="checkbox"
                checked={o.tp > 1 && o.sequenceParallel}
                disabled={o.tp <= 1}
                onChange={(e) => set({ sequenceParallel: e.target.checked })}
              />
              sequence parallel
            </label>
            <span
              className="op__derived mono"
              title="Data-parallel degree left after tensor and pipeline parallelism."
            >
              DP {dp}×
            </span>
            <label
              className="op__check"
              title="Training token budget. Off follows the Chinchilla-optimal budget."
            >
              <input
                type="checkbox"
                checked={o.tokens !== null}
                onChange={(e) => set({ tokens: e.target.checked ? 15e12 : null })}
              />
              token budget
            </label>
            {o.tokens !== null && (
              <input
                className="field field--num"
                type="number"
                min={1}
                step={1e11}
                value={o.tokens}
                spellCheck={false}
                title={formatCount(o.tokens)}
                onKeyDown={(e) => e.stopPropagation()}
                onChange={(e) => {
                  const n = Number(e.target.value);
                  if (Number.isFinite(n) && n >= 1) set({ tokens: n });
                }}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}
