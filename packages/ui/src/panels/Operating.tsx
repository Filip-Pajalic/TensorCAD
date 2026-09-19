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
import type { Recompute } from "../state/operating.js";
import type { Dtype, OptimizerKind } from "@tensorcad/engine";
import { formatCount } from "@tensorcad/engine";
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

export default function Operating(): React.ReactElement {
  const o = useEditor((s) => s.operating);
  const set = useEditor((s) => s.setOperating);
  const reset = useEditor((s) => s.resetOperating);
  // With no override the analysis falls back to the design's own T; showing it
  // as the placeholder is how the field says what blank means.
  const effectiveT = useDerived().analysis.options.T;
  const dp = Math.max(1, Math.floor(o.gpus / Math.max(1, o.tp * o.pp)));
  const [open, setOpen] = useState(() => {
    try {
      return localStorage.getItem("tensorcad.op.open") !== "0";
    } catch {
      return true;
    }
  });

  const toggle = (): void =>
    setOpen((was) => {
      try {
        localStorage.setItem("tensorcad.op.open", was ? "0" : "1");
      } catch {
        // Not remembering it is harmless.
      }
      return !was;
    });

  // Folded, the header still has to say what the numbers below it mean.
  const summary = [
    `B${o.B}`,
    `T${effectiveT.toLocaleString("en-US")}`,
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
          <span className="op__label">seq</span>
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
        <Pick
          label="train"
          value={o.dtype}
          options={DTYPES.map((d) => ({ id: d, label: d }))}
          onChange={(dtype) => set({ dtype })}
          title="Precision of weights and activations while training."
        />
        <Pick
          label="serve"
          value={o.inferenceDtype}
          options={DTYPES.map((d) => ({ id: d, label: d }))}
          onChange={(inferenceDtype) => set({ inferenceDtype })}
          title="Precision of the served weights and the cache."
        />
      </div>

      <div className="op__grid op__grid--wide">
        <Pick
          label="device"
          value={o.hardware}
          options={HARDWARE.map((h) => ({ id: h.id, label: h.name }))}
          onChange={(hardware) => set({ hardware })}
        />
        <Num label="GPUs" value={o.gpus} onChange={(gpus) => set({ gpus })} />
      </div>

      <div className="op__grid">
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
    </div>
  );
}
