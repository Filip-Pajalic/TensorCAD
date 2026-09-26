/**
 * New design: a kind of model and a size, rather than an empty sheet.
 *
 * Two questions, because they are the two things somebody starting out knows:
 * what sort of model, and how big — as a parameter count, or as whatever
 * trains on the one GPU they have. The engine answers with the reference design
 * of that kind scaled to the size, proportions kept, and the dialog shows what
 * that came to before anything is created: its depth and width, its heads, and
 * what training it takes on one device, measured the way the readout will
 * measure it afterwards.
 *
 * The blank sheet is still here, one button away, for somebody who wants it.
 */

import { useEffect, useMemo, useState } from "react";
import { formatBytes, formatCount, type NewDesignRequest, type NewDesignResult } from "@tensor-cad/engine";
import { HARDWARE, HARDWARE_BY_ID, modelFamilies, newDesign } from "../engine.js";
import { useEditor } from "../state/store.js";
import { toAnalysisOptions } from "../state/operating.js";
import { runCommand } from "../state/commands.js";
import { Input } from "../ui/input.js";
import { Button } from "../ui/button.js";

type Size = { kind: "params"; params: number } | { kind: "fit"; device: string };

/** A size the way a model is named: 125M, 1B, 1.5B. */
function sizeLabel(n: number): string {
  const [unit, div] = n >= 1e9 ? ["B", 1e9] : n >= 1e6 ? ["M", 1e6] : ["K", 1e3];
  const v = n / div;
  return `${Number.isInteger(v) || v >= 10 ? Math.round(v) : v.toFixed(1)}${unit}`;
}

/** "2.5B", "500m", "3e9", "1,200,000" — whatever a person types for a size. */
export function parseSize(text: string): number | null {
  const m = /^\s*([\d.,]+(?:e\d+)?)\s*([kmbt])?\s*$/i.exec(text);
  if (!m) return null;
  const n = Number(m[1]!.replace(/,/g, ""));
  if (!Number.isFinite(n) || n <= 0) return null;
  const mult = { k: 1e3, m: 1e6, b: 1e9, t: 1e12 }[(m[2] ?? "").toLowerCase()] ?? 1;
  return n * mult;
}

/** The size offered first: the largest up to a billion, which trains somewhere. */
function defaultSize(sizes: number[]): number {
  const small = sizes.filter((s) => s <= 1e9);
  return small.length ? small[small.length - 1]! : sizes[0]!;
}

/** The new design's shape, in words. */
export function shapeInWords(symbols: Record<string, number>): string {
  const n = (v: number) => v.toLocaleString("en-US");
  const parts: string[] = [];
  if (symbols.L !== undefined) parts.push(`${n(symbols.L)} layers`);
  if (symbols.D !== undefined) parts.push(`width ${n(symbols.D)}`);
  if (symbols.H !== undefined) {
    parts.push(
      symbols.Hkv !== undefined && symbols.Hkv !== symbols.H
        ? `${n(symbols.H)} heads sharing ${n(symbols.Hkv)} key/value head${symbols.Hkv === 1 ? "" : "s"}`
        : `${n(symbols.H)} heads`,
    );
  }
  if (symbols.E !== undefined) {
    parts.push(`${n(symbols.E)} experts, ${n(symbols.K ?? 1)} per token`);
  } else if (symbols.F !== undefined) {
    parts.push(`feed-forward ${n(symbols.F)}`);
  }
  return parts.join(" · ");
}

export default function NewDesign(): React.ReactElement {
  const families = useMemo(() => modelFamilies(), []);
  const operating = useEditor((s) => s.operating);
  const [familyId, setFamilyId] = useState(families[0]!.id);
  const family = families.find((f) => f.id === familyId) ?? families[0]!;
  const [size, setSize] = useState<Size>({ kind: "params", params: defaultSize(family.sizes) });
  const [custom, setCustom] = useState("");
  const [name, setName] = useState("");
  const [walk, setWalk] = useState(true);
  const [outcome, setOutcome] = useState<{ result: NewDesignResult | null; error: string | null }>({
    result: null,
    error: null,
  });
  const [busy, setBusy] = useState(false);

  const pick = (id: string): void => {
    setFamilyId(id);
    const next = families.find((f) => f.id === id)!;
    // A size chosen for one kind is kept for the next when it was typed or a
    // device; a chip belongs to the kind that offered it.
    if (size.kind === "params" && !custom) setSize({ kind: "params", params: defaultSize(next.sizes) });
  };

  const request: NewDesignRequest =
    size.kind === "params"
      ? { family: family.id, params: size.params, ...(name.trim() ? { name: name.trim() } : {}) }
      : { family: family.id, fit: size.device, ...(name.trim() ? { name: name.trim() } : {}) };
  const key = JSON.stringify(request);

  // Measured when a choice changes, a moment later, so the dialog has drawn the
  // choice before a fit's search holds the thread.
  useEffect(() => {
    setBusy(true);
    const id = setTimeout(() => {
      try {
        setOutcome({ result: newDesign(request, toAnalysisOptions(operating)), error: null });
      } catch (e) {
        setOutcome({ result: null, error: (e as Error).message });
      }
      setBusy(false);
    }, 30);
    return () => clearTimeout(id);
    // The request as a value, not the object remade on every render.
  }, [key, operating]);

  const result = outcome.result;
  const device = result ? (HARDWARE_BY_ID[result.device]?.name ?? result.device) : "";

  const create = (): void => {
    if (!result) return;
    const editor = useEditor.getState();
    editor.setDoc(result.doc, `Started ${result.doc.meta.name}: ${family.name.toLowerCase()}, ${formatCount(result.params)} parameters`);
    editor.closeDialog();
    if (walk) runCommand("help.start");
  };

  return (
    <div className="newdesign" data-testid="new-design">
      <section className="newdesign__section">
        <h3 className="newdesign__question">What kind of model?</h3>
        <div className="newdesign__kinds" role="radiogroup" aria-label="Kind of model">
          {families.map((f) => (
            <button
              key={f.id}
              type="button"
              role="radio"
              aria-checked={f.id === family.id}
              className={"newdesign__kind" + (f.id === family.id ? " is-on" : "")}
              data-family={f.id}
              onClick={() => pick(f.id)}
            >
              <span className="newdesign__kindname">{f.name}</span>
              <span className="newdesign__kindsummary">{f.summary}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="newdesign__section">
        <h3 className="newdesign__question">How big?</h3>
        <div className="newdesign__sizes" role="radiogroup" aria-label="Size">
          {family.sizes.map((s) => {
            const on = size.kind === "params" && !custom && size.params === s;
            return (
              <button
                key={s}
                type="button"
                role="radio"
                aria-checked={on}
                className={"newdesign__size" + (on ? " is-on" : "")}
                onClick={() => {
                  setCustom("");
                  setSize({ kind: "params", params: s });
                }}
              >
                {sizeLabel(s)}
              </button>
            );
          })}
          <Input
            className={"newdesign__custom" + (custom ? " is-on" : "")}
            placeholder="or type one: 2.5B"
            value={custom}
            spellCheck={false}
            aria-label="A size of your own, in parameters"
            onChange={(e) => {
              setCustom(e.target.value);
              const n = parseSize(e.target.value);
              if (n !== null) setSize({ kind: "params", params: n });
            }}
          />
        </div>
        <label className="newdesign__fit">
          <input
            type="radio"
            checked={size.kind === "fit"}
            onChange={() => {
              setCustom("");
              setSize({ kind: "fit", device: size.kind === "fit" ? size.device : "rtx4090" });
            }}
          />
          <span>As large as trains on one</span>
          <select
            className="field"
            value={size.kind === "fit" ? size.device : "rtx4090"}
            aria-label="The device it has to train on"
            onChange={(e) => setSize({ kind: "fit", device: e.target.value })}
            onFocus={() => size.kind !== "fit" && setSize({ kind: "fit", device: "rtx4090" })}
          >
            {HARDWARE.map((h) => (
              <option key={h.id} value={h.id}>
                {h.name}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className={"newdesign__preview" + (busy ? " is-busy" : "")} aria-live="polite" data-testid="new-design-preview">
        {outcome.error ? (
          <p className="newdesign__error">{outcome.error}</p>
        ) : result ? (
          <>
            <div className="newdesign__headline">
              <span className="newdesign__params">{formatCount(result.params)}</span>
              <span className="muted">parameters</span>
              <span className="newdesign__base mono" title="The reference design it is scaled from">
                from {result.base}
              </span>
            </div>
            <p className="newdesign__shape">{shapeInWords(result.symbols)}</p>
            <p className={"newdesign__train" + (result.fits ? "" : " is-over")}>
              {result.fits ? "Trains" : "Does not train"} on one {device}: {formatBytes(result.trainBytes)} of{" "}
              {formatBytes(result.budget)}
              {!result.fits && ", so it needs several — the Cluster view says how to split it"}.
            </p>
            {result.notes.length > 0 && (
              <ul className="newdesign__notes">
                {result.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            )}
          </>
        ) : (
          <p className="muted">Measuring…</p>
        )}
      </section>

      <div className="newdesign__foot">
        <Input
          className="newdesign__name"
          placeholder={result?.doc.meta.name ?? "name"}
          value={name}
          spellCheck={false}
          aria-label="Name"
          onChange={(e) => setName(e.target.value)}
        />
        <label className="newdesign__walk">
          <input type="checkbox" checked={walk} onChange={(e) => setWalk(e.target.checked)} />
          Walk me through it
        </label>
        <span className="flex-1" />
        <Button
          variant="ghost"
          onClick={() => {
            useEditor.getState().newDoc();
            useEditor.getState().closeDialog();
          }}
        >
          Blank sheet
        </Button>
        <Button variant="default" disabled={!result || busy} onClick={create} data-testid="create-design">
          Create
        </Button>
      </div>
    </div>
  );
}
