/**
 * The key: what the letters and the marks on this sheet mean.
 *
 * A drawing that writes `B T (H dh)` on a wire is unreadable to anyone who has
 * not been told what the four letters are, and until now nothing in the editor
 * told anyone. The symbols are already computed — `derive()` returns the whole
 * table with a documentation string per name — so two thirds of this is the
 * design describing itself and cannot go stale. Only the grammar is authored.
 *
 * Every mark is drawn from the same custom properties the canvas draws with,
 * for the same reason the callouts are derived: a legend that restates the
 * stylesheet in its own words is a legend that will one day be wrong.
 */

import { useEditor } from "../state/store.js";
import { useLevel } from "../state/hooks.js";
import { dtypeColor } from "./blocks.js";

/** The element types a pin is coloured by, in the order the catalog declares. */
const DTYPES: [string, string][] = [
  ["int64", "token ids and indices"],
  ["fp32", "activations, full width"],
  ["bf16", "activations, half width"],
  ["fp8", "a narrow quantized type"],
  ["bool", "a mask"],
];

function Wire({ kind, label }: { kind: string; label: string }): React.ReactElement {
  return (
    <div className="key__row">
      <svg className="key__swatch" viewBox="0 0 34 8" aria-hidden>
        <path className={`key__wire key__wire--${kind}`} d="M1 4 H33" />
      </svg>
      <span>{label}</span>
    </div>
  );
}

export default function Key(): React.ReactElement {
  const open = useEditor((s) => s.showKey);
  const setOpen = useEditor((s) => s.setShowKey);
  // A walkthrough is the guided version of the same job, and two panels over
  // one sheet is one too many. The key shuts to its tab rather than going away,
  // and without touching the stored preference — closing the walkthrough puts
  // it back exactly as it was.
  const walking = useEditor((s) => s.walkthrough !== null);
  const { derived } = useLevel();
  const symbols = derived.symbols;

  if (!open || walking) {
    return (
      <button
        type="button"
        className="key key--shut"
        onClick={() => {
          useEditor.getState().endWalkthrough();
          setOpen(true);
        }}
      >
        Key
      </button>
    );
  }

  return (
    <div className="key">
      <div className="key__bar">
        <span className="key__title">Key</span>
        <button
          type="button"
          className="key__shut"
          onClick={() => setOpen(false)}
          aria-label="close the key"
        >
          &times;
        </button>
      </div>

      <div className="key__group">
        <div className="key__heading">Symbols</div>
        {symbols.order.map((name) => {
          const value = symbols.values[name];
          // B and T are conditions of a run, not properties of the design, and
          // they stay symbolic through shape checking. Saying so here is the
          // whole reason a reader can make sense of `B T D`.
          const runtime = symbols.designValues[name] === undefined;
          return (
            <div className="key__row" key={name}>
              <span className="key__sym mono">{name}</span>
              <span>
                {symbols.docs[name] || "no description"}
                {Number.isFinite(value) && (
                  <span className="key__val mono"> = {value.toLocaleString("en-US")}</span>
                )}
                {runtime && (
                  <span className="key__note"> — a condition of the run, not of the design</span>
                )}
              </span>
            </div>
          );
        })}
      </div>

      <div className="key__group">
        <div className="key__heading">Shapes</div>
        <p className="key__prose">
          A shape is its axes in order, largest first: <span className="mono">B T D</span> is a
          batch of sequences of vectors. Brackets are one axis folded out of several, so{" "}
          <span className="mono">B T (H dh)</span> and <span className="mono">B H T dh</span> hold
          the same numbers laid out differently &mdash; which is all a{" "}
          <span className="key__sym">reshape</span> ever does.
        </p>
      </div>

      <div className="key__group">
        <div className="key__heading">Lines</div>
        <Wire kind="signal" label="the main path" />
        <Wire kind="bypass" label="a line that skips a stage" />
        <Wire kind="index" label="carries indices, not activations" />
      </div>

      <div className="key__group">
        <div className="key__heading">Marks</div>
        <div className="key__row">
          <span className="key__swatch key__swatch--mark">
            <i className="key__junction" />
          </span>
          <span>the net branches here</span>
        </div>
        <div className="key__row">
          <span className="key__swatch key__swatch--mark">
            <i className="key__dangling" />
          </span>
          <span>nothing is connected</span>
        </div>
        <div className="key__row">
          <span className="key__swatch key__swatch--mark key__glyph">&#8853;</span>
          <span>add, which is how a residual rejoins</span>
        </div>
        <div className="key__row">
          <span className="key__swatch key__swatch--mark key__glyph">&#8855;</span>
          <span>multiply, which is how a gate applies</span>
        </div>
      </div>

      <div className="key__group">
        <div className="key__heading">Pin colour</div>
        {DTYPES.map(([dtype, what]) => (
          <div className="key__row" key={dtype}>
            <span className="key__swatch key__swatch--mark">
              <i className="key__dot" style={{ background: dtypeColor(dtype) }} />
            </span>
            <span>
              <span className="mono">{dtype}</span> &mdash; {what}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
