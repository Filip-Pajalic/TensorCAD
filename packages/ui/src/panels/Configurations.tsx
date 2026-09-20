/**
 * One design, several sizes.
 *
 * The library carries `gpt2-small`, `gpt2-medium`, `gpt2-large` and `gpt2-xl`
 * as four files. They are the same architecture — the same blocks, the same
 * wiring, the same expression for the feed-forward width — and they differ in
 * four numbers. A configuration is those numbers, named, so the architecture is
 * written once.
 *
 * Symbols only, deliberately. A variant that changed the graph would be a
 * different design, and calling it a configuration would be a way of losing
 * track of that.
 *
 * It sits above the symbol table because that is what it selects between, and
 * because a number you are about to edit should say first which size you are
 * editing it at: while a configuration is in force, an edit lands in it rather
 * than in the design underneath.
 */

import { useState } from "react";
import { useEditor } from "../state/store.js";

export default function Configurations(): React.ReactElement | null {
  const doc = useEditor((s) => s.doc);
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");

  const names = Object.keys(doc.configurations ?? {});
  const active = doc.active ?? "";

  const save = (): void => {
    const wanted = name.trim();
    if (!wanted) return;
    useEditor.getState().captureConfiguration(wanted);
    setName("");
    setNaming(false);
  };

  // Nothing to select between and nothing named yet: one control rather than an
  // empty list, because a picker with no options is furniture.
  if (names.length === 0 && !naming) {
    return (
      <div className="configs">
        <button
          className="configs__add"
          onClick={() => setNaming(true)}
          title="Name these symbol values, so the design can carry more than one size"
        >
          Save these symbols as a configuration…
        </button>
      </div>
    );
  }

  return (
    <div className="configs">
      {names.length > 0 && (
        <div className="configs__row">
          <span className="configs__label">Built at</span>
          <select
            className="configs__select"
            value={active}
            onChange={(e) => useEditor.getState().setActiveConfiguration(e.target.value || null)}
          >
            {/* The design's own symbols are always an option: a configuration
                is a view of them, and there has to be a way back. */}
            <option value="">as written</option>
            {names.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          {active !== "" && (
            <button
              className="configs__btn"
              onClick={() => useEditor.getState().removeConfiguration(active)}
              title="Forget this configuration. The design's own symbols are not touched."
            >
              remove
            </button>
          )}
          <button className="configs__btn" onClick={() => setNaming(true)} title="Name these values">
            save as…
          </button>
        </div>
      )}

      {naming && (
        <div className="configs__row">
          <input
            className="configs__name"
            value={name}
            placeholder="name, e.g. 7b"
            autoFocus
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") save();
              if (e.key === "Escape") setNaming(false);
            }}
          />
          <button className="configs__btn" onClick={save} disabled={name.trim() === ""}>
            save
          </button>
          <button className="configs__btn" onClick={() => setNaming(false)}>
            cancel
          </button>
        </div>
      )}

      {active !== "" && (
        <div className="configs__note">
          Editing a symbol below changes this configuration, not the design underneath.
        </div>
      )}
    </div>
  );
}
