/**
 * The parts library.
 *
 * Every catalog block, grouped by category and searchable. Drag onto the canvas
 * or click to drop one in the middle of the current level.
 */

import { useMemo, useState } from "react";
import { useEditor } from "../state/store.js";
import { useLevel } from "../state/hooks.js";
import { newNodeFor } from "../state/addBlock.js";
import { categoryColor } from "../canvas/blocks.js";
import { DRAG_MIME } from "../canvas/Canvas.js";
import { catalogByCategory, isUserBlock, type BlockDef } from "../engine.js";

const KIND_MARK: Record<string, string> = {
  primitive: "P",
  composite: "C",
  container: "▣",
};

function matches(def: BlockDef, needle: string): boolean {
  if (!needle) return true;
  const hay = `${def.type} ${def.category} ${def.docs.summary}`.toLowerCase();
  return hay.includes(needle);
}

export default function Palette(): React.ReactElement {
  const [query, setQuery] = useState("");
  const { level } = useLevel();
  const open = useEditor((s) => s.paletteOpen);
  const toggle = useEditor((s) => s.togglePalette);
  const doc = useEditor((s) => s.doc);
  // Keyed on the document, so a block defined a moment ago is in the list.
  const groups = useMemo(() => catalogByCategory(doc), [doc]);
  const needle = query.trim().toLowerCase();

  const add = (type: string): void => {
    if (!level.editable) {
      useEditor.getState().setStatus("This level is read-only");
      return;
    }
    const node = newNodeFor(type);
    if (!node) return;
    const count = level.graph.nodes.length;
    useEditor.getState().addNode(level.segments, node, [40 + (count % 4) * 260, 40 + count * 24]);
  };

  const visible = Object.entries(groups)
    .map(([category, defs]) => [category, defs.filter((d) => matches(d, needle))] as const)
    .filter(([, defs]) => defs.length > 0);

  if (!open) {
    return (
      <div className="panel palette palette--closed">
        <button className="panel__head panel__head--button" onClick={toggle} title="Open the parts library">
          <span className="fold__caret" aria-hidden>
            ▸
          </span>
          <h2>Palette</h2>
          <span className="muted mono">{Object.values(groups).flat().length} blocks</span>
        </button>
      </div>
    );
  }

  return (
    <div className="panel palette">
      <button className="panel__head panel__head--button" onClick={toggle} title="Close the parts library">
        <span className="fold__caret" aria-hidden>
          ▾
        </span>
        <h2>Palette</h2>
        <span className="muted mono">{Object.values(groups).flat().length} blocks</span>
      </button>
      <div className="palette__search">
        <input
          className="field"
          placeholder="Search blocks..."
          value={query}
          spellCheck={false}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.stopPropagation()}
        />
      </div>
      {!level.editable && <div className="notice">Read-only level &mdash; go up to add blocks.</div>}
      <div className="panel__body">
        {visible.map(([category, defs]) => (
          <div className="palette__group" key={category}>
            <div className="palette__category" style={{ ["--accent" as string]: categoryColor(category) }}>
              {category}
            </div>
            {defs.map((def) => (
              <div
                key={def.type}
                className="palette__item"
                style={{ ["--accent" as string]: categoryColor(def.category) }}
                draggable={level.editable}
                title={def.docs.summary + (def.docs.formula ? `\n\n${def.docs.formula}` : "")}
                onDragStart={(e) => {
                  e.dataTransfer.setData(DRAG_MIME, def.type);
                  e.dataTransfer.effectAllowed = "move";
                }}
                onClick={() => add(def.type)}
              >
                <span className="palette__kind" title={def.kind}>
                  {KIND_MARK[def.kind] ?? "?"}
                </span>
                <span className="palette__type mono">{def.type}</span>
                {isUserBlock(doc, def.type) && (
                  <span className="palette__own" title="Defined by this design">
                    own
                  </span>
                )}
                <span className="palette__summary">{def.docs.summary}</span>
              </div>
            ))}
          </div>
        ))}
        {visible.length === 0 && <div className="empty">No block matches &ldquo;{query}&rdquo;.</div>}
      </div>
    </div>
  );
}
