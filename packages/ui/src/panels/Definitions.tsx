/**
 * The design's own block library.
 *
 * A design carries its composites in `doc.defs`, and until now you could make
 * one, import a file of them and export one — but not look at what you had.
 * This is the list: what each block declares, where it is used, and the three
 * things you do to one you already have.
 *
 * "Edit" opens the template itself, not an instance of it, so a change reaches
 * every instance at once — which is what a definition is for. "Show" goes to an
 * instance instead, for when the question is what this block does *here*.
 *
 * Deleting is refused while a block is in use, because the alternative is a
 * design full of blocks the catalog has never heard of.
 */

import { useMemo } from "react";
import { useEditor } from "../state/store.js";
import { defsOf, renameBlock, usageOf, withoutBlock } from "../state/blocks.js";
import { DEF_PREFIX } from "../state/definition.js";
import { validateUserBlock } from "../engine.js";
import type { Doc, Graph, UserBlockDef } from "@tensor-cad/engine";

function names(record: Record<string, unknown> | undefined): string[] {
  return Object.keys(record ?? {});
}

function One({ type, def, doc }: { type: string; def: UserBlockDef; doc: Doc }): React.ReactElement {
  const { paths, inDefs } = usageOf(doc, type);
  const used = paths.length + inDefs;
  const problems = validateUserBlock(def, type);
  const state = useEditor.getState();

  const edit = (): void => {
    state.setPath([DEF_PREFIX, type]);
    state.closeDialog();
  };

  // And an instance, for when the question is what this block does in one
  // particular place rather than what it is.
  const show = (): void => {
    const first = paths[0];
    if (first === undefined) return;
    state.setPath(first.split("/"));
    state.select(first);
    state.closeDialog();
  };

  const rename = (): void => {
    const wanted = window.prompt(`Rename "${type}" to`, type);
    if (!wanted) return;
    const renamed = renameBlock(doc, type, wanted);
    if (renamed.to === type) return;
    state.setDoc(renamed.doc, `Renamed "${type}" to "${renamed.to}"`);
  };

  const remove = (): void => {
    if (used > 0) {
      state.setStatus(`"${type}" is used ${used} time${used === 1 ? "" : "s"}; remove those first.`);
      return;
    }
    state.setDoc(withoutBlock(doc, type), `Deleted the block "${type}"`);
  };

  return (
    <div className="def">
      <div className="def__head">
        <span className="def__name mono">{type}</span>
        <span className="def__used">
          {used === 0 ? "unused" : `${used} instance${used === 1 ? "" : "s"}`}
          {inDefs > 0 && `, ${inDefs} inside another definition`}
        </span>
        <span className="def__actions">
          <button className="def__btn" onClick={edit} title="Edit this block's template">
            edit
          </button>
          <button
            className="def__btn"
            onClick={show}
            disabled={paths.length === 0}
            title={
              paths.length === 0
                ? "Nothing on the canvas uses it, so there is no instance to open. Place one from the palette."
                : "Open the first one on the canvas"
            }
          >
            show
          </button>
          <button className="def__btn" onClick={rename} title="Rename it, and every instance with it">
            rename
          </button>
          <button
            className="def__btn"
            onClick={remove}
            disabled={used > 0}
            title={used > 0 ? "In use; remove the instances first" : "Delete this definition"}
          >
            delete
          </button>
        </span>
      </div>
      {def.docs?.summary && <div className="def__summary">{def.docs.summary}</div>}
      <div className="def__facts">
        <span>
          <b>in</b> {names(def.ports?.in).join(", ") || "—"}
        </span>
        <span>
          <b>out</b> {names(def.ports?.out).join(", ") || "—"}
        </span>
        <span>
          <b>parameters</b> {names(def.params).join(", ") || "none"}
        </span>
        <span>
          <b>template</b> {(def.graph as Graph | undefined)?.nodes.length ?? 0} blocks
        </span>
      </div>
      {problems.map((p) => (
        <div className="def__problem" key={p}>
          {p}
        </div>
      ))}
    </div>
  );
}

export default function Definitions(): React.ReactElement {
  const doc = useEditor((s) => s.doc);
  const defs = useMemo(() => Object.entries(defsOf(doc)).sort(), [doc]);

  if (defs.length === 0) {
    return (
      <div className="empty">
        This design defines no blocks of its own. Open a level and use{" "}
        <b>Blocks &gt; Make a block from this level</b> to turn what is on screen into one, or
        import a library.
      </div>
    );
  }

  return (
    <div className="defs">
      {defs.map(([type, def]) => (
        <One key={type} type={type} def={def} doc={doc} />
      ))}
    </div>
  );
}
