/**
 * Where you were, captured and restored.
 *
 * The document says what you had. This says where you were in it, which is the
 * difference between reopening a design and reopening your work: the level you
 * had drilled into, what was selected, how far the containers were unfolded,
 * and the operating point every number was being measured under.
 *
 * It is composed out of the store's existing actions rather than writing state
 * directly, so restoring a session goes through the same doors an edit does —
 * `setDetail` still remembers the detail level, `setOperating` still persists
 * the operating point, and a level that no longer exists is still refused by
 * `setPath`'s own rules rather than by a second copy of them here.
 */

import { useEditor } from "./store.js";
import * as ops from "./ops.js";
import type { ViewState } from "./storage.js";
import type { Doc } from "@tensorcad/engine";

/** What the editor would need to put you back. */
export function captureView(): ViewState {
  const s = useEditor.getState();
  return {
    path: s.path,
    selection: s.selection,
    selectedNet: s.selectedNet,
    detail: s.detail,
    viewMode: s.viewMode,
    shapeMode: s.shapeMode,
    operating: s.operating,
  };
}

/**
 * Put it back, as far as the document allows.
 *
 * Every part is optional and every part is checked, because a view is stored
 * beside a document and the two can drift: a design edited elsewhere may no
 * longer have the block that was selected, or the level that was open. The
 * rule is that a stale part is dropped rather than throwing away the whole
 * restore — landing on the top sheet with the right operating point is much
 * better than landing nowhere.
 */
export function applyView(view: ViewState | undefined, doc: Doc): void {
  if (!view) return;
  const editor = useEditor.getState();

  if (view.operating) editor.setOperating(view.operating);
  if (typeof view.detail === "number") editor.setDetail(view.detail);
  if (view.viewMode) editor.setViewMode(view.viewMode);
  if (view.shapeMode) editor.setShapeMode(view.shapeMode);

  // The level, only if it is still a level.
  if (view.path?.length && ops.graphAtPath(doc, view.path)) {
    editor.setPath(view.path);
  }

  // Selection last: `setPath` clears it, so restoring it first would be undone.
  if (view.selection && ops.nodeAtPath(doc, ops.segmentsOf(view.selection))) {
    useEditor.getState().select(view.selection);
  } else if (view.selectedNet) {
    const at = view.selectedNet.lastIndexOf(":");
    const producer = at > 0 ? view.selectedNet.slice(0, at) : "";
    if (producer && ops.nodeAtPath(doc, ops.segmentsOf(producer))) {
      useEditor.getState().selectNet(view.selectedNet);
    }
  }
}
