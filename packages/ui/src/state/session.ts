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
import { parseDoc } from "./serialize.js";
import { storage, type ViewState } from "./storage.js";
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

/** A path that names a shared design: `/d/<id>`. */
const SHARED = /^\/d\/([A-Za-z0-9_-]{1,128})\/?$/;

/**
 * Open the design this URL names, if it names one.
 *
 * Called by whatever assembled the editor, after it has registered a provider
 * and after the engine has loaded — the store builds a design the moment its
 * module runs, so there is nothing to put a document into before then.
 *
 * Returns false when the URL is an ordinary one, when nothing is offering to
 * keep designs, or when the provider does not do sharing. Every one of those
 * is the normal case for a plain checkout, and none of them is an error: the
 * editor opens as it always does.
 *
 * A link that cannot be opened *is* worth saying out loud, though. Silently
 * showing the default design to somebody who followed a link is how a person
 * concludes the tool is broken.
 */
export async function openFromLocation(): Promise<boolean> {
  if (typeof location === "undefined") return false;
  const match = SHARED.exec(location.pathname);
  if (!match) return false;

  const provider = storage();
  if (!provider?.loadShared) return false;

  try {
    const got = await provider.loadShared(match[1]!);
    const doc = parseDoc(got.body);
    useEditor.getState().setDoc(doc, `Opened ${got.name}`);
    applyView(got.view, doc);
    return true;
  } catch (e) {
    useEditor.getState().setStatus(`That link did not open: ${(e as Error).message}`);
    return false;
  }
}
