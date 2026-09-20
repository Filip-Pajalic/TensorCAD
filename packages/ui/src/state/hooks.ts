import { useMemo, useSyncExternalStore } from "react";
import { useEditor } from "./store.js";
import { derive, prefixDerived, type Derived } from "./derive.js";
import { resolveLevel, type Level } from "./level.js";
import { DEF_PREFIX, previewDoc } from "./definition.js";
import { storage, subscribeStorage, type StorageProvider } from "./storage.js";

/**
 * The design's own numbers.
 *
 * The readout, the status bar and the findings dock all want these whatever
 * level is open: a definition being edited does not change what the design
 * weighs, and a readout that said otherwise would be answering a question
 * nobody asked.
 */
export function useDerived(): Derived {
  const doc = useEditor((s) => s.doc);
  const operating = useEditor((s) => s.operating);
  return useMemo(() => derive(doc, operating), [doc, operating]);
}

/**
 * The level on screen, and the numbers for what is drawn on it.
 *
 * These are the design's, except inside a definition. A template is analysed as
 * a design of its own — only then do its parameters have values — and the
 * answers are re-keyed under the definition's path, so the canvas and the
 * inspector go on asking the way they always have.
 */
export function useLevel(): { level: Level; derived: Derived } {
  const doc = useEditor((s) => s.doc);
  const path = useEditor((s) => s.path);
  const operating = useEditor((s) => s.operating);
  const design = useDerived();

  // Memoised on the document and the name, so the preview is one object across
  // renders and `derive`'s cache holds.
  const type = path[0] === DEF_PREFIX ? (path[1] ?? "") : null;
  const preview = useMemo(() => (type === null ? null : previewDoc(doc, type)), [doc, type]);

  const derived = useMemo(() => {
    if (preview === null || type === null) return design;
    return prefixDerived(derive(preview, operating), `${DEF_PREFIX}/${type}`);
  }, [preview, type, operating, design]);

  const level = useMemo(() => resolveLevel(doc, path, derived), [doc, path, derived]);
  return { level, derived };
}

/**
 * The place designs are kept, or null when nothing is offering.
 *
 * A plain checkout has no provider and every caller of this renders nothing,
 * which is how the editor stays exactly what it was when nobody has plugged a
 * store in.
 */
export function useStorage(): StorageProvider | null {
  return useSyncExternalStore(subscribeStorage, storage, storage);
}
