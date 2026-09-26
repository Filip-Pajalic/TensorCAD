/**
 * Sharing the open design, from one button.
 *
 * Signed in to a store that shares, the design is saved there and the store's
 * own link is what goes out: short, view-only, and still there when the design
 * changes. Anywhere else — a plain checkout, the public editor, somebody not
 * signed in — the design goes in the link itself, which needs nobody's server
 * and opens a copy the recipient can edit.
 *
 * Either way it is one press, and the person pressing it is told which kind of
 * link they are holding.
 */

import { useEditor } from "./store.js";
import { serializeDoc } from "./serialize.js";
import { captureView } from "./session.js";
import { openDesignId, setOpenDesign, storage } from "./storage.js";
import { encodeDesign, linkBase } from "./link.js";

export interface Shared {
  url: string;
  /** `stored` is a store's link to a saved design; `inline` carries the design itself. */
  kind: "stored" | "inline";
}

/** Whether Share will go through a store rather than into the link. */
export function sharesThroughStore(): boolean {
  const provider = storage();
  return !!(provider?.share && provider.account?.());
}

export async function shareDesign(): Promise<Shared> {
  const doc = useEditor.getState().doc;
  const provider = storage();
  if (provider?.share && provider.account?.()) {
    const { id } = await provider.save(openDesignId(), doc.meta.name, serializeDoc(doc), captureView());
    setOpenDesign(id);
    const { url } = await provider.share(id);
    return { url, kind: "stored" };
  }
  return { url: `${linkBase()}#design=${await encodeDesign(serializeDoc(doc))}`, kind: "inline" };
}
