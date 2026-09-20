/**
 * Somewhere for designs to live, if anything is offering.
 *
 * The editor keeps no designs. `File > Open` reads a file you chose and
 * `Save a copy` hands one back, and reloading the tab starts again from a
 * preset — which is honest for a tool with no server, and useless the moment
 * you want a design to still be there tomorrow.
 *
 * This is the seam a store plugs into. The editor knows there may be *a*
 * place to keep designs; it does not know what or where, and nothing here
 * names a vendor, a protocol or a host. A deployment registers a provider at
 * startup; a plain checkout registers none and behaves exactly as it always
 * has.
 *
 * ## Why an interface rather than a client
 *
 * Two reasons, and the second is the real one.
 *
 * It keeps this repository free of anybody's authentication. The panel below
 * renders whatever a provider reports and cannot tell you where it came from.
 *
 * And it is a better shape anyway. "Bring your own backend" is a thing people
 * ask of an editor like this — a shared drive, a git repository, a browser's
 * own storage — and every one of those is a few dozen lines against this
 * interface rather than a fork.
 *
 * ## The document is still the source of truth
 *
 * A provider stores **text**, produced by `serializeDoc` and read by
 * `parseDoc`. It is not given a `Doc` and is not expected to understand one.
 * That keeps invariant 1 intact — the document is the IR, and everything else
 * is a view of it — and it means a provider cannot quietly disagree with the
 * engine about what a design is.
 */

import type { OperatingPoint } from "./operating.js";
import type { ShapeMode } from "../canvas/shapes.js";

/**
 * Where you were, as distinct from what you had.
 *
 * Reopening a design at the top of the sheet with nothing selected is
 * technically the same design and not the same place. This is the small,
 * cheap part of a session: the level, what was selected, how far unfolded, and
 * the conditions the numbers were being measured under.
 *
 * Deliberately *not* the edit history. That is `base` plus `steps` plus a
 * hundred materialised documents, and restoring it across a change to the
 * document format is a promise this cannot keep.
 */
export interface ViewState {
  /** Breadcrumb path of the open graph level. */
  path?: string[];
  selection?: string | null;
  selectedNet?: string | null;
  detail?: number;
  viewMode?: "sheet" | "volume";
  shapeMode?: ShapeMode;
  operating?: OperatingPoint;
}

/** One design, as a list needs to show it. */
export interface StoredDesign {
  id: string;
  name: string;
  /** ISO 8601. What a list sorts by. */
  updatedAt: string;
  /** Present when this design has been shared. */
  shareUrl?: string;
}

/** Who is signed in, for the corner of the screen. Null means nobody. */
export interface Account {
  name: string;
  email?: string;
  avatarUrl?: string;
  signOutUrl?: string;
}

/**
 * A place designs can be kept.
 *
 * Everything returns a promise, including the parts a local implementation
 * could answer instantly — a caller that has to know which is which is a
 * caller that breaks when the provider changes.
 */
export interface StorageProvider {
  /** A name for the place, shown in the panel: "Cloud", "This browser". */
  readonly label: string;

  list(): Promise<StoredDesign[]>;
  load(id: string): Promise<{ body: string; name: string; view?: ViewState }>;
  /**
   * Write a design. A null id means one that has never been saved, and the
   * returned id is what to use from then on — the same convention the desktop
   * shell already uses, where an empty path means "never saved".
   */
  save(id: string | null, name: string, body: string, view: ViewState): Promise<{ id: string }>;
  remove(id: string): Promise<void>;

  /** Mint or return a link anyone can open. Absent means sharing is not offered. */
  share?(id: string): Promise<{ url: string }>;
  /** Read a shared design without being signed in. */
  loadShared?(shareId: string): Promise<{ body: string; name: string; view?: ViewState }>;
  /** Who is signed in. Absent means the provider has no notion of accounts. */
  account?(): Account | null;
}

let provider: StorageProvider | null = null;
const listeners = new Set<() => void>();

/**
 * Offer a place to keep designs.
 *
 * Called once at startup, before the first frame, by whatever is assembling
 * this editor. Calling it again replaces the provider, which is what a sign-out
 * that drops back to local behaviour would do.
 */
export function registerStorage(next: StorageProvider | null): void {
  provider = next;
  for (const listener of listeners) listener();
}

/** The provider, or null when nothing is offering. */
export function storage(): StorageProvider | null {
  return provider;
}

/** For React. Re-renders when a provider arrives or goes away. */
export function subscribeStorage(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
