/**
 * Designs kept somewhere.
 *
 * This panel exists only when something is offering to keep them — a plain
 * checkout has no provider, the tab is not in the row, and the editor is
 * exactly what it was. So there is no "sign in to continue" state here and no
 * empty promise: either there is a store, or there is no panel.
 *
 * It knows nothing about where the designs are. `label` is whatever the
 * provider calls itself, and every other word on screen comes from the list it
 * returns. That is what keeps this file free of anybody's authentication.
 */

import { useCallback, useEffect, useState } from "react";
import { useEditor } from "../state/store.js";
import { useStorage } from "../state/hooks.js";
import { captureView, applyView } from "../state/session.js";
import { serializeDoc, parseDoc } from "../state/serialize.js";
import type { StoredDesign } from "../state/storage.js";
import Section from "./Section.js";

/** The design this editor is currently working on, when it came from a store. */
let openId: string | null = null;

/** Called when a design is opened from elsewhere, so Save knows where to write. */
export function setOpenDesign(id: string | null): void {
  openId = id;
}

export default function Designs(): React.ReactElement {
  const provider = useStorage();
  const doc = useEditor((s) => s.doc);
  const setDoc = useEditor((s) => s.setDoc);
  const setStatus = useEditor((s) => s.setStatus);

  const [rows, setRows] = useState<StoredDesign[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!provider) return;
    try {
      setRows(await provider.list());
      setProblem(null);
    } catch (e) {
      setProblem((e as Error).message);
    }
  }, [provider]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  if (!provider) {
    // Unreachable through the tab row, which omits this panel entirely when
    // there is no provider. Here so the component is honest on its own.
    return <div className="panel__body" />;
  }

  const run = async (what: string, fn: () => Promise<void>): Promise<void> => {
    setBusy(true);
    try {
      await fn();
      setProblem(null);
    } catch (e) {
      setProblem(`${what}: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const save = (): Promise<void> =>
    run("Could not save", async () => {
      const { id } = await provider.save(openId, doc.meta.name, serializeDoc(doc), captureView());
      openId = id;
      setStatus(`Saved ${doc.meta.name}`);
      await refresh();
    });

  const open = (row: StoredDesign): Promise<void> =>
    run("Could not open", async () => {
      const got = await provider.load(row.id);
      const next = parseDoc(got.body);
      setDoc(next, `Opened ${got.name}`);
      // After setDoc, which resets the level — putting you back is the point.
      applyView(got.view, next);
      openId = row.id;
    });

  const share = (row: StoredDesign): Promise<void> =>
    run("Could not share", async () => {
      const { url } = await provider.share!(row.id);
      try {
        await navigator.clipboard.writeText(url);
        setStatus("Share link copied");
      } catch {
        // A clipboard a browser would not give us is not a failed share.
        setStatus(url);
      }
      await refresh();
    });

  const account = provider.account?.();

  return (
    <div className="panel__body designs">
      <Section id="designs.list" title={provider.label} note={rows?.length || undefined}>
        {rows === null ? (
          <p className="empty">Loading…</p>
        ) : rows.length === 0 ? (
          <p className="empty">Nothing saved yet. Press Save below to keep this design.</p>
        ) : (
          <ul className="designs__list">
            {rows.map((row) => (
              <li key={row.id} className={row.id === openId ? "designs__row designs__row--open" : "designs__row"}>
                <button type="button" className="designs__open" onClick={() => void open(row)} disabled={busy}>
                  <span className="designs__name">{row.name}</span>
                  <span className="designs__when">{when(row.updatedAt)}</span>
                </button>
                {provider.share && (
                  <button
                    type="button"
                    className="designs__action"
                    onClick={() => void share(row)}
                    disabled={busy}
                    title={row.shareUrl ? "Copy the share link" : "Make a link anyone can open"}
                  >
                    {row.shareUrl ? "link" : "share"}
                  </button>
                )}
                <button
                  type="button"
                  className="designs__drop"
                  onClick={() => void run("Could not delete", async () => {
                    await provider.remove(row.id);
                    if (openId === row.id) openId = null;
                    await refresh();
                  })}
                  disabled={busy}
                  title="Delete this design"
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="designs__actions">
          <button type="button" onClick={() => void save()} disabled={busy}>
            {openId ? "Save" : "Save as new"}
          </button>
          {openId && (
            <button type="button" onClick={() => { openId = null; setStatus("The next save will make a new design"); }}>
              Detach
            </button>
          )}
        </div>

        {problem && <p className="designs__problem">{problem}</p>}
      </Section>

      {account && (
        <Section id="designs.account" title="Account">
          <div className="designs__account">
            <span>{account.name}</span>
            {account.email && <span className="designs__email">{account.email}</span>}
            {account.signOutUrl && (
              <a href={account.signOutUrl} className="designs__signout">
                Sign out
              </a>
            )}
          </div>
        </Section>
      )}
    </div>
  );
}

/** "3 minutes ago" is more use than a timestamp in a list you scan. */
function when(iso: string): string {
  const then = Date.parse(iso);
  if (!Number.isFinite(then)) return "";
  const seconds = Math.max(0, (Date.now() - then) / 1000);
  if (seconds < 90) return "just now";
  const minutes = seconds / 60;
  if (minutes < 60) return `${Math.round(minutes)} min ago`;
  const hours = minutes / 60;
  if (hours < 24) return `${Math.round(hours)} h ago`;
  return new Date(then).toLocaleDateString();
}
