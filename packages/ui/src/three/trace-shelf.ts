/**
 * Where traces are kept between visits.
 *
 * A trace is megabytes — too much for `localStorage`, whose quota a few of
 * them would exhaust — so it goes in the browser's own database, IndexedDB,
 * keyed by the fingerprint of the model it was made from. That is the same key
 * the editor matches a trace to a design by, so a design opened again finds
 * its trace with nothing to load: in a tab, in the hosted editor, and in the
 * desktop window, which is a browser too.
 *
 * It keeps the most recent few and lets the rest go. A trace is cheap to make
 * again, and a shelf that only grows is a shelf nobody knows is there until
 * the disk fills.
 *
 * Every call is allowed to fail — a private window, a quota, a browser with
 * storage turned off — and failing means the trace is kept for the session
 * only, which is what happened before there was a shelf at all.
 */

export interface TraceShelf {
  /** The trace of the model with this fingerprint, as the text it was loaded from. */
  get(hash: string): Promise<string | null>;
  put(hash: string, text: string): Promise<void>;
  /** How many are kept: none means there is nothing to look a design up in. */
  count(): Promise<number>;
}

/** How many traces are kept. A few designs' worth; each is up to a few megabytes. */
export const KEEP = 8;

const DB = "tensorcad-traces";
const STORE = "traces";

interface Shelved {
  hash: string;
  text: string;
  /** When it was last put or taken, so the oldest is the one let go. */
  used: number;
}

function request<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((ok, no) => {
    req.onsuccess = () => ok(req.result);
    req.onerror = () => no(req.error);
  });
}

function open(): Promise<IDBDatabase> {
  return new Promise((ok, no) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "hash" });
    req.onsuccess = () => ok(req.result);
    req.onerror = () => no(req.error);
  });
}

/** The shelf in this browser's IndexedDB. */
function indexedShelf(): TraceShelf {
  let db: Promise<IDBDatabase> | null = null;
  const store = async (mode: IDBTransactionMode): Promise<IDBObjectStore> => {
    db ??= open();
    return (await db).transaction(STORE, mode).objectStore(STORE);
  };

  return {
    async get(hash) {
      const found = (await request((await store("readonly")).get(hash))) as Shelved | undefined;
      if (!found) return null;
      // Taken is used: it moves to the back of the queue to be let go.
      await request((await store("readwrite")).put({ ...found, used: Date.now() }));
      return found.text;
    },
    async put(hash, text) {
      await request((await store("readwrite")).put({ hash, text, used: Date.now() } satisfies Shelved));
      const all = (await request((await store("readonly")).getAll())) as Shelved[];
      const extra = all.sort((a, b) => b.used - a.used).slice(KEEP);
      if (extra.length === 0) return;
      const tx = await store("readwrite");
      await Promise.all(extra.map((s) => request(tx.delete(s.hash))));
    },
    async count() {
      return request((await store("readonly")).count());
    },
  };
}

/**
 * A shelf that keeps nothing past the session, for where there is no
 * IndexedDB: the test runner, and a browser with storage turned off.
 */
export function memoryShelf(): TraceShelf {
  const kept = new Map<string, string>();
  return {
    async get(hash) {
      const text = kept.get(hash);
      if (text === undefined) return null;
      kept.delete(hash);
      kept.set(hash, text);
      return text;
    },
    async put(hash, text) {
      kept.delete(hash);
      kept.set(hash, text);
      while (kept.size > KEEP) kept.delete(kept.keys().next().value!);
    },
    async count() {
      return kept.size;
    },
  };
}

let shelf: TraceShelf = typeof indexedDB === "undefined" ? memoryShelf() : indexedShelf();

/** The shelf traces are kept on. */
export function traceShelf(): TraceShelf {
  return shelf;
}

/** Put a different shelf in its place: a test's, which can be emptied between cases. */
export function setTraceShelf(next: TraceShelf): void {
  shelf = next;
}
