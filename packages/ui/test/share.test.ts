/**
 * Sharing a design from the toolbar.
 *
 * Two kinds of link. Signed in to a store that shares, Share saves the design
 * there and hands out the store's link. Anywhere else the design goes in the
 * link itself, compressed into its fragment, and opening it needs no server.
 *
 * Bun has no CompressionStream, which every browser does, so the compressed
 * form is driven here through node:zlib's raw deflate — the same format — and
 * the plain form, which is what an environment without compression writes, is
 * driven through no codec at all.
 */

import { beforeEach, describe, expect, test } from "bun:test";
import { deflateRawSync, inflateRawSync } from "node:zlib";
import { loadEngine } from "../src/engine.js";

await loadEngine();
const { getPreset, PRESET_NAMES } = await import("../src/engine.js");
const { useEditor } = await import("../src/state/store.js");
const { serializeDoc } = await import("../src/state/serialize.js");
const { registerStorage, openDesignId, setOpenDesign } = await import("../src/state/storage.js");
const { openFromLocation } = await import("../src/state/session.js");
const { encodeDesign, decodeDesign } = await import("../src/state/link.js");
const { shareDesign, sharesThroughStore } = await import("../src/state/share.js");
import type { Codec } from "../src/state/link.js";
import type { StorageProvider } from "../src/state/storage.js";

const state = () => useEditor.getState();

const zlib: Codec = {
  async compress(bytes) {
    return new Uint8Array(deflateRawSync(bytes));
  },
  async decompress(bytes) {
    return new Uint8Array(inflateRawSync(bytes));
  },
};

describe("a design carried in a link", () => {
  test("comes back exactly, compressed or plain, for every preset", async () => {
    let longest = 0;
    for (const name of PRESET_NAMES) {
      const text = serializeDoc(getPreset(name));
      const deflated = await encodeDesign(text, zlib);
      expect(deflated[0]).toBe("z");
      expect(await decodeDesign(deflated, zlib)).toBe(text);
      const plain = await encodeDesign(text, null);
      expect(plain[0]).toBe("p");
      expect(await decodeDesign(plain, null)).toBe(text);
      // Only characters a fragment carries untouched.
      expect(deflated).toMatch(/^[A-Za-z0-9_-]+$/);
      longest = Math.max(longest, deflated.length);
    }
    // Nemotron-H's fifty-two layers written out is the longest, and it still
    // fits any address bar several times over.
    expect(longest).toBeLessThan(16_000);
  });

  test("a link this editor did not make is refused with a reason", async () => {
    await expect(decodeDesign("xAAAA", zlib)).rejects.toThrow(/not made by this editor/);
    await expect(decodeDesign("zAAAA", null)).rejects.toThrow(/cannot read a compressed link/);
  });
});

describe("opening one", () => {
  let replaced: string | null = null;
  function at(hash: string): void {
    (globalThis as { location?: unknown }).location = { pathname: "/", search: "", hash } as Location;
    (globalThis as { history?: unknown }).history = {
      replaceState: (_: unknown, __: string, url: string) => {
        replaced = url;
      },
    } as History;
  }

  beforeEach(() => {
    registerStorage(null);
    replaced = null;
    state().setDoc(getPreset("llama-3-8b"), "test");
  });

  test("opens the design with nothing offering to store it, and clears the address bar", async () => {
    at(`#design=${await encodeDesign(serializeDoc(getPreset("t5-small")), null)}`);
    expect(await openFromLocation()).toBe(true);
    expect(state().doc.meta.name).toBe("t5-small");
    expect(state().status).toContain("from a link");
    expect(replaced).toBe("/");
  });

  test("a broken one says so rather than showing the default design", async () => {
    at("#design=pnot-json");
    expect(await openFromLocation()).toBe(false);
    expect(state().doc.meta.name).toBe("llama-3-8b");
    expect(state().status).toContain("did not open");
  });
});

describe("Share", () => {
  function store(signedIn: boolean): StorageProvider & { saved: string[] } {
    const saved: string[] = [];
    return {
      label: "A Map",
      saved,
      async list() {
        return [];
      },
      async load() {
        throw new Error("unused");
      },
      async save(id, name) {
        saved.push(name);
        return { id: id ?? "d1" };
      },
      async remove() {},
      async share(id) {
        return { url: `https://example.test/d/${id}` };
      },
      account: () => (signedIn ? { name: "someone" } : null),
    };
  }

  beforeEach(() => {
    (globalThis as { location?: unknown }).location = undefined;
    setOpenDesign(null);
    state().setDoc(getPreset("gpt2-small"), "test");
  });

  test("with nothing to store it in, the design goes in the link, to the public editor", async () => {
    registerStorage(null);
    expect(sharesThroughStore()).toBe(false);
    const got = await shareDesign();
    expect(got.kind).toBe("inline");
    expect(got.url.startsWith("https://tensorcad.dev/#design=")).toBe(true);
  });

  test("signed out of a store, it still works, the same way", async () => {
    const provider = store(false);
    registerStorage(provider);
    const got = await shareDesign();
    expect(got.kind).toBe("inline");
    expect(provider.saved).toEqual([]);
  });

  test("signed in, it saves the design and hands out the store's link", async () => {
    const provider = store(true);
    registerStorage(provider);
    expect(sharesThroughStore()).toBe(true);
    const got = await shareDesign();
    expect(got).toEqual({ url: "https://example.test/d/d1", kind: "stored" });
    expect(provider.saved).toEqual(["gpt2-small"]);
    // And the design is now the stored one, so the next save updates it.
    expect(openDesignId()).toBe("d1");
  });
});
