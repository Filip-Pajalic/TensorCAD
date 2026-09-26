/**
 * A design carried in a link.
 *
 * Sharing should not need an account. A design is a few kilobytes of JSON, and
 * compressed and written into a link's fragment it fits in any address bar, so
 * `#design=…` is a design anybody can open with nothing uploaded anywhere: the
 * fragment never leaves the browser, and the editor reads it back on load.
 *
 * The first character says how the rest is written: `z` for deflated, `p` for
 * plain. Every current browser can deflate, so a link made in one is `z`; the
 * plain form is what an environment without compression makes, and every
 * reader understands both.
 */

/** Compression, supplied rather than assumed: the tests bring their own. */
export interface Codec {
  compress(bytes: Uint8Array): Promise<Uint8Array>;
  decompress(bytes: Uint8Array): Promise<Uint8Array>;
}

async function through(
  bytes: Uint8Array,
  stream: { readable: ReadableStream<Uint8Array>; writable: WritableStream<BufferSource> },
): Promise<Uint8Array> {
  const piped = new Blob([bytes as BlobPart]).stream().pipeThrough(stream);
  return new Uint8Array(await new Response(piped).arrayBuffer());
}

/** The browser's own deflate, where there is one. */
export const platformCodec: Codec | null =
  typeof CompressionStream === "undefined" || typeof DecompressionStream === "undefined"
    ? null
    : {
        compress: (bytes) => through(bytes, new CompressionStream("deflate-raw")),
        decompress: (bytes) => through(bytes, new DecompressionStream("deflate-raw")),
      };

const DEFLATED = "z";
const PLAIN = "p";

function toBase64Url(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(text: string): Uint8Array {
  const b64 = text.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(b64 + "=".repeat((4 - (b64.length % 4)) % 4));
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

/** A design's text as a fragment's value. */
export async function encodeDesign(text: string, codec: Codec | null = platformCodec): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  return codec ? DEFLATED + toBase64Url(await codec.compress(bytes)) : PLAIN + toBase64Url(bytes);
}

/** A fragment's value back to the design's text. */
export async function decodeDesign(value: string, codec: Codec | null = platformCodec): Promise<string> {
  const body = fromBase64Url(value.slice(1));
  switch (value[0]) {
    case PLAIN:
      return new TextDecoder().decode(body);
    case DEFLATED:
      if (!codec) throw new Error("this browser cannot read a compressed link");
      return new TextDecoder().decode(await codec.decompress(body));
  }
  throw new Error("it was not made by this editor");
}

/** The fragment a link carries a design in: `#design=` and the value. */
export const DESIGN_FRAGMENT = /^#design=([A-Za-z0-9_-]+)$/;

/**
 * Where a link should point. This editor, when it is on the web, so a link
 * made on a deployment opens on that deployment; the public one otherwise,
 * which is where a link made in the desktop app has to go.
 */
export function linkBase(): string {
  if (typeof location !== "undefined" && /^https?:$/.test(location.protocol)) return `${location.origin}/`;
  return "https://tensorcad.dev/";
}
