/**
 * Reading the IR's own spellings.
 *
 * An endpoint is `"nodeId:portName"` and a path is `"a/b/c"`. These are the
 * two conventions the whole document rests on, and they are string handling
 * rather than analysis — the editor splits a hundred of them per repaint, and
 * crossing into the engine for that would be absurd. The Go engine has the same
 * two functions for the same reason.
 */

/** One end of an edge. */
export interface Endpoint {
  node: string;
  port: string;
}

/**
 * Splits `"node:port"`.
 *
 * At the last colon, because a node id inside a container carries slashes and
 * may itself have come from a path.
 */
export function splitEndpoint(endpoint: string): Endpoint {
  const i = endpoint.lastIndexOf(":");
  if (i < 0) throw new Error(`Malformed endpoint "${endpoint}", expected "node:port"`);
  return { node: endpoint.slice(0, i), port: endpoint.slice(i + 1) };
}

/** Builds the dotted path identifying a node inside nested graphs. */
export function joinPath(prefix: string, id: string): string {
  return prefix ? `${prefix}/${id}` : id;
}
