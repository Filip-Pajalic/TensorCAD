/**
 * Where generated files go.
 *
 * `tensorcad_generate_code` emits a PyTorch module and the design that
 * produced it. Given no destination it hands the contents back in the reply,
 * which is fine for one small file and useless for anything an agent wants to
 * run — a model it cannot put anywhere is a model it can only read.
 *
 * Locally the destination is a directory, and writing to it is two lines of
 * `node:fs`. Hosted, there is no filesystem at all: a Worker has no `mkdir`,
 * and a file written into an isolate is gone when that isolate is. So the
 * destination is an interface, and the only thing the tool knows about it is
 * that it takes a path and some text and says where they went.
 *
 * ## The reply says where, not what
 *
 * `location` is for a person reading the transcript — an absolute path, or an
 * object key. `url` is what makes a hosted sink worth having: something the
 * agent can actually fetch. A sink that has no meaningful URL omits it rather
 * than inventing one, because a link that does not resolve is worse than no
 * link.
 */

/** One emitted file, once it has been put somewhere. */
export interface WrittenArtifact {
  /** Where it went, in whatever terms the sink uses. */
  location: string;
  /** Somewhere it can be fetched from, when the sink has one. */
  url?: string;
}

export interface ArtifactSink {
  /**
   * A short name for the place, for the sentence the tool writes.
   *
   * "disk" and "object storage" read very differently in a transcript, and the
   * difference matters to whoever has to go and find the file.
   */
  readonly label: string;

  /**
   * Put one file somewhere.
   *
   * `prefix` is whatever the caller passed as `out_dir`, uninterpreted: a
   * directory to a filesystem, a key prefix to a bucket. `path` is the file's
   * own relative path, which may contain directories.
   */
  write(prefix: string, path: string, contents: string): Promise<WrittenArtifact>;
}
