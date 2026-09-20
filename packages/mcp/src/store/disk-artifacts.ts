/**
 * Generated files, written to a directory.
 *
 * What the tool did inline until there was a second kind of destination. It is
 * here rather than in `tools.ts` because it is the only part of that tool that
 * cannot run in a Worker — `mkdir` and `writeFile` do not exist there — and
 * keeping it in one small module is what lets the hosted build substitute
 * something else rather than fork the tool.
 */

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import type { ArtifactSink, WrittenArtifact } from "../artifacts.js";

export class DiskSink implements ArtifactSink {
  readonly label = "disk";

  constructor(private readonly root: string) {}

  async write(prefix: string, path: string, contents: string): Promise<WrittenArtifact> {
    // A relative `out_dir` is relative to the server's root rather than to
    // whatever the process happens to have chdir'd to, so the same request
    // lands in the same place whoever started the server.
    const base = isAbsolute(prefix) ? prefix : resolve(this.root, prefix);
    const target = join(base, path);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, contents, "utf8");
    return { location: target };
  }
}
