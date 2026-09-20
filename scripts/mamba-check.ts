import { analyze, countParams, DOC_VERSION, inferShapes, loadEngine } from "@tensor-cad/engine/node";
import type { Doc } from "@tensor-cad/engine/node";

await loadEngine();

// Nemotron-H-8B's Mamba-2 block in isolation.
const doc: Doc = {
  version: DOC_VERSION,
  meta: { name: "mamba2-probe" },
  symbols: {
    B: { kind: "runtime", default: 1 }, T: { kind: "runtime", default: 8192 },
    D: { kind: "design", value: 4096 },
  },
  graph: {
    nodes: [
      { id: "tokens", type: "input", params: { shape: "B T", dtype: "int64" } },
      { id: "embed", type: "embedding", params: { vocab: 131072, dim: "D" } },
      { id: "blk", type: "mamba2_block", params: { d_model: "D", expand: 2, head_dim: 64, state: 128, groups: 8, conv_kernel: 4 } },
      { id: "out", type: "output" },
    ],
    edges: [["tokens:x", "embed:ids"], ["embed:y", "blk:x"], ["blk:y", "out:x"]],
  },
};
const inf = inferShapes(doc, "expanded");
const errs = inf.issues.filter((i) => i.severity === "error");
console.log("shape errors:", errs.length);
for (const e of errs) console.log("  ", e.path, e.port ?? "", e.message);
const p = countParams(doc);
const blk = Object.entries(p.byPath).filter(([k]) => k.startsWith("blk/")).reduce((a, [, v]) => a + v, 0);
console.log("mamba2 block params:", blk.toLocaleString("en-US"), "(expected 109,635,968)");
const a = analyze(doc, { T: 8192 });
console.log("state per sequence:", a.kv.bytesPerSequenceFixed.toLocaleString("en-US"), "bytes");
console.log("cache per token:", a.kv.bytesPerToken, "bytes (a state-space layer has none)");
