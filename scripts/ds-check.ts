import { formatBytes, formatCount, getPreset, loadEngine, validate } from "@tensor-cad/engine/node";

await loadEngine();

const r = validate(getPreset("deepseek-v3"), { T: 4096 });
const a = r.analysis;
console.log("total       ", a.params.total.toLocaleString("en-US"), "=", formatCount(a.params.total));
console.log("active      ", a.params.active.toLocaleString("en-US"), "=", formatCount(a.params.active));
console.log("kv/token    ", a.kv.bytesPerToken, "bytes =", formatBytes(a.kv.bytesPerToken));
console.log("attn/layer  ", formatCount((a.params.byPath["layers/block/attn/q_down"] ?? 0) / 58));
console.log("errors      ", r.counts.error, " warnings", r.counts.warning, " info", r.counts.info);
for (const f of r.findings.filter((x) => x.severity !== "info")) console.log("  ", f.severity, f.rule, f.path ?? "", f.message);
