import { generateTorch, getPreset, loadEngine } from "@tensor-cad/engine/node";
import { mkdirSync, writeFileSync } from "node:fs";

await loadEngine();

for (const name of ["mixtral-8x7b", "deepseek-v3"]) {
  const out = generateTorch(getPreset(name), { moeDispatch: "dense" });
  const dir = `out/${name}-dense`;
  mkdirSync(dir, { recursive: true });
  for (const f of out.files) writeFileSync(`${dir}/${f.path}`, f.contents);
  console.log(`wrote ${dir}/model.py  warnings: ${out.warnings.length}`);
}
