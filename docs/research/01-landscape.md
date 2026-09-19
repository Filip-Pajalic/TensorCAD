# Research 01 — Landscape: visual NN designers, node editors, config/codegen targets

Researched 2026-09-18.

## 1. Existing tools

### Legacy visual NN builders (all dead or dormant)

| Tool | What it does | Status | Shape check / params / codegen | License |
|---|---|---|---|---|
| [Sony Neural Network Console](https://github.com/sony/neural-network-console) | Windows GUI, drag-drop layers, NAS, training | EOL April 3 2025; repo archived | "Network statistics: computational complexity and memory usage"; export NNP/ONNX/TF | Apache-2.0 |
| [ENNUI](https://github.com/martinjm97/ENNUI) ([live](https://math.mit.edu/ennui/)) | Browser drag-drop (dense/conv/pool/norm/dropout/concat/sum), trains in-browser, "Export to Python" | Last commit 2023-01. Dead | Keras-style codegen; no attention layers; no param/FLOP stats | MIT |
| [Deep Cognition / DLS](https://deepcognition.ai/) | Drag-drop Keras builder (2017–2020) | Company pivoted; product gone | Keras codegen | Proprietary |
| [NVIDIA DIGITS](https://github.com/NVIDIA/DIGITS) | Web UI for Caffe/TF CNN training (form-based) | Archived 2025-01 | No | BSD-3 |
| [PerceptiLabs](https://github.com/PerceptiLabs/PerceptiLabs) | TF visual modeler generating TF code | Shut down; last push 2021-08 | Codegen yes | Proprietary |
| [Cloud-CV Fabrik](https://github.com/Cloud-CV/Fabrik) | Browser drag-drop, import/export Caffe/Keras/TF | Last push 2020-12 | Codegen yes | GPL-3.0 |
| Microsoft Lobe | No-code image classifier | Deprecated ~2023 | n/a | Proprietary |
| [TorchStudio](https://github.com/TorchStudio/torchstudio) | PyTorch IDE; graph view shows tensor size changes for a coded model | Last release 2024-04. Dormant | Shape display; no codegen | MIT |
| [KAIBER NN Editor](https://kaiber.biz/nne-py_en/) | Browser editor, real-time output shapes, exports PyTorch | Beta 5, Sept 2022. Dormant | Shapes + codegen; no stats; no transformer blocks | Closed beta |

No official Keras/TensorFlow visual builder exists; ML.NET "Model Builder" is AutoML, not architecture design.

### Visualization-only (maintained, not editors)
- [Netron](https://github.com/lutzroeder/netron): MIT, very active. Renders ONNX/PyTorch graphs with shapes. No editing, no codegen.
- [NN-SVG](https://github.com/alexlenail/NN-SVG), [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet): publication schematics only.
- [Google Visual Blocks](https://github.com/google/visualblocks): Apache-2.0, active; composes ML *pipelines* (models are opaque nodes); no layer-level design.

### 2024–2026 projects closest to the idea
- [BuildANeuralNet](https://github.com/JavaNoTea/BuildANeuralNet): React 18 + React Flow + FastAPI; drag-drop layers, real-time connection validation, PyTorch `nn.Module` codegen. GPL-3.0, 1 star, MLP/CNN only; transformer blocks on TODO.
- [easy-torchcraft](https://github.com/Tylersuard/easy-torchcraft): Lovable-generated React app, PyTorch export. No license, 28 stars.
- [v0-pytorch-neural-network-designer](https://github.com/pmquang87/v0-pytorch-neural-network-designer): v0-generated toy.
- [NeuralFlows.ai](https://neuralflows.ai/): claims transformer design + in-browser training + PyTorch export. Unverifiable (site unreachable).

### LLM-specific visualizers/calculators (none are editors)
- [rasbt/LLM-architecture-gallery](https://github.com/rasbt/LLM-architecture-gallery): Apache-2.0, 1.5k stars, active. Figures + structured data for every major open architecture (GQA/MLA/MoE). **Best reference catalog for our block library and presets.**
- [Devisri-B/LLM-Architectures](https://github.com/Devisri-B/LLM-Architectures): React Flow diagrams of GPT/Llama, viz only.
- [Liears/llm-architecture-atlas](https://github.com/Liears/llm-architecture-atlas): static catalog from `models.yml`.
- [llm-arch-reviewer](https://github.com/YAMY1234/llm-arch-reviewer): block diagrams with vLLM/SGLang profile overlays.
- KV/memory calculators: [inference-estimator](https://github.com/totalwindupflightsystems/inference-estimator), [LMCache kvcache-view](https://github.com/LMCache/kvcache-view).
- Educational: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/), [LLM Visualizer](https://jayvisaria.github.io/LLM-Visualizer/).
- Searches for "visual LLM builder" are dominated by agent/pipeline builders (Flowise, ADK Visual Agent Builder). Different problem.

### Conclusion
**No polished "CAD for LLM architectures" exists as of Sept 2026.** Every drag-drop layer builder is dead or an MLP/CNN toy. None offer MoE/GQA/MLA/SSM blocks, param/FLOP/KV-cache accounting, and PyTorch codegen together. The niche is open.

## 2. Node editor libraries

### React Flow / [@xyflow/react](https://github.com/xyflow/xyflow)
- Current: **12.11.6** (2026-09-01). MIT. React 17+ (React 19 works). 38k stars, weekly releases.
- Custom nodes are React components; handles with ids; `isValidConnection` hook for typed ports; MiniMap, Controls, Background, NodeToolbar, NodeResizer.
- Sub-flows: [`parentId`](https://reactflow.dev/learn/layouting/sub-flows), `extent: 'parent'`, `expandParent`, `group` node type. Parents must precede children in the `nodes` array. Multi-level nesting works in practice.
- [Layouting guide](https://reactflow.dev/learn/layouting/layouting): dagre (simple/fast), elkjs (most configurable, sub-flows + edge routing, ~1 MB, run in a worker), d3-hierarchy, d3-force.
- Licensing: core is [MIT "forever"](https://reactflow.dev/pro); Pro buys examples/support, not required.
- Siblings: Svelte Flow 1.6.x (Svelte 5), Vue Flow 1.48.x.

### Alternatives
- [Rete.js](https://retejs.org/docs/) 2.x: framework-agnostic, typed sockets, `DataflowEngine`, scopes plugin for nesting. More assembly, smaller ecosystem.
- [LiteGraph.js](https://github.com/jagenjo/litegraph.js): original dormant. Comfy-Org TypeScript fork published as [@comfyorg/litegraph](https://www.npmjs.com/package/@comfyorg/litegraph) (npm MIT; monorepo GPL-3.0, check before vendoring). Canvas2D, imperative, fast with big graphs.
- [Drawflow](https://github.com/jerosoler/Drawflow): dormant, no typed ports.
- [Baklava.js](https://github.com/newcat/baklavajs): Vue 3, typed interfaces, active. Vue only.
- Blueprint-style: Flume (low activity), node-blueprint (Vue). Nothing mainstream.
- Layout: [@dagrejs/dagre 3.x](https://github.com/dagrejs/dagre) (MIT), [elkjs 0.12](https://github.com/kieler/elkjs) (EPL-2.0 OR GPL-3.0), d3-dag (MIT, Sugiyama).

### ComfyUI as reference
ComfyUI shipped **subgraphs** in Aug 2025 ([v0.3.51](https://blog.comfy.org/p/subgraph-official-release)): collapse selection to a super-node with typed I/O, nestable, breadcrumb navigation. Its [workflow JSON](https://docs.comfy.org/specs/workflow_json) is its own format (`nodes[]`, `links[]`, `groups`, `extra`), not the editor library's. **Lesson: persist our own IR; treat editor state as a view.**

### Recommendation
**React Flow (@xyflow/react 12.x) + elkjs (dagre for simple cases).** Typed handles + `isValidConnection` map onto shape checks; sub-flows give the "Transformer Block ×N" container; nodes are React components so shape badges and live counts are trivial; MIT; largest community. Caveat: DOM-based, so collapse repeated blocks into a single ×N node (needed for the math anyway).

## 3. Declarative configs and codegen: targets and import formats

| Project | Form | Modern blocks | Use |
|---|---|---|---|
| [HF transformers](https://github.com/huggingface/transformers) (v5.x) | `config.json` per model; [modular transformers](https://huggingface.co/docs/transformers/modular_transformers) generates modeling files; unified [AttentionInterface](https://huggingface.co/docs/transformers/main/attention_interface) | GQA, MLA, MoE, Mamba hybrids | **Primary IMPORT** (config.json is the de facto standard). Secondary TARGET for interop. |
| [TorchTitan](https://github.com/pytorch/torchtitan) (BSD-3) | `ModelArgs` dataclasses + flavor dicts, TOML selects flavor; DeepSeek-V3 args include `num_experts`, `q_lora_rank`, `kv_lora_rank` | Llama 3/4, DeepSeek-V3, Qwen3, GPT-OSS | TARGET for scale-out training; args are a checklist of what a design must specify. |
| [litgpt](https://github.com/Lightning-AI/litgpt) (Apache-2.0) | One [`Config` dataclass](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/config.py) (~100 fields: `n_query_groups`, `mlp_class_name`, `norm_class_name`, rope params), `name_to_config` registry | GQA, MoE, RoPE variants; no SSM | Cheapest TARGET: emit a `Config`, get pretrain/finetune/serve free. Also IMPORT. |
| [MaxText](https://github.com/AI-Hypercomputer/maxtext) (JAX) | YAML configs, `decoder_block` selects layer type | MLA, shared/routed experts, YaRN | Reference/IMPORT only. |
| [nanoGPT](https://github.com/karpathy/nanoGPT) / [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT) | `GPTConfig`; modded-nanogpt is a single hand-tuned `train_gpt.py` | Dense GPT + speedrun tricks | Minimal TARGET template; reference for modern tricks. |
| [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) | [`ModuleSpec` + `TransformerLayerSubmodules`](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/models.gpt.html): declarative submodule tree | MoE, GQA, MLA, hybrids | Closest existing "architecture-as-spec" in production. Heavy; custom license. |
| [nanotron](https://github.com/huggingface/nanotron), [Lingua](https://github.com/facebookresearch/lingua) | YAML / dataclass configs | Dense + some MoE | Secondary targets. |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT), [mamba](https://github.com/state-spaces/mamba) (Apache-2.0) | Importable `nn.Module` blocks (GLA, DeltaNet, GatedDeltaNet, Mamba/Mamba2) | SSM/linear attention | Generated PyTorch should `import` these for SSM nodes rather than re-implement kernels. |
| [torch.fx](https://docs.pytorch.org/docs/stable/fx.html) | Graph IR with Python codegen | op-level | Possible backend but loses block structure; templating `nn.Module` is simpler. |
| [torch.export](https://docs.pytorch.org/docs/stable/export.html) | `ExportedProgram` (ATen IR, shape-aware) | op-level | **Verification target**: export generated model, cross-check shapes/params vs editor estimates. |
| [ONNX](https://onnx.ai/onnx/) | Op graph | poor for MoE/custom kernels | Optional export for Netron viewing. |

**Suggested pipeline**: own JSON IR (with a "block ×N" container) → codegen to (a) plain PyTorch `nn.Module` importing `fla`/`mamba_ssm` for exotic blocks, (b) HF `config.json` + modeling file, (c) litgpt `Config` where representable; **import** HF `config.json` and litgpt `Config`; **validate** with `torch.export` shape propagation.

## 4. Small-scale validation references (for the "test bench" idea)

- [modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt): record #91 (Aug 2026) reaches 3.28 val loss on FineWeb in 1.126 min on 8×H100. Tricks in the record: Muon, value embeddings, U-net skips, FlashAttention 3 + sliding window, FP8, logit softcap, multi-token prediction, bigram hash embeddings, MUDD skips, learnable XSA gates. No single-consumer-GPU track exists; we would define our own tiny track.
- [METR note on NanoGPT progress (2026-04)](https://metr.org/notes/2026-04-21-ai-rd-nanogpt-progress/): 31× speedup May 2024 → Mar 2026; early gains were architecture, later gains optimizer and kernels. Imported techniques contributed 6.7×, adapted 3.0×, invented 1.6×. Small-scale-only tricks often do not transfer; "scale-complementary" ones (e.g. Muon) did.
- [Parameter Golf (arXiv 2607.01517)](https://arxiv.org/abs/2607.01517): 84 techniques catalogued across 1,430 submissions; individual techniques rarely exceed 1% and their gains shrink when re-measured among competitive submissions. Lesson: an ablation harness must control for the surrounding stack, and report confidence intervals.
- μP / μTransfer: [microsoft/mup](https://github.com/microsoft/mup), [Cerebras practitioner's guide](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization), [How To Scale](https://howtoscalenn.github.io/). Relevant for a "scale ladder" feature: tune at small width, transfer to large width.

### Unverified
NeuralFlows.ai claims; Sony NNC PyTorch engine support; React Flow practical nesting depth; Megatron-LM license terms.
