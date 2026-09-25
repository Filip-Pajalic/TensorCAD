# License

TensorCAD is released under the MIT License.

```
MIT License

Copyright (c) 2026 TensorCAD contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Third-party notices

### llm-viz — ported code

`packages/ui/src/three/model3d.ts` and `packages/ui/src/three/View3D.tsx` are a
port of the layout and arrow rendering in Brendan Bycroft's LLM visualisation.
The conventions, the geometry and the drawing decisions are his; they are kept
deliberately close to the original so the two read the same.

> MIT License
>
> Copyright (c) 2023-2026 Brendan Bycroft
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Source: <https://github.com/bbycroft/llm-viz>

### Architectures described, not copied

The presets in `packages/core-go/presets/data/` describe published architectures.
They are written from each model's configuration and paper — no code is copied
from any of them, and an architecture's dimensions are facts rather than
expression. They are listed here because the work deserves attribution:

| Preset | Source | Upstream license |
|---|---|---|
| `gpt2-*` | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | MIT |
| `nanogpt` | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | MIT |
| `gpt-oss-20b` | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | Apache-2.0 |
| `bloom-7b1` | [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1) | BigScience RAIL License v1.0 |
| `llama-*` | Meta Llama model cards | Llama Community License |
| `mistral-7b` | [mistralai/Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) | Apache-2.0 |
| `qwen*` | [QwenLM](https://huggingface.co/Qwen) | Apache-2.0 |
| `gemma-2-9b` | [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) | Gemma Terms of Use |
| `mixtral-8x7b` | [mistralai/Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | Apache-2.0 |
| `deepseek-v3` | [DeepSeek-V3 paper](https://arxiv.org/abs/2412.19437) | MIT (code) |
| `nemotron-h-8b` | [nvidia/Nemotron-H-8B-Base](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K) | NVIDIA Open Model License |
| `ijepa-vit-h14` | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | CC BY-NC 4.0 |
| `alexnet` | [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause |

Note on I-JEPA: its repository is non-commercially licensed. TensorCAD contains
no code from it. The preset is a description of the architecture, built from the
published `vision_transformer.py` configuration and the `in1k_vith14_ep300.yaml`
training config. If you intend to use the I-JEPA *weights* or *code*, that
license applies to you and not to this repository.

### Interaction design

The schematic editor's behaviour — pick apertures, net highlighting, junction
dots, dangling-pin marks, selection modifiers, wire reconnection — follows
[KiCad's eeschema](https://docs.kicad.org/). No KiCad code is used; the
documentation was read and the conventions were reimplemented.

### Runtime dependencies

The engine has none: `packages/core-go` is the Go standard library and nothing
else. The editor, desktop shell and Python runtime depend on
React, React Flow, ELK, Three.js, Base UI, Tailwind, Zustand, lucide-react,
Wails and PyTorch, each under its own license. See the respective `package.json`,
`go.mod` and `pyproject.toml` for exact versions.
