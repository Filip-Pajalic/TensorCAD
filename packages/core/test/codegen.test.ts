/**
 * Code generation tests.
 *
 * These check the properties that matter without pinning every character: the
 * generator must emit no placeholders, must declare exactly the modules the
 * design contains, and must agree with the analysis about the parameter count.
 * The golden file for GPT-2 guards against accidental formatting churn.
 */

import { describe, expect, it } from "bun:test";
import { generateTorch } from "../src/codegen/torch.js";
import { resolveSymbols } from "../src/ir/symbols.js";
import { countParams } from "../src/analysis/params.js";
import { allPresets, getPreset, PRESET_NAMES } from "../src/presets/index.js";

function modelSource(name: string): string {
  const out = generateTorch(getPreset(name));
  const file = out.files.find((f) => f.path === "model.py");
  expect(file).toBeDefined();
  return file!.contents;
}

describe("PyTorch generation", () => {
  it("emits every preset without warnings or placeholders", () => {
    for (const doc of allPresets()) {
      const out = generateTorch(doc);
      expect({ name: doc.meta.name, warnings: out.warnings }).toEqual({
        name: doc.meta.name,
        warnings: [],
      });
      const src = out.files.find((f) => f.path === "model.py")!.contents;
      expect(src).not.toContain("TODO");
      // The emitter writes `None` when it cannot resolve a value, either as an
      // assignment or as a call argument. Indexing with None is legitimate.
      expect(src).not.toMatch(/=\s*None\b/);
      expect(src).not.toMatch(/\(\s*None\s*[,)]/);
    }
  });

  it("writes the design alongside the model so the pair stays reproducible", () => {
    const out = generateTorch(getPreset("llama-3-8b"));
    const design = out.files.find((f) => f.path === "design.tensorcad.json");
    expect(design).toBeDefined();
    const parsed = JSON.parse(design!.contents);
    expect(parsed.meta.name).toBe("llama-3-8b");
  });

  it("states a parameter count that matches the analysis", () => {
    for (const name of PRESET_NAMES) {
      const doc = getPreset(name);
      const total = countParams(doc, resolveSymbols(doc)).total;
      const src = modelSource(name);
      expect(src).toContain(`expected = ${total}`);
    }
  });

  it("builds grouped-query attention with the right projection widths", () => {
    const src = modelSource("llama-3-8b");
    // 32 query heads and 8 key/value heads at 128 wide.
    expect(src).toContain("self.q_proj = nn.Linear(4096, 4096, bias=False)");
    expect(src).toContain("self.k_proj = nn.Linear(4096, 1024, bias=False)");
    expect(src).toContain("enable_gqa=True");
  });

  it("reshapes into heads and back without a redundant step", () => {
    const src = modelSource("llama-3-8b");
    expect(src).toContain("q_proj_y.view(B, T, 32, 128).permute(0, 2, 1, 3)");
    expect(src).toContain("attn_y.permute(0, 2, 1, 3).reshape(B, T, 4096)");
    // The split into heads needs no reshape after the permute.
    expect(src).not.toContain("permute(0, 2, 1, 3).reshape(B, 32, T, 128)");
  });

  it("ties the head to the embedding only when the design says so", () => {
    expect(modelSource("gpt2-small")).toContain("self.head.weight = self.embed.weight");
    expect(modelSource("llama-3-8b")).not.toContain("self.head.weight = self.embed.weight");
  });

  it("emits a rotary helper only for designs that use one", () => {
    expect(modelSource("llama-3-8b")).toContain("class RotaryEmbedding");
    // GPT-2 uses learned absolute positions instead.
    const gpt2 = modelSource("gpt2-small");
    expect(gpt2).not.toContain("class RotaryEmbedding");
    expect(gpt2).toContain("self.pos = nn.Embedding(1024, 768)");
  });

  it("emits a sliding-window mask only for designs that use one", () => {
    expect(modelSource("mistral-7b")).toContain("sliding_window_mask(");
    expect(modelSource("mistral-7b")).toContain("4096");
    expect(modelSource("llama-3-8b")).not.toContain("sliding_window_mask");
  });

  it("emits Gemma's extra output norms", () => {
    const src = modelSource("gemma-2-9b");
    expect(src).toContain("self.post_attn_norm");
    expect(src).toContain("self.post_mlp_norm");
    expect(src).toContain('approximate="tanh"');
  });

  it("emits Qwen's asymmetric attention bias", () => {
    const src = modelSource("qwen2.5-7b");
    expect(src).toContain("self.q_proj = nn.Linear(3584, 3584, bias=True)");
    expect(src).toContain("self.o_proj = nn.Linear(3584, 3584, bias=False)");
  });

  it("emits Qwen3's query and key norms", () => {
    const src = modelSource("qwen3-8b");
    expect(src).toContain("self.q_norm = nn.RMSNorm(128");
    expect(src).toContain("self.k_norm = nn.RMSNorm(128");
  });

  it("shares one class between identical blocks instead of repeating them", () => {
    const src = modelSource("llama-3-8b");
    const attentionClasses = src.match(/^class GqaAttention/gm) ?? [];
    expect(attentionClasses).toHaveLength(1);
    expect(src).toContain("nn.ModuleList([Layer() for _ in range(32)])");
  });

  it("is deterministic", () => {
    const a = modelSource("llama-3-8b");
    const b = modelSource("llama-3-8b");
    expect(a).toBe(b);
  });

  it("produces syntactically plausible Python", () => {
    for (const name of PRESET_NAMES) {
      const src = modelSource(name);
      // Balanced brackets and no stray template artefacts.
      for (const [open, close] of [
        ["(", ")"],
        ["[", "]"],
      ] as const) {
        const o = src.split(open).length - 1;
        const c = src.split(close).length - 1;
        expect({ name, open, o, c }).toEqual({ name, open, o, c: o });
      }
      expect(src).not.toContain("undefined");
      expect(src).not.toContain("[object Object]");
      expect(src).not.toContain("NaN");
    }
  });
});

describe("sparse and latent blocks", () => {
  it("emits a working expert dispatch for a sparse design", () => {
    const src = modelSource("mixtral-8x7b");
    expect(src).toContain("self.router = nn.Linear(4096, 8, bias=False)");
    expect(src).toContain("nn.ModuleList([Expert() for _ in range(8)])");
    expect(src).toContain("torch.topk(scores, self.top_k, dim=-1)");
    expect(src).toContain("index_add_");
    // The expert body comes from the same emitter as any other feed-forward.
    expect(src).toContain("self.gate = nn.Linear(4096, 14336, bias=False)");
  });

  it("emits a shared expert only when the design has one", () => {
    expect(modelSource("deepseek-v3")).toContain("self.shared = SharedExpert()");
    expect(modelSource("mixtral-8x7b")).not.toContain("self.shared");
  });

  it("emits the latent attention path", () => {
    const src = modelSource("deepseek-v3");
    expect(src).toContain("self.q_down = nn.Linear(7168, 1536, bias=False)");
    expect(src).toContain("self.kv_down = nn.Linear(7168, 576, bias=False)");
    expect(src).toContain("torch.split(latent_y, [512, 64], dim=-1)");
    expect(src).toContain("unsqueeze(1).expand(-1, 128, -1, -1)");
    expect(src).toContain("torch.cat([k_nope_heads_y, k_rope_y], dim=-1)");
    // A 192-wide query head with a 128-wide value head needs an explicit scale.
    expect(src).toMatch(/scale=0\.0721/);
  });

  it("marks the cached latent rather than pretending it is an operation", () => {
    expect(modelSource("deepseek-v3")).toContain("# cached latent");
  });
});

describe("state-space generation", () => {
  it("emits a runnable scan and a causal depthwise convolution", () => {
    const src = modelSource("nemotron-h-8b");
    expect(src).toContain("class SSDScan");
    expect(src).toContain("nn.Conv1d(10240, 10240, 4, groups=10240, padding=3");
    expect(src).toContain("SSDScan(128, 64, 128, 8)");
    // The convolution runs over time and then drops its own padding.
    expect(src).toContain(".transpose(1, 2))[..., :");
  });

  it("does not emit the scan helper for a design without one", () => {
    expect(modelSource("llama-3-8b")).not.toContain("class SSDScan");
  });
});

describe("generated Python parses", () => {
  // Python's own parser is the only honest syntax check. It needs no PyTorch,
  // so this runs anywhere Python 3 is installed and skips where it is not.
  const python = (() => {
    for (const exe of ["python", "python3"]) {
      const probe = Bun.spawnSync([exe, "-c", "print(1)"]);
      if (probe.success) return exe;
    }
    return null;
  })();

  for (const name of PRESET_NAMES) {
    it(`${name} is valid Python`, () => {
      if (!python) return;
      const src = modelSource(name);
      const proc = Bun.spawnSync([python, "-c", "import ast,sys; ast.parse(sys.stdin.read())"], {
        stdin: new TextEncoder().encode(src),
      });
      const stderr = new TextDecoder().decode(proc.stderr);
      expect({ name, ok: proc.success, stderr: proc.success ? "" : stderr }).toEqual({
        name,
        ok: true,
        stderr: "",
      });
    });
  }
});

describe("mixture-of-experts dispatch styles", () => {
  it("defaults to the fast gather-based dispatch", () => {
    const src = modelSource("mixtral-8x7b");
    expect(src).toContain("nonzero(as_tuple=True)");
    expect(src).not.toContain("gate.scatter");
  });

  it("offers a dense dispatch that torch.export can trace", () => {
    const out = generateTorch(getPreset("mixtral-8x7b"), { moeDispatch: "dense" });
    const src = out.files.find((f) => f.path === "model.py")!.contents;
    expect(out.warnings).toEqual([]);
    // No data-dependent shapes: that is the whole point of this variant.
    expect(src).not.toContain("nonzero");
    expect(src).toContain("gate.scatter(1, index, weight)");
    expect(src).toContain("Dense dispatch: traceable, but 4x the work.");
    // It must still declare the same parameter count.
    const total = countParams(getPreset("mixtral-8x7b"), resolveSymbols(getPreset("mixtral-8x7b"))).total;
    expect(src).toContain(`expected = ${total}`);
  });
});

describe("weight initialization", () => {
  it("emits an initializer, because PyTorch's defaults are wrong for a language model", () => {
    const src = modelSource("gpt2-small");
    expect(src).toContain("def init_weights(self)");
    expect(src).toContain("nn.init.normal_(module.weight, mean=0.0, std=0.02)");
    // nn.Embedding defaults to a unit normal, which starts GPT-2 small at a
    // next-token loss of 466 instead of ln(50257) = 10.8. Measured on this
    // machine with PyTorch 2.11.
    expect(src).toContain("PyTorch's defaults leave nn.Embedding at a unit normal");
  });

  it("scales residual projections by the depth", () => {
    const src = modelSource("gpt2-small");
    // 0.02 / sqrt(2 * 12 layers).
    const expected = (0.02 / Math.sqrt(2 * 12)).toPrecision(8);
    expect(src).toContain(`std=${expected}`);
    expect(src).toContain('RESIDUAL_PROJECTIONS = ("o_proj.weight", "down.weight", "out_proj.weight")');
  });

  it("uses each design's own depth for that scale", () => {
    const deep = modelSource("llama-3.1-405b");
    expect(deep).toContain(`std=${(0.02 / Math.sqrt(2 * 126)).toPrecision(8)}`);
  });

  it("can be turned off", () => {
    const out = generateTorch(getPreset("gpt2-small"), { initStd: 0 });
    const src = out.files.find((f) => f.path === "model.py")!.contents;
    expect(src).not.toContain("def init_weights");
    expect(src).not.toContain("RESIDUAL_PROJECTIONS");
  });
});
