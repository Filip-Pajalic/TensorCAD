/**
 * Prompts.
 *
 * Four workflows worth having a slash command for. In Claude Code these show
 * up as `/mcp__tensorcad__design_model` and friends.
 *
 * MCP prompt arguments are strings on the wire, so the schemas take strings and
 * the text says what a good value looks like.
 */

import * as z from "zod";
import type { McpServer } from "@modelcontextprotocol/server";
import { HARDWARE, PRESET_NAMES } from "@tensorcad/engine/node";

const user = (text: string) => ({
  messages: [{ role: "user" as const, content: { type: "text" as const, text } }],
});

const HARDWARE_IDS = HARDWARE.map((h) => h.id).join(", ");

export function registerPrompts(server: McpServer): void {
  server.registerPrompt(
    "design_model",
    {
      title: "Design a model",
      description: "Design a decoder-only LLM to a parameter budget and context length, then check and cost it.",
      argsSchema: z.object({
        target_params: z.string().describe('Parameter budget, e.g. "3B", "8B", "70B".'),
        context_len: z.string().optional().describe('Context length in tokens, e.g. "8192".'),
        family: z.string().optional().describe(`Architecture to start from: ${PRESET_NAMES.join(", ")}, or "empty".`),
      }),
    },
    ({ target_params, context_len, family }) =>
      user(
        [
          `Design a decoder-only language model of about ${target_params} parameters` +
            `${context_len ? ` with a ${context_len}-token context` : ""}` +
            `${family ? `, starting from the ${family} architecture` : ""}.`,
          "",
          "Work like this:",
          `1. ${family && family !== "empty" ? `tensorcad_new_design with preset "${family}"` : "tensorcad_new_design, choosing the closest preset with tensorcad_list_designs first"}.`,
          "2. tensorcad_get_design (outline) to see what you have.",
          "3. tensorcad_apply_ops with set_symbol operations to move L, D, H, Hkv, dh, F and V toward the budget.",
          "   Keep D divisible by H, keep the head dimension in {64, 96, 128, 160, 192, 256} so fused attention",
          "   kernels apply, keep D and F multiples of 128, and keep H divisible by Hkv.",
          "4. tensorcad_validate and fix every error and any warning you can.",
          `5. tensorcad_analyze${context_len ? ` with T=${context_len}` : ""} and report parameters, KV cache per token,`,
          "   training memory per GPU and the Chinchilla-optimal token budget.",
          "",
          "Report the final symbol table, the parameter count against the target, and anything you traded off.",
        ].join("\n"),
      ),
  );

  server.registerPrompt(
    "review_design",
    {
      title: "Review a design",
      description: "Read a design, run the rules, and report what is wrong and what to do about it.",
      argsSchema: z.object({
        design_id: z.string().describe("Design handle. Use tensorcad_list_designs if you do not have one."),
        hardware: z.string().optional().describe(`Device to judge memory against: ${HARDWARE_IDS}.`),
      }),
    },
    ({ design_id, hardware }) =>
      user(
        [
          `Review design ${design_id}.`,
          "",
          `1. tensorcad_get_design with format "outline".`,
          `2. tensorcad_validate${hardware ? ` with hardware "${hardware}"` : ""}.`,
          "3. tensorcad_get_block on anything a finding points at, to see the shapes and parameters for yourself.",
          "4. tensorcad_analyze for the numbers behind the memory and cost findings.",
          "",
          "Report: every error with the exact edit that fixes it, then warnings by how much they cost,",
          "then anything the rules do not catch (unusual width ratios, a vocabulary that dominates the",
          "parameter count, attention that dominates FLOPs at this context length).",
          "Do not change the design unless I ask.",
        ].join("\n"),
      ),
  );

  server.registerPrompt(
    "scale_design",
    {
      title: "Scale a design",
      description: "Scale a design up or down by a factor while keeping its proportions sane.",
      argsSchema: z.object({
        design_id: z.string().describe("Design handle."),
        factor: z.string().describe('Parameter multiplier, e.g. "2", "0.5", or a target such as "70B".'),
      }),
    },
    ({ design_id, factor }) =>
      user(
        [
          `Scale design ${design_id} by ${factor}.`,
          "",
          "1. tensorcad_checkpoint first, labelled with what you are about to try, so tensorcad_restore can undo it.",
          "2. tensorcad_analyze to record the starting numbers.",
          "3. Decide how to spend the factor. Parameters go roughly as L x D^2, so depth is linear and width is",
          "   quadratic; published families widen and deepen together rather than stretching one axis.",
          "4. tensorcad_apply_ops with set_symbol operations. Keep D divisible by H, keep the head dimension in",
          "   {64, 96, 128, 160, 192, 256}, keep D and F multiples of 128, and keep H divisible by Hkv.",
          "5. tensorcad_validate, then tensorcad_analyze again.",
          "",
          "Report the before and after symbol tables, the parameter count against the target, and how KV cache,",
          "training memory and training cost moved. If you cannot hit the target without breaking a constraint,",
          "say which constraint and what the nearest good design is.",
        ].join("\n"),
      ),
  );

  server.registerPrompt(
    "explain_costs",
    {
      title: "Explain the costs",
      description: "Explain what a design costs to train and to serve, and where the money goes.",
      argsSchema: z.object({
        design_id: z.string().describe("Design handle."),
        batch: z.string().optional().describe("Micro-batch size for training, and concurrency for serving."),
        seq: z.string().optional().describe("Sequence length in tokens."),
        hardware: z.string().optional().describe(`Device: ${HARDWARE_IDS}.`),
      }),
    },
    ({ design_id, batch, seq, hardware }) =>
      user(
        [
          `Explain what design ${design_id} costs.`,
          "",
          `1. tensorcad_analyze with${seq ? ` T=${seq}` : " the document's own T"}${batch ? `, B=${batch} and concurrency=${batch}` : ""}` +
            `${hardware ? `, hardware "${hardware}"` : ""}.`,
          "2. Run it again with a different GPU count or ZeRO stage if training does not fit the device.",
          "",
          "Cover, in plain language:",
          "- training memory: weights, gradients, optimizer state and activations, which one dominates, and what",
          "  ZeRO stage or tensor parallelism would make it fit.",
          "- training compute: FLOPs per token, the Chinchilla-optimal token budget, GPU-hours and dollars, and",
          "  what the MFU assumption is doing to that number.",
          "- serving: weights plus KV cache per concurrent sequence, how many sequences fit in device memory,",
          "  decode tokens per second, and whether decode is memory-bound or compute-bound.",
          "- the one change that would most reduce each of those.",
        ].join("\n"),
      ),
  );
}

export const PROMPT_NAMES = ["design_model", "review_design", "scale_design", "explain_costs"] as const;
