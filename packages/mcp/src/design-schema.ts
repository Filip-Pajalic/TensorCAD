/**
 * JSON Schema for the design document, served as `tensorcad://schema/design`.
 *
 * Hand-written rather than derived, because the engine describes the
 * document with TypeScript types and a per-block parameter catalog rather than
 * one monolithic runtime schema. Block parameters are therefore `object` here;
 * `tensorcad_search_catalog` is where the per-type parameter schemas live.
 */

export const DESIGN_JSON_SCHEMA = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  $id: "tensorcad://schema/design",
  title: "TensorCAD design document",
  description:
    "The `.tensorcad.json` format. The graph is the source of truth; the editor's node positions are a view of it.",
  type: "object",
  required: ["version", "meta", "symbols", "graph"],
  additionalProperties: false,
  properties: {
    version: { const: 1, description: "Document format version." },
    meta: {
      type: "object",
      required: ["name"],
      additionalProperties: false,
      properties: {
        name: { type: "string" },
        family: { type: "string", description: "Architecture family, e.g. llama, qwen, gpt2." },
        notes: { type: "string" },
        published: {
          type: "object",
          description: "Reference numbers from the model card or paper, asserted by the regression suite.",
          additionalProperties: false,
          properties: {
            params: { type: "number" },
            activeParams: { type: "number" },
            kvBytesPerToken: { type: "number" },
            source: { type: "string" },
            tolerance: { type: "number", description: "Allowed relative difference. Defaults to 0.5%." },
          },
        },
      },
    },
    symbols: {
      type: "object",
      description:
        "The design's named dimensions. A symbol is a number, an expression over earlier symbols, or a runtime " +
        "dimension (B, T) that stays indeterminate through analysis.",
      additionalProperties: {
        anyOf: [
          { type: "number" },
          { type: "string", description: "Expression over earlier symbols, e.g. \"ceil_mult(1.3*8/3*D, 1024)\"." },
          {
            type: "object",
            required: ["kind", "default"],
            additionalProperties: false,
            properties: {
              kind: { const: "runtime" },
              default: { type: "number" },
              doc: { type: "string" },
            },
          },
          {
            type: "object",
            required: ["kind", "value"],
            additionalProperties: false,
            properties: {
              kind: { const: "design" },
              value: { anyOf: [{ type: "number" }, { type: "string" }] },
              doc: { type: "string" },
            },
          },
        ],
      },
    },
    graph: { $ref: "#/$defs/graph" },
    ui: {
      type: "object",
      description: "Editor state. Ignored by analysis and codegen.",
      additionalProperties: false,
      properties: {
        positions: {
          type: "object",
          additionalProperties: { type: "array", items: { type: "number" }, minItems: 2, maxItems: 2 },
        },
        collapsed: { type: "array", items: { type: "string" } },
      },
    },
  },
  $defs: {
    graph: {
      type: "object",
      required: ["nodes", "edges"],
      additionalProperties: false,
      properties: {
        nodes: { type: "array", items: { $ref: "#/$defs/node" } },
        edges: {
          type: "array",
          description: "Each edge is [\"fromBlock:port\", \"toBlock:port\"], with ids local to this graph.",
          items: {
            type: "array",
            items: { type: "string" },
            minItems: 2,
            maxItems: 2,
          },
        },
      },
    },
    node: {
      type: "object",
      required: ["id", "type"],
      additionalProperties: false,
      properties: {
        id: { type: "string", description: "Unique within its graph." },
        type: { type: "string", description: "A catalog block type; see tensorcad://catalog." },
        params: {
          type: "object",
          description:
            "Block parameters. Numeric parameters accept an expression string over the design symbols. " +
            "The per-type schema is in tensorcad://catalog/{type}.",
          additionalProperties: true,
        },
        graph: { $ref: "#/$defs/graph", description: "Subgraph, for container blocks such as repeat." },
        variants: {
          type: "object",
          description: "Named subgraph variants, for hybrid repeat patterns.",
          additionalProperties: { $ref: "#/$defs/graph" },
        },
        label: { type: "string", description: "Shown on the canvas instead of the id." },
      },
    },
  },
} as const;
