/**
 * Resources.
 *
 * Almost no design tool exposes its document as an MCP resource, which is a
 * gap worth closing: resources are `@`-mentionable in Claude Code, they are
 * cache friendly, and they let a client read a design without spending a tool
 * call on it.
 */

import { analyze, CATALOG, validate } from "@tensorcad/core";
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/server";
import { DESIGN_JSON_SCHEMA } from "./design-schema.js";
import { allCatalogEntries, analysisJson, catalogEntry, findingsJson, outlineOf } from "./summarize.js";
import type { DocumentStore } from "./store/types.js";

const JSON_MIME = "application/json";

const json = (uri: URL, value: unknown) => ({
  contents: [{ uri: uri.href, mimeType: JSON_MIME, text: `${JSON.stringify(value, null, 2)}\n` }],
});

export function registerResources(server: McpServer, store: DocumentStore): void {
  // The design itself. Registered before the bare {id} template so the more
  // specific URIs win, although the RFC 6570 expansion would not cross a "/"
  // in any case.
  server.registerResource(
    "design-validation",
    new ResourceTemplate("tensorcad://designs/{id}/validation", { list: undefined, complete: { id: completeDesignId(store) } }),
    {
      title: "Design rule report",
      description: "Every finding for a design: shape errors, memory fit, kernel constraints, Chinchilla sanity.",
      mimeType: JSON_MIME,
    },
    (uri, { id }) => {
      const record = store.get(String(id));
      const report = validate(record.doc);
      return json(uri, {
        design_id: record.design_id,
        revision: record.revision,
        name: record.name,
        ok: report.ok,
        counts: report.counts,
        findings: findingsJson(report),
      });
    },
  );

  server.registerResource(
    "design-analysis",
    new ResourceTemplate("tensorcad://designs/{id}/analysis", { list: undefined, complete: { id: completeDesignId(store) } }),
    {
      title: "Design analysis",
      description: "Parameters, FLOPs, KV cache, memory, throughput and cost at the document's own defaults.",
      mimeType: JSON_MIME,
    },
    (uri, { id }) => {
      const record = store.get(String(id));
      const result = analyze(record.doc);
      return json(uri, { design_id: record.design_id, revision: record.revision, ...analysisJson(result) });
    },
  );

  server.registerResource(
    "design",
    new ResourceTemplate("tensorcad://designs/{id}", {
      list: () => ({
        resources: store.list().map((d) => ({
          uri: `tensorcad://designs/${d.design_id}`,
          name: d.name,
          title: `${d.name} (revision ${d.revision})`,
          description: `${d.source} design${d.dirty ? ", unsaved changes" : ""}`,
          mimeType: JSON_MIME,
        })),
      }),
      complete: { id: completeDesignId(store) },
    }),
    {
      title: "Design document",
      description: "The literal .tensorcad.json document, with a compact outline beside it.",
      mimeType: JSON_MIME,
    },
    (uri, { id }) => {
      const record = store.get(String(id));
      return json(uri, {
        design_id: record.design_id,
        revision: record.revision,
        dirty: record.dirty,
        ...(record.path ? { path: record.path } : {}),
        outline: outlineOf(record.doc),
        document: record.doc,
      });
    },
  );

  // Catalog -----------------------------------------------------------------
  server.registerResource(
    "catalog",
    "tensorcad://catalog",
    {
      title: "Block catalog",
      description:
        "Every block type with its parameter schema, port patterns and documentation. " +
        "Primitives carry the formulas; composites expand into primitives; repeat is the only container.",
      mimeType: JSON_MIME,
    },
    (uri) => {
      const blocks = allCatalogEntries();
      return json(uri, {
        count: blocks.length,
        categories: [...new Set(blocks.map((b) => b.category))].sort(),
        blocks,
      });
    },
  );

  server.registerResource(
    "catalog-block",
    new ResourceTemplate("tensorcad://catalog/{type}", {
      list: () => ({
        resources: Object.keys(CATALOG)
          .sort()
          .map((type) => ({
            uri: `tensorcad://catalog/${type}`,
            name: type,
            title: type,
            description: CATALOG[type].docs.summary,
            mimeType: JSON_MIME,
          })),
      }),
      complete: {
        type: (value: string) =>
          Object.keys(CATALOG)
            .filter((t) => t.startsWith(value))
            .sort()
            .slice(0, 50),
      },
    }),
    { title: "Catalog block", description: "One block type in full.", mimeType: JSON_MIME },
    (uri, { type }) => {
      const def = CATALOG[String(type)];
      if (!def) {
        throw new Error(`Unknown block type "${String(type)}". Known: ${Object.keys(CATALOG).sort().join(", ")}`);
      }
      return json(uri, catalogEntry(def));
    },
  );

  // Document format ---------------------------------------------------------
  server.registerResource(
    "design-schema",
    "tensorcad://schema/design",
    {
      title: "Design document schema",
      description: "JSON Schema for the .tensorcad.json format.",
      mimeType: "application/schema+json",
    },
    (uri) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "application/schema+json",
          text: `${JSON.stringify(DESIGN_JSON_SCHEMA, null, 2)}\n`,
        },
      ],
    }),
  );
}

function completeDesignId(store: DocumentStore) {
  return (value: string) =>
    store
      .list()
      .map((d) => d.design_id)
      .filter((id) => id.startsWith(value));
}
