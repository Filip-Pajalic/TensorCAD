/**
 * The parameter editor for the selected block.
 *
 * Numeric parameters are expressions over the document's symbols, so they are
 * edited as text and the evaluated number is shown beside the field. That is
 * the whole point of the symbol table: `ffn_hidden` reads `F`, and `F` reads
 * `ceil_mult(1.3*8/3*D, 1024)`.
 */

import { useState } from "react";
import { useEditor } from "../state/store.js";
import { useLevel } from "../state/hooks.js";
import { TextArea, TextField } from "./Field.js";
import { categoryColor } from "../canvas/blocks.js";
import { formatShape } from "../canvas/shapes.js";
import type { NodeDef, ParamSpec, ParamValue, Resolved } from "@tensorcad/engine";
import { formatCount } from "@tensorcad/engine";
import { CATALOG, type BlockDef } from "../engine.js";

function isPlainNumber(text: string): boolean {
  return /^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(text.trim());
}

function rawText(value: ParamValue | undefined): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "number" || typeof value === "string") return String(value);
  return JSON.stringify(value);
}

function evaluatedText(resolved: Resolved | undefined, key: string): string | null {
  if (!resolved) return null;
  const v = resolved.p[key];
  if (typeof v !== "number" || !Number.isFinite(v)) return null;
  return Number.isInteger(v) ? v.toLocaleString("en-US") : String(Number(v.toPrecision(6)));
}

function JsonField({
  value,
  disabled,
  onCommit,
}: {
  value: ParamValue | undefined;
  disabled: boolean;
  onCommit: (next: ParamValue | undefined) => void;
}): React.ReactElement {
  const [error, setError] = useState<string | null>(null);
  const text = value === undefined ? "" : JSON.stringify(value, null, 2);
  return (
    <div>
      <TextArea
        value={text}
        disabled={disabled}
        invalid={error !== null}
        rows={Math.min(8, Math.max(2, text.split("\n").length))}
        onCommit={(next) => {
          const trimmed = next.trim();
          if (trimmed === "") {
            setError(null);
            onCommit(undefined);
            return;
          }
          try {
            onCommit(JSON.parse(trimmed) as ParamValue);
            setError(null);
          } catch (e) {
            setError((e as Error).message);
          }
        }}
      />
      {error && <div className="param__error">{error}</div>}
    </div>
  );
}

function ParamRow({
  name,
  spec,
  node,
  path,
  resolved,
  disabled,
}: {
  name: string;
  spec: ParamSpec;
  node: NodeDef;
  path: string;
  resolved: Resolved | undefined;
  disabled: boolean;
}): React.ReactElement {
  const raw = node.params?.[name];
  const hasDefault = (spec as { default?: ParamValue }).default !== undefined;
  const missing =
    raw === undefined && !hasDefault && spec.type !== "bool" && spec.type !== "obj";
  const set = (value: ParamValue | undefined): void =>
    useEditor.getState().setParam(path, name, value);

  let control: React.ReactElement;
  switch (spec.type) {
    case "int":
    case "num": {
      const evaluated = evaluatedText(resolved, name);
      control = (
        <div className="param__numeric">
          <TextField
            mono
            value={rawText(raw)}
            disabled={disabled}
            invalid={missing}
            placeholder={hasDefault ? String((spec as { default?: ParamValue }).default) : "required"}
            title="A number or an expression over the symbols"
            onCommit={(text) => {
              const t = text.trim();
              if (t === "") set(undefined);
              else set(isPlainNumber(t) ? Number(t) : t);
            }}
          />
          {evaluated !== null && (
            <span className="param__eval mono" title="Evaluated with the current symbol values">
              = {evaluated}
            </span>
          )}
        </div>
      );
      break;
    }
    case "bool": {
      const triState = (spec as { default?: boolean | null }).default === null;
      const effective = raw === undefined ? (spec as { default?: boolean | null }).default : raw;
      if (triState) {
        control = (
          <select
            className="field"
            disabled={disabled}
            value={effective === true ? "true" : effective === false ? "false" : "inherit"}
            onChange={(e) =>
              set(e.target.value === "inherit" ? null : e.target.value === "true")
            }
          >
            <option value="inherit">inherit</option>
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
        );
      } else {
        control = (
          <label className="param__check">
            <input
              type="checkbox"
              disabled={disabled}
              checked={effective === true}
              onChange={(e) => set(e.target.checked)}
            />
            <span className="mono">{effective === true ? "true" : "false"}</span>
          </label>
        );
      }
      break;
    }
    case "enum": {
      // An enum's value is a string; a default that is not one belongs to a
      // parameter that was declared wrong, and showing "unset" is the honest
      // reading of it.
      const fallback = typeof spec.default === "string" ? spec.default : "";
      const effective = raw === undefined ? fallback : String(raw ?? "");
      control = (
        <select
          className="field mono"
          disabled={disabled}
          value={effective}
          onChange={(e) => set(e.target.value === "" ? undefined : e.target.value)}
        >
          {effective === "" && <option value="">(unset)</option>}
          {(spec.values ?? []).map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      );
      break;
    }
    case "obj": {
      control = <JsonField value={raw} disabled={disabled} onCommit={set} />;
      break;
    }
    default: {
      control = (
        <TextField
          mono
          value={rawText(raw)}
          disabled={disabled}
          invalid={missing}
          placeholder={hasDefault ? String((spec as { default?: ParamValue }).default) : "required"}
          onCommit={(text) => set(text.trim() === "" ? undefined : text)}
        />
      );
    }
  }

  return (
    <div className={`param${missing ? " param--missing" : ""}`}>
      <div className="param__head">
        <span className="param__name mono">{name}</span>
        <span className="param__type">{spec.type}</span>
        {missing && <span className="badge badge--error">required</span>}
        {raw !== undefined && hasDefault && <span className="badge">set</span>}
      </div>
      {control}
      {spec.doc && <div className="param__doc">{spec.doc}</div>}
    </div>
  );
}

export default function Inspector(): React.ReactElement {
  const selection = useEditor((s) => s.selection);
  const shapeMode = useEditor((s) => s.shapeMode);
  const { level, derived } = useLevel();

  if (!selection) {
    return (
      <div className="panel__body">
        <div className="empty">Select a block on the canvas to edit its parameters.</div>
      </div>
    );
  }

  const localId = level.prefix
    ? selection.startsWith(level.prefix + "/")
      ? selection.slice(level.prefix.length + 1)
      : null
    : selection;
  const node = localId && !localId.includes("/") ? level.graph.nodes.find((n) => n.id === localId) : null;

  if (!node) {
    return (
      <div className="panel__body">
        <div className="empty">
          <code>{selection}</code> is not on this level.
        </div>
      </div>
    );
  }

  const def: BlockDef | undefined = CATALOG[node.type];
  const resolved = derived.infer.resolved.get(selection);
  const ports = derived.infer.ports.get(selection);
  const params = derived.paramsByPath.get(selection) ?? 0;
  const issues = derived.issues.filter((i) => i.path === selection);
  const disabled = !level.editable;
  const act = useEditor.getState();

  return (
    <div className="panel__body inspector">
      <div className="inspector__title" style={{ ["--accent" as string]: categoryColor(def?.category) }}>
        <div className="inspector__type mono">{node.type}</div>
        <div className="inspector__meta">
          <span className="badge">{def?.kind ?? "unknown"}</span>
          <span className="badge">{def?.category ?? "?"}</span>
          {params > 0 && (
            <span className="badge badge--num mono" title={`${params.toLocaleString("en-US")} parameters`}>
              {formatCount(params)}
            </span>
          )}
        </div>
      </div>

      {def?.docs.summary && <p className="inspector__summary">{def.docs.summary}</p>}
      {def?.docs.formula && <pre className="inspector__formula mono">{def.docs.formula}</pre>}
      {def?.docs.refs && def.docs.refs.length > 0 && (
        <div className="inspector__refs">
          {def.docs.refs.map((r) => (
            <a key={r} href={r} target="_blank" rel="noreferrer">
              {r.replace(/^https?:\/\//, "").slice(0, 46)}
            </a>
          ))}
        </div>
      )}

      <section className="section">
        <h3>Identity</h3>
        <label className="row">
          <span className="row__label">label</span>
          <TextField
            value={node.label ?? ""}
            disabled={disabled}
            placeholder={node.id}
            onCommit={(text) => act.renameNode(selection, text)}
          />
        </label>
        <label className="row">
          <span className="row__label">id</span>
          <TextField
            mono
            value={node.id}
            disabled={disabled}
            onCommit={(text) => act.setNodeId(selection, text)}
          />
        </label>
        <div className="row">
          <span className="row__label">path</span>
          <code className="row__value mono">{selection}</code>
        </div>
      </section>

      {def && Object.keys(def.params ?? {}).length > 0 && (
        <section className="section">
          <h3>Parameters</h3>
          {Object.entries(def.params as Record<string, ParamSpec>).map(([name, spec]) => (
            <ParamRow
              key={name}
              name={name}
              spec={spec}
              node={node}
              path={selection}
              resolved={resolved}
              disabled={disabled}
            />
          ))}
        </section>
      )}

      {ports && (Object.keys(ports.in).length > 0 || Object.keys(ports.out).length > 0) && (
        <section className="section">
          <h3>Ports</h3>
          <table className="table">
            <tbody>
              {Object.keys(ports.in).map((p) => {
                const shape = derived.infer.inputs.get(`${selection}:${p}`);
                return (
                  <tr key={`in-${p}`}>
                    <td className="dim">in</td>
                    <td className="mono">{p}</td>
                    <td className="mono num">{formatShape(shape, shapeMode) ?? "—"}</td>
                  </tr>
                );
              })}
              {Object.keys(ports.out).map((p) => {
                const shape = derived.infer.outputs.get(`${selection}:${p}`);
                return (
                  <tr key={`out-${p}`}>
                    <td className="dim">out</td>
                    <td className="mono">{p}</td>
                    <td className="mono num">{formatShape(shape, shapeMode) ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}

      {issues.length > 0 && (
        <section className="section">
          <h3>Issues</h3>
          {issues.map((i) => (
            <div key={i.key} className={`issue issue--${i.severity}`}>
              <span className={`dot dot--${i.severity}`} />
              <span>{i.message}</span>
            </div>
          ))}
        </section>
      )}

      {!disabled && (
        <section className="section">
          <button className="btn btn--danger" onClick={() => act.removeNode(selection)}>
            Delete block
          </button>
        </section>
      )}
    </div>
  );
}
