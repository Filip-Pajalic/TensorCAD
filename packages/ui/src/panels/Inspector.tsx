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
import { categoryColor, typeName } from "../canvas/blocks.js";
import { formatShape } from "../canvas/shapes.js";
import type { NodeDef, ParamSpec, ParamValue, Resolved } from "@tensor-cad/engine";
import { formatCount } from "@tensor-cad/engine";
import { blockDef, type BlockDef } from "../engine.js";

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
  irrelevant = false,
}: {
  name: string;
  spec: ParamSpec;
  node: NodeDef;
  path: string;
  resolved: Resolved | undefined;
  disabled: boolean;
  /** True when this block's own settings make the field mean nothing. */
  irrelevant?: boolean;
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
    <div
      className={
        `param${missing ? " param--missing" : ""}` + (irrelevant ? " param--irrelevant" : "")
      }
      title={
        irrelevant && spec.when
          ? `Only used when ${spec.when.param} is ${spec.when.is.join(" or ")}`
          : undefined
      }
    >
      <div className="param__head">
        <span className="param__name mono">{name}</span>
        <span className="param__type">{spec.type}</span>
        {missing && !irrelevant && <span className="badge badge--error">required</span>}
        {raw !== undefined && hasDefault && <span className="badge">set</span>}
        {/*
          Named rather than just greyed: "unused" beside the field is an answer,
          where a dimmer row is a thing to wonder about. The value is still in
          the document and still editable — a block may be set up before the
          switch that turns it on.
        */}
        {irrelevant && spec.when && (
          <span className="badge badge--quiet" title={`Only used when ${spec.when.param} is ${spec.when.is.join(" or ")}`}>
            unused
          </span>
        )}
      </div>
      {control}
      {spec.doc && <div className="param__doc">{spec.doc}</div>}
    </div>
  );
}

/**
 * Whether a parameter means anything given what the block's others are set to.
 *
 * The resolved value rather than the raw one, so a condition reads through a
 * default the document never wrote: a block that says nothing about `mlp` is
 * still a gated one, and its `experts` field still means nothing.
 */
function meaningful(spec: ParamSpec, resolved: Resolved | undefined): boolean {
  if (!spec.when) return true;
  const value = resolved?.p?.[spec.when.param];
  if (value === undefined) return true;
  return spec.when.is.includes(String(value));
}

/**
 * A block's parameters, under headings, with the irrelevant ones greyed.
 *
 * `transformer_block` has twenty-five and about ten of them mean nothing at any
 * moment. Greyed rather than hidden: a field that disappears when you change
 * `mlp` is one you go looking for, and the value is still in the document
 * either way.
 */
function Parameters({
  def,
  node,
  path,
  resolved,
  disabled,
}: {
  def: BlockDef;
  node: NodeDef;
  path: string;
  resolved: Resolved | undefined;
  disabled: boolean;
}): React.ReactElement {
  const specs = def.params as Record<string, ParamSpec>;
  // The block's declared order, which the catalog carries beside the map
  // because a JSON object's is only its insertion order and a Go map has none.
  const order = def.paramOrder ?? Object.keys(specs);

  // Groups in the order their first field appears, so the headings follow the
  // block's own order rather than the alphabet.
  const groups: { name: string; fields: string[] }[] = [];
  for (const name of order) {
    const spec = specs[name];
    if (!spec) continue;
    const group = spec.group ?? "";
    const existing = groups.find((g) => g.name === group);
    if (existing) existing.fields.push(name);
    else groups.push({ name: group, fields: [name] });
  }

  const dim = groups.reduce(
    (n, g) => n + g.fields.filter((f) => !meaningful(specs[f], resolved)).length,
    0,
  );

  return (
    <section className="section">
      <h3>
        Parameters
        {dim > 0 && (
          <span className="section__note" title="Fields this block's own settings make irrelevant">
            {dim} not in use
          </span>
        )}
      </h3>
      {groups.map((group) => (
        <div className="param__group" key={group.name || "_"}>
          {group.name && <div className="param__groupName">{group.name}</div>}
          {group.fields.map((name) => (
            <ParamRow
              key={name}
              name={name}
              spec={specs[name]}
              node={node}
              path={path}
              resolved={resolved}
              disabled={disabled}
              irrelevant={!meaningful(specs[name], resolved)}
            />
          ))}
        </div>
      ))}
    </section>
  );
}

export default function Inspector(): React.ReactElement {
  const selection = useEditor((s) => s.selection);
  const also = useEditor((s) => s.also);
  const shapeMode = useEditor((s) => s.shapeMode);
  const doc = useEditor((s) => s.doc);
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

  // Through the document's own catalog (invariant 1), or a design's own
  // block inspects as an unknown kind with no documentation.
  const def: BlockDef | undefined = blockDef(node.type, doc);
  // The inspector edits one block, because a parameter belongs to one block.
  // Saying which one is being edited, when several are selected, is the least
  // it can do — the alternative is a panel that silently ignores the rest.
  const multiple = also.length > 0;
  const resolved = derived.infer.resolved.get(selection);
  const ports = derived.infer.ports.get(selection);
  const params = derived.paramsByPath.get(selection) ?? 0;
  const issues = derived.issues.filter((i) => i.path === selection);
  const disabled = !level.editable;
  const act = useEditor.getState();

  return (
    <div className="panel__body inspector">
      <div className="inspector__title" style={{ ["--accent" as string]: categoryColor(def?.category) }}>
        {/*
          The name leads and the identifier follows it, because this is the
          panel where the identifier is actually wanted: a reader who found a
          block by its drawing comes here to learn what to type.
        */}
        <div className="inspector__type">{typeName(def, node.type)}</div>
        <div className="inspector__meta">
          <span className="badge badge--id mono" title="the type, as a path and an MCP call write it">
            {node.type}
          </span>
          <span className="badge">{def?.kind ?? "unknown"}</span>
          <span className="badge">{def?.category ?? "?"}</span>
          {params > 0 && (
            <span className="badge badge--num mono" title={`${params.toLocaleString("en-US")} parameters`}>
              {formatCount(params)}
            </span>
          )}
        </div>
      </div>

      {multiple && (
        <p className="inspector__multiple">
          {also.length + 1} blocks are selected. Parameters belong to one block, so this edits the
          last one you picked; Delete, Duplicate and Lock act on all of them.
        </p>
      )}

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
        <Parameters
          def={def}
          node={node}
          path={selection}
          resolved={resolved}
          disabled={disabled}
        />
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
                    <td className="mono num">
                      {formatShape(shape, shapeMode, derived.symbols) ?? "—"}
                    </td>
                  </tr>
                );
              })}
              {Object.keys(ports.out).map((p) => {
                const shape = derived.infer.outputs.get(`${selection}:${p}`);
                return (
                  <tr key={`out-${p}`}>
                    <td className="dim">out</td>
                    <td className="mono">{p}</td>
                    <td className="mono num">
                      {formatShape(shape, shapeMode, derived.symbols) ?? "—"}
                    </td>
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
