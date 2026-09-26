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
import { categoryColor, categoryName, typeName } from "../canvas/blocks.js";
import { formatShape } from "../canvas/shapes.js";
import type { NodeDef, ParamSpec, ParamValue, Resolved } from "@tensor-cad/engine";
import { formatCount } from "@tensor-cad/engine";
import { blockDef, type BlockDef } from "../engine.js";
import { resolveLevel } from "../state/level.js";
import * as ops from "../state/ops.js";
import { runCommand } from "../state/commands.js";
import MaskPreview from "./MaskPreview.js";

/** What an empty attention expression field suggests. */
const EXPRESSION_EXAMPLES: Record<string, string> = {
  mask: "e.g. kv <= q and q - kv < 1024",
  score: "e.g. score - (q - kv) / 8",
};

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
  onlyWhen,
}: {
  name: string;
  spec: ParamSpec;
  node: NodeDef;
  path: string;
  resolved: Resolved | undefined;
  disabled: boolean;
  /** True when this block's own settings make the field mean nothing. */
  irrelevant?: boolean;
  /** When it does apply, in words: "Only used when feed-forward is gated". */
  onlyWhen?: string;
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
            {/* What the default is, is the parameter's own to say, in its
                doc: o_bias follows attn_bias, sinks are none. */}
            <option value="inherit">default</option>
            <option value="true">on</option>
            <option value="false">off</option>
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
            <span>{effective === true ? "on" : "off"}</span>
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
      // Words where the catalog has them, and the value where it does not: an
      // activation is "SiLU" in a paper and `silu` in a document.
      const words = spec.valueLabels ?? {};
      control = (
        <select
          className={"field" + (Object.keys(words).length ? "" : " mono")}
          disabled={disabled}
          value={effective}
          onChange={(e) => set(e.target.value === "" ? undefined : e.target.value)}
        >
          {effective === "" && <option value="">(unset)</option>}
          {(spec.values ?? []).map((v) => (
            <option key={v} value={v}>
              {words[v] ?? v}
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
    case "mask":
    case "score": {
      // The engine keeps an expression the way it understood it — symbols
      // replaced by their values, constants folded — which is worth showing
      // when it differs from what was typed: it is what the kernel is given.
      const understood = resolved?.p[name];
      const text = rawText(raw);
      control = (
        <div>
          <TextField
            mono
            value={text}
            disabled={disabled}
            placeholder={EXPRESSION_EXAMPLES[spec.type]}
            title={
              spec.type === "mask"
                ? "Which scores count: q, kv, h, b, heads and the design's symbols"
                : "What each score becomes: score, q, kv, h, heads and the design's symbols"
            }
            onCommit={(t) => set(t.trim() === "" ? undefined : t.trim())}
          />
          {typeof understood === "string" && understood !== text.trim() && (
            <div className="param__eval mono" title="As the engine understood it">
              = {understood}
            </div>
          )}
          {spec.type === "mask" && !irrelevant && (
            <MaskPreview path={path} heads={headCount(resolved)} />
          )}
        </div>
      );
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
      title={irrelevant ? onlyWhen : undefined}
    >
      {/*
        The label leads and the name follows, smaller: the label is what the
        field is, the name is what a document, a path and an MCP call write.
        A design's own block may declare no label, and then the name is all
        there is to say.
      */}
      <div className="param__head">
        {spec.label && <span className="param__label">{spec.label}</span>}
        <span
          className={"param__name mono" + (spec.label ? "" : " param__name--alone")}
          title="What a document writes"
        >
          {name}
        </span>
        {missing && !irrelevant && <span className="badge badge--error">required</span>}
        {raw !== undefined && hasDefault && <span className="badge">set</span>}
        {/*
          Named rather than just greyed: "unused" beside the field is an answer,
          where a dimmer row is a thing to wonder about. The value is still in
          the document and still editable — a block may be set up before the
          switch that turns it on.
        */}
        {irrelevant && onlyWhen && (
          <span className="badge badge--quiet" title={onlyWhen}>
            unused
          </span>
        )}
      </div>
      {control}
      {spec.doc && <div className="param__doc">{spec.doc}</div>}
    </div>
  );
}

function headCount(resolved: Resolved | undefined): number {
  const n = resolved?.p.heads;
  return typeof n === "number" && Number.isFinite(n) ? n : 1;
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

/** "Feed-forward" to "feed-forward" for the middle of a sentence; "RMS" stays. */
function lowerFirst(text: string): string {
  return /^[A-Z][a-z]/.test(text) ? text[0]!.toLowerCase() + text.slice(1) : text;
}

/** Whether a block has written a parameter as something other than its default. */
function changed(spec: ParamSpec, raw: ParamValue | undefined): boolean {
  if (raw === undefined) return false;
  return JSON.stringify(raw ?? null) !== JSON.stringify(spec.default ?? null);
}

/**
 * Fields under their headings, in the order their first field appears, so the
 * headings follow the block's own order rather than the alphabet.
 */
function byGroup(names: string[], specs: Record<string, ParamSpec>): { name: string; fields: string[] }[] {
  const groups: { name: string; fields: string[] }[] = [];
  for (const name of names) {
    const group = specs[name]?.group ?? "";
    const existing = groups.find((g) => g.name === group);
    if (existing) existing.fields.push(name);
    else groups.push({ name: group, fields: [name] });
  }
  return groups;
}

/**
 * A block's parameters: the ones that matter, then the rare ones, with the ones
 * that do not apply kept out of the way.
 *
 * `transformer_block` has forty-odd, and at any moment about ten mean nothing
 * — `experts` on a dense block — and another eighteen are things most designs
 * never touch: attention sinks, a score cap, an expression mask. Listing all of
 * them is how a twelve-field decision reads as a forty-field form.
 *
 * So three piles. The fields this block's settings make meaningful and that a
 * design is made of are shown. The rare ones are under Advanced, which starts
 * closed and opens by itself for a block that has changed one, so a design
 * that uses sinks shows its sinks. The ones that do not apply are hidden behind
 * a count that shows them, greyed, in their places: a field that vanishes when
 * `mlp` changes is one you go looking for, and the count says where it went.
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
  const order = (def.paramOrder ?? Object.keys(specs)).filter((name) => specs[name]);
  const applies = (name: string): boolean => meaningful(specs[name], resolved);
  // Said in the other field's words, not its identifier: "feed-forward is
  // mixture of experts" rather than "mlp is moe".
  const onlyWhen = (name: string): string | undefined => {
    const when = specs[name].when;
    if (!when) return undefined;
    const other = specs[when.param];
    const values = when.is.map((v) => other?.valueLabels?.[v] ?? v);
    return `Only used when ${lowerFirst(other?.label ?? when.param)} is ${values.join(" or ")}`;
  };

  const advanced = order.filter((name) => specs[name].advanced);
  const advancedChanged = advanced.filter(
    (name) => applies(name) && changed(specs[name], node.params?.[name]),
  );

  const [showUnused, setShowUnused] = useState(false);
  const [openAdvanced, setOpenAdvanced] = useState(advancedChanged.length > 0);

  // What the count at the bottom would put back, which does not include what
  // is under a closed Advanced: a count that promised twelve and showed nine
  // would be the first thing wrong with it.
  const unused = order.filter((name) => !applies(name) && (!specs[name].advanced || openAdvanced));
  const shown = (name: string): boolean => showUnused || applies(name);
  const basic = order.filter((name) => !specs[name].advanced && shown(name));
  const rare = advanced.filter(shown);

  const rows = (names: string[]): React.ReactElement[] =>
    byGroup(names, specs).map((group) => (
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
            irrelevant={!applies(name)}
            onlyWhen={onlyWhen(name)}
          />
        ))}
      </div>
    ));

  return (
    <section className="section">
      <h3>Parameters</h3>
      {rows(basic)}

      {rare.length > 0 && (
        <div className="param__more">
          <button
            type="button"
            className="param__toggle"
            data-testid="advanced"
            aria-expanded={openAdvanced}
            onClick={() => setOpenAdvanced(!openAdvanced)}
          >
            <span className="fold__caret" aria-hidden>
              {openAdvanced ? "▾" : "▸"}
            </span>
            Advanced
            <span className="param__count">
              {advancedChanged.length > 0 ? `${advancedChanged.length} changed` : rare.length}
            </span>
          </button>
          {openAdvanced && rows(rare)}
        </div>
      )}

      {unused.length > 0 && (
        <button
          type="button"
          className="param__toggle param__toggle--quiet"
          data-testid="show-unused"
          aria-pressed={showUnused}
          title="Fields this block's own settings make meaningless. Their values are still in the document."
          onClick={() => setShowUnused(!showUnused)}
        >
          {showUnused
            ? `Hide the ${unused.length} that don’t apply`
            : `${unused.length} more that don’t apply`}
        </button>
      )}
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
        {/*
          The panel somebody new is looking at before they have done anything,
          so it says the two things there are to do: click a block, or be shown
          round. The walkthrough is the one on the Help menu; this is the place
          it can be found without knowing that.
        */}
        <div className="inspector__start" data-testid="inspector-start">
          <p>Click any block in the drawing to see what it is and change it.</p>
          <p className="muted">
            New to this? The walkthrough goes through the design one stage at a time, with its
            own numbers.
          </p>
          <button
            type="button"
            className="btn btn--primary"
            data-testid="start-here"
            onClick={() => runCommand("help.start")}
          >
            Start here
          </button>
          <p className="muted">
            Or make one of your own: a kind of model and a size.{" "}
            <button type="button" className="linkish" data-testid="new-design-link" onClick={() => runCommand("file.new")}>
              New design…
            </button>
          </p>
        </div>
      </div>
    );
  }

  const localId = level.prefix
    ? selection.startsWith(level.prefix + "/")
      ? selection.slice(level.prefix.length + 1)
      : null
    : selection;
  let where = level;
  let node = localId && !localId.includes("/") ? level.graph.nodes.find((n) => n.id === localId) : null;
  // A block drawn inside an unfolded frame is on a deeper level than the one
  // open, and clicking it is still asking what it is. So the inspector finds
  // the level it lives on and shows it there — editable if that level is,
  // which a stack's interior is and a built-in block's is not.
  const segments = ops.segmentsOf(selection);
  if (!node) {
    const parent = resolveLevel(doc, segments.slice(0, -1), derived);
    if (!parent.error) {
      where = parent;
      node = parent.graph.nodes.find((n) => n.id === segments[segments.length - 1]) ?? null;
    }
  }

  if (!node) {
    return (
      <div className="panel__body">
        <div className="empty">
          <code>{selection}</code> is not in this design.
        </div>
      </div>
    );
  }
  const elsewhere = where !== level;

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
  const disabled = !where.editable;
  const act = useEditor.getState();
  const openWhereItIs = (): void => {
    act.setPath(segments.slice(0, -1));
    act.select(selection);
  };

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
          <span className="badge">{categoryName(def?.category)}</span>
          {params > 0 && (
            <span className="badge badge--num mono" title={`${params.toLocaleString("en-US")} parameters`}>
              {formatCount(params)}
            </span>
          )}
        </div>
      </div>

      {elsewhere && (
        <p className="inspector__where" data-testid="inspector-where">
          {where.editable ? (
            <>Inside {where.owner?.label ?? where.owner?.id}, drawn open on this sheet. </>
          ) : (
            <>
              Part of {where.owner?.label ?? where.owner?.id}, a built-in{" "}
              {typeName(where.owner ? blockDef(where.owner.type, doc) : undefined, where.owner?.type ?? "")}
              , so it is read-only: its settings come from that block&rsquo;s.{" "}
            </>
          )}
          <button type="button" className="linkish" onClick={openWhereItIs}>
            Open its level
          </button>
        </p>
      )}

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
          // Keyed on the block, so what is open starts again for each one.
          key={selection}
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
