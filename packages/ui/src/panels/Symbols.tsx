/**
 * The global parameter table.
 *
 * One edit here rescales the whole model: change `D` and every shape label,
 * every dependent symbol and the parameter count follow on the next frame.
 */

import { useState } from "react";
import { useEditor } from "../state/store.js";
import { useDerived } from "../state/hooks.js";
import { TextField } from "./Field.js";
import Configurations from "./Configurations.js";
import type { SymbolDef } from "@tensor-cad/engine";

type Kind = "design" | "runtime";

interface SymbolView {
  kind: Kind;
  value: string;
  doc: string;
}

function read(def: SymbolDef): SymbolView {
  if (typeof def === "number") return { kind: "design", value: String(def), doc: "" };
  if (typeof def === "string") return { kind: "design", value: def, doc: "" };
  if (def && typeof def === "object" && def.kind === "runtime") {
    return { kind: "runtime", value: String(def.default), doc: def.doc ?? "" };
  }
  if (def && typeof def === "object" && def.kind === "design") {
    return { kind: "design", value: String(def.value), doc: def.doc ?? "" };
  }
  return { kind: "design", value: "", doc: "" };
}

function write(view: SymbolView): SymbolDef {
  const numeric = /^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(view.value.trim());
  if (view.kind === "runtime") {
    return { kind: "runtime", default: numeric ? Number(view.value) : 1, doc: view.doc };
  }
  return {
    kind: "design",
    value: numeric ? Number(view.value) : view.value,
    doc: view.doc,
  };
}

function fmt(n: number | undefined): string {
  if (n === undefined) return "—";
  if (!Number.isFinite(n)) return "NaN";
  return Number.isInteger(n) ? n.toLocaleString("en-US") : String(Number(n.toPrecision(6)));
}

export default function Symbols(): React.ReactElement {
  const doc = useEditor((s) => s.doc);
  const derived = useDerived();
  const [newName, setNewName] = useState("");
  const act = useEditor.getState();

  // As the design is currently built, so the table and the readout agree. A
  // configuration overrides by name, and a symbol it does not mention keeps
  // whatever the design says.
  const overrides = doc.active ? (doc.configurations?.[doc.active]?.symbols ?? {}) : {};
  const entries = Object.entries(doc.symbols).map(
    ([name, def]) => [name, overrides[name] ?? def] as const,
  );

  const update = (name: string, patch: Partial<SymbolView>): void => {
    const view = { ...read(overrides[name] ?? doc.symbols[name]), ...patch };
    act.setSymbol(name, write(view));
  };

  const add = (): void => {
    const name = newName.trim();
    if (!name || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) return;
    if (doc.symbols[name] !== undefined) return;
    act.setSymbol(name, { kind: "design", value: 1, doc: "" });
    setNewName("");
  };

  return (
    <div className="panel__body">
      <Configurations />
      <table className="table table--symbols">
        <thead>
          <tr>
            <th>name</th>
            <th>kind</th>
            <th>value / expression</th>
            <th className="num">=</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, def]) => {
            const view = read(def);
            const evaluated = derived.symbols.values[name];
            const bad = evaluated === undefined || !Number.isFinite(evaluated);
            return (
              <tr key={name} className={bad ? "row--bad" : undefined}>
                <td>
                  <TextField
                    mono
                    value={name}
                    title={derived.symbols.docs[name] || view.doc}
                    onCommit={(next) => act.renameSymbol(name, next.trim())}
                  />
                </td>
                <td>
                  <select
                    className="field"
                    value={view.kind}
                    onChange={(e) => update(name, { kind: e.target.value as Kind })}
                  >
                    <option value="design">design</option>
                    <option value="runtime">runtime</option>
                  </select>
                </td>
                <td>
                  <TextField
                    mono
                    value={view.value}
                    invalid={bad}
                    onCommit={(next) => update(name, { value: next })}
                  />
                </td>
                <td className="mono num">{fmt(evaluated)}</td>
                <td>
                  <button
                    className="btn btn--icon"
                    title={`Delete symbol ${name}`}
                    onClick={() => act.setSymbol(name, undefined)}
                  >
                    &times;
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="symbols__docs">
        {entries.map(([name]) => {
          const text = derived.symbols.docs[name];
          if (!text) return null;
          return (
            <div className="symbols__doc" key={name}>
              <code className="mono">{name}</code>
              <span>{text}</span>
            </div>
          );
        })}
      </div>

      <div className="symbols__add">
        <input
          className="field mono"
          placeholder="new symbol"
          value={newName}
          spellCheck={false}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => {
            e.stopPropagation();
            if (e.key === "Enter") add();
          }}
        />
        <button className="btn" onClick={add}>
          Add
        </button>
      </div>

      {derived.symbols.errors.length > 0 && (
        <div className="section">
          <h3>Symbol errors</h3>
          {derived.symbols.errors.map((e) => (
            <div className="issue issue--error" key={e}>
              <span className="dot dot--error" />
              <span>{e}</span>
            </div>
          ))}
        </div>
      )}

      <p className="hint">
        Runtime symbols (<code className="mono">B</code>, <code className="mono">T</code>) stay
        indeterminate in shape labels and only take their default in the analysis. Design symbols may
        be expressions over earlier symbols, e.g.{" "}
        <code className="mono">ceil_mult(1.3*8/3*D, 1024)</code>.
      </p>
    </div>
  );
}
