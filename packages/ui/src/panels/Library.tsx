/**
 * The library: the reference architectures, with what each one is.
 *
 * Every preset has carried a paragraph about itself since it was written —
 * `meta.notes`, the thing that says the vocabulary padding in nanoGPT is a
 * speed decision rather than a modelling one and costs 36,096 parameters — and
 * until now the editor rendered it nowhere. Choosing between `qwen3-30b-a3b`
 * and `qwen3-next-80b-a3b` meant loading both and looking.
 *
 * Grouped by family, in the order the engine lists them, which is the order the
 * library is written in: families together, roughly by size. Sorting here would
 * throw that away and put `alexnet` first.
 */

import { useMemo, useState } from "react";
import { getPreset, PRESET_NAMES } from "../engine.js";
import { useEditor } from "../state/store.js";
import { formatCount } from "@tensor-cad/engine";
import { Input } from "../ui/input.js";

interface Entry {
  name: string;
  family: string;
  notes: string;
  params: number | null;
  source: string | null;
}

/**
 * Read once. A preset is a document, so its own metadata is the only place this
 * could come from, and twenty-three of them is twenty-three crossings of the
 * boundary — cheap, but not on every keystroke of the filter.
 */
function readLibrary(): Entry[] {
  return PRESET_NAMES.map((name) => {
    try {
      const doc = getPreset(name);
      return {
        name,
        family: doc.meta.family ?? "other",
        notes: doc.meta.notes ?? "",
        params: doc.meta.published?.params ?? null,
        source: doc.meta.published?.source ?? null,
      };
    } catch {
      // A preset that will not load is still a preset that exists; saying so is
      // better than a list with a silent hole in it.
      return { name, family: "other", notes: "", params: null, source: null };
    }
  });
}

export default function Library(): React.ReactElement {
  const entries = useMemo(readLibrary, []);
  const [query, setQuery] = useState("");
  const close = useEditor((s) => s.closeDialog);
  const load = useEditor((s) => s.loadPreset);
  const open = useEditor((s) => s.doc.meta.name);

  const needle = query.trim().toLowerCase();
  const shown = needle
    ? entries.filter((e) => `${e.name} ${e.family} ${e.notes}`.toLowerCase().includes(needle))
    : entries;

  // Grouped by family, each family in the place it first appears and each
  // design in the order the library declares it — which within a family is
  // ascending size. The list itself interleaves families (llama-2, mistral,
  // llama-3), so following its order strictly would print LLAMA twice.
  const families: [string, Entry[]][] = [];
  const at = new Map<string, Entry[]>();
  for (const entry of shown) {
    let list = at.get(entry.family);
    if (!list) {
      list = [];
      at.set(entry.family, list);
      families.push([entry.family, list]);
    }
    list.push(entry);
  }

  return (
    <div className="library">
      <Input
        className="mb-2 w-full"
        placeholder="Filter by name, family or description"
        value={query}
        spellCheck={false}
        aria-label="Filter the library"
        onChange={(e) => setQuery(e.target.value)}
      />

      {shown.length === 0 && <div className="empty">Nothing in the library matches that.</div>}

      {families.map(([family, list], i) => (
        <section className="library__family" key={`${family}-${i}`}>
          <div className="library__familyname">{family}</div>
          {list.map((e) => (
            <div
              className={`library__item${e.name === open ? " is-open" : ""}`}
              key={e.name}
              role="button"
              tabIndex={0}
              onClick={() => {
                load(e.name);
                close();
              }}
              onKeyDown={(ev) => {
                if (ev.key !== "Enter" && ev.key !== " ") return;
                ev.preventDefault();
                load(e.name);
                close();
              }}
            >
              <div className="library__head">
                <span className="library__name mono">{e.name}</span>
                {e.params !== null && (
                  <span className="library__params" title={e.params.toLocaleString("en-US")}>
                    {formatCount(e.params)}
                  </span>
                )}
              </div>
              {e.notes && <p className="library__notes">{e.notes}</p>}
              {e.source && (
                <a
                  className="library__source"
                  href={e.source}
                  target="_blank"
                  rel="noreferrer"
                  // The row loads the design; the link goes somewhere else, and
                  // one gesture must not do both.
                  onClick={(ev) => ev.stopPropagation()}
                >
                  {e.source.replace(/^https?:\/\//, "")}
                </a>
              )}
            </div>
          ))}
        </section>
      ))}
    </div>
  );
}
