/**
 * Every command, one keystroke away.
 *
 * The menu groups thirty-odd commands behind six dropdowns, which is the right
 * way to *browse* them and the wrong way to reach one you already know the name
 * of. `state/commands.ts` is already the single list behind the keyboard, the
 * menu, the shortcut sheet and the native menu, so a palette over it costs one
 * component and cannot fall out of step with any of them.
 *
 * A command that is not available right now is shown and not run, rather than
 * hidden. "Export blocks is greyed out" is an answer; a command that vanishes
 * when the design has no blocks of its own is a command you conclude does not
 * exist.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useEditor } from "../state/store.js";
import { COMMANDS, prettyShortcut, type Command, type CommandGroup } from "../state/commands.js";

const GROUP_LABEL: Record<CommandGroup, string> = {
  file: "File",
  edit: "Edit",
  view: "View",
  panel: "Panels",
  blocks: "Blocks",
  help: "Help",
};

/** True where a character begins a word, which is where initials land. */
function startsWord(haystack: string, at: number): boolean {
  return at === 0 || !/[a-z0-9]/.test(haystack[at - 1]);
}

/**
 * Whether every character of the query appears in order, and how well.
 *
 * Subsequence rather than substring, because "eb" is how anyone who has used a
 * palette reaches for "Export blocks". Lower is better, and two things make a
 * match better: each character landing at the start of a word, which is what
 * typing initials means, and the whole run being tight rather than scattered
 * across a sentence.
 *
 * Word starts are weighted above tightness on purpose. Without that, "eb" ranks
 * "Vi[e]w [B]oth docks" above "Export blocks", because those two letters happen
 * to sit three apart — which is arithmetically true and not what was asked for.
 */
function match(label: string, query: string): number | null {
  if (query === "") return 0;
  const haystack = label.toLowerCase();
  let at = -1;
  let first = -1;
  let initials = 0;
  for (const ch of query) {
    at = haystack.indexOf(ch, at + 1);
    if (at === -1) return null;
    if (first === -1) first = at;
    if (startsWord(haystack, at)) initials++;
  }
  const missed = query.length - initials;
  return missed * 10_000 + (at - first) * 100 + first;
}

interface Hit {
  command: Command;
  score: number;
  enabled: boolean;
}

export default function CommandPalette(): React.ReactElement {
  const close = useEditor((s) => s.closeDialog);
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => input.current?.focus(), []);

  const hits = useMemo((): Hit[] => {
    const q = query.trim().toLowerCase();
    const seen = new Set<string>();
    const out: Hit[] = [];
    for (const command of COMMANDS) {
      // The same label twice is one command with two chords, and the palette
      // is a list of things you can do rather than of ways to do them.
      if (seen.has(command.label)) continue;
      const score = match(`${GROUP_LABEL[command.group]} ${command.label}`, q);
      if (score === null) continue;
      seen.add(command.label);
      out.push({ command, score, enabled: command.enabled ? command.enabled() : true });
    }
    // By how well it matched, and only then by whether it is available. The
    // other way round buries a command you named exactly under one you did
    // not — which is the same failure as hiding it, arrived at politely. The
    // row is greyed and cannot be run; that is what says it is unavailable.
    out.sort((a, b) => {
      if (a.score !== b.score) return a.score - b.score;
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1;
      return a.command.label.localeCompare(b.command.label);
    });
    return out;
  }, [query]);

  // A filter that moves the list has to move the cursor with it, or Enter runs
  // whatever happens to be sitting at the old index.
  useEffect(() => setCursor(0), [query]);

  useEffect(() => {
    list.current?.querySelector('[data-at="1"]')?.scrollIntoView({ block: "nearest" });
  }, [cursor, hits]);

  const run = (hit: Hit | undefined): void => {
    if (!hit || !hit.enabled) return;
    // Close first: a command that opens another dialog would otherwise be
    // closed by this one on the way out.
    close();
    hit.command.run();
  };

  const onKey = (e: React.KeyboardEvent): void => {
    if (e.key === "ArrowDown" || (e.key === "n" && e.ctrlKey)) {
      e.preventDefault();
      setCursor((c) => Math.min(hits.length - 1, c + 1));
    } else if (e.key === "ArrowUp" || (e.key === "p" && e.ctrlKey)) {
      e.preventDefault();
      setCursor((c) => Math.max(0, c - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      run(hits[cursor]);
    } else if (e.key === "Home") {
      e.preventDefault();
      setCursor(0);
    } else if (e.key === "End") {
      e.preventDefault();
      setCursor(Math.max(0, hits.length - 1));
    }
  };

  return (
    <div className="cpal">
      <input
        ref={input}
        className="field cpal__query"
        value={query}
        placeholder="Type a command"
        spellCheck={false}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={onKey}
        aria-label="Command"
      />
      <div className="cpal__list" ref={list} role="listbox">
        {hits.length === 0 && <div className="empty">Nothing matches.</div>}
        {hits.map((hit, i) => (
          <button
            key={hit.command.id}
            type="button"
            role="option"
            aria-selected={i === cursor}
            data-at={i === cursor ? "1" : undefined}
            className={
              "cpal__row" +
              (i === cursor ? " cpal__row--at" : "") +
              (hit.enabled ? "" : " cpal__row--off")
            }
            disabled={!hit.enabled}
            onMouseMove={() => setCursor(i)}
            onClick={() => run(hit)}
            title={hit.command.hint}
          >
            <span className="cpal__group">{GROUP_LABEL[hit.command.group]}</span>
            <span className="cpal__label">
              {hit.command.label}
              {hit.command.checked?.() ? <span className="cpal__on"> on</span> : null}
            </span>
            <span className="cpal__key mono">{prettyShortcut(hit.command.shortcut)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
