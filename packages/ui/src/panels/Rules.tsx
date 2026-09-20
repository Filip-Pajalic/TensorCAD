/**
 * The design-rule check.
 *
 * The same idea as a PCB tool's DRC list: every rule the design breaks, sorted
 * worst first, each one clickable so it opens the level that owns the offending
 * block, selects it and centres the viewport. A rule that fires tells you what
 * it found and what to do about it; the rule book at the bottom says what every
 * rule is looking for, whether or not it fired.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useEditor } from "../state/store.js";
import { useDerived } from "../state/hooks.js";
import type { Severity, UiIssue } from "../state/derive.js";
import Section from "./Section.js";
import { RULES } from "../engine.js";

const SEVERITIES: Severity[] = ["error", "warning", "info"];

const TITLE_BY_RULE: Record<string, string> = Object.fromEntries(RULES.map((r) => [r.id, r.title]));

function Group({
  severity,
  issues,
  focus,
}: {
  severity: Severity;
  issues: UiIssue[];
  /** The block whose marker was pressed on the drawing, if any. */
  focus: string | null;
}): React.ReactElement | null {
  if (issues.length === 0) return null;
  return (
    <div className="issues">
      {issues.map((i) => (
        <div
          key={i.key}
          data-focus={focus !== null && i.path === focus ? "1" : undefined}
          className={
            `issue issue--${severity}${i.path ? " clickable" : ""}` +
            (focus !== null && i.path === focus ? " issue--focus" : "")
          }
          onClick={() => i.path && useEditor.getState().focusOn(i.path)}
          title={i.path ? `Open ${i.path}` : undefined}
        >
          <span className={`dot dot--${severity}`} />
          <div className="issue__body">
            <div className="issue__message">{i.message}</div>
            {i.hint && <div className="issue__hint">{i.hint}</div>}
            <div className="issue__where mono">
              <span className="badge badge--tiny" title={TITLE_BY_RULE[i.rule] ?? i.rule}>
                {i.rule}
              </span>
              {i.path ?? "document"}
              {i.port ? `:${i.port}` : ""}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function Rules(): React.ReactElement {
  const derived = useDerived();
  const focus = useEditor((s) => s.findingFocus);
  const focusNonce = useEditor((s) => s.findingNonce);
  const body = useRef<HTMLDivElement>(null);
  const [muted, setMuted] = useState<Set<Severity>>(() => new Set<Severity>(["info"]));

  // Pressing a marker on the drawing opens this list on that block. Scroll to
  // it, because the list is long and landing on the right tab with the finding
  // off screen is the same as not arriving.
  useEffect(() => {
    if (!focus) return;
    const row = body.current?.querySelector('[data-focus="1"]');
    row?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [focus, focusNonce, derived]);

  const toggle = (s: Severity): void =>
    setMuted((was) => {
      const next = new Set(was);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });

  const grouped = useMemo(() => {
    const out: Record<Severity, UiIssue[]> = {
      error: [],
      warning: [],
      info: [],
    };
    for (const i of derived.issues) out[i.severity].push(i);
    return out;
  }, [derived.issues]);

  const shown = SEVERITIES.filter((s) => !muted.has(s));
  const nothingShown = shown.every((s) => grouped[s].length === 0);

  const focused = focus ? derived.issues.filter((i) => i.path === focus) : [];

  return (
    <div className="panel__body" ref={body}>
      {focus && (
        <div className="issues__focus">
          <span className="mono">{focus}</span>
          <span className="dim">
            {focused.length} finding{focused.length === 1 ? "" : "s"}
          </span>
          <button
            className="badge badge--quiet"
            onClick={() => useEditor.getState().clearFindingFocus()}
            title="Stop highlighting this block"
          >
            clear
          </button>
        </div>
      )}
      <div className="issues__summary">
        {SEVERITIES.map((s) => {
          const n = grouped[s].length;
          const off = muted.has(s);
          return (
            <button
              key={s}
              className={`badge badge--${n ? s : "quiet"}${off ? " is-muted" : ""}`}
              onClick={() => toggle(s)}
              title={off ? `Show ${s}s` : `Hide ${s}s`}
            >
              {n} {s}
              {n === 1 ? "" : "s"}
            </button>
          );
        })}
      </div>

      {nothingShown && (
        <div className="empty">
          {derived.ok
            ? "Nothing to report. Every port connects, every edge unifies, and every rule is satisfied."
            : "Nothing at the severities you are showing."}
        </div>
      )}

      {shown.map((s) => (
        <Group key={s} severity={s} issues={grouped[s]} focus={focus} />
      ))}

      <Section id="rulebook" title="Rule book" defaultOpen={false} note={`${RULES.length} rules`}>
        <div className="rulebook">
          {RULES.map((r) => {
            const fired = derived.issues.filter((i) => i.rule === r.id).length;
            return (
              <div className="rulebook__row" key={r.id}>
                <div className="rulebook__head">
                  <span className="mono">{r.id}</span>
                  <span className={`badge badge--tiny${fired ? " badge--warning" : ""}`}>
                    {fired === 0 ? "clear" : fired}
                  </span>
                </div>
                <div className="rulebook__title">{r.title}</div>
                <div className="rulebook__desc">{r.description}</div>
              </div>
            );
          })}
        </div>
      </Section>
    </div>
  );
}
