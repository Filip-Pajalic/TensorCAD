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
import type { RuleSeverity } from "@tensorcad/engine";

const SEVERITIES: Severity[] = ["error", "warning", "info"];

const TITLE_BY_RULE: Record<string, string> = Object.fromEntries(RULES.map((r) => [r.id, r.title]));

/** What this design has decided one rule means to it. */
function RuleChoice({
  rule,
  value,
}: {
  rule: string;
  value: RuleSeverity | undefined;
}): React.ReactElement {
  return (
    <select
      className={`field field--tiny${value ? " field--set" : ""}`}
      value={value ?? ""}
      title={
        value
          ? `This design treats ${rule} as ${value}`
          : `This design takes ${rule} as it comes`
      }
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => {
        e.stopPropagation();
        useEditor
          .getState()
          .setRuleSeverity(rule, e.target.value === "" ? undefined : (e.target.value as RuleSeverity));
      }}
    >
      <option value="">as it comes</option>
      <option value="error">error</option>
      <option value="warning">warning</option>
      <option value="info">info</option>
      <option value="off">off</option>
    </select>
  );
}

function Group({
  severity,
  issues,
  focus,
  severities,
}: {
  severity: Severity;
  issues: UiIssue[];
  /** The block whose marker was pressed on the drawing, if any. */
  focus: string | null;
  /** What the document has already decided, keyed by rule. */
  severities: Record<string, RuleSeverity>;
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
              {/*
                The decision belongs where the finding is read. The rule book
                carries the same control, but only for the eighteen design
                rules; a block's own constraint — SDPA-03, ATTN-01 — is a
                finding with a rule id and no row in that book, and those are
                the ones most worth being able to accept.
              */}
              <RuleChoice rule={i.rule} value={severities[i.rule]} />
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
  const severities: Record<string, RuleSeverity> = useEditor((s) => s.doc.rules) ?? {};
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
  const decided = Object.keys(severities).length;
  const dropped = derived.overridden.filter((o) => o.to === "off").length;
  const unreadable = derived.overridden.filter((o) => o.to.startsWith("?"));

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
      {derived.overridden.length > 0 && (
        <div className="issues__overrides">
          {dropped > 0 && (
            <div>
              {dropped} finding{dropped === 1 ? "" : "s"} dropped because this design says so.
            </div>
          )}
          {derived.overridden.length > dropped && (
            <div>
              {derived.overridden.length - dropped} finding
              {derived.overridden.length - dropped === 1 ? "" : "s"} shown at a severity this
              design chose.
            </div>
          )}
          {unreadable.map((o) => (
            <div className="issues__bad" key={o.rule + o.path}>
              {o.rule} is set to {o.to.slice(1)}, which is not a severity; the rule was left as it
              is.
            </div>
          ))}
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
          {/* "Nothing to report" has to mean nothing was found, not nothing is
              being shown. Notes are muted by default, so judging this by
              `derived.ok` — which only asks about errors — told a design with a
              note that every rule was satisfied. */}
          {derived.issues.length === 0
            ? "Nothing to report. Every port connects, every edge unifies, and every rule is satisfied."
            : `Nothing at the severities you are showing. ${derived.issues.length} finding${
                derived.issues.length === 1 ? " is" : "s are"
              } hidden.`}
        </div>
      )}

      {shown.map((s) => (
        <Group key={s} severity={s} issues={grouped[s]} focus={focus} severities={severities} />
      ))}

      <Section
        id="rulebook"
        title="Rule book"
        defaultOpen={false}
        note={
          decided === 0 ? `${RULES.length} rules` : `${RULES.length} rules, ${decided} decided`
        }
      >
        <div className="rulebook">
          {RULES.map((r) => {
            const fired = derived.issues.filter((i) => i.rule === r.id).length;
            const decision = severities[r.id];
            return (
              <div className="rulebook__row" key={r.id}>
                <div className="rulebook__head">
                  <span className="mono">{r.id}</span>
                  <span className={`badge badge--tiny${fired ? " badge--warning" : ""}`}>
                    {fired === 0 ? "clear" : fired}
                  </span>
                  {/*
                    What this design has decided the rule means to it. A rule
                    that is right in general is sometimes wrong here, and the
                    alternative to recording that is reading past a warning
                    until the warnings stop meaning anything.
                  */}
                  <RuleChoice rule={r.id} value={decision} />
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
