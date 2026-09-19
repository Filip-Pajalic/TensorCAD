/**
 * A collapsible section with a persistent open state.
 *
 * The readout column is long: parameters, compute, memory, serving, training.
 * Tabbing them would hide the comparison that makes them worth reading, so they
 * stack and each one folds away. Which ones are folded is remembered, because a
 * reader who cares about memory today cares about it tomorrow.
 */

import { useCallback, useState } from "react";

const STORAGE_PREFIX = "tensorcad.section.";

function stored(id: string, fallback: boolean): boolean {
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + id);
    return raw === null ? fallback : raw === "1";
  } catch {
    return fallback;
  }
}

export default function Section({
  id,
  title,
  note,
  defaultOpen = true,
  children,
}: {
  /** Stable key for remembering the open state. */
  id: string;
  title: string;
  /** A number or word shown on the header, readable while folded. */
  note?: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}): React.ReactElement {
  const [open, setOpen] = useState(() => stored(id, defaultOpen));

  const toggle = useCallback(() => {
    setOpen((was) => {
      const next = !was;
      try {
        localStorage.setItem(STORAGE_PREFIX + id, next ? "1" : "0");
      } catch {
        // Not remembering it is harmless.
      }
      return next;
    });
  }, [id]);

  return (
    <section className={`fold${open ? " is-open" : ""}`}>
      <button className="fold__head" onClick={toggle} aria-expanded={open}>
        <span className="fold__caret" aria-hidden>
          {open ? "▾" : "▸"}
        </span>
        <span className="fold__title">{title}</span>
        {note !== undefined && <span className="fold__note mono">{note}</span>}
      </button>
      {open && <div className="fold__body">{children}</div>}
    </section>
  );
}
