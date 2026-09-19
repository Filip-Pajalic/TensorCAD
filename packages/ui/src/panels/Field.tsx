/**
 * Small controlled inputs that only commit on blur or Enter, so typing an
 * expression such as `ceil_mult(1.3*8/3*D, 1024)` does not re-analyse the whole
 * document on every keystroke.
 */

import { useEffect, useState } from "react";

export function TextField({
  value,
  onCommit,
  placeholder,
  title,
  mono = false,
  disabled = false,
  invalid = false,
}: {
  value: string;
  onCommit: (next: string) => void;
  placeholder?: string;
  title?: string;
  mono?: boolean;
  disabled?: boolean;
  invalid?: boolean;
}): React.ReactElement {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);

  const commit = (): void => {
    if (draft !== value) onCommit(draft);
  };

  return (
    <input
      className={`field${mono ? " mono" : ""}${invalid ? " field--invalid" : ""}`}
      value={draft}
      placeholder={placeholder}
      title={title}
      disabled={disabled}
      spellCheck={false}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          commit();
          (e.target as HTMLInputElement).blur();
        } else if (e.key === "Escape") {
          setDraft(value);
          (e.target as HTMLInputElement).blur();
        }
        e.stopPropagation();
      }}
    />
  );
}

export function TextArea({
  value,
  onCommit,
  rows = 3,
  disabled = false,
  invalid = false,
}: {
  value: string;
  onCommit: (next: string) => void;
  rows?: number;
  disabled?: boolean;
  invalid?: boolean;
}): React.ReactElement {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  return (
    <textarea
      className={`field field--area mono${invalid ? " field--invalid" : ""}`}
      rows={rows}
      value={draft}
      disabled={disabled}
      spellCheck={false}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => draft !== value && onCommit(draft)}
      onKeyDown={(e) => e.stopPropagation()}
    />
  );
}
