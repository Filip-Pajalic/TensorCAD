/**
 * Clarify selection: what is under the cursor, when several things are.
 *
 * KiCad's convention, and it exists because on a dense sheet a click is
 * ambiguous far more often than a tool admits. A wire crossing a block, two
 * wires meeting, a part inside a frame: clicking picks whatever happens to be
 * on top, and the only way to reach the other thing is to move the drawing.
 *
 * Reached with Alt held, which is KiCad's modifier for it. Deliberately not a
 * long press: a long press on a block is how a touch device starts a drag, and
 * a gesture that means two things on two devices means neither.
 */

import { useEffect, useRef } from "react";
import { categoryColor } from "./blocks.js";

export interface Candidate {
  /** Node path, or the net key for a wire. */
  id: string;
  kind: "block" | "wire";
  label: string;
  sub: string;
  category?: string;
}

export default function Clarify({
  x,
  y,
  items,
  onPick,
  onClose,
}: {
  x: number;
  y: number;
  items: Candidate[];
  onPick: (item: Candidate) => void;
  onClose: () => void;
}): React.ReactElement {
  const host = useRef<HTMLDivElement>(null);

  // Anywhere else, and Escape, put it away. A chooser that stays up after the
  // choice has been made somewhere else is a chooser in the way.
  useEffect(() => {
    const away = (e: PointerEvent): void => {
      if (!host.current?.contains(e.target as Node)) onClose();
    };
    const key = (e: KeyboardEvent): void => {
      if (e.key === "Escape") onClose();
    };
    // Capture, so it closes before the click lands on the canvas underneath.
    window.addEventListener("pointerdown", away, true);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("pointerdown", away, true);
      window.removeEventListener("keydown", key);
    };
  }, [onClose]);

  return (
    <div className="clarify" style={{ left: x, top: y }} ref={host} role="menu">
      <div className="clarify__head">{items.length} things here</div>
      {items.map((item) => (
        <button
          type="button"
          role="menuitem"
          className="clarify__item"
          key={`${item.kind}:${item.id}`}
          onClick={() => onPick(item)}
        >
          <span
            className={`clarify__swatch clarify__swatch--${item.kind}`}
            style={{ background: item.kind === "block" ? categoryColor(item.category) : undefined }}
          />
          <span className="clarify__label">{item.label}</span>
          <span className="clarify__sub">{item.sub}</span>
        </button>
      ))}
    </div>
  );
}
