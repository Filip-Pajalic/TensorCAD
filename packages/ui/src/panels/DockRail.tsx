/**
 * A collapsed dock.
 *
 * Fusion collapses its browser to a narrow strip with a double arrow, and the
 * strip still says what is behind it. That is the part worth copying: a dock
 * that vanishes entirely leaves you hunting through a menu to get it back,
 * while a rail keeps the affordance on screen and costs twenty pixels.
 *
 * The whole rail is the button, because a twenty-pixel-wide chevron is a poor
 * target and there is nothing else in there to hit by mistake.
 */

import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "../lib/utils.js";

export default function DockRail({
  side,
  label,
  note,
  onExpand,
}: {
  side: "left" | "right";
  label: string;
  /** A number or word kept visible while collapsed, e.g. the parameter count. */
  note?: string;
  onExpand: () => void;
}): React.ReactElement {
  const Chevron = side === "left" ? ChevronRight : ChevronLeft;
  return (
    <button
      type="button"
      onClick={onExpand}
      title={`Show ${label.toLowerCase()}`}
      aria-label={`Show ${label.toLowerCase()}`}
      className={cn(
        "group flex w-full flex-col items-center gap-3 bg-panel py-2 text-dim",
        "transition-colors outline-none hover:bg-elev hover:text-foreground",
        "focus-visible:ring-2 focus-visible:ring-ring/40",
        side === "left" ? "border-r border-border" : "border-l border-border",
      )}
    >
      <Chevron className="size-3.5 shrink-0" />
      <span
        className="text-[9.5px] font-semibold tracking-[0.18em] uppercase select-none"
        style={{ writingMode: "vertical-rl", textOrientation: "mixed" }}
      >
        {label}
      </span>
      {note && (
        <span
          className="font-mono text-[10px] text-text-dim select-none"
          style={{ writingMode: "vertical-rl", textOrientation: "mixed" }}
        >
          {note}
        </span>
      )}
    </button>
  );
}
