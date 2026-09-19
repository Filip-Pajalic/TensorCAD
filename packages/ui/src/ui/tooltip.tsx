/** Tooltip. One provider at the root, one component everywhere else. */

import { Tooltip as Base } from "@base-ui-components/react/tooltip";
import { cn } from "../lib/utils.js";

export const TooltipProvider = Base.Provider;

export function Tooltip({
  content,
  children,
  side = "bottom",
  className,
}: {
  content: React.ReactNode;
  children: React.ReactElement<Record<string, unknown>>;
  side?: "top" | "bottom" | "left" | "right";
  className?: string;
}): React.ReactElement {
  if (!content) return children;
  return (
    <Base.Root>
      <Base.Trigger render={children} />
      <Base.Portal>
        <Base.Positioner side={side} sideOffset={6} className="z-50">
          <Base.Popup
            className={cn(
              "max-w-[22rem] border border-border bg-popover px-2 py-1.5",
              "font-sans text-[11px] leading-snug text-foreground shadow-lg",
              className,
            )}
          >
            {content}
          </Base.Popup>
        </Base.Positioner>
      </Base.Portal>
    </Base.Root>
  );
}

/** A keyboard key, for shortcut hints inside a tooltip or a menu. */
export function Kbd({ children }: { children: React.ReactNode }): React.ReactElement {
  return (
    <kbd className="ml-1 border border-border bg-elev px-1 py-px font-mono text-[9.5px] text-dim">
      {children}
    </kbd>
  );
}
