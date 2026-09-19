/** Dialog, on Base UI. Square, bordered, with a thin title rule. */

import { Dialog as Base } from "@base-ui-components/react/dialog";
import { X } from "lucide-react";
import { cn } from "../lib/utils.js";

export const Dialog = Base.Root;
export const DialogTrigger = Base.Trigger;
export const DialogClose = Base.Close;

export function DialogContent({
  className,
  title,
  description,
  children,
  width = "34rem",
}: {
  className?: string;
  title: string;
  description?: string;
  children: React.ReactNode;
  width?: string;
}): React.ReactElement {
  return (
    <Base.Portal>
      <Base.Backdrop className="fixed inset-0 z-40 bg-black/50" />
      <Base.Popup
        style={{ width }}
        className={cn(
          "fixed top-1/2 left-1/2 z-50 max-h-[85vh] max-w-[92vw] -translate-x-1/2 -translate-y-1/2",
          "flex flex-col border border-border bg-card text-foreground shadow-2xl outline-none",
          className,
        )}
      >
        <div className="flex shrink-0 items-start gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0 flex-1">
            <Base.Title className="text-[11px] font-semibold uppercase tracking-[0.09em] text-dim">
              {title}
            </Base.Title>
            {description && (
              <Base.Description className="mt-1 text-xs text-text-dim">
                {description}
              </Base.Description>
            )}
          </div>
          <Base.Close
            className="-mt-1 -mr-1 p-1 text-dim outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/40"
            aria-label="Close"
          >
            <X className="size-4" />
          </Base.Close>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">{children}</div>
      </Base.Popup>
    </Base.Portal>
  );
}

export function DialogFooter({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>): React.ReactElement {
  return (
    <div
      {...props}
      className={cn(
        "mt-4 flex items-center justify-end gap-2 border-t border-border pt-3",
        className,
      )}
    />
  );
}
