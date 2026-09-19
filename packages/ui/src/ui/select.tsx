/** Select, on Base UI, styled to match the rest of the instrument panel. */

import { Select as Base } from "@base-ui-components/react/select";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "../lib/utils.js";

export interface SelectOption<T extends string | number> {
  value: T;
  label: string;
  /** Second line in the list, for a device name or a long description. */
  hint?: string;
}

export function Select<T extends string | number>({
  value,
  options,
  onValueChange,
  className,
  disabled,
  title,
}: {
  value: T;
  options: SelectOption<T>[];
  onValueChange: (value: T) => void;
  className?: string;
  disabled?: boolean;
  title?: string;
}): React.ReactElement {
  return (
    <Base.Root
      value={value}
      onValueChange={(next) => onValueChange(next as T)}
      disabled={disabled}
      items={options}
    >
      <Base.Trigger
        title={title}
        className={cn(
          "flex h-7 w-full min-w-0 items-center justify-between gap-1.5 border border-border",
          "bg-input px-2 font-sans text-xs text-foreground outline-none transition-colors",
          "hover:border-dim focus-visible:border-primary focus-visible:ring-2 focus-visible:ring-ring/40",
          "data-[disabled]:opacity-50",
          className,
        )}
      >
        <Base.Value className="truncate text-left" />
        <Base.Icon className="shrink-0 text-dim">
          <ChevronDown className="size-3" />
        </Base.Icon>
      </Base.Trigger>
      <Base.Portal>
        <Base.Positioner sideOffset={2} className="z-50">
          <Base.Popup
            className={cn(
              "max-h-[min(24rem,var(--available-height))] min-w-[var(--anchor-width)] overflow-y-auto",
              "border border-border bg-popover py-1 text-xs shadow-lg outline-none",
            )}
          >
            {options.map((option) => (
              <Base.Item
                key={String(option.value)}
                value={option.value}
                className={cn(
                  "flex cursor-default items-start gap-2 px-2 py-1 outline-none select-none",
                  "data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground",
                )}
              >
                <Base.ItemIndicator className="mt-[3px] shrink-0 text-primary">
                  <Check className="size-3" />
                </Base.ItemIndicator>
                <div className="min-w-0 flex-1">
                  <Base.ItemText className="block truncate">{option.label}</Base.ItemText>
                  {option.hint && (
                    <span className="block truncate text-[10px] text-dim">{option.hint}</span>
                  )}
                </div>
              </Base.Item>
            ))}
          </Base.Popup>
        </Base.Positioner>
      </Base.Portal>
    </Base.Root>
  );
}
