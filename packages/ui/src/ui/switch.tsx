/** Switch and checkbox: square, because everything here is. */

import { Switch as Base } from "@base-ui-components/react/switch";
import { Checkbox as BaseCheckbox } from "@base-ui-components/react/checkbox";
import { Check } from "lucide-react";
import { cn } from "../lib/utils.js";

export function Switch({
  checked,
  onCheckedChange,
  className,
  ...props
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  className?: string;
  title?: string;
  disabled?: boolean;
}): React.ReactElement {
  return (
    <Base.Root
      checked={checked}
      onCheckedChange={onCheckedChange}
      {...props}
      className={cn(
        // inline-flex, not the browser's default inline: a switch is a box with
        // a thumb inside it, and inline layout ignores the height it is given.
        "relative inline-flex h-4 w-7 shrink-0 items-center border border-border bg-input p-px",
        "outline-none transition-colors",
        "data-[checked]:border-primary data-[checked]:bg-primary",
        "focus-visible:ring-2 focus-visible:ring-ring/40",
        className,
      )}
    >
      <Base.Thumb
        className={cn(
          "block h-3 w-3 shrink-0 bg-dim transition-transform",
          "data-[checked]:translate-x-3 data-[checked]:bg-primary-foreground",
        )}
      />
    </Base.Root>
  );
}

export function Checkbox({
  checked,
  onCheckedChange,
  className,
  ...props
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  className?: string;
  title?: string;
  disabled?: boolean;
}): React.ReactElement {
  return (
    <BaseCheckbox.Root
      checked={checked}
      onCheckedChange={onCheckedChange}
      {...props}
      className={cn(
        "inline-flex size-3.5 shrink-0 items-center justify-center border border-border bg-input",
        "outline-none transition-colors data-[checked]:border-primary data-[checked]:bg-primary",
        "focus-visible:ring-2 focus-visible:ring-ring/40",
        className,
      )}
    >
      <BaseCheckbox.Indicator className="text-primary-foreground">
        <Check className="size-2.5" strokeWidth={3} />
      </BaseCheckbox.Indicator>
    </BaseCheckbox.Root>
  );
}
