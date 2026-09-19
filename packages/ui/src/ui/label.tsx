/** A field caption: small, uppercase, quiet. The convention on every CAD form. */

import { cn } from "../lib/utils.js";

export function Label({
  className,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>): React.ReactElement {
  return (
    <label
      {...props}
      className={cn(
        "text-[9.5px] font-medium uppercase tracking-[0.07em] text-dim select-none",
        className,
      )}
    />
  );
}
