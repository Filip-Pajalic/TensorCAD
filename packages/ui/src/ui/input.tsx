/** Text and number fields. Square, dense, monospace where a number lives. */

import { cn } from "../lib/utils.js";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  /** Numbers get tabular figures and right alignment so columns line up. */
  numeric?: boolean;
  invalid?: boolean;
}

export function Input({ className, numeric, invalid, ...props }: InputProps): React.ReactElement {
  return (
    <input
      {...props}
      aria-invalid={invalid || undefined}
      className={cn(
        "h-7 w-full min-w-0 border border-border bg-input px-2 font-sans text-xs text-foreground",
        "outline-none transition-colors placeholder:text-dim",
        "focus-visible:border-primary focus-visible:ring-2 focus-visible:ring-ring/40",
        "disabled:opacity-50",
        numeric && "text-right font-mono tabular-nums",
        invalid && "border-error focus-visible:border-error focus-visible:ring-error/40",
        className,
      )}
    />
  );
}

export function Textarea({
  className,
  ...props
}: React.TextareaHTMLAttributes<HTMLTextAreaElement>): React.ReactElement {
  return (
    <textarea
      {...props}
      className={cn(
        "w-full border border-border bg-input px-2 py-1.5 font-mono text-xs text-foreground",
        "outline-none transition-colors placeholder:text-dim",
        "focus-visible:border-primary focus-visible:ring-2 focus-visible:ring-ring/40",
        className,
      )}
    />
  );
}
