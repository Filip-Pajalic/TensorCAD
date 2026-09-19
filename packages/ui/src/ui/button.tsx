/**
 * Button.
 *
 * shadcn's API on Base UI's `useRender`, with every corner square. The sizes
 * are tighter than shadcn's defaults because this is a dense tool: a toolbar
 * row has to fit beside a drawing, not float in a marketing page.
 */

import { cva, type VariantProps } from "class-variance-authority";
import { useRender } from "@base-ui-components/react/use-render";
import { cn } from "../lib/utils.js";

export const buttonVariants = cva(
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap font-sans text-xs " +
    "transition-colors select-none outline-none " +
    "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0 " +
    "disabled:pointer-events-none disabled:opacity-40 " +
    "[&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg]:size-3.5",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        outline:
          "border border-border bg-elev text-foreground hover:bg-accent hover:text-accent-foreground",
        ghost: "text-dim hover:bg-elev hover:text-foreground",
        destructive: "border border-error bg-error-soft text-error hover:bg-error hover:text-white",
        // A pressed view-state toggle, the way a CAD toolbar shows one.
        on: "border border-primary bg-accent text-accent-foreground",
      },
      size: {
        sm: "h-6 px-2",
        md: "h-7 px-2.5",
        lg: "h-8 px-3 text-sm",
        icon: "h-7 w-7 px-0",
        "icon-sm": "h-6 w-6 px-0",
        // A segment of a joined control: no height of its own, so it fills the
        // box it sits in. A fixed height here is what used to leave a segmented
        // group two or three pixels short of the controls beside it.
        seg: "h-full self-stretch px-0",
      },
    },
    defaultVariants: { variant: "outline", size: "md" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  /** Render as something else, keeping the styling. */
  render?: useRender.RenderProp;
}

export function Button({
  className,
  variant,
  size,
  render,
  ...props
}: ButtonProps): React.ReactElement {
  return useRender({
    render: render ?? <button type="button" />,
    props: { className: cn(buttonVariants({ variant, size, className })), ...props },
  });
}
