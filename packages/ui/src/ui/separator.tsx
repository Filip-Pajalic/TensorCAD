import { Separator as Base } from "@base-ui-components/react/separator";
import { cn } from "../lib/utils.js";

export function Separator({
  className,
  orientation = "horizontal",
  ...props
}: React.ComponentProps<typeof Base>): React.ReactElement {
  return (
    <Base
      {...props}
      orientation={orientation}
      className={cn(
        "shrink-0 bg-border",
        orientation === "horizontal" ? "h-px w-full" : "h-full w-px",
        className,
      )}
    />
  );
}
