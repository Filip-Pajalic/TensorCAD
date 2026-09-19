/** Tabs. A square underline, not a pill. */

import { Tabs as Base } from "@base-ui-components/react/tabs";
import { cn } from "../lib/utils.js";

export const Tabs = Base.Root;

export function TabsList({
  className,
  ...props
}: React.ComponentProps<typeof Base.List>): React.ReactElement {
  return (
    <Base.List
      {...props}
      className={cn("flex shrink-0 border-b border-border bg-input", className)}
    />
  );
}

export function TabsTrigger({
  className,
  ...props
}: React.ComponentProps<typeof Base.Tab>): React.ReactElement {
  return (
    <Base.Tab
      {...props}
      className={cn(
        "relative flex flex-1 items-center justify-center gap-1.5 border-b-2 border-transparent",
        "px-1 py-2 font-sans text-xs text-dim outline-none transition-colors select-none",
        "hover:text-foreground",
        "data-[selected]:border-primary data-[selected]:bg-elev data-[selected]:text-foreground",
        "focus-visible:ring-2 focus-visible:ring-ring/40",
        className,
      )}
    />
  );
}

export function TabsContent({
  className,
  ...props
}: React.ComponentProps<typeof Base.Panel>): React.ReactElement {
  return <Base.Panel {...props} className={cn("flex min-h-0 flex-1 flex-col outline-none", className)} />;
}
