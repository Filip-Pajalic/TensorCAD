/**
 * The application menu and the right-click menu.
 *
 * Base UI's Menu, sized like a desktop menu rather than a web one: small text,
 * tight rows, and a shortcut column on the right, because a menu is also where
 * people learn the keys.
 */

import { Menu as Base } from "@base-ui-components/react/menu";
import { Check, ChevronRight } from "lucide-react";
import { cn } from "../lib/utils.js";

export const Menu = Base.Root;
export const MenuTrigger = Base.Trigger;
export const MenuSubmenu = Base.SubmenuRoot;

const POPUP =
  "z-50 min-w-[13rem] border border-border bg-popover py-1 text-xs shadow-xl outline-none";
const ITEM =
  "flex cursor-default items-center gap-2 px-2.5 py-1 outline-none select-none " +
  "data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground " +
  "data-[disabled]:pointer-events-none data-[disabled]:opacity-40";

export function MenuContent({
  children,
  align = "start",
  className,
}: {
  children: React.ReactNode;
  align?: "start" | "center" | "end";
  className?: string;
}): React.ReactElement {
  return (
    <Base.Portal>
      <Base.Positioner align={align} sideOffset={3} className="z-50">
        <Base.Popup className={cn(POPUP, className)}>{children}</Base.Popup>
      </Base.Positioner>
    </Base.Portal>
  );
}

export function MenuItem({
  children,
  shortcut,
  onClick,
  disabled,
  className,
}: {
  children: React.ReactNode;
  shortcut?: string;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
}): React.ReactElement {
  return (
    <Base.Item className={cn(ITEM, className)} onClick={onClick} disabled={disabled}>
      <span className="flex-1 truncate">{children}</span>
      {shortcut && <span className="font-mono text-[9.5px] text-dim">{shortcut}</span>}
    </Base.Item>
  );
}

export function MenuCheckItem({
  children,
  checked,
  onCheckedChange,
  shortcut,
}: {
  children: React.ReactNode;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  shortcut?: string;
}): React.ReactElement {
  return (
    <Base.CheckboxItem
      className={cn(ITEM, "pl-1.5")}
      checked={checked}
      onCheckedChange={onCheckedChange}
      closeOnClick={false}
    >
      <span className="flex size-3.5 shrink-0 items-center justify-center text-primary">
        {checked && <Check className="size-3" strokeWidth={3} />}
      </span>
      <span className="flex-1 truncate">{children}</span>
      {shortcut && <span className="font-mono text-[9.5px] text-dim">{shortcut}</span>}
    </Base.CheckboxItem>
  );
}

export function MenuRadioGroup({
  value,
  onValueChange,
  children,
}: {
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}): React.ReactElement {
  return (
    <Base.RadioGroup value={value} onValueChange={(v) => onValueChange(String(v))}>
      {children}
    </Base.RadioGroup>
  );
}

export function MenuRadioItem({
  value,
  children,
}: {
  value: string;
  children: React.ReactNode;
}): React.ReactElement {
  return (
    <Base.RadioItem className={cn(ITEM, "pl-1.5")} value={value}>
      <span className="flex size-3.5 shrink-0 items-center justify-center text-primary">
        <Base.RadioItemIndicator>
          <Check className="size-3" strokeWidth={3} />
        </Base.RadioItemIndicator>
      </span>
      <span className="flex-1 truncate">{children}</span>
    </Base.RadioItem>
  );
}

export function MenuSeparator(): React.ReactElement {
  return <Base.Separator className="my-1 h-px bg-border" />;
}

/**
 * A section caption.
 *
 * Deliberately not `Menu.GroupLabel`: that part requires a surrounding
 * `Menu.Group`, and these captions head a run of items rather than a labelled
 * group, so it would only add a wrapper to satisfy a context check.
 */
export function MenuLabel({ children }: { children: React.ReactNode }): React.ReactElement {
  return (
    <div className="px-2.5 py-1 text-[9.5px] font-semibold tracking-[0.08em] uppercase text-dim">
      {children}
    </div>
  );
}

export function MenuSubTrigger({ children }: { children: React.ReactNode }): React.ReactElement {
  return (
    <Base.SubmenuTrigger className={ITEM}>
      <span className="flex-1 truncate">{children}</span>
      <ChevronRight className="size-3 text-dim" />
    </Base.SubmenuTrigger>
  );
}

export function MenuSubContent({ children }: { children: React.ReactNode }): React.ReactElement {
  return (
    <Base.Portal>
      <Base.Positioner sideOffset={-2} align="start" className="z-50">
        <Base.Popup className={POPUP}>{children}</Base.Popup>
      </Base.Positioner>
    </Base.Portal>
  );
}
