/**
 * The menu, as a tree.
 *
 * A flat list of every command is a menu you scroll, and a menu you scroll is
 * one that closes while you are reaching for the bottom of it. Fusion's ribbon
 * groups its commands behind a dozen short dropdowns for the same reason.
 *
 * So the structure is declared once here and rendered by both the menu bar and
 * the right-click menu, which keeps them from drifting and means a new command
 * has exactly one place to be added.
 */

import { Menu } from "@base-ui-components/react/menu";
import { Check, ChevronRight } from "lucide-react";
import { cn } from "../lib/utils.js";
import { COMMAND_BY_ID, prettyShortcut, runCommand } from "../state/commands.js";

export type MenuNode =
  | { kind: "item"; id: string }
  | { kind: "sep" }
  | { kind: "label"; text: string }
  | { kind: "sub"; label: string; children: MenuNode[] };

export const POPUP =
  "z-50 max-h-[min(34rem,var(--available-height))] min-w-[13rem] overflow-y-auto " +
  "border border-border bg-popover py-1 text-xs shadow-xl outline-none";
export const ITEM =
  "flex cursor-default items-center gap-2 px-2.5 py-1 outline-none select-none " +
  "data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground " +
  "data-[disabled]:pointer-events-none data-[disabled]:opacity-40";

const item = (id: string): MenuNode => ({ kind: "item", id });
const sep = (): MenuNode => ({ kind: "sep" });
const sub = (label: string, children: MenuNode[]): MenuNode => ({ kind: "sub", label, children });

/** The file, edit, view, blocks and panels trees, shared by both menus. */
export const FILE_MENU: MenuNode[] = [
  item("file.new"),
  item("file.open"),
  item("file.save"),
  sep(),
  item("file.export"),
];

export const EDIT_MENU: MenuNode[] = [
  item("edit.undo"),
  item("edit.redo"),
  sep(),
  item("edit.duplicate"),
  item("edit.lock"),
  item("edit.delete"),
  sep(),
  item("edit.deselect"),
];

export const ZOOM_MENU: MenuNode[] = [
  item("view.fit"),
  item("view.zoomIn"),
  item("view.zoomOut"),
  item("view.zoomReset"),
];

export const SHOW_MENU: MenuNode[] = [
  item("view.callouts"),
  item("view.shapes"),
  item("view.palette"),
  sep(),
  item("view.theme"),
];

export const VIEW_MENU: MenuNode[] = [
  item("view.volume"),
  sep(),
  sub("Zoom", ZOOM_MENU),
  item("view.arrange"),
  sep(),
  item("view.detailIn"),
  item("view.detailOut"),
  sep(),
  item("view.compare"),
  item("view.mark-baseline"),
  sep(),
  sub("Show", SHOW_MENU),
];

export const BLOCKS_MENU: MenuNode[] = [
  item("blocks.fromLevel"),
  sep(),
  item("blocks.import"),
  item("blocks.export"),
];

export const PANELS_MENU: MenuNode[] = [
  item("panel.inspector"),
  item("panel.symbols"),
  item("panel.rules"),
  item("panel.cluster"),
  item("panel.ladder"),
  sep(),
  item("view.dock.left"),
  item("view.dock.right"),
  item("view.focus"),
];

export const HELP_MENU: MenuNode[] = [
  item("help.palette"),
  sep(),
  item("help.settings"),
  item("help.shortcuts"),
];

/**
 * The application menu, as titles across a bar rather than one button that
 * hides all of them. Six short dropdowns, none of which needs scrolling.
 *
 * The same trees still feed the right-click menu below, so a command is still
 * declared exactly once.
 */
export const MENUBAR: { label: string; children: MenuNode[] }[] = [
  { label: "File", children: FILE_MENU },
  { label: "Edit", children: EDIT_MENU },
  { label: "View", children: VIEW_MENU },
  { label: "Blocks", children: BLOCKS_MENU },
  { label: "Panels", children: PANELS_MENU },
  { label: "Help", children: HELP_MENU },
];

/** The sheet's right-click menu. Block actions are prepended when on one. */
export const CANVAS_MENU: MenuNode[] = [
  sub("View", VIEW_MENU),
  sub("Zoom", ZOOM_MENU),
  item("view.arrange"),
  sep(),
  sub("Blocks", BLOCKS_MENU),
  sub("Panels", PANELS_MENU),
  sep(),
  item("help.palette"),
  item("help.settings"),
];

export const BLOCK_ACTIONS: MenuNode[] = [
  item("edit.lock"),
  item("edit.duplicate"),
  item("edit.delete"),
  sep(),
];

function Item({ id, onDone }: { id: string; onDone: () => void }): React.ReactElement | null {
  const command = COMMAND_BY_ID.get(id);
  if (!command) return null;
  const checked = command.checked?.();
  return (
    <Menu.Item
      className={cn(ITEM, checked !== undefined && "pl-1.5")}
      disabled={command.enabled ? !command.enabled() : false}
      onClick={() => {
        runCommand(id);
        onDone();
      }}
    >
      {checked !== undefined && (
        <span className="flex size-3.5 shrink-0 items-center justify-center text-primary">
          {checked && <Check className="size-3" strokeWidth={3} />}
        </span>
      )}
      <span className="flex-1 truncate">{command.label}</span>
      {command.shortcut && (
        <span className="font-mono text-[9.5px] text-dim">{prettyShortcut(command.shortcut)}</span>
      )}
    </Menu.Item>
  );
}

/** Render a tree. Recursion handles submenus at any depth. */
export function renderMenu(nodes: MenuNode[], onDone: () => void, keyPrefix = ""): React.ReactNode {
  return nodes.map((node, i) => {
    const key = `${keyPrefix}${i}`;
    switch (node.kind) {
      case "sep":
        return <Menu.Separator key={key} className="my-1 h-px bg-border" />;
      case "label":
        return (
          <div
            key={key}
            className="px-2.5 py-1 text-[9.5px] font-semibold tracking-[0.08em] uppercase text-dim"
          >
            {node.text}
          </div>
        );
      case "sub":
        return (
          <Menu.SubmenuRoot key={key}>
            <Menu.SubmenuTrigger className={ITEM}>
              <span className="flex-1 truncate">{node.label}</span>
              <ChevronRight className="size-3 text-dim" />
            </Menu.SubmenuTrigger>
            <Menu.Portal>
              <Menu.Positioner className="z-50" align="start" sideOffset={-2}>
                <Menu.Popup className={POPUP}>
                  {renderMenu(node.children, onDone, `${key}.`)}
                </Menu.Popup>
              </Menu.Positioner>
            </Menu.Portal>
          </Menu.SubmenuRoot>
        );
      default:
        return <Item key={key} id={node.id} onDone={onDone} />;
    }
  });
}
