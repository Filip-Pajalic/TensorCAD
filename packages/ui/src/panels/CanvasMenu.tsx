/**
 * The right-click menu.
 *
 * A drawing application owns its right button. Without this the browser's own
 * menu appears — Back, Refresh, Save as, Inspect — which is useless here and a
 * reminder that you are looking at a web page.
 *
 * Built on `Menu` with a virtual anchor at the pointer rather than on Base UI's
 * `ContextMenu`, whose trigger never receives the event on this canvas. The
 * contents come from the shared tree, so this and the application menu cannot
 * drift apart.
 */

import { useCallback, useEffect, useState } from "react";
import { Menu } from "@base-ui-components/react/menu";
import { useEditor } from "../state/store.js";
import { BLOCK_ACTIONS, CANVAS_MENU, POPUP, renderMenu } from "./menu-tree.js";

interface At {
  x: number;
  y: number;
  /** Whether the click landed on a part, which decides what is offered. */
  onBlock: boolean;
}

export default function CanvasMenu({
  children,
}: {
  children: React.ReactNode;
}): React.ReactElement {
  const selection = useEditor((s) => s.selection);
  const [at, setAt] = useState<At | null>(null);
  const close = useCallback(() => setAt(null), []);

  useEffect(() => {
    const onMenu = (e: MouseEvent): void => {
      const target = e.target as HTMLElement | null;
      // Text fields keep the browser's, where cut, copy and paste actually live.
      if (target && /^(INPUT|TEXTAREA)$/.test(target.tagName)) return;
      e.preventDefault();
      // Only the sheet gets a menu; elsewhere the browser's one is simply gone,
      // which is most of the point.
      if (!target?.closest(".canvas, .canvas__flow, canvas")) return;

      // Right-clicking a block selects it, as it does everywhere else: a menu
      // offering "Delete" for whatever was selected a minute ago is worse than
      // no menu.
      const node = target.closest<HTMLElement>(".react-flow__node");
      const path = node?.getAttribute("data-id") ?? null;
      const onBlock = Boolean(target.closest(".part, .glyph, .frame"));
      if (onBlock && path) useEditor.getState().select(path);
      setAt({ x: e.clientX, y: e.clientY, onBlock: onBlock && Boolean(path) });
    };
    document.addEventListener("contextmenu", onMenu);
    return () => document.removeEventListener("contextmenu", onMenu);
  }, []);

  const nodes = at?.onBlock && selection ? [...BLOCK_ACTIONS, ...CANVAS_MENU] : CANVAS_MENU;

  return (
    <>
      {children}
      <Menu.Root open={at !== null} onOpenChange={(open) => !open && close()} modal={false}>
        <Menu.Portal>
          <Menu.Positioner
            className="z-50"
            side="right"
            align="start"
            sideOffset={2}
            anchor={{
              getBoundingClientRect: () =>
                DOMRect.fromRect({ x: at?.x ?? 0, y: at?.y ?? 0, width: 0, height: 0 }),
            }}
          >
            <Menu.Popup className={POPUP}>{renderMenu(nodes, close)}</Menu.Popup>
          </Menu.Positioner>
        </Menu.Portal>
      </Menu.Root>
    </>
  );
}
