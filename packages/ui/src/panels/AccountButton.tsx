/**
 * The account, in the editor's own toolbar.
 *
 * A deployment that keeps designs used to put a bar of its own above the
 * editor to say who was signed in, which gave the page two top bars and the
 * product's name twice. The corner of this toolbar is where an account goes,
 * so it goes here, from whatever the registered provider reports. A plain
 * checkout has no provider, and there is nothing in the corner.
 */

import { Menu } from "@base-ui-components/react/menu";
import { ChevronDown, TriangleAlert } from "lucide-react";
import { Button } from "../ui/button.js";
import { useStorage } from "../state/hooks.js";
import { useEditor } from "../state/store.js";
import { ITEM, POPUP } from "./menu-tree.js";

function Initial({ name, avatarUrl }: { name: string; avatarUrl?: string }): React.ReactElement {
  return avatarUrl ? (
    <img src={avatarUrl} alt="" className="size-5 object-cover" referrerPolicy="no-referrer" />
  ) : (
    <span className="flex size-5 items-center justify-center bg-accent text-[11px] font-semibold text-accent-foreground">
      {name.slice(0, 1).toUpperCase()}
    </span>
  );
}

export default function AccountButton(): React.ReactElement | null {
  const provider = useStorage();
  if (!provider?.account) return null;
  const account = provider.account();

  if (!account) {
    return provider.signIn ? (
      <Button variant="outline" size="md" onClick={() => provider.signIn?.()} data-testid="sign-in">
        Sign in
      </Button>
    ) : null;
  }

  return (
    <Menu.Root>
      <Menu.Trigger
        className={
          "flex h-7 max-w-[14rem] shrink-0 items-center gap-1.5 border border-border bg-elev pr-1.5 pl-1 text-xs " +
          "outline-none hover:bg-accent hover:text-accent-foreground focus-visible:ring-2 focus-visible:ring-ring " +
          "data-[popup-open]:bg-accent"
        }
        aria-label={`Account: ${account.name}`}
        data-testid="account"
      >
        <Initial name={account.name} avatarUrl={account.avatarUrl} />
        <span className="truncate">{account.name}</span>
        {account.problem && <TriangleAlert className="size-3.5 shrink-0 text-warn" aria-label="Something needs attention" />}
        <ChevronDown className="size-3.5 shrink-0 text-dim" />
      </Menu.Trigger>
      <Menu.Portal>
        <Menu.Positioner align="end" sideOffset={4} className="z-50">
          <Menu.Popup className={POPUP}>
            <div className="px-2.5 py-1.5">
              <div className="font-medium text-foreground">{account.name}</div>
              {account.email && <div className="text-dim">{account.email}</div>}
            </div>
            {account.problem && <div className="max-w-[18rem] px-2.5 pb-1.5 text-warn">{account.problem}</div>}
            <Menu.Separator className="my-1 h-px bg-border" />
            <Menu.Item className={ITEM} onClick={() => useEditor.getState().setRightTab("designs")}>
              Your designs
            </Menu.Item>
            {provider.signOut ? (
              <Menu.Item className={ITEM} onClick={() => void provider.signOut?.()}>
                Sign out
              </Menu.Item>
            ) : account.signOutUrl ? (
              <Menu.Item className={ITEM} render={<a href={account.signOutUrl} />}>
                Sign out
              </Menu.Item>
            ) : null}
          </Menu.Popup>
        </Menu.Positioner>
      </Menu.Portal>
    </Menu.Root>
  );
}
