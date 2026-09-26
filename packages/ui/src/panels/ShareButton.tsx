/**
 * Share, in the toolbar where it is looked for.
 *
 * One press makes a link and copies it, and the dialog says which kind of link
 * it is, because the two behave differently in the hands of whoever opens it:
 * a store's link shows the saved design, and a link that carries the design
 * gives them a copy of their own.
 */

import { useState } from "react";
import { Check, Copy, Link2 } from "lucide-react";
import { Button } from "../ui/button.js";
import { Input } from "../ui/input.js";
import { Dialog, DialogContent } from "../ui/dialog.js";
import { Tooltip } from "../ui/tooltip.js";
import { useStorage } from "../state/hooks.js";
import { shareDesign, sharesThroughStore, type Shared } from "../state/share.js";

export default function ShareButton(): React.ReactElement {
  const provider = useStorage();
  const [open, setOpen] = useState(false);
  const [shared, setShared] = useState<Shared | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const copy = async (url: string): Promise<void> => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
    } catch {
      // A clipboard the browser will not hand over: the link is on screen to
      // copy by hand, which is not a failed share.
      setCopied(false);
    }
  };

  const start = async (): Promise<void> => {
    setOpen(true);
    setShared(null);
    setProblem(null);
    setCopied(false);
    try {
      const got = await shareDesign();
      setShared(got);
      await copy(got.url);
    } catch (e) {
      setProblem((e as Error).message);
    }
  };

  const signIn = provider?.signIn && !provider.account?.() ? provider.signIn.bind(provider) : null;

  return (
    <>
      <Tooltip
        content={
          sharesThroughStore()
            ? "Save this design and copy a link anyone can open"
            : "Copy a link that carries this design"
        }
      >
        <Button variant="default" size="md" onClick={() => void start()} data-testid="share">
          <Link2 />
          Share
        </Button>
      </Tooltip>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent title="Share this design" width="30rem">
          {problem ? (
            <p className="text-xs text-error">Could not make a link: {problem}</p>
          ) : !shared ? (
            <p className="text-xs text-text-dim">Making a link…</p>
          ) : (
            <div className="flex flex-col gap-3">
              <div className="flex gap-2">
                <Input
                  readOnly
                  value={shared.url}
                  aria-label="Share link"
                  data-testid="share-link"
                  className="min-w-0 flex-1 font-mono text-[11px]"
                  onFocus={(e) => e.currentTarget.select()}
                />
                <Button onClick={() => void copy(shared.url)} aria-label="Copy the link">
                  {copied ? <Check /> : <Copy />}
                  {copied ? "Copied" : "Copy"}
                </Button>
              </div>
              <p className="text-xs text-text-dim">
                {shared.kind === "stored"
                  ? "Saved to your designs. Anyone with the link can open it to look; it stays yours to change."
                  : "The design travels in the link itself. Nothing is uploaded, and whoever opens it gets a copy of their own to change."}
              </p>
              {signIn && (
                <p className="text-xs text-text-dim">
                  <button type="button" className="text-primary hover:underline" onClick={signIn}>
                    Sign in
                  </button>{" "}
                  to keep your designs and share short links to them.
                </p>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
