/**
 * The command bar.
 *
 * One row, one height. A menu bar on the left carrying every command and the
 * key that runs it, the drawing's identity beside it, the view state after
 * that, and the live readout hard right.
 *
 * Two rules keep it from becoming a ribbon again. Every control on the row is
 * `h-7`, including the segmented groups and the readout cells, so the bar has a
 * single baseline rather than the four heights it used to have. And nothing is
 * captioned: a label under a pair of undo arrows teaches nobody anything, costs
 * the drawing a whole row of height, and leaves the captions floating at
 * whatever x their cluster happened to centre on. What a control does is said
 * by a tooltip, on demand.
 */

import { useRef, useState } from "react";
import { Box, Minus, Plus, Redo2, Squircle, Undo2 } from "lucide-react";
import { useEditor } from "../state/store.js";
import { useDerived } from "../state/hooks.js";
import { parseDoc } from "../state/serialize.js";
import { mergeLibrary } from "../state/blocks.js";
import { loadTrace } from "../three/trace.js";
import { COMMAND_BY_ID, prettyShortcut, runCommand } from "../state/commands.js";
import { Button } from "../ui/button.js";
import { Input } from "../ui/input.js";
import { Select } from "../ui/select.js";
import { Separator } from "../ui/separator.js";
import { Tooltip, Kbd } from "../ui/tooltip.js";
import { Menu } from "@base-ui-components/react/menu";
import { Menubar } from "@base-ui-components/react/menubar";
import { MENUBAR, POPUP, renderMenu } from "./menu-tree.js";
import { formatCount, formatFlops } from "@tensor-cad/engine";
import { PRESET_NAMES } from "../engine.js";
import ShareButton from "./ShareButton.js";
import AccountButton from "./AccountButton.js";

/** The key a command is bound to, ready to drop into a tooltip. */
function key(id: string): React.ReactNode {
  const shortcut = COMMAND_BY_ID.get(id)?.shortcut;
  return shortcut ? <Kbd>{prettyShortcut(shortcut)}</Kbd> : null;
}

/**
 * The menu bar.
 *
 * Titles across the bar rather than a hamburger. A hamburger is what you reach
 * for when the commands will not fit; six titles fit, and putting them where
 * they can be read is the difference between a tool you learn by using and one
 * you learn by opening a drawer. Base UI's Menubar is what makes it behave like
 * a desktop menu: once one is open, moving sideways opens the next.
 */
function AppMenuBar(): React.ReactElement {
  return (
    <Menubar className="flex items-stretch self-stretch">
      {MENUBAR.map((menu) => (
        <Menu.Root key={menu.label}>
          <Menu.Trigger
            className={
              "flex cursor-default items-center px-2 font-sans text-xs text-text-dim " +
              "outline-none transition-colors select-none hover:bg-elev hover:text-foreground " +
              "focus-visible:ring-2 focus-visible:ring-ring " +
              "data-[popup-open]:bg-accent data-[popup-open]:text-accent-foreground"
            }
          >
            {menu.label}
          </Menu.Trigger>
          <Menu.Portal>
            <Menu.Positioner align="start" sideOffset={2} className="z-50">
              <Menu.Popup className={POPUP}>
                {renderMenu(menu.children, () => undefined, `${menu.label}.`)}
              </Menu.Popup>
            </Menu.Positioner>
          </Menu.Portal>
        </Menu.Root>
      ))}
    </Menubar>
  );
}

/**
 * A segmented control: one box the height of every other control, divided
 * rather than a row of separate buttons. The children stretch to the box, which
 * is what stops a nested button setting its own height and leaving the group
 * three pixels short of its neighbours.
 */
function Segmented({ children }: { children: React.ReactNode }): React.ReactElement {
  return (
    <div className="flex h-7 shrink-0 items-stretch divide-x divide-border border border-border bg-input">
      {children}
    </div>
  );
}

/** One cell of the readout on the right. */
function Cell({
  children,
  tone,
  onClick,
  className,
}: {
  children: React.ReactNode;
  tone?: string;
  onClick?: () => void;
  className?: string;
}): React.ReactElement {
  const cls = [
    "flex items-center gap-1.5 px-2.5 font-mono text-xs whitespace-nowrap",
    onClick ? "cursor-pointer transition-colors" : "cursor-default",
    tone ?? "text-foreground",
    className ?? "",
  ].join(" ");
  return onClick ? (
    <button type="button" className={cls} onClick={onClick}>
      {children}
    </button>
  ) : (
    <div className={cls}>{children}</div>
  );
}

export default function Toolbar(): React.ReactElement {
  const doc = useEditor((s) => s.doc);
  const past = useEditor((s) => s.at);
  const future = useEditor((s) => s.steps.length - s.at);
  const status = useEditor((s) => s.status);
  const detail = useEditor((s) => s.detail);
  const viewMode = useEditor((s) => s.viewMode);
  const derived = useDerived();
  const fileInput = useRef<HTMLInputElement>(null);
  const [preset, setPreset] = useState("");

  const act = useEditor.getState();
  const { error: errors, warning: warnings } = derived.counts;

  const load = (file: File): void => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        act.setDoc(parseDoc(String(reader.result)), `Loaded ${file.name}`);
      } catch (e) {
        act.setStatus(`Could not load ${file.name}: ${(e as Error).message}`);
      }
    };
    reader.readAsText(file);
  };

  return (
    <header className="flex h-9 shrink-0 items-center gap-1.5 border-b border-border bg-panel pr-2 pl-2.5">
      <input
        id="tensorcad-open-input"
        ref={fileInput}
        type="file"
        accept=".json,application/json"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) load(file);
          e.target.value = "";
        }}
      />

      <input
        id="tensorcad-trace-input"
        type="file"
        accept=".json,application/json"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (!file) return;
          const reader = new FileReader();
          reader.onload = () => {
            void loadTrace(String(reader.result), useEditor.getState().doc).then(({ message }) =>
              useEditor.getState().setStatus(message),
            );
          };
          reader.readAsText(file);
        }}
      />

      <input
        id="tensorcad-blocks-input"
        type="file"
        accept=".json,application/json"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (!file) return;
          const reader = new FileReader();
          reader.onload = () => {
            const { doc: next, added, errors } = mergeLibrary(
              useEditor.getState().doc,
              String(reader.result),
            );
            if (added.length === 0) {
              act.setStatus(errors[0] ?? "No blocks in that file.");
              return;
            }
            act.setDoc(next, `Imported ${added.length} block(s): ${added.join(", ")}`);
          };
          reader.readAsText(file);
        }}
      />

      {/* The one thing on the row that is decoration rather than function, so
          it is the first to go when the window cannot hold everything. Below
          that the menu bar starts at the left edge, which is where a menu bar
          starts anyway.

          The widths are measured, not Tailwind's: at `xl`, exactly 1280, this
          and the FLOPs cell both appeared and pushed Share off the right edge
          of a thirteen-inch laptop, and a deployment's Sign in needs room
          beside it too. */}
      <div className="hidden shrink-0 select-none items-baseline gap-1 font-sans text-[13px] font-semibold tracking-tight min-[1360px]:flex">
        <span className="text-primary">Tensor</span>
        <span className="text-foreground">CAD</span>
      </div>

      <AppMenuBar />

      <Separator orientation="vertical" className="h-5" />

      {/* The drawing's identity, and the one shortcut into it.

          These used to be two boxes of the same width showing the same word,
          which is a coin toss over which one you are about to type into. The
          preset no longer mirrors the name: it is something you do, not
          something the document is, so it reads as an action whether or not it
          has been used, and what it loaded is spelled out in the field beside
          it. That, the narrower box and the chevron are the difference now. */}
      <Input
        className="min-w-[7rem] shrink basis-40 font-medium"
        value={doc.meta.name}
        spellCheck={false}
        aria-label="Design name"
        onChange={(e) => act.setMetaName(e.target.value)}
      />
      <Select
        className="w-32 shrink-0"
        title="Load a reference architecture"
        value={preset}
        onValueChange={(name) => {
          if (!name) return;
          act.loadPreset(name);
          setPreset("");
        }}
        options={[
          { value: "", label: "Load preset…" },
          ...PRESET_NAMES.map((name) => ({ value: name, label: name })),
        ]}
      />

      <Separator orientation="vertical" className="h-5" />

      <Segmented>
        <Tooltip content={<span>Undo{key("edit.undo")}</span>}>
          <Button
            variant="ghost"
            size="seg"
            className="w-7"
            aria-label="Undo"
            disabled={past === 0}
            onClick={() => runCommand("edit.undo")}
          >
            <Undo2 />
          </Button>
        </Tooltip>
        <Tooltip content={<span>Redo{key("edit.redo")}</span>}>
          <Button
            variant="ghost"
            size="seg"
            className="w-7"
            aria-label="Redo"
            disabled={future === 0}
            onClick={() => runCommand("edit.redo")}
          >
            <Redo2 />
          </Button>
        </Tooltip>
      </Segmented>

      {/* How far into the design the drawing opens. Flat is one editable level;
          above that, containers are drawn around what they contain. */}
      <Segmented>
        <Tooltip content={<span>Close one level of detail{key("view.detailOut")}</span>}>
          <Button
            variant="ghost"
            size="seg"
            className="w-6"
            aria-label="Close one level"
            disabled={detail === 0}
            onClick={() => runCommand("view.detailOut")}
          >
            <Minus />
          </Button>
        </Tooltip>
        <Tooltip content="How many levels of containers the drawing opens">
          <span className="flex min-w-[3.4rem] items-center justify-center px-1.5 font-mono text-[11px] text-text-dim select-none">
            {detail === 0 ? "flat" : `${detail} deep`}
          </span>
        </Tooltip>
        <Tooltip content={<span>Open one more level of detail{key("view.detailIn")}</span>}>
          <Button
            variant="ghost"
            size="seg"
            className="w-6"
            aria-label="Open one more level"
            disabled={detail >= 5}
            onClick={() => runCommand("view.detailIn")}
          >
            <Plus />
          </Button>
        </Tooltip>
      </Segmented>

      {/* Which way the design is drawn. A buried icon is not a view switch;
          this is the one control that has to say what both modes are. */}
      <Segmented>
        <Tooltip content={<span>The editable schematic{key("view.volume")}</span>}>
          <Button
            variant={viewMode === "sheet" ? "on" : "ghost"}
            size="seg"
            className="gap-1 px-2"
            aria-pressed={viewMode === "sheet"}
            onClick={() => useEditor.getState().setViewMode("sheet")}
          >
            <Squircle />
            Sheet
          </Button>
        </Tooltip>
        <Tooltip
          content={<span>Every tensor as a box at its real proportions{key("view.volume")}</span>}
        >
          <Button
            variant={viewMode === "volume" ? "on" : "ghost"}
            size="seg"
            className="gap-1 px-2"
            aria-pressed={viewMode === "volume"}
            onClick={() => useEditor.getState().setViewMode("volume")}
          >
            <Box />
            3D
          </Button>
        </Tooltip>
      </Segmented>

      {/* What just happened. The status bar along the bottom carries what is
          true; this carries what was done, and then gets out of the way.

          It is also the gap that holds the readout against the right edge, and
          the gap is what is left over once everything else has been laid out —
          so on a narrow window there may be thirty pixels of it. Three clipped
          characters of a message are worse than no message, so it is measured
          against itself and stays hidden until a sentence would actually fit. */}
      <div className="@container min-w-0 flex-1 px-1" aria-live="polite">
        {status && (
          <span className="hidden truncate text-[11px] text-text-dim @min-[9rem]:block">
            {status}
          </span>
        )}
      </div>

      {/* The readout: one box, three cells, one height. The same chrome on all
          three, so that colour means something when it appears — only the check
          state is ever coloured, and the numbers are told apart by weight. */}
      <div className="flex h-7 shrink-0 items-stretch divide-x divide-border border border-border bg-elev">
        <Tooltip content="Open the design-rule check">
          <Cell
            tone={
              errors
                ? "bg-error-soft text-error hover:bg-error hover:text-error-soft"
                : warnings
                  ? "bg-warn-soft text-warn hover:bg-warn hover:text-warn-soft"
                  : "text-ok hover:bg-accent hover:text-accent-foreground"
            }
            onClick={() => runCommand("panel.rules")}
          >
            {/* Takes the cell's own colour, so it cannot vanish into the fill
                when a red cell is hovered and inverts. */}
            <span className="size-1.5 shrink-0 bg-current" aria-hidden />
            {errors > 0
              ? `${errors} error${errors === 1 ? "" : "s"}`
              : warnings > 0
                ? `${warnings} warning${warnings === 1 ? "" : "s"}`
                : "checks pass"}
          </Cell>
        </Tooltip>

        <Tooltip
          content={`${derived.params.total.toLocaleString("en-US")} parameters, ${derived.params.active.toLocaleString("en-US")} active per token`}
        >
          <Cell className="font-semibold">
            {formatCount(derived.params.total)}
            {derived.params.active !== derived.params.total && (
              <span className="text-[10px] font-normal text-dim">
                {formatCount(derived.params.active)} active
              </span>
            )}
          </Cell>
        </Tooltip>

        {/* The first thing to go when the window is too narrow for the whole
            row. It is the least urgent of the three and the Readout panel
            carries it in full, so losing it costs nothing that is not one
            glance to the right. */}
        <div className="hidden items-stretch min-[1520px]:flex">
          <Tooltip
            content={`Forward FLOPs per token at T=${derived.analysis.options.T.toLocaleString("en-US")}`}
          >
            <Cell>
              {formatFlops(derived.analysis.flops.fwdTotal)}
              <span className="text-[10px] text-dim">per token</span>
            </Cell>
          </Tooltip>
        </div>
      </div>

      {/* Sharing and the account, in the corner people look for them. */}
      <ShareButton />
      <AccountButton />
    </header>
  );
}
