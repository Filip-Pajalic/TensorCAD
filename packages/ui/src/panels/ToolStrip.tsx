/**
 * The tool strip.
 *
 * Every drawing program has one, down the left edge of the sheet: which tool
 * the pointer currently is, then the things you do to the view. Modal tools
 * matter here for the same reason they matter in a schematic editor — on a
 * dense drawing a long drag meant as a wire will otherwise pick up a block and
 * move it, and you will not notice until the layout is wrong.
 */

import {
  Box,
  Crosshair,
  Frame,
  Hand,
  Lock,
  LockOpen,
  Maximize2,
  MousePointer2,
  Spline,
  Trash2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useEditor, type Tool } from "../state/store.js";
import { runCommand, prettyShortcut, COMMAND_BY_ID } from "../state/commands.js";
import { Button } from "../ui/button.js";
import { Tooltip, Kbd } from "../ui/tooltip.js";
import { Separator } from "../ui/separator.js";

const TOOLS: { id: Tool; label: string; key: string; icon: React.ReactNode; hint: string }[] = [
  {
    id: "select",
    label: "Select",
    key: "V",
    icon: <MousePointer2 />,
    hint: "Click to select, drag to move, drag the sheet to rubber-band.",
  },
  {
    id: "pan",
    label: "Pan",
    key: "H",
    icon: <Hand />,
    hint: "Drag anywhere to move the sheet. Blocks stay put.",
  },
  {
    id: "wire",
    label: "Wire",
    key: "W",
    icon: <Spline />,
    hint: "Drag from a part to another to connect them. Every pin is shown while this is on.",
  },
];

function Action({
  commandId,
  label,
  icon,
  disabled,
}: {
  commandId: string;
  label: string;
  icon: React.ReactNode;
  disabled?: boolean;
}): React.ReactElement {
  const command = COMMAND_BY_ID.get(commandId);
  return (
    <Tooltip
      side="right"
      content={
        <span>
          {label}
          {command?.shortcut && <Kbd>{prettyShortcut(command.shortcut)}</Kbd>}
        </span>
      }
    >
      <Button
        variant="ghost"
        size="icon"
        aria-label={label}
        disabled={disabled}
        onClick={() => runCommand(commandId)}
      >
        {icon}
      </Button>
    </Tooltip>
  );
}

export default function ToolStrip(): React.ReactElement {
  const tool = useEditor((s) => s.tool);
  const setTool = useEditor((s) => s.setTool);
  const viewMode = useEditor((s) => s.viewMode);
  const selection = useEditor((s) => s.selection);
  const detail = useEditor((s) => s.detail);
  const locked = useEditor((s) => (selection ? s.isLocked(selection) : false));

  return (
    <div className="flex w-9 shrink-0 flex-col items-center gap-0.5 border-r border-border bg-panel py-1.5">
      <Tooltip
        side="right"
        content={
          <span>
            <span className="font-medium">Volume view</span>
            <Kbd>3</Kbd>
            <span className="mt-1 block text-dim">
              The same design as boxes at their real proportions: the tower shows where the
              parameters actually are.
            </span>
          </span>
        }
      >
        <Button
          variant={viewMode === "volume" ? "on" : "ghost"}
          size="icon"
          aria-label="Volume view"
          aria-pressed={viewMode === "volume"}
          onClick={() => runCommand("view.volume")}
        >
          <Box />
        </Button>
      </Tooltip>

      <Separator className="my-1 w-5" />

      {TOOLS.map((t) => (
        <Tooltip
          key={t.id}
          side="right"
          content={
            <span>
              <span className="font-medium">{t.label}</span>
              <Kbd>{t.key}</Kbd>
              <span className="mt-1 block text-dim">{t.hint}</span>
            </span>
          }
        >
          <Button
            variant={tool === t.id ? "on" : "ghost"}
            size="icon"
            aria-label={t.label}
            aria-pressed={tool === t.id}
            onClick={() => setTool(t.id)}
          >
            {t.icon}
          </Button>
        </Tooltip>
      ))}

      <Separator className="my-1 w-5" />

      <Action commandId="view.fit" label="Fit to window" icon={<Maximize2 />} />
      <Action commandId="view.zoomIn" label="Zoom in" icon={<ZoomIn />} />
      <Action commandId="view.zoomOut" label="Zoom out" icon={<ZoomOut />} />
      <Action commandId="view.zoomReset" label="Zoom to 100%" icon={<Crosshair />} />
      <Action commandId="view.arrange" label="Arrange" icon={<Frame />} />

      <Separator className="my-1 w-5" />

      <Action
        commandId="edit.lock"
        label={locked ? "Unlock" : "Lock"}
        icon={locked ? <Lock /> : <LockOpen />}
        disabled={!selection}
      />
      <Action
        commandId="edit.delete"
        label="Delete"
        icon={<Trash2 />}
        disabled={!selection || detail > 0}
      />
    </div>
  );
}
