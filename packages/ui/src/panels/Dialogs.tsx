/**
 * Settings and the shortcut sheet.
 *
 * Both are generated: the shortcut sheet from the command list, so a key can
 * never be documented wrong, and settings from the same store the canvas reads,
 * so nothing has a hidden second copy of a preference.
 */

import { useEffect, useState } from "react";
import Compare from "./Compare.js";
import CommandPalette from "./CommandPalette.js";
import { Dialog, DialogContent } from "../ui/dialog.js";
import { Label } from "../ui/label.js";
import { Select } from "../ui/select.js";
import { Switch } from "../ui/switch.js";
import { Separator } from "../ui/separator.js";
import { useEditor } from "../state/store.js";
import {
  commandsIn,
  prettyShortcut,
  type CommandGroup,
} from "../state/commands.js";
import {
  onThemeChange,
  setThemePreference,
  themePreference,
  type ThemePreference,
} from "../state/theme.js";

function Setting({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}): React.ReactElement {
  return (
    <div className="flex items-start justify-between gap-6 py-2">
      <div className="min-w-0 flex-1">
        <div className="text-xs text-foreground">{label}</div>
        {hint && <div className="mt-0.5 text-[11px] leading-snug text-dim">{hint}</div>}
      </div>
      <div className="shrink-0 pt-0.5">{children}</div>
    </div>
  );
}

function SettingsBody(): React.ReactElement {
  const s = useEditor();
  const [preference, setPreference] = useState<ThemePreference>(themePreference);
  useEffect(() => onThemeChange((_, pref) => setPreference(pref)), []);

  return (
    <div className="divide-y divide-border-soft">
      <section className="pb-2">
        <Label className="mb-1 block">Appearance</Label>
        <Setting label="Theme" hint="Dark follows the system unless you pick one.">
          <Select
            className="w-32"
            value={preference}
            onValueChange={(next) => setThemePreference(next)}
            options={[
              { value: "system", label: "Follow system" },
              { value: "light", label: "Light" },
              { value: "dark", label: "Dark" },
            ]}
          />
        </Setting>
      </section>

      <section className="py-2">
        <Label className="mb-1 block">Sheet</Label>
        <Setting label="Grid" hint="A fine grid to place against and a coarse one to judge distance by.">
          <Switch checked={s.showGrid} onCheckedChange={s.setShowGrid} />
        </Setting>
        <Setting label="Snap to grid" hint="Hand-placed blocks land on the 16-unit pitch.">
          <Switch checked={s.snap} onCheckedChange={s.setSnap} />
        </Setting>
        <Setting label="Minimap" hint="Bottom-left overview of the whole drawing.">
          <Switch checked={s.showMinimap} onCheckedChange={s.setShowMinimap} />
        </Setting>
        <Setting label="Title block" hint="The drawing's own summary, in the corner of the sheet.">
          <Switch checked={s.showTitleBlock} onCheckedChange={s.setShowTitleBlock} />
        </Setting>
      </section>

      <section className="py-2">
        <Label className="mb-1 block">Drawing</Label>
        <Setting
          label="Detail"
          hint="How many levels of container are drawn open. Flat is one editable graph."
        >
          <Select
            className="w-32"
            value={s.detail}
            onValueChange={(next) => s.setDetail(Number(next))}
            options={[
              { value: 0, label: "Flat" },
              { value: 1, label: "1 level" },
              { value: 2, label: "2 levels" },
              { value: 3, label: "3 levels" },
              { value: 4, label: "4 levels" },
              { value: 5, label: "5 levels" },
            ]}
          />
        </Setting>
        <Setting label="Annotations" hint="Callouts on leader lines, derived from the design.">
          <Switch checked={s.showCallouts} onCheckedChange={() => s.toggleCallouts()} />
        </Setting>
        <Setting label="Wire labels" hint="Show evaluated sizes instead of symbol names.">
          <Switch
            checked={s.shapeMode === "numeric"}
            onCheckedChange={(on) => s.setShapeMode(on ? "numeric" : "symbolic")}
          />
        </Setting>
      </section>

      <section className="pt-2">
        <Label className="mb-1 block">Analysis</Label>
        <p className="text-[11px] leading-relaxed text-dim">
          Batch, sequence length, precision, device and the sharding plan live in the operating
          point at the top of the readout, because every number below them depends on them and a
          reading is not worth much without its conditions beside it.
        </p>
      </section>
    </div>
  );
}

const GROUP_LABEL: Record<CommandGroup, string> = {
  file: "File",
  edit: "Edit",
  view: "View",
  panel: "Panels",
  blocks: "Blocks",
  help: "Help",
};

function ShortcutsBody(): React.ReactElement {
  const groups: CommandGroup[] = ["file", "edit", "view", "panel", "blocks", "help"];
  return (
    <div className="grid gap-x-8 gap-y-4 sm:grid-cols-2">
      {groups.map((group) => (
        <section key={group}>
          <Label className="mb-1 block">{GROUP_LABEL[group]}</Label>
          <dl className="divide-y divide-border-soft">
            {commandsIn(group)
              .filter((c) => c.shortcut)
              .map((c) => (
                <div key={c.id} className="flex items-baseline justify-between gap-4 py-1">
                  <dt className="min-w-0 text-xs">
                    <span className="block truncate text-foreground">{c.label}</span>
                    {c.hint && <span className="block text-[10.5px] text-dim">{c.hint}</span>}
                  </dt>
                  <dd className="shrink-0 border border-border bg-elev px-1.5 py-px font-mono text-[10px] text-text-dim">
                    {prettyShortcut(c.shortcut)}
                  </dd>
                </div>
              ))}
          </dl>
        </section>
      ))}
      <p className="text-[11px] leading-relaxed text-dim sm:col-span-2">
        Tools: <span className="font-mono">V</span> select, <span className="font-mono">H</span>{" "}
        pan, <span className="font-mono">W</span> wire. Holding space pans with any tool, and the
        middle mouse button always pans.
      </p>
    </div>
  );
}

export default function Dialogs(): React.ReactElement {
  const dialog = useEditor((s) => s.dialog);
  const close = useEditor((s) => s.closeDialog);

  return (
    <>
      <Dialog open={dialog === "settings"} onOpenChange={(open) => !open && close()}>
        {dialog === "settings" && (
          <DialogContent title="Settings" width="30rem">
            <SettingsBody />
          </DialogContent>
        )}
      </Dialog>

      <Dialog open={dialog === "palette"} onOpenChange={(open) => !open && close()}>
        {dialog === "palette" && (
          <DialogContent title="Commands" width="32rem">
            <CommandPalette />
          </DialogContent>
        )}
      </Dialog>

      <Dialog open={dialog === "compare"} onOpenChange={(open) => !open && close()}>
        {dialog === "compare" && (
          <DialogContent title="Compare" width="46rem">
            <Compare />
          </DialogContent>
        )}
      </Dialog>

      <Dialog open={dialog === "shortcuts"} onOpenChange={(open) => !open && close()}>
        {dialog === "shortcuts" && (
          <DialogContent title="Keyboard shortcuts" width="44rem">
            <ShortcutsBody />
          </DialogContent>
        )}
      </Dialog>
    </>
  );
}
