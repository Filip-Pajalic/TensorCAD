/**
 * Application shell.
 *
 * Left: the model tree, with the parts palette folded away beneath it until you
 * want it. Middle: the drawing. Right: the readout on top and the editing
 * surfaces beneath, split rather than tabbed, because the numbers are the
 * reason to open the tool and watching them move while you change a parameter
 * is the whole of "validating and testing". Tabs stay where they belong — over
 * the three things you edit, only one of which you edit at a time.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { ChevronsRight } from "lucide-react";
import Canvas from "../canvas/Canvas.js";
import View3D from "../three/View3D.js";
import Palette from "../panels/Palette.js";
import Toolbar from "../panels/Toolbar.js";
import Breadcrumb from "../panels/Breadcrumb.js";
import Inspector from "../panels/Inspector.js";
import Symbols from "../panels/Symbols.js";
import Analysis from "../panels/Analysis.js";
import Operating from "../panels/Operating.js";
import Rules from "../panels/Rules.js";
import Cluster from "../panels/Cluster.js";
import StatusBar from "../panels/StatusBar.js";
import ModelTree from "../panels/ModelTree.js";
import ToolStrip from "../panels/ToolStrip.js";
import Dialogs from "../panels/Dialogs.js";
import Ladder from "../panels/Ladder.js";
import CanvasMenu from "../panels/CanvasMenu.js";
import DockRail from "../panels/DockRail.js";
import { useEditor, type RightTab } from "../state/store.js";
import { useDerived } from "../state/hooks.js";
import { handleKey } from "../state/commands.js";
import { Tabs, TabsList, TabsTrigger } from "../ui/tabs.js";
import { TooltipProvider } from "../ui/tooltip.js";
import { formatCount } from "@tensorcad/engine";

const TABS: { id: RightTab; label: string }[] = [
  { id: "inspector", label: "Inspector" },
  { id: "symbols", label: "Symbols" },
  { id: "rules", label: "Rules" },
  { id: "cluster", label: "Cluster" },
  { id: "ladder", label: "Ladder" },
];

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

/** A drag handle between two panes. Horizontal drags move x, vertical move y. */
const RAIL = 26;

function Resizer({
  onResize,
  axis,
  invert = false,
  onToggle,
}: {
  onResize: (delta: number) => void;
  axis: "x" | "y";
  invert?: boolean;
  /** Double-clicking a divider collapses the dock beside it. */
  onToggle?: () => void;
}): React.ReactElement {
  const last = useRef(0);
  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>): void => {
    e.preventDefault();
    last.current = axis === "x" ? e.clientX : e.clientY;
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent): void => {
      const now = axis === "x" ? ev.clientX : ev.clientY;
      const delta = now - last.current;
      last.current = now;
      onResize(invert ? -delta : delta);
    };
    const up = (): void => {
      el.releasePointerCapture(e.pointerId);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  };
  return (
    <div
      className={`resizer resizer--${axis}`}
      onPointerDown={onPointerDown}
      onDoubleClick={onToggle}
      title={onToggle ? "Drag to resize, double-click to collapse" : undefined}
      role="separator"
      aria-orientation={axis === "x" ? "vertical" : "horizontal"}
    />
  );
}

export default function App(): React.ReactElement {
  const rightTab = useEditor((s) => s.rightTab);
  const path = useEditor((s) => s.path);
  const paletteOpen = useEditor((s) => s.paletteOpen);
  const viewMode = useEditor((s) => s.viewMode);
  const leftCollapsed = useEditor((s) => s.leftCollapsed);
  const rightCollapsed = useEditor((s) => s.rightCollapsed);
  const toggleDock = useEditor((s) => s.toggleDock);
  const derived = useDerived();
  const [leftWidth, setLeftWidth] = useState(262);
  const [rightWidth, setRightWidth] = useState(400);
  /** Height of the readout, as a fraction of the right column. */
  const [readoutFraction, setReadoutFraction] = useState(0.66);
  const rightColumn = useRef<HTMLElement | null>(null);

  // One listener, one command list. Tool keys are here rather than in the list
  // because a tool is a mode rather than an action, and a mode has no menu item.
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (handleKey(e)) return;
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) || target.isContentEditable)
      ) {
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const tool = { v: "select", h: "pan", w: "wire" }[e.key.toLowerCase()];
      if (tool) {
        e.preventDefault();
        useEditor.getState().setTool(tool as "select" | "pan" | "wire");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const errorCount = derived.counts.error;
  const warningCount = derived.counts.warning;

  const renderTab = useCallback((): React.ReactElement => {
    switch (rightTab) {
      case "symbols":
        return <Symbols />;
      case "rules":
        return <Rules />;
      case "cluster":
        return <Cluster />;
      case "ladder":
        return <Ladder />;
      default:
        return <Inspector />;
    }
  }, [rightTab]);

  const dragReadout = useCallback((delta: number) => {
    const height = rightColumn.current?.clientHeight ?? 1;
    setReadoutFraction((f) => clamp(f + delta / height, 0.18, 0.86));
  }, []);

  return (
    <TooltipProvider delay={420} closeDelay={80}>
      <div className="app">
        <Toolbar />
        <div
          className="app__body"
          style={{
            gridTemplateColumns: [
              leftCollapsed ? `${RAIL}px` : `${leftWidth}px`,
              leftCollapsed ? "0px" : "5px",
              "1fr",
              rightCollapsed ? "0px" : "5px",
              rightCollapsed ? `${RAIL}px` : `${rightWidth}px`,
            ].join(" "),
          }}
        >
          {/* Five grid items, always. A collapsed dock swaps its contents for a
            rail and its divider for a zero-width spacer rather than being
            removed, because dropping a grid item shifts every later one into
            the wrong column. */}
          {leftCollapsed ? (
            <DockRail side="left" label="Model" onExpand={() => toggleDock("left")} />
          ) : (
            <aside className={`app__left${paletteOpen ? " has-palette" : ""}`}>
              <ModelTree />
              <Palette />
            </aside>
          )}
          {leftCollapsed ? (
            <div aria-hidden />
          ) : (
            <Resizer
              axis="x"
              onResize={(d) => setLeftWidth((w) => clamp(w + d, 190, 460))}
              onToggle={() => toggleDock("left")}
            />
          )}

          <main className="app__center flex-row">
            <ToolStrip />
            <div className="flex min-w-0 flex-1 flex-col">
              <Breadcrumb />
              <CanvasMenu>
                {viewMode === "volume" ? (
                  <View3D />
                ) : (
                  <ReactFlowProvider key={path.join("/") || "root"}>
                    <Canvas />
                  </ReactFlowProvider>
                )}
              </CanvasMenu>
            </div>
          </main>

          {rightCollapsed ? (
            <div aria-hidden />
          ) : (
            <Resizer
              axis="x"
              invert
              onResize={(d) => setRightWidth((w) => clamp(w + d, 300, 680))}
              onToggle={() => toggleDock("right")}
            />
          )}
          {rightCollapsed ? (
            <DockRail
              side="right"
              label="Readout"
              note={formatCount(derived.params.total)}
              onExpand={() => toggleDock("right")}
            />
          ) : (
            <aside
              className="app__right"
              ref={rightColumn}
              style={{
                gridTemplateRows: `${readoutFraction}fr 5px ${1 - readoutFraction}fr`,
              }}
            >
              <section className="panel panel--readout">
                {/* The collapse control sits on the panel it collapses, the way
                    Fusion puts the arrow on its browser and not in the ribbon. */}
                <div className="panel__head">
                  <h2>Readout</h2>
                  <span className="panel__meta">under the operating point</span>
                  <button
                    className="dock__collapse"
                    title="Collapse the readout (Ctrl+Shift+B)"
                    aria-label="Collapse the readout"
                    onClick={() => toggleDock("right")}
                  >
                    <ChevronsRight className="size-3.5" />
                  </button>
                </div>
                <Operating />
                <Analysis />
              </section>
              <Resizer axis="y" onResize={dragReadout} />
              <section className="panel panel--edit">
                <Tabs
                  value={rightTab}
                  onValueChange={(value) => useEditor.getState().setRightTab(value as RightTab)}
                  className="flex min-h-0 flex-1 flex-col"
                >
                  <TabsList>
                    {TABS.map((tab) => (
                      <TabsTrigger key={tab.id} value={tab.id}>
                        {tab.label}
                        {tab.id === "rules" && errorCount > 0 && (
                          <span className="border border-error bg-error px-1 font-mono text-[9.5px] font-bold text-error-soft">
                            {errorCount}
                          </span>
                        )}
                        {tab.id === "rules" && errorCount === 0 && warningCount > 0 && (
                          <span className="border border-warn bg-warn px-1 font-mono text-[9.5px] font-bold text-warn-soft">
                            {warningCount}
                          </span>
                        )}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                  <div className="flex min-h-0 flex-1 flex-col">{renderTab()}</div>
                </Tabs>
              </section>
            </aside>
          )}
        </div>
        <StatusBar />
        <Dialogs />
      </div>
    </TooltipProvider>
  );
}
