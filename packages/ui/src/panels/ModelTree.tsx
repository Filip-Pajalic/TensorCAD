/**
 * The model tree.
 *
 * Every CAD application has one: FreeCAD's model tree, Fusion's browser,
 * KiCad's hierarchy navigator. It answers a question the canvas cannot, which
 * is "what is in this design and how is it nested", and it is where you lock
 * things, because locking is a property of the item rather than of the view.
 *
 * The tree mirrors the document, not the expansion: containers open to show
 * what they stack, and a composite is a leaf you can open on the canvas.
 */

import { useMemo, useState } from "react";
import { ChevronsLeft } from "lucide-react";
import { useEditor } from "../state/store.js";
import { useDerived } from "../state/hooks.js";
import { categoryColor, typeName } from "../canvas/blocks.js";
import * as ops from "../state/ops.js";
import type { Doc, NodeDef } from "@tensor-cad/engine";
import { formatCount } from "@tensor-cad/engine";
import { catalogOf, type BlockDef } from "../engine.js";

/**
 * A stable empty array. A zustand selector must return the same reference when
 * nothing changed, or React re-renders forever.
 */
const NO_LOCKS: readonly string[] = [];

interface TreeItem {
  path: string;
  id: string;
  label: string;
  /** The identifier, for the tooltip and for anything that has to be typed. */
  type: string;
  /** What the row prints: the catalog's own name for the type. */
  typeName: string;
  category: string;
  depth: number;
  children: TreeItem[];
  /** True when this node holds a stored subgraph the tree can show inline. */
  expandable: boolean;
  /** True when the canvas can open this node as its own level. */
  drillable: boolean;
}

function build(
  cat: Record<string, BlockDef>,
  nodes: NodeDef[],
  prefix: string,
  depth: number,
): TreeItem[] {
  return nodes.map((node) => {
    const path = prefix ? `${prefix}/${node.id}` : node.id;
    const def = cat[node.type];
    return {
      path,
      id: node.id,
      label: node.label ?? node.id,
      type: node.type,
      typeName: typeName(def, node.type),
      category: def?.category ?? "unknown",
      depth,
      children: node.graph ? build(cat, node.graph.nodes, path, depth + 1) : [],
      expandable: Boolean(node.graph && node.graph.nodes.length > 0),
      drillable: ops.isDrillable(node, def),
    };
  });
}

function Row({
  item,
  open,
  onToggleOpen,
}: {
  item: TreeItem;
  open: boolean;
  onToggleOpen: () => void;
}): React.ReactElement {
  const derived = useDerived();
  const selection = useEditor((s) => s.selection);
  const lockList = useEditor((s) => (s.doc.ui as { locked?: string[] } | undefined)?.locked);
  const isLocked = (lockList ?? NO_LOCKS).includes(item.path);
  const params = derived.paramsByPath.get(item.path) ?? 0;
  const severity = derived.severityByPath.get(item.path) ?? null;

  return (
    <div
      className={`tree__row${selection === item.path ? " is-selected" : ""}`}
      style={{ paddingLeft: 6 + item.depth * 13 }}
      onClick={() => useEditor.getState().focusOn(item.path)}
      onDoubleClick={() => {
        if (item.drillable || item.expandable) useEditor.getState().focusOn(item.path);
      }}
      title={`${item.path}\n${item.typeName} (${item.type})`}
    >
      <button
        className={`tree__twisty${item.expandable ? "" : " is-empty"}`}
        onClick={(e) => {
          e.stopPropagation();
          if (item.expandable) onToggleOpen();
        }}
        tabIndex={-1}
        aria-label={open ? "collapse" : "expand"}
      >
        {item.expandable ? (open ? "▾" : "▸") : ""}
      </button>

      <span className="tree__swatch" style={{ background: categoryColor(item.category) }} />
      <span className="tree__label">{item.label}</span>
      <span className="tree__type">{item.typeName}</span>

      {severity && <span className={`mark mark--${severity}`}>{severity === "error" ? "✖" : "⚠"}</span>}
      {params > 0 && <span className="tree__params">{formatCount(params)}</span>}

      <button
        className={`tree__lock${isLocked ? " is-locked" : ""}`}
        title={isLocked ? "Unlock: allow this block to be moved" : "Lock this block's position"}
        onClick={(e) => {
          e.stopPropagation();
          useEditor.getState().toggleLock(item.path);
        }}
      >
        {isLocked ? "\u{1F512}" : "\u{1F513}"}
      </button>
    </div>
  );
}

function Branch({ items }: { items: TreeItem[] }): React.ReactElement {
  const [openPaths, setOpenPaths] = useState<Set<string>>(() => new Set(items.map((i) => i.path)));
  return (
    <>
      {items.map((item) => {
        const open = openPaths.has(item.path);
        return (
          <div key={item.path}>
            <Row
              item={item}
              open={open}
              onToggleOpen={() =>
                setOpenPaths((prev) => {
                  const next = new Set(prev);
                  if (next.has(item.path)) next.delete(item.path);
                  else next.add(item.path);
                  return next;
                })
              }
            />
            {open && item.children.length > 0 && <Branch items={item.children} />}
          </div>
        );
      })}
    </>
  );
}

export default function ModelTree(): React.ReactElement {
  const doc = useEditor((s) => s.doc);
  const lockList = useEditor((s) => (s.doc.ui as { locked?: string[] } | undefined)?.locked);
  const locked = lockList ?? NO_LOCKS;
  // Through the document's own catalog, not the built-in one (invariant 1):
  // a design's own blocks would otherwise list as an unknown category.
  const items = useMemo(() => build(catalogOf(doc), doc.graph.nodes, "", 0), [doc]);

  return (
    <div className="tree">
      {/* The collapse control lives on the panel, the way Fusion puts the
          arrow on its browser rather than in the ribbon. */}
      <div className="panel__head">
        <h2>Model</h2>
        <span className="panel__meta">
          {locked.length > 0
            ? `${items.length} top-level, ${locked.length} locked`
            : `${items.length} top-level`}
        </span>
        <button
          className="dock__collapse"
          title="Collapse the model tree (Ctrl+B)"
          aria-label="Collapse the model tree"
          onClick={() => useEditor.getState().toggleDock("left")}
        >
          <ChevronsLeft className="size-3.5" />
        </button>
      </div>
      <div className="tree__body">
        <Branch items={items} />
      </div>
    </div>
  );
}
