/**
 * Which graph level is on screen, and how to get back out of it.
 */

import { useEditor } from "../state/store.js";
import { useLevel } from "../state/hooks.js";
import { formatCount } from "@tensorcad/engine";

export default function Breadcrumb(): React.ReactElement {
  const { level, derived } = useLevel();
  const shapeMode = useEditor((s) => s.shapeMode);
  const act = useEditor.getState();
  const params = level.prefix ? derived.paramsByPath.get(level.prefix) : derived.params.total;

  return (
    <div className="breadcrumb">
      <button
        className="btn btn--icon"
        disabled={level.segments.length === 0}
        title="Go up one level"
        onClick={() => act.setPath(level.segments.slice(0, -1))}
      >
        &uarr;
      </button>
      <nav className="breadcrumb__trail">
        {level.crumbs.map((crumb, i) => (
          <span className="breadcrumb__crumb" key={crumb.segments.join("/") || "root"}>
            {i > 0 && <span className="breadcrumb__sep">/</span>}
            <button
              className={`breadcrumb__button${i === level.crumbs.length - 1 ? " is-current" : ""}`}
              onClick={() => act.setPath(crumb.segments)}
            >
              {crumb.label}
              {crumb.sub && <span className="breadcrumb__sub mono">{crumb.sub}</span>}
            </button>
          </span>
        ))}
      </nav>
      <div className="breadcrumb__right">
        <div className="toggle" title="How shapes are written on edges and handles">
          {(["symbolic", "numeric"] as const).map((mode) => (
            <button
              key={mode}
              className={`toggle__option${shapeMode === mode ? " is-active" : ""}`}
              onClick={() => act.setShapeMode(mode)}
            >
              {mode === "symbolic" ? "B T D" : "B T 4096"}
            </button>
          ))}
        </div>
        {!level.editable && <span className="badge badge--warning">read-only expansion</span>}
        {level.kind === "container" && level.owner && (
          <span className="badge" title="This subgraph is stacked this many times">
            stacked &times;
            {String(derived.infer.resolved.get(level.prefix)?.p.count ?? "?")}
          </span>
        )}
        {params !== undefined && params > 0 && (
          <span className="badge badge--num mono" title={`${params.toLocaleString("en-US")} parameters here`}>
            {formatCount(params)}
          </span>
        )}
        <span className="dim mono">{level.graph.nodes.length} blocks</span>
      </div>
    </div>
  );
}
