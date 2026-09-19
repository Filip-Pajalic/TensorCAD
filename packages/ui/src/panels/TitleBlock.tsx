/**
 * The title block.
 *
 * Bottom-right of the sheet, as on any engineering drawing: what this is, who
 * drew it, at what scale, and the figures that identify the revision. On a
 * mechanical drawing that is the part number and the material. Here it is the
 * parameter count, the layer stack and whether the design passes its checks.
 */

import { useEditor } from "../state/store.js";
import { useDerived } from "../state/hooks.js";
import { formatCount } from "@tensorcad/engine";

function Field({
  label,
  value,
  wide,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  wide?: boolean;
  tone?: "ok" | "warning" | "error";
}): React.ReactElement {
  return (
    <div className={`title-block__field${wide ? " title-block__field--wide" : ""}`}>
      <div className="title-block__label">{label}</div>
      <div className={`title-block__value${tone ? ` is-${tone}` : ""}`}>{value}</div>
    </div>
  );
}

export default function TitleBlock(): React.ReactElement | null {
  const doc = useEditor((s) => s.doc);
  const status = useEditor((s) => s.canvasStatus);
  const derived = useDerived();

  const errors = derived.issues.filter((i) => i.severity === "error").length;
  const warnings = derived.issues.filter((i) => i.severity === "warning").length;

  const symbols = derived.symbols.values;
  const layers = symbols.L ?? symbols.Lm ?? null;
  const published = doc.meta.published?.params;
  const delta =
    published && published > 0
      ? Math.abs(derived.params.total - published) / published
      : null;

  return (
    <div className="title-block">
      <div className="title-block__head">
        <span className="title-block__name">{doc.meta.name}</span>
        {doc.meta.family && <span className="title-block__family">{doc.meta.family}</span>}
      </div>

      <div className="title-block__grid">
        <Field label="Parameters" value={formatCount(derived.params.total)} />
        <Field
          label="Active"
          value={
            derived.params.active === derived.params.total
              ? "dense"
              : formatCount(derived.params.active)
          }
        />
        <Field label="Layers" value={layers === null ? "—" : String(layers)} />
        <Field label="d_model" value={symbols.D === undefined ? "—" : String(symbols.D)} />
        <Field label="Scale" value={`${Math.round(status.zoom * 100)}%`} />
        <Field
          label="Checks"
          value={
            errors > 0
              ? `${errors} error${errors === 1 ? "" : "s"}`
              : warnings > 0
                ? `${warnings} warning${warnings === 1 ? "" : "s"}`
                : "pass"
          }
          tone={errors > 0 ? "error" : warnings > 0 ? "warning" : "ok"}
        />
        {published !== undefined && (
          <Field
            label="Against published"
            wide
            value={
              delta !== null && delta < 0.00005
                ? `${formatCount(published)} — exact`
                : `${formatCount(published)} — ${((delta ?? 0) * 100).toFixed(2)}% off`
            }
            tone={delta !== null && delta < 0.005 ? "ok" : "warning"}
          />
        )}
      </div>
    </div>
  );
}
