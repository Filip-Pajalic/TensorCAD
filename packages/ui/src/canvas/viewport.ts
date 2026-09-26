/**
 * A viewport the sheet can be drawn at.
 *
 * React Flow keeps whatever viewport it is handed. One with NaN in it — a fit
 * that divided nothing by nothing is the one known way to get one — reaches
 * every place the viewport does: the grid's pattern, every wire's path, the
 * minimap's viewBox and "ZOOM NaN%" in the status bar. Nothing puts it right
 * until something sets the viewport outright, and on a sheet nobody is
 * touching that can be never.
 */

export interface Viewport {
  x: number;
  y: number;
  zoom: number;
}

/**
 * Null when this viewport is fine; otherwise the one to replace it with.
 *
 * The replacement is only somewhere finite to stand while the sheet is fitted
 * again, which the caller asks for — it is not a guess at where the drawing is.
 */
export function repairViewport(v: Viewport): Viewport | null {
  if (Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.zoom) && v.zoom > 0) return null;
  return { x: 0, y: 0, zoom: 1 };
}

/** A rectangle on the sheet, in flow coordinates. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Space kept clear at each edge, as a fraction of the pane. */
export interface Margins {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/**
 * The zoom a sheet opens at, at the least.
 *
 * A part's type line is 10.5 pixels at full size and its summary 10; below
 * about nine pixels on screen they stop being words. That is 0.85. The old
 * floor was 0.6, where a part's name was eight pixels and everything under it
 * six, and a first visit opened on a column of boxes nobody could read.
 */
export const READABLE_ZOOM = 0.85;

/**
 * Where to stand to see a drawing.
 *
 * All of it, centred, when all of it fits at `floor` or better. When it does
 * not, the drawing is shown at `floor` from its top — and from its left, if it
 * is wider than the pane — because a design is read from its input, and the
 * middle of a model at a readable size is a worse start than its beginning.
 */
export function standFor(
  bounds: Rect,
  pane: { width: number; height: number },
  margins: Margins,
  floor: number,
  ceiling: number,
): Viewport | null {
  if (!(bounds.width > 0 && bounds.height > 0 && pane.width > 0 && pane.height > 0)) return null;
  const left = pane.width * margins.left;
  const top = pane.height * margins.top;
  const wide = pane.width * (1 - margins.left - margins.right);
  const tall = pane.height * (1 - margins.top - margins.bottom);
  const all = Math.min(wide / bounds.width, tall / bounds.height);
  const zoom = Math.min(ceiling, Math.max(floor, all));
  const w = bounds.width * zoom;
  const h = bounds.height * zoom;
  const x = (w <= wide ? left + (wide - w) / 2 : left) - bounds.x * zoom;
  const y = (h <= tall ? top + (tall - h) / 2 : top) - bounds.y * zoom;
  return repairViewport({ x, y, zoom }) ? null : { x, y, zoom };
}
