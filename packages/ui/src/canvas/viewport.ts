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
