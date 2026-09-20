/**
 * The sheet, as a vector.
 *
 * A schematic tool that cannot hand you the drawing is missing something
 * obvious: a figure goes in a paper, a slide, a pull request. A screenshot
 * gives you a picture of the drawing at one zoom on one screen; this gives you
 * the drawing.
 *
 * ## Why it reads the rendered sheet rather than re-drawing it
 *
 * The wires are the hard part. `wiring.ts` decides which of a port's four sides
 * a wire leaves by, routes it orthogonally around what is in the way, and puts
 * junction dots where a net branches — and it has already done all of that, as
 * `<path d="…">`, by the time you press export. Re-deriving it here would be a
 * second router, and the two would disagree. So the paths are taken as they
 * are, and only the blocks are drawn again, because they are HTML and an HTML
 * div is not a thing an SVG can hold without `foreignObject` — which is a
 * screenshot with extra steps and does not open in Illustrator.
 *
 * ## Colours
 *
 * Read off the rendered elements rather than off the tokens, so the export
 * matches the theme on screen. A literal in here would be a third opinion
 * about what colour a block is.
 */

/** Something the canvas painted: a block body, a frame, a pin mark. */
interface Shape {
  x: number;
  y: number;
  w: number;
  h: number;
  r: number;
  fill: string;
  stroke: string;
  strokeWidth: number;
}

/** One run of text, where the browser laid it out. */
interface Run {
  text: string;
  x: number;
  y: number;
  size: number;
  weight: string;
  fill: string;
}

const escape = (text: string): string =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/**
 * The sheet inside a React Flow container, as an SVG document.
 *
 * Returns undefined when there is nothing rendered to read, which is what a
 * caller gets if it asks before the first frame.
 */
export function sheetToSvg(root: HTMLElement, title: string): string | undefined {
  const viewport = root.querySelector<HTMLElement>(".react-flow__viewport");
  const nodes = [...root.querySelectorAll<HTMLElement>(".react-flow__node")];
  if (!viewport || nodes.length === 0) return undefined;

  // Every measurement is taken against the viewport's own box, so the export
  // is in sheet coordinates and does not carry the current pan and zoom.
  const origin = viewport.getBoundingClientRect();
  const scale = zoomOf(viewport);
  const at = (rect: DOMRect): { x: number; y: number; w: number; h: number } => ({
    x: (rect.left - origin.left) / scale,
    y: (rect.top - origin.top) / scale,
    w: rect.width / scale,
    h: rect.height / scale,
  });

  const shapes: Shape[] = [];
  const runs: Run[] = [];
  for (const node of nodes) readNode(node, at, shapes, runs);
  const wires = [...root.querySelectorAll<SVGPathElement>(".react-flow__edge-path")].map((path) => ({
    d: path.getAttribute("d") ?? "",
    stroke: getComputedStyle(path).stroke,
    width: Number.parseFloat(getComputedStyle(path).strokeWidth) || 1,
    dash: getComputedStyle(path).strokeDasharray,
  }));
  const labels = [...root.querySelectorAll<HTMLElement>(".react-flow__edge-textwrapper, .react-flow__edgelabel-renderer > div")]
    .map((el) => ({ text: (el.textContent ?? "").trim(), box: at(el.getBoundingClientRect()) }))
    .filter((l) => l.text.length > 0);

  const pad = 24;
  const all = [...shapes.map((b) => ({ x: b.x, y: b.y, w: b.w, h: b.h })), ...labels.map((l) => l.box)];
  const minX = Math.min(...all.map((b) => b.x)) - pad;
  const minY = Math.min(...all.map((b) => b.y)) - pad;
  const maxX = Math.max(...all.map((b) => b.x + b.w)) + pad;
  const maxY = Math.max(...all.map((b) => b.y + b.h)) + pad;
  const width = Math.ceil(maxX - minX);
  const height = Math.ceil(maxY - minY);

  const sheet = getComputedStyle(root).getPropertyValue("--sheet").trim() || "#ffffff";
  const out: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="ui-monospace, monospace">`,
    `<title>${escape(title)}</title>`,
    `<rect width="${width}" height="${height}" fill="${sheet}"/>`,
    `<g transform="translate(${(-minX).toFixed(1)} ${(-minY).toFixed(1)})">`,
  ];

  // Wires under blocks, the way the canvas stacks them.
  for (const w of wires) {
    if (!w.d) continue;
    const dash = w.dash && w.dash !== "none" ? ` stroke-dasharray="${w.dash.replace(/px/g, "")}"` : "";
    out.push(`<path d="${w.d}" fill="none" stroke="${w.stroke}" stroke-width="${w.width}"${dash}/>`);
  }

  for (const b of shapes) {
    const round = b.r > 0 ? ` rx="${b.r.toFixed(1)}"` : "";
    out.push(
      `<rect x="${b.x.toFixed(1)}" y="${b.y.toFixed(1)}" width="${b.w.toFixed(1)}" height="${b.h.toFixed(1)}"${round} fill="${b.fill}" stroke="${b.stroke}" stroke-width="${b.strokeWidth.toFixed(2)}"/>`,
    );
  }

  for (const line of runs) {
    out.push(
      `<text x="${line.x.toFixed(1)}" y="${line.y.toFixed(1)}" text-anchor="middle" font-size="${line.size.toFixed(1)}" font-weight="${line.weight}" fill="${line.fill}">${escape(line.text)}</text>`,
    );
  }

  for (const l of labels) {
    out.push(
      `<text x="${(l.box.x + l.box.w / 2).toFixed(1)}" y="${(l.box.y + l.box.h * 0.75).toFixed(1)}" text-anchor="middle" font-size="9" fill="${getComputedStyle(root).getPropertyValue("--wire-label").trim() || "#666"}">${escape(l.text)}</text>`,
    );
  }

  out.push("</g>", "</svg>");
  return out.join("\n");
}

/** The zoom React Flow has applied, from the viewport's transform. */
function zoomOf(viewport: HTMLElement): number {
  const m = /scale\(([\d.]+)\)/.exec(viewport.style.transform);
  if (m) return Number.parseFloat(m[1]!) || 1;
  const matrix = new DOMMatrixReadOnly(getComputedStyle(viewport).transform);
  return matrix.a || 1;
}

/**
 * Everything one node paints, and everything it says.
 *
 * "The block" is not one element. The fill and the border are on
 * `.part__body`, the pins are their own marks around it, and a container is a
 * `.frame` — so this looks for anything with a background or a visible border
 * rather than naming the classes, which would go stale the first time the
 * canvas is restyled.
 *
 * The first export of this read the *node*, whose background is transparent,
 * and wrote every block with no fill at all. Nothing looked wrong until a
 * block with dark text on a light fill turned up: attention, invisible against
 * the sheet.
 */
function readNode(
  node: HTMLElement,
  at: (rect: DOMRect) => { x: number; y: number; w: number; h: number },
  shapes: Shape[],
  runs: Run[],
): void {
  for (const el of [node, ...node.querySelectorAll<HTMLElement>("*")]) {
    // React Flow's handles are invisible targets sized for the pointer; the
    // pin they sit on is the thing that is drawn.
    if (el.classList.contains("react-flow__handle")) continue;
    const s = getComputedStyle(el);
    const filled = s.backgroundColor !== "rgba(0, 0, 0, 0)" && s.backgroundColor !== "transparent";
    const width = Number.parseFloat(s.borderTopWidth) || 0;
    const outlined = width > 0 && s.borderTopStyle !== "none" && s.borderTopColor !== "rgba(0, 0, 0, 0)";
    if (!filled && !outlined) continue;
    const box = at(el.getBoundingClientRect());
    if (box.w < 0.5 || box.h < 0.5) continue;
    shapes.push({
      ...box,
      r: Number.parseFloat(s.borderTopLeftRadius) || 0,
      fill: filled ? s.backgroundColor : "none",
      stroke: outlined ? s.borderTopColor : "none",
      strokeWidth: outlined ? width : 0,
    });
  }

  for (const el of node.querySelectorAll<HTMLElement>("*")) {
    if (el.children.length > 0) continue;
    const text = (el.textContent ?? "").trim();
    if (!text) continue;
    const box = at(el.getBoundingClientRect());
    const s = getComputedStyle(el);
    runs.push({
      text,
      // Each run where the browser put it, not where the block's centre is.
      // A row of three inline runs — "vocab 128256 · D 4096 · 525.3M" — centred
      // on the block collapses into one illegible pile, which is what the
      // first export of this did.
      x: box.x + box.w / 2,
      // The baseline, near enough: three quarters down the line box is where a
      // monospace baseline sits, and an SVG has no line box to ask.
      y: box.y + box.h * 0.75,
      size: Number.parseFloat(s.fontSize) || 10,
      weight: s.fontWeight || "normal",
      fill: s.color,
    });
  }
}
