/**
 * The volume view.
 *
 * Renders the layout from `model3d.ts`, which is a port of Brendan Bycroft's
 * LLM visualisation (MIT). Two things carry the look, and both are cheap:
 *
 *   - every face is shaded with the block's own cell grid, so a tensor reads as
 *     a matrix rather than as a box. The grid is drawn in a fragment shader with
 *     its density clamped, because a 128k-wide plate has no business asking for
 *     128k lines, and it fades out with distance so a wall of cells never turns
 *     into moiré;
 *   - weights and activations are different materials. That distinction is the
 *     reason to look at this at all: you can see the residual pathway running
 *     through a model that is otherwise all weights.
 *
 * The render loop runs on demand rather than every frame. A static tower does
 * not need sixty redraws a second.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { setViewportApi } from "../state/commands.js";
import { useEditor } from "../state/store.js";
import { useDerived } from "../state/hooks.js";
import { onThemeChange, resolvedTheme, themeValue } from "../state/theme.js";
import { buildModel3D, describeBlk, type Arrow, type Blk, type Model3D } from "./model3d.js";
import { cellsFor, useTrace, type Trace } from "./trace.js";
import { formatCount } from "@tensor-cad/engine";

/**
 * A cell grid on every face.
 *
 * `uCells` is how many cells run along each of the box's three axes. The
 * fragment shader picks the two that face the camera from the interpolated
 * local position, draws lines at that pitch, and gives up once a cell would be
 * smaller than a pixel or two — past that the honest thing is a flat face.
 */
const CELL_VERT = /* glsl */ `
varying vec3 vLocal;
varying vec3 vNormal;
void main() {
  vLocal = position + 0.5;
  vNormal = normalMatrix * normal;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const CELL_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform vec3 uGrid;
uniform vec3 uEdge;
uniform vec3 uCells;
uniform float uHighlight;
uniform float uHover;
// A block's real values, when a trace covers it: red is the value scaled to
// [-1, 1] by the block's own largest, green is 1 where there is a value and 0
// where the position was never visible — a masked attention score.
uniform sampler2D uData;
uniform float uHasData;
varying vec3 vLocal;
varying vec3 vNormal;

/**
 * How many divisions to actually rule this axis into.
 *
 * A real tensor here has thousands of cells per axis, so drawing one line per
 * cell gives either moire or — once the shader gives up — a flat painted face,
 * which is what this looked like before. Instead the face is ruled into as many
 * divisions as can be seen, halving down from the true cell count until they
 * are at least eight pixels apart. Zoom in far enough on a small tensor and the
 * divisions become the actual cells; zoom out and you get a readable eight or
 * sixteen. Either way the face says "matrix" rather than "painted box".
 */
float divisionsFor(float coord, float cells) {
  float px = fwidth(coord);
  if (px <= 0.0) return cells;
  float visible = 1.0 / (px * 8.0);
  if (cells <= visible) return cells;
  return max(1.0, exp2(floor(log2(max(1.0, visible)))));
}

float rule(float coord, float divs) {
  float scaled = coord * divs;
  float w = fwidth(scaled);
  float d = abs(fract(scaled) - 0.5) * 2.0;
  return smoothstep(1.0 - w * 2.2, 1.0, d);
}

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// A value per cell, drawn only where the divisions really are the cells. The
// design carries shapes, not weights, so this is texture rather than data and
// it is kept faint enough never to be mistaken for one. A block a trace covers
// draws its real values instead, much stronger, and the legend says which is
// which.
float speckle(vec2 coord, vec2 divs, vec2 cells) {
  if (divs.x < cells.x || divs.y < cells.y) return 0.0;
  return hash(floor(coord * divs)) - 0.5;
}

// The cube's own edges. The reference draws these in white — its colorEdge — on
// top of everything else, and it is what makes a plate read as a solid with
// sides rather than a painted rectangle, especially where the side is only a
// few pixels across.
float border(vec2 uv) {
  vec2 w = fwidth(uv) * 1.5;
  vec2 d = min(uv, 1.0 - uv);
  return 1.0 - smoothstep(0.0, max(w.x, w.y), min(d.x, d.y));
}

void main() {
  vec3 n = normalize(vNormal);
  vec3 an = abs(n);

  // Only the two axes in the plane of this face are ruled.
  vec2 uv;
  vec2 cells;
  if (an.z > an.x && an.z > an.y) {
    uv = vLocal.xy;
    cells = uCells.xy;
  } else if (an.x > an.y) {
    uv = vLocal.zy;
    cells = uCells.zy;
  } else {
    uv = vLocal.xz;
    cells = uCells.xz;
  }

  vec2 divs = vec2(divisionsFor(uv.x, cells.x), divisionsFor(uv.y, cells.y));
  float g = max(rule(uv.x, divs.x), rule(uv.y, divs.y));
  float v;
  float empty = 0.0;
  if (uHasData > 0.5) {
    // Only the faces that show both of the tensor's axes carry its values.
    // The others would be a slice along the batch, and a trace is one sequence.
    v = 0.0;
    if (an.z > an.x && an.z > an.y) {
      // The layout's rows run downward from the top of the block; a data
      // texture's first row is at v = 0.
      vec2 d = texture2D(uData, vec2(uv.x, 1.0 - uv.y)).rg;
      v = d.r * 2.0;
      empty = 1.0 - d.g;
    }
  } else {
    v = speckle(uv, divs, cells);
  }
  float b = border(uv);

  // Mostly ambient with one soft key, so a face turned away still reads and
  // nothing goes black.
  float lambert = 0.78 + 0.22 * max(0.0, dot(n, normalize(vec3(0.3, 0.85, 0.45))));
  vec3 col = uColor * lambert;
  col *= 1.0 + v * 0.34;
  col = mix(col, uGrid, empty * 0.75);
  col = mix(col, uGrid, g * 0.32);
  col = mix(col, uEdge, b * 0.9);
  col = mix(col, vec3(1.0), uHover);
  col = mix(col, vec3(1.0, 0.95, 0.65), uHighlight * 0.45);
  gl_FragColor = vec4(col, 1.0);

  // THREE.Color converts a hex from sRGB into the linear working space, and a
  // raw ShaderMaterial writes straight to the framebuffer with no conversion
  // back. Without this every colour renders about two stops too dark.
  #include <colorspace_fragment>
}
`;


/**
 * A block's values as a texture: one texel per cell, nearest-sampled.
 *
 * Scaled by the block's own largest magnitude, so a weight matrix and the
 * residual stream beside it are each legible rather than one washing out the
 * other. That makes the brightness a comparison *within* a block; the hover
 * readout says what the scale was.
 */
function dataTexture(values: Float32Array, cx: number, cy: number): { tex: THREE.DataTexture; max: number } {
  let max = 0;
  for (const v of values) if (Number.isFinite(v)) max = Math.max(max, Math.abs(v));
  const scale = max > 0 ? 1 / max : 1;
  const rg = new Float32Array(cx * cy * 2);
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    const seen = Number.isFinite(v);
    rg[i * 2] = seen ? v * scale : 0;
    rg[i * 2 + 1] = seen ? 1 : 0;
  }
  const tex = new THREE.DataTexture(rg, cx, cy, THREE.RGFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  return { tex, max };
}

/**
 * A ribbon between two blocks.
 *
 * The reference draws its arrows as flat rectangular pathways with a triangular
 * head — a quad strip along the centre line in the x-y plane, since every arrow
 * lies in that plane. Built into one geometry per colour so the whole flow is
 * two draw calls rather than two hundred.
 */
/**
 * The ribbons, as two geometries.
 *
 * This is `drawArrow` from the reference, kept close to it on purpose.
 *
 * The trick worth understanding is the frame. A run builds an orthonormal basis
 * from its own direction — `side`, `dir`, `normal` — flattens both endpoints
 * into it, lays the ribbon out in that frame's x-y plane, and transforms the
 * result back. Because the basis is a reflection it is its own inverse, so the
 * same matrix does both. Everything downstream can then think in two dimensions
 * while the ribbon itself sits wherever in space it needs to.
 *
 * That is what makes the curves possible: when the two ends are at different
 * depths, or the ribbon has to arrive side-on, the run is swept as a cubic
 * bezier whose tangents are `dir` at the start and the arrival direction at the
 * end. Six attention heads funnelling back into one output is that case, and it
 * is most of what the reference's picture is showing.
 *
 * The reference's own note on the design — "a flat, rectangular, ribbon-like
 * pathway with lines down the edges" — is followed here for every run. Its code
 * has the border lines commented out on the straight segments and left in on
 * the curved ones; without them a ribbon is a smear rather than a path.
 */

/** How finely a curved run is sampled. The reference subdivides adaptively. */
const BEZIER_STEPS = 28;
/** The head is this deep, and juts this far past the ribbon on each side. */
const HEAD_RATIO = 0.5;

interface Ribbons {
  tris: number[];
  lines: number[];
}

function ribbonGeometry(arrows: Arrow[]): {
  fill: THREE.BufferGeometry;
  edge: THREE.BufferGeometry;
} {
  const out: Ribbons = { tris: [], lines: [] };
  for (const a of arrows) drawArrow(out, a);

  const fill = new THREE.BufferGeometry();
  fill.setAttribute("position", new THREE.Float32BufferAttribute(out.tris, 3));
  const edge = new THREE.BufferGeometry();
  edge.setAttribute("position", new THREE.Float32BufferAttribute(out.lines, 3));
  return { fill, edge };
}

const _dir = new THREE.Vector3();
const _side = new THREE.Vector3();
const _normal = new THREE.Vector3();
const _up = new THREE.Vector3(0, 0, 1);

function drawArrow(out: Ribbons, a: Arrow): void {
  const startW = new THREE.Vector3(...a.start);
  const endW = new THREE.Vector3(...a.end);

  // The frame is built from the run's direction flattened into x-y, so `normal`
  // always comes out along z and the ribbon always starts in-plane.
  _dir.subVectors(endW, startW);
  _dir.z = 0;
  if (_dir.lengthSq() < 1e-12) _dir.set(0, 1, 0);
  _dir.normalize();
  _side.crossVectors(_dir, _up).multiplyScalar(-1).normalize();
  _normal.crossVectors(_side, _dir).normalize();
  const mtx = new THREE.Matrix4().makeBasis(_side, _dir, _normal);

  const len = startW.distanceTo(endW);
  const headExtra = a.width * HEAD_RATIO;
  const headDepth = a.head ? Math.min(len * 0.7, headExtra) : 0;

  const start = startW.clone().applyMatrix4(mtx);
  const end = endW.clone().applyMatrix4(mtx);
  const endDir = a.endDir
    ? new THREE.Vector3(...a.endDir).transformDirection(mtx)
    : null;

  const emit = (p: THREE.Vector3): [number, number, number] => {
    const q = p.clone().applyMatrix4(mtx);
    return [q.x, q.y, q.z];
  };
  const tri = (p: THREE.Vector3, q: THREE.Vector3, r: THREE.Vector3): void => {
    out.tris.push(...emit(p), ...emit(q), ...emit(r));
  };
  const line = (p: THREE.Vector3, q: THREE.Vector3): void => {
    out.lines.push(...emit(p), ...emit(q));
  };
  const at = (x: number, y: number, z: number): THREE.Vector3 => new THREE.Vector3(x, y, z);

  /** One flat segment: a quad `width` across, spanning x and y between a and b. */
  const seg = (p: THREE.Vector3, q: THREE.Vector3, border: boolean): void => {
    const half = a.width / 2;
    const tl = at(p.x - half, p.y, p.z);
    const tr = at(q.x + half, p.y, p.z);
    const bl = at(p.x - half, q.y, q.z);
    const br = at(q.x + half, q.y, q.z);
    tri(tl, bl, tr);
    tri(bl, br, tr);
    if (border) {
      line(tl, bl);
      line(tr, br);
    }
  };

  if (endDir || Math.abs(start.z - end.z) > 1e-4) {
    // --- the curved case -----------------------------------------------------
    const dist = Math.max(headDepth, Math.abs(start.y - end.y - headDepth) / 2);
    const p0 = start.clone();
    const p1 = at(start.x, start.y + dist, start.z);
    const p2 = endDir
      ? end.clone().addScaledVector(endDir, -headDepth - dist)
      : at(end.x, end.y - headDepth - dist, end.z);
    const p3 = endDir
      ? end.clone().addScaledVector(endDir, -headDepth)
      : at(end.x, end.y - headDepth, end.z);

    const pts: THREE.Vector3[] = [];
    for (let i = 0; i <= BEZIER_STEPS; i++) {
      const t = i / BEZIER_STEPS;
      const mt = 1 - t;
      const w0 = mt * mt * mt;
      const w1 = 3 * mt * mt * t;
      const w2 = 3 * mt * t * t;
      const w3 = t * t * t;
      pts.push(
        at(
          w0 * p0.x + w1 * p1.x + w2 * p2.x + w3 * p3.x,
          w0 * p0.y + w1 * p1.y + w2 * p2.y + w3 * p3.y,
          w0 * p0.z + w1 * p1.z + w2 * p2.z + w3 * p3.z,
        ),
      );
    }
    for (let i = 0; i < pts.length - 1; i++) seg(pts[i], pts[i + 1], true);
  } else {
    seg(start, at(end.x, end.y - headDepth, end.z), true);
  }

  if (a.corner !== "none") {
    drawArrowCorner(at(start.x, start.y - a.width / 2, start.z), a.corner, a.width, tri, line);
  }

  if (a.head) {
    const from = endDir ? end.clone().addScaledVector(endDir, -headDepth) : at(end.x, end.y - headDepth, end.z);
    const half = a.width / 2;
    const left = at(from.x - half - headExtra, from.y, from.z);
    const right = at(end.x + half + headExtra, from.y, from.z);
    const tip = at(from.x, end.y, end.z);
    tri(left, tip, right);
    line(at(from.x - half, from.y, from.z), left);
    line(left, tip);
    line(tip, right);
    line(right, at(end.x + half, from.y, from.z));
  }
}

/**
 * The rounded inside of a dogleg.
 *
 * Coming in from the side and turning down, the inner corner is a quarter disc
 * swept from the pivot, so the two runs meet as one pathway rather than two
 * rectangles crossing.
 */
function drawArrowCorner(
  centre: THREE.Vector3,
  mode: "left" | "right",
  width: number,
  tri: (p: THREE.Vector3, q: THREE.Vector3, r: THREE.Vector3) => void,
  line: (p: THREE.Vector3, q: THREE.Vector3) => void,
): void {
  const mul = mode === "left" ? 1 : -1;
  const pivot = new THREE.Vector3(centre.x + (width / 2) * mul, centre.y + width / 2, centre.z);
  const count = 8;
  let prev: THREE.Vector3 | null = null;
  for (let i = 0; i < count; i++) {
    const theta = (i / (count - 1)) * (Math.PI / 2);
    const curr = new THREE.Vector3(
      pivot.x - width * Math.cos(theta) * mul,
      pivot.y - width * Math.sin(theta),
      centre.z,
    );
    if (prev) {
      tri(prev, pivot, curr);
      line(prev, curr);
    }
    prev = curr;
  }
}

interface Hover {
  blk: Blk;
  x: number;
  y: number;
  /** The cell under the pointer and its value, on a block a trace covers. */
  cell: { x: number; y: number; value: number; max: number } | null;
}

/** A block's values as the view drew them, kept for the hover readout. */
interface CellValues {
  cells: Float32Array;
  max: number;
}

/** A number the way the readout prints it: three significant figures, a real minus. */
const fmt = (v: number): string =>
  Number.isFinite(v) ? v.toPrecision(3).replace(/^-/, "−") : "masked";

/** A name pinned to a block, in screen pixels. */
interface Tag {
  key: string;
  text: string;
  sub: string;
  x: number;
  y: number;
  kind: string;
}

/**
 * Labels are not permanent.
 *
 * The reference keeps every label at `visible = 0` and raises it only for the
 * group under the mouse. Drawing them all is what turns a drawing into a wall
 * of overlapping text — there are several thousand blocks here and no amount of
 * culling makes that readable. So: hover a block, and its group is named.
 */
const HOVER_DIM = 0.22;

/**
 * What the stage names in the left margin have to stay clear of: the title
 * block across the top, and the legend and hints across the bottom.
 */
const MARGIN_TOP = 48;
const MARGIN_BOTTOM = 132;

interface Ctx {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  group: THREE.Group;
  picks: Map<number, Blk>;
  pickable: THREE.Object3D[];
  raycaster: THREE.Raycaster;
  target: THREE.Vector3;
  spherical: THREE.Spherical;
  /**
   * Whether the camera has been moved since the model was last built.
   *
   * The pane refits on resize, which is right until someone has zoomed in: a
   * status line growing a word, or a dock opening, then throws their view away.
   * So a resize only reframes a camera nobody has touched.
   */
  moved: boolean;
  draw: () => void;
  model: Model3D | null;
  frame: () => void;
  /** Called after each render, so the label layer can re-project. */
  onTags: (() => void) | null;
}

/**
 * Where a cell is, in the block's own axes: `T 3 (B) · C 17`.
 *
 * A position names the symbol that was there, which is what makes the
 * attention matrix readable — row `C` looked at column `A`.
 */
function cellLabel(blk: Blk, cell: { x: number; y: number }, trace: Trace): string {
  const axis = (dim: string, i: number): string | null => {
    if (!dim) return null;
    const letter = dim === "T" ? trace.letters[i] : dim === "n_vocab" ? trace.file.task.symbols[i] : undefined;
    return letter ? `${dim} ${i} (${letter})` : `${dim} ${i}`;
  };
  return [axis(blk.dimX, cell.x), axis(blk.dimY, cell.y)].filter(Boolean).join(" · ") || "value";
}

export default function View3D(): React.ReactElement {
  const doc = useEditor((s) => s.doc);
  const selection = useEditor((s) => s.selection);
  const derived = useDerived();
  const model = useMemo(() => buildModel3D(doc, derived), [doc, derived]);
  /**
   * The design's trace, when it has one that still describes it.
   *
   * Looked up per document rather than per rebuild: the check generates the
   * model and hashes it, which is cheap once and wasteful on every hover.
   */
  const trace = useTrace(doc);

  const mount = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const [tags, setTags] = useState<Tag[]>([]);
  /**
   * The stage names, which unlike the tags are always up.
   *
   * Half a dozen of them against several thousand tags, so the reason the tags
   * are hover-only does not apply: a still of the tower has to say which slab
   * is which, and hovering is not something a screenshot does.
   */
  const [marks, setMarks] = useState<Tag[]>([]);
  /** The group under the mouse, which is the only thing that gets named. */
  const hoverGroup = useRef<string | null>(null);
  const [theme, setTheme] = useState(resolvedTheme);
  useEffect(() => onThemeChange(setTheme), []);

  const gl = useRef<Ctx | null>(null);

  // --- set up once ---------------------------------------------------------
  useEffect(() => {
    const host = mount.current;
    if (!host) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(renderer.domElement);
    Object.assign(renderer.domElement.style, { display: "block", width: "100%", height: "100%" });

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 20000);
    const group = new THREE.Group();
    scene.add(group);

    const target = new THREE.Vector3();
    const spherical = new THREE.Spherical(200, 1.35, 0.55);

    let queued = false;
    const draw = (): void => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        camera.position.setFromSpherical(spherical).add(target);
        camera.lookAt(target);
        renderer.render(scene, camera);
        gl.current?.onTags?.();
      });
    };

    /** Pull the camera back until the drawn extent fits at the current angle. */
    const frame = (): void => {
      const m = gl.current?.model;
      if (!m) return;
      const vFov = (camera.fov * Math.PI) / 180;
      const aspect = Math.max(0.35, camera.aspect || 1);
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);
      const tanH = Math.tan(hFov / 2);
      const tanV = Math.tan(vFov / 2);

      const eye = new THREE.Vector3().setFromSpherical(spherical);
      const forward = eye.clone().negate().normalize();
      const right = new THREE.Vector3()
        .crossVectors(forward, new THREE.Vector3(0, 1, 0))
        .normalize();
      const up = new THREE.Vector3().crossVectors(right, forward).normalize();

      const { min, max, centre } = m.bounds;
      let distance = 0;
      for (let i = 0; i < 8; i++) {
        const corner = new THREE.Vector3(
          (i & 1 ? max[0] : min[0]) - centre[0],
          (i & 2 ? max[1] : min[1]) - centre[1],
          (i & 4 ? max[2] : min[2]) - centre[2],
        );
        const along = corner.dot(forward);
        distance = Math.max(
          distance,
          Math.abs(corner.dot(right)) / tanH + along,
          Math.abs(corner.dot(up)) / tanV + along,
        );
      }
      spherical.radius = Math.max(4, distance * 1.06);
      camera.near = Math.max(0.05, spherical.radius / 4000);
      camera.far = spherical.radius * 8;
      camera.updateProjectionMatrix();
    };

    const resize = (): void => {
      const { clientWidth: w, clientHeight: h } = host;
      if (w === 0 || h === 0) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      if (!gl.current?.moved) frame();
      draw();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);

    gl.current = {
      renderer,
      scene,
      camera,
      group,
      picks: new Map(),
      pickable: [],
      raycaster: new THREE.Raycaster(),
      target,
      spherical,
      moved: false,
      draw,
      model: null,
      frame,
      onTags: null,
    };
    resize();

    return () => {
      observer.disconnect();
      renderer.dispose();
      host.removeChild(renderer.domElement);
      gl.current = null;
    };
  }, []);

  // --- geometry ------------------------------------------------------------
  const values = useRef(new Map<Blk, CellValues>()).current;
  /** How many blocks drew real values, as state so the legend follows a rebuild. */
  const [covered, setCovered] = useState(0);
  const rebuild = useCallback((m: Model3D, trace: Trace | null) => {
    const ctx = gl.current;
    if (!ctx) return;
    const { group, picks } = ctx;

    for (const child of [...group.children]) {
      group.remove(child);
      if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments) {
        // The unit box is shared and recreated each rebuild; only the ribbon
        // geometries are owned per object.
        if (child.userData.owned) child.geometry.dispose();
        (Array.isArray(child.material) ? child.material : [child.material]).forEach((mat) => {
          // A material does not own the textures in its uniforms.
          if (mat instanceof THREE.ShaderMaterial) mat.uniforms.uData?.value?.dispose();
          mat.dispose();
        });
      }
    }
    picks.clear();
    values.clear();
    ctx.pickable = [];

    const sheet = new THREE.Color(themeValue("--sheet", "#f7f6f2"));
    ctx.scene.background = sheet;

    const colours: Record<string, THREE.Color> = {
      w: new THREE.Color(themeValue("--vol-weight", "#7d86ad")),
      i: new THREE.Color(themeValue("--vol-activation", "#78976f")),
      a: new THREE.Color(themeValue("--vol-aggregate", "#9aa0a8")),
    };
    const grid = new THREE.Color(themeValue("--vol-face", "#ffffff"));
    const edge = new THREE.Color(themeValue("--vol-edge", "#ffffff"));
    const box = new THREE.BoxGeometry(1, 1, 1);

    for (const b of m.blocks) {
      const cells = trace && b.source ? cellsFor(trace, b.source, b.cx, b.cy) : null;
      const data = cells ? dataTexture(cells, b.cx, b.cy) : null;
      const material = new THREE.ShaderMaterial({
        vertexShader: CELL_VERT,
        fragmentShader: CELL_FRAG,
        uniforms: {
          uColor: { value: colours[b.kind] },
          uGrid: { value: grid },
          uEdge: { value: edge },
          uCells: { value: new THREE.Vector3(b.cx, b.cy, b.cz) },
          uHighlight: { value: 0 },
          uHover: { value: 0 },
          uData: { value: data?.tex ?? null },
          uHasData: { value: data ? 1 : 0 },
        },
      });
      if (data) values.set(b, { cells: cells!, max: data.max });
      const mesh = new THREE.Mesh(box, material);
      // The layout's y is positive downward; three's is up.
      mesh.position.set(b.x + b.dx / 2, -(b.y + b.dy / 2), b.z + b.dz / 2);
      mesh.scale.set(b.dx, b.dy, b.dz);
      mesh.userData.path = b.path;
      group.add(mesh);
      picks.set(mesh.id, b);
      ctx.pickable.push(mesh);
    }

    setCovered(values.size);

    // Flow ribbons: blue out of a weight, green out of a value, as the
    // reference colours them. Drawn without depth writes so they layer over the
    // plates instead of z-fighting the faces they start and end on.
    for (const kind of ["w", "i"] as const) {
      const subset = m.arrows.filter((a) => a.kind === kind);
      if (subset.length === 0) continue;
      const base = new THREE.Color(
        kind === "w"
          ? themeValue("--vol-arrow-weight", "#3333aa")
          : themeValue("--vol-arrow-data", "#33aa33"),
      );
      const { fill, edge } = ribbonGeometry(subset);

      const body = new THREE.Mesh(
        fill,
        new THREE.MeshBasicMaterial({
          color: base,
          transparent: true,
          opacity: 0.3,
          depthWrite: false,
          side: THREE.DoubleSide,
        }),
      );
      const border = new THREE.LineSegments(
        edge,
        new THREE.LineBasicMaterial({
          color: base.clone().multiplyScalar(0.8),
          transparent: true,
          opacity: 0.85,
          depthWrite: false,
        }),
      );
      for (const obj of [body, border]) {
        // The layout's y is positive downward; three's is up.
        obj.scale.set(1, -1, 1);
        obj.userData.owned = true;
        group.add(obj);
      }
    }

    group.position.set(-m.bounds.centre[0], -m.bounds.centre[1], -m.bounds.centre[2]);
    ctx.target.set(0, 0, 0);
    // Slightly above the horizon and a little off-axis: the tower is tall and
    // narrow, so a steep angle just foreshortens it into nothing.
    ctx.spherical.phi = 1.35;
    ctx.spherical.theta = 0.55;
    ctx.model = m;
    ctx.moved = false;
    ctx.frame();
    ctx.draw();
  }, []);

  /**
   * Project the hovered group's names to screen. Re-run after every render, so
   * the labels stay on their blocks while the camera moves.
   */
  const projectTags = useCallback(() => {
    const ctx = gl.current;
    const host = mount.current;
    const key = hoverGroup.current;
    if (!ctx || !host || !ctx.model || !key) {
      setTags((held) => (held.length === 0 ? held : []));
      return;
    }
    const { clientWidth: w, clientHeight: h } = host;
    const centre = new THREE.Vector3();
    const next: Tag[] = [];

    for (const b of ctx.model.blocks) {
      if (b.group !== key || !b.name) continue;
      centre
        .set(b.x + b.dx / 2, -(b.y + b.dy / 2), b.z + b.dz / 2)
        .add(ctx.group.position)
        .project(ctx.camera);
      if (centre.z < -1 || centre.z > 1) continue;
      next.push({
        key: `${b.name}:${b.layer}:${Math.round(b.x)}:${Math.round(b.y)}`,
        text: b.name,
        sub: [b.dimX, b.dimY].filter(Boolean).join(" \u00d7 "),
        x: ((centre.x + 1) / 2) * w,
        y: ((1 - centre.y) / 2) * h,
        kind: b.kind,
      });
    }
    setTags(next);
  }, []);

  /** The same projection for the stage names, which do not depend on a hover. */
  const projectMarks = useCallback(() => {
    const ctx = gl.current;
    const host = mount.current;
    if (!ctx || !host || !ctx.model) {
      setMarks((held) => (held.length === 0 ? held : []));
      return;
    }
    const { clientWidth: w, clientHeight: h } = host;
    const at = new THREE.Vector3();
    const next: Tag[] = [];
    const taken: number[] = [];
    for (const m of ctx.model.landmarks) {
      at.set(m.x, -m.y, m.z).add(ctx.group.position).project(ctx.camera);
      if (at.z < -1 || at.z > 1) continue;
      const y = ((1 - at.y) / 2) * h;
      // The margin is not the whole height: the title sits across the top and
      // the legend across the bottom left, and a stage name printed over the
      // legend is two things in one place. No room, no label — which is
      // honest, and orbiting a little brings it back.
      if (y < MARGIN_TOP || y > h - MARGIN_BOTTOM) continue;
      // Two stages at the same height in this projection would print over each
      // other, and two names in one place is worse than one name.
      if (taken.some((other) => Math.abs(other - y) < 16)) continue;
      taken.push(y);
      next.push({
        key: `mark:${m.path}`,
        text: m.text,
        sub: "",
        // In the margin, at the stage's height. A schematic puts its row names
        // down the side rather than beside whatever sticks out furthest, and
        // a point pinned in the model would swing across the picture as the
        // camera orbits.
        x: 12,
        y,
        kind: "mark",
      });
    }
    setMarks(next);
  }, []);

  useEffect(() => {
    if (gl.current) {
      gl.current.onTags = () => {
        projectTags();
        projectMarks();
      };
    }
    return () => {
      if (gl.current) gl.current.onTags = null;
    };
  }, [projectTags, projectMarks]);

  useEffect(() => {
    rebuild(model, trace);
  }, [model, trace, rebuild, theme]);

  // --- selection highlight -------------------------------------------------
  useEffect(() => {
    const ctx = gl.current;
    if (!ctx) return;
    for (const mesh of ctx.pickable) {
      const path = mesh.userData.path as string | null;
      const on = Boolean(
        selection && path && (path === selection || path.startsWith(`${selection}/`)),
      );
      const material = (mesh as THREE.Mesh).material as THREE.ShaderMaterial;
      material.uniforms.uHighlight.value = on ? 1 : 0;
    }
    ctx.draw();
  }, [selection]);

  // --- interaction ---------------------------------------------------------
  const pointer = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  const hitAt = useCallback((clientX: number, clientY: number): THREE.Intersection | null => {
    const ctx = gl.current;
    const host = mount.current;
    if (!ctx || !host) return null;
    const rect = host.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    ctx.raycaster.setFromCamera(ndc, ctx.camera);
    return ctx.raycaster.intersectObjects(ctx.pickable, false)[0] ?? null;
  }, []);

  const pick = useCallback(
    (clientX: number, clientY: number): Blk | null => {
      const hit = hitAt(clientX, clientY);
      return hit ? (gl.current?.picks.get(hit.object.id) ?? null) : null;
    },
    [hitAt],
  );

  /**
   * The cell under the pointer, on the face that carries values.
   *
   * The same arithmetic as the shader, run backwards: the hit in the box's own
   * unit frame, x across from the left and rows down from the top.
   */
  const cellAt = (hit: THREE.Intersection, blk: Blk): Hover["cell"] => {
    const held = values.get(blk);
    if (!held || !hit.face || Math.abs(hit.face.normal.z) < 0.5) return null;
    const local = hit.object.worldToLocal(hit.point.clone()).addScalar(0.5);
    const x = Math.min(blk.cx - 1, Math.max(0, Math.floor(local.x * blk.cx)));
    const y = Math.min(blk.cy - 1, Math.max(0, Math.floor((1 - local.y) * blk.cy)));
    return { x, y, value: held.cells[y * blk.cx + x]!, max: held.max };
  };

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>): void => {
    e.currentTarget.setPointerCapture(e.pointerId);
    pointer.current = { x: e.clientX, y: e.clientY, moved: false };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>): void => {
    const ctx = gl.current;
    if (!ctx) return;
    const held = pointer.current;

    if (held) {
      const dx = e.clientX - held.x;
      const dy = e.clientY - held.y;
      held.x = e.clientX;
      held.y = e.clientY;
      if (Math.abs(dx) + Math.abs(dy) > 2) held.moved = true;
      ctx.moved = true;

      if (e.shiftKey || e.buttons === 4) {
        const right = new THREE.Vector3();
        const up = new THREE.Vector3();
        ctx.camera.matrixWorld.extractBasis(right, up, new THREE.Vector3());
        const scale = ctx.spherical.radius * 0.0016;
        ctx.target.addScaledVector(right, -dx * scale);
        ctx.target.addScaledVector(up, dy * scale);
      } else {
        ctx.spherical.theta -= dx * 0.006;
        ctx.spherical.phi = Math.min(
          Math.PI - 0.05,
          Math.max(0.05, ctx.spherical.phi - dy * 0.006),
        );
      }
      ctx.draw();
      return;
    }

    const hit = hitAt(e.clientX, e.clientY);
    const blk = hit ? (ctx.picks.get(hit.object.id) ?? null) : null;
    setHover(blk && hit ? { blk, x: e.clientX, y: e.clientY, cell: cellAt(hit, blk) } : null);

    const key = blk?.group ?? null;
    if (key !== hoverGroup.current) {
      hoverGroup.current = key;
      // Light the whole group, not just the box the pointer happens to be on.
      for (const mesh of ctx.pickable) {
        const b = ctx.picks.get(mesh.id);
        const material = (mesh as THREE.Mesh).material as THREE.ShaderMaterial;
        material.uniforms.uHover.value = b && key && b.group === key ? HOVER_DIM : 0;
      }
      ctx.draw();
    } else {
      projectTags();
    }
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>): void => {
    const held = pointer.current;
    pointer.current = null;
    e.currentTarget.releasePointerCapture(e.pointerId);
    if (held && !held.moved) {
      const blk = pick(e.clientX, e.clientY);
      if (blk?.path) useEditor.getState().focusOn(blk.path);
      else useEditor.getState().select(null);
    }
  };

  /**
   * Fit and zoom, so the view commands mean something here too.
   *
   * The sheet hands these up from React Flow; the volume view is the other half
   * of the same switch, and without this `F` and the zoom shortcuts would go
   * nowhere the moment someone turned the model round.
   */
  useEffect(() => {
    const scale = (by: number): void => {
      const ctx = gl.current;
      if (!ctx) return;
      ctx.moved = true;
      ctx.spherical.radius = Math.min(40000, Math.max(0.5, ctx.spherical.radius * by));
      ctx.camera.near = Math.max(0.05, ctx.spherical.radius / 4000);
      ctx.camera.far = ctx.spherical.radius * 8;
      ctx.camera.updateProjectionMatrix();
      ctx.draw();
    };
    const fit = (): void => {
      const ctx = gl.current;
      if (!ctx) return;
      ctx.moved = false;
      ctx.target.set(0, 0, 0);
      ctx.frame();
      ctx.draw();
    };
    setViewportApi({
      fit,
      zoomReset: fit,
      zoomIn: () => scale(0.8),
      zoomOut: () => scale(1.25),
      // Nothing is being edited in here, so there is nothing to duplicate.
      duplicateSelection: () => {},
    });
    return () => setViewportApi(null);
  }, []);

  const onWheel = (e: React.WheelEvent<HTMLDivElement>): void => {
    const ctx = gl.current;
    if (!ctx) return;
    ctx.moved = true;
    ctx.spherical.radius = Math.min(
      40000,
      Math.max(0.5, ctx.spherical.radius * Math.exp(e.deltaY * 0.0011)),
    );
    ctx.camera.near = Math.max(0.05, ctx.spherical.radius / 4000);
    ctx.camera.far = ctx.spherical.radius * 8;
    ctx.camera.updateProjectionMatrix();
    ctx.draw();
  };

  const { shape } = model;

  return (
    <div className="relative min-h-0 flex-1 overflow-hidden">
      <div
        ref={mount}
        className="absolute inset-0 cursor-grab active:cursor-grabbing"
        // A canvas is one opaque element to a screen reader. What it shows is
        // what the sheet says in words, so this says what the picture adds.
        role="img"
        aria-label={`Volume view of ${model.name}: ${model.blocksDrawn} blocks of ${shape.nHeads} attention heads, width ${shape.C}, drawn at their real proportions${trace && covered > 0 ? `, with the real values of a run: ${trace.summary}` : ""}.`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => {
          setHover(null);
          hoverGroup.current = null;
          const ctx = gl.current;
          if (!ctx) return;
          for (const mesh of ctx.pickable) {
            ((mesh as THREE.Mesh).material as THREE.ShaderMaterial).uniforms.uHover.value = 0;
          }
          ctx.draw();
        }}
        onWheel={onWheel}
      />

      {/* Names pinned to the blocks, the way the reference labels every tensor.
          A drawing of unlabelled boxes is a texture, not a diagram. */}
      {/* The stage names, under the hover tags so a tag that lands on one is
          the thing you asked for rather than the thing that was already there. */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        {marks.map((mark) => (
          <div
            key={mark.key}
            className="absolute -translate-y-1/2 whitespace-nowrap font-mono text-[10px] leading-tight tracking-wide text-dim"
            style={{ left: mark.x, top: mark.y }}
          >
            {mark.text}
          </div>
        ))}
      </div>

      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        {tags.map((tag) => (
          <div
            key={tag.key}
            className="absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap px-1.5 py-px text-[9.5px] leading-tight font-medium text-white shadow-sm"
            style={{
              left: tag.x,
              top: tag.y,
              background:
                tag.kind === "w"
                  ? "rgba(26, 28, 46, 0.86)"
                  : tag.kind === "i"
                    ? "rgba(20, 34, 22, 0.86)"
                    : "rgba(32, 34, 38, 0.82)",
            }}
          >
            {tag.text}
            {tag.sub && <span className="ml-1.5 opacity-60">{tag.sub}</span>}
          </div>
        ))}
      </div>

      <div className="pointer-events-none absolute top-3 left-1/2 -translate-x-1/2 border border-border bg-popover/90 px-3 py-1.5 text-center">
        <div className="font-sans text-sm font-semibold text-foreground">{model.name}</div>
        <div className="font-mono text-[10px] text-dim">
          n_params = {formatCount(model.totalParams)} · {model.blocksDrawn} blocks ·{" "}
          {shape.nHeads} heads · C {shape.C.toLocaleString("en-US")}
          {model.columns > 1 && ` · ${model.columns} columns`}
        </div>
      </div>

      <div className="pointer-events-none absolute bottom-3 left-3 max-w-[26rem] space-y-0.5 font-mono text-[10px] leading-relaxed text-dim">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="inline-block size-2" style={{ background: "var(--vol-weight)" }} />
            weights
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block size-2" style={{ background: "var(--vol-activation)" }} />
            activations
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block size-2" style={{ background: "var(--vol-aggregate)" }} />
            aggregates
          </span>
        </div>
        <div>drag to orbit · shift-drag to pan · wheel to zoom · click a tensor to select it</div>
        {trace && covered > 0 ? (
          <div className="text-foreground">
            real values: {trace.summary} · brighter is larger
            {covered < model.blocks.length && " · speckled cells are decoration"}
          </div>
        ) : trace && shape.T > trace.positions ? (
          // A trace of this design is loaded, but it ran fewer positions than
          // the drawing has. The model is causal, so a shorter drawing is
          // exactly the start of the run; a longer one has positions nobody
          // computed, and drawing none rather than some is the honest default.
          <div className="pointer-events-auto text-foreground">
            a trace of this design covers {trace.positions} positions and this is drawn at{" "}
            {shape.T.toLocaleString("en-US")} ·{" "}
            <button
              type="button"
              className="underline underline-offset-2 hover:text-primary"
              onClick={() => useEditor.getState().setOperating({ T: trace.positions })}
            >
              draw it at {trace.positions}
            </button>
          </div>
        ) : (
          <div>cell shading is decoration: a design has shapes, not weights</div>
        )}
        {model.notes.map((n) => (
          <div key={n} className="text-warn">
            {n}
          </div>
        ))}
        <div className="opacity-70">
          layout ported from Brendan Bycroft&rsquo;s LLM visualisation (MIT)
        </div>
      </div>

      {hover && (
        <div
          className="pointer-events-none fixed z-50 border border-border bg-popover px-2 py-1 font-mono text-[10.5px] text-foreground shadow-lg"
          style={{ left: hover.x + 14, top: hover.y + 14 }}
        >
          <div className="font-sans font-medium">{describeBlk(hover.blk)}</div>
          <div className="text-dim">
            {[hover.blk.dimX, hover.blk.dimY].filter(Boolean).join(" × ") || " "}
            {hover.blk.layer >= 0 && ` · block ${hover.blk.layer + 1}`}
          </div>
          {hover.cell && trace && (
            <div>
              {cellLabel(hover.blk, hover.cell, trace)} = <b>{fmt(hover.cell.value)}</b>
              <span className="text-dim"> · largest here {fmt(hover.cell.max)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
