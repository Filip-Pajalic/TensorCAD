/**
 * An attention's mask, drawn beside the expression that makes it.
 *
 * What is drawn is what FlexAttention's block mask sees: the sequence cut into
 * blocks along both sides, each shaded by the share of its scores the mask
 * keeps. An empty block is one the kernel skips, a full one is computed without
 * the mask, and one in between is computed and then masked — which is why a
 * mask that keeps a sliver of every block costs more than its density says.
 *
 * The grid comes from the engine at the operating point's sequence length, so
 * the picture and the FLOP count beside it are the same measurement.
 */

import { useMemo, useState } from "react";
import type { MaskView } from "@tensor-cad/engine";
import { useEditor } from "../state/store.js";
import { toAnalysisOptions } from "../state/operating.js";
import { engine } from "../engine.js";

const SIZE = 160;

function percent(x: number): string {
  if (x >= 0.995 || x === 0) return `${Math.round(x * 100)}%`;
  return x < 0.01 ? `${(x * 100).toPrecision(2)}%` : `${(x * 100).toFixed(1)}%`;
}

export default function MaskPreview({
  path,
  heads,
}: {
  path: string;
  /** How many heads a mask that reads `h` can be stepped through. */
  heads: number;
}): React.ReactElement | null {
  const doc = useEditor((s) => s.doc);
  const operating = useEditor((s) => s.operating);
  const [head, setHead] = useState(0);

  const view: MaskView | null = useMemo(() => {
    try {
      return engine().attentionMask(doc, path, toAnalysisOptions(operating), head);
    } catch {
      return null;
    }
  }, [doc, operating, path, head]);

  if (!view || !view.found || view.cells === 0) return null;
  const cell = SIZE / view.cells;

  return (
    <figure className="mask" data-testid="mask-preview">
      <svg
        className="mask__grid"
        width={SIZE}
        height={SIZE}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        // Blocks that touch should read as one region, not as tiles with a
        // hairline between each: antialiasing draws one at every shared edge.
        shapeRendering="crispEdges"
        role="img"
        aria-label={`Attention mask at ${view.T} positions: keeps ${percent(view.density)} of the scores`}
      >
        <rect className="mask__paper" x={0} y={0} width={SIZE} height={SIZE} />
        {view.kept.map((kept, i) =>
          kept > 0 ? (
            <rect
              key={i}
              className={kept >= 1 ? "mask__full" : "mask__partial"}
              x={(i % view.cells) * cell}
              y={Math.floor(i / view.cells) * cell}
              width={cell}
              height={cell}
              fillOpacity={kept >= 1 ? 1 : 0.25 + 0.6 * kept}
            />
          ) : null,
        )}
      </svg>
      <figcaption className="mask__caption">
        <div>
          <span className="mono">{view.T.toLocaleString("en-US")}</span> positions,{" "}
          <span className="mono">{Number(view.span.toPrecision(4)).toLocaleString("en-US")}</span> a
          block; keys across, queries down
        </div>
        <div>
          keeps <span className="mono" data-testid="mask-density">{percent(view.density)}</span> of the scores
        </div>
        {view.perHead && heads > 1 && (
          <div className="mask__head">
            <button
              className="btn btn--icon"
              disabled={view.head <= 0}
              onClick={() => setHead(Math.max(0, view.head - 1))}
              aria-label="previous head"
            >
              ‹
            </button>
            <span>
              head <span className="mono">{view.head}</span> of {heads}
            </span>
            <button
              className="btn btn--icon"
              disabled={view.head >= heads - 1}
              onClick={() => setHead(Math.min(heads - 1, view.head + 1))}
              aria-label="next head"
            >
              ›
            </button>
          </div>
        )}
        {view.mask && (
          <div className="mask__expr mono" title="Every condition a score has to meet, causal and window included">
            {view.mask}
          </div>
        )}
        {view.score && (
          <div className="mask__expr mono" title="What each score becomes, the cap included">
            {view.score}
          </div>
        )}
      </figcaption>
    </figure>
  );
}
