/**
 * The inspector for a selected net.
 *
 * A tensor has always been a thing in this design rather than a line between
 * two things: it has a shape, a dtype, one producer, however many consumers,
 * and a share of the activation memory that the analysis attributes to *it*
 * and not to a block. The drawing was the last place it was not selectable, so
 * the only way to ask what a wire cost was to guess which block to click.
 *
 * The memory figure is the one that could not be had any other way. A block
 * that fans out — `split`, which is how a selective scan gets Δ, B and C from
 * one projection — is a single row in the per-block breakdown covering three
 * tensors of wildly different sizes. Nemotron-H's holds 2 MiB and 167 MiB on
 * two of its three output pins, and the block's own number says neither.
 */

import { useEditor } from "../state/store.js";
import { useDerived, useLevel } from "../state/hooks.js";
import { formatShape } from "../canvas/shapes.js";
import { CATALOG } from "../engine.js";

const gib = (bytes: number): string =>
  bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(2)} GiB`
    : bytes >= 1024 ** 2
      ? `${(bytes / 1024 ** 2).toFixed(1)} MiB`
      : `${(bytes / 1024).toFixed(1)} KiB`;

function Row({ label, children }: { label: string; children: React.ReactNode }): React.ReactElement {
  return (
    <div className="tensor__row">
      <span className="tensor__key">{label}</span>
      <span className="tensor__value">{children}</span>
    </div>
  );
}

export default function Tensor(): React.ReactElement {
  const net = useEditor((s) => s.selectedNet);
  const select = useEditor((s) => s.select);
  const shapeMode = useEditor((s) => s.shapeMode);
  const derived = useDerived();
  const { derived: levelDerived } = useLevel();

  if (!net) {
    return (
      <div className="panel__body">
        <div className="empty">Select a net on the canvas to see what it carries.</div>
      </div>
    );
  }

  const at = net.lastIndexOf(":");
  const producer = at > 0 ? net.slice(0, at) : net;
  const port = at > 0 ? net.slice(at + 1) : "";

  // The level's inference knows the shape of what is drawn; the design's knows
  // the rest. Asking the level first is what makes this work inside a
  // definition, where the design's numbers are about a different graph.
  const shape = levelDerived.infer.outputs.get(net) ?? derived.infer.outputs.get(net);

  const consumers: string[] = [];
  for (const [consumer, from] of derived.infer.producerOf) {
    if (from === net) consumers.push(consumer);
  }
  consumers.sort();

  const activations = derived.analysis.memory.train.activationsByTensor?.[net] ?? 0;
  const blockTotal = derived.analysis.memory.train.activationsByPath?.[producer] ?? 0;
  const resolved = derived.infer.resolved.get(producer);
  const def = resolved ? CATALOG[resolved.type] : undefined;
  const dtype = derived.infer.ports.get(producer)?.out?.[port]?.dtype;

  return (
    <div className="panel__body tensor">
      <div className="tensor__head">
        <code className="tensor__name">{net}</code>
        {shape && <span className="tensor__shape">{formatShape(shape, shapeMode) ?? "?"}</span>}
      </div>

      <Row label="from">
        <button type="button" className="tensor__link" onClick={() => select(producer)}>
          {producer}
        </button>
        {def && <span className="tensor__type"> · {resolved?.type}</span>}
      </Row>

      <Row label="to">
        {consumers.length === 0 ? (
          // Not an error. A tensor nobody reads is a real thing to have drawn
          // and a real thing to be told about, which is why it says so here
          // rather than leaving the row blank.
          <span className="tensor__none">nothing reads it</span>
        ) : (
          <span className="tensor__consumers">
            {consumers.map((c) => {
              const dot = c.lastIndexOf(":");
              const path = dot > 0 ? c.slice(0, dot) : c;
              return (
                <button key={c} type="button" className="tensor__link" onClick={() => select(path)}>
                  {c}
                </button>
              );
            })}
          </span>
        )}
      </Row>

      {dtype && <Row label="dtype">{dtype}</Row>}

      <Row label="activations">
        {activations > 0 ? (
          <>
            {gib(activations)}
            {/* Only worth saying when the block holds more than this one net;
                otherwise it is the same number twice. */}
            {blockTotal > activations * 1.0001 && (
              <span className="tensor__aside"> of {gib(blockTotal)} held by {producer.split("/").pop()}</span>
            )}
          </>
        ) : (
          <span className="tensor__none">
            not retained — nothing downstream needs it kept for the backward pass
          </span>
        )}
      </Row>
    </div>
  );
}
