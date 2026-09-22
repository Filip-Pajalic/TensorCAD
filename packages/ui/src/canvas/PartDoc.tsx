/**
 * What a part says about itself on hover.
 *
 * The catalog has carried a summary for every block since M6, and a formula for
 * every primitive that counts parameters. Until now the only way to read one was
 * to select the block and look at the inspector, which is the wrong shape for
 * the question: "what is this box" is asked while the pointer is already on it,
 * about a box you have not decided to care about yet.
 *
 * Sources are deliberately not here. A tooltip closes when the pointer leaves
 * it, so a link in one is a link that cannot be clicked; the inspector has them
 * and is where a reader who wants the paper has already arrived.
 */

export default function PartDoc({
  name,
  type,
  docs,
  drillable,
}: {
  name: string;
  /** The identifier, which is what has to be typed and is not what is drawn. */
  type: string;
  docs: { summary?: string; formula?: string };
  drillable?: boolean;
}): React.ReactElement | null {
  // Nothing to say beyond the name already on the part, and a tooltip that
  // repeats the label is worse than none.
  if (!docs.summary && !docs.formula && !drillable) return null;

  return (
    <div className="partdoc">
      <div className="partdoc__head">
        <span className="partdoc__name">{name}</span>
        {name !== type && <span className="partdoc__id mono">{type}</span>}
      </div>
      {docs.summary && <p className="partdoc__summary">{docs.summary}</p>}
      {docs.formula && <pre className="partdoc__formula mono">{docs.formula}</pre>}
      {drillable && <p className="partdoc__hint">Double-click to open this block.</p>}
    </div>
  );
}
