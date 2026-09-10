import { useId } from "react";
import { referenceTrace } from "../../data/python-foundations-model";
import "./python-mechanism-figures.css";

function ArrowMarker({ id }) {
  return <defs><marker id={id} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="currentColor" /></marker></defs>;
}

export function ReferenceSetupFigure() {
  const unique = useId().replace(/:/g, "");
  return <figure className="pmf-figure" data-python-figure="reference-setup">
    <figcaption><strong>Two names can lead to one list—or to two lists.</strong><span>Both setups start with readings = [18, 21]. Follow the arrows before changing anything.</span></figcaption>
    <div className="pmf-reference-comparison">{[false, true].map(copy => {
      const state = referenceTrace(copy).states[2], marker = `pmf-reference-${unique}-${copy}`;
      return <div key={String(copy)}><code>{copy ? "backup = readings.copy()" : "backup = readings"}</code>
        <svg viewBox="0 0 320 168" role="img" aria-label={copy ? "readings refers to list A, [18, 21]. backup refers to separate list B, [18, 21]." : "readings and backup both refer to the same list A, [18, 21]."}>
          <ArrowMarker id={marker} />
          {["readings", "backup"].map((name, i) => <g key={name}><text x="5" y={48 + i * 84}>{name}</text><path className="pmf-arrow" d={`M84,${44 + i * 84} C122,${44 + i * 84} 122,${copy ? 44 + i * 84 : 86} 160,${copy ? 44 + i * 84 : 86}`} markerEnd={`url(#${marker})`} /></g>)}
          {Object.entries(state.objects).map(([id, values], i) => <g key={id}><rect className="pmf-object" x="168" y={copy ? 15 + i * 84 : 57} width="147" height="58" rx="3"/><text className="pmf-svg-label" x="179" y={copy ? 32 + i * 84 : 74}>LIST {id}</text>{values.map((v, j) => <g key={j}><rect className="pmf-cell" x={179 + j * 64} y={copy ? 40 + i * 84 : 82} width="54" height="23"/><text x={206 + j * 64} y={copy ? 57 + i * 84 : 99} textAnchor="middle">{v}</text></g>)}</g>)}
        </svg><p>{copy ? "Equal contents; separate outer containers." : "One container; two ways to reach it."}</p>
      </div>;
    })}</div>
    <p className="pmf-caption">An arrow means “refers to”; A and B are identity labels, not addresses. These lists contain numbers. A shallow copy can still share nested mutable objects, as the deeper example shows.</p>
  </figure>;
}
