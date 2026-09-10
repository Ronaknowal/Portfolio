import { useId } from 'react';

function ObjectGraph({ copied }) {
  const arrowId = useId();
  const suffix = copied ? '′' : '';
  const points = [{ value: 'A', x: 60, y: 70 }, { value: 'B', x: 210, y: 55 }, { value: 'C', x: 210, y: 170 }];
  return <div style={{ flex: '1 1 260px', minWidth: 0 }}>
    <p><strong>{copied ? 'New objects' : 'Original objects'}</strong></p>
    <svg viewBox="0 0 300 235" style={{ width: '100%', maxWidth: 360, display: 'block' }} role="img" aria-label={`${copied ? 'Copied' : 'Original'} graph: A points to B and C, B also points to C, and C points back to A. All edges stay within this graph.`}>
      <defs><marker id={arrowId} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>
      <g fill="none" stroke="currentColor" strokeWidth="1.8" opacity="0.75" markerEnd={`url(#${arrowId})`}>
        <path d="M 84 67 L 184 57"/>
        <path d="M 210 81 L 210 143"/>
        <path d="M 79 87 L 187 153"/>
        <path d="M 187 182 C 95 245, 0 180, 44 92"/>
      </g>
      {points.map(point => <g key={point.value}>
        <circle cx={point.x} cy={point.y} r="25" fill="var(--bg, #090b0d)" stroke="currentColor" strokeWidth="1.5"/>
        <text x={point.x} y={point.y + 7} textAnchor="middle" fill="currentColor" fontSize="22">{point.value}{suffix}</text>
      </g>)}
    </svg>
  </div>;
}

export default function GraphCloneFigure() {
  return <figure style={{ margin: '1.75rem 0', paddingBlock: '1rem', borderBlock: '1px solid currentColor' }}>
    <figcaption><strong>Preserve sharing and cycles; replace object identity</strong></figcaption>
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}><ObjectGraph copied={false}/><ObjectGraph copied/></div>
    <p>A → A′, B → B′ and C → C′ are entries in the copy map. They are not extra graph edges. Both copied references to C′ reach the same new object; C′ points back to A′. Values are copied unchanged—the prime marks new identity for this diagram.</p>
  </figure>;
}
