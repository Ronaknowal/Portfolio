import './clustering-evaluation-intuition.css';

export function ClusteringDistanceSummaryFigure() {
  const x = value => 25 + value * 29;
  return <figure className="ce-intuition" data-figure="distance-summary-contrast"><svg viewBox="0 0 325 240" role="img" aria-label="The same six line locations 0,1,2,7,8,9. Center separation is seven, nearest cross-group separation five, and largest within-group diameter two.">
    <text x="20" y="18">Same data, different summaries</text>
    <path d="M25 56H286" stroke="#666" />
    {[0, 1, 2, 7, 8, 9].map(value => <g key={value}><circle cx={x(value)} cy="56" r="5" fill="#ddd" /><text x={x(value)} y="79" textAnchor="middle">{value}</text></g>)}
    {[
      { a: 1, b: 8, y: 106, label: 'DB: centers are 7 apart' },
      { a: 2, b: 7, y: 153, label: 'Dunn: closest across = 5' },
      { a: 0, b: 2, y: 200, label: 'Dunn: widest inside = 2' },
    ].map(({ a, b, y, label }) => <g key={label}><path d={`M${x(a)} ${y-5}V${y+5}M${x(a)} ${y}H${x(b)}M${x(b)} ${y-5}V${y+5}`} stroke="#e8b44a" strokeWidth="2" /><text x="20" y={y+24}>{label}</text></g>)}
  </svg><figcaption>DB combines the average distance to each center (2/3 on either side) with center separation 7: (2/3+2/3)/7=4/21. Dunn instead uses the closest cross-group pair and the widest within-group pair: 5/2. Neither is the silhouette's average distance from a selected observation to all members of a group.</figcaption></figure>;
}

export function CopheneticJoinFigure() {
  return <figure className="ce-intuition" data-figure="cophenetic-first-join"><svg viewBox="0 0 325 230" role="img" aria-label="Single-link tree for A at 0, B at 1, C at 3. A and B join at height one; that branch joins C at height two. A to C originally has distance three but first joins at height two.">
    <text x="18" y="20">merge height</text>
    {[0, 1, 2].map(v => <g key={v}><path d={`M40 ${180-v*60}H300`} stroke="#444" strokeDasharray="3 4" /><text x="23" y={185-v*60}>{v}</text></g>)}
    <path d="M75 180V120H165V180M120 120V60H270V180" fill="none" stroke="#e8b44a" strokeWidth="3" />
    <text x="75" y="205" textAnchor="middle">A: 0</text><text x="165" y="205" textAnchor="middle">B: 1</text><text x="270" y="205" textAnchor="middle">C: 3</text>
  </svg><figcaption>Trace upward from each leaf until the paths meet. AB meets at 1; AC and BC both meet at 2. The horizontal leaf positions merely lay out the tree; their printed values give the original locations. The tree preserves AB's distance but compresses original AC=3 and BC=2 into the same height.</figcaption></figure>;
}
