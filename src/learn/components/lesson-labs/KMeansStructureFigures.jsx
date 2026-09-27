import './k-means-structure.css';

export function SingleLinkPathFigure() {
  const x = value => 35 + value * 45;
  return <figure className="cluster-structure" data-figure="single-link-path">
    <p><strong>A path of short edges can connect far-apart endpoints</strong></p>
    <svg viewBox="0 0 340 140" role="img" aria-label="Points at zero, one, two and six. At distance threshold one, edges zero-to-one and one-to-two remain; the edge from two to six is cut. Endpoints zero and two belong together despite their distance two.">
      {[[0, 1], [1, 2], [2, 6]].map(([a, b]) => <g key={a}><path d={`M${x(a)} 66H${x(b)}`} stroke={b - a <= 1 ? '#e8b44a' : '#777'} strokeWidth="3" strokeDasharray={b - a > 1 ? '5 5' : undefined} /><text x={(x(a) + x(b)) / 2} y="46" textAnchor="middle">{b - a}</text></g>)}
      {[0, 1, 2, 6].map(value => <g key={value}><circle cx={x(value)} cy="66" r="7" fill={value === 6 ? '#ddd' : '#e8b44a'} /><text x={x(value)} y="96" textAnchor="middle">{value}</text></g>)}
      <text x="170" y="127" textAnchor="middle">one-dimensional position</text>
    </svg>
    <figcaption>This line's minimum spanning tree uses edges of length 1, 1 and 4. Keeping edges at most 1 gives groups {'{0,1,2}'} and {'{6}'}. The first group's endpoints are two units apart. Complete linkage would evaluate that endpoint distance too; single linkage only needs an unbroken path.</figcaption>
  </figure>;
}

export function CompressedScatterFigure() {
  const x = value => 35 + value * 65;
  return <figure className="cluster-structure" data-figure="compressed-subcluster-scatter">
    <p><strong>A mean plus a count preserves whole-group costs; it cannot undo a group</strong></p>
    <svg viewBox="0 0 330 185" role="img" aria-label="Original points zero and two have mean one and internal squared error two. Moving their shared representative from one to four adds eighteen, for total squared error twenty.">
      <text x="35" y="25">Original members</text><path d="M35 52H295" stroke="#666" />
      {[0, 2].map(value => <g key={value}><circle cx={x(value)} cy="52" r="6" fill="#ddd" /><text x={x(value)} y="79" textAnchor="middle">{value}</text></g>)}
      <text x="35" y="111">Summary → new center</text><path d={`M${x(1)} 136H${x(4)}`} stroke="#e8b44a" strokeWidth="3" />
      <circle cx={x(1)} cy="136" r="6" fill="#e8b44a" /><path d={`M${x(4)} 128l8 8-8 8-8-8Z`} fill="#111" stroke="#e8b44a" strokeWidth="2" />
      <text x={x(1)} y="168" textAnchor="middle">mean 1</text><text x={x(4)} y="168" textAnchor="middle">c = 4</text>
    </svg>
    <figcaption>For {'{0,2}'}, N=2, LS=2, SS=4. Internal scatter is 4−2²/2=2. With shared center 4, add N(1−4)²=18; total 20 agrees with (0−4)²+(2−4)². If the original points could choose separate centers 0 and 2, their error would instead be zero. Compressing them into an indivisible weighted mean excludes that assignment.</figcaption>
  </figure>;
}
