import './pca-intuition.css';

function ScorePanel({ whitened }) {
  const raw = [[-3, -1], [-3, 1], [3, -1], [3, 1]].map(row => row.map(value => value / Math.SQRT2));
  const points = whitened ? raw.map(([a, b]) => [a / Math.sqrt(6), b / Math.sqrt(2 / 3)]) : raw;
  const x = value => 145 + value * 46;
  const y = value => 130 - value * 46;
  return <div><p><strong>{whitened ? 'After whitening' : 'Ordinary PCA scores'}</strong></p><svg viewBox="0 0 290 270" role="img" aria-label={whitened ? 'Whitened scores at plus or minus .866 in both coordinates, making a square.' : 'Original score coordinates at plus or minus 2.121 horizontally and .707 vertically, making a wide rectangle.'}>
    <path d="M27 130H260M145 20V240" fill="none" stroke="#666" />
    {[-2, -1, 1, 2].map(value => <g key={value}><text x={x(value)} y="148" textAnchor="middle">{value}</text><text x="134" y={y(value) + 5} textAnchor="end">{value}</text></g>)}
    <path d={`M${x(points[0][0])} ${y(points[0][1])}L${x(points[1][0])} ${y(points[1][1])}L${x(points[3][0])} ${y(points[3][1])}L${x(points[2][0])} ${y(points[2][1])}Z`} fill="none" stroke="#666" strokeDasharray="4 4" />
    {points.map(([a, b], index) => <g key={index}><circle cx={x(a)} cy={y(b)} r="5" fill="#e8b44a" /><text x={x(a) + (a < 0 ? -10 : 10)} y={y(b) + (b < 0 ? 18 : -10)} textAnchor={a < 0 ? 'end' : 'start'}>{['A', 'B', 'C', 'D'][index]}</text></g>)}
    <text x="145" y="264" textAnchor="middle">{whitened ? 'whitened coordinate 1' : 'score 1'}</text><text x="27" y="17">{whitened ? 'coordinate 2' : 'score 2'}</text>
  </svg></div>;
}

export function WhiteningGeometryFigure() {
  return <figure className="pca-intuition" data-figure="whitening-geometry"><div className="pca-intuition-panels"><ScorePanel /><ScorePanel whitened /></div><figcaption>The same four observations, now drawn in score coordinates. Divide score 1 by √6 and score 2 by √(2/3). Both new coordinate magnitudes become √3/2≈.8660. A/B's original score distance √2 becomes √3, while A/C's 3√2 becomes √3: unequal separations become equal. Every panel uses the same numeric scale on both axes. Whitening rescales distances; it is more than another rotation.</figcaption></figure>;
}

export function DenoisingDirectionFigure() {
  const x = value => 35 + value * 47;
  const y = value => 220 - value * 47;
  return <figure className="pca-intuition" data-figure="noise-direction-projection"><div className="pca-intuition-panels">{[
    { name: 'Noise across the signal', observed: [3, 1], restored: [2, 2], result: '(3,1) projects to (2,2)' },
    { name: 'Noise along the signal', observed: [3, 3], restored: [3, 3], result: '(3,3) projects to (3,3)' },
  ].map(({ name, observed, restored, result }) => <div key={name}><p><strong>{name}</strong></p><svg viewBox="0 0 270 270" role="img" aria-label={`${name}: true signal two, two; ${result}.`}>
    <path d="M35 20V220H235" stroke="#666" fill="none" /><path d={`M${x(0)} ${y(0)}L${x(4)} ${y(4)}`} stroke="#888" strokeDasharray="5 5" />
    <path d={`M${x(2)} ${y(2)}L${x(observed[0])} ${y(observed[1])}`} stroke="#aaa" strokeWidth="2" />
    <path d={`M${x(observed[0])} ${y(observed[1])}L${x(restored[0])} ${y(restored[1])}`} stroke="#e8b44a" strokeWidth="3" />
    <circle cx={x(2)} cy={y(2)} r="7" fill="#ddd" />
    <path d={`M${x(observed[0])} ${y(observed[1]) - 8}l8 8-8 8-8-8Z`} fill="#101010" stroke="#e8b44a" strokeWidth="2" />
    {[0, 2, 4].map(value => <g key={value}><text x={x(value)} y="242" textAnchor="middle">{value}</text><text x="23" y={y(value) + 5} textAnchor="end">{value}</text></g>)}<text x="132" y="265" textAnchor="middle">sensor 1</text><text x="40" y="18">sensor 2</text>
    </svg><p>{result}</p></div>)}</div><figcaption>Constructed observations with true signal (2,2), shown by a white circle, and measured input shown by an amber diamond. The fixed retained line is x₂=x₁. Orthogonal noise (1,−1) disappears on projection; parallel noise (1,1) survives unchanged. Axes use equal scales. These examples inspect a given projection; estimating its direction from noisy data remains a separate fitting step.</figcaption></figure>;
}
