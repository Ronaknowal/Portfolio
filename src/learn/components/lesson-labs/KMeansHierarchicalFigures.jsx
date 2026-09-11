import { LessonTable } from './LessonElements.jsx';
import './k-means-hierarchical-figures.css';

// Rounded actual output of clusteringExamples.production, not invented curve data.
// Fixture: 150 make_blobs rows, std .8, data seed42; KMeans n_init1, seeds0/1/2.
export const clusteringDiagnosticRows = [
  { k: 2, minimumInertia: 2660.009039, maximumInertia: 2660.009039, medianSilhouette: .719940 },
  { k: 3, minimumInertia: 181.504432, maximumInertia: 181.504432, medianSilhouette: .876025 },
  { k: 4, minimumInertia: 156.192219, maximumInertia: 171.664605, medianSilhouette: .695022 },
  { k: 5, minimumInertia: 136.295134, maximumInertia: 149.916286, medianSilhouette: .519682 }
];

export function ClusteringDiagnosticsFigure() {
  const xPosition = k => 49 + (k - 2) * 77;
  return <figure className="cluster-figure">
    <figcaption><strong>Calculated diagnostics from the runnable 150-row example</strong> · three single starts at each k, scikit-learn 1.9.1</figcaption>
    <div className="cluster-figure-pair">
      {[{ key: 'minimumInertia', label: 'Fitted inertia (squared feature units)', maximum: 2800, ticks: [0, 1400, 2800], color: '#e2b55a' }, { key: 'medianSilhouette', label: 'Median mean silhouette (unitless)', maximum: 1, ticks: [0, .5, 1], color: '#7dd3fc' }].map(chart => {
        const yPosition = value => 205 - 170 * value / chart.maximum;
        return <div key={chart.key}><p>{chart.label}</p><svg viewBox="0 0 320 255" role="img" aria-label={`${chart.label}: ${clusteringDiagnosticRows.map(row => `k ${row.k}, ${row[chart.key]}`).join('; ')}. Exact rounded values and inertia ranges are in the table.`}>
          {chart.ticks.map(tick => <g key={tick}><line x1="49" x2="290" y1={yPosition(tick)} y2={yPosition(tick)} stroke="#333" /><text x="42" y={yPosition(tick) + 4} textAnchor="end">{tick}</text></g>)}
          <polyline points={clusteringDiagnosticRows.map(row => `${xPosition(row.k)},${yPosition(row[chart.key])}`).join(' ')} fill="none" stroke={chart.color} strokeWidth="2" />
          {clusteringDiagnosticRows.map(row => <g key={row.k}>
            {chart.key === 'minimumInertia' && <path d={`M${xPosition(row.k) - 5} ${yPosition(row.maximumInertia)} h10 M${xPosition(row.k)} ${yPosition(row.maximumInertia)} V${yPosition(row.minimumInertia)}`} stroke={chart.color} fill="none" />}
            <circle cx={xPosition(row.k)} cy={yPosition(row[chart.key])} r="4" fill={chart.color} />
            <text x={xPosition(row.k)} y="227" textAnchor="middle">{row.k}</text>
          </g>)}
          <text x="164" y="248" textAnchor="middle">k · number of centers</text>
        </svg></div>;
      })}
    </div>
    <p>The gold line is the best inertia among the three starts; whiskers extend to the worst. These ranges are seed variation, not confidence intervals. Lines connect the tested integer k values as a reading aid. Both y-axes are linear. Small inertia ranges can be easier to inspect numerically.</p>
    <LessonTable caption="Same executed diagnostic values as the graphs; six-decimal rounding" headers={['k', 'inertia: min to max', 'median silhouette']} rows={clusteringDiagnosticRows.map(row => [row.k, `${row.minimumInertia.toFixed(6)} to ${row.maximumInertia.toFixed(6)}`, row.medianSilhouette.toFixed(6)])} />
  </figure>;
}

export function MeanRepresentativeFigure() {
  const points = [1, 2, 9];
  return <figure className="cluster-figure">
    <figcaption><strong>One representative, two placements</strong> · exact calculation for points 1, 2 and 9</figcaption>
    <div className="cluster-figure-pair">
      {[2, 4].map(center => <div key={center}>
        <p>Center c = {center}{center === 4 ? ' (the mean)' : ''}</p>
        <svg viewBox="0 0 320 145" role="img" aria-label={`Points 1, 2 and 9 connected to center ${center}; sum of squared distances ${points.reduce((sum, point) => sum + (point - center) ** 2, 0)}.`}>
          <line x1="20" y1="105" x2="300" y2="105" className="cluster-axis" />
          {points.map((point, index) => <g key={point}>
            <line x1={20 + point * 28} y1="90" x2={20 + center * 28} y2="38" className="cluster-connector" />
            <circle cx={20 + point * 28} cy="90" r="5" fill="#e8e8e8" />
            <text x={20 + point * 28} y={index === 1 ? 137 : 122} textAnchor="middle">{point}</text>
          </g>)}
          <path d={`M ${20 + center * 28} 29 l 8 9 -8 9 -8 -9 Z`} fill="#e2b55a" />
          <text x={20 + center * 28} y="19" textAnchor="middle">c = {center}</text>
        </svg>
        <p className="cluster-calculation">{points.map(point => `(${point}−${center})²`).join(' + ')} = <strong>{points.reduce((sum, point) => sum + (point - center) ** 2, 0)}</strong></p>
      </div>)}
    </div>
    <p>The links show which representative owns each point. Their drawn slopes are layout; the calculation uses horizontal numerical differences. Moving from 2 to the mean 4 reduces squared error from 50 to 38. The mean need not be an observed point.</p>
  </figure>;
}

export function LinkageDefinitionFigure() {
  return <figure className="cluster-figure">
    <figcaption><strong>“Closest groups” has several meanings</strong> · one-dimensional observations</figcaption>
    <svg viewBox="0 0 340 150" className="cluster-linkage-ruler" role="img" aria-label="Group A contains 0 and 2; group B contains 5 and 9. The nearest cross-group pair is 2 and 5, distance 3; the farthest is 0 and 9, distance 9.">
      <line x1="28" x2="316" y1="104" y2="104" className="cluster-axis" />
      <path d="M28 48 V34 H316 V48" fill="none" stroke="#c084fc" />
      <text x="172" y="23" textAnchor="middle">complete: 9</text>
      <path d="M92 74 V61 H188 V74" fill="none" stroke="#4ade80" />
      <text x="140" y="53" textAnchor="middle">single: 3</text>
      {[0, 2, 5, 9].map((point, index) => <g key={point}>
        {index < 2 ? <circle cx={28 + point * 32} cy="104" r="6" fill="#e2b55a" /> : <rect x={22 + point * 32} y="98" width="12" height="12" fill="#7dd3fc" />}
        <text x={28 + point * 32} y="130" textAnchor="middle">{index < 2 ? 'A' : 'B'}: {point}</text>
      </g>)}
    </svg>
    <LessonTable caption="All cross-group distances; average linkage uses every entry" headers={['distance', 'B: 5', 'B: 9']} rows={[["A: 0", '5', '9'], ["A: 2", '3', '7']]} />
    <p>Single = 3; complete = 9; average = (5+9+3+7)/4 = 6. The means are 1 and 7. Ward’s increase in squared error is Δ = (2·2/4)(7−1)² = 36; a SciPy-style Ward tree places this merge at √72 ≈ 8.485. Those are different questions, with different units.</p>
  </figure>;
}

export function ClusteringShapeFigure() {
  const rings = [1, 2.4].map(radius => Array.from({ length: 16 }, (_, index) => {
    const angle = index * Math.PI / 8;
    return [Math.cos(angle) * radius, Math.sin(angle) * radius];
  }));
  return <figure className="cluster-figure">
    <figcaption><strong>A geometric limitation</strong> · constructed concentric rings, not fitted measurements</figcaption>
    <div className="cluster-figure-pair">
      <div>
        <p>A proposed “inner versus outer” grouping</p>
        <svg viewBox="0 0 280 245" role="img" aria-label="Two concentric rings centered at the same coordinate. Inner points are circles, outer points are squares.">
          {rings.map((points, group) => points.map(([x, y], index) => group === 0 ? <circle key={`${group}-${index}`} cx={140 + x * 43} cy={120 - y * 43} r="4" fill="#e2b55a" /> : <rect key={`${group}-${index}`} x={136 + x * 43} y={116 - y * 43} width="8" height="8" fill="#7dd3fc" />))}
          <path d="M134 120 H146 M140 114 V126" stroke="#e8e8e8" strokeWidth="2" />
          <text x="140" y="241" textAnchor="middle">Both ring means are (0, 0)</text>
        </svg>
      </div>
      <div>
        <p>Two distinct Euclidean centers</p>
        <svg viewBox="0 0 280 245" role="img" aria-label="The same ring points split by a vertical nearest-center boundary. That straight boundary divides both rings rather than separating inner from outer.">
          <rect x="18" y="10" width="122" height="218" fill="#e2b55a" opacity=".06" />
          <rect x="140" y="10" width="122" height="218" fill="#7dd3fc" opacity=".06" />
          <line x1="140" x2="140" y1="10" y2="226" stroke="#aaa" strokeDasharray="4 4" />
          {rings.flat().map(([x, y], index) => <circle key={index} cx={140 + x * 43} cy={120 - y * 43} r="4" fill={x < -1e-10 ? '#e2b55a' : '#7dd3fc'} />)}
          {[[-1.2, 'L'], [1.2, 'R']].map(([x, label]) => <g key={label}><path d={`M${140 + x * 43} 110 l8 10 -8 10 -8 -10Z`} fill="#050505" stroke="#e8e8e8" /><text x={140 + x * 43} y="148" textAnchor="middle">{label}</text></g>)}
          <text x="140" y="241" textAnchor="middle">Schematic centers; not a fitted result</text>
        </svg>
      </div>
    </div>
    <p>With two distinct centers, the equal-distance boundary is a line, so it cannot enclose just the inner ring. Coincident ring means tie everywhere and also cannot recover the rings under a fixed nearest-center tie rule. A different geometry or clustering objective is needed if ring membership is the intended structure.</p>
  </figure>;
}
