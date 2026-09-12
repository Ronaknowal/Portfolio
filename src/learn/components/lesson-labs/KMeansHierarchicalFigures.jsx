import { LessonTable } from './LessonElements.jsx';
import { faithfulPoints, faithfulDiagnostics, faithfulTwoGroups, faithfulSubsample, faithfulWardLinkage } from '../../data/k-means-hierarchical-faithful.js';
import './k-means-hierarchical-figures.css';

const groupColors = ['#e2b55a', '#7dd3fc'];

/** 272 real observations. `grouped` colors each eruption by the executed
 * standardized k = 2 fit and marks the two centers in original units. */
export function FaithfulScatterFigure({ grouped = false }) {
  const x = value => 40 + (value - 1.5) * (250 / 4);
  const y = value => 200 - (value - 40) * (170 / 60);
  return <figure className="cluster-figure cluster-standalone">
    <figcaption><strong>{grouped ? 'The same 272 eruptions, colored by a fitted two-center partition' : 'Old Faithful: 272 recorded eruptions'}</strong> · real measurements, minutes on both axes</figcaption>
    <svg viewBox="0 0 320 240" role="img" aria-label={grouped ? `Scatter of eruption duration against waiting time, colored by two fitted groups. Centers near ${faithfulTwoGroups.centers[0][0].toFixed(2)} minutes and ${faithfulTwoGroups.centers[0][1].toFixed(1)} minutes wait, and ${faithfulTwoGroups.centers[1][0].toFixed(2)} minutes and ${faithfulTwoGroups.centers[1][1].toFixed(1)} minutes wait.` : 'Scatter of eruption duration in minutes against waiting time to the next eruption in minutes. Two dense regions: short eruptions with waits near 55 minutes and long eruptions with waits near 80 minutes.'}>
      {[40, 60, 80, 100].map(value => <g key={value}><line x1="40" x2="290" y1={y(value)} y2={y(value)} stroke="#333" /><text x="34" y={y(value) + 4} textAnchor="end">{value}</text></g>)}
      {[2, 3, 4, 5].map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1="30" y2="200" stroke="#2a2a2a" /><text x={x(value)} y="216" textAnchor="middle">{value}</text></g>)}
      {faithfulPoints.map((point, index) => <circle key={index} cx={x(point[0])} cy={y(point[1])} r="2.6" fill={grouped ? groupColors[faithfulTwoGroups.labels[index]] : '#d8d8d8'} fillOpacity=".75" />)}
      {grouped && faithfulTwoGroups.centers.map((center, index) => <g key={index} stroke={groupColors[index]} strokeWidth="3"><path d={`M${x(center[0]) - 7},${y(center[1])}h14 M${x(center[0])},${y(center[1]) - 7}v14`} /><circle cx={x(center[0])} cy={y(center[1])} r="10" fill="none" strokeWidth="1" /></g>)}
      <text x="165" y="234" textAnchor="middle">eruption duration (minutes)</text>
      <text x="12" y="115" textAnchor="middle" transform="rotate(-90 12 115)">waiting time (minutes)</text>
    </svg>
    {grouped
      ? <p>Crosses are the two fitted centers after standardizing both columns, mapped back to minutes: about {faithfulTwoGroups.centers[0][0].toFixed(2)} min eruptions followed by {faithfulTwoGroups.centers[0][1].toFixed(1)} min waits ({faithfulTwoGroups.sizes[0]} rows) and {faithfulTwoGroups.centers[1][0].toFixed(2)} min eruptions followed by {faithfulTwoGroups.centers[1][1].toFixed(1)} min waits ({faithfulTwoGroups.sizes[1]} rows). The split is a fitted description of this geometry, not a geological classification.</p>
      : <p>Each dot is one eruption: how long it lasted, and how long visitors then waited for the next one. Durations were recorded to the nearest second and are heavily rounded. Nothing in the file says which eruptions belong together, yet two dense regions are visible. Deciding whether that visual impression is a useful grouping is exactly the job of this lesson.</p>}
  </figure>;
}

/** Executed on the standardized Old Faithful features by the displayed
 * production program: KMeans(n_init=1) with seeds 0..4 at each k. */
export function ClusteringDiagnosticsFigure() {
  const xPosition = k => 49 + (k - 1) * 34.5;
  const logY = value => 205 - 170 * (Math.log10(value) - 1) / 2;
  const linearY = value => 205 - 170 * value;
  return <figure className="cluster-figure">
    <figcaption><strong>Calculated diagnostics for Old Faithful, k = 1 to 8</strong> · five single starts at each k on standardized features, scikit-learn 1.9.1</figcaption>
    <div className="cluster-figure-pair">
      <div><p>Fitted inertia, log scale (squared standardized units)</p>
        <svg viewBox="0 0 320 255" role="img" aria-label={`Inertia by k on a logarithmic axis: ${faithfulDiagnostics.map(row => `k ${row.k}, best ${Math.min(...row.inertias).toFixed(1)}`).join('; ')}. The largest proportional drop is from one center to two.`}>
          {[10, 100, 1000].map(value => <g key={value}><line x1="49" x2="300" y1={logY(value)} y2={logY(value)} stroke="#333" /><text x="42" y={logY(value) + 4} textAnchor="end">{value}</text></g>)}
          <polyline points={faithfulDiagnostics.map(row => `${xPosition(row.k)},${logY(Math.min(...row.inertias))}`).join(' ')} fill="none" stroke="#e2b55a" strokeWidth="2" />
          {faithfulDiagnostics.map(row => <g key={row.k}>
            {row.inertias.map((value, seed) => <circle key={seed} cx={xPosition(row.k)} cy={logY(value)} r="3" fill="#e2b55a" fillOpacity=".55" />)}
            <text x={xPosition(row.k)} y="227" textAnchor="middle">{row.k}</text>
          </g>)}
          <text x="174" y="248" textAnchor="middle">k · number of centers</text>
        </svg></div>
      <div><p>Median silhouette over the five starts (unitless)</p>
        <svg viewBox="0 0 320 255" role="img" aria-label={`Median silhouette by k: ${faithfulDiagnostics.filter(row => row.medianSilhouette !== null).map(row => `k ${row.k}, ${row.medianSilhouette.toFixed(3)}`).join('; ')}. Undefined at k equals 1. Highest at k equals 2.`}>
          {[0, 0.5, 1].map(value => <g key={value}><line x1="49" x2="300" y1={linearY(value)} y2={linearY(value)} stroke="#333" /><text x="42" y={linearY(value) + 4} textAnchor="end">{value}</text></g>)}
          <polyline points={faithfulDiagnostics.filter(row => row.medianSilhouette !== null).map(row => `${xPosition(row.k)},${linearY(row.medianSilhouette)}`).join(' ')} fill="none" stroke="#7dd3fc" strokeWidth="2" />
          {faithfulDiagnostics.map(row => <g key={row.k}>
            {row.medianSilhouette !== null ? <circle cx={xPosition(row.k)} cy={linearY(row.medianSilhouette)} r="4" fill="#7dd3fc" /> : <text x={xPosition(row.k) + 6} y={linearY(0.5) - 8} textAnchor="start" fill="#888">undefined</text>}
            <text x={xPosition(row.k)} y="227" textAnchor="middle">{row.k}</text>
          </g>)}
          <text x="174" y="248" textAnchor="middle">k · number of centers</text>
        </svg></div>
    </div>
    <p>Each gold dot is one start; the line follows the best of the five. The inertia axis is logarithmic so that equal vertical drops mean equal proportional improvements: one center to two removes about 85% of the error, two to three about 29%, and later steps less. The silhouette is undefined for one cluster and peaks at two. From k = 3 onward the five starts disagree, which is itself information about the objective landscape.</p>
    <LessonTable caption="Same executed values as the graphs; six-decimal rounding" headers={['k', 'inertia: best to worst of five starts', 'median silhouette']} rows={faithfulDiagnostics.map(row => [row.k, `${Math.min(...row.inertias).toFixed(6)} to ${Math.max(...row.inertias).toFixed(6)}`, row.medianSilhouette === null ? 'undefined' : row.medianSilhouette.toFixed(6)])} />
  </figure>;
}

/** SciPy Ward tree of a deterministic 40-row Old Faithful subsample on
 * standardized coordinates. Leaves carry the color of the executed 272-row
 * k = 2 fit so the two representations can be compared. */
export function FaithfulDendrogramFigure() {
  const n = faithfulSubsample.length;
  const nodes = faithfulSubsample.map((row, index) => ({ id: index, height: 0, members: [index] }));
  faithfulWardLinkage.forEach(([left, right, height], index) => nodes.push({ id: n + index, left, right, height, members: [...nodes[left].members, ...nodes[right].members] }));
  const root = nodes.at(-1);
  const order = [];
  const visit = id => { const node = nodes[id]; if (node.left === undefined) order.push(id); else { visit(node.left); visit(node.right); } };
  visit(root.id);
  const positions = new Map(order.map((id, index) => [id, 40 + 260 * index / (n - 1)]));
  faithfulWardLinkage.forEach(([left, right], index) => positions.set(n + index, (positions.get(left) + positions.get(right)) / 2));
  const maxHeight = root.height * 1.06;
  const y = value => 200 - 176 * value / maxHeight;
  const cutHeight = 5;
  const clustersAtCut = 1 + faithfulWardLinkage.filter(([, , height]) => height > cutHeight).length;
  const secondCut = 2;
  const clustersAtSecond = 1 + faithfulWardLinkage.filter(([, , height]) => height > secondCut).length;
  return <figure className="cluster-figure cluster-standalone">
    <figcaption><strong>A realistic Ward dendrogram: 40 Old Faithful eruptions</strong> · SciPy 1.18.1 on standardized coordinates, leaf color from the 272-row two-center fit</figcaption>
    <svg viewBox="0 0 320 236" role="img" aria-label={`Ward dendrogram with forty leaves. The final merge is at height ${root.height.toFixed(2)}, far above the next merge at ${faithfulWardLinkage.at(-2)[2].toFixed(2)}. A cut at ${cutHeight} leaves ${clustersAtCut} clusters; a cut at ${secondCut} leaves ${clustersAtSecond}.`}>
      {[0, 4, 8, 12].map(value => <g key={value}><line x1="34" x2="306" y1={y(value)} y2={y(value)} stroke="#2a2a2a" /><text x="28" y={y(value) + 4} textAnchor="end">{value}</text></g>)}
      {faithfulWardLinkage.map(([left, right, height], index) => {
        const members = nodes[n + index].members;
        const labels = members.map(member => faithfulTwoGroups.labels[faithfulSubsample[member]]);
        const same = labels.every(label => label === labels[0]);
        return <path key={index} d={`M${positions.get(left)},${y(nodes[left].height)}V${y(height)}H${positions.get(right)}V${y(nodes[right].height)}`} fill="none" stroke={same ? groupColors[labels[0]] : '#9a9a9a'} strokeWidth="1.4" />;
      })}
      <line x1="34" x2="306" y1={y(cutHeight)} y2={y(cutHeight)} stroke="#e2b55a" strokeDasharray="5 4" />
      <text x="308" y={y(cutHeight) + 4}>{clustersAtCut}</text>
      <line x1="34" x2="306" y1={y(secondCut)} y2={y(secondCut)} stroke="#e2b55a" strokeDasharray="2 4" strokeOpacity=".7" />
      <text x="308" y={y(secondCut) + 4}>{clustersAtSecond}</text>
      {order.map((id, index) => <circle key={id} cx={positions.get(id)} cy="206" r="2.6" fill={groupColors[faithfulTwoGroups.labels[faithfulSubsample[id]]]} />)}
      <text x="170" y="228" textAnchor="middle">40 leaves, colored by the two-center fit</text>
    </svg>
    <p>Read the vertical gaps, not the leaf order. The last merge sits at height {root.height.toFixed(2)} while every earlier merge is below {faithfulWardLinkage.at(-2)[2].toFixed(2)}, so any cut in that long empty stretch gives the same two clusters, and they coincide with the two-center k-means colors on these rows. The dotted line at height {secondCut} instead yields {clustersAtSecond} clusters whose boundaries are much less stable to the cut position. Gaps like the tall one are the evidence a dendrogram offers; the horizontal placement of leaves is drawing convention.</p>
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
