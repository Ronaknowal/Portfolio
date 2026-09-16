import { useState } from 'react';
import { fourPoints, projectAtAngle, wineFullScores } from '../../data/pca-models';
import { wineFeatures, wineFullFit, wineCultivar, gaussianSpectrum } from '../../data/pca-wine-data';
import { SquarePlot, lineThrough, ScoreStrip, Table } from './PcaLabs.jsx';
import './pca-labs.css';

const number = (value, digits = 4) => Number.isInteger(value) ? String(value) : value.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
const percent = value => `${number(100 * value, 2)}%`;
const names = ['A', 'B', 'C', 'D'];

/** F1: four observations, the diagonal ruler through the mean, their
 * perpendicular feet, A's residual with a right-angle mark, and the score strip. */
export function ProjectionShadowFigure() {
  const state = projectAtAngle(fourPoints, 45);
  return <figure className="pca-figure pca-standalone">
    <figcaption><strong>A point, its shadow and its saved coordinate</strong> · exact calculation for A, B, C, D with the diagonal ruler</figcaption>
    <SquarePlot points={fourPoints} extra={[state.mean]} describe="Four labeled observations, the line through the mean (3, 2) at 45 degrees, the perpendicular foot of each point on the line, and the dashed residual of A from (1, 1) to (1.5, 0.5).">
      {(project, domain, extent) => <>
        <line className="pca-line" {...lineThrough(state.mean, state.direction, domain, extent, project)} />
        {fourPoints.map((point, index) => {
          const [px, py] = project(point), [fx, fy] = project(state.projections[index]);
          return <g key={index}>
            <line className="pca-residual" x1={px} y1={py} x2={fx} y2={fy} strokeOpacity={index === 0 ? 1 : 0.45} />
            <circle className="pca-foot" cx={fx} cy={fy} r="3" />
            <circle className={`pca-point ${index === 0 ? 'is-selected' : ''}`} cx={px} cy={py} r="4.5" />
            <text x={px + 7} y={py - 6}>{names[index]}</text>
          </g>;
        })}
        {(() => {
          const [fx, fy] = project(state.projections[0]);
          const d = state.direction, r = [-d[1], d[0]];
          const s = 9;
          return <path d={`M${fx + s * d[0]},${fy - s * d[1]} L${fx + s * d[0] + s * r[0]},${fy - s * d[1] - s * r[1]} L${fx + s * r[0]},${fy - s * r[1]}`} fill="none" stroke="#da9c86" strokeWidth="1" />;
        })()}
        {(() => { const [mx, my] = project(state.mean); return <g><circle className="pca-mean" cx={mx} cy={my} r="6" /><text x={mx + 9} y={my + 14}>mean (3, 2)</text></g>; })()}
        {(() => { const [fx, fy] = project(state.projections[0]); return <text x={fx + 14} y={fy + 8}>Â</text>; })()}
      </>}
    </SquarePlot>
    <ScoreStrip scores={state.scores} names={names} caption="Each observation's score is its signed distance along the ruler from the mean. A and B share a score; so do C and D." />
    <p>A = (1, 1) and B = (2, 0) have different readings but the same diagonal score, −2.1213. The dashed segment from A to its foot Â = (1.5, 0.5) is the difference the one-coordinate representation loses; its squared length is 0.5, and the same is true for each of the other three points. The small square at Â marks the right angle: the residual is perpendicular to the ruler.</p>
  </figure>;
}

/** F2: three views of the same centered points with the fixed total 20 split
 * into retained and residual squared length. */
export function ConservationFigure() {
  const angles = [0, 45, 135];
  const states = angles.map(angle => projectAtAngle(fourPoints, angle));
  return <figure className="pca-figure">
    <figcaption><strong>The conserved quantity behind the two objectives</strong> · retained + residual = 20 for every ruler</figcaption>
    <div className="pca-multiples">
      {states.map((state, index) => <figure key={angles[index]}>
        <figcaption>{angles[index]}°: retains {number(state.retained)}, loses {number(state.sse)}</figcaption>
        <svg viewBox="0 0 160 160" role="img" aria-label={`Centered points with a ruler at ${angles[index]} degrees: retained squared length ${number(state.retained)}, residual ${number(state.sse)}.`}>
          {[20, 80, 140].map(position => <path key={position} className="pca-grid" d={`M20,${position}H140 M${position},20V140`} />)}
          {(() => { const scale = 20; const project = point => [80 + scale * point[0], 80 - scale * point[1]]; const d = state.direction; const [x1, y1] = project([-3 * d[0], -3 * d[1]]), [x2, y2] = project([3 * d[0], 3 * d[1]]);
            return <>
              <line className="pca-line" x1={x1} y1={y1} x2={x2} y2={y2} />
              {state.centered.map((point, which) => { const [px, py] = project(point); const foot = [state.scores[which] * d[0], state.scores[which] * d[1]]; const [fx, fy] = project(foot); return <g key={which}><line className="pca-residual" x1={px} y1={py} x2={fx} y2={fy} /><circle className="pca-foot" cx={fx} cy={fy} r="2.5" /><circle className="pca-point" cx={px} cy={py} r="3.5" /><text x={px + (point[0] < 0 ? -7 : 7)} y={py + (point[1] < 0 ? 15 : -7)} textAnchor={point[0] < 0 ? 'end' : 'start'}>{names[which]}</text></g>; })}
            </>; })()}
        </svg>
      </figure>)}
    </div>
    <div className="pca-bars" role="img" aria-label="Three bars of equal total 20: retained 10 and residual 10 at 0 degrees; retained 18 and residual 2 at 45 degrees; retained 2 and residual 18 at 135 degrees.">
      {states.map((state, index) => <div key={angles[index]} className="pca-bar-row"><span>{angles[index]}°</span><span className="pca-bar-track"><span className="pca-bar-fill" style={{ width: `${100 * state.retained / state.total}%` }} /><span className="pca-bar-fill is-residual" style={{ left: `${100 * state.retained / state.total}%`, width: `${100 * state.sse / state.total}%` }} /></span><span className="pca-bar-value">{number(state.retained)} retained · {number(state.sse)} lost</span></div>)}
    </div>
    <p>Gold is retained squared length, red is residual. Every bar has the same total, 20, because turning the ruler cannot change the centered points' squared lengths. The direction that keeps most therefore loses least: dividing the retained 18 by n − 1 = 3 gives the first sample variance 6, and dividing the lost 2 by 3 gives the second, 2/3.</p>
  </figure>;
}

/** F3: matrix shapes for centering, projection and mean-restored reconstruction,
 * with A's row highlighted and a second branch for a new observation. */
function Matrix({ rows, label, highlightRow = null, columns }) {
  return <span><span className="pca-matrix" style={{ gridTemplateColumns: `repeat(${columns}, auto)` }}>{rows.flatMap((row, r) => row.map((value, c) => <span key={`${r}-${c}`} className={r === highlightRow ? 'is-a' : undefined}>{typeof value === 'number' ? number(value) : value}</span>))}</span><span className="pca-matrix-label">{label}</span></span>;
}
export function TransformShapesFigure() {
  const mean = [3, 2];
  const centered = fourPoints.map(point => [point[0] - mean[0], point[1] - mean[1]]);
  const v = [Math.SQRT1_2, Math.SQRT1_2];
  const scores = centered.map(row => row[0] * v[0] + row[1] * v[1]);
  const rebuilt = scores.map(score => [mean[0] + score * v[0], mean[1] + score * v[1]]);
  return <figure className="pca-figure">
    <figcaption><strong>Shapes and the fitted transform</strong> · which matrix multiplies which, and what a new observation reuses</figcaption>
    <p>Forward: subtract the fitted mean, then multiply by the retained direction. A's row is highlighted throughout.</p>
    <div className="pca-shapes">
      <Matrix rows={fourPoints} columns={2} label="X · 4×2" highlightRow={0} /><span className="pca-operator">−</span>
      <Matrix rows={[mean]} columns={2} label="mean · 1×2 (broadcast to every row)" /><span className="pca-operator">=</span>
      <Matrix rows={centered} columns={2} label="centered A · 4×2" highlightRow={0} /><span className="pca-operator">×</span>
      <Matrix rows={[[v[0]], [v[1]]]} columns={1} label="V₁ · 2×1 (one direction as a column)" /><span className="pca-operator">=</span>
      <Matrix rows={scores.map(score => [score])} columns={1} label="scores Z · 4×1" highlightRow={0} />
    </div>
    <p>Inverse: multiply the scores by the direction as a row, then add the mean back. Without the last step the result would sit around zero, not around the data.</p>
    <div className="pca-shapes">
      <Matrix rows={scores.map(score => [score])} columns={1} label="Z · 4×1" highlightRow={0} /><span className="pca-operator">×</span>
      <Matrix rows={[v]} columns={2} label="V₁ᵀ · 1×2" /><span className="pca-operator">+</span>
      <Matrix rows={[mean]} columns={2} label="mean" /><span className="pca-operator">=</span>
      <Matrix rows={rebuilt} columns={2} label="reconstruction X̂ · 4×2" highlightRow={0} />
    </div>
    <p>A new observation reuses the fitted mean and direction without refitting: (6, 4) − (3, 2) = (3, 2); score 3·0.7071 + 2·0.7071 = 3.5355; reconstruction (3, 2) + 3.5355·(0.7071, 0.7071) = (5.5, 4.5). In code the directions are stored as rows, so the forward multiplication uses the transpose and the reconstruction does not.</p>
  </figure>;
}

/** F4: real Wine variance shares raw versus standardized, standardized score
 * scatter with optional cultivar shapes, and PC1's signed coefficients. */
export function WineOverviewFigure() {
  const [showCultivar, setShowCultivar] = useState(false);
  const scores = wineFullScores(2);
  const x = value => 160 + 28 * value, y = value => 120 - 28 * value;
  const marks = ['circle', 'square', 'triangle'];
  const order = wineFullFit.components[0].map((weight, index) => [weight, index]).sort((left, right) => right[0] - left[0]);
  return <figure className="pca-figure">
    <figcaption><strong>Real Wine measurements: scale, scores and coefficients</strong> · descriptive fit on all 178 rows, scikit-learn 1.9.1</figcaption>
    <div className="pca-figure-pair">
      <div><p>Raw readings: share of variance by component</p>
        <div className="pca-bars" role="img" aria-label={`Raw explained variance ratios: ${wineFullFit.rawRatios.slice(0, 4).map((ratio, index) => `PC${index + 1} ${percent(ratio)}`).join(', ')}.`}>
          {wineFullFit.rawRatios.slice(0, 4).map((ratio, index) => <div key={index} className="pca-bar-row"><span>PC{index + 1}</span><span className="pca-bar-track"><span className="pca-bar-fill" style={{ width: `${100 * ratio}%` }} /></span><span className="pca-bar-value">{percent(ratio)}</span></div>)}
        </div></div>
      <div><p>Standardized readings: share of variance by component</p>
        <div className="pca-bars" role="img" aria-label={`Standardized explained variance ratios: ${wineFullFit.standardizedRatios.slice(0, 4).map((ratio, index) => `PC${index + 1} ${percent(ratio)}`).join(', ')}.`}>
          {wineFullFit.standardizedRatios.slice(0, 4).map((ratio, index) => <div key={index} className="pca-bar-row"><span>PC{index + 1}</span><span className="pca-bar-track"><span className="pca-bar-fill is-second" style={{ width: `${100 * ratio}%` }} /></span><span className="pca-bar-value">{percent(ratio)}</span></div>)}
        </div></div>
    </div>
    <p>Both bar charts share a 0 to 100% axis. The raw first component is almost entirely proline, whose values are in the hundreds while most other measurements are below ten. After standardization the first two components retain {percent(wineFullFit.standardizedRatios[0] + wineFullFit.standardizedRatios[1])} between them.</p>
    <svg viewBox="0 0 320 262" role="img" aria-label={`Standardized PC1 against PC2 for 178 wines. ${showCultivar ? 'Cultivars 1, 2 and 3 are drawn as circles, squares and triangles.' : 'Points are unlabeled.'}`} style={{ width: '100%', maxWidth: 520, margin: 'auto', display: 'block' }}>
      {[-4, -2, 0, 2, 4].map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1="10" y2="230" className="pca-grid" /><line x1="30" x2="290" y1={y(value)} y2={y(value)} className="pca-grid" /><text x={x(value)} y="243" textAnchor="middle">{value}</text><text x="24" y={y(value) + 4} textAnchor="end">{value}</text></g>)}
      {scores.map((score, index) => {
        const cx = x(score[0]), cy = y(score[1]);
        const kind = showCultivar ? marks[wineCultivar[index] - 1] : 'circle';
        const fill = showCultivar ? ['#e7b94a', '#8eb9a5', '#91aecf'][wineCultivar[index] - 1] : '#d8dcd9';
        if (kind === 'square') return <rect key={index} x={cx - 3} y={cy - 3} width="6" height="6" fill={fill} fillOpacity=".8" />;
        if (kind === 'triangle') return <path key={index} d={`M${cx},${cy - 4}l4,7h-8z`} fill={fill} fillOpacity=".8" />;
        return <circle key={index} cx={cx} cy={cy} r="3" fill={fill} fillOpacity=".8" />;
      })}
      <text x="160" y="258" textAnchor="middle">horizontal PC1 ({percent(wineFullFit.standardizedRatios[0])}) · vertical PC2 ({percent(wineFullFit.standardizedRatios[1])})</text>
    </svg>
    <div className="pca-buttons"><button type="button" onClick={() => setShowCultivar(!showCultivar)}>{showCultivar ? 'Hide cultivar labels' : 'Overlay the cultivar labels (not used by PCA)'}</button><span>{showCultivar ? 'circle = cultivar 1, square = 2, triangle = 3' : 'Inspect the unsupervised geometry first.'}</span></div>
    <p>This picture retains {percent(wineFullFit.standardizedRatios[0] + wineFullFit.standardizedRatios[1])} of the standardized variance; close points in this plane can still differ in the eleven omitted coordinates.</p>
    <div className="pca-bars" role="img" aria-label={`Signed PC1 coefficients for the 13 standardized features: ${order.map(([weight, index]) => `${wineFeatures[index]} ${number(weight)}`).join('; ')}.`}>
      {order.map(([weight, index]) => <div key={index} className="pca-bar-row"><span>{wineFeatures[index]}</span><span className="pca-bar-track"><span className="pca-bar-marker" style={{ left: '50%' }} /><span className={`pca-bar-fill ${weight < 0 ? 'is-residual' : ''}`} style={{ left: weight < 0 ? `${50 + 50 * weight / 0.5}%` : '50%', width: `${50 * Math.abs(weight) / 0.5}%` }} /></span><span className="pca-bar-value">{number(weight)}</span></div>)}
    </div>
    <p>PC1's unit direction, one signed coefficient per standardized feature, sorted. The bar axis runs from −0.5 to 0.5 about the central zero marker. Flavanoids, total phenols, od280/od315 and proanthocyanins pull one way; nonflavanoid phenols, malic acid and alcalinity pull the other. A coefficient says how the score is computed from a feature; it does not name a cause.</p>
  </figure>;
}

/** F5: a biplot read with one observation and one feature arrow on the four-point fixture. */
export function BiplotReadingFigure() {
  const z = [-3 / Math.SQRT2, -1 / Math.SQRT2];
  const arrow = [Math.SQRT1_2, Math.SQRT1_2];
  const x = value => 150 + 40 * value, y = value => 110 - 40 * value;
  return <figure className="pca-figure pca-standalone">
    <figcaption><strong>Reading a coefficient, a correlation and a biplot</strong> · four-point fixture, observation A and the first feature</figcaption>
    <svg viewBox="0 0 300 234" role="img" aria-label="Score coordinates of A, B, C and D on PC1 and PC2, with the first feature's arrow (0.7071, 0.7071). A's dot product with the arrow is −2, the centered first reading.">
      {[-2, 0, 2].map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1="20" y2="200" className="pca-grid" /><line x1="30" x2="270" y1={y(value)} y2={y(value)} className="pca-grid" /><text x={x(value)} y="212" textAnchor="middle">{value}</text><text x="24" y={y(value) + 4} textAnchor="end">{value}</text></g>)}
      {[[-3 / Math.SQRT2, -1 / Math.SQRT2, 'A'], [-3 / Math.SQRT2, 1 / Math.SQRT2, 'B'], [3 / Math.SQRT2, -1 / Math.SQRT2, 'C'], [3 / Math.SQRT2, 1 / Math.SQRT2, 'D']].map(([px, py, name]) => <g key={name}><circle className={`pca-point ${name === 'A' ? 'is-selected' : ''}`} cx={x(px)} cy={y(py)} r="4.5" /><text x={x(px) + 7} y={y(py) - 6}>{name}</text></g>)}
      <line x1={x(0)} y1={y(0)} x2={x(2 * arrow[0])} y2={y(2 * arrow[1])} stroke="#8eb9a5" strokeWidth="2" />
      <text x={x(2 * arrow[0]) - 4} y={y(2 * arrow[1]) - 6} fill="#8eb9a5" textAnchor="end">feature 1 arrow (×2)</text>
      <text x="150" y="230" textAnchor="middle">horizontal: PC1 score · vertical: PC2 score</text>
    </svg>
    <p>A's scores are z = ({number(z[0])}, {number(z[1])}). The first feature's arrow is its two direction coefficients, a₁ = (0.7071, 0.7071); the drawing doubles its length and says so. Their dot product is {number(z[0] * arrow[0] + z[1] * arrow[1])}, exactly A's centered first reading; adding the mean 3 recovers the original 1. That is what this biplot scaling means: observation-dot-arrow gives the rank-two reconstruction of a centered feature.</p>
    <Table caption="Two different numbers about the same feature and component" headings={['quantity', 'value', 'what it answers']} rows={[['PC1 direction coefficient for feature 1', '0.7071', 'How much one unit of the centered feature contributes to the score'], ['correlation between feature 1 and the PC1 score', number(Math.sqrt(0.9)), 'How linearly the feature tracks the score across observations'], ['arrow angle to the PC1 axis', '45°', 'Layout under this scaling; not a general formula for feature correlations']]} />
  </figure>;
}

/** F6: sample spectrum of independent Gaussian noise against the flat population share. */
export function GaussianSpectrumFigure() {
  const x = index => 34 + index * 13.5;
  const y = value => 190 - 900 * value;
  return <figure className="pca-figure pca-standalone">
    <figcaption><strong>Sample spectrum versus population symmetry</strong> · seeded simulation, 40 rows × 20 independent standard normal features</figcaption>
    <svg viewBox="0 0 320 225" role="img" aria-label={`Twenty sample explained-variance ratios from ${number(gaussianSpectrum.ratios[0])} down to ${number(gaussianSpectrum.ratios[19])}, against a flat population share of 0.05 per direction. The first two sum to ${number(gaussianSpectrum.ratios[0] + gaussianSpectrum.ratios[1])}.`}>
      {[0, 0.05, 0.1, 0.15].map(value => <g key={value}><line x1="30" x2="305" y1={y(value)} y2={y(value)} className="pca-grid" /><text x="26" y={y(value) + 4} textAnchor="end">{value}</text></g>)}
      {gaussianSpectrum.ratios.map((ratio, index) => <rect key={index} x={x(index) - 5} y={y(ratio)} width="10" height={190 - y(ratio)} fill={index < 2 ? '#e7b94a' : '#91aecf'} />)}
      <line x1="30" x2="305" y1={y(0.05)} y2={y(0.05)} stroke="#f2e7ca" strokeDasharray="5 4" />
      <text x="300" y={y(0.05) - 4} fill="#f2e7ca" textAnchor="end">population share 0.05 each</text>
      {[1, 5, 10, 15, 20].map(rank => <text key={rank} x={x(rank - 1)} y="205" textAnchor="middle">{rank}</text>)}
      <text x="170" y="220" textAnchor="middle">component rank · share of sample variance</text>
    </svg>
    <p>The population covariance is the identity: every direction holds 5% of the population variance, the dashed line. The fitted sample components are sorted, so the leading ones exceed that line by construction and the trailing ones fall below it. The first two hold {percent(gaussianSpectrum.ratios[0] + gaussianSpectrum.ratios[1])} of this sample's variance without any two-dimensional cause. Another seed gives different bars; the flat line is a benchmark, not a significance threshold.</p>
  </figure>;
}

/** F7: score along the usual mode versus residual away from it, for two constructed observations. */
export function ResidualAlarmFigure() {
  const v = [Math.SQRT1_2, Math.SQRT1_2];
  const observations = [{ name: 'P', point: [3, 3] }, { name: 'Q', point: [3, -3] }].map(entry => {
    const score = entry.point[0] * v[0] + entry.point[1] * v[1];
    const foot = [score * v[0], score * v[1]];
    const residual = [entry.point[0] - foot[0], entry.point[1] - foot[1]];
    return { ...entry, score, foot, residual, residualSquared: residual[0] ** 2 + residual[1] ** 2 };
  });
  const x = value => 150 + 22 * value, y = value => 110 - 22 * value;
  return <figure className="pca-figure pca-standalone">
    <figcaption><strong>The residual reveals a different anomaly</strong> · constructed two-sensor system whose normal readings lie near the diagonal</figcaption>
    <svg viewBox="0 0 300 232" role="img" aria-label="A diagonal band of normal readings through zero. P at (3, 3) lies on the band with score 4.24 and zero residual. Q at (3, −3) has zero score and residual squared length 18.">
      {[-4, -2, 0, 2, 4].map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1="10" y2="210" className="pca-grid" /><line x1="40" x2="260" y1={y(value)} y2={y(value)} className="pca-grid" /></g>)}
      <polygon points={`${x(-4.5)},${y(-5.1)} ${x(4.5)},${y(3.9)} ${x(4.5)},${y(5.1)} ${x(-4.5)},${y(-3.9)}`} fill="#8eb9a5" fillOpacity=".15" />
      <line className="pca-line" x1={x(-4.5)} y1={y(-4.5)} x2={x(4.5)} y2={y(4.5)} />
      {observations.map(entry => <g key={entry.name}>
        <line className="pca-residual" x1={x(entry.point[0])} y1={y(entry.point[1])} x2={x(entry.foot[0])} y2={y(entry.foot[1])} />
        <circle className="pca-foot" cx={x(entry.foot[0])} cy={y(entry.foot[1])} r="3" />
        <circle className="pca-point is-selected" cx={x(entry.point[0])} cy={y(entry.point[1])} r="5" />
        <text x={x(entry.point[0]) + 8} y={y(entry.point[1]) + 19}>{entry.name} ({entry.point.join(', ')})</text>
      </g>)}
      <text x="150" y="228" textAnchor="middle">band: usual readings · line: fixed direction (1, 1)/√2</text>
    </svg>
    <Table caption="Two diagnostics for each observation, against the fixed usual direction" headings={['observation', 'score along the usual mode', 'residual squared length']} rows={observations.map(entry => [`${entry.name} ${entry.point.join(', ')}`, number(entry.score), number(entry.residualSquared)])} />
    <p>P travels far along the usual mode and leaves no residual. Q has a score of exactly zero, so a score-only display would call it unremarkable, yet its residual squared length is 18: the two sensors disagree as hard as they can at that magnitude.</p>
  </figure>;
}
