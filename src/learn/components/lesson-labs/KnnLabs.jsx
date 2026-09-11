import { useId, useMemo, useState } from 'react';
import { Investigation, Predict } from './LessonInvestigation.jsx';
import { LessonTable } from './LessonElements.jsx';
import { neighborRows, neighborReport, unitsReport, regressionRows, localMean, kdSearch, candidateReport, volumeReport } from '../../data/knn-models.js';
import './knn-labs.css';
import { knnValidationRows } from '../../data/knn-validation-data.js';
const fixed = (value, digits = 3) => value.toFixed(digits);
const classColors = {
  A: '#edc269',
  B: '#8dcae4',
  C: '#c6a6e8'
};
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 1
}) {
  const id = useId();
  return <label htmlFor={id}><span>{label}: <output>{value}</output></span><input id={id} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Plot({
  title,
  description,
  children
}) {
  return <figure className="knn-figure"><div className="knn-plot" tabIndex={0} role="region" aria-label={`${title}; horizontally scrollable`}><svg viewBox="0 0 400 400" role="img" aria-label={title}><title>{title}</title><desc>{description}</desc>{children}</svg></div><p className="knn-scroll-note">On a narrow screen, scroll the figure sideways to see the full axes; labels keep their readable size.</p><figcaption>{description}</figcaption></figure>;
}
function Point({
  row,
  x,
  y,
  selected = true
}) {
  return <g opacity={selected ? 1 : 0.38} fill={classColors[row.label]} stroke="#11171b" strokeWidth="1.3">{row.label === 'A' ? <circle cx={x} cy={y} r="5" /> : row.label === 'B' ? <rect x={x - 5} y={y - 5} width="10" height="10" /> : <path d={`M${x},${y - 6}l6,11h-12Z`} />}<text x={x + 8} y={y + 4} stroke="none">{row.id}</text></g>;
}
function NeighborMap({
  query,
  selected = neighborRows,
  radius,
  metric = 'euclidean',
  split = null,
  title = 'Select nearby cases in the same coordinate system'
}) {
  const clipId = useId();
  const x = value => 55 + value * 50;
  const y = value => 345 - value * 50;
  const boundary = metric === 'manhattan' ? [[radius, 0], [0, radius], [-radius, 0], [0, -radius]] : [[radius, radius], [-radius, radius], [-radius, -radius], [radius, -radius]];
  return <Plot title={title} description="Both axes use the same scale. Circles identify class A, squares B and triangles C. The white cross is the new query; bright row IDs are the selected cases. The outlined distance contour, when present, passes through the kth neighbor.">
    <defs><clipPath id={clipId}><rect x="55" y="45" width="300" height="300" /></clipPath></defs>
    {[0, 2, 4, 6].map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1="45" y2="345" className="knn-grid" /><line x1="55" x2="355" y1={y(value)} y2={y(value)} className="knn-grid" /><text x={x(value)} y="367" textAnchor="middle">{value}</text><text x="45" y={y(value) + 4} textAnchor="end">{value}</text></g>)}
    <g clipPath={`url(#${clipId})`}>
      {radius !== undefined && (metric === 'euclidean' ? <circle cx={x(query[0])} cy={y(query[1])} r={radius * 50} className="knn-contour" /> : <polygon points={boundary.map(([dx, dy]) => `${x(query[0] + dx)},${y(query[1] + dy)}`).join(' ')} className="knn-contour" />)}
      {split && (split.axis === 0 ? <line x1={x(split.split)} x2={x(split.split)} y1="45" y2="345" className="knn-split" /> : <line x1="55" x2="355" y1={y(split.split)} y2={y(split.split)} className="knn-split" />)}
      {selected.map(row => <line key={row.id} x1={x(query[0])} y1={y(query[1])} x2={x(row.point[0])} y2={y(row.point[1])} className="knn-link" />)}
    </g>
    {neighborRows.map(row => <Point key={row.id} row={row} x={x(row.point[0])} y={y(row.point[1])} selected={selected.some(other => other.id === row.id)} />)}
    <path d={`M${x(query[0]) - 6},${y(query[1]) - 6}l12,12m-12,0l12,-12`} stroke="white" strokeWidth="2.5" />
    <text x="205" y="393" textAnchor="middle">Feature x₁</text><text x="15" y="195" transform="rotate(-90 15 195)" textAnchor="middle">Feature x₂</text>
  </Plot>;
}
export function KnnLifecycleFigure() {
  return <figure className="knn-flow"><div><strong>Fit on training rows</strong><span>learn allowed transforms</span><span>↓ store transformed rows + targets</span><span>build an index if useful</span></div><div><strong>Answer one new query</strong><span>apply those same transforms</span><span>↓ retrieve neighbor IDs/distances</span><span>aggregate their known targets</span></div><figcaption>The new target is unavailable. A search index accelerates retrieval; the distance, k and target aggregation define the prediction rule.</figcaption></figure>;
}

export function KnnValidationFigure() {
  const x = value => 55 + value * 300 / 45;
  const y = value => 345 - value * 275 / 3.5;
  return <div aria-label="KNN validation log loss from the recorded experiment">
    <Plot title="Eight actually fitted candidates on the same validation rows" description="Dots are the recorded validation log losses from the complete seed19/noise0.28 experiment below, rounded to six decimals. Gold circles use uniform weights; blue squares use distance weights. Connecting segments only guide the eye between tested integer k values; they are not unmeasured intermediate results.">
      {[0, 1, 2, 3].map(value => <g key={value}><line x1="55" x2="355" y1={y(value)} y2={y(value)} className="knn-grid" /><text x="45" y={y(value) + 4} textAnchor="end">{value}</text></g>)}
      {[1, 5, 15, 41].map(value => <text key={value} x={x(value)} y="369" textAnchor="middle">{value}</text>)}
      {['uniform', 'distance'].map(weights => <g key={weights}><polyline points={knnValidationRows.filter(row => row.weights === weights).map(row => `${x(row.k)},${y(row.loss)}`).join(' ')} fill="none" stroke={weights === 'uniform' ? '#edc269' : '#8dcae4'} strokeDasharray={weights === 'uniform' ? undefined : '5 4'} strokeWidth="2" />{knnValidationRows.filter(row => row.weights === weights).map(row => weights === 'uniform' ? <circle key={row.k} cx={x(row.k)} cy={y(row.loss)} r="5" fill="#edc269" /> : <rect key={row.k} x={x(row.k) - 3} y={y(row.loss) - 3} width="6" height="6" fill="#8dcae4" />)}</g>)}
      <text x="65" y="30">Uniform ○ · Distance □</text><text x="205" y="394" textAnchor="middle">Tested neighbors k</text><text x="15" y="195" transform="rotate(-90 15 195)" textAnchor="middle">Validation log loss</text>
    </Plot>
    <LessonTable caption="Actual validation candidates; smaller log loss is better" headers={['k', 'Weighting', 'Validation log loss']} rows={knnValidationRows.map(row => [row.k, row.weights, fixed(row.loss, 6)])} />
    <p>This experiment's tested losses mostly decrease across the chosen range. A mandatory U-shaped illustration would misrepresent these results. The test set supplies no point in this selection chart.</p>
  </div>;
}
export function NeighborVotingLab() {
  const [queryX, setQueryX] = useState(3.1);
  const [queryY, setQueryY] = useState(2.9);
  const [k, setK] = useState(5);
  const [metric, setMetric] = useState('euclidean');
  const [weights, setWeights] = useState('uniform');
  const report = neighborReport({
    query: [queryX, queryY],
    k,
    metric,
    weights
  });
  function reset() {
    setQueryX(3.1);
    setQueryY(2.9);
    setK(5);
    setMetric('euclidean');
    setWeights('uniform');
  }
  return <Investigation id="knn-neighbors" kicker="RETRIEVE → WEIGH → DECIDE" title="A close minority can lose the count and win the weighted vote">
    <Predict>Two C points nearly touch the query. Will five equal votes select C? Change only the weighting after predicting.</Predict>
    <div className="knn-controls"><Range label="Query x1" value={queryX} onChange={setQueryX} min={0} max={6} step={0.1} /><Range label="Query x2" value={queryY} onChange={setQueryY} min={0} max={6} step={0.1} /><Range label="Neighbors k" value={k} onChange={setK} min={1} max={8} /><label>Distance<select value={metric} onChange={event => setMetric(event.target.value)}><option value="euclidean">Euclidean — circle</option><option value="manhattan">Manhattan — diamond</option><option value="maximum">Maximum — square</option></select></label><label>Vote weighting<select value={weights} onChange={event => setWeights(event.target.value)}><option value="uniform">Equal weights</option><option value="distance">Inverse distance</option></select></label><button onClick={reset}>Reset</button></div>
    <NeighborMap query={[queryX, queryY]} selected={report.neighbors} radius={report.radius} metric={metric} />
    <div className="knn-votes" aria-label="Calculated class fractions">{report.votes.map(row => <div key={row.label}><span>Class {row.label}: {fixed(row.probability)}</span><span className="knn-bar"><span style={{
            width: `${100 * row.probability}%`,
            background: classColors[row.label]
          }} /></span></div>)}</div>
    <p className="knn-result" aria-live="polite">Prediction: <strong>{report.label}</strong>. Neighborhood radius {fixed(report.radius)}. {report.exact && weights === 'distance' ? 'Exact matches share all weight; other selected rows get zero.' : 'The fractions describe this neighborhood; calibration is a separate question.'}</p>
    <LessonTable caption="All eight distances; selected weights normalized to sum to one" headers={['Row', 'Coordinates', 'Distance', 'Selected weight']} rows={report.ranked.map(row => [row.id, row.point.join(', '), fixed(row.distance, 6), fixed(report.neighbors.find(other => other.id === row.id)?.weight || 0)])} />
    <p>For reproducibility in this decimal fixture, distances rounded to 12 places break retrieval ties by row ID; probabilities within 10⁻¹² tie by A, B, C. This numerical convention is separate from the mathematical distance. A point on the contour is not necessarily selected when several rows tie at the cutoff.</p>
  </Investigation>;
}
export function NeighborUnitsLab() {
  const [standardize, setStandardize] = useState(false);
  const report = unitsReport(standardize);
  return <Investigation id="knn-units" kicker="THE UNITS CHOOSE WHAT LOOKS CLOSE" title="Inspect which feature pays for the distance">
    <Predict>The query is 1.2 hours and 600 Wh. Raw numbers favor a case with closer energy. Will fitting a scale on the three training rows change that?</Predict>
    <div className="knn-controls"><label>Coordinate rule<select value={String(standardize)} onChange={event => setStandardize(event.target.value === 'true')}><option value="false">Raw hours and Wh</option><option value="true">Training standard deviations</option></select></label><button onClick={() => setStandardize(false)}>Reset</button></div>
    <p>Training scales: {fixed(report.scales[0])} hours and {fixed(report.scales[1])} Wh. The query never participates in fitting these scales.</p>
    <div className="knn-contributions">{report.rows.map(row => {
        const sum = row.contributions[0] + row.contributions[1];
        return <div key={row.id}><strong>{row.id}: {row.point[0]} hours, {row.point[1]} Wh</strong><div className="knn-stacked" aria-label={`Squared contributions: time ${fixed(row.contributions[0])}, energy ${fixed(row.contributions[1])}`}><span style={{
              width: `${100 * row.contributions[0] / sum}%`
            }} /><span style={{
              width: `${100 * row.contributions[1] / sum}%`
            }} /></div><span>Time term {fixed(row.contributions[0])} + energy term {fixed(row.contributions[1])} = {fixed(sum)}. Distance {fixed(row.distance)}.</span></div>;
      })}</div>
    <p aria-live="polite">Nearest: <strong>{report.nearest.id} ({report.nearest.label})</strong>. Bar proportions show each row's own squared-distance composition, not comparable total bar magnitudes.</p>
    <p>Gold is the time contribution; blue is energy. Standardization changes the model's similarity judgment. It does not establish that time and energy deserve equal predictive influence.</p>
  </Investigation>;
}
export function CosineDirectionFigure() {
  return <Plot title="Direction and magnitude give different neighbors" description="For query (1,0), case U=(3,0) has cosine distance0 but Euclidean distance2. Case V=(1,1) has cosine distance1−1/√2≈0.293 but Euclidean distance1. Normalizing removes the magnitude difference; zero vectors require an explicit policy.">
    <line x1="60" x2="350" y1="330" y2="330" className="knn-grid" /><line x1="60" x2="60" y1="70" y2="330" className="knn-grid" />
    <line x1="60" y1="330" x2="300" y2="330" stroke="#edc269" strokeWidth="4" /><line x1="60" y1="330" x2="140" y2="250" stroke="#8dcae4" strokeWidth="3" />
    <circle cx="140" cy="330" r="6" fill="white" /><circle cx="300" cy="330" r="6" fill="#edc269" /><circle cx="140" cy="250" r="6" fill="#8dcae4" />
    <text x="128" y="355">q (1,0)</text><text x="260" y="310">U (3,0)</text><text x="150" y="246">V (1,1)</text><text x="60" y="50">Same angle ≠ same length</text><text x="200" y="391" textAnchor="middle">Two coordinates, equal units</text>
  </Plot>;
}
export function LocalRegressionLab() {
  const [query, setQuery] = useState(2.5);
  const [k, setK] = useState(3);
  const [weights, setWeights] = useState('uniform');
  const report = localMean(query, k, weights);
  const x = value => 55 + (value + 1) * 300 / 9;
  const y = value => 345 - value * 10;
  return <Investigation id="knn-regression" kicker="AVERAGE TARGETS IN A LOCAL SET" title="Move beyond the training range without inventing extrapolation">
    <Predict>Observed targets are x² at integer x from 0 to 5. At x=8, can their nonnegative weighted average predict64?</Predict>
    <div className="knn-controls"><Range label="Regression query" value={query} onChange={setQuery} min={-1} max={8} step={0.5} /><Range label="Regression neighbors" value={k} onChange={setK} min={1} max={6} /><label>Regression weights<select value={weights} onChange={event => setWeights(event.target.value)}><option value="uniform">Uniform</option><option value="distance">Inverse distance</option></select></label><button onClick={() => {
        setQuery(2.5);
        setK(3);
        setWeights('uniform');
      }}>Reset</button></div>
    <Plot title="Observed targets and one local mean" description="Gold circles are selected observations; dim circles remain in training data but do not contribute. The white diamond is the current weighted prediction at the query. Dashed links show which target values contribute; they are not an interpolated fitted curve.">
      {[0, 10, 20, 30].map(value => <g key={value}><line x1="55" x2="355" y1={y(value)} y2={y(value)} className="knn-grid" /><text x="45" y={y(value) + 4} textAnchor="end">{value}</text></g>)}
      {[-1, 2, 5, 8].map(value => <text key={value} x={x(value)} y="368" textAnchor="middle">{value}</text>)}
      {report.rows.map(row => <line key={row.id} x1={x(row.x)} y1={y(row.y)} x2={x(query)} y2={y(report.prediction)} className="knn-link" />)}
      {regressionRows.map(row => <circle key={row.id} cx={x(row.x)} cy={y(row.y)} r="5" fill={report.rows.some(other => other.id === row.id) ? '#edc269' : '#4a555c'} />)}
      <path d={`M${x(query)},${y(report.prediction) - 7}l7,7l-7,7l-7,-7Z`} fill="white" />
      <text x="205" y="394" textAnchor="middle">Query feature x</text><text x="15" y="195" transform="rotate(-90 15 195)" textAnchor="middle">Target units</text>
    </Plot>
    <p aria-live="polite">Prediction = <strong>{fixed(report.prediction, 6)}</strong>. Training targets range from 0 to 25. The example's generating x² rule is unknown to the neighbor estimator.</p>
    <LessonTable caption="Multiply each selected target by its weight" headers={['Row x', 'Target', 'Distance', 'Weight', 'Contribution']} rows={report.rows.map(row => [row.x, row.y, fixed(row.distance), fixed(row.weight), fixed(row.y * row.weight)])} />
    <p>Equal-distance rows prefer the smaller row ID. With inverse weights, exact matches share the full mass. Uniform k=6 returns55/6 everywhere; inverse-weighted k=6 generally does not.</p>
  </Investigation>;
}
export function KdTreeSearchLab() {
  const [query, setQuery] = useState('near-a');
  const [step, setStep] = useState(0);
  const points = {
    'near-a': [1.3, 2.1],
    'near-b': [5.1, 4.9],
    'between': [3.1, 2.9]
  };
  const report = useMemo(() => kdSearch(points[query]), [query]);
  const event = report.events[Math.min(step, report.events.length - 1)];
  const visited = report.events.slice(0, step + 1).filter(row => row.kind === 'visit').map(row => neighborRows.find(other => other.id === row.id));
  return <Investigation id="knn-kdtree" kicker="A LOWER BOUND JUSTIFIES EACH SKIP" title="Search a median tree without guessing where the winner is">
    <Predict>A branch can contain several unseen points. What inequality makes it safe to skip all of them?</Predict>
    <div className="knn-controls"><label>Search query<select value={query} onChange={event => {
          setQuery(event.target.value);
          setStep(0);
        }}><option value="near-a">(1.3,2.1)</option><option value="near-b">(5.1,4.9)</option><option value="between">(3.1,2.9)</option></select></label><button disabled={step === 0} onClick={() => setStep(value => value - 1)}>Previous step</button><button disabled={step >= report.events.length - 1} onClick={() => setStep(value => value + 1)}>Next step</button><button onClick={() => {
        setQuery('near-a');
        setStep(0);
      }}>Reset</button></div>
    <NeighborMap query={points[query]} selected={visited} split={event} title="Visited rows and the current split plane" />
    <p className="knn-result" aria-live="polite">Step {step + 1}/{report.events.length}: {event.explanation} Best distance {fixed(event.best.distance, 6)}.</p>
    <p>The dashed blue line extends the current node's split plane across the drawing to show its distance bound. It is not the entire tree partition. Bright rows have actually been visited; the full trace finds {report.best.id} with {report.visits} distance evaluations rather than 8 for this query.</p>
    <details><summary>Read the trace so far</summary><ol>{report.events.slice(0, step + 1).map((row, index) => <li key={index}>{row.explanation}</li>)}</ol></details>
  </Investigation>;
}
export function CandidateRecallLab() {
  const [mode, setMode] = useState('all');
  const report = candidateReport(mode);
  return <Investigation id="knn-candidates" kicker="RETRIEVAL QUALITY ≠ PREDICTION QUALITY" title="An approximate search can miss the evidence that mattered">
    <Predict>If the candidate stage drops the two nearby C cases, can exact reranking inside the remaining pool recover them?</Predict>
    <div className="knn-controls"><label>Candidate pool<select value={mode} onChange={event => setMode(event.target.value)}><option value="all">All eight rows</option><option value="lose-close-c">Drop both close C rows</option><option value="keep-close-c">Drop far B rows</option></select></label><button onClick={() => setMode('all')}>Reset</button></div>
    <div className="knn-candidate-columns"><div><strong>Exact three neighbors</strong>{report.exact.neighbors.map(row => <span key={row.id}>{row.id} · d={fixed(row.distance)}</span>)}<span>Inverse-weighted class {report.exact.label}</span></div><div><strong>Candidate reranking</strong>{report.candidate.neighbors.map(row => <span key={row.id}>{row.id} · d={fixed(row.distance)}</span>)}<span>Inverse-weighted class {report.candidate.label}</span></div></div>
    <p aria-live="polite">Search pool {report.candidates.length}/8 rows; recall@3 = <strong>{fixed(report.recall)}</strong>. Final class {report.candidate.label}.</p>
    <p>This is a declared candidate restriction to isolate recall. It is not a simulation of HNSW or IVF training, and the row count is not measured latency. Real approximate search must evaluate both neighbor recall and the task's held-out result.</p>
  </Investigation>;
}
export function NeighborhoodVolumeLab() {
  const [dimension, setDimension] = useState(10);
  const [fraction, setFraction] = useState(0.01);
  const report = volumeReport(dimension, fraction);
  return <Investigation id="knn-volume" kicker="A SPECIFIED VOLUME MODEL" title="A small volume can require a wide neighborhood">
    <Predict>A cube holds1% of a uniform unit cube's mass. In ten dimensions, is its side closer to 0.1 or0.6?</Predict>
    <div className="knn-controls"><Range label="Dimensions" value={dimension} onChange={setDimension} min={1} max={100} /><label>Target volume fraction<select value={fraction} onChange={event => setFraction(Number(event.target.value))}><option value={0.01}>1%</option><option value={0.1}>10%</option><option value={0.5}>50%</option></select></label><button onClick={() => {
        setDimension(10);
        setFraction(0.01);
      }}>Reset</button></div>
    <Plot title="Calculated side length versus dimension" description="The curve is s=f^(1/d) for an axis-aligned cube with volume fraction f inside a uniform unit cube. The point uses the selected integer dimension. This is a volume identity, not an observed nearest-neighbor distance or a timing benchmark.">
      {[0, 0.5, 1].map(value => <g key={value}><line x1="55" x2="355" y1={345 - value * 300} y2={345 - value * 300} className="knn-grid" /><text x="45" y={349 - value * 300} textAnchor="end">{value}</text></g>)}
      {[1, 25, 50, 75, 100].map(value => <text key={value} x={55 + (value - 1) * 300 / 99} y="370" textAnchor="middle">{value}</text>)}
      <polyline points={Array.from({
        length: 100
      }, (_, index) => `${55 + index * 300 / 99},${345 - fraction ** (1 / (index + 1)) * 300}`).join(' ')} fill="none" stroke="#edc269" strokeWidth="2.5" />
      <circle cx={55 + (dimension - 1) * 300 / 99} cy={345 - report.side * 300} r="5" fill="white" />
      <text x="205" y="394" textAnchor="middle">Dimensions d</text><text x="15" y="195" transform="rotate(-90 15 195)" textAnchor="middle">Side length s</text>
    </Plot>
    <p aria-live="polite">Side length {fixed(report.side, 6)}; raising this side to power{dimension} gives{fraction}. To place10 observations in expectation in a fixed side-0.1 cube requires n=10^{report.log10RowsForTenNeighborsAtSideOneTenth} in this model.</p>
    <p>Nonuniform density, boundaries, correlations and lower-dimensional support can change the neighborhood story. There is no universal feature-count cutoff where all nearest-neighbor methods stop working.</p>
  </Investigation>;
}
