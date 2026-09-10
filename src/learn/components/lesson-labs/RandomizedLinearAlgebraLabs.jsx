import { useId, useState } from 'react';
import { weightedProbeState, spectralSketchState, sketchSpectra, rowSketchState, traceProbeState, traceMatrices } from '../../data/randomized-linear-algebra-models.js';
import './randomized-linear-algebra-labs.css';
const number = value => value === null ? 'undefined' : String(Number(value.toFixed(4)));
const pair = vector => `[${vector.map(number).join(', ')}]`;
const colors = ['#e4bd6d', '#9ec7df', '#b5cda6'];
function MatrixTable({
  matrix,
  label
}) {
  return <div className="rla-matrix-wrap" tabIndex={0} role="region" aria-label={label}>
    <table className="rla-matrix"><caption>{label}</caption><tbody>{matrix.map((row, rowIndex) => <tr key={rowIndex}>{row.map((value, column) => <td key={column}>{number(value)}</td>)}</tr>)}</tbody></table>
  </div>;
}
function Plane({
  title,
  description,
  range,
  segments = [],
  points = [],
  direction = null
}) {
  const id = useId();
  const screen = point => [160 + 118 * point[0] / range, 160 - 118 * point[1] / range];
  return <svg viewBox="0 0 320 320" className="rla-plane" role="img" aria-labelledby={`${id}-title ${id}-description`}>
    <title id={`${id}-title`}>{title}</title><desc id={`${id}-description`}>{description} Equal horizontal and vertical scales, from minus {range} to plus {range}. Numerical values follow.</desc>
    {[-1, -0.5, 0.5, 1].map(fraction => <g key={fraction} className="rla-grid">
      <line x1={160 + fraction * 118} x2={160 + fraction * 118} y1="32" y2="288" />
      <line x1="32" x2="288" y1={160 - fraction * 118} y2={160 - fraction * 118} />
      <text x={160 + fraction * 118} y="177" textAnchor="middle">{number(range * fraction)}</text>
      <text x="153" y={164 - fraction * 118} textAnchor="end">{number(range * fraction)}</text>
    </g>)}
    <path className="rla-axis" d="M 28 160 H 292 M 160 28 V 292" />
    <text x="301" y="155">x</text><text x="166" y="20">y</text>
    {direction && <line x1={screen(direction.map(value => -range * value))[0]} y1={screen(direction.map(value => -range * value))[1]} x2={screen(direction.map(value => range * value))[0]} y2={screen(direction.map(value => range * value))[1]} stroke="#adca9b" strokeDasharray="5 4" />}
    {segments.map(({
      start,
      end,
      color,
      dashed,
      label
    }, index) => {
      const first = screen(start);
      const last = screen(end);
      const angle = Math.atan2(last[1] - first[1], last[0] - first[0]);
      const head = [last, [last[0] - 8 * Math.cos(angle - 0.4), last[1] - 8 * Math.sin(angle - 0.4)], [last[0] - 8 * Math.cos(angle + 0.4), last[1] - 8 * Math.sin(angle + 0.4)]];
      return <g key={index}><title>{label}: {pair(start)} to {pair(end)}</title>
        <line x1={first[0]} y1={first[1]} x2={last[0]} y2={last[1]} stroke={color} strokeWidth={dashed ? 1.5 : 2.5} strokeDasharray={dashed ? '3 4' : undefined} />
        {!dashed && Math.hypot(last[0] - first[0], last[1] - first[1]) > 1 && <polygon points={head.map(point => point.join(',')).join(' ')} fill={color} />}
      </g>;
    })}
    {points.map(({
      value,
      label,
      hollow
    }, index) => <g key={index}>
      <circle cx={screen(value)[0]} cy={screen(value)[1]} r="4" fill={hollow ? '#11151b' : colors[0]} stroke={hollow ? colors[2] : colors[0]} strokeWidth="2" />
      <title>{label}: {pair(value)}</title>
    </g>)}
  </svg>;
}
export function WeightedColumnProbeLab() {
  const [weights, setWeights] = useState([1, 1, 0]);
  const state = weightedProbeState(weights);
  let current = [0, 0];
  const path = state.contributions.map((contribution, index) => {
    const start = current;
    current = current.map((value, row) => value + contribution[row]);
    return {
      start,
      end: current,
      color: colors[index],
      label: `Weighted column ${index + 1}`
    };
  });
  const range = Math.max(4, Math.ceil(Math.max(...path.flatMap(segment => [...segment.start, ...segment.end].map(Math.abs)))));
  return <section className="rla-lab" aria-label="Weighted column probe investigation">
    <p className="rla-eyebrow">A RANDOM PROBE IS A COLUMN COMBINATION</p>
    <h3>What does one probe reveal—or hide?</h3>
    <p>The matrix columns are [3, 0], [0, 1] and [3, 1]. Choose weights and follow the three colored contributions from tail to tip. Predict what happens with weights [1, 1, −1].</p>
    <div className="rla-controls">{weights.map((value, index) => <label key={index}>Column {index + 1} weight
      <select value={value} onChange={event => setWeights(previous => previous.map((weight, position) => position === index ? Number(event.target.value) : weight))}>
        {[-2, -1, 0, 1, 2].map(weight => <option key={weight} value={weight}>{weight}</option>)}
      </select></label>)}<button onClick={() => setWeights([1, 1, 0])}>Reset probe</button></div>
    <div className="rla-plot-pair"><figure>
      <Plane title="Add the weighted columns" description="Three colored vectors are placed tail to tip. The gold point is their sum, A omega. A changing axis range keeps every contribution visible." range={range} segments={path} points={[{
          value: state.output,
          label: 'Probe output'
        }]} />
      <figcaption>Gold → blue → green contributions sum to <strong>{pair(state.output)}</strong>. Axes rescale together as needed.</figcaption>
    </figure><figure>
      <Plane title="Project the original columns onto the observed line" description="Solid gold arrows are original columns; hollow green points are projections. Dotted segments show the information discarded by this one-dimensional approximation." range={4} direction={state.direction} segments={state.columns.flatMap((column, index) => [{
          start: [0, 0],
          end: column,
          color: colors[0],
          label: `Column ${index + 1}`
        }, {
          start: column,
          end: state.projected[index],
          color: '#a9b0b8',
          dashed: true,
          label: `Residual ${index + 1}`
        }])} points={state.projected.map((value, index) => ({
          value,
          hollow: true,
          label: `Projection ${index + 1}`
        }))} />
      <figcaption>Squared lengths of the discarded segments sum to <strong>{number(state.residualSquared)}</strong>.</figcaption>
    </figure></div>
    <p className="rla-feedback" aria-live="polite">{state.direction ? `The observed unit direction is ${pair(state.direction)}. One probe describes a line, even though these columns span a plane.` : 'The combination cancelled to zero. There is no observed direction to normalize; this says nothing about whether the original matrix is zero. The displayed zero approximation discards all columns.'}</p>
    <details><summary>Read the column calculations</summary><ol>{state.columns.map((column, index) => <li key={index}>
      {weights[index]} × {pair(column)} = {pair(state.contributions[index])}; projected column {pair(state.projected[index])}.
    </li>)}</ol></details>
    <p className="rla-caption">These selectable integer probes make cancellation visible. Gaussian random probes in the algorithm below have continuous distributions; exact cancellation of a nonzero map has probability zero under that ideal model, but poor coverage can still occur.</p>
  </section>;
}
export function SpectralSketchLab() {
  const [preset, setPreset] = useState('fast');
  const [rank, setRank] = useState(2);
  const [oversampling, setOversampling] = useState(1);
  const [iterations, setIterations] = useState(0);
  const [seed, setSeed] = useState(7);
  const state = spectralSketchState(preset, rank, oversampling, iterations, seed);
  const percent = value => state.totalSquared ? 100 * value / state.totalSquared : 0;
  function reset() {
    setPreset('fast');
    setRank(2);
    setOversampling(1);
    setIterations(0);
    setSeed(7);
  }
  return <section className="rla-lab" aria-label="Spectral sketch budget investigation">
    <p className="rla-eyebrow">RANK LIMIT · SKETCH QUALITY · DATA PASSES</p>
    <h3>Which error can another probe actually fix?</h3>
    <p>Start with rank 2, then add oversampling without changing rank. Switch to equal singular values and predict whether extra passes can beat the best rank-2 error. All matrices are 6×6 with fixed rotated singular directions.</p>
    <div className="rla-controls"><label>Spectrum<select value={preset} onChange={event => setPreset(event.target.value)}>{Object.entries(sketchSpectra).map(([key, item]) => <option key={key} value={key}>{item.title}</option>)}</select></label>
      <label>Target rank k<select value={rank} onChange={event => {
          const nextRank = Number(event.target.value);
          setRank(nextRank);
          setOversampling(previous => Math.min(previous, 6 - nextRank));
        }}>{[1, 2, 3, 4].map(value => <option key={value} value={value}>{value}</option>)}</select></label>
      <label>Extra probes p<select value={oversampling} onChange={event => setOversampling(Number(event.target.value))}>{Array.from({
            length: 7 - rank
          }, (_, value) => <option key={value} value={value}>{value}</option>)}</select></label>
      <label>Subspace iterations q<select value={iterations} onChange={event => setIterations(Number(event.target.value))}>{[0, 1, 2, 3].map(value => <option key={value} value={value}>{value}</option>)}</select></label>
      <label>Probe seed<select value={seed} onChange={event => setSeed(Number(event.target.value))}>{[1, 7, 42, 99].map(value => <option key={value} value={value}>{value}</option>)}</select></label>
      <button onClick={reset}>Reset spectrum</button></div>
    <figure className="rla-spectrum"><figcaption>Input singular values: the first k set the exact-SVD benchmark</figcaption>
      {state.spectrum.map((value, index) => <div className="rla-spectrum-row" key={index}><span>σ{index + 1}</span><div className="rla-bar-track"><div className={index < rank ? 'rla-bar-kept' : 'rla-bar-tail'} style={{
            width: `${100 * value / 12}%`
          }} /></div><span>{number(value)}</span></div>)}
      <p>Linear bar scale: 0 to 12. Gold identifies the first k values; blue is the unavoidable exact rank-k tail, not the actual randomized residual.</p>
    </figure>
    <figure className="rla-error-budget"><figcaption>Squared approximation error as a share of ‖A‖²F</figcaption>
      <div className="rla-error-track"><div className="rla-range-error" style={{
          width: `${percent(state.rangeErrorSquared)}%`
        }} /><div className="rla-truncation-error" style={{
          width: `${percent(state.truncationErrorSquared)}%`
        }} />
        <span className="rla-floor-mark" style={{
          left: `${percent(state.floorSquared)}%`
        }} /></div>
      <p>Gold: outside Q. Blue: discarded inside Q. The vertical mark is the exact rank-k floor. The full width is 100% of input squared norm.</p>
    </figure>
    <dl className="rla-readings"><dt>Probe width ℓ = k + p</dt><dd>{state.width}</dd><dt>Observed independent directions</dt><dd>{state.observedRank}</dd>
      <dt>Outside-Q error squared</dt><dd>{number(state.rangeErrorSquared)}</dd><dt>Inside-Q truncation error squared</dt><dd>{number(state.truncationErrorSquared)}</dd>
      <dt>Total error squared</dt><dd>{number(state.errorSquared)}</dd><dt>Best rank-k error squared</dt><dd>{number(state.floorSquared)}</dd>
      <dt>Relative Frobenius error</dt><dd>{state.relativeError === null ? 'Undefined: the input norm is zero' : number(state.relativeError)}</dd>
      <dt>Full passes in the standard algorithm</dt><dd>{state.passes}</dd></dl>
    <p className="rla-feedback" aria-live="polite">{preset === 'zero' ? 'Both absolute errors are zero; dividing by the zero input norm would not define a relative error. A program can short-circuit this case.' : preset === 'flat' ? 'There is no dominant singular direction to amplify. More accurate range finding cannot remove the error imposed by the chosen final rank.' : 'Increasing width can improve the captured subspace; final truncation still limits the answer to rank at most k. Changing the seed changes which probes are drawn.'}</p>
    <details><summary>Inspect actual matrices and numerical boundaries</summary>
      <MatrixTable matrix={state.matrix} label="Input A, rounded to four decimals" />
      <MatrixTable matrix={state.omega} label="Probe matrix Omega" />
      <MatrixTable matrix={state.approximation} label="Final rank-at-most-k approximation" />
      <p>The browser uses seeded pseudo-Gaussian probes, twice-reorthogonalized Gram–Schmidt, and a bounded symmetric eigensolver for the small Gram matrix. It drops residual columns below 10⁻¹¹ times the largest incoming column norm. The displayed modest spectra are checked against NumPy SVD; this small Gram-based routine is not a recommended general SVD implementation. Seed values are reproducible here, not shared random streams with NumPy.</p>
    </details>
  </section>;
}
function RegressionPlot({
  state
}) {
  const id = useId();
  const fits = [state.fullFit, state.sketchFit].filter(Boolean);
  const endpointValues = fits.flatMap(fit => [fit.intercept, fit.intercept + 20 * fit.slope]);
  const lowest = Math.floor(Math.min(0, ...endpointValues, ...state.observations.map(point => point.y)) - 1);
  const highest = Math.ceil(Math.max(15, ...endpointValues, ...state.observations.map(point => point.y)) + 1);
  const x = value => 46 + value / 20 * 290;
  const y = value => 230 - (value - lowest) / (highest - lowest) * 190;
  return <svg viewBox="0 0 370 276" className="rla-regression" role="img" aria-labelledby={`${id}-title ${id}-description`}>
    <title id={`${id}-title`}>Fit with selected observations and evaluate against all observations</title>
    <desc id={`${id}-description`}>Filled gold points are selected; hollow points are omitted. Dashed blue is the all-data optimum; green is the selected-row fit. Vertical scale adapts to include both lines. Values appear in the observation table and readings.</desc>
    <path className="rla-axis" d="M 46 34 V 230 H 340" />
    {[0, 5, 10, 15, 20].map(value => <text key={value} x={x(value)} y="249" textAnchor="middle">{value}</text>)}
    {[lowest, (lowest + highest) / 2, highest].map(value => <g key={value}><line className="rla-grid" x1="46" x2="340" y1={y(value)} y2={y(value)} /><text x="38" y={y(value) + 4} textAnchor="end">{number(value)}</text></g>)}
    <text x="183" y="272">input x</text><text x="8" y="20">response y</text>
    {fits.map((fit, index) => <line key={index} x1={x(0)} x2={x(20)} y1={y(fit.intercept)} y2={y(fit.intercept + 20 * fit.slope)} stroke={index ? colors[2] : colors[1]} strokeWidth="2" strokeDasharray={index ? undefined : '5 4'} />)}
    {state.observations.map((point, index) => <g key={index}><circle cx={x(point.x)} cy={y(point.y)} r="4.5" fill={state.selectedRows.includes(index) ? colors[0] : '#11151b'} stroke={colors[0]} strokeWidth="1.7" /><title>Row {index}: ({point.x}, {number(point.y)}), {state.selectedRows.includes(index) ? 'selected' : 'omitted'}</title></g>)}
  </svg>;
}
export function RowSketchLab() {
  const [selectedRows, setSelectedRows] = useState([0, 1, 2, 3]);
  const state = rowSketchState(selectedRows);
  return <section className="rla-lab" aria-label="Observation sketch investigation">
    <p className="rla-eyebrow">A GOOD FIT TO A SKETCH CAN MISS THE ORIGINAL TASK</p>
    <h3>Which observations did the small problem forget?</h3>
    <p>Fit an intercept and slope using selected rows. Try just the two endpoints: a perfect fit to those two points still leaves error on the omitted observations. Row 7, far from the other inputs, has unusually large geometric influence.</p>
    <div className="rla-controls"><button onClick={() => setSelectedRows([0, 1, 2, 3])}>First four / reset</button><button onClick={() => setSelectedRows([0, 7])}>Two endpoints</button><button onClick={() => setSelectedRows([0, 1, 2, 3, 4, 5, 6, 7])}>All rows</button><button onClick={() => setSelectedRows([])}>Clear selection</button></div>
    <div className="rla-plot-pair"><RegressionPlot state={state} /><fieldset className="rla-row-selection"><legend>Rows included in the small problem</legend>
      {state.observations.map((point, index) => <label key={index}><input type="checkbox" checked={selectedRows.includes(index)} onChange={() => setSelectedRows(previous => previous.includes(index) ? previous.filter(value => value !== index) : [...previous, index])} />
        <span>Row {index}: x = {point.x}, y = {number(point.y)}<small>leverage {number(state.leverage[index])}</small></span></label>)}
    </fieldset></div>
    <dl className="rla-readings"><dt>Selected-row fit</dt><dd>{state.sketchFit ? `y = ${number(state.sketchFit.intercept)} + ${number(state.sketchFit.slope)}x` : 'No unique intercept and slope'}</dd>
      <dt>Selected-row squared error</dt><dd>{state.sketchResidualSquared === null ? 'Not reported without a unique fit' : number(state.sketchResidualSquared)}</dd>
      <dt>That fit on every original row</dt><dd>{state.originalResidualSquared === null ? 'Not reported without a unique fit' : number(state.originalResidualSquared)}</dd>
      <dt>Best possible all-row squared error</dt><dd>{number(state.fullResidualSquared)}</dd></dl>
    <p className="rla-feedback" aria-live="polite">{state.sketchFit ? 'Compare original-data error, not just how well the reduced problem was solved. Lower sketch error alone does not imply a better answer.' : 'Fewer than two distinct input locations leave the two coefficients underdetermined. More than one line fits; the investigation does not invent a unique answer.'}</p>
    <p className="rla-caption">Manual row selection illustrates lost information; it is not a randomized guarantee. The fit uses unweighted selected rows. Under uniform sampling without replacement, multiplying every selected row and response by the same √(8/s) leaves the minimizer unchanged but rescales the displayed sketch error. Nonuniform sampling needs different weights, explained in the lesson.</p>
  </section>;
}
function TraceHistory({
  state
}) {
  const id = useId();
  const halfRange = Math.max(2, 2 * Math.abs(state.matrix[0][1]) + 1);
  const minimum = Math.min(0, state.exactTrace - halfRange);
  const maximum = state.exactTrace + halfRange;
  const x = count => 46 + (count - 1) * 285 / 31;
  const y = value => 200 - 152 * (value - minimum) / (maximum - minimum);
  return <svg viewBox="0 0 370 252" className="rla-history" role="img" aria-labelledby={`${id}-title ${id}-description`}>
    <title id={`${id}-title`}>Running mean of trace probes</title><desc id={`${id}-description`}>Horizontal axis is number of probes, one through 32. Vertical axis is the trace estimate. Blue connects actual running means; a gold dashed line is the known reference trace of this tiny matrix. A numerical table follows.</desc>
    <path className="rla-axis" d="M 46 36 V 200 H 338" />
    {[minimum, (minimum + maximum) / 2, maximum].map(value => <g key={value}><text x="37" y={y(value) + 4} textAnchor="end">{number(value)}</text><line className="rla-grid" x1="46" x2="338" y1={y(value)} y2={y(value)} /></g>)}
    {[1, 8, 16, 24, 32].map(value => <text key={value} x={x(value)} y="222" textAnchor="middle">{value}</text>)}
    <text x="126" y="247">number of probes</text><text x="8" y="20">trace estimate</text>
    <line x1="46" x2="338" y1={y(state.exactTrace)} y2={y(state.exactTrace)} stroke={colors[0]} strokeWidth="2" strokeDasharray="5 4" />
    <polyline points={state.samples.map(sample => `${x(sample.count)},${y(sample.mean)}`).join(' ')} fill="none" stroke={colors[1]} strokeWidth="2" />
    {state.samples.map(sample => <circle key={sample.count} cx={x(sample.count)} cy={y(sample.mean)} r="2.5" fill={colors[1]} />)}
  </svg>;
}
export function TraceEstimatorLab() {
  const [preset, setPreset] = useState('coupled');
  const [seed, setSeed] = useState(7);
  const [count, setCount] = useState(0);
  const state = traceProbeState(preset, seed, count);
  const sample = state.samples.at(-1);
  return <section className="rla-lab" aria-label="Random sign trace investigation">
    <p className="rla-eyebrow">RANDOM SIGNS · MATRIX PRODUCT · RUNNING AVERAGE</p>
    <h3>Does another sample always make the estimate better?</h3>
    <p>Each probe chooses two signs, applies A once, and takes zᵀAz. Advance several steps before switching to the diagonal matrix. Explain why its random signs cannot change the answer.</p>
    <div className="rla-controls"><label>Trace matrix<select value={preset} onChange={event => {
          setPreset(event.target.value);
          setCount(0);
        }}>{Object.entries(traceMatrices).map(([key, item]) => <option key={key} value={key}>{item.title}</option>)}</select></label>
      <label>Sign seed<select value={seed} onChange={event => {
          setSeed(Number(event.target.value));
          setCount(0);
        }}>{[1, 7, 42, 99].map(value => <option key={value} value={value}>{value}</option>)}</select></label>
      <button disabled={count === 32} onClick={() => setCount(previous => previous + 1)}>Draw one probe</button><button disabled={!count} onClick={() => setCount(previous => previous - 1)}>Previous probe</button>
      <button onClick={() => {
        setPreset('coupled');
        setSeed(7);
        setCount(0);
      }}>Reset estimator</button></div>
    <div className="rla-probe-flow"><div><span>sign vector z</span><strong>{sample ? pair(sample.input) : 'Draw a probe'}</strong></div><span className="rla-flow-arrow" aria-hidden="true">→</span>
      <MatrixTable matrix={state.matrix} label="A" /><span className="rla-flow-arrow" aria-hidden="true">→</span><div><span>product Az</span><strong>{sample ? pair(sample.output) : '—'}</strong></div>
      <span className="rla-flow-arrow" aria-hidden="true">→</span><div><span>inner product zᵀAz</span><strong>{sample ? sample.value : '—'}</strong></div></div>
    <TraceHistory state={state} />
    <p className="rla-feedback" aria-live="polite">{count ? `${count} probe${count === 1 ? '' : 's'}: mean ${number(state.mean)}, exact reference trace ${state.exactTrace}, absolute error ${number(state.absoluteError)}.` : `No probes yet. The known reference trace is ${state.exactTrace}; the estimate is not zero—it has not been calculated.`}</p>
    <details><summary>Read every probe and the uncertainty model</summary>
      <div className="rla-matrix-wrap" tabIndex={0} role="region" aria-label="Trace probe data"><table className="rla-matrix"><thead><tr><th>Probe</th><th>z</th><th>zᵀAz</th><th>Mean</th></tr></thead><tbody>{state.samples.map(item => <tr key={item.count}><th>{item.count}</th><td>{pair(item.input)}</td><td>{item.value}</td><td>{number(item.mean)}</td></tr>)}</tbody></table></div>
      <p>For independent ideal uniform signs, the single-probe variance here is {state.variancePerProbe}. The variance of an average of s independent probes is that value divided by s. This statement concerns repeated random experiments; it does not force each successive error in one run to decrease. A seeded pseudo-random sequence makes this demonstration repeatable.</p>
    </details>
    <p className="rla-caption">For an explicitly stored 2×2 matrix, reading the diagonal is easier. The useful setting is a large operator for which products Az are available cheaply but diagonal entries are not. The zero-trace example needs absolute error: a relative error would divide by zero.</p>
  </section>;
}
export function RandomizedFactorFlow() {
  return <figure className="rla-inline-figure"><figcaption>Keep the big dimension outside the expensive factorization</figcaption>
    <ol className="rla-factor-flow"><li><strong>A Ω → Y</strong><span>(m×n)(n×ℓ) → m×ℓ</span><p>Combine columns into ℓ probes.</p></li>
      <li><strong>Y → Q</strong><span>m×r, r ≤ ℓ</span><p>Describe the observed space with orthonormal columns.</p></li>
      <li><strong>Qᵀ A → B</strong><span>(r×m)(m×n) → r×n</span><p>Express every original column in that basis.</p></li>
      <li><strong>B → Ũₖ Σₖ Vₖᵀ</strong><span>Small SVD, retain at most k</span><p>Lift its left directions with Q Ũₖ.</p></li></ol>
    <p>Probe width ℓ is not the final rank k. Numerical rank deficiency can make r smaller; the resulting factors still have rank at most k.</p>
  </figure>;
}
export function SketchDirectionFigure() {
  return <figure className="rla-inline-figure"><figcaption>Choose which dimension to compress by the task</figcaption><div className="rla-sketch-directions">
    <div><span className="rla-eyebrow">RIGHT SKETCH</span><strong>A Ω</strong><p>Combine <b>columns</b>. Discover a subspace in the m-dimensional output space, then reconstruct A.</p></div>
    <div><span className="rla-eyebrow">LEFT SKETCH</span><strong>S [A | b]</strong><p>Combine <b>observations</b>. Solve a smaller residual problem with the same unknown coefficients, then evaluate on the original rows.</p></div>
  </div><p>A sketch is designed around what it must preserve. Merely making an array smaller does not establish that the learning or numerical task survives.</p></figure>;
}
