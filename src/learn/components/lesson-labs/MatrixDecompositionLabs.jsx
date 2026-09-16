import { useId, useState } from 'react';
import { covarianceGeometry, eliminationPresets, eliminationTrace, qrGeometry, svdGeometry } from '../../data/matrix-decomposition-models.js';
import './matrix-decomposition-labs.css';
function valueText(value) {
  if (value === null) return 'pending';
  return (Math.abs(value) < 0.0000005 ? 0 : value).toFixed(3).replace(/\.?0+$/, '');
}
function vectorText(vector) {
  return `[${vector.map(valueText).join(', ')}]`;
}
export function DecompositionMatrix({
  values,
  label,
  augmented = false
}) {
  return <div className="decomposition-matrix">
    <table>
      <caption>{label}</caption>
      <tbody>{values.map((row, rowIndex) => <tr key={rowIndex}>{row.map((value, columnIndex) => <td key={columnIndex} className={augmented && columnIndex === row.length - 1 ? 'decomposition-target' : undefined}>
          {valueText(value)}
        </td>)}</tr>)}</tbody>
    </table>
  </div>;
}
function CoordinatePlane({
  title,
  description,
  vectors = [],
  curves = [],
  range = 3.5
}) {
  const titleId = useId();
  const descriptionId = useId();
  const unit = 130 / range;
  const position = point => [160 + point[0] * unit, 160 - point[1] * unit];
  return <figure className="decomposition-plane">
    <figcaption>{title}</figcaption>
    <svg viewBox="0 0 320 338" role="img" aria-labelledby={`${titleId} ${descriptionId}`}>
      <title id={titleId}>{title}</title><desc id={descriptionId}>{description}</desc>
      {[-3, -2, -1, 0, 1, 2, 3].filter(value => Math.abs(value) <= range).map(value => <g key={value}>
        <line className="decomposition-grid" x1={160 + value * unit} y1="25" x2={160 + value * unit} y2="295" />
        <line className="decomposition-grid" x1="25" y1={160 - value * unit} x2="295" y2={160 - value * unit} />
        {value !== 0 && <text className="decomposition-tick" x={160 + value * unit} y="320" textAnchor="middle">{value}</text>}
      </g>)}
      <line className="decomposition-axis" x1="20" y1="160" x2="298" y2="160" />
      <line className="decomposition-axis" x1="160" y1="20" x2="160" y2="298" />
      <text x="304" y="150" className="decomposition-axis-label">x</text>
      <text x="172" y="17" className="decomposition-axis-label">y</text>
      {curves.map((curve, index) => <polyline key={index} points={curve.points.map(point => position(point).join(',')).join(' ')} fill="none" stroke={curve.color} strokeWidth="2" strokeDasharray={curve.dashed ? '6 4' : undefined} />)}
      {vectors.map((vector, index) => {
        const start = position(vector.start ?? [0, 0]);
        const end = position(vector.end);
        const dx = end[0] - start[0];
        const dy = end[1] - start[1];
        const length = Math.hypot(dx, dy);
        const arrow = length > 0.01 ? [end, [end[0] - 9 * dx / length + 4 * dy / length, end[1] - 9 * dy / length - 4 * dx / length], [end[0] - 9 * dx / length - 4 * dy / length, end[1] - 9 * dy / length + 4 * dx / length]] : null;
        return <g key={index} stroke={vector.color} fill={vector.color}>
          <line x1={start[0]} y1={start[1]} x2={end[0]} y2={end[1]} strokeWidth="3" strokeDasharray={vector.dashed ? '5 3' : undefined} />
          {arrow ? <polygon points={arrow.map(point => point.join(',')).join(' ')} stroke="none" /> : <circle cx={end[0]} cy={end[1]} r="4" />}
        </g>;
      })}
    </svg>
    <ul className="decomposition-legend">{vectors.map((vector, index) => <li key={index}><span style={{
          borderColor: vector.color
        }} aria-hidden="true" />{vector.label}: {vectorText(vector.end)}{vector.start ? ` from ${vectorText(vector.start)}` : ''}</li>)}</ul>
  </figure>;
}
export function TriangularDependencyFigure() {
  return <figure className="decomposition-inline">
    <figcaption>Start where only one unknown remains</figcaption>
    <div className="decomposition-dependency">
      <div><strong>2x₁ + x₂ = 5</strong><span>Wait for x₂, then solve x₁.</span></div>
      <span className="decomposition-up" aria-label="Substitute the lower equation's result into the upper equation">↑</span>
      <div><strong>x₂ = 1</strong><span>Known first → x₁ = (5 − 1) / 2 = 2.</span></div>
    </div>
    <p>The zero coefficient below the first diagonal entry removes x₁ from the bottom equation. This dependency is why back substitution runs upward.</p>
  </figure>;
}
export function EliminationLab() {
  const [presetName, setPresetName] = useState('coupled');
  const [stepIndex, setStepIndex] = useState(0);
  const trace = eliminationTrace(presetName);
  const state = trace.steps[stepIndex];
  const changePreset = event => {
    setPresetName(event.target.value);
    setStepIndex(0);
  };
  return <section className="decomposition-lab" aria-label="Pivoted elimination investigation">
    <span className="decomposition-eyebrow">ROW OPERATIONS · FOLLOW THE EQUATIONS</span>
    <h3>Eliminate one unknown without changing the answer</h3>
    <p>Predict what must happen to the right-hand side when a row changes. Then advance one operation. Preset changes apply immediately and reset the trace.</p>
    <label className="decomposition-select">System <select value={presetName} onChange={changePreset}>
      {Object.entries(eliminationPresets).map(([key, preset]) => <option key={key} value={key}>{preset.title}</option>)}
    </select></label>
    <div className="decomposition-matrix-row">
      <DecompositionMatrix values={trace.matrix.map((row, index) => [...row, trace.target[index]])} label="Original [A | b]" augmented />
      <span aria-hidden="true">→</span><DecompositionMatrix values={state.augmented} label="Current equations" augmented />
    </div>
    <div className="decomposition-controls">
      <button onClick={() => setStepIndex(index => index - 1)} disabled={stepIndex === 0}>Previous operation</button>
      <button onClick={() => setStepIndex(index => index + 1)} disabled={stepIndex === trace.steps.length - 1}>Next operation</button>
      <button onClick={() => setStepIndex(0)}>Reset trace</button>
      <span>State {stepIndex + 1} / {trace.steps.length}</span>
    </div>
    <p className="decomposition-feedback" aria-live="polite">{state.message}</p>
    {state.solution && <p className="decomposition-readout">x = {vectorText(state.solution)}</p>}
    <details><summary>Inspect the stored factors at this step</summary>
      <div className="decomposition-matrix-row"><DecompositionMatrix values={state.permutation} label="P: row ordering" /><DecompositionMatrix values={state.lower} label="L: stored multipliers" /></div>
      <p>After elimination, the current left side is U and PA = LU. The current target is y = L⁻¹Pb. Singular presets still have factors; they do not have a unique solution.</p>
    </details>
    <p className="decomposition-note">Exact small preset systems evaluated in JavaScript arithmetic. This bounded view does not simulate numerical roundoff or implement a general-purpose solver.</p>
  </section>;
}
export function QrProjectionLab() {
  const [columnX, setColumnX] = useState(1);
  const [columnY, setColumnY] = useState(0);
  const model = qrGeometry([columnX, columnY]);
  return <section className="decomposition-lab" aria-label="QR column projection investigation">
    <span className="decomposition-eyebrow">QR · REMOVE THE SHARED DIRECTION</span>
    <h3>What does the second column add?</h3>
    <p>The first column a₁ = [1, 1] fixes a line. Move a₂ and see which part lies along that line and which part supplies a new direction. Set both coordinates equal to test dependence.</p>
    <div className="decomposition-controls">
      <label>a₂ first coordinate: {columnX}<input aria-label="Second column first coordinate" type="range" min="-2" max="2" step="0.5" value={columnX} onChange={event => setColumnX(Number(event.target.value))} /></label>
      <label>a₂ second coordinate: {columnY}<input aria-label="Second column second coordinate" type="range" min="-2" max="2" step="0.5" value={columnY} onChange={event => setColumnY(Number(event.target.value))} /></label>
      <button onClick={() => {
        setColumnX(1);
        setColumnY(0);
      }}>Reset columns</button>
    </div>
    <CoordinatePlane title="a₂ = parallel part + perpendicular remainder" description={`Second column ${vectorText(model.secondColumn)} splits into projection ${vectorText(model.projection)} and perpendicular remainder ${vectorText(model.perpendicular)}.`} vectors={[{
      end: model.firstColumn,
      label: 'a₁',
      color: '#c8c3b6',
      dashed: true
    }, {
      end: model.secondColumn,
      label: 'a₂',
      color: '#f0c66f'
    }, {
      end: model.projection,
      label: 'Parallel part',
      color: '#a6c4e8',
      dashed: true
    }, {
      start: model.projection,
      end: model.secondColumn,
      label: 'Remainder tip',
      color: '#a9cf9e'
    }]} />
    <p className="decomposition-readout">q₁·a₂ = {valueText(model.overlap)}; remainder = {vectorText(model.perpendicular)}; its length = {valueText(model.remainingNorm)}.</p>
    <p className="decomposition-feedback" aria-live="polite">{model.dependent ? 'No new direction remains. This construction cannot normalize a zero remainder. The columns are dependent; a full-rank triangular solve is not valid.' : `Normalize the remainder to get q₂ = ${vectorText(model.secondUnit)}. Both q columns have length 1 and their dot product is 0 (up to rounding).`}</p>
    {model.orthogonal && <div className="decomposition-matrix-row"><DecompositionMatrix values={model.orthogonal} label="Q" /><DecompositionMatrix values={model.triangular} label="R: Q coordinates of A" /></div>}
    <p className="decomposition-note">Analytic Gram-Schmidt geometry with equal axis scales. Values are rounded for display; computations use full precision. Production QR commonly uses Householder reflections.</p>
  </section>;
}
export function CovarianceFactorLab() {
  const [correlation, setCorrelation] = useState(0.5);
  const model = covarianceGeometry(correlation);
  return <section className="decomposition-lab" aria-label="Covariance factor investigation">
    <span className="decomposition-eyebrow">CHOLESKY · SHARE AN INDEPENDENT COMPONENT</span>
    <h3>Correlation changes a shape, not just one table entry</h3>
    <p>Let z₁ and z₂ have zero mean, unit variance and zero covariance. Set x₁ = 2z₁ and x₂ = ρz₁ + √(1 − ρ²)z₂. Predict the shape at ρ = 0, then at an endpoint.</p>
    <div className="decomposition-controls"><label>Correlation ρ: {valueText(correlation)}<input type="range" aria-label="Covariance correlation" min="-1" max="1" step="0.05" value={correlation} onChange={event => setCorrelation(Number(event.target.value))} /></label>
      <button onClick={() => setCorrelation(0.5)}>Reset correlation</button></div>
    <CoordinatePlane title="A unit circle mapped by L" description={`Dashed unit circle and its transformed outline for correlation ${correlation}. Variances stay 4 and 1; covariance is ${2 * correlation}.`} curves={[{
      points: model.circle,
      color: '#c8c3b6',
      dashed: true
    }, {
      points: model.transformed,
      color: '#a9cf9e'
    }]} vectors={[{
      end: [2, correlation],
      label: 'First column of L',
      color: '#f0c66f'
    }, {
      end: [0, model.lower[1][1]],
      label: 'Second column of L',
      color: '#a6c4e8'
    }]} />
    <div className="decomposition-matrix-row"><DecompositionMatrix values={model.lower} label="L" /><DecompositionMatrix values={model.covariance} label="C = LLᵀ" /></div>
    <p className="decomposition-feedback" aria-live="polite">{model.positiveDefinite ? `Two independent directions remain: det(C) = ${valueText(model.determinant)} > 0. This C is positive definite.` : 'One direction collapses: det(C) = 0. C is positive semidefinite, but ordinary Cholesky requires positive definiteness. The displayed algebraic factor still satisfies LLᵀ = C.'}</p>
    <p className="decomposition-note">Dashed: input unit circle. Solid: its exact linear image, sampled at 65 angular positions for drawing. These are geometric outlines, not observed samples or probability contours. Independence and Gaussianity are stronger assumptions than the covariance identity requires.</p>
  </section>;
}
export function SingularDirectionsLab() {
  const [inputAngle, setInputAngle] = useState(30);
  const [outputAngle, setOutputAngle] = useState(-20);
  const [smallerScale, setSmallerScale] = useState(1);
  const [vectorAngle, setVectorAngle] = useState(45);
  const [retained, setRetained] = useState(2);
  const model = svdGeometry({
    inputAngle,
    outputAngle,
    smallerScale,
    vectorAngle,
    retained
  });
  function reset() {
    setInputAngle(30);
    setOutputAngle(-20);
    setSmallerScale(1);
    setVectorAngle(45);
    setRetained(2);
  }
  return <section className="decomposition-lab" aria-label="Singular directions investigation">
    <span className="decomposition-eyebrow">SVD · CHANGE BASIS, STRETCH, CHANGE BASIS</span>
    <h3>Follow the same input through all three factors</h3>
    <p>This investigation constructs A from known factors. Its largest singular value stays 3. Change the smaller one, then keep only one singular direction and inspect what is lost.</p>
    <div className="decomposition-controls">
      <label>Input basis angle: {inputAngle}°<input aria-label="Input basis angle" type="range" min="-90" max="90" step="15" value={inputAngle} onChange={event => setInputAngle(Number(event.target.value))} /></label>
      <label>Output basis angle: {outputAngle}°<input aria-label="Output basis angle" type="range" min="-90" max="90" step="10" value={outputAngle} onChange={event => setOutputAngle(Number(event.target.value))} /></label>
      <label>Smaller singular value: {smallerScale}<input aria-label="Smaller singular value" type="range" min="0" max="3" step="0.25" value={smallerScale} onChange={event => setSmallerScale(Number(event.target.value))} /></label>
      <label>Input vector angle: {vectorAngle}°<input aria-label="Input vector angle" type="range" min="0" max="360" step="15" value={vectorAngle} onChange={event => setVectorAngle(Number(event.target.value))} /></label>
      <label>Singular directions kept<select aria-label="Singular directions kept" value={retained} onChange={event => setRetained(Number(event.target.value))}><option value="0">0: zero matrix</option><option value="1">1: largest direction</option><option value="2">2: both directions</option></select></label>
      <button onClick={reset}>Reset SVD</button>
    </div>
    <ol className="decomposition-factor-chain">
      <li><strong>x</strong><span>{vectorText(model.input)}</span></li>
      <li><strong>Vᵀx</strong><span>{vectorText(model.rotated)}</span></li>
      <li><strong>ΣVᵀx</strong><span>{vectorText(model.scaled)}</span></li>
      <li><strong>UΣVᵀx = Ax</strong><span>{vectorText(model.output)}</span></li>
    </ol>
    <div className="decomposition-planes">
      <CoordinatePlane title="Input directions" description={`The selected unit vector is ${vectorText(model.input)}. Right singular directions are the columns of V.`} curves={[{
        points: model.circle,
        color: '#c8c3b6',
        dashed: true
      }]} vectors={[{
        end: model.input,
        label: 'Input x',
        color: '#f0c66f'
      }, {
        end: model.right.map(row => row[0]),
        label: 'v₁',
        color: '#a6c4e8',
        dashed: true
      }]} />
      <CoordinatePlane title="Output and truncated approximation" description={`A maps the circle to an ellipse or line. Exact output ${vectorText(model.output)}; truncated output ${vectorText(model.approximationOutput)}.`} curves={[{
        points: model.transformed,
        color: '#a9cf9e'
      }, {
        points: model.truncated,
        color: '#a6c4e8',
        dashed: true
      }]} vectors={[{
        end: model.output,
        label: 'Ax',
        color: '#f0c66f'
      }, {
        end: model.approximationOutput,
        label: 'Aₖx',
        color: '#a6c4e8',
        dashed: true
      }]} />
    </div>
    <div className="decomposition-matrix-row"><DecompositionMatrix values={model.matrix} label="Constructed A" /><DecompositionMatrix values={model.approximation} label="Aₖ" /></div>
    <p className="decomposition-readout" aria-live="polite">Singular values: {vectorText(model.singularValues)}. Original rank: {model.rank}. Frobenius error: {valueText(model.frobeniusError)}. Largest possible output error over unit inputs: {valueText(model.spectralError)}.</p>
    <p>Try aligning x with v₁: rank-one truncation can reproduce this input exactly while losing information for another input. A small error on one vector is not the maximum error over all directions.</p>
    <p className="decomposition-note">An analytic real 2×2 factor family, drawn at equal axis scales; no general SVD computation or empirical compression benchmark runs in your browser. Solid green is A's unit-circle image; dashed blue is Aₖ's. Rotations suffice for this family; general orthogonal SVD factors can also reflect.</p>
  </section>;
}
