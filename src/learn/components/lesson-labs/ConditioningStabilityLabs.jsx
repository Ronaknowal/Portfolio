import { useEffect, useRef, useState } from 'react';
import { backwardWitnessState, cancellationState, formatConditioning as f, measurementSensitivityState, propagationState, roundingCellState, summationState, unstableRefinementState } from '../../data/conditioning-stability-models.js';
import './conditioning-stability-labs.css';
function Slider({
  label,
  value,
  minimum,
  maximum,
  onChange
}) {
  return <label>{label}: <strong>{value}</strong><input aria-label={label} type="range" min={minimum} max={maximum} step="1" value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Table({
  caption,
  headers,
  rows
}) {
  return <div className="conditioning-table" role="region" tabIndex={0} aria-label={caption}><table><caption>{caption}</caption><thead><tr>{headers.map(header => <th scope="col" key={header}>{header}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>)}</tbody></table></div>;
}
function Drawing({
  label,
  compact = false,
  xLabel,
  yLabel,
  children
}) {
  return <>{yLabel && <p className="conditioning-axis-label">Vertical axis: {yLabel}</p>}<div className={`conditioning-drawing${compact ? " compact" : ""}`} role="region" tabIndex={0} aria-label={`${label}; scroll horizontally if needed`}><svg viewBox="0 0 440 250" role="img" aria-label={label}>{children}</svg></div>{xLabel && <p className="conditioning-axis-label">Horizontal axis: {xLabel}</p>}</>;
}
export function NumericalEvidenceFigure() {
  return <figure className="conditioning-figure">
    <ol className="conditioning-pipeline">
      <li><strong>Intended data</strong><span>10000000000000001</span><small>input representation ↓</small></li>
      <li><strong>Stored input</strong><span>10000000000000000</span><small>Use this same input in both branches below.</small></li>
    </ol>
    <div className="conditioning-reference-branches">
      <div><strong>Exact reference branch</strong><span>Stored input → exact mathematical answer</span></div>
      <div><strong>Computed branch</strong><span>Stored input → algorithm → approximation</span></div>
    </div>
    <figcaption>Forward error compares the two answers; the program does not first need to know the exact answer. Changing precision after the input conversion does not restore the missing one. Model error is a separate comparison between the mathematical question and the physical system.</figcaption>
  </figure>;
}

export function RoundingCellsLab() {
  const [bin, setBin] = useState(0);
  const [half, setHalf] = useState(1);
  const result = roundingCellState(bin, half);
  const start = 2 ** bin;
  const point = value => 42 + 352 * (value - start) / start;
  return <section className="conditioning-lab" aria-label="Rounding cells investigation"><h3>Which stored neighbor wins?</h3><p>Inspect the rounded value while moving through the halfway points. This is an exact four-significant-bit toy system with nearest-even rounding.</p><div className="conditioning-controls"><label>Exponent bin<select aria-label="Exponent bin" value={bin} onChange={event => setBin(Number(event.target.value))}><option value="0">1 to 2</option><option value="1">2 to 4</option></select></label><Slider label="Half-step position" value={half} minimum={0} maximum={16} onChange={setHalf} /></div><Drawing label="Exact rounding grid and halfway input"><line x1="42" x2="394" y1="105" y2="105" className="axis" />{result.grid.map((value, index) => <g key={value}><line x1={point(value)} x2={point(value)} y1="96" y2="114" className="tick" /><text x={point(value)} y={index % 2 ? 158 : 138} textAnchor="middle">{value}</text></g>)}<line x1={point(result.input)} x2={point(result.input)} y1="55" y2="94" className="input" /><circle cx={point(result.input)} cy="55" r="6" className="input-fill" /><text x="42" y="25">Input {result.input}</text><circle cx={point(result.rounded)} cy="105" r="8" className="chosen" /><text x="42" y="203">Stored {result.rounded}</text><text x="42" y="228">Gap between neighbors = {result.spacing}</text></Drawing><p className="conditioning-readout" aria-live="polite">Input = {result.input}; stored = {result.rounded}. {result.halfway ? `Exactly halfway: the retained significand is even.` : 'Already on the grid; no rounding change.'}</p><button onClick={() => {
      setBin(0);
      setHalf(1);
    }}>Reset rounding</button></section>;
}
export function CancellationLab() {
  const [exponent, setExponent] = useState(54);
  const [sign, setSign] = useState(1);
  const state = cancellationState(exponent, sign);
  const samples = Array.from({
    length: 60
  }, (_, i) => ({
    exponent: i + 1,
    state: cancellationState(i + 1, sign)
  }));
  const chartValues = samples.flatMap(row => [row.state.directError.relativeUpper, row.state.repairedError.relativeUpper]).filter(value => value > 0 && Number.isFinite(value));
  const bottom = Math.min(-20, ...chartValues.map(value => Math.floor(Math.log10(value) / 5) * 5));
  const chartX = value => 106 + 288 * (value - 1) / 59;
  const chartY = value => 25 + 158 * (1 - Math.log10(value)) / (1 - bottom);
  return <section className="conditioning-lab" aria-label="Cancellation investigation"><h3>Follow one input through two formulas</h3><p>The two paths have the same exact answer. Their rounded intermediate values differ in usefulness. Error readouts use an exact rational enclosure of the square root; displayed bounds are rounded.</p><div className="conditioning-controls"><label>Input sign<select aria-label="Input sign" value={sign} onChange={event => setSign(Number(event.target.value))}><option value="1">Positive</option><option value="-1">Negative</option><option value="0">Exactly zero</option></select></label><Slider label="Dyadic exponent k" value={exponent} minimum={0} maximum={60} onChange={setExponent} /></div><p><strong>x = {f(state.input, 10)}</strong> ({sign === 0 ? 'zero' : `${sign < 0 ? '−' : ''}2^−${exponent}`})</p><div className="conditioning-arithmetic"><div><span>1 + x</span><strong>{state.sum.toPrecision(17)}</strong><span>square root</span><strong>{state.root.toPrecision(17)}</strong></div><div><span>Direct: subtract 1</span><strong>{f(state.direct, 12)}</strong><span>Repaired: x ÷ (root + 1)</span><strong>{f(state.repaired, 12)}</strong></div></div><p className="conditioning-readout" aria-live="polite">Reference ≈ {f(state.reference, 12)}. Direct relative error bound ≤ {f(state.directError.relativeUpper)}; repaired ≤ {f(state.repairedError.relativeUpper)}.</p>{sign === 0 ? <p role="status">The exact answer is zero. Both absolute errors are zero; relative output error is undefined.</p> : <><Drawing compact label="Measured relative error bounds across dyadic exponents" xLabel="Exponent k"><line x1="106" x2="394" y1="183" y2="183" className="axis" /><line x1="106" x2="106" y1="25" y2="183" className="axis" />{[0, Math.floor(bottom / 2), bottom].map(tick => <g key={tick}><line x1="106" x2="394" y1={chartY(10 ** tick)} y2={chartY(10 ** tick)} className="grid" /><text x="98" y={chartY(10 ** tick) + 5} textAnchor="end">1e{tick}</text></g>)}{['directError', 'repairedError'].map((method, index) => samples.filter(row => row.state[method].relativeUpper > 0).map(row => <circle key={`${method}-${row.exponent}`} data-exponent={row.exponent} data-error={row.state[method].relativeUpper} cx={chartX(row.exponent)} cy={chartY(row.state[method].relativeUpper)} r="2.6" className={index ? 'repaired-point' : 'direct-point'} />))}<text x="106" y="207">1</text><text x="394" y="207" textAnchor="end">60</text></Drawing><p className="conditioning-legend"><span className="direct-key">● direct</span><span className="repaired-key">● rationalized</span></p><p className="conditioning-note">Every point is an evaluated binary64 path; the vertical axis is logarithmic. The vertical coordinate is a rounded upper bound from the exact enclosure, not a fitted trend. Exact zero errors are omitted from the log plot. The plot spans k=1…60; the selected k=0 endpoint is described below.</p></>}{state.input === -1 && <p role="status">At x = −1, both formulas return −1 exactly. The derivative is unbounded at this endpoint, so the local derivative condition formula does not apply.</p>}<details><summary>Inspect the exact reference interval</summary><p>Lower endpoint</p><code className="conditioning-rational">{state.referenceLower}</code><p>Upper endpoint</p><code className="conditioning-rational">{state.referenceUpper}</code><p>These are exact fractions. The interval width is at most 2^−160. It can enclose the mathematical answer without containing a rounded approximation to it.</p></details><button onClick={() => {
      setExponent(54);
      setSign(1);
    }}>Reset formulas</button></section>;
}
export function MeasurementSensitivityLab() {
  const [exponent, setExponent] = useState(4);
  const [perturbation, setPerturbation] = useState(1);
  const [singular, setSingular] = useState(false);
  const state = measurementSensitivityState(exponent, perturbation, singular);
  const px = x => 64 + (x + 0.2) * 130;
  const py = y => 216 - (y + 0.2) * 80;
  const line = (epsilon, delta) => [-0.2, 2.2].map(x => `${px(x)},${py((2 + epsilon + delta - x) / (1 + epsilon))}`).join(' ');
  return <section className="conditioning-lab" aria-label="Measurement sensitivity investigation"><h3>Two similar mixtures hide a difference</h3><p>Each line contains all amounts satisfying one reading. Their intersection is the answer. Subtracting the rows gives εx₂ = ε + δ: the small denominator is visible in the algebra as well as the geometry.</p><div className="conditioning-controls"><Slider label="Separation exponent" value={exponent} minimum={1} maximum={7} onChange={setExponent} /><Slider label="Second reading change in units of 1/256" value={perturbation} minimum={-2} maximum={2} onChange={setPerturbation} /><label className="conditioning-checkbox"><input type="checkbox" checked={singular} onChange={event => setSingular(event.target.checked)} />Make the rows identical: ε = 0</label></div><Drawing compact label="Measurement lines and their exact intersection" xLabel="amount x₁" yLabel="amount x₂"><line x1={px(0)} x2={px(2.2)} y1={py(0)} y2={py(0)} className="axis" /><line x1={px(0)} x2={px(0)} y1={py(0)} y2={py(2.2)} className="axis" /><polyline points={line(0, 0)} className="measurement-first" /><polyline points={line(state.epsilon, state.delta)} className="measurement-second" />{[0, 1, 2].map(t => <g key={t}><text x={px(t)} y={py(0) + 23} textAnchor="middle">{t}</text><text x={px(-0.2) - 10} y={py(t) + 5} textAnchor="end">{t}</text></g>)}<circle cx={px(1)} cy={py(1)} r="6" className="reference-dot" />{state.solution && <circle data-solution={state.solution.join(',')} cx={px(state.solution[0])} cy={py(state.solution[1])} r="8" className="chosen" />}</Drawing><p className="conditioning-legend"><span className="direct-key">— first row</span><span className="repaired-key">— perturbed second row</span><span>○ original (1,1)</span></p><p className="conditioning-readout" aria-live="polite">ε = {f(state.epsilon)}; δ = {f(state.delta)}. {state.solution ? `Answer = (${f(state.solution[0])}, ${f(state.solution[1])}); forward relative error = ${f(state.outputRelative)}; bound = ${f(state.bound)}; κ∞ = ${f(state.condition)}.` : state.status === 'nonunique' ? 'Coincident rows: infinitely many solutions.' : 'Different readings for identical rows: no solution.'}</p>{singular && <p role="status">The gap between inconsistent nearly coincident lines may be below screen resolution. The equations, not pixel separation, establish the failure.</p>}<button onClick={() => {
      setExponent(4);
      setPerturbation(1);
      setSingular(false);
    }}>Reset measurements</button></section>;
}
export function BackwardErrorLab() {
  const [power, setPower] = useState(6);
  const [scaled, setScaled] = useState(false);
  const state = backwardWitnessState(power, scaled);
  return <section className="conditioning-lab" aria-label="Backward error investigation"><h3>What is allowed to change?</h3><p>The proposed answer stays x̂ = (1,0), although the exact answer is (1,1). Compare two rules for changing the data. The highlighted off-diagonal zero is decisive.</p><div className="conditioning-controls"><Slider label="Small row exponent" value={power} minimum={1} maximum={9} onChange={setPower} /><label className="conditioning-checkbox"><input type="checkbox" checked={scaled} onChange={event => setScaled(event.target.checked)} />Rescale the second equation to unit coefficient</label></div><div className="conditioning-witness"><div><h4>Recorded second equation</h4><p><mark>0</mark> × x₁ + {f(state.row)} × x₂ = {f(state.row)}</p><p>Substitute (1,0): 0 ≠ {f(state.row)}</p></div><div><h4>Normwise witness</h4><p><mark>{f(state.changedEntry)}</mark> × x₁ + {f(state.row)} × x₂ = {f(state.changedRhs)}</p><p>Substitute (1,0): {f(state.changedEntry)} = {f(state.changedRhs)}</p></div></div><p>The componentwise rule requires |ΔA₂₁| ≤ η|0| = 0. It cannot use that highlighted change; in this case it needs η = 1.</p><Table caption="Two perturbation models, the same proposed answer" headers={['Quantity', 'Value']} rows={[["Residual infinity norm", f(state.row)], ['Forward infinity error', '1'], ['Normwise backward error η', f(state.normwise)], ['Componentwise backward error ηc', '1'], ['Matrix condition κ∞', f(state.condition)], ['Conditional joint forward bound', '2']]} /><p className="conditioning-readout" aria-live="polite">Normwise η = {f(state.normwise)}; componentwise ηc = 1. A second-reading uncertainty of {f(state.displayedReadingUncertainty)} in the displayed units permits an amount uncertainty of 0.01.</p><p>Both rows and their uncertainties change units together. The uncertainty in the inferred amount is unchanged. This example permits unstructured changes to A and b; it does not preserve every property such as symmetry.</p><button onClick={() => {
      setPower(6);
      setScaled(false);
    }}>Reset backward error</button></section>;
}
function SumTree({
  node
}) {
  return <li><span>{String(node.value)}</span>{node.children.length > 0 && <><small>rounded sum of children</small><ul>{node.children.map(child => <SumTree key={child.start} node={child} />)}</ul></>}</li>;
}
export function SummationLab() {
  const [preset, setPreset] = useState('cancellation');
  const [step, setStep] = useState(1);
  const treeRegion = useRef(null);
  useEffect(() => {
    const region = treeRegion.current;
    if (region) region.scrollLeft = (region.scrollWidth - region.clientWidth) / 2;
  }, [preset]);
  const state = summationState(preset);
  const current = state.steps[Math.min(step - 1, state.steps.length - 1)];
  return <section className="conditioning-lab" aria-label="Summation investigation"><h3>Follow the lost contribution</h3><p>Inspect which method will win while changing the order. The table inspects an explicit left fold, not Python's version-dependent built-in sum.</p><div className="conditioning-controls"><label>Input order<select aria-label="Input order" value={preset} onChange={event => {
          setPreset(event.target.value);
          setStep(1);
        }}><option value="cancellation">Large, one, opposite large</option><option value="reordered">Large, opposite large, one</option><option value="positive">Four ones after 2^53</option><option value="changed">Large, three, opposite large</option><option value="zero">Exact zero sum</option></select></label><Slider label="Inspect addition" value={Math.min(step, state.steps.length)} minimum={1} maximum={state.steps.length} onChange={setStep} /></div><div className="conditioning-addition"><p><strong>{String(current.before)}</strong> + <strong>{String(current.value)}</strong> → stored <strong>{String(current.updated)}</strong></p><p>Exact local sum minus stored subtotal: <strong>{current.lost}</strong></p><p>Neumaier's accumulated correction: <strong>{f(current.correction, 10)}</strong></p><p>Exact sum of all inputs seen so far: <strong>{current.exactPrefix}</strong></p></div><details open><summary>Inspect the balanced addition tree</summary><p className="conditioning-note">Read from the leaves toward the final sum at the top. The branches extend sideways: scroll or use the arrow keys to inspect each addition.</p><div ref={treeRegion} className="conditioning-tree-scroll" tabIndex={0} role="region" aria-label="Balanced addition tree; scroll horizontally if needed"><ul className="conditioning-tree"><SumTree node={state.balanced} /></ul></div></details><Table caption="Actual Number results for this input order" headers={['Method', 'Result']} rows={[["Exact sum of stored inputs", state.exact], ['Naive left fold', f(state.naive, 15)], ['Balanced tree', f(state.balanced.value, 15)], ['Classic Kahan', f(state.kahan, 15)], ['Neumaier', f(state.neumaier, 15)]]} /><p className="conditioning-readout" aria-live="polite">Exact sum = {state.exact}; naive = {f(state.naive, 15)}; balanced = {f(state.balanced.value, 15)}; Neumaier = {f(state.neumaier, 15)}.</p><p>Componentwise relative sum condition ≈ {f(state.condition)}. The naive absolute error bound γₙ₋₁Σ|xᵢ| ≈ {f(state.gammaBound)}. {state.condition === null && 'The zero exact sum has no relative output-error denominator.'} The correction channel is another rounded computation; none of these methods is guaranteed exact for every sequence.</p><button onClick={() => {
      setPreset('cancellation');
      setStep(1);
    }}>Reset summation</button></section>;
}
export function ErrorPropagationLab() {
  const [q, setQ] = useState(0.5);
  const [mode, setMode] = useState('constant');
  const [steps, setSteps] = useState(12);
  const [refinement, setRefinement] = useState(8);
  const state = propagationState(q, mode, steps);
  const unstable = unstableRefinementState(refinement);
  const height = Math.max(0.02, ...state.frames.map(frame => frame.bound));
  const px = index => 115 + 270 * index / steps;
  const py = error => 113 - 82 * error / height;
  const points = field => state.frames.map(frame => `${px(frame.step)},${py(frame[field])}`).join(' ');
  return <section className="conditioning-lab" aria-label="Error propagation investigation"><h3>Does each step damp or amplify what came before?</h3><p>Start with zero error. The multiplier acts on the old error before a new disturbance arrives. Compare the signed trace with an envelope that allows disturbances to align unfavorably.</p><div className="conditioning-controls"><label>Multiplier q<select aria-label="Multiplier q" value={q} onChange={event => setQ(Number(event.target.value))}>{[-0.5, 0.5, 0.9, 1, 1.1].map(value => <option value={value} key={value}>{value}</option>)}</select></label><label>Disturbance pattern<select aria-label="Disturbance pattern" value={mode} onChange={event => setMode(event.target.value)}><option value="constant">+0.01 every step</option><option value="alternating">Alternating ±0.01</option><option value="pulse">+0.01 on first step only</option></select></label><Slider label="Propagation steps" value={steps} minimum={1} maximum={24} onChange={setSteps} /></div><Drawing compact label="Signed propagated error and worst-case envelope"><line x1="115" x2="385" y1="113" y2="113" className="axis" />{[-height, 0, height].map(value => <text key={value} x="107" y={py(value) + 5} textAnchor="end">{f(value, 2)}</text>)}<polyline points={points('bound')} className="error-bound" /><polyline points={state.frames.map(frame => `${px(frame.step)},${py(-frame.bound)}`).join(' ')} className="error-bound" /><polyline points={points('error')} className="error-trace" />{state.frames.map(frame => <circle data-step={frame.step} data-error={frame.error} data-bound={frame.bound} key={frame.step} cx={px(frame.step)} cy={py(frame.error)} r="3" className="repaired-point" />)}<text x="115" y="225">step 0</text><text x="385" y="225" textAnchor="end">step {steps}</text></Drawing><p className="conditioning-readout" aria-live="polite">Final signed error = {f(state.frames.at(-1).error)}; envelope radius = {f(state.frames.at(-1).bound)}.</p><p className="conditioning-legend"><span className="repaired-key">— actual signed recurrence</span><span className="direct-key">- - permitted envelope</span></p><h4>Now refine a different, unstable recurrence</h4><p>For the constant exact solution u(t)=1, initialize y₀=1 and y₁=1+h², then use yₙ₊₂=3yₙ₊₁−2yₙ. Hold final time at T=1 while changing N.</p><label>Fixed-time refinement<select aria-label="Fixed-time refinement" value={refinement} onChange={event => setRefinement(Number(event.target.value))}>{[4, 8, 16, 32].map(value => <option key={value} value={value}>N = {value}, h = 1/{value}</option>)}</select></label><div className="conditioning-refinement"><div><span>Smaller start perturbation</span><strong>1/{refinement ** 2}</strong></div><span aria-hidden="true">→</span><div><span>Error at the same final time</span><strong>{2 ** refinement - 1}/{refinement ** 2}</strong></div></div><p data-refinement-error={unstable.finalError}>Final error = {f(unstable.finalError, 9)}. The formula (2ᴺ−1)/N² exposes amplification by an extra recurrence mode. The exact solution stays constant; the growth is caused by the method.</p><details><summary>Inspect every fixed-time step</summary><Table caption="Exact dyadic recurrence samples" headers={['Step', 'Time', 'Error']} rows={unstable.frames.map(frame => [frame.step, f(frame.time), f(frame.error, 9)])} /></details><button onClick={() => {
      setQ(0.5);
      setMode('constant');
      setSteps(12);
      setRefinement(8);
    }}>Reset propagation</button></section>;
}
export function DefectResidualFigure() {
  return <figure className="conditioning-figure"><div className="conditioning-defects"><div><strong>Restricted exact solution uₕ</strong><span>Substitute into Aₕuₕ</span><b>Defect τₕ = bₕ − Aₕuₕ</b></div><div><strong>Computed solution x̂ₕ</strong><span>Substitute into Aₕx̂ₕ</span><b>Residual rₕ = bₕ − Aₕx̂ₕ</b></div></div><figcaption>Subtract the two equations: Aₕ(x̂ₕ−uₕ)=τₕ−rₕ. A small solver residual controls only one contribution; the discrete operator also converts those contributions into solution error.</figcaption></figure>;
}
