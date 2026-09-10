import { useId, useState } from 'react';
import { LessonTable } from './LessonElements';
import { INFORMATION_PARTITIONS, LIMIT_MODES, TARGET_MEASURES, informationState, preimageState, mixedMeasureState, cantorCoverState, simpleIntegralState, limitIntegralState, jointCellState, signedArrayState, conditionalMeanState, parseLosses, reweightState, measureNumber as number } from '../../data/measure-theory-models';
import './measure-theory-labs.css';
const gold = '#e2b55a',
  blue = '#8ecbd1',
  pink = '#ff9db2';
const setLabel = values => values.length ? '{' + values.join(', ') + '}' : '∅';
function Select({
  label,
  value,
  setValue,
  options
}) {
  return <label>{label}<select value={value} onChange={event => setValue(event.target.value)}>{Object.entries(options).map(([key, item]) => <option key={key} value={key}>{typeof item === 'string' ? item : item.label}</option>)}</select></label>;
}
function Range({
  label,
  value,
  setValue,
  min,
  max,
  step = 1
}) {
  return <label>{label}<strong>{number(value)}</strong><input type="range" aria-label={label} min={min} max={max} step={step} value={value} onChange={event => setValue(Number(event.target.value))} /></label>;
}
function Metric({
  label,
  children
}) {
  return <div><dt>{label}</dt><dd>{children}</dd></div>;
}
function Lab({
  title,
  children,
  name
}) {
  const id = useId();
  return <section className="measure-lab" data-measure-lab={name} aria-labelledby={id}><h3 id={id}>{title}</h3>{children}<p className="measure-note">The display uses the stated deterministic model. Decimal readouts are rounded; finite pictures do not establish an infinite limiting claim.</p></section>;
}
function Axes({
  maxY = 1,
  xLabel = 'x',
  yLabel = 'value'
}) {
  return <g className="measure-axes"><path d="M45 20 V240 H345" /><text x="36" y="259">0</text><text x="336" y="259">1</text><text x="15" y="32">{number(maxY, 2)}</text><text x="185" y="282">{xLabel}</text><text x="45" y="20">{yLabel}</text></g>;
}
export function EventInformationLab() {
  const [partition, setPartition] = useState('pairs');
  const [mask, setMask] = useState(3);
  const state = informationState(partition, mask);
  return <Lab title="Which questions can the observation answer?" name="events">
    <p>Choose an event by selecting faces. Each outlined cell groups outcomes that produce the same observation. Can its label alone tell you whether the event occurred?</p>
    <div className="measure-controls"><Select label="Observed information" value={partition} setValue={setPartition} options={INFORMATION_PARTITIONS} /></div>
    <div className="measure-cells">{state.cells.map(cell => <div key={cell.faces.join()} className={cell.split ? 'measure-cell split' : 'measure-cell'}><span>Observation {setLabel(cell.faces)}</span><div>{cell.faces.map(face => <button key={face} aria-label={'Face ' + face} aria-pressed={state.event.includes(face)} onClick={() => setMask(mask ^ 1 << face - 1)}>{face}</button>)}</div><strong>{cell.split ? 'Split: label cannot answer' : cell.included.length ? 'Whole cell included' : 'Whole cell excluded'}</strong></div>)}</div>
    <div className="measure-controls"><button onClick={() => setMask(0)}>Empty event</button><button onClick={() => setMask(63)}>Every outcome</button><button onClick={() => setMask(7)}>Faces 1, 2, 3</button></div>
    <dl className="measure-metrics" aria-live="polite"><Metric label="Selected event">{setLabel(state.event)}</Metric><Metric label="Observable from this information?">{state.observable ? 'Yes: union of whole cells' : 'No: at least one cell is split'}</Metric><Metric label="Ambient fair-die probability">{state.event.length}/6 = {number(state.ambientProbability)}</Metric><Metric label="Events in this information algebra">{state.observableEventCount}</Metric></dl>
    <p>The event still has a probability in the full die space even when this observation cannot resolve it. Select faces 1, 2, 3, then compare pairs with exact-face information.</p>
    <details><summary>Inspect every observable event</summary><p className="measure-set-list">{state.algebra.map(event => setLabel(event)).join(' · ')}</p></details>
  </Lab>;
}
export function PreimageFigure() {
  const [threshold, setThreshold] = useState(1);
  const state = preimageState(threshold);
  return <Lab title="Follow a value question backwards" name="preimage">
    <p>The map is Z=max(face−3, 0). First select output values Z≤t; then collect every input that maps there. Multiple inputs can share one output.</p>
    <div className="measure-controls"><Range label="Output threshold t" value={threshold} setValue={setThreshold} min={0} max={3} /></div>
    <div className="measure-map">{state.rows.map(row => <div key={row.face} className={row.selected ? 'selected' : ''}><span>Face {row.face}</span><span aria-hidden="true">→</span><strong>Z={row.value}</strong><span>{row.selected ? 'In preimage' : 'Outside'}</span></div>)}</div>
    <dl className="measure-metrics" aria-live="polite"><Metric label="Input event Z⁻¹((−∞, t])">{setLabel(state.preimage)}</Metric><Metric label="Its probability">{state.preimage.length}/6 = {number(state.probability)}</Metric></dl>
    <LessonTable caption="Induced law: each input has mass 1/6" headers={['Output z', 'P(Z=z)']} rows={state.law.map(row => [row.value, number(row.probability)])} />
  </Lab>;
}
export function PrefixContinuityFigure() {
  return <figure className="measure-inline"><figcaption>One infinite sequence is trapped inside every matching prefix</figcaption>
    <p>Fix a particular target sequence beginning H T H H. A prefix event leaves every later flip unrestricted.</p>
    {[0, 1, 2, 3, 4].map(n => <div className="measure-prefix" key={n}><span>{n ? ['H', 'T', 'H', 'H'].slice(0, n).join(' ') + ' …' : 'Any sequence'}</span><div><i style={{
          width: 100 * 2 ** -n + '%'
        }} /></div><strong>2<sup>−{n}</sup></strong></div>)}
    <p>Bar lengths show exact prefix probabilities, not a drawing of all sequences. The events decrease. Their intersection fixes every flip; continuity from above gives that singleton probability lim 2<sup>−n</sup>=0.</p>
  </figure>;
}
export function MixedMeasureLab() {
  const [weight, setWeight] = useState(.25);
  const [interval, setInterval] = useState('half');
  const intervals = {
    half: [0, .5],
    point: [0, 0],
    positive: [.25, .75],
    all: [0, 1],
    outside: [-.25, -.1]
  };
  const state = mixedMeasureState(weight, ...intervals[interval]);
  const left = Math.max(0, state.lower),
    right = Math.min(1, state.upper);
  return <Lab title="A point mass and a density use different bookkeeping" name="mixture">
    <p>A device returns exactly 0 with probability w. Otherwise it returns a Uniform[0,1] value. The atomic mass below is a probability; the continuous plot's vertical axis is probability per unit x.</p>
    <div className="measure-controls"><Range label="Atomic probability w" value={weight} setValue={setWeight} min={0} max={1} step={.05} /><Select label="Closed event interval" value={interval} setValue={setInterval} options={{
        half: '[0, 0.5]',
        point: '[0, 0]: one point',
        positive: '[0.25, 0.75]',
        all: '[0, 1]',
        outside: '[−0.25, −0.1]'
      }} /></div>
    <div className="measure-two-plots"><div><h4>Continuous contribution only</h4><svg className="measure-plot" viewBox="0 0 370 290" role="img" aria-label="Uniform continuous density and selected interval area"><Axes yLabel="density" /><rect x="45" y={240 - 210 * state.continuousDensity} width="300" height={210 * state.continuousDensity} fill={blue} opacity=".1" />{right > left && <rect x={45 + 300 * left} y={240 - 210 * state.continuousDensity} width={300 * (right - left)} height={210 * state.continuousDensity} fill={blue} opacity=".5" />}<path d={'M45 ' + (240 - 210 * state.continuousDensity) + ' H345'} stroke={blue} strokeWidth="3" /></svg><p>Atomic contribution at x=0: <strong className="measure-gold">{number(weight)}</strong>. {state.atomIncluded ? 'Included in the selected event.' : 'Outside the selected event.'} It is not drawn as a finite-width density spike.</p></div>
    <div><h4>CDF includes both contributions</h4><svg className="measure-plot" viewBox="0 0 370 290" role="img" aria-label="CDF with a jump of atomic probability at zero"><Axes yLabel="P(X≤x)" /><path d={'M45 ' + (240 - 210 * weight) + ' L345 30'} fill="none" stroke={gold} strokeWidth="3" /><path d={'M45 240 V' + (240 - 210 * weight)} stroke={pink} strokeWidth="3" strokeDasharray="5 4" /><circle cx="45" cy="240" r="5" fill="#151515" stroke={gold} /><circle cx="45" cy={240 - 210 * weight} r="5" fill={gold} /></svg><p>The jump at zero is w. The dashed segment marks that jump; its interior is not part of the CDF graph.</p></div></div>
    <dl className="measure-metrics" aria-live="polite"><Metric label="Atom in event">{number(state.atomMass)}</Metric><Metric label="Continuous area in event">{number(state.continuousMass)}</Metric><Metric label="Total event probability">{number(state.probability)}</Metric><Metric label="Entire law has a Lebesgue density?">{state.hasLebesgueDensity ? 'Yes, w=0' : 'No, positive mass at a length-zero point'}</Metric></dl>
    <p>Prediction: selecting the single point [0,0] makes the continuous area zero. Does the event probability also become zero? Then set w=0 and compare.</p>
  </Lab>;
}
export function CantorCoverFigure() {
  return <figure className="measure-inline"><figcaption>Probability can remain while supporting length disappears</figcaption>
    {[0, 1, 2, 3, 4].map(level => {
      const state = cantorCoverState(level);
      return <div className="measure-cantor-row" key={level}><p>Level {level}: length {number(state.totalLength)}, probability 1</p><svg viewBox="0 0 360 22" role="img" aria-label={'Cantor cover level ' + level + ', ' + state.intervalCount + ' retained intervals'}>{state.intervals.map((interval, index) => <rect key={index} x={360 * interval.low} y="2" width={360 * (interval.high - interval.low)} height="18" fill={gold} />)}</svg></div>;
    })}<p>At level n there are 2ⁿ intervals, each of length 3⁻ⁿ and probability 2⁻ⁿ under the Cantor law. The drawing shows five finite covers; the text proves the properties of their infinite intersection.</p>
  </figure>;
}
export function SimpleIntegralLab() {
  const [level, setLevel] = useState(2);
  const state = simpleIntegralState(level);
  const curve = Array.from({
    length: 101
  }, (_, i) => 45 + 3 * i + ',' + (240 - 210 * (i / 100) ** 2)).join(' ');
  return <Lab title="Build an integral from measurable value bands" name="simple">
    <p>For X uniform on [0,1], approximate f(x)=x² by rounding its value down to multiples of 1/N, with N=2ⁿ. The input bins have unequal widths because we are dividing the output values equally.</p>
    <div className="measure-controls"><Range label="Refinement n" value={level} setValue={setLevel} min={0} max={6} /></div>
    <svg className="measure-plot" viewBox="0 0 370 290" role="img" aria-label="Lower simple-function steps under x squared"><Axes yLabel="function value" />{state.steps.map(step => <rect key={step.index} x={45 + step.low * 300} y={240 - step.value * 210} width={step.mass * 300} height={step.value * 210} fill={gold} opacity=".4" stroke={gold} strokeWidth=".7" />)}<polyline points={curve} fill="none" stroke={blue} strokeWidth="3" /></svg>
    <dl className="measure-metrics" aria-live="polite"><Metric label="Number of value bands">{state.count}</Metric><Metric label="Lower simple integral">{number(state.lowerIntegral, 6)}</Metric><Metric label="Exact ∫x² dx">1/3</Metric><Metric label="Remaining integral error">{number(state.error, 6)} ≤ {number(state.uniformErrorBound)}</Metric></dl>
    <details><summary>Inspect band widths and contributions</summary><LessonTable caption="Each band contributes its value times its probability" headers={['Band k', 'Input interval (endpoints rounded)', 'Value k/N', 'Width = P(band)', 'Contribution']} rows={state.steps.map(step => [step.index, '[' + number(step.low) + ', ' + number(step.high) + ')', number(step.value), number(step.mass), number(step.contribution)])} /><p>The point x=1 can be assigned value 1. It has probability zero, so it changes none of these integrals.</p></details>
    <p>The gold steps are the lower simple function; the blue curve is x². Doubling N refines every band and never decreases this lower approximation. A finite error bound, 1/N, explains convergence without trusting how smooth the picture looks.</p>
  </Lab>;
}
export function IntegralLimitLab() {
  const [mode, setMode] = useState('spike');
  const [n, setN] = useState(4);
  const [x, setX] = useState(.25);
  const state = limitIntegralState(mode, n, x);
  const py = value => 240 - 210 * value / state.yMaximum;
  const points = state.points.map(point => 45 + 300 * point.x + ',' + py(point.y)).join(' ');
  return <Lab title="Follow a fixed point and the whole integral separately" name="limits">
    <div className="measure-controls"><Select label="Function sequence" value={mode} setValue={setMode} options={LIMIT_MODES} /><Range label="Sequence index n" value={n} setValue={setN} min={1} max={64} /><Range label="Fixed observation x" value={x} setValue={setX} min={0} max={1} step={.01} /></div>
    <p>{mode === 'spike' ? 'fₙ(x)=n on the open interval (0,1/n), and 0 elsewhere. Width shrinks as height rises.' : mode === 'bounded' ? 'fₙ(x)=xⁿ. Every curve stays between 0 and the same integrable bound 1.' : 'fₙ(x)=min(n,1/√x), with fₙ(0)=n. These are increasing truncations of an integrable singularity.'}</p>
    <svg className="measure-plot" viewBox="0 0 370 290" role="img" aria-label="Function curve with selected fixed point and integral area"><Axes maxY={state.yMaximum} yLabel="fₙ(x)" />{mode === 'spike' ? <><rect x="45" y="30" width={300 / n} height="210" fill={gold} opacity=".35" /><path d={'M' + (45 + 300 / n) + ' 240 H345 M45 30 H' + (45 + 300 / n)} fill="none" stroke={gold} strokeWidth="3" /><circle cx="45" cy="30" r="4" fill="#151515" stroke={gold} /><circle cx={45 + 300 / n} cy="30" r="4" fill="#151515" stroke={gold} /><circle cx="45" cy="240" r="4" fill={gold} /><circle cx={45 + 300 / n} cy="240" r="4" fill={gold} /></> : <><polygon points={'45,240 ' + points + ' 345,240'} fill={gold} opacity=".22" /><polyline points={points} fill="none" stroke={gold} strokeWidth="3" /></>}<line x1={45 + 300 * x} x2={45 + 300 * x} y1="25" y2="240" stroke={blue} strokeDasharray="5 4" /><circle cx={45 + 300 * x} cy={py(state.value)} r="5" fill={blue} /></svg>
    <p>The vertical scale changes with n in the spike and truncation examples; read its top label. Smooth curves use analytic samples with extra points near the corner. The integral readout uses the exact formula, not pixel area.</p>
    <dl className="measure-metrics" aria-live="polite"><Metric label="Value at the selected fixed x">{number(state.value)}</Metric><Metric label="Its pointwise limit">{number(state.pointwiseLimit)}</Metric><Metric label="Current integral">{number(state.integral)}</Metric><Metric label="Limit of integrals / integral of limit">{number(state.limitOfIntegrals)} / {number(state.integralOfLimit)}</Metric></dl>
    <p>{mode === 'spike' ? 'Keep x=0.25 fixed and advance past n=4: the value is 0 from that point onward, while every integral is 1. Tracking x=1/(2n) would move the point each time and would answer a different question.' : mode === 'bounded' ? 'At x=1 the limit is 1; at every x<1 it is 0. That exceptional point has length zero, so the limiting integral is still 0.' : 'The height at x=0 tends to +∞, but that singleton has length zero. The limiting function has integral 2; nonnegative monotone convergence applies.'}</p>
  </Lab>;
}
function CoordinateCell({
  state,
  transformed
}) {
  const sx = transformed ? state.scaleX : 1,
    sy = transformed ? state.scaleY : 1;
  const unit = 76;
  return <svg className="measure-cell-plot" viewBox="0 0 315 310" role="img" aria-label={transformed ? 'Scaled rectangle in u and v coordinates' : 'Unit square in x and y coordinates'}>
    <path d="M44 24 V264 H290" fill="none" stroke="#888" />
    {state.cells.map(cell => <rect key={cell.row * 4 + cell.column} x={44 + cell.column * sx * unit / 4} y={264 - (cell.row + 1) * sy * unit / 4} width={sx * unit / 4} height={sy * unit / 4} fill={blue} fillOpacity={.12 + 2 * cell.probability} stroke="#555" />)}
    <rect x={44 + state.x0 * sx * unit} y={264 - state.y1 * sy * unit} width={sx * unit / 4} height={sy * unit / 4} fill={gold} opacity=".8" stroke={gold} strokeWidth="2" />
    <text x="22" y="284">0</text><text x={44 + sx * unit} y="284" textAnchor="middle">{number(sx)}</text><text x="31" y={268 - sy * unit} textAnchor="end">{number(sy)}</text><text x="185" y="307">{transformed ? 'u' : 'x'}</text><text x="13" y="23">{transformed ? 'v' : 'y'}</text>
  </svg>;
}
export function JointMeasureLab() {
  const [column, setColumn] = useState(2),
    [row, setRow] = useState(1);
  const [sx, setSx] = useState(2),
    [sy, setSy] = useState(3);
  const state = jointCellState(column, row, sx, sy);
  return <Lab title="Change coordinates; preserve the selected probability" name="joint">
    <p>The original joint density is 4xy on the unit square. Select one of sixteen rectangles, then stretch coordinates by u=ax and v=by. Both diagrams use the same drawing length per coordinate unit, so area scaling is visible.</p>
    <div className="measure-controls"><Range label="Cell column, from left" value={column} setValue={setColumn} min={0} max={3} /><Range label="Cell row, from bottom" value={row} setValue={setRow} min={0} max={3} /><Range label="Horizontal scale a" value={sx} setValue={setSx} min={.5} max={3} step={.5} /><Range label="Vertical scale b" value={sy} setValue={setSy} min={.5} max={3} step={.5} /></div>
    <div className="measure-two-plots"><div><h4>Original coordinates</h4><CoordinateCell state={state} transformed={false} /><p>x∈[{number(state.x0)}, {number(state.x1)}], y∈[{number(state.y0)}, {number(state.y1)}]</p></div><div><h4>Transformed coordinates</h4><CoordinateCell state={state} transformed /><p>u∈[{number(state.u0)}, {number(state.u1)}], v∈[{number(state.v0)}, {number(state.v1)}]</p></div></div>
    <dl className="measure-metrics" aria-live="polite"><Metric label="Area before → after">{number(state.area)} → {number(state.transformedArea)}</Metric><Metric label="Density at cell center before → after">{number(state.centerDensity)} → {number(state.transformedDensity)}</Metric><Metric label="Area multiplier |det J|">{number(state.jacobian)}</Metric><Metric label="Unchanged cell probability">{number(state.probability)}</Metric></dl>
    <p>The center density times area is exact for this bilinear density on a rectangle. It is generally only an approximation for a varying density. The cell masses here come from exact integrals; shading only helps locate the selected cell.</p>
  </Lab>;
}
export function SignedArrayFigure() {
  const square = signedArrayState(4, 4),
    wide = signedArrayState(4, 5);
  return <figure className="measure-inline"><figcaption>Finite reordering works; two infinite limiting orders can disagree</figcaption>
    <LessonTable caption="First four rows, with the next column included" headers={['Row / column', '1', '2', '3', '4', '5', 'Row sum']} rows={wide.values.map((row, i) => [i + 1, ...row, wide.rowSums[i]])} />
    <p>The 4×4 square sums to {square.finiteSum}: its last +1 has no −1 inside that square. The 4×5 window sums to {wide.finiteSum}. Each fixed finite window gives the same total by rows or columns. Only the choice of windows approaching infinity differs.</p>
  </figure>;
}
export function ConditionalMeanLab() {
  const [partition, setPartition] = useState('pairs'),
    [sampling, setSampling] = useState('fair');
  const [draft, setDraft] = useState('0, 2, 4, 4, 8, 12'),
    [values, setValues] = useState([0, 2, 4, 4, 8, 12]);
  const [error, setError] = useState(''),
    [nullValue, setNullValue] = useState(0);
  const state = conditionalMeanState(partition, sampling, values, nullValue);
  function applyLosses(event) {
    event.preventDefault();
    try {
      setValues(parseLosses(draft));
      setError('');
    } catch (caught) {
      setError(caught.message);
    }
  }
  return <Lab title="A prediction must be constant inside each information cell" name="conditional">
    <p>The outcomes carry losses. If an observation hides which member of its cell occurred, your prediction must use the same number for every member. The conditional mean chooses that number by probability-weighted averaging.</p>
    <div className="measure-controls"><Select label="Prediction information" value={partition} setValue={setPartition} options={INFORMATION_PARTITIONS} /><Select label="Outcome probabilities" value={sampling} setValue={setSampling} options={{
        fair: 'Fair: every face has probability 1/6',
        'missing-six': 'Faces 1–5: 1/5 each; face 6: 0'
      }} /></div>
    <form onSubmit={applyLosses} className="measure-controls"><label>Six losses, face order<input type="text" value={draft} onChange={event => setDraft(event.target.value)} aria-invalid={Boolean(error)} /></label><button type="submit">Apply losses</button></form>{error && <p role="alert">{error} The last valid calculation stays visible.</p>}
    <div className="measure-cells">{state.cells.map(cell => <div className="measure-cell" key={cell.faces.join()}><span>Observed {setLabel(cell.faces)}</span><strong>Predict {number(cell.prediction)}</strong><span>Cell probability {number(cell.mass)}</span><span>{cell.mass ? 'Weighted loss / cell probability' : 'Null cell: value is a chosen version'}</span></div>)}</div>
    {state.hasNullCell && <div className="measure-controls"><Range label="Chosen prediction on the null cell" value={nullValue} setValue={setNullValue} min={-20} max={20} /></div>}
    <LessonTable caption="Outcome, conditional prediction and squared-error contribution" headers={['Face', 'P(face)', 'Loss Y', 'Prediction Z', 'Residual Y−Z', 'P(face) × residual²']} rows={state.rows.map(item => [item.face, number(item.weight), item.loss, number(item.prediction), number(item.residual), number(item.weightedSquare)])} />
    <dl className="measure-metrics" aria-live="polite"><Metric label="E[Y] = E[Z]">{number(state.mean)} = {number(state.predictionMean)}</Metric><Metric label="Mean squared residual">{number(state.risk)}</Metric><Metric label="Variance of Y">{number(state.variance)}</Metric><Metric label="Variance explained by Z">{number(state.predictionVariance)}</Metric></dl>
    <p>Compare no information → pairs → exact face. This is a nested refinement: expected squared error cannot increase. Parity and pairs are not nested; changing between them is not automatically adding information.</p>
  </Lab>;
}
export function DensityRatioLab() {
  const [source, setSource] = useState('fair'),
    [target, setTarget] = useState('lossHeavy');
  const state = reweightState(source, target);
  return <Lab title="Can weights transfer the source measure to this target?" name="ratios">
    <div className="measure-controls"><Select label="Source probability P" value={source} setValue={setSource} options={{
        fair: 'Uniform over all six faces',
        'missing-six': 'Uniform over faces 1–5 only'
      }} /><Select label="Target probability Q" value={target} setValue={setTarget} options={TARGET_MEASURES} /></div>
    <LessonTable caption="A density ratio transfers probability, not outcomes" headers={['Face', 'P', 'Q', 'w = Q/P', 'Loss']} rows={state.rows.map(row => [row.face, number(row.p), number(row.q), row.unsupported ? 'No valid weight' : number(row.ratio), row.loss])} />
    <dl className="measure-metrics" aria-live="polite"><Metric label="Does Q give zero mass to every P-null event?">{state.supported ? 'Yes: Q ≪ P' : 'No: support condition fails'}</Metric><Metric label="Direct target expectation E_Q[Y]">{number(state.targetMean)}</Metric><Metric label="Reweighted source expectation E_P[wY]">{state.supported ? number(state.weightedMean) : 'Identity unavailable'}</Metric><Metric label="Source expectation of weight E_P[w]">{state.supported ? number(state.ratioMean) : 'No probability-preserving ratio'}</Metric></dl>
    <p>{state.supported ? 'Every target-positive face is possible under the source. Each product P(face)×w(face) recovers Q(face).' : 'Target mass ' + number(state.unsupportedMass) + ' sits on source-null faces. Any finite weight times zero remains zero, so source samples cannot recover it by reweighting.'}</p>
  </Lab>;
}
