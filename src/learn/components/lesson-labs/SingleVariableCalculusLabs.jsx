import { useId, useState } from 'react';
import { compositionChange, exponentialRate, extremaCandidates, extremaValue, improperPowerIntegral, limitGuarantee, motionAccumulation, motionPosition, motionRate, motionVelocity, taylorApproximation } from '../../data/single-variable-calculus-models.js';
import './single-variable-calculus-labs.css';
function number(value, digits = 4) {
  if (value === null) return 'undefined';
  if (value === 0) return '0';
  if (Math.abs(value) >= 10000 || Math.abs(value) < 0.0001) return value.toExponential(3);
  return Number(value.toFixed(digits)).toString();
}
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.1,
  suffix = ''
}) {
  return <label className="calculus-field">
    <span>{label}: <strong>{number(value)}{suffix}</strong></span>
    <input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}
function Select({
  label,
  value,
  onChange,
  options
}) {
  return <label className="calculus-field"><span>{label}</span>
    <select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
    </select>
  </label>;
}
function Investigation({
  title,
  guidance,
  children,
  onReset,
  resetLabel
}) {
  return <section data-live-exploration className="calculus-lab" aria-label={title}>
    <h3>{title}</h3><p>{guidance}</p>{children}
    <button type="button" onClick={onReset}>{resetLabel}</button>
  </section>;
}
function CalculusPlot({
  label,
  xRange,
  yRange,
  xLabel,
  yLabel,
  curves = [],
  children
}) {
  const clipId = useId().replaceAll(':', '');
  const yTicks = [yRange[0], (yRange[0] + yRange[1]) / 2, yRange[1]];
  // Reserve space for the actual labels when a divergent integral becomes large.
  const left = Math.max(46, ...yTicks.map(tick => number(tick, 2).length * 9 + 12));
  const right = 292;
  const top = 32;
  const bottom = 220;
  const x = value => left + (value - xRange[0]) * (right - left) / (xRange[1] - xRange[0]);
  const y = value => bottom - (value - yRange[0]) * (bottom - top) / (yRange[1] - yRange[0]);
  const xTicks = [xRange[0], (xRange[0] + xRange[1]) / 2, xRange[1]];
  return <svg className="calculus-plot" viewBox="0 0 320 276" role="img" aria-label={label}>
    <defs><clipPath id={clipId}><rect x={left} y={top} width={right - left} height={bottom - top} /></clipPath></defs>
    <text x={left} y="18" className="axis-title">{yLabel}</text>
    {yTicks.map(tick => <g key={tick}><line className="grid" x1={left} x2={right} y1={y(tick)} y2={y(tick)} /><text x={left - 7} y={y(tick) + 5} textAnchor="end">{number(tick, 2)}</text></g>)}
    {xTicks.map((tick, index) => <g key={tick}><line className="grid" x1={x(tick)} x2={x(tick)} y1={top} y2={bottom} /><text x={x(tick)} y={bottom + 22} textAnchor={index === 0 ? "start" : index === 2 ? "end" : "middle"}>{number(tick, 2)}</text></g>)}
    <path className="axis" d={`M${left},${top}V${bottom}H${right}`} />
    <text x={(left + right) / 2} y="268" textAnchor="middle" className="axis-title">{xLabel}</text>
    <g clipPath={`url(#${clipId})`}>
      {yRange[0] < 0 && yRange[1] > 0 && <line className="zero-line" x1={left} x2={right} y1={y(0)} y2={y(0)} />}
      {curves.map((curve, index) => {
        const domain = curve.domain || xRange;
        const points = Array.from({
          length: 161
        }, (_, pointIndex) => {
          const input = domain[0] + (domain[1] - domain[0]) * pointIndex / 160;
          return `${x(input)},${y(curve.fn(input))}`;
        }).join(' ');
        return <polyline key={index} className={curve.className || 'gold'} points={points} fill="none" />;
      })}
      {children?.({
        x,
        y,
        left,
        right,
        top,
        bottom
      })}
    </g>
  </svg>;
}
export function MotionRateLab() {
  const [baseTime, setBaseTime] = useState(2);
  const [increment, setIncrement] = useState(0.5);
  const state = motionRate(baseTime, increment);
  return <Investigation title="Bring a second observation closer" guidance="At t=2 the object is moving backwards. Inspect whether the interval slope is also negative, then shrink the interval. Next move to a turning point at t=1." onReset={() => {
    setBaseTime(2);
    setIncrement(0.5);
  }} resetLabel="Reset motion rate">
    <div className="calculus-controls">
      <Range label="Base time" value={baseTime} onChange={setBaseTime} min={1} max={3} step={0.1} suffix=" s" />
      <Select label="Signed time increment" value={increment} onChange={value => setIncrement(Number(value))} options={[1, 0.5, 0.1, 0.01, -0.01, -0.1, -0.5, -1].map(value => [value, `${value > 0 ? '+' : ''}${value} s`])} />
    </div>
    <CalculusPlot label="Position curve with a secant through two observations and a local tangent" xRange={[0, 4]} yRange={[-2, 6]} xLabel="Time (s)" yLabel="Position (m)" curves={[{
      fn: motionPosition
    }, {
      fn: time => state.position + state.secant * (time - baseTime),
      className: 'blue dashed'
    }, {
      fn: time => state.position + state.velocity * (time - baseTime),
      className: 'rose'
    }]}>
      {({
        x,
        y
      }) => <>
        <circle className="point" cx={x(baseTime)} cy={y(state.position)} r="5" />
        <circle className="point blue-point" cx={x(baseTime + increment)} cy={y(state.nextPosition)} r="5" />
      </>}
    </CalculusPlot>
    <p className="calculus-legend"><span>Gold: position</span><span className="blue-key">Dashed blue: secant</span><span className="rose-key">Rose: tangent</span></p>
    <div className="calculus-readout" aria-live="polite">
      <p>Positions: {number(state.position)} → {number(state.nextPosition)} m</p>
      <p>Interval slope: {number(state.secant)} m/s · local velocity: {number(state.velocity)} m/s</p>
      <p>Linear change: {number(state.predictedChange)} m · exact change: {number(state.exactChange)} m</p>
      <p>Remainder: {number(state.remainder)} m</p>
    </div>
    <p>The two points determine the secant. The tangent uses the derivative of the stated cubic, not a fitted line. At t=2 the tangent crosses the curve; crossing does not disqualify a tangent. All graph values come from the analytic model on 0–4 seconds.</p>
  </Investigation>;
}
export function LimitGateLab() {
  const [kind, setKind] = useState('smooth');
  const [epsilon, setEpsilon] = useState(0.5);
  const [delta, setDelta] = useState(0.1);
  const state = limitGuarantee(kind, epsilon, delta);
  const curves = kind === 'jump' ? [{
    fn: input => input ** 2,
    domain: [1, 2]
  }, {
    fn: input => input ** 2 + 2,
    domain: [2, 3]
  }] : [{
    fn: input => input ** 2
  }];
  return <Investigation title="Make a promise about every nearby input" guidance="The target is a limit of 4 at x=2. Choose an output tolerance, then decide how narrow the input strip must be. Does changing only the value at x=2 affect this promise?" onReset={() => {
    setKind('smooth');
    setEpsilon(0.5);
    setDelta(0.1);
  }} resetLabel="Reset limit guarantee">
    <div className="calculus-controls">
      <Select label="Function near two" value={kind} onChange={setKind} options={[['smooth', 'x² everywhere'], ['hole', 'x² nearby; f(2)=6'], ['jump', 'x² left; x²+2 right']]} />
      <Range label="Output tolerance epsilon" value={epsilon} onChange={setEpsilon} min={0.01} max={1} step={0.01} />
      <Range label="Input radius delta" value={delta} onChange={setDelta} min={0.001} max={0.5} step={0.001} />
    </div>
    <CalculusPlot label="Output tolerance band and punctured input strip for a limit at two" xRange={[1, 3]} yRange={[0, 8]} xLabel="Input x" yLabel="Output f(x)" curves={curves}>
      {({
        x,
        y,
        left,
        right,
        top,
        bottom
      }) => <>
        <rect className="output-band" x={left} y={y(4 + epsilon)} width={right - left} height={y(4 - epsilon) - y(4 + epsilon)} />
        <rect className="input-band" x={x(2 - delta)} y={top} width={x(2 + delta) - x(2 - delta)} height={bottom - top} />
        <line className="blue dashed" x1={x(2 - delta)} x2={x(2 - delta)} y1={top} y2={bottom} />
        <line className="blue dashed" x1={x(2 + delta)} x2={x(2 + delta)} y1={top} y2={bottom} />
        <circle className="open-point" cx={x(2)} cy={y(4)} r="6" />
        <circle className="point" cx={x(2)} cy={y(state.centerValue)} r="4" />
        {kind === 'jump' && <circle className="open-point" cx={x(2)} cy={y(6)} r="6" />}
        {state.witness && <circle className="rose-point" cx={x(state.witness.input)} cy={y(state.witness.output)} r="5" />}
      </>}
    </CalculusPlot>
    <p className="calculus-legend"><span>Gold band: allowed output</span><span className="blue-key">Blue strip: nearby inputs</span><span className="rose-key">Rose dot: counterexample</span></p>
    <div className="calculus-readout" aria-live="polite">
      <p><strong>{state.guaranteed ? 'This input radius guarantees the requested accuracy.' : 'This input radius fails the requested accuracy.'}</strong></p>
      <p>Supremum of nearby output error: {number(state.supremum)} · requested: {number(epsilon)}</p>
      {state.witness && <p>Interior witness: x={state.witness.resolvedInFloat ? number(state.witness.input, 7) : state.witness.inputExact}. Its exact error is {state.witness.errorExact} ≥ epsilon. {state.witness.resolvedInFloat ? '' : 'Its plotted position is rounded; the displayed fraction is strictly inside the strip.'}</p>}
      <p>Left limit: {state.leftLimit} · right limit: {state.rightLimit} · f(2)={state.centerValue}</p>
    </div>
    <p>The punctured strip excludes x=2 and its boundary lines. Decimal controls are compared exactly, so an open-strip supremum equal to epsilon is valid. The guarantee comes from algebra, not from sampling the drawing. For x², delta=min(1,epsilon/5) is one sufficient choice. The right jump fails for every shown epsilon, however small the positive input radius.</p>
  </Investigation>;
}
export function ChainLocalChangeLab() {
  const [input, setInput] = useState(0);
  const [increment, setIncrement] = useState(0.1);
  const state = compositionChange(input, increment);
  return <Investigation title="Follow a change through two functions" guidance="The inner function triples an input change. The outer function squares the intermediate value. Explore how their local sensitivities combine, then compare the prediction with the actual finite change." onReset={() => {
    setInput(0);
    setIncrement(0.1);
  }} resetLabel="Reset composition">
    <div className="calculus-controls">
      <Range label="Input x" value={input} onChange={setInput} min={-1} max={1} step="any" />
      <Range label="Input change h" value={increment} onChange={setIncrement} min={-0.5} max={0.5} step={0.01} />
      <button type="button" onClick={() => setInput(-1 / 3)}>Make the outer slope zero</button>
    </div>
    <div className="calculus-chain" aria-label="Composition values and changing quantities">
      <div><strong>Input x</strong><span>{number(input)} → {number(input + increment)}</span><small>Change h={number(increment)}</small></div>
      <p className="chain-arrow">↓ multiply by 3, then add 1</p>
      <div><strong>Intermediate u</strong><span>{number(state.intermediate)} → {number(state.intermediate + state.intermediateChange)}</span><small>Change 3h={number(state.intermediateChange)}</small></div>
      <p className="chain-arrow">↓ square; local slope 2u={number(state.outerDerivative)}</p>
      <div><strong>Output u²</strong><span>{number(state.output)} → {number(state.nextOutput)}</span><small>Actual change {number(state.exactChange)}</small></div>
    </div>
    <div className="calculus-readout" aria-live="polite">
      <p>Derivative: outer slope × inner slope = {number(state.outerDerivative)} × 3 = {number(state.derivative)}</p>
      <p>First-order change: {number(state.predictedChange)} · remainder 9h²: {number(state.remainder)}</p>
      <p>Actual change = first-order change + remainder.</p>
    </div>
    <p>At u=0 the output's first-order change vanishes, but a nonzero h still changes the square by 9h². The chain rule predicts a local linear contribution; it does not remove higher-order terms. The diagram uses values and labelled operations, not distances proportional to their magnitudes.</p>
  </Investigation>;
}
export function ExtremaCandidatesLab() {
  const [kind, setKind] = useState('motion');
  const [bounds, setBounds] = useState({
    left: 0,
    right: 4
  });
  const [draftLeft, setDraftLeft] = useState('0');
  const [draftRight, setDraftRight] = useState('4');
  const [error, setError] = useState('');
  const state = extremaCandidates(kind, bounds.left, bounds.right);
  const applyBounds = () => {
    try {
      if (!draftLeft.trim() || !draftRight.trim()) throw new Error('Enter both endpoints.');
      const left = Number(draftLeft);
      const right = Number(draftRight);
      extremaCandidates(kind, left, right);
      setBounds({
        left,
        right
      });
      setError('');
    } catch (cause) {
      setError(`${cause.message} The last valid interval is retained.`);
    }
  };
  const reset = () => {
    setKind('motion');
    setBounds({
      left: 0,
      right: 4
    });
    setDraftLeft('0');
    setDraftRight('4');
    setError('');
  };
  return <Investigation title="Compare every candidate inside the allowed interval" guidance="The derivative can find interior candidates, but the domain decides which points are allowed. Observe what happens when both turning points lie outside the interval. Then compare a stationary inflection with a cusp." onReset={reset} resetLabel="Reset extrema">
    <div className="calculus-controls">
      <Select label="Extrema function" value={kind} onChange={setKind} options={[['motion', 't(t−3)²'], ['inflection', '(t−2)³'], ['cusp', '|t−2|']]} />
      <label className="calculus-field"><span>Left endpoint</span><input aria-label="Left endpoint" value={draftLeft} onChange={event => setDraftLeft(event.target.value)} inputMode="decimal" /></label>
      <label className="calculus-field"><span>Right endpoint</span><input aria-label="Right endpoint" value={draftRight} onChange={event => setDraftRight(event.target.value)} inputMode="decimal" /></label>
      <button type="button" onClick={applyBounds}>Apply interval</button>
    </div>
    {error && <p role="alert">{error}</p>}
    <CalculusPlot label="Analytic function and all endpoint or interior critical candidates" xRange={[0, 4]} yRange={kind === 'inflection' ? [-9, 9] : [-1, 5]} xLabel="Input t" yLabel="Function value" curves={[{
      fn: input => extremaValue(kind, input)
    }]}>
      {({
        x,
        y,
        top,
        bottom
      }) => <>
        <rect className="input-band" x={x(bounds.left)} y={top} width={x(bounds.right) - x(bounds.left)} height={bottom - top} />
        {state.candidates.map(candidate => <circle key={candidate.input} className={candidate.isMinimum ? 'point blue-point' : candidate.isMaximum ? 'rose-point' : 'point'} cx={x(candidate.input)} cy={y(candidate.value)} r="5" />)}
      </>}
    </CalculusPlot>
    <p>Active interval: [{bounds.left}, {bounds.right}]. Inputs may be any finite decimals from 0 to 4 with left &lt; right. Blue candidates attain the minimum; rose candidates attain the maximum. Classifications use exact decimal values of the accepted inputs. Heights are rounded for display; nearby points can overlap in the drawing.</p>
    <div className="calculus-candidates" aria-live="polite">
      {state.candidates.map(candidate => <div key={candidate.input}>
        <strong>t={candidate.input} → approximately {number(candidate.value)}</strong>
        <span>{candidate.reasons.join('; ')}</span>
        <span>{candidate.isMinimum ? 'Global minimum on this interval. ' : ''}{candidate.isMaximum ? 'Global maximum on this interval.' : ''}</span>
      </div>)}
    </div>
    <p className="calculus-signs">Between candidates: {state.intervals.map(interval => `(${interval.start}, ${interval.end}): ${interval.sign > 0 ? 'increasing' : 'decreasing'}`).join(' · ')}</p>
    <p>The sign statements refer to each open interval between listed boundaries. They come from the factored derivative, not a grid search. A derivative of zero at the middle of (t−2)³ does not give an extremum. The cusp's minimum has no finite derivative there.</p>
  </Investigation>;
}
export function SignedAccumulationLab() {
  const [upper, setUpper] = useState(4);
  const [panelCount, setPanelCount] = useState(8);
  const [method, setMethod] = useState('midpoint');
  const state = motionAccumulation(upper, panelCount, method);
  return <Investigation title="Add forward and backward contributions" guidance="From t=1 to t=3 the velocity is negative. Inspect whether those strips increase the displacement, the distance, both or neither. Move the endpoint before refining the rectangles." onReset={() => {
    setUpper(4);
    setPanelCount(8);
    setMethod('midpoint');
  }} resetLabel="Reset accumulation">
    <div className="calculus-controls">
      <Range label="Upper time T" value={upper} onChange={setUpper} min={0.25} max={4} step={0.25} suffix=" s" />
      <Select label="Rectangle count" value={panelCount} onChange={value => setPanelCount(Number(value))} options={[3, 4, 8, 16, 32, 64].map(value => [value, String(value)])} />
      <Select label="Sample in each strip" value={method} onChange={setMethod} options={[['left', 'Left endpoint'], ['midpoint', 'Midpoint'], ['right', 'Right endpoint']]} />
    </div>
    <div className="calculus-two-plots">
      <CalculusPlot label="Signed velocity rectangles from zero to the chosen upper time" xRange={[0, 4]} yRange={[-4, 10]} xLabel="Time (s)" yLabel="Velocity (m/s)" curves={[{
        fn: motionVelocity
      }]}>
        {({
          x,
          y
        }) => <>
          {state.panels.map((panel, index) => <rect key={index} className={panel.height >= 0 ? 'positive-panel' : 'negative-panel'} x={x(panel.start)} y={Math.min(y(0), y(panel.height))} width={x(panel.end) - x(panel.start)} height={Math.abs(y(panel.height) - y(0))} />)}
          <circle className="point" cx={x(upper)} cy={y(state.endpointVelocity)} r="5" />
        </>}
      </CalculusPlot>
      <CalculusPlot label="Exact accumulated displacement; marker at the chosen upper endpoint" xRange={[0, 4]} yRange={[-1, 5]} xLabel="Upper time T (s)" yLabel="Accumulation A(T) (m)" curves={[{
        fn: motionPosition,
        className: 'blue'
      }]}>
        {({
          x,
          y
        }) => <circle className="point blue-point" cx={x(upper)} cy={y(state.exactDisplacement)} r="5" />}
      </CalculusPlot>
    </div>
    <p className="calculus-legend"><span>Above zero: positive contribution</span><span className="blue-key">Below zero: negative contribution</span></p>
    <div className="calculus-readout" aria-live="polite">
      <p>Each strip: {number(state.width)} s × sampled velocity.</p>
      <p>Signed sum: {number(state.signedEstimate)} m · exact displacement: {number(state.exactDisplacement)} m</p>
      <p>Absolute-strip sum: {number(state.distanceEstimate)} m · exact distance: {number(state.exactDistance)} m</p>
      <p>Signed-sum error: {number(state.signedError)} m</p>
      <p>Rate of the exact accumulation at T: {number(state.endpointVelocity)} m/s</p>
    </div>
    <p>The second curve is the analytic accumulation, not an interpolation of the rectangle totals. Negative velocity makes that curve decrease while travelled distance still increases. Some equal grids happen to give the exact distance for this cubic; three panels or a changed endpoint exposes why this coincidence is not a general integration guarantee.</p>
  </Investigation>;
}
export function ExponentialRateLab() {
  const [rate, setRate] = useState(0.2);
  const [period, setPeriod] = useState(1);
  const state = exponentialRate(rate, period);
  const upperAmount = 10 * Math.exp(Math.max(rate * 4, 0));
  return <Investigation title="Separate a local relative rate from a period's gain" guidance="At k=0.2 per time unit, will the amount gain exactly 20% over one whole unit? Shorten the period, then try a negative rate and compare the actual curve with its tangent." onReset={() => {
    setRate(0.2);
    setPeriod(1);
  }} resetLabel="Reset exponential rate">
    <div className="calculus-controls">
      <Range label="Continuous rate k" value={rate} onChange={setRate} min={-0.8} max={0.8} step={0.05} suffix=" /time" />
      <Range label="Period length" value={period} onChange={setPeriod} min={0.25} max={2} step={0.25} suffix=" time units" />
    </div>
    <CalculusPlot label="Exponential amount and local tangent at time two, compared over a finite period" xRange={[0, 4]} yRange={[0, upperAmount * 1.1]} xLabel="Time" yLabel="Amount" curves={[{
      fn: time => 10 * Math.exp(rate * time)
    }, {
      fn: time => state.currentAmount + state.instantaneousRate * (time - 2),
      className: 'blue dashed'
    }]}>
      {({
        x,
        y
      }) => <>
        <circle className="point" cx={x(2)} cy={y(state.currentAmount)} r="5" />
        <circle className="point" cx={x(2 + period)} cy={y(state.futureAmount)} r="5" />
        <line className="rose" x1={x(2 + period)} x2={x(2 + period)} y1={y(state.futureAmount)} y2={y(state.tangentPrediction)} />
      </>}
    </CalculusPlot>
    <p className="calculus-legend"><span>Gold: 10 exp(kt)</span><span className="blue-key">Dashed blue: local prediction at t=2</span></p>
    <div className="calculus-readout" aria-live="polite">
      <p>Amount at t=2: {number(state.currentAmount)} · after this period: {number(state.futureAmount)}</p>
      <p>Instantaneous rate: {number(state.instantaneousRate)} amount/time</p>
      <p>Average rate over the period: {number(state.intervalAverageRate)} amount/time</p>
      <p>Period's fraction: exp(kΔ)−1 = {number(state.perPeriodFraction)} ({number(100 * state.perPeriodFraction)}%)</p>
      <p>Fraction divided by period: {number(state.averageRelativeRate)} /time · instantaneous relative rate: {number(rate)} /time</p>
    </div>
    <p>The period's relative change uses the amount at t=2 as its denominator. It is not the average of all instantaneous relative rates along the curve; those are constantly k. This is a declared growth/decay model, not evidence that a measured system will follow it.</p>
  </Investigation>;
}
export function TaylorErrorLab() {
  const [kind, setKind] = useState('exp');
  const [degree, setDegree] = useState(3);
  const [input, setInput] = useState(0.5);
  const state = taylorApproximation(kind, degree, input);
  const xRange = kind === 'exp' ? [-2, 2] : [-0.9, 1.5];
  const changeKind = value => {
    setKind(value);
    setInput(current => Math.min(value === 'exp' ? 2 : 1.5, Math.max(value === 'exp' ? -2 : -0.9, current)));
  };
  return <Investigation title="Ask how much of the function a polynomial captures" guidance="Near zero, adding terms can improve an approximation. Inspect whether it must improve log(1+x) at x=1.5. Compare the computed difference with the analytic truncation bound, then consider the arithmetic used to evaluate them." onReset={() => {
    setKind('exp');
    setDegree(3);
    setInput(0.5);
  }} resetLabel="Reset Taylor approximation">
    <div className="calculus-controls">
      <Select label="Approximated function" value={kind} onChange={changeKind} options={[['exp', 'exp(x), around zero'], ['log', 'log(1+x), around zero']]} />
      <Range label="Polynomial degree" value={degree} onChange={setDegree} min={1} max={12} step={1} />
      <Range label="Evaluation input" value={input} onChange={setInput} min={xRange[0]} max={xRange[1]} step={0.05} />
    </div>
    <CalculusPlot label="Function and its finite Taylor polynomial evaluated in floating-point arithmetic" xRange={xRange} yRange={kind === 'exp' ? [-1, 8] : [-4, 2]} xLabel="Input x" yLabel="Function / polynomial" curves={[{
      fn: value => taylorApproximation(kind, degree, value).exactValue
    }, {
      fn: value => taylorApproximation(kind, degree, value).polynomial,
      className: 'blue dashed'
    }]}>
      {({
        x,
        y
      }) => <>
        <line className="rose" x1={x(input)} x2={x(input)} y1={y(state.exactValue)} y2={y(state.polynomial)} />
        <circle className="point" cx={x(input)} cy={y(state.exactValue)} r="5" />
        <circle className="point blue-point" cx={x(input)} cy={y(state.polynomial)} r="4" />
      </>}
    </CalculusPlot>
    <p className="calculus-legend"><span>Gold: function</span><span className="blue-key">Dashed blue: finite polynomial</span><span className="rose-key">Rose: computed difference at selected x</span></p>
    <div className="calculus-readout" aria-live="polite">
      <p>Function: {number(state.exactValue, 7)} · polynomial: {number(state.polynomial, 7)}</p>
      <p>Computed polynomial minus function: {number(state.signedError, 7)}</p>
      <p>Analytic truncation bound, evaluated approximately: {number(state.remainderBound, 7)}</p>
      {kind === 'log' && <p>{state.insideOpenLogSeriesInterval ? 'Inside |x|<1, where the geometric-remainder argument establishes convergence.' : 'Outside the open interval |x|<1. Existence of log(1+x) does not make this power series converge here; endpoints need separate assessment.'}</p>}
    </div>
    <p>The theorem bounds the exact polynomial's truncation error. The computed difference also includes floating-point roundoff: it can exceed a tiny analytic bound or round to zero while the true error remains nonzero. Try exp(x), degree 12 and x=0.05. No rounding allowance has been added to the mathematical bound, and its displayed decimal is not a rigorously rounded interval bound.</p>
    <p>{kind === 'exp' ? 'Pₙ(x)=Σ from j=0 to n of xʲ/j!. The derivative bound uses exp(max(0,x)).' : 'Pₙ(x)=Σ from j=1 to n of (−1)ʲ⁺¹xʲ/j. The derivative bound uses n! / min(1,1+x)ⁿ⁺¹.'} These are analytic polynomial evaluations. The plot clips values outside its labelled vertical range; the numerical result still reports them. A large valid bound can be uninformative without the approximation itself being equally bad.</p>
  </Investigation>;
}
export function ImproperIntegralLab() {
  const [kind, setKind] = useState('endpoint');
  const [power, setPower] = useState(0.5);
  const [cutoffExponent, setCutoffExponent] = useState(2);
  const state = improperPowerIntegral(kind, power, cutoffExponent);
  const end = improperPowerIntegral(kind, power, 4);
  const yMaximum = Math.max(end.truncated, end.total || 0, 1) * 1.12;
  return <Investigation title="Measure what a cutoff leaves out" guidance="An unbounded height near zero can enclose finite area. A curve that tends to zero at infinity can still enclose infinite area. Change which endpoint is difficult, then test the same power on both sides." onReset={() => {
    setKind('endpoint');
    setPower(0.5);
    setCutoffExponent(2);
  }} resetLabel="Reset improper integral">
    <div className="calculus-controls">
      <Select label="Difficult endpoint" value={kind} onChange={setKind} options={[['endpoint', 'From delta to 1; delta → 0'], ['tail', 'From 1 to B; B → infinity']]} />
      <Select label="Power p in x to minus p" value={power} onChange={value => setPower(Number(value))} options={[0.5, 1, 1.5, 2].map(value => [value, `p=${value}`])} />
      <Range label="Cutoff exponent q" value={cutoffExponent} onChange={setCutoffExponent} min={0} max={4} step={0.25} />
    </div>
    <CalculusPlot label="Calculated finite integral as the decimal cutoff is extended; a finite total is shown only when proved to exist" xRange={[0, 4]} yRange={[0, yMaximum]} xLabel="Cutoff exponent q" yLabel="Truncated integral" curves={[{
      fn: value => improperPowerIntegral(kind, power, value).truncated
    }, ...(state.converges ? [{
      fn: () => state.total,
      className: 'blue dashed'
    }] : [])]}>
      {({
        x,
        y
      }) => <>
        <circle className="point" cx={x(cutoffExponent)} cy={y(state.truncated)} r="5" />
        {state.converges && <line className="rose" x1={x(cutoffExponent)} x2={x(cutoffExponent)} y1={y(state.truncated)} y2={y(state.total)} />}
      </>}
    </CalculusPlot>
    <p>Horizontal q means {kind === 'tail' ? 'B=10ᑫ' : 'delta=10⁻ᑫ'}. This is a plot of the finite integral against cutoff, not a plot of density or integrand height. Its vertical scale adapts to the selected power.</p>
    <div className="calculus-readout" aria-live="polite">
      <p>Current {kind === 'tail' ? 'upper bound B' : 'lower bound delta'}: {number(state.cutoff, 7)}</p>
      <p>Retained integral: {number(state.truncated, 7)}</p>
      <p>{state.converges ? `Finite total: ${number(state.total)} · omitted contribution: ${number(state.missing, 7)}` : 'The improper integral diverges; a finite cutoff is still finite.'}</p>
      <p>For this endpoint, convergence requires {kind === 'tail' ? 'p > 1' : 'p < 1'}. At p=1 the logarithm diverges.</p>
    </div>
    <p>The finite total and omitted contribution follow from the antiderivative and its limit. A curve that looks settled over four cutoff decades is not, by itself, proof of convergence. No infinite endpoint is replaced with a finite plotted height.</p>
  </Investigation>;
}
export function MotionJourneyFigure() {
  return <figure className="calculus-figure">
    <svg viewBox="0 0 320 220" role="img" aria-label="The object moves from position zero to four, back to zero, then to four again">
      <text x="20" y="20">Position 0 m</text><text x="210" y="20">4 m</text>
      <path className="gold" d="M35 55H280l-10-6m10 6-10 6" />
      <text x="38" y="82">t=0 → 1 s: +4 m</text>
      <path className="blue" d="M280 110H35l10-6m-10 6 10 6" />
      <text x="38" y="137">t=1 → 3 s: −4 m</text>
      <path className="gold" d="M35 165H280l-10-6m10 6-10 6" />
      <text x="38" y="192">t=3 → 4 s: +4 m</text>
    </svg>
    <figcaption>Each row is one leg of the same journey on the same position scale. The vertical separation only organizes the legs; it is not another direction of motion. Signed changes sum to 4 m, while the three travelled lengths sum to 12 m.</figcaption>
  </figure>;
}
export function ProductIncrementFigure() {
  return <figure className="calculus-figure">
    <svg viewBox="0 0 320 275" role="img" aria-label="An expanded rectangle separates the two edge strips and the second-order corner">
      <rect className="gold-fill" x="40" y="60" width="120" height="120" />
      <rect className="blue-fill" x="160" y="60" width="40" height="120" />
      <rect className="blue-fill" x="40" y="30" width="120" height="30" />
      <rect className="rose-fill" x="160" y="30" width="40" height="30" />
      <text x="100" y="125" textAnchor="middle">ab</text>
      <text x="100" y="50" textAnchor="middle">aΔb</text>
      <text x="180" y="125" textAnchor="middle" transform="rotate(-90 180 125)">bΔa</text>
      <path className="rose" d="M200 45H225V22" />
      <text x="225" y="17" textAnchor="middle">ΔaΔb</text>
      <text x="100" y="203" textAnchor="middle">a</text><text x="180" y="203" textAnchor="middle">Δa</text>
      <text x="20" y="125" textAnchor="middle">b</text>
      <text x="40" y="242">Two edge strips + one corner</text>
    </svg>
    <figcaption>This structural area diagram uses positive increments; its lengths are schematic. The exact algebra is Δ(ab)=bΔa+aΔb+ΔaΔb. When both increments are proportional to a small h, the corner is proportional to h². Signed increments use the same algebra without requiring literal added rectangles.</figcaption>
  </figure>;
}
export function FundamentalStripFigure() {
  return <figure className="calculus-figure">
    <svg viewBox="0 0 320 250" role="img" aria-label="A small added integral under one plus t is compared with a rectangle and its triangular error">
      <path className="axis" d="M35 20V190H285" />
      <path className="gold-fill" d="M55 190V130L175 70V190Z" />
      <path className="blue-fill" d="M175 190V70H235V190Z" />
      <path className="rose-fill" d="M175 70H235V40Z" />
      <path className="gold" d="M55 130L255 30" />
      <text x="55" y="212" textAnchor="middle">0</text><text x="175" y="212" textAnchor="middle">1</text><text x="235" y="212" textAnchor="middle">1.5</text>
      <text x="30" y="74" textAnchor="end">2</text><text x="30" y="44" textAnchor="end">2.5</text>
      <text x="75" y="154">A(1)</text><text x="203" y="154" textAnchor="middle">1</text>
      <text x="240" y="20" textAnchor="middle">error 1/8</text>
      <text x="100" y="243">t (chosen units)</text>
    </svg>
    <figcaption>For f(t)=1+t and h=1/2, the added area is 1+1/8=9/8. The rectangle f(1)h is 1; its triangular error is h²/2. Divide by h: the added-area rate is 2+h/2, which tends to f(1)=2. The proof in the text extends the residual bound beyond this linear example.</figcaption>
  </figure>;
}
export function WeightedRodFigure() {
  return <figure className="calculus-figure">
    <svg viewBox="0 0 320 220" role="img" aria-label="A rod's increasing density shifts its center of mass to the right of the geometric midpoint">
      <path className="axis" d="M30 25V150H290" />
      <path className="gold-fill" d="M45 150V95L275 40V150Z" />
      <path className="blue dashed" d="M160 150V62" />
      <path className="rose" d="M172.7777777778 150V64.4444444444" />
      <text x="20" y="100" textAnchor="end">2</text><text x="20" y="45" textAnchor="end">4</text>
      <text x="45" y="173" textAnchor="middle">0</text><text x="275" y="173" textAnchor="middle">2 m</text>
      <text x="52" y="25">Density ρ(x)=2+x kg/m</text>
      <text x="35" y="205">Midpoint 1 · center 10/9 m</text>
    </svg>
    <figcaption>The area under density is mass: 6 kg. Weight each strip again by its position to get first moment 20/3 kg·m. Their ratio is the center 10/9 m. The dashed midpoint and rose center use the same position scale; the denser right half moves the balance point rightwards.</figcaption>
  </figure>;
}
