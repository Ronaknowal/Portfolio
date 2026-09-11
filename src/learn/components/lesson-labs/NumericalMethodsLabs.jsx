import { useId, useState } from 'react';
import { Investigation, Stepper } from './LessonInvestigation.jsx';
import { adaptiveSimpson, bracketTrace, compositeQuadrature, derivativeEstimate, formatNumerical as format, integralProblems, newtonTrace, pumpCalibration, pumpRate, pumpVolume, rootProblems } from '../../data/numerical-methods-models.js';
import './numerical-methods-labs.css';
function Select({
  label,
  value,
  onChange,
  options
}) {
  return <label className="nm-control"><span>{label}</span><select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>{options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}</select></label>;
}
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 1
}) {
  return <label className="nm-control"><span>{label}: <strong>{format(value)}</strong></span><input aria-label={label} type="range" value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Plot({
  label,
  domain = [0, 1],
  range = [0, 1],
  xLabel = 'x',
  yLabel = 'f(x)',
  curves = [],
  children
}) {
  const clipId = useId();
  const x = value => 42 + 238 * (value - domain[0]) / (domain[1] - domain[0]);
  const y = value => 166 - 132 * (value - range[0]) / (range[1] - range[0]);
  return <svg className="nm-plot" viewBox="0 0 300 220" role="img" aria-label={label}>
    <defs><clipPath id={clipId}><rect x="42" y="34" width="238" height="132" /></clipPath></defs>
    <path d="M42,29 V166 H282" stroke="var(--nm-line)" fill="none" />
    <text x="42" y="18">{yLabel}</text><text x="160" y="210" textAnchor="middle">{xLabel}</text>
    {[0, 1].map(position => <g key={position}><text x={position ? 280 : 42} y="188" textAnchor={position ? 'end' : 'start'}>{format(domain[position], 2)}</text><text x="36" y={position ? 39 : 169} textAnchor="end">{format(range[position], 2)}</text></g>)}
    {range[0] < 0 && range[1] > 0 && <path d={`M42,${y(0)} H280`} stroke="var(--nm-line)" strokeDasharray="3 3" />}
    {curves.map(({
      f,
      color = 'var(--nm-gold)',
      dashed = false
    }, index) => <path key={index} d={Array.from({
      length: 241
    }, (_, point) => {
      const coordinate = domain[0] + (domain[1] - domain[0]) * point / 240;
      const value = f(coordinate);
      return `${point === 0 ? 'M' : 'L'}${x(coordinate)},${y(value)}`;
    }).join(' ')} fill="none" stroke={color} strokeWidth="2" clipPath={`url(#${clipId})`} strokeDasharray={dashed ? '4 3' : undefined} />)}
    {children && children({
      x,
      y
    })}
  </svg>;
}
function Table({
  caption,
  headings,
  rows
}) {
  return <div className="nm-table-scroll" tabIndex={0} role="region" aria-label={caption}><table><caption>{caption}</caption><thead><tr>{headings.map(heading => <th key={heading}>{heading}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{row.map((cell, column) => <td key={column}>{cell}</td>)}</tr>)}</tbody></table></div>;
}
export function PumpConnectionFigure() {
  return <figure className="nm-inline"><div className="nm-pair">
    <Plot label="Synthetic rate t exp(−t) with the first three minutes shaded" domain={[0, 4]} range={[0, .4]} xLabel="Minutes" yLabel="Litres / minute" curves={[{
        f: pumpRate
      }]}>{({
          x,
          y
        }) => <path d={`M${x(0)},${y(0)} ${Array.from({
          length: 61
        }, (_, index) => `L${x(index / 20)},${y(pumpRate(index / 20))}`).join(' ')} L${x(3)},${y(0)} Z`} fill="var(--nm-gold)" opacity=".2" />}</Plot>
    <Plot label="Accumulated volume with a 0.8 litre target and the time at three minutes" domain={[0, 4]} range={[0, 1]} xLabel="Minutes" yLabel="Litres delivered" curves={[{
        f: pumpVolume
      }]}>{({
          x,
          y
        }) => <><path d={`M42,${y(.8)} H280 M${x(3)},166 V${y(pumpVolume(3))}`} stroke="var(--nm-blue)" strokeDasharray="4 3" /><circle cx={x(3)} cy={y(pumpVolume(3))} r="4" fill="var(--nm-blue)" /></>}</Plot>
  </div><figcaption>The shaded rate-area through three minutes equals V(3) ≈ 0.80085 litres. Its right-hand curve crosses 0.8 litres slightly before three minutes. Both plots are calculated from the same synthetic model; neither is measured pump data.</figcaption></figure>;
}
export function BracketLab() {
  const [key, setKey] = useState('square');
  const [step, setStep] = useState(0);
  const problem = rootProblems[key];
  const result = bracketTrace(problem.f, problem.lower, problem.upper, {
    maximumSteps: 8,
    tolerance: .001
  });
  const state = result.steps[Math.min(step, result.steps.length - 1)];
  function change(value) {
    setKey(value);
    setStep(0);
  }
  return <Investigation id="numerical-bracket" kicker="ROOTS · RETAIN AN INTERVAL" title="What stays known after one evaluation?">
    <p>Predict which half contains a root before advancing. Continuity of these polynomial fixtures is known; opposite signs alone would not establish continuity for an arbitrary function.</p>
    <Select label="Bracket problem" value={key} onChange={change} options={Object.entries(rootProblems).map(([id, item]) => [id, item.label])} />
    <Plot label={`${problem.label}: function and current bracket`} domain={problem.domain} range={problem.range} curves={[{
      f: problem.f
    }]}>{({
        x,
        y
      }) => state && <><path d={`M${x(state.lower)},${y(0)} H${x(state.upper)}`} stroke="var(--nm-blue)" strokeWidth="6" />{[[state.lower, state.leftValue], [state.midpoint, state.value], [state.upper, state.rightValue]].map(([position, value], index) => <circle key={index} cx={x(position)} cy={y(value)} r="4" fill={index === 1 ? 'var(--nm-gold)' : 'var(--nm-blue)'} />)}</>}</Plot>
    {state ? <><div className="nm-intervals" aria-label="Successively retained intervals">{result.steps.slice(0, step + 1).map(row => <div className="nm-interval-row" key={row.iteration}><span>{row.iteration} halves</span><div><i style={{
              left: `${100 * (row.lower - problem.lower) / (problem.upper - problem.lower)}%`,
              width: `${100 * row.width / (problem.upper - problem.lower)}%`
            }} /></div></div>)}</div>
      <p className="nm-readout" aria-live="polite">Current bracket [{format(state.lower)}, {format(state.upper)}]. Midpoint {format(state.midpoint)} has f(midpoint) = {format(state.value)}. Width {format(state.width)}; mathematical midpoint error bound {format(state.radius)}.</p>
      <Stepper step={step} count={result.steps.length} setStep={setStep} onReset={() => change('square')} />
      <p>The last available state reports: {result.status}. Eight halvings here are a bounded demonstration, not a promise that every chosen tolerance is met.</p>
    </> : <><p className="nm-readout">{result.status}. Here f(1) = 0, yet f(0) and f(2) are both positive. A failed bracket test does not prove there is no root.</p><button type="button" onClick={() => change('square')}>Reset</button></>}
    <p>Transfer: change to the cubic and predict the first retained half. Its negative root is not the 0 ↔ 1 Newton cycle explored next.</p>
  </Investigation>;
}
export function NewtonLab() {
  const [key, setKey] = useState('cycle');
  const [method, setMethod] = useState('plain');
  const [step, setStep] = useState(0);
  const problem = rootProblems[key];
  const result = newtonTrace(problem, problem.start, {
    safeguarded: method === 'safe',
    maximumSteps: 8
  });
  const state = result.steps[Math.min(step, result.steps.length - 1)];
  function reset() {
    setKey('cycle');
    setMethod('plain');
    setStep(0);
  }
  return <Investigation id="numerical-newton" kicker="ROOTS · FOLLOW A TANGENT" title="A local line can point the wrong way">
    <p>The gold curve is the function. The dashed blue line is the current tangent, clipped at the plot boundary if it leaves the picture. The readout preserves the full proposed number.</p>
    <div className="nm-controls"><Select label="Newton problem" value={key} onChange={value => {
        setKey(value);
        setStep(0);
      }} options={Object.entries(rootProblems).map(([id, item]) => [id, item.label])} /><Select label="Newton method" value={method} onChange={value => {
        setMethod(value);
        setStep(0);
      }} options={[["plain", "Plain Newton"], ["safe", "Bracket safeguard"]]} /></div>
    <Plot label="Function, tangent and next Newton candidate" domain={problem.domain} range={problem.range} curves={[{
      f: problem.f
    }, ...(state ? [{
      f: x => state.value + state.slope * (x - state.current),
      color: 'var(--nm-blue)',
      dashed: true
    }] : [])]}>{({
        x,
        y
      }) => state && <><circle cx={x(state.current)} cy={y(state.value)} r="4" fill="var(--nm-gold)" />{state.next >= problem.domain[0] && state.next <= problem.domain[1] && <circle cx={x(state.next)} cy={y(0)} r="5" fill="var(--nm-blue)" />}</>}</Plot>
    {state ? <><p className="nm-readout" aria-live="polite">At x = {format(state.current)}, f = {format(state.value)} and slope = {format(state.slope)}. Tangent proposes {state.proposal === null ? 'no finite intercept' : format(state.proposal)}. Use {format(state.next)}: {state.reason}.</p><Stepper step={step} count={result.steps.length} setStep={setStep} onReset={reset} /><Table caption="Visited values so far" headings={['Step', 'Current x', 'Next x', 'Decision']} rows={result.steps.slice(0, step + 1).map(row => [row.iteration + 1, format(row.current), format(row.next), row.reason])} /></> : <><p>{result.status}. A touching root has no opposite-sign bracket in this fixture, so this safeguard cannot start.</p><button type="button" onClick={reset}>Reset</button></>}
    <p>Full trace status: {result.status}. A residual stopping condition is distinct from a guaranteed location error. Try the touching root with plain Newton: its distance from one halves rather than squares.</p>
  </Investigation>;
}
export function DifferenceLab() {
  const [exponent, setExponent] = useState(1);
  const [stencil, setStencil] = useState('central');
  const [offset, setOffset] = useState('0');
  const h = 10 ** -exponent;
  const result = derivativeEstimate(1, h, stencil, Number(offset));
  const localShape = (x, derivative) => stencil === 'second' ? Math.sin(1) + Math.cos(1) * (x - 1) + derivative * (x - 1) ** 2 / 2 : Math.sin(1) + derivative * (x - 1);
  const referenceDerivative = stencil === 'second' ? -Math.sin(1) : Math.cos(1);
  const sweep = Array.from({
    length: 16
  }, (_, index) => ({
    exponent: index + 1,
    ...derivativeEstimate(1, 10 ** -(index + 1), stencil, Number(offset))
  }));
  const logErrors = sweep.filter(row => row.error > 0 && Number.isFinite(row.error)).map(row => Math.log10(row.error));
  const logMinimum = logErrors.length ? Math.floor(Math.min(...logErrors)) - 1 : -1;
  const logMaximum = logErrors.length ? Math.ceil(Math.max(...logErrors)) + 1 : 1;
  function reset() {
    setExponent(1);
    setStencil('central');
    setOffset('0');
  }
  return <Investigation id="numerical-difference" kicker="DERIVATIVES · MOVE THE SAMPLE POINTS" title="Smaller spacing competes with lost information">
    <div className="nm-controls"><Select label="Derivative stencil" value={stencil} onChange={setStencil} options={[["central", "Central first derivative"], ["forward", "Forward first derivative"], ["boundary", "One-sided second-order"], ["second", "Central second derivative"]]} /><Select label="Function offset" value={offset} onChange={setOffset} options={[["0", "sin(x)"], ["100000000", "100,000,000 + sin(x)"]]} /><Range label="Negative power of ten" value={exponent} onChange={setExponent} min={1} max={16} /></div>
    <p>h = {format(h)}. The added constant changes evaluation rounding but not the true derivative. Geometry below shows the unshifted sine to keep its shape visible; the sample table contains the actual shifted values.</p>
    <p>Gold is sine. Dashed blue uses the analytic derivative; purple uses the computed derivative. First-derivative comparisons are lines through the center value; second-derivative comparisons are local quadratics with the exact value and first derivative held fixed. They are local models, not extra function samples.</p>
    <div className="nm-pair"><Plot label="Sine samples and analytic versus computed local derivative shapes" domain={[.75, 1.25]} range={[.6, 1.02]} curves={[{
        f: Math.sin
      }, {
        f: x => localShape(x, referenceDerivative),
        color: 'var(--nm-blue)',
        dashed: true
      }, ...(Number.isFinite(result.estimate) ? [{
        f: x => localShape(x, result.estimate),
        color: '#bc9bf1'
      }] : [])]}>{({
          x,
          y
        }) => result.nodes.filter(node => node.x >= .75 && node.x <= 1.25).map((node, index) => <circle key={index} cx={x(node.x)} cy={y(Math.sin(node.x))} r="4" fill="var(--nm-gold)" />)}</Plot>
    <svg className="nm-plot" viewBox="0 0 300 220" role="img" aria-label="Calculated absolute derivative error versus negative power of ten; vertical axis log base 10 error"><path d="M42,30 V166 H280" stroke="var(--nm-line)" fill="none" /><text x="42" y="18">log₁₀ error</text><text x="162" y="210" textAnchor="middle">h = 10⁻ᵏ; k →</text><text x="42" y="186">1</text><text x="280" y="186" textAnchor="end">16</text><text x="36" y="38" textAnchor="end">{logMaximum}</text><text x="36" y="166" textAnchor="end">{logMinimum}</text>{sweep.map(row => row.error !== null && Number.isFinite(row.error) && <circle key={row.exponent} cx={42 + (row.exponent - 1) * 238 / 15} cy={row.error === 0 ? 166 : 166 - 132 * (Math.log10(row.error) - logMinimum) / (logMaximum - logMinimum)} r={row.exponent === exponent ? 5 : 3} fill={row.error === 0 ? 'none' : 'var(--nm-gold)'} stroke="var(--nm-gold)" />)}</svg></div>
    <p className="nm-readout" aria-live="polite">{result.status}. Estimate {result.estimate === null ? 'unavailable' : format(result.estimate)}, reference {format(result.reference ?? (stencil === 'second' ? -Math.sin(1) : Math.cos(1)))}; absolute error {result.error === null ? 'unavailable' : format(result.error)}.</p>
    <Table caption="Actual stencil samples" headings={['x coordinate', 'Value', 'Weight']} rows={result.nodes.map(node => [node.x.toPrecision(17), node.value.toPrecision(17), node.weight])} />
    <p>Plot points are actual JavaScript binary64 calculations. Open points on the bottom boundary mean computed zero error. The vertical range covers this entire stencil/offset sweep and stays fixed when only h changes. Missing points mean coincident samples or nonfinite arithmetic. This is not an exact high-precision derivative oracle.</p><button type="button" onClick={reset}>Reset</button>
  </Investigation>;
}
export function QuadratureLab() {
  const [key, setKey] = useState('polynomial');
  const [panels, setPanels] = useState('4');
  const [method, setMethod] = useState('trapezoid');
  const problem = integralProblems[key];
  const result = compositeQuadrature(problem.f, 0, 1, Number(panels), method);
  function interpolant(x) {
    if (method === 'trapezoid') {
      const index = Math.min(result.panels - 1, Math.floor(x / result.h));
      const left = result.nodes[index],
        right = result.nodes[index + 1];
      return left.value + (right.value - left.value) * (x - left.x) / result.h;
    }
    const index = Math.min(result.panels - 2, 2 * Math.floor(x / (2 * result.h)));
    const nodes = result.nodes.slice(index, index + 3);
    return nodes.reduce((sum, node, i) => sum + node.value * nodes.reduce((product, other, j) => i === j ? product : product * (x - other.x) / (node.x - other.x), 1), 0);
  }
  function reset() {
    setKey('polynomial');
    setPanels('4');
    setMethod('trapezoid');
  }
  return <Investigation id="numerical-quadrature" kicker="AREA · INTEGRATE THE LOCAL SHAPE" title="The weights come from an interpolating curve">
    <p>Gold is the actual function; blue is the line or parabola being integrated. Predict the sign of the error on x² before changing panel count.</p><div className="nm-controls"><Select label="Area function" value={key} onChange={setKey} options={['polynomial', 'sine'].map(id => [id, integralProblems[id].label])} /><Select label="Area method" value={method} onChange={setMethod} options={[["trapezoid", "Trapezoids"], ["simpson", "Simpson parabolas"]]} /><Select label="Number of panels" value={panels} onChange={setPanels} options={['2', '4', '8', '16'].map(value => [value, value])} /></div>
    <Plot label="Actual curve and numerical interpolant with sampled nodes" range={[0, 1.1]} curves={[{
      f: problem.f
    }, {
      f: interpolant,
      color: 'var(--nm-blue)',
      dashed: true
    }]}>{({
        x,
        y
      }) => <><path d={`M${x(0)},${y(0)} ${Array.from({
          length: 129
        }, (_, index) => `L${x(index / 128)},${y(interpolant(index / 128))}`).join(' ')} L${x(1)},${y(0)} Z`} fill="var(--nm-blue)" opacity=".13" />{result.nodes.map((node, index) => <g key={index}><path d={`M${x(node.x)},166 V${y(node.value)}`} stroke="var(--nm-blue)" opacity=".4" /><circle cx={x(node.x)} cy={y(node.value)} r="3" fill="var(--nm-blue)" /></g>)}</>}</Plot>
    <p className="nm-readout" aria-live="polite">Area {format(result.estimate)}, analytic reference {format(problem.integral)}; signed error {format(result.estimate - problem.integral)}. {result.nodes.length} evaluations. Multiply the weighted sum by {format(result.h / (method === 'simpson' ? 3 : 1))}.</p>
    <Table caption="Sample weights before the common scale factor" headings={['x', 'f(x)', 'Weight']} rows={result.nodes.map(node => [format(node.x), format(node.value), node.weight])} /><button type="button" onClick={reset}>Reset</button>
    <p>Transfer: keep four panels and switch from x² to sin(πx). Exactness on a quadratic did not establish exactness on all smooth functions.</p>
  </Investigation>;
}
export function AdaptiveLab() {
  const [key, setKey] = useState('peak');
  const [depth, setDepth] = useState(1);
  const [tolerance, setTolerance] = useState('0.000001');
  const problem = integralProblems[key];
  const verticalScale = key === 'blind' ? 1e5 : 1;
  const result = adaptiveSimpson(problem.f, 0, 1, {
    tolerance: Number(tolerance),
    maximumDepth: depth
  });
  function reset() {
    setKey('peak');
    setDepth(1);
    setTolerance('0.000001');
  }
  return <Investigation id="numerical-adaptive" kicker="AREA · ALLOCATE EVALUATIONS" title="Refine a partition, then challenge its evidence">
    <p>Allowing another depth reruns the same deterministic algorithm with a larger work limit. Each split halves that interval's absolute budget. Gold is the function; blue marks only actual integration samples.</p><div className="nm-controls"><Select label="Adaptive function" value={key} onChange={value => {
        setKey(value);
        setDepth(1);
      }} options={['peak', 'blind'].map(id => [id, integralProblems[id].label])} /><Select label="Requested area tolerance" value={tolerance} onChange={setTolerance} options={[["0.0001", "10⁻⁴"], ["0.000001", "10⁻⁶"], ["0.00000001", "10⁻⁸"]]} /></div>
    <Plot label="Function and actual adaptive sample locations" yLabel={key === 'blind' ? 'f(x) × 10⁵' : 'f(x)'} range={[0, problem.maximum * 1.1 * verticalScale]} curves={[{
      f: x => problem.f(x) * verticalScale
    }]}>{({
        x,
        y
      }) => result.samples.map((sample, index) => <circle key={index} cx={x(sample.x)} cy={y(sample.value * verticalScale)} r="2.5" fill="var(--nm-blue)" />)}</Plot>
    {key === 'blind' && <p>The vertical coordinates are multiplied by 10⁵ for readability. The area and error readouts below retain their original scale.</p>}
    <div className="nm-partition" aria-label={`${result.leaves.length} final intervals covering zero to one`}>{result.leaves.map((leaf, index) => <span key={index} style={{
        width: `${100 * (leaf.upper - leaf.lower)}%`
      }} className={leaf.accepted ? 'accepted' : 'unresolved'} title={`[${leaf.lower}, ${leaf.upper}], budget ${leaf.budget}`} />)}</div>
    <p className="nm-readout" aria-live="polite">Allowed depth {depth}; {result.samples.length} evaluations. {result.status}. Area {format(result.estimate)}; estimated error {format(result.estimatedError)}; actual error against the analytic fixture {format(Math.abs(result.estimate - problem.integral))}.</p>
    <div className="nm-buttons"><button type="button" disabled={depth === 1} onClick={() => setDepth(depth - 1)}>One less level</button><button type="button" disabled={depth === 8} onClick={() => setDepth(depth + 1)}>Allow another level</button><button type="button" onClick={reset}>Reset</button></div>
    <p>Blue partition segments meet their local estimator test; outlined gold segments hit the depth limit. This color is not a certificate of true error. The blind polynomial remains invisible at its five initial nodes even when you allow greater depth.</p>
    <details><summary>Inspect all terminal intervals</summary><Table caption="Local adaptive budgets and estimated errors" headings={['Interval', 'Budget', 'Estimate', 'Test']} rows={result.leaves.map(leaf => [`${format(leaf.lower)}–${format(leaf.upper)}`, format(leaf.budget), format(leaf.estimatedError), leaf.accepted ? 'accepted estimate' : 'depth limit'])} /></details>
  </Investigation>;
}
export function CalibrationLab() {
  const [target, setTarget] = useState('.8');
  const [panels, setPanels] = useState('8');
  const [candidate, setCandidate] = useState(3);
  const result = pumpCalibration(Number(target), Number(panels), candidate);
  function reset() {
    setTarget('.8');
    setPanels('8');
    setCandidate(3);
  }
  return <Investigation id="numerical-calibration" kicker="COMPOSITION · VOLUME ERROR BECOMES TIME ERROR" title="Is the inner integral accurate enough to decide?">
    <div className="nm-controls"><Select label="Target litres" value={target} onChange={setTarget} options={[[".7", "0.7 L"], [".8", "0.8 L"], [".85", "0.85 L"]]} /><Select label="Integral panels" value={panels} onChange={setPanels} options={['8', '32', '128'].map(value => [value, value])} /><Range label="Candidate minutes" value={candidate} onChange={setCandidate} min={2} max={4} step={.01} /></div>
    <Plot label="Analytic volume curve and numerical candidate volume with a discretization error interval" domain={[2, 4]} range={[.3, 1.2]} xLabel="Minutes" yLabel="Litres" curves={[{
      f: pumpVolume
    }]}>{({
        x,
        y
      }) => <><path d={`M42,${y(result.target)} H280`} stroke="var(--nm-blue)" strokeDasharray="4 3" /><path d={`M${x(candidate)},${y(result.estimate - result.integralBound)} V${y(result.estimate + result.integralBound)}`} stroke="var(--nm-blue)" strokeWidth="4" /><circle cx={x(candidate)} cy={y(result.estimate)} r="4" fill="var(--nm-blue)" /></>}</Plot>
    <p className="nm-readout" aria-live="polite">{result.sign}. Estimated volume {format(result.estimate)} L ± {format(result.integralBound)} L of bounded discretization error. Residual {format(result.residual)} L. Conditional time-error bound: {format(result.timeBound)} minutes.</p>
    <p>The slope on [2, 4] is at least {format(result.slopeLowerBound)} L/min. The displayed bound uses (absolute residual + integral bound) / minimum slope. It excludes a rigorous enclosure of floating-point evaluation error and is not a confidence interval.</p><button type="button" onClick={reset}>Reset</button>
    <p>Try target 0.8 L and candidate 3 min. Increasing the panel count can resolve the sign, but moving the candidate is still necessary if the requested time accuracy is tighter than its remaining location error.</p>
  </Investigation>;
}
