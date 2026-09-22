import { useId, useState } from 'react';
import { approachState, circleMotionState, curvaturePresets, curvatureState, descentState, formatCalculusNumber as number, localChangeState } from '../../data/multivariate-calculus-models.js';
import './multivariate-calculus-labs.css';
const pair = values => `(${values.map(number).join(', ')})`;
const axisNumber = value => Math.abs(value) < 1e-11 ? '0' : String(Number(value.toPrecision(2)));
const linePath = (points, position) => points.map((point, index) => `${index ? 'L' : 'M'}${position(point).join(',')}`).join(' ');
function Investigation({
  name,
  title,
  children
}) {
  return <section className="multivariate-lab" data-lab={name} aria-label={title}>
    <p className="multivariate-eyebrow">PREDICT · CHANGE · EXPLAIN</p>
    <h3>{title}</h3>{children}
  </section>;
}
function CartesianPlot({
  label,
  domain = [-3, 3],
  curves = [],
  vectors = [],
  points = [],
  children
}) {
  const unique = useId().replaceAll(':', '');
  const scale = 252 / (domain[1] - domain[0]);
  const position = ([x, y]) => [42 + (x - domain[0]) * scale, 278 - (y - domain[0]) * scale];
  const origin = position([0, 0]);
  return <svg className="multivariate-plot" viewBox="0 0 330 320" role="img" aria-label={label}>
    <title>{label}</title>
    <defs>
      <marker id={`${unique}-arrow`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="context-stroke" /></marker>
      <clipPath id={`${unique}-clip`}><rect x="40" y="24" width="256" height="256" /></clipPath>
    </defs>
    <rect x="42" y="26" width="252" height="252" className="mv-frame" />
    <path d={`M42,${origin[1]} H294 M${origin[0]},26 V278`} className="mv-axis" />
    <text x="301" y={Math.max(30, Math.min(278, origin[1] - 7))}>x</text>
    <text x={Math.max(46, Math.min(275, origin[0] + 9))} y="19">y</text>
    <text x="42" y="302" textAnchor="middle">{number(domain[0])}</text><text x="294" y="302" textAnchor="middle">{number(domain[1])}</text>
    <g clipPath={`url(#${unique}-clip)`}>
      {curves.map((curve, index) => <path key={index} d={linePath(curve.points, position)} className={curve.className || 'mv-contour'} />)}
      {vectors.map((vector, index) => {
        const from = position(vector.from || [0, 0]);
        const to = position(vector.to);
        return <line key={index} x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]} className={vector.className || 'mv-actual'} markerEnd={`url(#${unique}-arrow)`} />;
      })}
      {points.map((point, index) => {
        const placed = position(point.point);
        const outgoing = vectors.filter(vector => (vector.from || [0, 0]).every((value, axis) => value === point.point[axis]))
          .map(vector => {
            const end = position(vector.to);
            const delta = end.map((value, axis) => value - placed[axis]);
            const length = Math.hypot(...delta);
            return length ? delta.map(value => value / length) : null;
          }).filter(Boolean);
        let labelOffset = [8, -10];
        if (outgoing.length) {
          // Put the name opposite the outgoing arrows. Opposed arrows leave
          // their perpendicular direction free for the label instead.
          const sum = outgoing.reduce((total, vector) => total.map((value, axis) => value + vector[axis]), [0, 0]);
          const length = Math.hypot(...sum);
          const direction = length > 1e-6 ? sum.map(value => -value / length) : [-outgoing[0][1], outgoing[0][0]];
          labelOffset = [22 * direction[0], 22 * direction[1] + 4];
        }
        return <g key={index}><circle cx={placed[0]} cy={placed[1]} r={point.radius || 5} className={point.className || 'mv-point'} />{point.label && <text x={placed[0] + labelOffset[0]} y={placed[1] + labelOffset[1]} textAnchor={outgoing.length ? 'middle' : undefined}>{point.label}</text>}</g>;
      })}
      {children}
    </g>
  </svg>;
}
function FunctionPlot({
  label,
  series,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  selected
}) {
  const unique = useId().replaceAll(':', '');
  const allValues = series.flatMap(line => line.points.map(point => point[1]));
  const minimum = yDomain ? yDomain[0] : Math.min(...allValues);
  const maximum = yDomain ? yDomain[1] : Math.max(...allValues);
  const padding = yDomain ? 0 : Math.max(0.1, (maximum - minimum) * 0.1);
  const lower = minimum - padding;
  const upper = maximum + padding;
  const position = ([x, y]) => [46 + (x - xDomain[0]) / (xDomain[1] - xDomain[0]) * 248, 254 - (y - lower) / (upper - lower) * 208];
  return <svg className="multivariate-plot" viewBox="0 0 330 300" role="img" aria-label={label}>
    <title>{label}</title>
    <defs><clipPath id={`${unique}-curve`}><rect x="44" y="43" width="252" height="214" /></clipPath></defs>
    <text x="46" y="22">{yLabel}</text>
    <path d="M46,44 V254 H296" className="mv-axis" />
    <text x="42" y="48" textAnchor="end">{axisNumber(upper)}</text><text x="42" y="256" textAnchor="end">{axisNumber(lower)}</text>
    <text x="46" y="278" textAnchor="middle">{number(xDomain[0])}</text><text x="294" y="278" textAnchor="middle">{number(xDomain[1])}</text><text x="165" y="296" textAnchor="middle">{xLabel}</text>
    <g clipPath={`url(#${unique}-curve)`}>
      {series.map((line, index) => <path key={index} d={linePath(line.points, position)} className={line.className || 'mv-actual'} />)}
      {selected?.map((point, index) => {
        const placed = position(point.point);
        return <circle key={index} cx={placed[0]} cy={placed[1]} r="5" className={point.className || 'mv-point'} />;
      })}
    </g>
  </svg>;
}
export function PartialSlicesFigure() {
  const xs = Array.from({
    length: 61
  }, (_, index) => index / 20);
  return <figure className="multivariate-inline" aria-label="Two partial derivative slices through the same point">
    <div className="multivariate-linked">
      <div><h4>Freeze y=1; move x</h4><FunctionPlot label="x slice: x squared plus 2, tangent slope 2 at x=1" xDomain={[0, 3]} xLabel="x" yLabel="f(x,1)" series={[{
          points: xs.map(x => [x, x * x + 2])
        }, {
          points: xs.map(x => [x, 3 + 2 * (x - 1)]),
          className: 'mv-linear'
        }]} selected={[{
          point: [1, 3]
        }]} /></div>
      <div><h4>Freeze x=1; move y</h4><FunctionPlot label="y slice: 1 plus 2 y squared, tangent slope 4 at y=1" xDomain={[0, 3]} xLabel="y" yLabel="f(1,y)" series={[{
          points: xs.map(y => [y, 1 + 2 * y * y])
        }, {
          points: xs.map(y => [y, 3 + 4 * (y - 1)]),
          className: 'mv-linear'
        }]} selected={[{
          point: [1, 3]
        }]} /></div>
    </div>
    <figcaption>Both marked points represent the same input (1,1) and output 3. Solid gold is the exact slice; dashed blue is its tangent line. The slopes are 2 and 4. The panels use separately labeled vertical scales; compare the numbers, not apparent screen angles.</figcaption>
  </figure>;
}
export function LocalGradientLab() {
  const [base, setBase] = useState([1, 1]);
  const [draft, setDraft] = useState(['1', '1']);
  const [angle, setAngle] = useState(0);
  const [step, setStep] = useState(0.2);
  const [error, setError] = useState('');
  const state = localChangeState(...base, angle, step);
  const curves = [1, 3, 6, 10].map(level => ({
    points: Array.from({
      length: 121
    }, (_, index) => {
      const theta = index * Math.PI / 60;
      return [Math.sqrt(level) * Math.cos(theta), Math.sqrt(level / 2) * Math.sin(theta)];
    })
  }));
  const changeBase = event => {
    event.preventDefault();
    try {
      if (draft.some(value => !value.trim())) throw new Error('Enter both base coordinates.');
      const nextBase = draft.map(Number);
      localChangeState(...nextBase, angle, step);
      setBase(nextBase);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  };
  const align = offset => {
    if (state.gradientNorm) setAngle(((Math.atan2(state.gradient[1], state.gradient[0]) * 180 / Math.PI + offset) % 360 + 360) % 360);
  };
  return <Investigation name="local-gradient" title="Which way is uphill at this point?">
    <p>Rotate the direction and watch the directional rate update. Then shrink the signed step: does the slope change, or only the finite prediction error? The function stays f=x²+2y².</p>
    <form className="multivariate-controls" onSubmit={changeBase}>
      {['x', 'y'].map((axis, index) => <label key={axis}>Base {axis}<input inputMode="decimal" value={draft[index]} onChange={event => setDraft(draft.map((value, position) => position === index ? event.target.value : value))} /></label>)}
      <button type="submit">Apply base point</button>
      <button type="button" onClick={() => {
        setBase([1, 1]);
        setDraft(['1', '1']);
        setAngle(0);
        setStep(0.2);
        setError('');
      }}>Reset</button>
    </form>
    {error && <p role="alert">{error} The active point is unchanged.</p>}
    <div className="multivariate-controls">
      <label>Direction angle: {number(angle)}°<input type="range" min="0" max="360" step="any" value={angle} onChange={event => setAngle(Number(event.target.value))} /></label>
      <label>Signed step h: {number(step)}<input type="range" min="-0.5" max="0.5" step="0.01" value={step} onChange={event => setStep(Number(event.target.value))} /></label>
      <button disabled={!state.gradientNorm} onClick={() => align(0)}>Along gradient</button><button disabled={!state.gradientNorm} onClick={() => align(90)}>Tangent direction</button>
    </div>
    <div className="multivariate-linked">
      <div><h4>Input plane: level curves</h4><CartesianPlot label="Bowl contours, base point, selected unit direction and gradient direction" curves={curves} vectors={[{
          from: base,
          to: base.map((value, index) => value + state.direction[index]),
          className: 'mv-direction'
        }, ...(state.gradientNorm ? [{
          from: base,
          to: base.map((value, index) => value + state.gradient[index] / state.gradientNorm),
          className: 'mv-actual'
        }] : [])]} points={[{
          point: base,
          label: 'a'
        }, {
          point: state.next,
          className: 'mv-next'
        }]} />
        <p className="multivariate-caption">Curves have f=1,3,6,10 from inner to outer. Blue arrow: chosen unit u. Gold arrow: unit gradient direction (magnitude is listed below). Gold dot: a; outlined dot: a+hu. The graph of f would rise above this input plane.</p></div>
      <div><h4>Slice along the chosen direction</h4><FunctionPlot label="Exact function slice and its tangent prediction" xDomain={[-0.5, 0.5]} xLabel="signed distance t" yLabel="f(a+tu)" series={[{
          points: state.slices.map(row => [row.distance, row.actual])
        }, {
          points: state.slices.map(row => [row.distance, row.linear]),
          className: 'mv-linear'
        }]} selected={[{
          point: [step, state.actual]
        }, {
          point: [step, state.predicted],
          className: 'mv-next'
        }]} />
        <p className="multivariate-caption">Solid gold: exact values. Dashed blue: tangent prediction. Markers show the chosen h. Axes rescale to the slice. Computed readouts are rounded; floating-point trigonometry may leave a tiny residual in a mathematically zero rate.</p></div>
    </div>
    <dl className="multivariate-readout" aria-live="polite"><dt>Active point / gradient</dt><dd>{pair(base)} / {pair(state.gradient)}</dd><dt>Unit direction / gradient length</dt><dd>{pair(state.direction)} / {number(state.gradientNorm)}</dd><dt>Rate ∇f·u</dt><dd data-result="rate">{number(state.rate)}</dd><dt>Predicted change / actual change</dt><dd>{number(state.predictedChange)} / {number(state.actualChange)}</dd><dt>Exact remainder</dt><dd data-result="remainder">{number(state.remainder)}</dd></dl>
    <p className="multivariate-feedback">{state.gradientNorm === 0 ? 'At the origin every first-order rate is zero, yet any nonzero step raises this bowl by a second-order amount.' : Math.abs(state.rate) < 1e-9 ? 'The computed first-order rate is near zero. At the exact tangent direction it is zero, but a straight tangent step generally leaves the curved level set and its quadratic remainder can still be positive.' : 'The rate is change per unit signed distance at the base point. Multiplying by h predicts a finite change; curvature produces the remaining difference.'}</p>
  </Investigation>;
}
export function ApproachPathsLab() {
  const [kind, setKind] = useState('axis');
  const [coefficient, setCoefficient] = useState(1);
  const state = approachState(kind, coefficient);
  return <Investigation name="approach-paths" title="Can every straight approach look safe?">
    <p>At the origin define g=0; elsewhere g=x²y/(x⁴+y²). Inspect whether a line and a parabola approach the same output. The table approaches from positive x; a single conflicting path is enough to disprove a limit. Values are rounded, with scientific notation preserving tiny nonzero inputs.</p>
    <div className="multivariate-controls"><label>Approach path<select value={kind} onChange={event => setKind(event.target.value)}><option value="axis">x axis: y=0</option><option value="line">Straight line: y=cx</option><option value="parabola">Parabola: y=cx²</option></select></label><label>Coefficient c<select value={coefficient} disabled={kind === 'axis'} onChange={event => setCoefficient(Number(event.target.value))}>{[-2, -1, -0.5, 0, 0.5, 1, 2].map(value => <option key={value} value={value}>{value}</option>)}</select></label><button onClick={() => {
        setKind('axis');
        setCoefficient(1);
      }}>Reset</button></div>
    <div className="multivariate-linked"><div><h4>Selected input path</h4><CartesianPlot label="Selected approach path toward the origin" domain={[-2.2, 2.2]} curves={[{
          points: state.path,
          className: 'mv-direction'
        }]} points={[{
          point: [0, 0],
          label: '0'
        }, {
          point: state.path.at(-1)
        }]} /><p className="multivariate-caption">The blue path ends at the origin. The table resolves distances too small to see at this fixed scale.</p></div><div className="multivariate-table" tabIndex="0" role="region" aria-label="Approach values"><table><caption>Calculated values, not a limit proof</caption><thead><tr><th>x</th><th>y</th><th>g(x,y)</th></tr></thead><tbody>{state.samples.map(row => <tr key={row.x}><td>{number(row.x)}</td><td>{number(row.y)}</td><td>{number(row.value)}</td></tr>)}</tbody></table></div></div>
    <p className="multivariate-feedback" aria-live="polite">Algebraic path limit: <strong data-result="path-limit">{number(state.limit)}</strong>. {state.limit !== 0 ? 'This differs from g(0,0)=0, so the function is not continuous at the origin and cannot be differentiable there.' : 'This one path agrees with g(0,0). That agreement cannot establish a multivariable limit; try a nonzero parabola coefficient.'}</p>
  </Investigation>;
}
export function CircleGradientLab() {
  const [angle, setAngle] = useState(0);
  const state = circleMotionState(angle);
  const circle = Array.from({
    length: 121
  }, (_, index) => [Math.cos(index * Math.PI / 60), Math.sin(index * Math.PI / 60)]);
  return <Investigation name="circle-gradient" title="Uphill in the plane, but unable to leave the circle">
    <p>Maximize f=2x+y while x²+y²=1. Inspect where motion around the circle stops changing f, even though ∇f=(2,1) never vanishes. Angle is displayed in degrees; the calculated rate is per radian.</p>
    <div className="multivariate-controls"><label>Position angle: {number(angle)}°<input type="range" min="0" max="360" step="any" value={angle} onChange={event => setAngle(Number(event.target.value))} /></label><button onClick={() => setAngle(state.maximumAngle)}>Maximum candidate</button><button onClick={() => setAngle(state.maximumAngle + 180)}>Minimum candidate</button><button onClick={() => setAngle(0)}>Reset</button></div>
    <div className="multivariate-linked"><div><h4>Allowed velocity is tangent</h4><CartesianPlot label="Unit circle, point, tangent direction and fixed objective gradient" domain={[-2.4, 2.4]} curves={[{
          points: circle
        }]} vectors={[{
          from: state.point,
          to: state.point.map((value, index) => value + state.tangent[index]),
          className: 'mv-direction'
        }, {
          from: state.point,
          to: state.point.map((value, index) => value + state.gradient[index] * 0.5),
          className: 'mv-actual'
        }]} points={[{
          point: state.point,
          label: 'r'
        }]} /><p className="multivariate-caption">Blue arrow: unit tangent (−sin θ, cos θ). Gold: half the gradient for display. All positions use equal x/y scales; arrow scaling does not alter the rate calculation.</p></div><div><h4>Output along the full circle</h4><FunctionPlot label="Objective value along a full circle" series={[{
          points: state.samples.map(row => [row.angle, row.value])
        }]} xDomain={[0, 360]} yDomain={[-2.5, 2.5]} xLabel="angle in degrees" yLabel="2 cos θ + sin θ" selected={[{
          point: [angle, state.value]
        }]} /><p className="multivariate-caption">The plotted height is the objective, not the y coordinate of the moving point. Maximum and minimum are ±√5.</p></div></div>
    <dl className="multivariate-readout" aria-live="polite"><dt>Point / unit tangent</dt><dd>{pair(state.point)} / {pair(state.tangent)}</dd><dt>Objective</dt><dd data-result="circle-value">{number(state.value)}</dd><dt>Along-circle rate ∇f·r′</dt><dd data-result="circle-rate">{number(state.rate)}</dd><dt>Gradient component along tangent</dt><dd>{pair(state.tangentGradient)}</dd></dl>
    <p className="multivariate-feedback">{Math.abs(state.rate) < 1e-9 ? 'The allowed first-order rate is zero to the displayed calculation precision: gradient and constraint normal are parallel at the exact candidate. Both a maximum and a minimum satisfy this condition; compare their objective values.' : 'Only the gradient component along the tangent can change the objective to first order while respecting this constraint.'} Computed values are rounded; a tiny residual at an exact candidate comes from floating-point trigonometry.</p>
  </Investigation>;
}
export function CurvatureSlicesLab() {
  const [preset, setPreset] = useState('bowl');
  const [angle, setAngle] = useState(0);
  const state = curvatureState(preset, angle);
  return <Investigation name="curvature-slices" title="Does this stationary point curve up in every direction?">
    <p>Every example has zero gradient at the origin. Inspect the classification while rotating a slice; then compare the two quartic cases, whose Hessians are identical.</p>
    <div className="multivariate-controls"><label>Function at the origin<select value={preset} onChange={event => setPreset(event.target.value)}>{Object.entries(curvaturePresets).map(([key, model]) => <option key={key} value={key}>{model.title}</option>)}</select></label><label>Slice direction: {angle}°<input type="range" min="0" max="360" step="15" value={angle} onChange={event => setAngle(Number(event.target.value))} /></label><button onClick={() => {
        setPreset('bowl');
        setAngle(0);
      }}>Reset</button></div>
    <div className="multivariate-linked"><div><h4>Exact slice and quadratic model</h4><FunctionPlot label="Curvature slice through the stationary point" xDomain={[-1, 1]} xLabel="signed distance t" yLabel="f(tu)" series={[{
          points: state.samples.map(row => [row.t, row.actual])
        }, {
          points: state.samples.map(row => [row.t, row.quadratic]),
          className: 'mv-linear'
        }]} /><p className="multivariate-caption">Solid gold: exact function. Dashed blue: ½t²uᵀHu. They coincide for these quadratic functions. The quartic examples expose changes the quadratic model cannot see.</p></div><div><h4>At the origin</h4><p>f(x,y)={state.formula}</p><dl className="multivariate-readout"><dt>Hessian rows</dt><dd>{state.hessian.map(row => pair(row)).join('; ')}</dd><dt>Hessian eigenvalues</dt><dd>{pair(state.eigenvalues)}</dd><dt>Unit direction</dt><dd>{pair(state.direction)}</dd><dt>Directional curvature uᵀHu</dt><dd data-result="curvature">{number(state.curvature)}</dd></dl><p className="multivariate-feedback" aria-live="polite">{state.verdict}</p><p>A single slice can reveal a counterexample; sampling several good slices alone is not proof that every direction is good. The analytic Hessian test or the exact function establishes these verdicts.</p></div></div>
  </Investigation>;
}
export function GradientDescentLab() {
  const [rate, setRate] = useState(0.1);
  const [draft, setDraft] = useState('0.1');
  const [steps, setSteps] = useState(0);
  const [error, setError] = useState('');
  const state = descentState(rate, steps);
  const extent = Math.max(3.5, ...state.history.flatMap(row => row.point.map(value => Math.abs(value) * 1.15)));
  const changeRate = value => {
    setRate(value);
    setDraft(String(value));
    setSteps(0);
    setError('');
  };
  const applyRate = event => {
    event.preventDefault();
    try {
      if (!draft.trim()) throw new Error('Enter a learning rate.');
      const value = Number(draft);
      descentState(value, 0);
      changeRate(value);
    } catch (failure) {
      setError(failure.message);
    }
  };
  return <Investigation name="gradient-descent" title="Trace the update, including a rate that fails">
    <p>Start at (3,−2) on f=x²+2y². Inspect the next point as you step. Changing the learning rate restarts the same initial condition so the comparison is controlled.</p>
    <form className="multivariate-controls" onSubmit={applyRate}><label>Learning rate η<input inputMode="decimal" value={draft} onChange={event => setDraft(event.target.value)} /></label><button type="submit">Apply rate</button><button type="button" onClick={() => changeRate(0.1)}>Reset</button></form>
    {error && <p role="alert">{error} The active trajectory is unchanged.</p>}
    <div className="multivariate-controls">{[0.01, 0.1, 0.49, 0.5, 0.6].map(value => <button key={value} onClick={() => changeRate(value)}>η={value}</button>)}<button disabled={steps === 0} onClick={() => setSteps(steps - 1)}>Previous step</button><button disabled={steps === 12} onClick={() => setSteps(steps + 1)}>Next step</button><button disabled={steps === 12} onClick={() => setSteps(12)}>Run 12 steps</button></div>
    <div className="multivariate-linked"><div><h4>Successive input points</h4><CartesianPlot label="Gradient descent trajectory in input space" domain={[-extent, extent]} curves={[{
          points: state.history.map(row => row.point),
          className: 'mv-direction'
        }]} points={[{
          point: [0, 0],
          label: 'min',
          className: 'mv-next'
        }, ...state.history.map((row, index) => ({
          point: row.point,
          radius: index === steps ? 6 : 3
        }))]} /><p className="multivariate-caption">The connecting segments show discrete updates, not a continuous differential equation. Both axes expand together if the iteration diverges; inspect the stated coordinate range and exact values.</p></div><div className="multivariate-table" tabIndex="0" role="region" aria-label="Gradient descent history"><table><caption>Calculated iterate history</caption><thead><tr><th>Step</th><th>x</th><th>y</th><th>f</th></tr></thead><tbody>{state.history.map(row => <tr key={row.step}><td>{row.step}</td><td>{number(row.point[0])}</td><td>{number(row.point[1])}</td><td>{number(row.loss)}</td></tr>)}</tbody></table></div></div>
    <dl className="multivariate-readout" aria-live="polite"><dt>Active rate / step</dt><dd>{number(rate)} / {steps}</dd><dt>Coordinate factors (1−2η, 1−4η)</dt><dd>{pair(state.factors)}</dd><dt>Current point</dt><dd data-result="descent-point">{pair(state.current.point)}</dd><dt>Current loss / gradient</dt><dd>{number(state.current.loss)} / {pair(state.gradient)}</dd></dl>
    <p className="multivariate-feedback">{state.behavior}</p>
  </Investigation>;
}
export function GradientNormalFigure() {
  const unique = useId().replaceAll(':', '');
  const project = ([x, y, z]) => [172 + 42 * x - 27 * y, 242 + 15 * x + 18 * y - 27 * z];
  const point = [1, 1, 3];
  const base = project([1, 1, 0]);
  const surface = project(point);
  const normalTip = project([0.2, -0.6, 3.4]);
  const patch = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]].map(([x, y]) => project([x, y, 3 + 2 * (x - 1) + 4 * (y - 1)]));
  return <figure className="multivariate-inline" aria-label="Input gradient and surface normal are different objects">
    <div className="multivariate-linked"><div><h4>Input plane ℝ²</h4>
      <CartesianPlot label="Two-coordinate gradient at point one one" domain={[-0.5, 2.5]} curves={[{
          points: Array.from({
            length: 121
          }, (_, index) => [Math.sqrt(3) * Math.cos(index * Math.PI / 60), Math.sqrt(1.5) * Math.sin(index * Math.PI / 60)])
        }]} vectors={[{
          from: [1, 1],
          to: [1.5, 2],
          className: 'mv-actual'
        }]} points={[{
          point: [1, 1],
          label: 'a'
        }]} />
      <p>At a=(1,1): ∇f=(2,4). The arrow uses one quarter of this vector for display. It points across input coordinates, perpendicular to the level curve f=3.</p></div><div><h4>Tangent plane in ℝ³</h4>
      <svg className="multivariate-plot" viewBox="0 0 330 320" role="img" aria-label="Oblique view of tangent plane and three-coordinate normal at one one three">
        <title>Tangent plane patch, base point and graph normal</title>
        <defs><marker id={`${unique}-normal`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#99bfd4" /></marker></defs>
        {[[[0, 0, 0], [3, 0, 0], 'x'], [[0, 0, 0], [0, 3, 0], 'y'], [[0, 0, 0], [0, 0, 6], 'z']].map(([from, to, label]) => {
            const start = project(from);
            const end = project(to);
            return <g key={label}><line x1={start[0]} y1={start[1]} x2={end[0]} y2={end[1]} className="mv-axis" /><text x={end[0] + 7} y={end[1]}>{label}</text></g>;
          })}
        <polygon points={patch.map(p => p.join(',')).join(' ')} fill="#4c422c" fillOpacity="0.6" stroke="#b79758" />
        <line x1={base[0]} y1={base[1]} x2={surface[0]} y2={surface[1]} className="mv-linear" />
        <line x1={surface[0]} y1={surface[1]} x2={normalTip[0]} y2={normalTip[1]} className="mv-direction" markerEnd={`url(#${unique}-normal)`} />
        <circle cx={surface[0]} cy={surface[1]} r="5" className="mv-point" /><circle cx={base[0]} cy={base[1]} r="4" className="mv-next" />
        <text x="32" y="36">Blue arrow: 0.4 n</text><text x="32" y="58">Gold patch: tangent plane</text>
      </svg>
      <p>At (1,1,3): n=(−2,−4,1). It is normal to z=3+2(x−1)+4(y−1). The dashed line joins the input point to the graph point; it is not the normal.</p></div></div>
    <figcaption>The graph is the level surface F(x,y,z)=z−f(x,y)=0. Its normal ∇F=(−fₓ,−fᵧ,1) has three entries. The right panel uses an oblique projection, so apparent screen angles are not 3D angle measurements. Check n·(1,0,2)=n·(0,1,4)=0 for its two tangent directions.</figcaption>
  </figure>;
}
