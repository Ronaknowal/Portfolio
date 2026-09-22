import { useId, useState } from 'react';
import { allocationCertificateState, convexChordState, convexCurves, ridgeContourPoints, ridgeCurvatureState, ridgeDesigns, softThresholdState } from '../../data/convex-optimization-models.js';
import './convex-optimization-labs.css';
const formatNumber = value => {
  if (value === null) return '—';
  if (value === 0 || Object.is(value, -0)) return '0';
  if (Math.abs(value) < 0.001 || Math.abs(value) >= 10000) return value.toExponential(2);
  return Number(value.toFixed(3)).toString();
};
const coordinates = values => `(${values.map(formatNumber).join(", ")})`;
function ConvexField({
  label,
  children
}) {
  const id = useId();
  return <div className="convex-field"><label htmlFor={id}>{label}</label>{children(id)}</div>;
}
function ConvexRange({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.25
}) {
  return <ConvexField label={`${label}: ${formatNumber(value)}`}>{id => <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />}</ConvexField>;
}
function ConvexGraph({
  title,
  description,
  xBounds,
  yBounds,
  xTicks,
  yTicks,
  xLabel,
  yLabel,
  children,
  square = false
}) {
  const id = useId();
  const width = 320;
  const height = square ? 320 : 260;
  const box = {
    left: 47,
    right: 299,
    top: 25,
    bottom: height - 43
  };
  const x = value => box.left + (value - xBounds[0]) * (box.right - box.left) / (xBounds[1] - xBounds[0]);
  const y = value => box.bottom - (value - yBounds[0]) * (box.bottom - box.top) / (yBounds[1] - yBounds[0]);
  const points = values => values.map(([first, second]) => `${x(first)},${y(second)}`).join(" ");
  return <svg className="convex-graph" viewBox={`0 -12 ${width} ${height + 42}`} role="img" aria-labelledby={`${id}-title ${id}-description`}>
    <title id={`${id}-title`}>{title}</title><desc id={`${id}-description`}>{description}</desc>
    <defs><clipPath id={`${id}-clip`}><rect x={box.left} y={box.top} width={box.right - box.left} height={box.bottom - box.top} /></clipPath></defs>
    {xTicks.map(tick => <g key={tick}><line className="convex-grid" x1={x(tick)} x2={x(tick)} y1={box.top} y2={box.bottom} /><text x={x(tick)} y={box.bottom + 23} textAnchor={tick === xBounds[0] ? 'start' : tick === xBounds[1] ? 'end' : 'middle'}>{formatNumber(tick)}</text></g>)}
    {yTicks.map(tick => <g key={tick}><line className="convex-grid" y1={y(tick)} y2={y(tick)} x1={box.left} x2={box.right} /><text x={box.left - 8} y={y(tick) + 4} textAnchor="end">{formatNumber(tick)}</text></g>)}
    <line className="convex-axis" x1={box.left} x2={box.right} y1={box.bottom} y2={box.bottom} />
    <line className="convex-axis" x1={box.left} x2={box.left} y1={box.top} y2={box.bottom} />
    <text x={(box.left + box.right) / 2} y={height + 12} textAnchor="middle">{xLabel}</text>
    <text x={box.left} y={14}>{yLabel}</text>
    <g clipPath={`url(#${id}-clip)`}>{children({
        x,
        y,
        points,
        box
      })}</g>
  </svg>;
}
function ConvexReadout({
  values
}) {
  return <dl className="convex-readout">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
export function FeasibleMixingFigure() {
  const panels = [{
    title: "All points on the segment stay allowed",
    convex: true,
    points: [[0, 3], [3, 0]],
    midpoint: [1.5, 1.5]
  }, {
    title: "A forbidden hole breaks the segment",
    convex: false,
    points: [[0.5, 2], [3.5, 2]],
    midpoint: [2, 2]
  }];
  return <figure className="convex-inline" aria-label="A segment inside a triangle and a segment crossing a forbidden hole">
    <figcaption>Mix two allowed decisions. Is every intermediate decision still allowed?</figcaption>
    <div className="convex-pair">{panels.map(panel => <div key={panel.title}>
      <ConvexGraph title={panel.title} description={panel.convex ? "The midpoint (1.5,1.5) between (0,3) and (3,0) stays inside x plus y at most 4, with nonnegative coordinates." : "Both endpoints lie in the square outside a forbidden disk. Their midpoint (2,2) lies in the hole."} xBounds={[-0.3, 4.3]} yBounds={[-0.3, 4.3]} xTicks={[0, 2, 4]} yTicks={[0, 2, 4]} xLabel="first decision" yLabel="second decision" square>
        {({
            x,
            y,
            points
          }) => <>
          {panel.convex ? <polygon className="convex-region" points={points([[0, 0], [4, 0], [0, 4]])} /> : <><polygon className="convex-region" points={points([[0, 0], [4, 0], [4, 4], [0, 4]])} /><ellipse className="convex-hole" cx={x(2)} cy={y(2)} rx={x(2.8) - x(2)} ry={y(2) - y(2.8)} /></>}
          <polyline className="convex-chord" points={points(panel.points)} />
          {panel.points.map(point => <circle key={point.join(',')} className="convex-endpoint" cx={x(point[0])} cy={y(point[1])} r="4" />)}
          <path className={panel.convex ? 'convex-optimum' : 'convex-invalid'} d={`M${x(panel.midpoint[0]) - 5},${y(panel.midpoint[1]) - 5}l10,10m-10,0l10,-10`} />
        </>}
      </ConvexGraph><p><strong>{panel.convex ? "Convex triangle." : "Nonconvex region."}</strong> {panel.title}. Dots are endpoints; the cross is their midpoint. Shading means allowed.</p>
    </div>)}</div>
  </figure>;
}
export function ConvexChordLab() {
  const [preset, setPreset] = useState('quadratic');
  const [left, setLeft] = useState(-1.5);
  const [right, setRight] = useState(1.5);
  const [fraction, setFraction] = useState(0.5);
  const state = convexChordState(preset, left, right, fraction);
  const slope = state.slope ?? 0;
  const supportingLine = [-2, 2].map(input => [input, state.value + slope * (input - state.input)]);
  const reset = () => {
    setPreset('quadratic');
    setLeft(-1.5);
    setRight(1.5);
    setFraction(0.5);
  };
  return <section className="lesson-lab convex-lab" aria-label="Chord and supporting-line investigation">
    <h3>Mix the inputs, then compare the heights</h3>
    <p>Inspect whether the solid curve lies below the endpoint chord. Then try the two-well function at its midpoint. Moving a chord is a test case; proving convexity requires every pair in the domain.</p>
    <div className="convex-controls">
      <ConvexField label="Function">{id => <select id={id} value={preset} onChange={event => setPreset(event.target.value)}>{Object.entries(convexCurves).map(([key, curve]) => <option key={key} value={key}>{curve.title}</option>)}</select>}</ConvexField>
      <ConvexRange label="Left input a" value={left} onChange={setLeft} min={-2} max={-0.25} />
      <ConvexRange label="Right input b" value={right} onChange={setRight} min={0.25} max={2} />
      <ConvexRange label="Fraction θ toward b" value={fraction} onChange={setFraction} min={0} max={1} step={0.05} />
      <button onClick={reset}>Reset chord</button>
    </div>
    <ConvexGraph title="Function, endpoint chord and supporting-line test" description={`At x=${formatNumber(state.input)}, the function height is ${formatNumber(state.value)} and chord height is ${formatNumber(state.chordValue)}. Their difference is ${formatNumber(state.gap)}.`} xBounds={[-2, 2]} yBounds={[-2, 5]} xTicks={[-2, -1, 0, 1, 2]} yTicks={[-2, 0, 2, 4]} xLabel="input x" yLabel="function height">
      {({
        x,
        y,
        points
      }) => <>
        <polyline className="convex-support" points={points(supportingLine)} />
        <polyline className="convex-curve" points={points(state.samples)} />
        <polyline className="convex-chord" points={points(state.endpoints)} />
        <line className={state.gap < -1e-10 ? 'convex-invalid' : 'convex-gap'} x1={x(state.input)} x2={x(state.input)} y1={y(state.value)} y2={y(state.chordValue)} />
        {state.endpoints.map(point => <circle key={point[0]} className="convex-endpoint" cx={x(point[0])} cy={y(point[1])} r="4" />)}
        <circle className="convex-current" cx={x(state.input)} cy={y(state.value)} r="5" />
      </>}
    </ConvexGraph>
    <p className="convex-legend">Gold solid: function · blue: endpoint chord · green dashed: tangent, or slope 0 support at the |x| kink. Lines are clipped to the labeled window.</p>
    <ConvexReadout values={[["Mixed input", formatNumber(state.input)], ["Function height", formatNumber(state.value)], ["Chord height", formatNumber(state.chordValue)], ["Chord minus function", formatNumber(state.gap)]]} />
    <p role="status">{state.gap < -1e-10 ? "The negative gap is a concrete violation: the mixed input costs more than the mixed endpoint costs. This function is not convex on this domain." : "This chosen chord passes the inequality. A passed finite test alone does not prove convexity."} {state.slope === null ? "At zero, |x| has no derivative; the horizontal line is one of its valid supporting lines." : `The tangent slope here is ${formatNumber(state.slope)}.`}</p>
    <p>Transfer: keep the quartic selected and move the input toward zero. Its curvature becomes arbitrarily small there even though it remains strictly convex.</p>
  </section>;
}
export function AllocationCertificateLab() {
  const [budget, setBudget] = useState(4);
  const [first, setFirst] = useState(2);
  const [second, setSecond] = useState(1);
  const state = allocationCertificateState(budget, first, second);
  const setCandidate = candidate => {
    setFirst(candidate[0]);
    setSecond(candidate[1]);
  };
  const reset = () => {
    setBudget(4);
    setCandidate([2, 1]);
  };
  return <section className="lesson-lab convex-lab" aria-label="Feasible allocation and optimality certificate investigation">
    <h3>A low objective needs an allowed decision</h3>
    <p>The desired allocation is (4,3), and the cost is half the squared distance to it. Inspect where the best feasible point lies when the total budget is 4. Then inspect why its gradient need not be zero.</p>
    <div className="convex-controls">
      <ConvexRange label="Total budget B" value={budget} onChange={setBudget} min={0} max={8} step={0.5} />
      <ConvexRange label="First allocation u" value={first} onChange={setFirst} min={0} max={8} />
      <ConvexRange label="Second allocation v" value={second} onChange={setSecond} min={0} max={8} />
      <button onClick={() => setCandidate(state.optimum)}>Use exact optimum</button>
      <button onClick={() => setCandidate([4, 3])}>Try desired allocation</button>
      <button onClick={reset}>Reset allocation</button>
    </div>
    <ConvexGraph title="A closest point in the feasible triangle" description={`Budget ${budget}. Candidate ${coordinates(state.candidate)} is ${state.feasible ? 'feasible' : 'infeasible'}. Exact optimum ${coordinates(state.optimum)}. Triangle vertices are (0,0), (B,0), and (0,B).`} xBounds={[-0.5, 8.5]} yBounds={[-0.5, 8.5]} xTicks={[0, 4, 8]} yTicks={[0, 4, 8]} xLabel="first allocation u" yLabel="second allocation v" square>
      {({
        x,
        y,
        points
      }) => <>
        <polygon className="convex-region" points={points(state.vertices)} />
        <polyline className="convex-support" points={points([state.candidate, state.target])} />
        <circle className="convex-target" cx={x(4)} cy={y(3)} r="6" />
        <path className="convex-optimum" d={`M${x(state.optimum[0]) - 6},${y(state.optimum[1]) - 6}l12,12m-12,0l12,-12`} />
        <circle className={state.feasible ? 'convex-current' : 'convex-bad-point'} cx={x(first)} cy={y(second)} r="4" />
      </>}
    </ConvexGraph>
    <p className="convex-legend">Shaded triangle: allowed decisions · hollow circle: desired (4,3) · cross: exact optimum · filled dot: candidate. Dashed segment shows distance to the desired decision.</p>
    <ConvexReadout values={[["Candidate cost", formatNumber(state.objective)], ["Gradient g", coordinates(state.gradient)], ["Linear lower bound", formatNumber(state.lowerBound)], ["Certified gap", state.feasible ? formatNumber(state.gap) : "No feasible upper bound"], ["Exact optimum", coordinates(state.optimum)], ["Budget violation", formatNumber(state.violation)]]} />
    <details><summary>Inspect all three vertex directions</summary><div className="convex-table-wrap"><table><caption>Evaluate g · (vertex − candidate)</caption><thead><tr><th>Vertex</th><th>Directional value</th></tr></thead><tbody>{state.vertices.map((vertex, index) => <tr key={index}><td>{coordinates(vertex)}</td><td>{formatNumber(state.directionalValues[index])}</td></tr>)}</tbody></table></div><p>Add the smallest directional value to the candidate cost for the lower bound. Every point in the triangle is a convex combination of its vertices, so no linear value in the triangle lies below their minimum.</p></details>
    <p role="status">{!state.feasible ? "The candidate exceeds the shared budget. Its attractive objective is not a feasible upper bound on the best allowed cost." : state.gap < 1e-9 ? "The feasible candidate attains its lower bound, certifying global optimality for this model." : `The feasible objective is at most ${formatNumber(state.gap)} above the unknown optimum according to this certificate; here the exact oracle shows an actual gap ${formatNumber(state.actualError)}.`}</p>
    <p>Transfer: compare budgets 0, 0.5, 4 and 8. At a boundary, only allowed directions matter. The exact oracle is available because this two-variable example has a closed-form projection; a general solver does not get it for free.</p>
  </section>;
}
export function EpigraphFigure() {
  const values = Array.from({
    length: 71
  }, (_, index) => {
    const c = index / 10;
    return [c, Math.max(Math.abs(c - 1), Math.abs(c - 2), Math.abs(c - 6))];
  });
  return <figure className="convex-inline">
    <figcaption>Turn “minimize the worst error” into “lower a ceiling that covers every error”</figcaption>
    <ConvexGraph title="Epigraph of the worst absolute fitting error" description="For readings 1,2,6, the smallest allowed ceiling at level c is max absolute error. The lowest feasible point is c=3.5, t=2.5. Shading lies above this V-shaped boundary." xBounds={[0, 7]} yBounds={[0, 7]} xTicks={[0, 3.5, 7]} yTicks={[0, 2.5, 5, 7]} xLabel="shared fitted level c" yLabel="error ceiling t">
      {({
        x,
        y,
        points
      }) => <><polygon className="convex-region" points={points([[0, 7], ...values, [7, 7]])} /><polyline className="convex-curve" points={points(values)} /><circle className="convex-current" cx={x(3.5)} cy={y(2.5)} r="5" /></>}
    </ConvexGraph>
    <p>Allowed points satisfy t≥|c−1|, t≥|c−2| and t≥|c−6|. For fixed c, lowering t stops at the largest of those errors. The plotted boundary is calculated from these three readings; the middle reading does not control the minimum ceiling.</p>
  </figure>;
}
export function RidgeCurvatureLab() {
  const [preset, setPreset] = useState('full');
  const [penalty, setPenalty] = useState(0.5);
  const [stepFactor, setStepFactor] = useState(1);
  const [iterations, setIterations] = useState(0);
  const state = ridgeCurvatureState(preset, penalty, stepFactor, iterations);
  const extent = Math.max(3, ...state.path.flatMap(frame => frame.weights.map(Math.abs)), ...state.optimum.map(Math.abs));
  const boundary = Math.ceil(extent * 1.15 * 2) / 2;
  const contours = [0.5, 2, 8];
  const reset = () => {
    setPreset('full');
    setPenalty(0.5);
    setStepFactor(1);
    setIterations(0);
  };
  const setting = setter => value => {
    setter(value);
    setIterations(0);
  };
  return <section className="lesson-lab convex-lab" aria-label="Ridge curvature and gradient trajectory investigation">
    <h3>Watch the coefficients move across objective contours</h3>
    <p>Inspect the effect of duplicate columns with zero penalty. Then compare a stable step ηL=1 with ηL=2.1. The second step can overshoot even though the objective is convex.</p>
    <div className="convex-controls">
      <ConvexField label="Feature matrix">{id => <select id={id} value={preset} onChange={event => setting(setPreset)(event.target.value)}>{Object.entries(ridgeDesigns).map(([key, design]) => <option key={key} value={key}>{design.title}</option>)}</select>}</ConvexField>
      <ConvexField label="Ridge penalty λ">{id => <select id={id} value={penalty} onChange={event => setting(setPenalty)(Number(event.target.value))}>{[0, 0.1, 0.5, 1, 2].map(value => <option key={value} value={value}>{value}</option>)}</select>}</ConvexField>
      <ConvexField label="Step times largest curvature ηL">{id => <select id={id} value={stepFactor} onChange={event => setting(setStepFactor)(Number(event.target.value))}>{[0.5, 1, 1.9, 2, 2.1, 2.2].map(value => <option key={value} value={value}>{value}</option>)}</select>}</ConvexField>
    </div>
    <div className="convex-actions"><button disabled={iterations === 0} onClick={() => setIterations(value => value - 1)}>Previous step</button><button disabled={iterations === 24} onClick={() => setIterations(value => value + 1)}>Next step</button><button disabled={iterations === 24} onClick={() => setIterations(24)}>Run 24 steps</button><button onClick={reset}>Reset ridge</button></div>
    <ConvexGraph title="Actual gradient descent in the two-coefficient plane" description={`Iteration ${iterations}, coefficients ${coordinates(state.current.weights)}, objective ${formatNumber(state.current.value)}. ${state.unique ? "Contours are ellipses around the unique optimum." : "Contours are parallel lines around a whole line of equally good coefficient pairs."}`} xBounds={[-boundary, boundary]} yBounds={[-boundary, boundary]} xTicks={[-boundary, 0, boundary]} yTicks={[-boundary, 0, boundary]} xLabel="coefficient w₁" yLabel="coefficient w₂" square>
      {({
        x,
        y,
        points
      }) => <>
        {contours.map(excess => state.unique ? <polyline className="convex-contour" key={excess} points={points(ridgeContourPoints(state, excess))} /> : [-1, 1].map(sign => <polyline className="convex-contour" key={`${excess}-${sign}`} points={points([-boundary, boundary].map(first => [first, 5 / 3 - first + sign * Math.sqrt(excess / 3)]))} />))}
        {!state.unique && <polyline className="convex-support" points={points([-boundary, boundary].map(first => [first, 5 / 3 - first]))} />}
        <polyline className="convex-curve" points={points(state.path.map(frame => frame.weights))} />
        {state.path.map(frame => <circle key={frame.iteration} className="convex-endpoint" cx={x(frame.weights[0])} cy={y(frame.weights[1])} r="2.5" />)}
        <circle className="convex-current" cx={x(state.current.weights[0])} cy={y(state.current.weights[1])} r="5" />
        <path className="convex-optimum" d={`M${x(state.optimum[0]) - 5},${y(state.optimum[1]) - 5}l10,10m-10,0l10,-10`} />
      </>}
    </ConvexGraph>
    <p className="convex-legend">Blue contours: cost f*+0.5, f*+2 and f*+8 · gold path and filled dot: actual iterations · green cross: reference optimum. With duplicate columns and λ=0, the dashed line contains all minimizers; the cross is only its minimum-norm member. Axes expand together to keep the trajectory visible; contours are clipped.</p>
    <ConvexReadout values={[["Iteration", `${iterations} / 24`], ["Current coefficients", coordinates(state.current.weights)], ["Current objective", formatNumber(state.current.value)], ["Gap to exact optimum", formatNumber(state.current.error)], ["Smallest curvature μ", formatNumber(state.smallestCurvature)], ["Largest curvature L", formatNumber(state.largestCurvature)], ["Condition ratio L/μ", state.unique ? formatNumber(state.condition) : "No positive lower curvature"], ["Actual step η", formatNumber(state.step)]]} />
    <p role="status">{state.stepFactor >= 2 ? "For this quadratic, a curvature-L error component is multiplied by 1−ηL each step. At 2 its magnitude persists; above 2 it grows if present. A temporary decrease in total cost is not a convergence guarantee." : "Every positive-curvature component contracts with this step. Slow directions can still need many iterations."} {!state.unique && "The zero-curvature component stays fixed, so descent can reach a different member of the optimum line from the green cross."}</p>
    <details><summary>Inspect coefficients and cost at every computed step</summary><div className="convex-table-wrap"><table><caption>Actual bounded trajectory; no empirical runtime comparison</caption><thead><tr><th>Step</th><th>w₁</th><th>w₂</th><th>Cost</th></tr></thead><tbody>{state.path.map(frame => <tr key={frame.iteration}><td>{frame.iteration}</td><td>{formatNumber(frame.weights[0])}</td><td>{formatNumber(frame.weights[1])}</td><td>{formatNumber(frame.value)}</td></tr>)}</tbody></table></div></details>
    <p>Transfer: compare λ=0 and λ=0.5 with duplicate columns. The penalty removes the flat direction by changing the model. It is not a free numerical acceleration of the same optimization question.</p>
  </section>;
}
export function SoftThresholdLab() {
  const [input, setInput] = useState(3);
  const [penalty, setPenalty] = useState(1);
  const [candidate, setCandidate] = useState(3);
  const state = softThresholdState(input, penalty, candidate);
  const maximum = Math.ceil(Math.max(...state.samples.map(point => point[1])) / 5) * 5;
  const reset = () => {
    setInput(3);
    setPenalty(1);
    setCandidate(3);
  };
  return <section className="lesson-lab convex-lab" aria-label="Soft thresholding and subgradient investigation">
    <h3>A kink can be the exact optimum</h3>
    <p>Balance staying near a against paying λ for each unit of |x|. Explore when the optimum becomes exactly zero; compare a=1, λ=1 with a=3, λ=1.</p>
    <div className="convex-controls">
      <ConvexRange label="Unregularized input a" value={input} onChange={setInput} min={-4} max={4} />
      <ConvexRange label="Absolute-value penalty λ" value={penalty} onChange={setPenalty} min={0} max={4} />
      <ConvexRange label="Candidate x" value={candidate} onChange={setCandidate} min={-4} max={4} />
      <button onClick={() => setCandidate(state.optimum)}>Use proximal optimum</button><button onClick={reset}>Reset threshold</button>
    </div>
    <div className="convex-pair">
      <ConvexGraph title="The sum of squared-distance cost and absolute-value penalty" description={`Input ${input}, penalty ${penalty}; optimum ${formatNumber(state.optimum)}. Candidate ${candidate} has cost ${formatNumber(state.value)} and subgradient interval ${coordinates(state.subgradient)}.`} xBounds={[-5, 5]} yBounds={[0, maximum]} xTicks={[-4, 0, 4]} yTicks={[0, maximum / 2, maximum]} xLabel="candidate x" yLabel="total objective">
        {({
          x,
          y,
          points
        }) => <><polyline className="convex-curve" points={points(state.samples)} /><circle className="convex-current" cx={x(candidate)} cy={y(state.value)} r="5" /><path className="convex-optimum" d={`M${x(state.optimum) - 5},${y(state.optimumValue) - 5}l10,10m-10,0l10,-10`} /></>}
      </ConvexGraph>
      <ConvexGraph title="Soft thresholding maps a whole interval of inputs to zero" description={`For fixed penalty ${penalty}, input a between minus lambda and plus lambda maps to0. Outside that interval subtract lambda from the magnitude while preserving sign.`} xBounds={[-4, 4]} yBounds={[-4, 4]} xTicks={[-4, 0, 4]} yTicks={[-4, 0, 4]} xLabel="input a" yLabel="optimal x">
        {({
          x,
          y,
          points
        }) => <><polyline className="convex-support" points={points([[-4, -4], [4, 4]])} /><polyline className="convex-curve" points={points([[-4, -Math.max(4 - penalty, 0)], [-penalty, 0], [penalty, 0], [4, Math.max(4 - penalty, 0)]])} /><circle className="convex-current" cx={x(input)} cy={y(state.optimum)} r="5" /></>}
      </ConvexGraph>
    </div>
    <p className="convex-legend">Left: gold objective, candidate dot and optimum cross. Right: exact soft-threshold map; dashed diagonal is the unchanged input. Vertical scales differ and are labeled.</p>
    <ConvexReadout values={[["Optimal x", formatNumber(state.optimum)], ["Candidate subgradients", `[${state.subgradient.map(formatNumber).join(", ")}]`], ["Candidate objective", formatNumber(state.value)], ["Optimal objective", formatNumber(state.optimumValue)]]} />
    <p role="status">{state.stationary ? "Zero belongs to the candidate’s subgradient set, so this candidate is the global minimizer. The quadratic part makes it unique." : "The candidate’s subgradient set does not contain zero. Move to the proximal optimum and inspect the condition again."}</p>
    <p>Transfer: use a negative input and then λ=0. Soft thresholding preserves sign beyond the threshold and shrinks magnitude; it does not clip large values to ±λ.</p>
  </section>;
}
export function TotalVariationFigure() {
  const observed = [0.2, -0.1, 0.1, 3, 3.2, 2.8];
  const fitted = [1 / 6, 1 / 6, 1 / 6, 2.9, 2.9, 2.9];
  return <figure className="convex-inline"><figcaption>Penalizing adjacent changes can preserve a jump while reducing small fluctuations</figcaption>
    <ConvexGraph title="Calculated total-variation fit to six invented observations" description="Observed values 0.2,-0.1,0.1,3,3.2,2.8 are fit with three values 1/6 followed by three values 2.9 at penalty 0.3. This is a small calculated optimization example, not a measured sensor experiment." xBounds={[0.8, 6.2]} yBounds={[-0.5, 3.5]} xTicks={[1, 2, 3, 4, 5, 6]} yTicks={[0, 1, 2, 3]} xLabel="sample index" yLabel="signal value">
      {({
        x,
        y,
        points
      }) => <><polyline className="convex-chord" points={points(observed.map((value, index) => [index + 1, value]))} /><polyline className="convex-curve" points={points(fitted.map((value, index) => [index + 1, value]))} />{observed.map((value, index) => <circle key={index} className="convex-endpoint" cx={x(index + 1)} cy={y(value)} r="4" />)}</>}
    </ConvexGraph><p>Blue dots: invented observations. Gold: the verified solution. Connecting lines guide the eye between discrete samples. The penalty acts on neighboring differences, so the optimization couples adjacent fitted values.</p>
  </figure>;
}
