import { useState } from 'react';
import { constraintNumber as number, coupledProjectionState, constraintPenaltyState, modifiedConstraintCost, originalConstraintCost, consensusAdmmState, paretoDecisionState, continuousTradeoffState, continuousObjectives, deploymentCandidates, normalizedScore } from '../../data/constrained-multiobjective-models';
import './constrained-multiobjective-labs.css';
const amber = '#f0bf65',
  green = '#7bd6ad',
  blue = '#87bdfa',
  pink = '#e7a4ce';
const vector = point => `(${point.map(number).join(', ')})`;
function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.1,
  disabled = false
}) {
  return <label className="constrained-field">{label}<strong>{number(value)}</strong><input type="range" aria-label={label} min={min} max={max} step={step} value={value} disabled={disabled} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Readout({
  rows
}) {
  return <dl className="constrained-readout">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Steps({
  step,
  setStep,
  max
}) {
  return <div className="constrained-buttons"><button onClick={() => setStep(0)} disabled={step === 0}>Reset trace</button><button onClick={() => setStep(step - 1)} disabled={step === 0}>Previous step</button><button onClick={() => setStep(step + 1)} disabled={step === max}>Next step</button><button onClick={() => setStep(max)} disabled={step === max}>Final step</button><output>Step {step} / {max}</output></div>;
}
function Plot({
  title,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  children,
  square = false
}) {
  const left = 46,
    right = 310,
    top = 40,
    bottom = square ? 304 : 240;
  const x = value => left + (value - xDomain[0]) / (xDomain[1] - xDomain[0]) * (right - left);
  const y = value => bottom - (value - yDomain[0]) / (yDomain[1] - yDomain[0]) * (bottom - top);
  const xticks = [...new Set([xDomain[0], ...(xDomain[0] < 0 && xDomain[1] > 0 ? [0] : []), xDomain[1]])];
  const yticks = [...new Set([yDomain[0], ...(yDomain[0] < 0 && yDomain[1] > 0 ? [0] : []), yDomain[1]])];
  return <svg viewBox={`0 0 340 ${bottom + 46}`} className="constrained-plot" role="img" aria-label={title}><title>{title}</title><text x="46" y="22" className="constrained-label">{yLabel}</text><path d={`M${left} ${top}V${bottom}H${right}`} className="constrained-axis" />{xticks.map(tick => <text key={tick} x={x(tick)} y={bottom + 19} textAnchor="middle" className="constrained-tick">{number(tick)}</text>)}{yticks.map(tick => <text key={tick} x={left - 8} y={y(tick) + 4} textAnchor="end" className="constrained-tick">{number(tick)}</text>)}{children({
      x,
      y,
      left,
      right,
      top,
      bottom
    })}<text x="178" y={bottom + 41} textAnchor="middle" className="constrained-label">{xLabel}</text></svg>;
}
function Polyline({
  points,
  x,
  y,
  color = amber,
  dashed = false
}) {
  let started = false;
  const d = points.map(point => {
    if (!point) {
      started = false;
      return '';
    }
    const command = started ? 'L' : 'M';
    started = true;
    return `${command}${x(point[0])} ${y(point[1])}`;
  }).join(' ');
  return <path d={d} fill="none" stroke={color} strokeWidth="2.5" strokeDasharray={dashed ? '6 5' : undefined} />;
}
function coordinateDomain(points, budget) {
  const values = [0, budget, ...points.flat()];
  return [Math.floor((Math.min(...values) - 0.25) * 2) / 2, Math.ceil((Math.max(...values) + 0.25) * 2) / 2];
}
function ConstraintGeometry({
  budget,
  target,
  points,
  exact,
  title,
  domain
}) {
  const [low, high] = domain;
  return <Plot title={title} xDomain={domain} yDomain={domain} xLabel="Coordinate 1" yLabel="Coordinate 2 · equal axis scales" square>{({
      x,
      y
    }) => <>
    <rect x={x(0)} y={y(high)} width={x(high) - x(0)} height={y(0) - y(high)} fill={green} opacity=".07" />
    <path d={`M${x(0)} ${y(low)}V${y(high)}M${x(low)} ${y(0)}H${x(high)}`} className="constrained-grid" />
    <Polyline points={[[Math.max(low, budget - high), Math.min(high, budget - low)], [Math.min(high, budget - low), Math.max(low, budget - high)]]} x={x} y={y} color={blue} dashed />
    <Polyline points={[[0, budget], [budget, 0]]} x={x} y={y} color={green} />
    <circle cx={x(target[0])} cy={y(target[1])} r="5" fill={amber} />
    {points.map((point, index) => <g key={index}><Polyline points={[index ? points[index - 1] : target, point]} x={x} y={y} color={pink} dashed /><circle cx={x(point[0])} cy={y(point[1])} r={index === points.length - 1 ? 6 : 4} fill={pink} /></g>)}
    <circle cx={x(exact[0])} cy={y(exact[1])} r="9" stroke={green} strokeWidth="2.5" fill="none" />
  </>}</Plot>;
}
export function CoupledProjectionLab() {
  const [target, setTarget] = useState([2, -0.6]);
  const [budget, setBudget] = useState(1);
  const [order, setOrder] = useState('orthant-first');
  const [step, setStep] = useState(0);
  const state = coupledProjectionState(target, budget, order);
  const frame = state.frames[step];
  const coordinate = (index, value) => {
    setTarget(target.map((item, i) => i === index ? value : item));
    setStep(0);
  };
  return <section className="constrained-lab" data-lab="coupled-projection" data-state={JSON.stringify({
    ...state,
    step
  })} aria-label="Coupled projection investigation">
    <p className="constrained-eyebrow">INVESTIGATION · TWO RULES, ONE DECISION</p><h3>Does fixing the second rule break the first?</h3><p>Inspect which rule survives a full pass. The shaded quadrant is C, the dashed line is D, and the thick green segment is their intersection. These are coordinates in an exact toy problem.</p>
    <div className="constrained-controls"><Slider label="Target coordinate 1" value={target[0]} onChange={value => coordinate(0, value)} min={-2} max={4} /><Slider label="Target coordinate 2" value={target[1]} onChange={value => coordinate(1, value)} min={-2} max={4} /><Slider label="Required sum B" value={budget} onChange={value => {
        setBudget(value);
        setStep(0);
      }} min={0.5} max={3} /><label className="constrained-field">Projection order<select aria-label="Projection order" value={order} onChange={event => {
          setOrder(event.target.value);
          setStep(0);
        }}><option value="orthant-first">C then D</option><option value="line-first">D then C</option></select></label></div>
    <Steps step={step} setStep={setStep} max={2} />
    <div className="constrained-pair"><figure><ConstraintGeometry domain={coordinateDomain(state.frames.map(item => item.point), budget)} budget={budget} target={target} points={state.frames.slice(1, step + 1).map(item => item.point)} exact={state.exact} title="Separate projection steps and the closest point in the intersection" /><figcaption>Amber: original target. Pink: separate projection trace. Green ring: exact projection onto C ∩ D. Ring and trace can coincide; the numeric checks distinguish coincidence from a general rule.</figcaption></figure><Readout rows={[[frame.label, vector(frame.point)], ['Nonnegativity violation · 0 satisfies C', number(frame.nonnegativeViolation)], ['Sum minus B · 0 satisfies D', number(frame.equalityResidual)], ['Squared distance from original target', number(frame.distanceSquared)], ['Exact intersection projection', vector(state.exact)], ['Exact minimum squared distance', number(state.exactDistanceSquared)]]} /></div>
    <p role="status">{step < 2 ? 'Advance the trace, then check both residuals.' : frame.nonnegativeViolation < 1e-12 && Math.abs(frame.equalityResidual) < 1e-12 ? 'This input happens to finish feasible. That does not prove the composition is the intersection projection for other inputs.' : 'The full pass remains infeasible: satisfying one set at a time did not preserve both rules.'}</p><p className="constrained-transfer">Try target (0.5, 3) with B = 2. Calculate the nearest point on the segment before advancing. Changing a control resets the trace.</p>
  </section>;
}
export function ConstraintPenaltyLab() {
  const [mode, setMode] = useState('quadratic');
  const [strength, setStrength] = useState(2);
  const state = constraintPenaltyState(mode, strength);
  const [minimum, maximum, increment] = mode === 'quadratic' ? [0, 20, 0.5] : mode === 'hinge' ? [0, 5, 0.1] : [0.01, 4, 0.01];
  const curve = Array.from({
    length: 451
  }, (_, index) => {
    const x = -1 + index / 100;
    const value = modifiedConstraintCost(mode, strength, x);
    return value === null || value > 10 ? null : [x, value];
  });
  return <section className="constrained-lab" data-lab="constraint-penalty" data-state={JSON.stringify(state)} aria-label="Penalty and barrier investigation"><p className="constrained-eyebrow">INVESTIGATION · CHANGE THE PROBLEM</p><h3>Is a strong preference the same as a hard limit?</h3><p>The original cost is ½(x − 3)², but x must be at most 1. Inspect which modified optimum can cross that boundary.</p><div className="constrained-controls"><label className="constrained-field">Modified objective<select aria-label="Modified objective" value={mode} onChange={event => {
          setMode(event.target.value);
          setStrength(event.target.value === 'barrier' ? 0.5 : 2);
        }}><option value="quadratic">Squared violation penalty</option><option value="hinge">Hinge violation penalty</option><option value="barrier">Interior log barrier</option></select></label><Slider label={mode === 'barrier' ? 'Barrier weight τ' : 'Penalty strength'} value={strength} onChange={setStrength} min={minimum} max={maximum} step={increment} /></div>
    <div className="constrained-pair"><figure><Plot title="Original and modified scalar objectives with the hard boundary at x equals one" xDomain={[-1, 3.5]} yDomain={[0, 10]} xLabel="Decision x" yLabel="Objective value · displayed 0–10">{({
            x,
            y,
            top,
            bottom
          }) => <><rect x={x(-1)} y={top} width={x(1) - x(-1)} height={bottom - top} fill={green} opacity=".07" /><path d={`M${x(1)} ${top}V${bottom}`} stroke={green} strokeDasharray="5 4" /><Polyline x={x} y={y} color={blue} points={Array.from({
              length: 91
            }, (_, i) => [-1 + i / 20, originalConstraintCost(-1 + i / 20)])} /><Polyline x={x} y={y} points={curve} /><circle cx={x(1)} cy={y(2)} r="7" stroke={green} fill="none" strokeWidth="2" /><circle cx={x(state.optimum)} cy={y(state.modifiedCost)} r="5" fill={amber} /></>}</Plot><figcaption>Blue: original cost. Amber: modified objective and its exact minimizer. Green ring: the hard-constrained answer (1, 2). Values above 10 are omitted from this view. For the barrier, x ≥ 1 is outside its domain, not a missing feasible branch.</figcaption></figure><Readout rows={[["Modified problem's minimizer", number(state.optimum)], ['Original cost there', number(state.originalCost)], ['Modified objective there', number(state.modifiedCost)], ['Violation max(x − 1, 0)', number(state.violation)], ['Signed slack 1 − x', number(state.slack)], ['Hard-constrained optimum', 'x = 1; original cost = 2']]} /></div><p role="status">{mode === 'quadratic' ? 'Every finite displayed squared penalty leaves positive violation.' : mode === 'hinge' ? strength >= 2 ? 'This example has reached its exact hinge threshold: strength ≥ 2.' : 'Below strength 2, this hinge optimum still violates the limit.' : 'A positive barrier weight keeps the solution strictly inside; reducing it approaches the boundary.'}</p><p className="constrained-transfer">Find a penalty strength giving violation below 0.1. Explain why that is still different from zero violation.</p></section>;
}
export function ConsensusAdmmLab() {
  const [target, setTarget] = useState([2, -0.6]);
  const [budget, setBudget] = useState(1);
  const [rho, setRho] = useState(1);
  const [step, setStep] = useState(0);
  const state = consensusAdmmState(target, budget, rho, 80);
  const domain = coordinateDomain(state.frames.flatMap(item => [item.x, item.z]), budget);
  const [low, high] = domain;
  const frame = state.frames[step];
  const coordinate = (index, value) => {
    setTarget(target.map((item, i) => i === index ? value : item));
    setStep(0);
  };
  return <section className="constrained-lab" data-lab="consensus-admm" data-state={JSON.stringify({
    ...state,
    step
  })} aria-label="Consensus ADMM investigation"><p className="constrained-eyebrow">INVESTIGATION · MAKE TWO COPIES AGREE</p><h3>Agreement is necessary, but one small residual is not enough</h3><p>x obeys nonnegativity; z obeys the required sum. The initial copies agree yet have not solved the objective. Inspect the first update, then compare both stopping tests.</p><div className="constrained-controls"><Slider label="ADMM target coordinate 1" value={target[0]} onChange={value => coordinate(0, value)} min={-2} max={4} /><Slider label="ADMM target coordinate 2" value={target[1]} onChange={value => coordinate(1, value)} min={-2} max={4} /><Slider label="ADMM required sum" value={budget} onChange={value => {
        setBudget(value);
        setStep(0);
      }} min={0.5} max={3} /><label className="constrained-field">Fixed penalty ρ<select aria-label="Fixed penalty rho" value={rho} onChange={event => {
          setRho(Number(event.target.value));
          setStep(0);
        }}>{[0.1, 0.3, 1, 3, 10].map(value => <option key={value} value={value}>{value}</option>)}</select></label></div><Steps step={step} setStep={setStep} max={80} /><Slider label="ADMM iteration" value={step} onChange={setStep} min={0} max={80} step={1} />
    <div className="constrained-pair"><figure><Plot title="ADMM constraint copies x and z in decision space" xDomain={domain} yDomain={domain} xLabel="Coordinate 1" yLabel="Coordinate 2 · equal axis scales" square>{({
            x,
            y
          }) => <><path d={`M${x(0)} ${y(low)}V${y(high)}M${x(low)} ${y(0)}H${x(high)}`} className="constrained-grid" /><Polyline points={[[Math.max(low, budget - high), Math.min(high, budget - low)], [Math.min(high, budget - low), Math.max(low, budget - high)]]} x={x} y={y} color={blue} dashed /><Polyline points={[[0, budget], [budget, 0]]} x={x} y={y} color={green} /><Polyline points={[frame.x, frame.z]} x={x} y={y} color={pink} dashed /><circle cx={x(frame.x[0])} cy={y(frame.x[1])} r="6" fill={amber} /><rect x={x(frame.z[0]) - 5} y={y(frame.z[1]) - 5} width="10" height="10" stroke={blue} fill="none" strokeWidth="2.5" /><circle cx={x(state.exact[0])} cy={y(state.exact[1])} r="10" stroke={green} fill="none" strokeWidth="2" /></>}</Plot><figcaption>Amber disk: x. Blue square: z. Their connector is the primal disagreement. Green ring: the known optimum of this teaching problem. As the copies converge to the solution the marks overlap. Equal axis ranges fit the complete 80-step run and stay fixed while stepping.</figcaption></figure><Readout rows={[["x · constrained to C", vector(frame.x)], ['z · constrained to D', vector(frame.z)], ['Accumulated scaled multiplier u', vector(frame.u)], ['Primal norm / tolerance', `${number(frame.primalNorm)} / ${number(frame.primalTolerance)}`], ['Dual norm / tolerance', frame.dualNorm === null ? 'Not defined before the first update' : `${number(frame.dualNorm)} / ${number(frame.dualTolerance)}`], ['x sum minus B', number(frame.xFeasibility.equalityResidual)], ['z nonnegativity violation', number(frame.zFeasibility.nonnegativeViolation)], ['Feasible repair P(C ∩ D)(z)', vector(frame.repair)], ['Repaired cost / known optimum cost', `${number(frame.repairedCost)} / ${number(state.exactCost)}`]]} /></div>
    <p role="status">{step === 0 ? 'Initial primal residual is zero. There is no completed update or dual-residual test yet.' : frame.stoppingPassed ? 'Both numerical stopping tests pass at this frame. Values are rounded; this is approximate convergence, not an exact feasibility certificate.' : 'At least one stopping test fails. Inspect both residuals and the original constraints.'}</p><p className="constrained-transfer">Compare ρ = 0.1 and 10 at the same iteration. Which residual decreases sooner? Changing inputs resets all copies and the multiplier; ρ stays fixed during each run.</p></section>;
}
export function ParetoDecisionLab() {
  const [latency, setLatency] = useState(25),
    [memory, setMemory] = useState(400),
    [price, setPrice] = useState(0.2),
    [method, setMethod] = useState('error');
  const state = paretoDecisionState(latency, memory, price, method);
  return <section className="constrained-lab" data-lab="pareto-decision" data-state={JSON.stringify(state)} aria-label="Pareto deployment investigation"><p className="constrained-eyebrow">INVESTIGATION · FILTER, COMPARE, THEN CHOOSE</p><h3>Can a sensible candidate disappear from every weighted optimum?</h3><p>All five measurements are hypothetical. Smaller error and latency are preferred; memory is a hard limit. The frontier is recomputed among the currently feasible candidates.</p><div className="constrained-controls"><Slider label="Maximum latency · ms" value={latency} onChange={setLatency} min={0} max={80} step={1} /><Slider label="Maximum memory · MB" value={memory} onChange={setMemory} min={0} max={400} step={8} /><label className="constrained-field">Preference after filtering<select aria-label="Preference after filtering" value={method} onChange={event => setMethod(event.target.value)}><option value="error">Error first, then latency</option><option value="weighted">Error points + latency price</option><option value="latency">Latency first, then error</option></select></label><Slider label="Price · error percentage points / ms" value={price} onChange={setPrice} min={0} max={1} step={0.01} disabled={method !== 'weighted'} /></div>
    <div className="constrained-pair"><figure><Plot title="Hypothetical deployment alternatives in latency and error objective space" xDomain={[0, 80]} yDomain={[6, 14]} xLabel="Latency · ms (smaller is better)" yLabel="Error · percent (smaller is better)">{({
            x,
            y,
            top,
            bottom
          }) => <><rect x={x(0)} y={top} width={x(latency) - x(0)} height={bottom - top} fill={green} opacity=".06" /><path d={`M${x(latency)} ${top}V${bottom}`} stroke={green} strokeDasharray="5 5" />{state.candidates.map(item => <g key={item.id} opacity={item.feasible ? 1 : 0.4}><circle cx={x(item.latencyMs)} cy={y(100 * item.error)} r="5" fill={item.feasible ? amber : '#929da7'} />{state.front.includes(item.id) && <circle cx={x(item.latencyMs)} cy={y(100 * item.error)} r="9" fill="none" stroke={green} strokeWidth="2" />}{state.selected === item.id && <rect x={x(item.latencyMs) - 12} y={y(100 * item.error) - 12} width="24" height="24" fill="none" stroke={blue} strokeWidth="2" />}<text x={x(item.latencyMs) + 12} y={y(100 * item.error) - 7} className="constrained-label">{item.id}</text></g>)}</>}</Plot><figcaption>Green rings: feasible Pareto alternatives. Blue box: selected decision. Faded points fail a budget. Points are discrete models; a line between them would not create a deployable model.</figcaption></figure><Readout rows={[["Selected model", state.selected ? state.candidates.find(item => item.id === state.selected).name : 'No feasible candidate'], ['Feasible frontier IDs', state.front.join(', ') || 'Empty'], ['Primary-score ties', state.ties.join(', ') || 'None'], ['Current weighted score definition', `error percentage points + ${number(price)} × latency in ms`]]} /></div>
    <div className="constrained-table" tabIndex="0" role="region" aria-label="Candidate metrics and feasibility table"><table><thead><tr><th>Model</th><th>Error %</th><th>ms</th><th>MB</th><th>Score</th><th>Budget check</th></tr></thead><tbody>{state.candidates.map(item => <tr key={item.id}><th>{item.id} · {item.name}</th><td>{number(100 * item.error)}</td><td>{item.latencyMs}</td><td>{item.memoryMb}</td><td>{number(item.score)}</td><td>{item.feasible ? 'Pass' : [item.latencyMs > latency && 'Latency', item.memoryMb > memory && 'Memory'].filter(Boolean).join(' + ')}</td></tr>)}</tbody></table></div><p role="status">{state.selected ? 'Selection applies the stated preference only after both hard limits. Primary values within 1e−10 count as numerical ties; ties use error, latency, then ID.' : 'These budgets leave no feasible choice. Report that result or revise the requirements explicitly.'}</p><p className="constrained-transfer">With latency 80 ms and memory 400 MB, sweep the weighted price: does Compact ever win? Switch to error first and a 15 ms limit. Then lower memory to zero.</p></section>;
}
export function ContinuousTradeoffLab() {
  const [method, setMethod] = useState('weighted'),
    [alpha, setAlpha] = useState(0.5),
    [epsilon, setEpsilon] = useState(1);
  const state = continuousTradeoffState(method, alpha, epsilon);
  const curve = Array.from({
    length: 101
  }, (_, i) => continuousObjectives(i / 50));
  return <section className="constrained-lab" data-lab="continuous-tradeoff" data-state={JSON.stringify(state)} aria-label="Continuous trade-off investigation"><p className="constrained-eyebrow">INVESTIGATION · ONE DECISION, TWO CONSEQUENCES</p><h3>Move the preference and watch the actual decision change</h3><p>A control x lies between 0 and 2. Its squared distances to targets 0 and 2 are the two objectives. Inspect whether increasing the weight on the first target moves x left or right.</p><div className="constrained-controls"><label className="constrained-field">Selection method<select aria-label="Continuous selection method" value={method} onChange={event => setMethod(event.target.value)}><option value="weighted">Weighted sum</option><option value="epsilon">Bound the second objective</option></select></label><Slider label="First-objective weight α" value={alpha} onChange={setAlpha} min={0} max={1} step={0.01} disabled={method !== 'weighted'} /><Slider label="Second-objective bound ε" value={epsilon} onChange={setEpsilon} min={0} max={4} step={0.01} disabled={method !== 'epsilon'} /></div><div className="constrained-pair"><figure><svg className="constrained-plot" viewBox="0 0 340 112" role="img" aria-label={`Selected decision x equals ${number(state.optimum)} between targets zero and two`}><path d="M46 50H310" className="constrained-axis" /><circle cx={46 + state.optimum / 2 * 264} cy="50" r="7" fill={amber} /><text x="46" y="83" textAnchor="middle" className="constrained-label">0</text><text x="310" y="83" textAnchor="middle" className="constrained-label">2</text><text x="178" y="22" textAnchor="middle" className="constrained-label">Decision x = {number(state.optimum)}</text><text x="178" y="107" textAnchor="middle" className="constrained-tick">Left target ← control → right target</text></svg><Plot title="Pareto curve traced by squared distances as x varies from zero to two" xDomain={[0, 4]} yDomain={[0, 4]} xLabel="First objective f₁ = x²" yLabel="Second objective f₂ = (x − 2)²">{({
            x,
            y,
            left,
            right
          }) => <><Polyline x={x} y={y} points={curve} color={green} />{method === 'epsilon' && <path d={`M${left} ${y(epsilon)}H${right}`} stroke={blue} strokeDasharray="5 5" />}<circle cx={x(state.objectives[0])} cy={y(state.objectives[1])} r="6" fill={amber} /></>}</Plot><figcaption>The continuous curve is attainable here: every point comes from an actual x. Its two endpoints favor different targets. The blue dashed line, when present, is the second-objective limit.</figcaption></figure><Readout rows={[["Selected x", number(state.optimum)], ['First objective x²', number(state.objectives[0])], ['Second objective (x − 2)²', number(state.objectives[1])], [method === 'weighted' ? 'Weighted objective' : 'Constraint slack ε − f₂', number(method === 'weighted' ? state.weightedScore : state.slack)]]} /></div><p className="constrained-transfer">Before using the controls, solve α = 0.25 and ε = 0.25 independently. Both choose x = 1.5, but they express different preferences.</p></section>;
}
export function UnitConversionFigure() {
  const selected = deploymentCandidates.filter(item => ['S', 'M', 'L'].includes(item.id));
  return <figure className="constrained-inline"><h3>The same physical latency, three numerical scores</h3><p className="constrained-scroll-note">Scroll the table to compare all three score columns.</p><div className="constrained-table" tabIndex="0" role="region" aria-label="Latency unit conversion comparison"><table><thead><tr><th>Model</th><th>0.2 points/ms</th><th>0.2 points/s · changed preference</th><th>200 points/s · same preference</th></tr></thead><tbody>{selected.map(item => <tr key={item.id}><th>{item.name}</th><td>{number(normalizedScore(item, 'ms'))}</td><td>{number(normalizedScore(item, 'seconds', false))}</td><td>{number(normalizedScore(item, 'seconds', true))}</td></tr>)}</tbody></table></div><figcaption>Score = error in percentage points + price × latency. The first and last columns agree exactly as mathematical quantities; displayed floating values are rounded. Keeping 0.2 after changing milliseconds to seconds changes the trade-off and selects Large instead of Medium.</figcaption></figure>;
}
export function CommonDescentFigure() {
  return <figure className="constrained-inline"><h3>Two slopes: agreement, conflict, or a blind spot</h3><div className="constrained-flow"><div><strong>At x = 3: gradients 6 and 2</strong><p>For d = −1, the directional rates are −6 and −2. A sufficiently small leftward step improves both squared-distance objectives.</p></div><div><strong>At x = 1: gradients 2 and −2</strong><p>Any nonzero scalar direction helps one objective and hurts the other. This point is on this problem’s Pareto curve.</p></div><div><strong>For two objectives −x² at x = 0</strong><p>Both gradients are zero, but every nearby nonzero x improves both. First derivatives alone miss this local maximum.</p></div></div><figcaption>Each rate is ∇fᵢ · d. The counterexample changes the objective functions explicitly; a zero first-order test is not a universal Pareto certificate.</figcaption></figure>;
}
