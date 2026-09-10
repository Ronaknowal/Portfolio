import { useId, useState } from 'react';
import { formatDualityNumber as number, projectionCertificateState, resourceDualAscentState, scalarKktState, sensitivityState } from '../../data/duality-kkt-models.js';
import './duality-kkt-labs.css';
function Range({
  label,
  value,
  setValue,
  min,
  max,
  step = 0.25
}) {
  return <label className="duality-field"><span>{label} · {number(value)}</span><input aria-label={label} type="range" value={value} min={min} max={max} step={step} onChange={event => setValue(Number(event.target.value))} /></label>;
}
function Plot({
  title,
  children,
  square = false
}) {
  const id = useId();
  return <svg className="duality-plot" viewBox={`0 0 330 ${square ? 330 : 260}`} role="img" aria-labelledby={id}><title id={id}>{title}</title>{children}</svg>;
}
function Readout({
  rows
}) {
  return <dl className="duality-readout">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function polyline(points, x, y) {
  return points.map((point, index) => `${index ? 'L' : 'M'}${x(point[0])},${y(point[1])}`).join(' ');
}
function BudgetPlane({
  state
}) {
  const clip = useId();
  const x = value => 44 + (value + 4) * 246 / 13;
  const y = value => 282 - (value + 4) * 246 / 13;
  const polygon = [];
  const corners = [[-4, -4], [9, -4], [9, 9], [-4, 9]];
  for (let index = 0; index < 4; index++) {
    const a = corners[index];
    const b = corners[(index + 1) % 4];
    const insideA = a[0] + a[1] <= state.budget;
    const insideB = b[0] + b[1] <= state.budget;
    if (insideA) polygon.push(a);
    if (insideA !== insideB) {
      const fraction = (state.budget - a[0] - a[1]) / (b[0] + b[1] - a[0] - a[1]);
      polygon.push(a.map((value, coordinate) => value + fraction * (b[coordinate] - value)));
    }
  }
  return <Plot square title="Equal-unit coordinate plane with the budget half-plane, target, candidate, exact projection and Lagrangian minimizer">
    <defs><clipPath id={clip}><rect x="44" y="36" width="246" height="246" /></clipPath></defs>
    <g clipPath={`url(#${clip})`}>
      <polygon points={polygon.map(point => `${x(point[0])},${y(point[1])}`).join(' ')} className="duality-region" />
      {[-4, 0, 4, 8].map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1="36" y2="282" className="duality-grid" /><line x1="44" x2="290" y1={y(value)} y2={y(value)} className="duality-grid" /></g>)}
      <circle cx={x(3)} cy={y(4)} r={Math.sqrt(state.objective) * 246 / 13} className="duality-contour" />
      <line x1={x(-4)} y1={y(state.budget + 4)} x2={x(9)} y2={y(state.budget - 9)} className="duality-line green" />
      <line x1={x(3)} y1={y(4)} x2={x(state.optimum[0])} y2={y(state.optimum[1])} className="duality-line green dashed" />
      <circle cx={x(3)} cy={y(4)} r="5" className="duality-dot blue" />
      <rect x={x(state.candidate[0]) - 5} y={y(state.candidate[1]) - 5} width="10" height="10" className="duality-dot amber" />
      <circle cx={x(state.optimum[0])} cy={y(state.optimum[1])} r="9" className="duality-ring green" />
      <path d={`M${x(state.minimizer[0]) - 7},${y(state.minimizer[1])}h14 M${x(state.minimizer[0])},${y(state.minimizer[1]) - 7}v14`} className="duality-line pink" />
    </g>
    <rect x="44" y="36" width="246" height="246" className="duality-border" />
    <text x="44" y="23" className="duality-label">coordinate y</text><text x="290" y="324" textAnchor="end" className="duality-label">coordinate x</text>
    {[-4, 0, 4, 8].map(value => <g key={value}><text x={x(value)} y="301" textAnchor="middle" className="duality-tick">{value}</text><text x="35" y={y(value) + 4} textAnchor="end" className="duality-tick">{value}</text></g>)}
  </Plot>;
}
export function ProjectionCertificateLab() {
  const [budget, setBudget] = useState(5);
  const [candidate, setCandidate] = useState([2, 2]);
  const [multiplier, setMultiplier] = useState(1);
  const state = projectionCertificateState(budget, candidate, multiplier);
  function reset() {
    setBudget(5);
    setCandidate([2, 2]);
    setMultiplier(1);
  }
  return <section className="duality-lab" aria-label="Projection and bound investigation">
    <p className="duality-eyebrow">Investigate · a candidate and a certificate have different jobs</p><h3>Trap the best possible cost between two numbers</h3>
    <p>Start with the feasible square at (2,2), costing 5. The price λ=1 supplies the lower bound 1.5. Predict what happens if you move the square to the blue target: its cost becomes smaller, but does it remain feasible?</p>
    <div className="duality-controls"><Range label="Projection budget b" value={budget} setValue={setBudget} min={-2} max={10} step={0.5} /><Range label="Bound multiplier lambda" value={multiplier} setValue={setMultiplier} min={-2} max={12} step={0.5} /><Range label="Candidate x" value={candidate[0]} setValue={value => setCandidate([value, candidate[1]])} min={-3} max={8} step={0.5} /><Range label="Candidate y" value={candidate[1]} setValue={value => setCandidate([candidate[0], value])} min={-3} max={8} step={0.5} /></div>
    <div className="duality-buttons"><button type="button" onClick={() => {
        setCandidate([...state.optimum]);
        setMultiplier(state.optimalMultiplier);
      }}>Use exact optimal pair</button><button type="button" onClick={() => setCandidate([3, 4])}>Try unconstrained target</button><button type="button" onClick={reset}>Reset projection</button></div>
    <div className="duality-pair"><figure><BudgetPlane state={state} /><figcaption>Blue dot: target (3,4). Amber square: candidate. Green ring: exact feasible optimum. Pink cross: the minimizer of L for this λ. Green area: x+y≤b. The dotted radius reaches the optimum; the grey circle is the candidate's squared-distance level. Both axes use the same unit scale.</figcaption></figure>
      <Readout rows={[['Candidate objective f', number(state.objective)], ['Constraint x+y−b', `${number(state.constraint)} · ${state.primalFeasible ? 'feasible' : 'infeasible'}`], ['L at this candidate', number(state.lagrangian)], ['Global infimum q(λ)', `${number(state.dualValue)}${state.dualFeasible ? '' : ' · negative λ is not allowed'}`], ['Lagrangian minimizer z(λ)', `(${state.minimizer.map(number).join(', ')})`], ['Known optimum p*', number(state.optimalValue)], ['Certified upper − lower', number(state.certifiedGap)]]} />
    </div>
    <p role="status">{state.certifiedGap === null ? 'No certificate from this pair: a primal candidate must obey the budget, and an inequality multiplier must be nonnegative.' : `The true minimum is between ${number(state.dualValue)} and ${number(state.objective)}. This candidate is at most ${number(state.certifiedGap)} above the minimum.`}</p>
    <p>The algebraic difference f−q is {number(state.formalDifference)} = {number(state.minimizationGap)} from not minimizing L + {number(state.complementarityGap)} from λ times unused budget. Both terms are nonnegative only under the stated feasibility contracts. Rounded displays do not change those contracts.</p>
    <p className="duality-transfer">Set b=7 and use the optimal pair. The target is on the boundary, yet λ=0. Then tighten to b=6: how far must the point move, and why does the optimal price change?</p>
  </section>;
}
export function BoundLadderFigure() {
  return <figure className="duality-inline"><ol className="duality-ladder"><li><strong>q(1)=1.5</strong><span>Minimize L over every point.</span></li><li><strong>≤ L((2,2),1)=4</strong><span>Evaluate L at this feasible candidate.</span></li><li><strong>≤ f(2,2)=5</strong><span>Remove its negative budget term.</span></li></ol><figcaption>For this same problem p*=2. The number 4 is above 2: evaluating L at an arbitrary feasible point does not itself give a lower bound on the optimum. Only its global infimum q does.</figcaption></figure>;
}
function ScalarCurve({
  state
}) {
  const x = value => 44 + (value + 2.5) * 264 / 7;
  const top = Math.max((-2.5 - state.center) ** 2, (4.5 - state.center) ** 2);
  const y = value => 213 - value * 177 / top;
  const points = Array.from({
    length: 85
  }, (_, index) => {
    const value = -2.5 + 7 * index / 84;
    return [value, (value - state.center) ** 2];
  });
  return <Plot title="Scalar quadratic with feasible nonnegative half-line and chosen candidate"><rect x={x(0)} y="36" width={x(4.5) - x(0)} height="177" className="duality-region" /><line x1="44" x2="308" y1="213" y2="213" className="duality-axis" /><line x1={x(0)} x2={x(0)} y1="36" y2="213" className="duality-line green dashed" /><path d={polyline(points, x, y)} className="duality-line blue" /><circle cx={x(state.optimum)} cy={y(state.optimalValue)} r="8" className="duality-ring green" /><rect x={x(state.candidate) - 5} y={y(state.objective) - 5} width="10" height="10" className="duality-dot amber" /><text x="44" y="23" className="duality-label">cost (x−c)²</text><text x="53" y="51" className="duality-tick">top {number(top)}</text>{[-2, 0, 2, 4].map(value => <text key={value} x={x(value)} y="233" textAnchor="middle" className="duality-tick">{value}</text>)}<text x="308" y="255" textAnchor="end" className="duality-label">candidate coordinate x</text></Plot>;
}
function GradientBalance({
  state
}) {
  const values = [state.gradient, state.constraintContribution, state.stationarity];
  const bound = Math.max(2, ...values.map(Math.abs));
  const x = value => 170 + 122 * value / bound;
  return <Plot title="Signed objective derivative, constraint contribution and stationarity residual on one shared scale">
    <line x1="170" x2="170" y1="25" y2="216" className="duality-grid" />
    {values.map((value, index) => <g key={index}><text x="35" y={38 + index * 68} className="duality-label">{['objective slope 2(x−c)', 'constraint term −λ', 'sum: stationarity'][index]}</text><line x1="170" x2={x(value)} y1={57 + index * 68} y2={57 + index * 68} className={`duality-force ${['blue', 'amber', 'green'][index]}`} /><circle cx={x(value)} cy={57 + index * 68} r="4" className={`duality-dot ${['blue', 'amber', 'green'][index]}`} /><text x="307" y={61 + index * 68} textAnchor="end" className="duality-tick">{number(value)}</text></g>)}
    <text x="44" y="238" className="duality-tick">−{number(bound)}</text><text x="170" y="238" textAnchor="middle" className="duality-tick">0</text><text x="294" y="238" textAnchor="end" className="duality-tick">+{number(bound)}</text><text x="170" y="255" textAnchor="middle" className="duality-label">one common derivative scale</text>
  </Plot>;
}
export function ScalarKktLab() {
  const [center, setCenter] = useState(-1);
  const [candidate, setCandidate] = useState(0);
  const [multiplier, setMultiplier] = useState(2);
  const state = scalarKktState(center, candidate, multiplier);
  function preset(value) {
    setCenter(value);
    setCandidate(Math.max(value, 0));
    setMultiplier(2 * Math.max(-value, 0));
  }
  return <section className="duality-lab" aria-label="Four KKT conditions investigation"><p className="duality-eyebrow">Investigate · balance is only one condition</p><h3>Separate a boundary from its price</h3><p>Minimize (x−c)² subject to x≥0, written −x≤0. At c=−1 the objective prefers an infeasible negative point. Its positive derivative at x=0 is balanced by the constraint's contribution −λ. Predict which checks fail if you keep x=0 but change λ to zero.</p>
    <div className="duality-controls"><Range label="Quadratic center c" value={center} setValue={setCenter} min={-2} max={2} /><Range label="KKT candidate x" value={candidate} setValue={setCandidate} min={-2} max={4} /><Range label="KKT multiplier lambda" value={multiplier} setValue={setMultiplier} min={-2} max={6} /></div>
    <div className="duality-buttons"><button type="button" onClick={() => preset(-1)}>Active · positive price</button><button type="button" onClick={() => preset(0)}>Active · zero price</button><button type="button" onClick={() => preset(1)}>Inactive · zero price</button><button type="button" onClick={() => {
        setCandidate(state.optimum);
        setMultiplier(state.optimalMultiplier);
      }}>Solve current center</button></div>
    <div className="duality-pair"><figure><ScalarCurve state={state} /><figcaption>Green shading is feasible; green ring is the exact constrained minimum; amber square is your candidate. The y-axis rescales with the chosen center.</figcaption></figure><figure><GradientBalance state={state} /><figcaption>Signed bars share one scale. Cancellation means L is stationary here; it does not by itself establish feasibility or complementary slackness.</figcaption></figure></div>
    <div className="duality-checks">{[['Primal: x≥0', state.conditions.primal, `x=${number(candidate)}`], ['Dual: λ≥0', state.conditions.dual, `λ=${number(multiplier)}`], ['Stationarity: 2(x−c)−λ=0', state.conditions.stationarity, `residual=${number(state.stationarity)}`], ['Complementarity: λ(−x)=0', state.conditions.complementarity, `product=${number(state.complementaryProduct)}`]].map(([label, pass, detail]) => <div key={label}><strong>{pass ? '✓ Pass' : '× Fails'} · {label}</strong><span>{detail}</span></div>)}</div>
    <p role="status">{state.allConditions ? `All four checks hold exactly for these inputs. Convexity makes this a global certificate: x=${number(candidate)}, cost=${number(state.objective)}.` : 'At least one required check fails. This pair is not a KKT certificate, even if its derivative balance is zero.'} The constraint is {state.active ? 'active (zero slack)' : 'not active (its expression is nonzero)'}.</p>
    <p className="duality-transfer">Use the active-zero preset. Tighten the constraint to x≥δ for a small positive δ on paper. The new cost is δ²: zero first-order price does not mean every finite change is free. Then try c=−1, x=1, λ=4: why is perfect derivative balance insufficient?</p>
  </section>;
}
export function KktLogicFigure() {
  return <figure className="duality-inline"><div className="duality-logic"><div><strong>Given a KKT pair</strong><span>+ convex objective and inequalities; affine equalities</span><b>↓</b><strong>A global optimum</strong><p>No Slater test is needed to validate this supplied pair.</p></div><div><strong>Given an attained primal optimum</strong><span>+ appropriate constraint qualification, such as the stated Slater condition</span><b>↓</b><strong>Optimal multipliers and KKT</strong><p>This direction needs assumptions that ensure multipliers exist.</p></div></div><figcaption>Equal primal and dual optimal values are another statement. They do not, by themselves, say either optimum is attained. Keep values, points and existence separate.</figcaption></figure>;
}
export function SensitivityLab() {
  const [mode, setMode] = useState('quadratic');
  const [base, setBase] = useState(5);
  const [change, setChange] = useState(0.5);
  const [chosenPrice, setChosenPrice] = useState(0.5);
  const state = sensitivityState(mode, base, change, chosenPrice);
  const value = argument => mode === 'quadratic' ? Math.max(0, 7 - argument) ** 2 / 2 : Math.max(0, -argument);
  const points = Array.from({
    length: 81
  }, (_, index) => [base - 2 + index / 20, value(base - 2 + index / 20)]);
  const endpoints = [base - 2, base + 2].map(argument => [argument, state.originalValue - state.price * (argument - base)]);
  const low = Math.min(0, ...endpoints.map(point => point[1]));
  const high = Math.max(1, ...points.map(point => point[1]));
  const x = argument => 44 + (argument - base + 2) * 66;
  const y = cost => 213 - (cost - low) * 170 / (high - low);
  function chooseMode(value) {
    setMode(value);
    setBase(value === 'quadratic' ? 5 : 0);
    setChange(0.5);
    setChosenPrice(0.5);
  }
  return <section className="duality-lab" aria-label="Multiplier sensitivity investigation"><p className="duality-eyebrow">Investigate · supporting price or derivative?</p><h3>Change the resource and reoptimize</h3><p>The blue curve is the exact best cost after changing the right-hand side. The amber line is the lower estimate from an optimal multiplier at the original point. Predict whether a finite change must land on that line.</p>
    <div className="duality-controls"><label className="duality-field"><span>Value function</span><select aria-label="Value function" value={mode} onChange={event => chooseMode(event.target.value)}><option value="quadratic">Squared-distance budget</option><option value="kink">A value function with a corner</option></select></label><Range label="Original right-hand side" value={base} setValue={setBase} min={mode === 'quadratic' ? 0 : -2} max={mode === 'quadratic' ? 10 : 2} /><Range label="Right-hand-side change delta" value={change} setValue={setChange} min={-2} max={2} />{mode === 'kink' && base === 0 && <Range label="Optimal kink multiplier" value={chosenPrice} setValue={setChosenPrice} min={0} max={1} />}</div>
    <div className="duality-buttons"><button type="button" onClick={() => {
        chooseMode('quadratic');
        setBase(7);
      }}>Active constraint · zero price</button><button type="button" onClick={() => chooseMode('kink')}>Inspect the corner</button><button type="button" onClick={() => chooseMode('quadratic')}>Reset sensitivity</button></div>
    <div className="duality-pair"><figure><Plot title="Exact optimal-value curve, multiplier supporting line, original and changed right-hand sides"><line x1="44" x2="308" y1={y(0)} y2={y(0)} className="duality-axis" /><line x1={x(base)} x2={x(base)} y1="36" y2="213" className="duality-grid" /><path d={polyline(points, x, y)} className="duality-line blue" /><path d={polyline(endpoints, x, y)} className="duality-line amber dashed" /><circle cx={x(base)} cy={y(state.originalValue)} r="6" className="duality-dot green" /><circle cx={x(base + change)} cy={y(state.changedValue)} r="6" className="duality-dot blue" /><rect x={x(base + change) - 4} y={y(state.supportingValue) - 4} width="8" height="8" className="duality-dot amber" /><text x="44" y="23" className="duality-label">best cost after reoptimization</text><text x="51" y="54" className="duality-tick">top {number(high)}</text><text x="51" y="206" className="duality-tick">bottom {number(low)}</text>{[base - 2, base, base + 2].map(argument => <text key={argument} x={x(argument)} y="235" textAnchor="middle" className="duality-tick">{number(argument)}</text>)}<text x="308" y="255" textAnchor="end" className="duality-label">right-hand side</text></Plot><figcaption>Green: original optimum. Blue: changed optimum. Amber square: the supporting estimate at the changed right-hand side. Axes rescale around the chosen original point. In corner mode, p(u)=max(0,−u); only at u=0 can you choose any price in [0,1].</figcaption></figure><Readout rows={[['Optimal base multiplier', number(state.price)], ['Original → changed best cost', `${number(state.originalValue)} → ${number(state.changedValue)}`], ['Actual cost change', number(state.actualChange)], ['Price-based linear change −λδ', number(state.linearChange)], ['Changed cost − supporting line', number(state.supportingGap)], ['Derivative at the base', state.differentiable ? number(state.derivative) : `does not exist · left ${number(state.leftDerivative)}, right ${number(state.rightDerivative)}`]]} /></div><p role="status">{state.differentiable ? 'Here the unique slope is −λ. The line is a first-order approximation and a global supporting lower bound, not an exact prediction for every finite change.' : 'At the corner, every selected λ is an optimal dual price and gives a valid supporting line. Different left and right slopes mean there is no single derivative.'}</p><p className="duality-transfer">At budget 5, compare δ=+.5 and δ=−.5. Why are the cost changes not opposites? At budget 7, tighten by .5: explain a positive cost increase despite a zero tangent slope.</p></section>;
}
function ResourcePicture({
  frame
}) {
  const width = value => 264 * value / 8;
  return <Plot title="Local and repaired allocations on the same resource scale, with a shared budget marker">
    {[['Local choices', frame.allocation], ['Feasible priority repair', frame.repaired]].map(([label, allocation], index) => <g key={label}><text x="44" y={36 + index * 93} className="duality-label">{label}</text><rect x="44" y={52 + index * 93} width={width(allocation[0])} height="25" className="duality-dot blue" /><rect x={44 + width(allocation[0])} y={52 + index * 93} width={width(allocation[1])} height="25" className="duality-dot amber" /><line x1={44 + width(frame.budget)} x2={44 + width(frame.budget)} y1={45 + index * 93} y2={86 + index * 93} className="duality-line green" /><text x="44" y={99 + index * 93} className="duality-tick">x₁={number(allocation[0])} · x₂={number(allocation[1])}</text></g>)}<line x1="44" x2="308" y1="215" y2="215" className="duality-axis" />{[0, 2, 4, 6, 8].map(value => <text key={value} x={44 + width(value)} y="235" textAnchor="middle" className="duality-tick">{value}</text>)}<text x="308" y="255" textAnchor="end" className="duality-label">shared resource units</text>
  </Plot>;
}
function PriceTrace({
  state,
  step
}) {
  const top = Math.max(1, state.optimumPrice, ...state.frames.map(frame => frame.price)) * 1.1;
  const x = index => 44 + 264 * index / state.updates;
  const y = price => 213 - 175 * price / top;
  return <Plot title="Selected prefix of projected price updates and an optimal reference price"><line x1="44" x2="308" y1="213" y2="213" className="duality-axis" /><line x1="44" x2="308" y1={y(state.optimumPrice)} y2={y(state.optimumPrice)} className="duality-line green dashed" /><path d={polyline(state.frames.slice(0, step + 1).map(frame => [frame.step, frame.price]), x, y)} className="duality-line amber" />{state.frames.slice(0, step + 1).map(frame => <circle key={frame.step} cx={x(frame.step)} cy={y(frame.price)} r={frame.step === step ? 5 : 2.5} className="duality-dot amber" />)}<text x="44" y="23" className="duality-label">price λ</text><text x="51" y="53" className="duality-tick">top {number(top)}</text><text x="44" y="235" className="duality-tick">0</text><text x="308" y="235" textAnchor="end" className="duality-tick">{state.updates}</text><text x="308" y="255" textAnchor="end" className="duality-label">price update index</text></Plot>;
}
export function ResourceDualAscentLab() {
  const [budget, setBudget] = useState(5);
  const [rate, setRate] = useState(1);
  const [initialPrice, setInitialPrice] = useState(0);
  const [step, setStep] = useState(0);
  const state = resourceDualAscentState(budget, rate, initialPrice, 20);
  const frame = state.frames[step];
  function change(setter, value) {
    setter(value);
    setStep(0);
  }
  function reset() {
    setBudget(5);
    setRate(1);
    setInitialPrice(0);
    setStep(0);
  }
  return <section className="duality-lab" aria-label="Resource price iteration investigation"><p className="duality-eyebrow">Investigate · local decisions, shared constraint</p><h3>Raise the price when demand exceeds the budget</h3><p>Two tasks choose their nonnegative allocations independently at the current price. Their costs are (x₁−3)² and 2(x₂−4)². The price update uses their total demand minus the budget. Predict the first price from λ=0, b=5, α=1 before stepping.</p>
    <div className="duality-controls"><Range label="Shared resource budget" value={budget} setValue={value => change(setBudget, value)} min={0} max={8} /><Range label="Price step size alpha" value={rate} setValue={value => change(setRate, value)} min={0} max={4} /><Range label="Initial resource price" value={initialPrice} setValue={value => change(setInitialPrice, value)} min={0} max={20} step={0.5} /></div>
    <p className="duality-note">Changing a parameter restarts the trace at update 0. Each state solves the two local minimizations exactly; these are deterministic arithmetic models, not measured distributed timings.</p>
    <div className="duality-buttons"><button type="button" disabled={step === 0} onClick={() => setStep(step - 1)}>Previous price</button><output>State {step} of 20</output><button type="button" disabled={step === 20} onClick={() => setStep(step + 1)}>Next price</button><button type="button" onClick={() => setStep(20)}>Show state 20</button><button type="button" onClick={reset}>Reset resource</button></div>
    <div className="duality-pair"><figure><ResourcePicture frame={frame} /><figcaption>Blue is task 1; amber is task 2. Green marks the shared budget. The lower bar keeps task 1 up to the budget, then gives task 2 what remains. That feasible repair is an upper-bound witness, not the optimal projection.</figcaption></figure><figure><PriceTrace state={state} step={step} /><figcaption>Amber: prices through the selected state. Green: one optimal price, {number(state.optimumPrice)}. At b=0 every price at least 16 is optimal; the reference uses 16. A large step can produce a repeating cycle.</figcaption></figure></div>
    <Readout rows={[['Current → next projected price', `${number(frame.price)} → max(0, ${number(frame.price)} + ${number(rate)} × ${number(frame.violation)}) = ${number(frame.nextPrice)}`], ['Local total − budget', `${number(frame.violation)} · ${frame.primalFeasible ? 'feasible' : 'infeasible'}`], ['Dual lower bound q', number(frame.dualValue)], ['Feasible repair upper bound', number(frame.repairedObjective)], ['Repair certificate gap', number(frame.certificateGap)], ['Exact optimum cost', number(state.optimum.objective)]]} /><p role="status">The current nonnegative price supplies a valid lower bound. {frame.primalFeasible ? 'Its local allocation also obeys the shared budget.' : 'Its local allocation violates the shared budget; do not report its cost as a primal upper bound.'} The repaired allocation supplies the displayed upper bound.</p><p>The gap is evaluated from nonnegative local quadratic differences and price times unused repaired budget. This avoids cancellation when rounded upper and lower costs look equal. It is a floating-point evaluation of the mathematical certificate; a displayed zero is not a proof of exact convergence.</p><p className="duality-transfer">Compare α=1 with α=3 from price 0 and budget 5. Trace the first three prices by hand. Then set α=0: why does the bound remain valid even when the algorithm makes no progress?</p>
  </section>;
}
