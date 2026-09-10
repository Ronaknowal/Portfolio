import { useId, useMemo, useState } from 'react';
import { LessonTable } from './LessonElements.jsx';
import { CUMULATIVE_SCENARIOS, cumulativeTransport, sinkhornComparison, sinkhornScaling, stabilityComparison, twoLocationTransport } from '../../data/optimal-transport-models.js';
import './optimal-transport-labs.css';
const gold = '#e6b75f',
  blue = '#91bada',
  green = '#99ccb3';
const number = (value, digits = 4) => value !== 0 && Math.abs(value) < 0.0001 ? value.toExponential(2) : Number(value.toFixed(digits)).toString();
function Slider({
  label,
  value,
  setValue,
  min,
  max,
  step = 1
}) {
  const id = useId();
  return <label className="transport-control" htmlFor={id}><span>{label}: <strong>{number(value)}</strong></span>
    <input id={id} aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => setValue(Number(event.target.value))} /></label>;
}
function Investigation({
  id,
  title,
  prediction,
  reset,
  children
}) {
  const heading = useId();
  return <section className="transport-lab lesson-lab" data-transport-lab={id} aria-labelledby={heading}>
    <h3 id={heading}>{title}</h3><p><strong>Predict first.</strong> {prediction}</p>
    {children}<button type="button" className="transport-reset" onClick={reset}>Reset investigation</button>
  </section>;
}
function Metrics({
  values
}) {
  return <dl className="transport-metrics">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Flow({
  plan,
  source,
  target,
  selected = 0
}) {
  const heights = [78, 220];
  return <figure className="transport-figure"><figcaption>Link width encodes transported mass. Left and right are source/target lists; their screen separation is not the ground distance.</figcaption>
    <svg viewBox="0 0 320 285" role="img" aria-label="Four source-to-target flows with widths proportional to mass">
      {plan.flatMap((row, i) => row.map((mass, j) => <path key={i + '-' + j} d={`M66,${heights[i]} C140,${heights[i]} 180,${heights[j]} 254,${heights[j]}`} fill="none" stroke={i * 2 + j === selected ? gold : '#7299b3'} strokeOpacity={i * 2 + j === selected ? 1 : 0.45} strokeWidth={mass * 32} />))}
      <text x="45" y="24" textAnchor="middle">Source</text><text x="275" y="24" textAnchor="middle">Target</text>
      {source.map((mass, i) => <g key={'source-' + i}><circle cx="45" cy={heights[i]} r="18" fill="#182c36" stroke={blue} strokeWidth="2" /><text x="45" y={heights[i] + 6} textAnchor="middle">{i === 0 ? '0' : '2'}</text><text x="45" y={heights[i] + 40} textAnchor="middle">{number(mass, 2)}</text></g>)}
      {target.map((mass, j) => <g key={'target-' + j}><circle cx="275" cy={heights[j]} r="18" fill="#302617" stroke={gold} strokeWidth="2" /><text x="275" y={heights[j] + 6} textAnchor="middle">{j}</text><text x="275" y={heights[j] + 40} textAnchor="middle">{number(mass, 2)}</text></g>)}
    </svg><p className="transport-caption">Circle labels: locations. Values below: required probability mass. Select a matrix cell to highlight its route.</p>
  </figure>;
}
function MassMatrix({
  state,
  selected,
  setSelected
}) {
  return <div className="transport-table" role="region" tabIndex={0} aria-label="Transport matrix and marginal checks">
    <table><caption>Each row spends its source mass; each column receives its target mass.</caption><thead><tr><th scope="col">From / to</th><th scope="col">y = 0</th><th scope="col">y = 1</th><th scope="col">Row / need</th></tr></thead>
      <tbody>{state.plan.map((row, i) => <tr key={i}><th scope="row">x = {i * 2}</th>{row.map((mass, j) => <td key={j}>{setSelected ? <button type="button" aria-label={`Inspect route ${i * 2} to ${j}`} aria-pressed={selected === i * 2 + j} onClick={() => setSelected(i * 2 + j)}>{number(mass)}</button> : number(mass)}</td>)}<td>{number(state.rowSums[i])} / {number(state.source[i])}</td></tr>)}
        <tr><th scope="row">Column / need</th>{state.columnSums.map((mass, j) => <td key={j}>{number(mass)} / {number(state.target[j])}</td>)}<td>Mass = {number(state.rowSums.reduce((a, b) => a + b, 0))}</td></tr>
      </tbody></table>
  </div>;
}
export function TransportPlanLab() {
  const [sourceFirst, setSourceFirst] = useState(0.5),
    [targetFirst, setTargetFirst] = useState(0.5);
  const [fraction, setFraction] = useState(1),
    [costKind, setCostKind] = useState('distance'),
    [selected, setSelected] = useState(0);
  const state = useMemo(() => twoLocationTransport({
    sourceFirst,
    targetFirst,
    fraction,
    costKind
  }), [sourceFirst, targetFirst, fraction, costKind]);
  const i = Math.floor(selected / 2),
    j = selected % 2;
  return <Investigation id="mass-ledger" title="Move mass without losing the ledger" prediction="If the target needs more mass at 0 than the source has there, can the source at 2 send some mass to both targets?" reset={() => {
    setSourceFirst(0.5);
    setTargetFirst(0.5);
    setFraction(1);
    setCostKind('distance');
    setSelected(0);
  }}>
    <div className="transport-controls"><Slider label="Source mass at 0" value={sourceFirst} setValue={setSourceFirst} min={0} max={1} step={0.05} /><Slider label="Target mass at 0" value={targetFirst} setValue={setTargetFirst} min={0} max={1} step={0.05} />
      <Slider label="Position in feasible interval" value={fraction} setValue={setFraction} min={0} max={1} step={0.05} />
      <label className="transport-control">Cost rule<select aria-label="Cost rule" value={costKind} onChange={event => setCostKind(event.target.value)}><option value="distance">Ordinary distance</option><option value="squared">Squared distance</option><option value="indifferent">Declared tie-producing costs</option><option value="changed">Declared preference reversal</option></select></label></div>
    <Flow plan={state.plan} source={state.source} target={state.target} selected={selected} /><MassMatrix state={state} selected={selected} setSelected={setSelected} />
    <p className="transport-observation">Selected route {i * 2} → {j}: <strong>{number(state.plan[i][j])} mass × {state.costs[i][j]} cost = {number(state.plan[i][j] * state.costs[i][j])}</strong>. All four contributions form the total.</p>
    <Metrics values={[['Feasible π₀₀ interval', `[${number(state.lower)}, ${number(state.upper)}]`], ['Current π₀₀', number(state.entry)], ['Current transport cost', number(state.cost)], ['Minimum cost', number(state.optimumCost)], ['Cost slope as π₀₀ increases', state.slope], ['Largest marginal residual', number(state.residual)]]} />
    <p>{state.slope === 0 ? 'Here every feasible plan has the same cost: the marginals determine the entire objective. An optimum need not be unique.' : `The objective is linear in π₀₀. Its slope is ${state.slope}, so the ${state.slope < 0 ? 'upper' : 'lower'} feasible endpoint minimizes it.`} {state.lower === state.upper && 'This weight choice permits only one plan.'}</p>
    <button type="button" onClick={() => setFraction(state.slope <= 0 ? 1 : 0)}>Use a minimum-cost plan</button>
    <LessonTable caption="Declared cost for every route" headers={['From / to', 'Target 0', 'Target 1']} rows={state.costs.map((row, i) => ['Source ' + i * 2, ...row])} />
    <p className="transport-caption">The two declared-cost options deliberately override physical distance. They demonstrate cost dependence, not a new metric guarantee.</p>
  </Investigation>;
}
export function TransportGeometryFigure() {
  return <figure className="transport-inline"><figcaption><strong>A nearby shift is still nearby when the supports do not overlap</strong></figcaption>
    <svg viewBox="0 0 320 200" role="img" aria-label="One probability atom moves from 0 to 0.2 or 2 on the same coordinate axis">
      {[60, 140].map(y => <line key={y} x1="40" x2="280" y1={y} y2={y} stroke="#667986" />)}
      <line x1="40" x2="64" y1="60" y2="60" stroke={gold} strokeWidth="5" /><line x1="40" x2="280" y1="140" y2="140" stroke={gold} strokeWidth="5" />
      {[60, 140].map(y => <circle key={y} cx="40" cy={y} r="8" fill={blue} />)}
      <circle cx="64" cy="60" r="8" fill={gold} /><circle cx="280" cy="140" r="8" fill={gold} />
      <text x="40" y="30">Near: W₁ =0.2</text><text x="40" y="110">Far: W₁ =2</text><text x="40" y="180" textAnchor="middle">0</text><text x="160" y="180" textAnchor="middle">1</text><text x="280" y="180" textAnchor="middle">2</text>
    </svg><p>Every dot represents mass 1, not a finite-width density spike. For either nonzero move, total variation is 1 and the directed KL from the original atom is infinite. W₁ retains the size of the displacement.</p>
  </figure>;
}
export function TransportDualLab() {
  const [potential, setPotential] = useState(1);
  const state = useMemo(() => twoLocationTransport({
    dualPosition: potential
  }), [potential]);
  return <Investigation id="dual-certificate" title="Prove that no cheaper plan exists" prediction="The shown plan costs 0.5. Can a feasible set of prices give a lower bound of exactly 0.5?" reset={() => setPotential(1)}>
    <p>Keep f₀ =0 and adjust f₁. Each gⱼ is chosen as the smaller of C₀ⱼ and C₁ⱼ − f₁, so no pair of prices exceeds that route's cost.</p>
    <Slider label="Source price f1" value={potential} setValue={setPotential} min={-1} max={3} step={0.1} />
    <LessonTable caption="Price, cost and remaining slack" headers={['Route', 'fᵢ + gⱼ', 'Cᵢⱼ', 'Slack', 'Plan mass']} rows={state.costs.flatMap((row, i) => row.map((cost, j) => [`${i * 2} → ${j}`, number(state.dual.sourcePotential[i] + state.dual.targetPotential[j]), cost, number(state.dual.slack[i][j]), state.plan[i][j]]))} />
    <Metrics values={[['f₀, f₁', state.dual.sourcePotential.map(value => number(value)).join(', ')], ['g₀, g₁', state.dual.targetPotential.map(value => number(value)).join(', ')], ['Plan cost: upper bound', number(state.cost)], ['Price value: lower bound', number(state.dual.value)], ['Primal − dual gap', number(state.dualGap)]]} />
    <p className="transport-observation">{state.dualGap < 1e-12 ? 'The bounds meet. Both occupied routes have zero slack, so the cost 0.5 is certified optimal.' : 'The prices are valid but the lower bound is loose. A positive gap here says this certificate is unfinished; it does not prove the shown plan is suboptimal.'}</p>
    <button type="button" onClick={() => setPotential(state.optimalCertificate.sourcePotential[1])}>Use a tight certificate</button>
  </Investigation>;
}
export function CumulativeTransportLab() {
  const [scenario, setScenario] = useState('nearby'),
    [spacing, setSpacing] = useState(1),
    [selected, setSelected] = useState(0);
  const state = useMemo(() => cumulativeTransport(scenario, spacing), [scenario, spacing]);
  const x = index => 42 + index * 78,
    y = value => 185 - value * 145;
  const curve = field => state.gaps.map((gap, index) => `${index ? 'L' : 'M'}${x(index)},${y(gap[field])} L${x(index + 1)},${y(gap[field])}`).join(' ') + ` L${x(3)},${y(1)}`;
  return <Investigation id="cumulative-crossings" title="Count the mass that must cross each gap" prediction="Doubling every spacing changes no probabilities. What should it do to W₁?" reset={() => {
    setScenario('nearby');
    setSpacing(1);
    setSelected(0);
  }}>
    <div className="transport-controls"><label className="transport-control">Target histogram<select aria-label="Target histogram" value={scenario} onChange={event => setScenario(event.target.value)}>{Object.entries(CUMULATIVE_SCENARIOS).map(([key, value]) => <option key={key} value={key}>{value.title}</option>)}</select></label><Slider label="Distance between bins" value={spacing} setValue={setSpacing} min={0.5} max={3} step={0.5} /></div>
    <figure className="transport-figure"><figcaption>Cumulative probability up to each location. Shaded gap area = required crossing mass × gap length.</figcaption>
      <svg viewBox="0 0 320 250" role="img" aria-label="Source and target cumulative distributions with exact rectangular difference areas">
        {[0, 0.5, 1].map(value => <g key={value}><line x1="42" x2="276" y1={y(value)} y2={y(value)} stroke="#405663" /><text x="35" y={y(value) + 6} textAnchor="end">{value}</text></g>)}
        {state.gaps.map((gap, index) => <rect key={index} x={x(index)} y={y(Math.max(gap.sourceCdf, gap.targetCdf))} width="78" height={Math.abs(y(gap.sourceCdf) - y(gap.targetCdf))} fill={gold} opacity={index === selected ? 0.45 : 0.16} />)}
        <path d={curve('sourceCdf')} fill="none" stroke={blue} strokeWidth="3" /><path d={curve('targetCdf')} fill="none" stroke={gold} strokeWidth="3" strokeDasharray="7 4" />
        {state.locations.map((position, index) => <text key={index} x={x(index)} y="211" textAnchor="middle">{number(position)}</text>)}<text x="160" y="238" textAnchor="middle">Location</text>
      </svg><p className="transport-caption">Solid blue: source. Dashed gold: target. At the last location, both cumulative probabilities become 1.</p>
    </figure>
    <div className="transport-buttons" aria-label="Select a gap">{state.gaps.map((gap, index) => <button key={index} type="button" aria-pressed={index === selected} onClick={() => setSelected(index)}>{number(gap.left)} → {number(gap.right)}</button>)}</div>
    <p className="transport-observation">Across this gap, net mass <strong>{number(Math.abs(state.gaps[selected].difference))}</strong> must move {state.gaps[selected].difference > 0 ? 'right' : state.gaps[selected].difference < 0 ? 'left' : 'in neither direction'}. Cost contribution = <strong>{number(state.gaps[selected].area)}</strong>.</p>
    <Metrics values={[['Total area / W₁', number(state.distance)], ['Sorted plan cost', number(state.cost)], ['Plan marginal residual', number(state.residual)]]} />
    <LessonTable caption="Probability mass and exact gap contributions" headers={['Location', 'Source mass', 'Target mass', 'Following gap area']} rows={state.locations.map((position, index) => [number(position), state.source[index], state.target[index], index < 3 ? number(state.gaps[index].area) : 'No following gap'])} />
  </Investigation>;
}
export function SinkhornScalingLab() {
  const [epsilon, setEpsilon] = useState(0.5),
    [step, setStep] = useState(0);
  const state = useMemo(() => sinkhornScaling({
    source: [0.6, 0.4],
    target: [0.3, 0.7],
    epsilon,
    traceSteps: 40
  }), [epsilon]);
  const current = state.trace[Math.min(step, state.trace.length - 1)];
  const matrixState = {
    ...current,
    source: state.source,
    target: state.target
  };
  return <Investigation id="alternating-scaling" title="Repair one marginal, then inspect the other" prediction="After correcting the rows once, are the columns already correct?" reset={() => {
    setEpsilon(0.5);
    setStep(0);
  }}>
    <Slider label="Scaling entropy epsilon" value={epsilon} setValue={value => {
      setEpsilon(value);
      setStep(0);
    }} min={0.1} max={2} step={0.1} />
    <p>This investigation uses source weights (.6,.4) and target weights (.3,.7), so the two marginal corrections do real work. The program below uses the simpler equal-weight example introduced earlier.</p>
    <p><strong>Half-step {step}: {current.phase}.</strong> The initial K is an unnormalized positive preference matrix. A half-step corrects all rows or all columns; it is not one complete sweep.</p>
    <MassMatrix state={matrixState} />
    <div className="transport-buttons"><button type="button" onClick={() => setStep(value => value - 1)} disabled={step === 0}>Previous correction</button><button type="button" onClick={() => setStep(value => value + 1)} disabled={step === state.trace.length - 1}>Next correction</button><button type="button" onClick={() => setStep(state.trace.length - 1)}>Show 40 corrections</button></div>
    <Metrics values={[['Current marginal residual', number(current.residual)], ['Current linear term', number(current.cost)], ['Reference final linear term', number(state.cost)], ['Reference full sweeps', state.iterations]]} />
    <p className="transport-observation">{current.residual > 1e-8 ? 'This iterate has not met both marginals. Its current cost is not a feasible primal upper bound.' : 'Both marginals agree to the displayed numerical tolerance. The off-diagonal mass is a deliberate regularization effect, not a residual to remove.'}</p>
    <p>The stored first 40 corrections use stable log-domain calculations equivalent to positive row/column scaling in exact arithmetic. The final reference continues until residual≤10⁻¹¹ or 1,000 sweeps; {state.converged ? 'it met that criterion.' : 'it reached the cap without meeting it.'}</p>
  </Investigation>;
}
export function SinkhornStabilityLab() {
  const [offset, setOffset] = useState(1000);
  const state = useMemo(() => stabilityComparison(0.5, offset), [offset]);
  return <Investigation id="log-domain" title="Change the numerical scale without changing the best plan" prediction="If every route costs an extra 1,000 units per unit mass, how much extra does every feasible probability plan cost?" reset={() => setOffset(1000)}>
    <Slider label="Cost added to every route" value={offset} setValue={setOffset} min={0} max={1000} step={100} />
    <LessonTable caption="Ordinary exponential kernel versus the stable plan" headers={['Route', 'Cost', 'exp(−C / 0.5)', 'Stable plan']} rows={state.costs.flatMap((row, i) => row.map((cost, j) => [`${i * 2} → ${j}`, cost, number(state.kernel[i][j]), number(state.stable.plan[i][j])]))} />
    <Metrics values={[['Kernel entries rounded to zero', state.vanishedEntries + ' of 4'], ['Stable marginal residual', number(state.stable.residual)], ['Plan difference from offset0', number(state.maximumPlanDifference)], ['Linear cost increase', number(state.stable.cost - state.reference.cost)]]} />
    <p className="transport-observation">{state.vanishedEntries === 4 ? 'Ordinary scaling would divide by a zero row sum. The mathematical kernel is positive; floating point lost it. Log-domain updates retain the relative preferences.' : 'The exponential kernel is still representable. Increasing the common offset eventually erases its tiny values in ordinary floating point.'}</p>
    <p>Every feasible plan has total mass 1, so the added objective term is exactly the offset. The optimal plan is unchanged. Stable arithmetic fixes this representation failure; it does not promise rapid convergence for every small ε.</p>
  </Investigation>;
}
const comparisonCases = {
  contracted: {
    label: 'Source 0,2 → target 0,1',
    shift: 0,
    scale: 0.5
  },
  shifted: {
    label: 'Source 0,2 → target 1,3',
    shift: 1,
    scale: 1
  },
  same: {
    label: 'Source 0,2 → itself',
    shift: 0,
    scale: 1
  }
};
export function SinkhornBiasLab() {
  const [epsilon, setEpsilon] = useState(1),
    [comparison, setComparison] = useState('contracted');
  const state = useMemo(() => sinkhornComparison(epsilon, comparisonCases[comparison].shift, comparisonCases[comparison].scale), [epsilon, comparison]);
  return <Investigation id="objective-bias" title="Give each computed number its correct name" prediction="If you compare a distribution with itself, which of the three quantities below must be zero?" reset={() => {
    setEpsilon(1);
    setComparison('contracted');
  }}>
    <div className="transport-controls"><Slider label="Comparison entropy epsilon" value={epsilon} setValue={setEpsilon} min={0.1} max={4} step={0.1} /><label className="transport-control">Comparison<select aria-label="Comparison" value={comparison} onChange={event => setComparison(event.target.value)}>{Object.entries(comparisonCases).map(([key, value]) => <option key={key} value={key}>{value.label}</option>)}</select></label></div>
    <p>Each atom has mass½. Costs are squared distances. The table uses the independently derived two-by-two closed form, including the entropy term; the iterative result below is checked against it.</p>
    <LessonTable caption="Same epsilon and entropy convention for all three solves" headers={['Pair', 'Linear cost', 'ε Σπ(logπ −1)', 'Fε']} rows={[['Cross', state.cross], ['Source self', state.sourceSelf], ['Target self', state.targetSelf]].map(([label, result]) => [label, number(result.analytic.cost), number(epsilon * result.analytic.negativeEntropy), number(result.analytic.objective)])} />
    <Metrics values={[['Debiased Sε (closed form)', number(state.analyticDivergence)], ['Exact unregularized W₂²', number(state.exactSquaredDistance)], ['W₂ (after square root)', number(state.exactDistance)], ['Iterative candidate Sε', number(state.divergence)], ['Worst iterative residual', number(Math.max(state.cross.residual, state.sourceSelf.residual, state.targetSelf.residual))], ['Cross iteration count', state.cross.iterations]]} />
    <p className="transport-observation">{state.converged ? 'All three iterative solves met residual≤10⁻¹¹. That controls this numerical feasibility error, not sampling error or every modelling choice.' : 'At least one iterative solve reached 5,000 sweeps before meeting residual≤10⁻¹¹. Its candidate Sε is not certified converged. The closed-form result remains the reference.'}</p>
    <p>{comparison === 'same' ? 'The debiased expression cancels exactly in self-comparison. The ordinary regularized plan still spreads mass, and its entropy objective is not zero.' : comparison === 'shifted' ? 'This is a special translation of the same equally weighted shape: the debiased value equals the squared shift for every ε. Do not generalize that equality to different shapes.' : 'The distributions have different spreads. Removing self-bias does not generally make Sε equal W₂² at finite ε.'} The raw objective can be negative; it is not a distance.</p>
  </Investigation>;
}
export function BarycentricProjectionFigure() {
  return <figure className="transport-inline"><figcaption><strong>A valid split plan and its average destination describe different laws</strong></figcaption>
    <svg viewBox="0 0 320 245" role="img" aria-label="One source atom splits equally to minus1 and 1 but its barycentric projection stays at 0">
      <path d="M160,52 L60,135 M160,52 L260,135" stroke={gold} fill="none" strokeWidth="4" /><path d="M160,52 L160,200" stroke={blue} strokeWidth="3" strokeDasharray="6 5" />
      <circle cx="160" cy="40" r="12" fill={green} /><circle cx="60" cy="145" r="12" fill={gold} /><circle cx="260" cy="145" r="12" fill={gold} /><circle cx="160" cy="205" r="12" fill={blue} />
      <text x="160" y="20" textAnchor="middle">Source 0: mass 1</text><text x="48" y="100">½</text><text x="265" y="100">½</text><text x="60" y="178" textAnchor="middle">Target −1</text><text x="260" y="178" textAnchor="middle">Target+1</text><text x="160" y="239" textAnchor="middle">Mean destination 0</text>
    </svg><p>The solid plan has the required target marginal½δ₋₁ +½δ₁. The dashed conditional-mean map produces δ₀ instead. Averaging destinations discards the split; it does not preserve the target distribution.</p>
  </figure>;
}
export function DisplacementFigure() {
  return <figure className="transport-inline"><figcaption><strong>At halfway: move the atom or mix the endpoints?</strong></figcaption>
    <div className="transport-interpolation"><div><strong>Displacement</strong><p>δ₀ → δ₁ → δ₂</p><span>Halfway: all mass at 1</span></div><div><strong>Mixture</strong><p>δ₀ → ½δ₀ +½δ₂ → δ₂</p><span>Halfway: mass at 0 and 2</span></div></div>
    <p>Both paths have the same endpoints and mean1 at halfway. Only the first follows the constant-speed Wasserstein path for these single atoms; a mean alone cannot identify a distribution.</p>
  </figure>;
}
