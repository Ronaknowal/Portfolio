import { useId, useMemo, useState } from 'react';
import { LessonTable } from './LessonElements.jsx';
import { MATROID_PRESETS, KNAPSACK_ITEMS, ASSIGNMENT_COSTS, VERTEX_COVER_PRESETS, matroidExchangeState, knapsackSearchTrace, assignmentTrace, setCoverTrace, vertexCoverState, scaledKnapsackState } from '../../data/combinatorial-optimization-models.js';
import './combinatorial-optimization-labs.css';
const display = value => Number.isFinite(value) ? Number(value.toFixed(3)).toString() : 'unreachable';
const selectedIndices = (mask, size) => Array.from({
  length: size
}, (_, index) => index).filter(index => mask & 1 << index);
const letter = index => String.fromCharCode(65 + index);
function Investigation({
  id,
  title,
  guidance,
  children,
  reset
}) {
  const heading = useId();
  return <section className="combinatorial-lab lesson-lab" data-combinatorial-lab={id} aria-labelledby={heading}>
    <h3 id={heading}>{title}</h3><p>{guidance}</p>
    {children}<button type="button" onClick={reset}>Reset investigation</button>
  </section>;
}
function Stepper({
  value,
  maximum,
  setValue
}) {
  return <div className="combinatorial-stepper">
    <button type="button" onClick={() => setValue(value - 1)} disabled={value === 0}>Previous state</button>
    <output aria-live="polite">State {value + 1} of {maximum + 1}</output>
    <button type="button" onClick={() => setValue(value + 1)} disabled={value === maximum}>Next state</button>
  </div>;
}
function Metrics({
  values
}) {
  return <dl className="combinatorial-metrics">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Slider({
  label,
  value,
  onChange,
  min = 0,
  max = 10
}) {
  const id = useId();
  return <label htmlFor={id}><span>{label}: <strong>{value}</strong></span><input id={id} aria-label={label} type="range" min={min} max={max} step="1" value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Graph({
  count,
  edges,
  selected = [],
  labels = null,
  edgeLabels = null,
  caption
}) {
  const id = useId();
  const positions = Array.from({
    length: count
  }, (_, index) => {
    const angle = -Math.PI / 2 + index * 2 * Math.PI / count;
    return [150 + Math.cos(angle) * 105, 130 + Math.sin(angle) * 78];
  });
  return <figure className="combinatorial-graph"><svg viewBox="0 0 300 270" role="img" aria-labelledby={id}>
    <title id={id}>{caption}</title>
    {edges.map(([a, b], index) => <g key={index}><line x1={positions[a][0]} y1={positions[a][1]} x2={positions[b][0]} y2={positions[b][1]} stroke={selected.includes(index) ? '#e9bb67' : '#637484'} strokeWidth={selected.includes(index) ? 4 : 2} />
      {edgeLabels && (() => {
          const fraction = count === 4 && Math.abs(a - b) === 2 ? .32 : .5;
          const x = positions[a][0] + fraction * (positions[b][0] - positions[a][0]);
          const y = positions[a][1] + fraction * (positions[b][1] - positions[a][1]);
          return <g><rect x={x - 16} y={y - 12} width="32" height="24" fill="#151a20" /><text x={x} y={y + 5} textAnchor="middle">{edgeLabels[index]}</text></g>;
        })()}
    </g>)}
    {positions.map(([x, y], index) => <g key={index}><circle cx={x} cy={y} r="17" fill="#151a20" stroke="#96c3e8" strokeWidth="2" /><text x={x} y={y + 4} textAnchor="middle">{letter(index)}</text>
      {labels && <text x={x} y={y + (y < 80 ? -25 : 33)} textAnchor="middle">{labels[index]}</text>}</g>)}
  </svg><figcaption>{caption}</figcaption></figure>;
}
export function AssignmentMeaningFigure() {
  return <figure className="combinatorial-inline"><figcaption>Two legal assignments, radically different total costs.</figcaption>
    <div className="combinatorial-comparison"><div><strong>Same-column pairs are illegal</strong><p>Worker A → job 1<br />Worker B → job 1</p><p>Job 1 is occupied twice; job 2 is empty.</p></div>
      <div><strong>Legal, cost 2</strong><p>Worker A → job 1: 1<br />Worker B → job 2: 1</p><p>Two jobs, two different workers.</p></div>
      <div><strong>Legal, cost 200</strong><p>Worker A → job 2: 100<br />Worker B → job 1: 100</p><p>The same cardinality says nothing about cost.</p></div></div>
    <p>Costs are supplied units for this example. These are alternative plans, not measured runtimes.</p>
  </figure>;
}
export function BoundMeaningFigure() {
  return <figure className="combinatorial-inline"><figcaption>Read an optimization interval in the correct direction.</figcaption>
    <div className="combinatorial-bound"><span>Proved lower limit<strong>80</strong></span><span className="combinatorial-bound-range">OPT can lie here</span><span>Feasible cost<strong>100</strong></span></div>
    <p>For this minimization instance, the gap is at most 20. Since 80 is positive, 100 / 80 = 1.25 certifies a factor of at most 1.25 on this instance. The example is a stated certificate scenario; it is not a benchmark or a theorem for an algorithm.</p>
  </figure>;
}
export function MatroidExchangeLab() {
  const [kind, setKind] = useState('intervals');
  const [weight, setWeight] = useState(3);
  const [smaller, setSmaller] = useState(1);
  const [larger, setLarger] = useState(6);
  const weights = useMemo(() => MATROID_PRESETS[kind].weights.map((value, index) => index === 0 ? weight : value), [kind, weight]);
  const state = useMemo(() => matroidExchangeState({
    kind,
    weights,
    smaller,
    larger
  }), [kind, weights, smaller, larger]);
  const nameSet = mask => selectedIndices(mask, state.preset.names.length).map(index => state.preset.names[index]).join(', ') || '∅';
  const changeKind = value => {
    setKind(value);
    setWeight(MATROID_PRESETS[value].weights[0]);
    setSmaller(1);
    setLarger(6);
  };
  return <Investigation id="matroid" title="Can the smaller feasible set grow?" guidance={kind === 'intervals' ? 'Can either short interval be added to the long interval A without creating overlap?' : 'Which element of the larger feasible set can join the smaller one without breaking the rule?'} reset={() => changeKind('intervals')}>
    <div className="combinatorial-controls"><label>Feasibility rule<select aria-label="Feasibility rule" value={kind} onChange={event => changeKind(event.target.value)}>{Object.entries(MATROID_PRESETS).map(([key, preset]) => <option key={key} value={key}>{{
              uniform: 'At most two elements',
              graphic: 'Acyclic graph edges',
              intervals: 'Nonoverlapping intervals'
            }[key] || preset.label}</option>)}</select></label>
      <Slider label="First element weight" min={-3} max={10} value={weight} onChange={setWeight} /></div>
    {kind === 'intervals' ? <figure className="combinatorial-intervals"><figcaption>Half-open time intervals: touching endpoints do not overlap.</figcaption>{state.preset.intervals.map(([start, end], index) => <div key={index}><span>{letter(index)}</span><span className="combinatorial-time"><span style={{
            marginLeft: start * 50 + '%',
            width: (end - start) * 50 + '%'
          }}>{start} → {end} · weight {weights[index]}</span></span></div>)}</figure> : kind === 'graphic' ? <Graph count={4} edges={state.preset.edges} selected={selectedIndices(state.greedy, 6)} edgeLabels={weights} caption="Amber edges are the greedy forest. An added edge must not close a cycle." /> : <div className="combinatorial-item-row">{weights.map((value, index) => <span key={index} className={state.greedy & 1 << index ? 'is-selected' : ''}>{letter(index)}<strong>{value}</strong></span>)}</div>}
    <div className="combinatorial-controls">{[['Smaller set I', smaller, setSmaller], ['Larger set J', larger, setLarger]].map(([label, value, setter]) => <label key={label}>{label}<select aria-label={label} value={value} onChange={event => setter(Number(event.target.value))}>{state.feasible.map(mask => <option key={mask} value={mask}>{nameSet(mask)}</option>)}</select></label>)}</div>
    <div className="combinatorial-exchange"><span>I = {nameSet(smaller)}</span><span>← add an element from J∖I</span><span>J = {nameSet(larger)}</span></div>
    <p aria-live="polite">{!state.applicable ? 'Both sets are feasible, but |I| < |J| is not satisfied. Choose a strictly larger J to test augmentation.' : state.witness.length ? 'Legal augmentation: add ' + state.witness.map(index => state.preset.names[index]).join(' or ') + ' to I.' : 'No augmentation exists: every element in J∖I makes I infeasible. This is an actual failure of the axiom.'}</p>
    <Metrics values={[['Greedy value', state.greedyValue], ['Tiny exact optimum', state.optimumValue], ['Augmentation for every feasible pair?', state.augmentationHolds ? 'Yes' : 'No']]} />
    <p>Greedy selects {nameSet(state.greedy)}; a tiny exhaustive optimum selects {nameSet(state.optimum)}. Nonpositive elements are skipped because selecting nothing is allowed. All subsets are checked only for these bounded presets.</p>
    <p><strong>Transfer.</strong> Set the first weight to −3. Explain why a maximum-weight independent set may have fewer elements than a mandatory base. Then switch to the forest and identify a rejected cycle edge.</p>
  </Investigation>;
}
export function KnapsackBoundLab() {
  const [capacity, setCapacity] = useState(10),
    [step, setStep] = useState(0);
  const model = useMemo(() => knapsackSearchTrace({
    capacity
  }), [capacity]);
  const index = Math.min(step, model.trace.length - 1),
    state = model.trace[index];
  const items = KNAPSACK_ITEMS.filter((_, item) => state.incumbent.mask & 1 << item);
  return <Investigation id="branch-bound" title="What is still possible in an unexplored branch?" guidance="The fractional root value is 59 at capacity 10. Does a whole-item solution worth 59 necessarily exist?" reset={() => {
    setCapacity(10);
    setStep(0);
  }}>
    <Slider label="Whole-item capacity" max={15} value={capacity} onChange={value => {
      setCapacity(value);
      setStep(0);
    }} />
    <div className="combinatorial-item-row">{KNAPSACK_ITEMS.map(item => <span key={item.name}>{item.name}<strong>{item.value}</strong><small>weight {item.weight}</small></span>)}</div>
    <div className="combinatorial-capacity" aria-label={'Incumbent uses ' + state.incumbent.weight + ' of ' + capacity + ' capacity'}>{items.map(item => <span key={item.name} style={{
        width: 100 * item.weight / Math.max(capacity, 1) + '%'
      }}>{item.name}</span>)}</div>
    <Metrics values={[['Feasible incumbent value', state.incumbent.value], ['Remaining upper bound', display(state.upper)], ['Maximum remaining improvement', display(state.gap)], ['Optimality certified?', state.optimal ? 'Yes' : 'Not yet']]} />
    <Stepper value={index} maximum={model.trace.length - 1} setValue={setStep} />
    <p>{state.action}</p>
    <div className="combinatorial-tree" role="region" aria-label="Branch and bound decision tree" tabIndex="0">{state.nodes.map(node => <div key={node.id} className={'combinatorial-tree-node ' + (node.id === state.current ? 'is-current' : '')} style={{
        marginLeft: node.depth * 12
      }}>
      <strong>#{node.id} {node.decision}</strong><span>parent {node.parent ?? 'none'} · value {node.value} · weight {node.weight}</span><span>bound {display(node.upper)} · {node.status}</span></div>)}</div>
    <p>The indentation is decision depth, not elapsed time. A bound is optimistic; a value comes from an actual legal subset. Nodes retained in the frontier can still contain better answers.</p>
    <p><strong>Transfer.</strong> Stop midway and write a valid quality interval. Then use capacity 0: an empty subset is a feasible optimum, and a zero bound is meaningful.</p>
  </Investigation>;
}
const assignmentPresets = {
  reversal: {
    label: 'An assignment must move',
    costs: ASSIGNMENT_COSTS
  },
  signed: {
    label: 'Signed costs',
    costs: [[-4, 2, null], [0, 3, 2], [1, null, -1]]
  },
  missing: {
    label: 'Missing assignments',
    costs: [[1, null, null], [2, null, null], [null, 1, 2]]
  }
};
export function AssignmentResidualLab() {
  const [kind, setKind] = useState('reversal'),
    [step, setStep] = useState(0);
  const model = useMemo(() => assignmentTrace({
    costs: assignmentPresets[kind].costs
  }), [kind]);
  const state = model.trace[Math.min(step, model.trace.length - 1)];
  const captionId = useId();
  return <Investigation id="assignment" title="Follow the refund through the residual network" guidance="If A already occupies job 1, can B take job 1 without losing A's assignment count?" reset={() => {
    setKind('reversal');
    setStep(0);
  }}>
    <label>Assignment case<select aria-label="Assignment case" value={kind} onChange={event => {
        setKind(event.target.value);
        setStep(0);
      }}>{Object.entries(assignmentPresets).map(([key, preset]) => <option key={key} value={key}>{preset.label}</option>)}</select></label>
    <div className="combinatorial-assignment">
      <table><caption>Cost matrix; ✓ marks the current matching.</caption><thead><tr><th>Worker</th>{model.costs[0].map((_, job) => <th key={job}>Job {job + 1}</th>)}</tr></thead><tbody>{model.costs.map((row, worker) => <tr key={worker}><th>{letter(worker)}</th>{row.map((cost, job) => <td key={job} className={state.pairs.some(pair => pair.worker === worker && pair.job === job) ? 'is-selected' : ''}>{cost === null ? 'absent' : cost}{state.pairs.some(pair => pair.worker === worker && pair.job === job) && ' ✓'}</td>)}</tr>)}</tbody></table>
      <svg viewBox="0 0 290 230" role="img" aria-labelledby={captionId}><title id={captionId}>Current worker-job matching. Each worker and job has capacity one.</title>
        {state.pairs.map(pair => <line key={pair.worker} x1="42" x2="246" y1={48 + pair.worker * 68} y2={48 + pair.job * 68} stroke="#e9bb67" strokeWidth="3" />)}
        {[0, 1, 2].map(index => <g key={index}><circle cx="42" cy={48 + index * 68} r="18" fill="#151a20" stroke="#96c3e8" /><text x="42" y={53 + index * 68} textAnchor="middle">{letter(index)}</text><circle cx="246" cy={48 + index * 68} r="18" fill="#151a20" stroke="#a1d6b8" /><text x="246" y={53 + index * 68} textAnchor="middle">{index + 1}</text></g>)}
      </svg>
    </div>
    <Metrics values={[['Assignments currently filled', state.flow + ' / ' + model.required], ['Current total cost', state.cost], ['Last path cost change', state.pathCost ?? 0]]} />
    <Stepper value={Math.min(step, model.trace.length - 1)} maximum={model.trace.length - 1} setValue={setStep} />
    <ol className="combinatorial-residual-path">{state.path.filter(edge => edge.type === 'assignment').map((edge, index) => <li key={index} className={edge.delta === 'undo' ? 'is-refund' : ''}><strong>{edge.delta === 'undo' ? 'Undo / refund' : 'Add'}</strong><span>{letter(edge.worker)} ↔ job {edge.job + 1}</span><span>cost change {edge.cost > 0 ? '+' : ''}{edge.cost}</span></li>)}</ol>
    {state.path.length === 0 && <p>No augmenting path has been sent yet.</p>}
    {step === model.trace.length - 1 && <p aria-live="polite">{model.feasible ? 'Requested size reached. ' : 'Requested size is impossible: no residual source–sink path remains. '}Every positive-capacity residual edge has nonnegative reduced cost under the computed vertex potentials, certifying no negative residual cycle. This establishes minimum cost for the attained size.</p>}
    <details><summary>Inspect the final potential certificate</summary><p>The certificate belongs to the final flow, not an earlier displayed step. Reduced cost is edge cost + potential(start) − potential(end). Summing around any cycle cancels the potentials.</p><LessonTable caption="Final residual edges; source and sink are capacity nodes." headers={['From → to', 'Cost', 'Reduced cost']} rows={model.residual.map(edge => [edge.from + ' → ' + edge.to, edge.cost, edge.reducedCost])} /><p>Vertex potentials: {model.potentials.join(', ')}. Node 0–2 are A–C; nodes 3–5 are jobs 1–3; nodes 6 and 7 are source and sink.</p></details>
    <p><strong>Transfer.</strong> Step to the final reversal: 2 −1 +2 adds only 3 to the prior cost 2. Switch to missing edges and explain why a cheaper size-two answer cannot be labeled a feasible size-three solution.</p>
  </Investigation>;
}
export function CoverChoiceLab() {
  const [objective, setObjective] = useState('cover'),
    [caseName, setCaseName] = useState('equal'),
    [step, setStep] = useState(0);
  const sets = useMemo(() => [{
    name: 'A',
    cost: caseName === 'weighted' ? 3 : 1,
    elements: [0, 1, 2, 3]
  }, {
    name: 'B',
    cost: caseName === 'weighted' ? 2 : 1,
    elements: caseName === 'missing' ? [0, 1] : [0, 1, 4]
  }, {
    name: 'C',
    cost: caseName === 'weighted' ? 2 : 1,
    elements: [2, 3, 5]
  }], [caseName]);
  const model = useMemo(() => setCoverTrace({
    sets,
    maximumSelections: objective === 'cover' ? null : 2
  }), [sets, objective]);
  const state = model.trace[Math.min(step, model.trace.length - 1)];
  return <Investigation id="cover" title="The objective changes which decision is good" guidance="After selecting A, why must full cover continue, while a two-test budget must eventually stop?" reset={() => {
    setObjective('cover');
    setCaseName('equal');
    setStep(0);
  }}>
    <div className="combinatorial-controls"><label>Optimization objective<select aria-label="Optimization objective" value={objective} onChange={event => {
          setObjective(event.target.value);
          setStep(0);
        }}><option value="cover">Cover every behavior</option><option value="coverage">Choose at most two tests</option></select></label>
      <label>Candidate data<select aria-label="Candidate data" value={caseName} onChange={event => {
          setCaseName(event.target.value);
          setStep(0);
        }}><option value="equal">Equal costs: 1, 1, 1</option><option value="weighted">Weighted costs: 3, 2, 2</option><option value="missing">Behavior 5 is unavailable</option></select></label></div>
    <p>● means a test covers a behavior; ✓ identifies a selected test.</p><p className="combinatorial-scroll-note">Scroll the table sideways to inspect all six behavior columns.</p>
    <div className="combinatorial-incidence" role="region" aria-label="Candidate incidence and element charges" tabIndex="0"><table><caption>Tests and six behaviors</caption><thead><tr><th>Test / cost</th>{[1, 2, 3, 4, 5, 6].map(value => <th key={value}>{value}</th>)}</tr></thead><tbody>{sets.map((set, index) => <tr key={set.name} className={state.selected.includes(index) ? 'is-selected' : ''}><th>{set.name} / {set.cost} {state.selected.includes(index) && '✓'}</th>{[0, 1, 2, 3, 4, 5].map(element => <td key={element}>{set.elements.includes(element) ? '●' : '·'}</td>)}</tr>)}
      <tr><th>Covered?</th>{[0, 1, 2, 3, 4, 5].map(element => <td key={element}>{state.covered & 1 << element ? '✓' : '—'}</td>)}</tr>
      {objective === 'cover' && <tr><th>Element charge</th>{state.charges.map((value, index) => <td key={index}>{display(value)}</td>)}</tr>}</tbody></table></div>
    <Stepper value={Math.min(step, model.trace.length - 1)} maximum={model.trace.length - 1} setValue={setStep} />
    <Metrics values={[['Chosen tests', state.selected.map(index => letter(index)).join(', ') || 'none'], ['Behaviors covered', selectedIndices(state.covered, 6).length + ' / 6'], [objective === 'cover' ? 'Current cost' : 'Used test slots', objective === 'cover' ? state.cost : state.selected.length + ' / 2'], [objective === 'cover' ? 'Tiny exact cover cost' : 'Tiny maximum possible coverage', objective === 'cover' ? Number.isFinite(model.optimumCost) ? model.optimumCost : 'infeasible' : model.optimumCoverage]]} />
    <p>{objective === 'cover' ? 'Choose minimum cost per newly covered behavior. Only new behaviors receive this step’s equal charge; their sum equals the price paid.' : 'Choose the largest new coverage gain. Money is not the constraint in this cardinality problem, so changing the displayed monetary costs does not change its rule.'}</p>
    {step === model.trace.length - 1 && <p aria-live="polite">{objective === 'cover' ? model.feasible ? 'Every requirement is covered. The harmonic guarantee uses H(maximum set size) = ' + display(model.harmonic) + '.' : 'No candidate contains behavior 5: full coverage is infeasible, so a full-cover approximation ratio does not apply.' : 'The budgeted greedy rule has finished. With k=2, its proven fraction is at least 3/4 of the optimal coverage; uncovered behaviors are allowed by this objective.'}</p>}
    <p><strong>Transfer.</strong> Compare equal and weighted costs. Then make behavior 5 unavailable and say whether each objective has become infeasible.</p>
  </Investigation>;
}
export function VertexCoverBudgetLab() {
  const [kind, setKind] = useState('triangle'),
    [firstCost, setFirstCost] = useState(2),
    [step, setStep] = useState(0);
  const preset = VERTEX_COVER_PRESETS[kind];
  const model = useMemo(() => vertexCoverState({
    costs: preset.costs.map((value, index) => index === 0 ? firstCost : value),
    edges: preset.edges
  }), [preset, firstCost]);
  const state = model.trace[Math.min(step, model.trace.length - 1)];
  const changeKind = value => {
    setKind(value);
    setFirstCost(VERTEX_COVER_PRESETS[value].costs[0]);
    setStep(0);
  };
  return <Investigation id="vertex-cover" title="Edges spend a shared vertex budget" guidance="Can an edge load grow past the cheaper endpoint's remaining budget?" reset={() => changeKind('triangle')}>
    <div className="combinatorial-controls"><label>Cover graph<select aria-label="Cover graph" value={kind} onChange={event => changeKind(event.target.value)}>{Object.entries(VERTEX_COVER_PRESETS).map(([key, value]) => <option key={key} value={key}>{value.label}</option>)}</select></label>
      <Slider label="Vertex A cost" value={firstCost} onChange={value => {
        setFirstCost(value);
        setStep(0);
      }} /></div>
    <Graph count={model.costs.length} edges={model.edges} selected={state.edge === null ? [] : [state.edge]} labels={model.costs.map((cost, index) => (state.selected.includes(index) ? '✓ ' : '') + 'cost ' + cost)} edgeLabels={state.loads} caption="Edge labels are current dual loads. ✓ marks selected tight vertices; amber identifies the most recently increased edge." />
    <div className="combinatorial-reservoirs">{model.costs.map((cost, index) => <div key={index}><strong>{letter(index)}</strong><meter min="0" max={Math.max(1, cost)} value={state.totals[index]} aria-label={'Load used at vertex ' + letter(index)} /><span>{display(state.totals[index])} / {cost} {state.selected.includes(index) && 'selected'}</span></div>)}</div>
    <Stepper value={Math.min(step, model.trace.length - 1)} maximum={model.trace.length - 1} setValue={setStep} />
    <Metrics values={[['Dual lower bound', state.lower], ['Selected-vertex cost', state.cost], ['Uncovered edges', state.uncovered.length], ['Tiny exact cover optimum', model.exactCost]]} />
    <p>{state.uncovered.length ? 'The loads already give a lower bound. The selected vertices do not yet cover every edge, so their cost is not yet a feasible upper bound.' : 'All edges are covered. The selected vertices now give a feasible upper bound, at most twice the current dual lower bound.'}</p>
    <details><summary>Compare a separately solved fractional relaxation</summary>
      <LessonTable caption="The tiny half-integral oracle is exponential; the lesson explains why its grid contains an LP optimum." headers={['Vertex', 'LP value x', 'Round x ≥ 1/2?']} rows={model.fractional.map((value, index) => [letter(index), value, value >= .5 ? 'select' : 'skip'])} />
      <Metrics values={[['LP optimum', model.fractionalCost], ['Rounded cover cost', model.roundedCost], ['Twice LP value', 2 * model.fractionalCost]]} />
      <p>The primal-dual loads above need not maximize the dual. A numerical LP solver may also return a different optimum when several exist; this bounded oracle prefers half-valued ties to expose rounding behavior.</p>
    </details>
    <p><strong>Transfer.</strong> Set A's cost to 0. It is selected immediately, without a stalled edge update. In the expensive-hub star, compare the legal cover costs of the hub and all leaves.</p>
  </Investigation>;
}
export function KnapsackScalingLab() {
  const [denominator, setDenominator] = useState(4),
    [capacity, setCapacity] = useState(10);
  const model = useMemo(() => scaledKnapsackState({
    capacity,
    epsilonDenominator: denominator
  }), [capacity, denominator]);
  const points = model.rows.at(-1).map((weight, score) => ({
    weight,
    score
  })).filter(point => Number.isFinite(point.weight) && point.weight <= capacity);
  const largest = Math.max(1, model.rows[0].length - 1),
    height = Math.max(1, capacity);
  const id = useId();
  return <Investigation id="scaling" title="Round values, keep the real capacity" guidance="Does asking for half the allowed error guarantee a different selected subset, or only a tighter theorem and a larger DP?" reset={() => {
    setDenominator(4);
    setCapacity(10);
  }}>
    <div className="combinatorial-controls"><label>Allowed fractional loss ε<select aria-label="Allowed fractional loss epsilon" value={denominator} onChange={event => setDenominator(Number(event.target.value))}>{[2, 4, 10, 20].map(value => <option key={value} value={value}>1/{value}</option>)}</select></label>
      <Slider label="Scaling example capacity" max={15} value={capacity} onChange={setCapacity} /></div>
    <LessonTable caption="Only feasible positive-value items set the scale. Original weights are unchanged." headers={['Item', 'Weight', 'True value', 'Rounded value']} rows={model.active.map(item => [item.name, item.weight, item.value, item.rounded])} />
    <figure className="combinatorial-frontier"><svg viewBox="0 0 310 220" role="img" aria-labelledby={id}><title id={id}>Reachable rounded scores whose minimum weight fits the capacity. These are calculated dynamic-programming states, not timings.</title><line x1="42" y1="175" x2="290" y2="175" stroke="#637484" /><line x1="42" y1="45" x2="42" y2="175" stroke="#637484" />
      <text x="42" y="193">0</text><text x="290" y="193" textAnchor="end">{largest}</text><text x="33" y="179" textAnchor="end">0</text><text x="33" y="54" textAnchor="end">{capacity}</text><text x="160" y="213" textAnchor="middle">rounded value</text><text x="42" y="21">minimum weight</text>
      {points.map(point => <circle key={point.score} cx={42 + point.score / largest * 248} cy={175 - point.weight / height * 125} r="3" fill="#e9bb67" />)}</svg>
      <figcaption>Only reachable states that fit are plotted. Missing scores are absent, not interpolated. The algorithm uses every DP column, including unreachable ones.</figcaption></figure>
    <Metrics values={[['Scale K', display(model.scale)], ['True selected value', model.value], ['Weight used', model.weight], ['DP update cells', model.operations], ['Tiny exact optimum', model.oracle.value], ['Proved minimum value', display(model.guarantee)]]} />
    <p>Chosen items: {model.selected.map(index => KNAPSACK_ITEMS[index].name).join(', ') || 'none'}. Work is shown as the actual DP cell count, not a fabricated speed comparison.</p>
    <p><strong>Transfer.</strong> Compare ε=1/2 and ε=1/20. The selected subset may remain optimal in both runs; the worst-case guarantee becomes tighter even when this instance's answer does not change. Capacity 0 should produce a legal empty answer.</p>
  </Investigation>;
}
export function MetricShortcutFigure() {
  return <figure className="combinatorial-inline"><figcaption>Shortcutting depends on a measurable inequality.</figcaption>
    <div className="combinatorial-comparison"><div><strong>Metric triangle</strong><p>A → B costs 2<br />B → C costs 3<br />A → C costs at most 5</p><p>Skipping B cannot increase this part of the walk.</p></div><div><strong>Nonmetric counterexample</strong><p>A → B costs 1<br />B → C costs 1<br />A → C costs 100</p><p>Skipping B increases cost from 2 to 100. The proof has lost its key assumption.</p></div></div>
    <p>The lesson's Manhattan-coordinate program computes a doubled-tree tour of cost 14 and an odd-repaired tour of cost 10 on the same five-point fixture. Its tiny exact optimum is 10.</p>
  </figure>;
}
