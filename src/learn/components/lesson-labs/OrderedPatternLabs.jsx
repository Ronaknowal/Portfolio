import { useState } from 'react';
import { boundaryTrace, initialMergeState, mergeChoice, mergeRecords, pairTrace, windowTrace, rateState, rateSearch, rateJobs } from '../../data/ordered-pattern-models.js';
import './ordered-pattern-labs.css';

function TraceControls({ step, length, setStep, noun = 'comparison' }) {
  return <div className="lesson-controls ordered-trace-controls">
    <button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous {noun}</button>
    <button disabled={step === length - 1} onClick={() => setStep(step + 1)}>Next {noun}</button>
    <button onClick={() => setStep(0)}>Restart trace</button>
    <span>State {step + 1} of {length}</span>
  </div>;
}

export function BoundarySearchLab() {
  const [draft, setDraft] = useState('2, 4, 4, 4, 7, 9');
  const [values, setValues] = useState([2, 4, 4, 4, 7, 9]);
  const [target, setTarget] = useState(4);
  const [side, setSide] = useState('left');
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const model = boundaryTrace(values, target, side);
  const state = model.steps[step];
  const beforeRelation = side === 'left' ? '<' : '≤';
  const afterRelation = side === 'left' ? '≥' : '>';
  function apply(event) {
    event.preventDefault();
    try {
      const parsed = draft.trim() === '' ? [] : draft.split(',').map(part => {
        if (!/^\s*-?\d+\s*$/.test(part)) throw new RangeError('Use comma-separated integers; leave empty for an empty array.');
        return Number(part);
      });
      boundaryTrace(parsed, target, side);
      setValues(parsed);
      setStep(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setDraft('2, 4, 4, 4, 7, 9');
    setValues([2, 4, 4, 4, 7, 9]);
    setTarget(4);
    setSide('left');
    setStep(0);
    setError('');
  }
  return <section className="lesson-lab ordered-lab" aria-label="Boundary search investigation">
    <p className="lesson-eyebrow">BINARY SEARCH · LOCATE A GAP</p>
    <h3>Which boundary still might be the answer?</h3>
    <p>Predict the next low and high before advancing. Values outside the unresolved interval have a proved relation to the target. The answer is one of the gaps 0 through n.</p>
    <form className="lesson-controls" onSubmit={apply}>
      <label>Sorted integers<input aria-label="Boundary sorted values" value={draft} onChange={event => setDraft(event.target.value)} /></label>
      <button type="submit">Apply array</button>
      <button type="button" onClick={reset}>Reset boundaries</button>
    </form>
    {error && <p role="alert">{error} The active array is unchanged.</p>}
    <div className="lesson-controls">
      <label>Target: {target}<input aria-label="Boundary target" type="range" min="-10" max="20" value={target} onChange={event => { setTarget(Number(event.target.value)); setStep(0); }} /></label>
      <label>Boundary rule<select aria-label="Boundary rule" value={side} onChange={event => { setSide(event.target.value); setStep(0); }}><option value="left">First value ≥ target</option><option value="right">First value &gt; target</option></select></label>
    </div>
    <div className="ordered-scroll" tabIndex={0} role="region" aria-label="Array values and insertion gaps">
      <div className="ordered-boundary-strip">
        {Array.from({ length: values.length + 1 }, (_, index) => <div className="ordered-gap-group" key={index}>
          <div className={`ordered-gap ${index >= state.low && index <= state.high ? 'possible-gap' : ''}`}>
            <small><span className="ordered-wide-label">gap </span><span className="ordered-narrow-label">g</span>{index}</small><strong>{index === state.low && index === state.high ? 'L=H' : index === state.low ? 'L' : index === state.high ? 'H' : '│'}</strong>
          </div>
          {index < values.length && <div className={`ordered-value ${index < state.low ? 'proved-before' : index >= state.high ? 'proved-after' : 'unresolved'} ${index === state.middle ? 'is-middle' : ''}`}>
            <small><span className="ordered-wide-label">index </span><span className="ordered-narrow-label">i</span>{index}</small><strong>{values[index]}</strong><small>{index === state.middle ? 'mid' : index < state.low ? `${beforeRelation} ${target}` : index >= state.high ? `${afterRelation} ${target}` : '?'}</small>
          </div>}
        </div>)}
      </div>
    </div>
    <p className="ordered-readout" aria-live="polite" data-boundary={state.done ? model.boundary : ''}>
      {state.done ? `Low = high = ${model.boundary}. This is the insertion boundary${model.boundary === values.length ? '; no value lies to its right' : `; its right-hand value is ${values[model.boundary]}`}.` : `Search values in [${state.low},${state.high}). Mid = ${state.middle}, value ${values[state.middle]}: ${state.belongsBefore ? `discard through mid; next low = ${state.nextLow}` : `keep mid as a boundary candidate; next high = ${state.nextHigh}`}.`}
    </p>
    <TraceControls step={step} length={model.steps.length} setStep={setStep} />
    <p className="lesson-note">Exact index trace; at most 16 sorted integers from −1000 to 1000. The target control covers −10 to 20. Gaps (g) are inclusive candidates [low,high]; unresolved element indices (i) are half-open [low,high). Longer strips scroll horizontally. Validating the input is separate from the logarithmic search cost.</p>
  </section>;
}

export function StableMergeLab() {
  const [state, setState] = useState(initialMergeState);
  const complete = state.output.length === mergeRecords.left.length + mergeRecords.right.length;
  return <section className="lesson-lab ordered-lab" aria-label="Stable merge investigation">
    <p className="lesson-eyebrow">MERGE · CHOOSE THE NEXT RECORD</p>
    <h3>Sorted keys do not settle what happens to ties</h3>
    <p>These are two already sorted halves of one original sequence. A, B and C originally preceded D, E and F. Move a smallest head into the output. At equal keys, try each choice and inspect the consequences.</p>
    <div className="ordered-merge-lanes">
      {['left', 'right'].map(lane => <div className="ordered-merge-lane" key={lane}>
        <strong>{lane === 'left' ? 'Earlier half' : 'Later half'}</strong>
        <div className="ordered-records">{mergeRecords[lane].map((record, index) => <span className={`${index < state[lane] ? 'is-consumed' : ''} ${index === state[lane] ? 'is-head' : ''}`} key={record.id}><strong>{record.key}</strong><small>ID {record.id}</small><small>{index === state[lane] ? 'head' : index < state[lane] ? 'moved' : 'waiting'}</small></span>)}</div>
        <button disabled={state[lane] === mergeRecords[lane].length} onClick={() => setState(mergeChoice(state, lane))}>Take {lane} head</button>
      </div>)}
    </div>
    <div className="ordered-output" aria-label="Merged output">
      <strong>Output →</strong><div className="ordered-records">{state.output.map(record => <span key={record.id}><strong>{record.key}</strong><small>ID {record.id}</small></span>)}{!state.output.length && <span>empty</span>}</div>
    </div>
    <p className="ordered-readout" aria-live="polite">{state.feedback} {complete ? `Complete: sorted keys; ${state.stable ? 'original tie order preserved' : 'original tie order changed'}.` : ''}</p>
    <button onClick={() => setState(initialMergeState())}>Reset merge</button>
    <p className="lesson-note">This is a head-choice investigation, not unrestricted record dragging. Larger-head moves are rejected. Choosing a later equal key is allowed so you can observe a sorted but unstable result. A stable forward merge chooses the earlier half on a tie.</p>
  </section>;
}

export function PartitionRegionsFigure() {
  const states = [
    { label: 'Before: scan=3, greater=7', groups: [['< pivot', '[0,1)', '1'], ['= pivot', '[1,3)', '4  4'], ['unknown', '[3,7)', '6  2  4  1'], ['> pivot', '[7,7)', 'empty']] },
    { label: 'After: scan=3, greater=6', groups: [['< pivot', '[0,1)', '1'], ['= pivot', '[1,3)', '4  4'], ['unknown', '[3,6)', '1  2  4'], ['> pivot', '[6,7)', '6']] },
  ];
  return <figure className="ordered-inline" aria-label="Three-way partition swaps a large value out without advancing the scan">
    <figcaption>Pivot 4: a value arriving from the unknown tail must still be inspected.</figcaption>
    {states.map(state => <div className="ordered-partition-state" key={state.label}>
      <strong>{state.label}</strong>
      <div className="ordered-partition-regions">{state.groups.map(([label, range, values]) => <div key={label} className={label === 'unknown' ? 'is-unknown' : ''}><strong>{label}</strong><small>{range}</small><span>{values}</span></div>)}</div>
    </div>)}
    <p>The array starts this step as [1,4,4,6,2,4,1]. Swap the scanned 6 with the last unknown 1, producing [1,4,4,1,2,4,6]. The 6 is settled; the incoming 1 is not. Advancing scan now would incorrectly treat that 1 as equal to the pivot.</p>
  </figure>;
}

export function PairEliminationLab() {
  const values = [1, 2, 4, 5, 7, 9];
  const [target, setTarget] = useState(11);
  const [step, setStep] = useState(0);
  const model = pairTrace(values, target);
  const state = model.steps[step];
  return <section className="lesson-lab ordered-lab" aria-label="Pair elimination investigation">
    <p className="lesson-eyebrow">TWO POINTERS · ELIMINATE PARTNERS</p>
    <h3>One comparison rules out a whole edge of candidates</h3>
    <p>Rows select index i and columns select j. Only i &lt; j is allowed. Each cell is values[i] + values[j]. The outlined cell is the current left/right pair; × marks a pair already eliminated by a proved pointer move.</p>
    <div className="lesson-controls"><label>Pair target: {target}<input aria-label="Pair target" type="range" min="0" max="20" value={target} onChange={event => { setTarget(Number(event.target.value)); setStep(0); }} /></label><button onClick={() => { setTarget(11); setStep(0); }}>Reset pairs</button></div>
    <div className="ordered-pair-grid" role="table" aria-label="Candidate pair sums">
      <div role="row"><span role="columnheader">i / j</span>{values.map((value, index) => <span role="columnheader" key={index}><small>{index}</small>{value}</span>)}</div>
      {values.map((value, row) => <div role="row" key={row}>
        <span role="rowheader"><small>{row}</small>{value}</span>
        {values.map((other, column) => {
          const allowed = row < column;
          const active = allowed && row >= state.left && column <= state.right;
          const selected = allowed && row === state.left && column === state.right;
          return <span role="cell" aria-label={`Indices ${row},${column}: ${!allowed ? 'excluded because i must be less than j' : `${value + other}${active ? ', candidate' : ', eliminated'}${selected ? ', current' : ''}`}`} className={`${active ? 'is-candidate' : ''} ${selected ? 'is-current-pair' : ''}`} key={column}>{!allowed ? '—' : active ? value + other : '×'}</span>;
        })}
      </div>)}
    </div>
    <p className="ordered-readout" aria-live="polite" data-pair-action={state.action}>
      {state.action === 'absent' ? 'The pointers met. No pair of distinct remaining indices exists.' : `left=${state.left} (${values[state.left]}), right=${state.right} (${values[state.right]}): sum ${state.sum}. `}
      {state.action === 'left' && `Even its largest remaining partner is too small. Every pair using left=${state.left} fails, so discard that row.`}
      {state.action === 'right' && `Even its smallest remaining partner is too large. Every pair using right=${state.right} fails, so discard that column.`}
      {state.action === 'found' && 'A matching pair is found. The task asks for one pair, so we stop.'}
    </p>
    <TraceControls step={step} length={model.steps.length} setStep={setStep} noun="pair" />
    <p className="lesson-note">Exact candidate sums, not a runtime heatmap. The picture has quadratic cells only to expose the proof; the actual algorithm stores two indices and never constructs this grid.</p>
  </section>;
}

export function CompactionFigure() {
  const rows = [
    { label: 'Before reading index 4', values: ['1', '2', '2', '2', '5'], write: 2, read: 4, kept: 2 },
    { label: 'After writing 5 to index 2', values: ['1', '2', '5', '2', '5'], write: 3, read: 5, kept: 3 },
  ];
  return <figure className="ordered-inline" aria-label="Read/write compaction preserves an unread suffix">
    <figcaption>The read pointer inspects the source; the write pointer marks the next output slot.</figcaption>
    {rows.map(row => <div className="ordered-compaction-row" key={row.label}><strong>{row.label}</strong><div className="ordered-records">{row.values.map((value, index) => <span className={index < row.kept ? 'is-kept' : ''} key={index}><small>index {index}</small><strong>{value}</strong><small>{index === row.read ? 'read' : index === row.write ? 'write' : index < row.kept ? 'kept' : 'ignored'}</small></span>)}</div></div>)}
    <p>The illustration begins partway through [1,1,2,2,5], after the first 2 has been copied left. The final logical result is the first three cells [1,2,5]; the physical list still has five cells. No unread source value was overwritten.</p>
  </figure>;
}

export function IntervalUnionFigure() {
  const rows = [['input', 1, 4], ['input', 2, 3], ['input', 4, 5], ['union', 1, 5], ['separate', 6, 8]];
  return <figure className="ordered-inline" aria-label="Closed interval union from one through five; six through eight is separate">
    <figcaption>Closed endpoints: [1,4] touches [4,5], so both share the point 4.</figcaption>
    <svg className="ordered-interval-svg" viewBox="0 0 360 220" role="img" aria-label="Exact closed intervals and merged coverage on a common 0 to 9 axis">
      {Array.from({ length: 10 }, (_, index) => <g key={index}><line x1={65 + index * 31} x2={65 + index * 31} y1="20" y2="185" stroke="#494237" /><text x={65 + index * 31} y="210" textAnchor="middle" fill="currentColor">{index}</text></g>)}
      {rows.map(([label, start, stop], index) => <g key={index}><text x="0" y={36 + index * 34} fill="currentColor">{label}</text><line x1={65 + start * 31} x2={65 + stop * 31} y1={30 + index * 34} y2={30 + index * 34} stroke={label === 'union' ? '#9fc6a3' : '#e2ba68'} strokeWidth={label === 'union' ? 7 : 4} /><circle cx={65 + start * 31} cy={30 + index * 34} r="5" fill="#e2ba68" /><circle cx={65 + stop * 31} cy={30 + index * 34} r="5" fill="#e2ba68" /></g>)}
    </svg>
    <p>Nested [2,3] adds no new coverage. A start at 6 leaves a gap after 5. These are continuous coordinates, so [1,5] and [6,8] do not touch merely because 5 and 6 are consecutive integers.</p>
  </figure>;
}

export function MovingWindowLab() {
  const presets = { positive: [2, 1, 3, 2, 4], zeros: [0, 0, 5], empty: [] };
  const [preset, setPreset] = useState('positive');
  const [target, setTarget] = useState(6);
  const [step, setStep] = useState(0);
  const model = windowTrace(presets[preset], target);
  const state = model.steps[step];
  return <section className="lesson-lab ordered-lab" aria-label="Moving window investigation">
    <p className="lesson-eyebrow">WINDOW · GROW, QUALIFY, SHRINK</p>
    <h3>Why can the left boundary keep moving forward?</h3>
    <p>Find a shortest nonempty interval whose sum reaches the target. Predict whether the next event adds a value, records a qualifying interval, or removes a value. Every value here is nonnegative.</p>
    <div className="lesson-controls">
      <label>Window input<select aria-label="Window input" value={preset} onChange={event => { setPreset(event.target.value); setStep(0); }}><option value="positive">[2,1,3,2,4]</option><option value="zeros">[0,0,5]</option><option value="empty">Empty []</option></select></label>
      <label>Required sum: {target}<input aria-label="Window target" type="range" min="1" max="15" value={target} onChange={event => { setTarget(Number(event.target.value)); setStep(0); }} /></label>
      <button onClick={() => { setPreset('positive'); setTarget(6); setStep(0); }}>Reset window</button>
    </div>
    <div className="ordered-window" aria-label={`Window [${state.left},${state.right}), sum ${state.total}`}>
      {model.values.map((value, index) => <div className={index >= state.left && index < state.right ? 'in-window' : ''} key={index}><small>{index}</small><div className="ordered-window-bar"><i style={{ height: `${value * 18}px` }} /></div><strong>{value}</strong><small>{index >= state.left && index < state.right ? 'inside' : 'outside'}</small></div>)}
      {!model.values.length && <p>Empty input: there is no nonempty candidate.</p>}
    </div>
    <p className="ordered-readout" aria-live="polite" data-window-step={step}>{state.note} Current sum: <strong>{state.total}</strong>. Best so far: {state.best ? `[${state.best[0]},${state.best[1]}), length ${state.best[1] - state.best[0]}` : 'none'}.</p>
    <TraceControls step={step} length={model.steps.length} setStep={setStep} noun="event" />
    <p className="lesson-note">Exact uninstrumented-algorithm state with separate observation events. Bar height encodes the integer value, 18 pixels per unit; the text values are authoritative. Trace storage belongs to the teaching tool, not the O(1)-auxiliary-space algorithm.</p>
  </section>;
}

export function PrefixCancellationFigure() {
  return <figure className="ordered-inline" aria-label="Range sum is the difference between two boundary prefix totals">
    <figcaption>Prefix totals live at boundaries. P[r] − P[l] cancels everything before l.</figcaption>
    <div className="ordered-prefix-grid">
      <strong>boundary</strong>{[0, 1, 2, 3, 4].map(value => <span key={value}>{value}</span>)}
      <strong>P</strong>{[0, 4, 2, 5, 6].map((value, index) => <span className={index === 1 || index === 3 ? 'is-prefix-end' : ''} key={index}>{value}</span>)}
      <strong>next value</strong>{[4, -2, 3, 1, 'end'].map((value, index) => <span className={index === 1 || index === 2 ? 'is-kept' : ''} key={index}>{value}</span>)}
    </div>
    <p>[1,3) contains −2 and 3. P[3]=4−2+3=5; subtract P[1]=4 to get 1. P decreases from 4 to 2, so a signed prefix sequence is not necessarily suitable for binary search.</p>
  </figure>;
}

export function FeasibleRateLab() {
  const [speed, setSpeed] = useState(3);
  const [budget, setBudget] = useState(6);
  const state = rateState(speed, budget);
  const search = rateSearch(budget);
  return <section className="lesson-lab ordered-lab" aria-label="Feasible rate investigation">
    <p className="lesson-eyebrow">ANSWER SEARCH · TEST A CANDIDATE</p>
    <h3>Find the first rate that fits the work budget</h3>
    <p>Jobs contain 3, 6 and 7 units. One time slot works on one job at up to the chosen rate; unused room in its final slot cannot be shared with the next job. Before choosing a rate, predict its total slots.</p>
    <div className="lesson-controls"><label>Available slots: {budget}<input aria-label="Rate budget" type="range" min="2" max="10" value={budget} onChange={event => setBudget(Number(event.target.value))} /></label><button onClick={() => { setSpeed(3); setBudget(6); }}>Reset rates</button></div>
    <div className="ordered-rate-options" aria-label="Rate feasibility choices">{search.options.map(option => <button key={option.speed} aria-pressed={speed === option.speed} onClick={() => setSpeed(option.speed)}><strong>{option.speed} / slot</strong><span>{option.feasible ? 'fits' : 'too slow'}</span></button>)}</div>
    <div className="ordered-rate-jobs">{rateJobs.map((job, index) => <div key={index}><strong>Job {index + 1}: {job} units</strong><div>{Array.from({ length: state.slots[index] }, (_, slot) => <span key={slot}><strong>{Math.min(speed, job - slot * speed)}</strong><small>of {speed}</small></span>)}</div><span>{state.slots[index]} {state.slots[index] === 1 ? 'slot' : 'slots'}</span></div>)}</div>
    <p className="ordered-readout" aria-live="polite" data-rate-total={state.total}>Rate {speed}: {state.slots.join(' + ')} = {state.total} slots. {state.feasible ? 'Fits' : 'Exceeds'} budget {budget}. Smallest feasible rate: {search.answer ?? 'none; each job needs at least one slot'}.</p>
    <p className="lesson-note">Exact integer scheduling scenario. Buttons enumerate this tiny domain to expose monotonicity; the Python binary search evaluates only logarithmically many rates. This models serial, indivisible slots and fixed jobs, not a measured processor or general scheduling system.</p>
  </section>;
}
