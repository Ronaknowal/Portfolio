import { useId, useMemo, useState } from 'react';
import { allocateCapacity, assignRooms, capacityItems, deadlineSchedule, defaultJobs, exchangeAdjacent, formatRational, huffmanTree, intervalOptimum, intervalPresets, parseIntervals, reachablePrefix, selectIntervals, wholeCapacityOptimum } from '../../data/greedy-exchange-models.js';
import './greedy-exchange-labs.css';
function presetText(name) {
  return intervalPresets[name].map(row => row.join(', ')).join('\n');
}
function TimeRows({
  intervals,
  rows,
  selected = [],
  rejected = [],
  current = null,
  witness = null,
  label
}) {
  const maximum = Math.max(1, ...intervals.map(item => item.finish));
  const width = Math.max(320, maximum * 34 + 50);
  const offset = 34;
  const scale = (width - offset - 20) / maximum;
  const height = rows.length * 46 + 52;
  return <div className="gx-scroll" tabIndex={0} role="region" aria-label={label}>
      <svg viewBox={`0 0 ${width} ${height}`} style={{
      width
    }} role="img" aria-label={label}>
        {Array.from({
        length: maximum + 1
      }, (_, time) => <g key={time}>
            <line x1={offset + time * scale} x2={offset + time * scale} y1="25" y2={height - 18} className="gx-time-grid" />
            <text x={offset + time * scale} y="16" className="gx-time-tick">{time}</text>
          </g>)}
        {rows.map((row, index) => <g key={index}>
            <text x="2" y={index * 46 + 48} className="gx-row-label">{rows.length === intervals.length ? row[0]?.id : index + 1}</text>
            {row.map(item => <g key={item.id} className={`gx-interval${selected.includes(item.id) ? ' gx-interval--selected' : ''}${rejected.includes(item.id) ? ' gx-interval--rejected' : ''}${current === item.id ? ' gx-interval--current' : ''}`}>
                <rect x={offset + item.start * scale} y={index * 46 + 30} width={(item.finish - item.start) * scale} height="30" rx="2" />
                <text x={offset + (item.start + item.finish) * scale / 2} y={index * 46 + 50}>{item.id}</text>
              </g>)}
          </g>)}
        {witness !== null && <line x1={offset + witness * scale} x2={offset + witness * scale} y1="25" y2={height - 18} className="gx-witness" />}
      </svg>
    </div>;
}
export function CoinFailureFigure() {
  return <figure className="gx-inline">
      <figcaption>Same amount, different coin counts · denominations 1, 3 and 4</figcaption>
      <div className="gx-coins"><strong>Largest first</strong><span>4</span><span>1</span><span>1</span><b>6 with 3 coins</b></div>
      <div className="gx-coins"><strong>A better answer</strong><span>3</span><span>3</span><b>6 with 2 coins</b></div>
      <p>Both answers are feasible. The second uses fewer coins; feasibility alone does not prove optimality.</p>
    </figure>;
}
export function IntervalSelectionLab() {
  const heading = useId();
  const [draft, setDraft] = useState(presetText('appointments'));
  const [rule, setRule] = useState('finish');
  const [objective, setObjective] = useState('count');
  const [model, setModel] = useState(() => selectIntervals(parseIntervals(presetText('appointments'))));
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const state = model.frames[step];
  const optimum = useMemo(() => intervalOptimum(model.intervals, objective), [model.intervals, objective]);
  const selectedScore = model.intervals.filter(item => state.selected.includes(item.id)).reduce((sum, item) => sum + (objective === 'count' ? 1 : item.value), 0);
  const rejected = state.examined.filter(id => !state.selected.includes(id));
  function apply(event) {
    event.preventDefault();
    try {
      setModel(selectIntervals(parseIntervals(draft), rule));
      setStep(0);
      setError('');
    } catch (caught) {
      setError(caught.message);
    }
  }
  function reset(name = 'appointments') {
    const text = presetText(name);
    setDraft(text);
    setRule('finish');
    setObjective(name === 'weightedTrap' ? 'value' : 'count');
    setModel(selectIntervals(parseIntervals(text)));
    setStep(0);
    setError('');
  }
  return <section className="gx-lab" data-gx-lab="intervals" aria-labelledby={heading}>
      <p className="gx-eyebrow">SELECTION · KEEP A FEASIBLE SUBSET</p>
      <h3 id={heading}>Which local ordering leaves room for the most appointments?</h3>
      <p>Predict the first choice, then compare candidate rules on the same requests. Every interval occupies [start, finish); touching endpoints are compatible. Letters identify input occurrences.</p>
      <form onSubmit={apply} className="gx-form">
        <label>Intervals · start, finish, value<textarea rows="6" value={draft} onChange={event => setDraft(event.target.value)} /></label>
        <div className="gx-controls">
          <label>Candidate rule<select aria-label="Candidate rule" value={rule} onChange={event => setRule(event.target.value)}><option value="finish">Earliest finish</option><option value="start">Earliest start</option><option value="duration">Shortest duration</option></select></label>
          <button type="submit">Apply intervals and rule</button>
          <button type="button" onClick={() => reset()}>Reset appointments</button>
        </div>
      </form>
      <div className="gx-controls"><button type="button" onClick={() => reset('shortestTrap')}>Shortest-duration trap</button><button type="button" onClick={() => reset('weightedTrap')}>Weighted-value trap</button></div>
      {error && <p role="alert" className="gx-error">{error}</p>}
      <p className="gx-note">Applied rule: {model.rule}. Inputs and rule apply with the button; presets apply immediately. Equal scores use finish, start, then letter order. Empty input is allowed.</p>
      <TimeRows intervals={model.intervals} rows={model.intervals.map(item => [item])} selected={state.selected} rejected={rejected} current={state.current} label="Appointment timelines; integer time on the horizontal axis" />
      <p className="gx-note">Green = kept; dashed = rejected; amber outline = current request. Timeline width means duration, not value. Scroll the timeline locally when needed.</p>
      <p className="gx-status" aria-live="polite">{state.message}</p>
      <div className="gx-controls">
        <button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous request</button><span>Examined {step} / {model.order.length}</span>
        <button disabled={step === model.frames.length - 1} onClick={() => setStep(step + 1)}>Next request</button>
        <button disabled={step === model.frames.length - 1} onClick={() => setStep(model.frames.length - 1)}>Finish selection</button>
      </div>
      <div className="gx-comparison">
        <label>Score this selection by<select aria-label="Score this selection by" value={objective} onChange={event => setObjective(event.target.value)}><option value="count">Number of appointments</option><option value="value">Total value</option></select></label>
        <p><strong>Selected: {state.selected.join(', ') || 'none'} · score {selectedScore}</strong><br />Exact small-input optimum: {optimum.score}, one selection {optimum.selected.join(', ') || 'empty'}. {step === model.frames.length - 1 && selectedScore < optimum.score ? 'This candidate rule loses on this input.' : step < model.frames.length - 1 ? 'Finish the trace before comparing its final score.' : 'Matching this optimum is evidence for this input; the general guarantee still needs a proof.'}</p>
      </div>
      <details><summary>Inspect the applied requests and order</summary><table><thead><tr><th>ID</th><th>Occupied time</th><th>Value</th></tr></thead><tbody>{model.intervals.map(item => <tr key={item.id}><th>{item.id}</th><td>[{item.start}, {item.finish})</td><td>{item.value}</td></tr>)}</tbody></table><p>Examining order: {model.order.join(' → ') || 'empty'}.</p></details>
      <p className="gx-note">The lab checks compatibility against every kept request to compare all three rules fairly, and enumerates up to 64 subsets for its oracle. The native earliest-finish algorithm uses a sorted scan and one last-finish value; it does not perform this exhaustive comparison.</p>
    </section>;
}
export function IntervalExchangeFigure() {
  const earlier = {
    id: 'G',
    start: 0,
    finish: 3,
    value: 1
  };
  const optimalFirst = {
    id: 'O',
    start: 1,
    finish: 4,
    value: 1
  };
  const tail = [{
    id: 'P',
    start: 4,
    finish: 6,
    value: 1
  }, {
    id: 'Q',
    start: 6,
    finish: 8,
    value: 1
  }];
  return <figure className="gx-inline">
      <figcaption>Exchange the first appointment; preserve every later appointment</figcaption>
      <TimeRows intervals={[earlier, optimalFirst, ...tail]} rows={[[optimalFirst, ...tail], [earlier, ...tail]]} selected={['G']} label="First row O P Q; second row G P Q. G ends at 3, no later than O at 4." />
      <p>Row 1 has O→P→Q. Row 2 replaces O with G and keeps P and Q unchanged. G finishes at 3 ≤ 4, so the tail that fitted after O still fits. Both selections contain three appointments.</p>
    </figure>;
}
export function RoomsLowerBoundFigure() {
  const intervals = parseIntervals(presetText('appointments'));
  const model = assignRooms(intervals);
  return <figure className="gx-inline">
      <figcaption>Assign every request: {model.rooms.length} room lanes meet a matching lower bound</figcaption>
      <TimeRows intervals={intervals} rows={model.rooms} selected={model.witness.ids} witness={model.witness.time} label="Room assignments with a vertical simultaneous-occupancy witness" />
      <p>At time {model.witness.time}, requests {model.witness.ids.join(', ')} are simultaneously active: every schedule needs at least {model.witness.ids.length} rooms. Each numbered row is one reusable room. Endpoints are half-open; an appointment ending now releases its room.</p>
    </figure>;
}
function ScheduleStrip({
  schedule,
  title
}) {
  const smallestDuration = Math.min(...schedule.jobs.map(job => job.processing));
  const width = Math.max(300, schedule.total / smallestDuration * 54);
  return <div>
      <h4>{title}</h4>
      <div className="gx-scroll" tabIndex={0} role="region" aria-label={`${title} duration-scaled schedule`}>
        <div className="gx-schedule" style={{
        width
      }}>{schedule.scheduled.map(job => <div key={job.id} className={`gx-job gx-job--${job.id}`} style={{
          flex: job.processing
        }}><strong>{job.id}</strong><small>{job.start} → {job.completion}</small><span>d={job.deadline}</span></div>)}</div>
      </div>
      <p>Maximum signed lateness <strong>{schedule.maximumLateness}</strong>; maximum tardiness <strong>{schedule.maximumTardiness}</strong>.</p>
    </div>;
}
export function DeadlineExchangeLab() {
  const heading = useId();
  const [draft, setDraft] = useState(defaultJobs.map(job => ({
    ...job
  })));
  const [schedule, setSchedule] = useState(() => deadlineSchedule(defaultJobs));
  const [before, setBefore] = useState(null);
  const [pair, setPair] = useState(0);
  const [message, setMessage] = useState('Swap A and B: B has the earlier deadline. Predict which individual completion gets later.');
  const [error, setError] = useState('');
  function update(index, key, text) {
    setDraft(draft.map((job, position) => position === index ? {
      ...job,
      [key]: text
    } : job));
  }
  function apply(event) {
    event.preventDefault();
    try {
      const jobs = draft.map(job => {
        if (!/^\d+$/.test(String(job.processing)) || !/^\d+$/.test(String(job.deadline))) throw new Error('Processing times and deadlines must be whole numbers.');
        return {
          ...job,
          processing: Number(job.processing),
          deadline: Number(job.deadline)
        };
      });
      setSchedule(deadlineSchedule(jobs));
      setBefore(null);
      setPair(0);
      setError('');
      setMessage('Applied the new jobs in A, B, C order. Select an adjacent pair to exchange.');
    } catch (caught) {
      setError(caught.message);
    }
  }
  function swap() {
    const result = exchangeAdjacent(schedule, pair);
    setBefore(result.before);
    setSchedule(result.after);
    setMessage(`${result.pair.join(' then ')} became ${[...result.pair].reverse().join(' then ')}. ${result.inverted ? 'This removes a deadline inversion; maximum lateness cannot increase.' : 'This pair was not a strict deadline inversion; the improvement guarantee does not apply.'}`);
  }
  function reset() {
    setDraft(defaultJobs.map(job => ({
      ...job
    })));
    setSchedule(deadlineSchedule(defaultJobs));
    setBefore(null);
    setPair(0);
    setError('');
    setMessage('Default jobs restored. A, B, C has maximum lateness 2.');
  }
  function orderByDeadline() {
    const order = [...schedule.jobs].sort((a, b) => a.deadline - b.deadline || a.id.localeCompare(b.id)).map(job => job.id);
    setBefore(schedule);
    setSchedule(deadlineSchedule(schedule.jobs, order));
    setMessage('Ordered all jobs by deadline. Any equal-deadline order has the same optimal maximum-lateness guarantee.');
  }
  return <section className="gx-lab" data-gx-lab="deadlines" aria-labelledby={heading}>
      <p className="gx-eyebrow">EXCHANGE · CHANGE AN ORDER WITHOUT DELAYING THE TAIL</p>
      <h3 id={heading}>Swap two neighbors and inspect every affected quantity</h3>
      <p>One processor starts at time zero. Every job is already available, runs without interruption and must be completed. Width is processing time; d is its deadline. Negative lateness means an early completion.</p>
      <form onSubmit={apply}>
        <div className="gx-job-inputs">{draft.map((job, index) => <fieldset key={job.id}><legend>Job {job.id}</legend><label>Processing time<input aria-label={`Job ${job.id} processing time`} inputMode="numeric" value={job.processing} onChange={event => update(index, 'processing', event.target.value)} /></label><label>Deadline<input aria-label={`Job ${job.id} deadline`} inputMode="numeric" value={job.deadline} onChange={event => update(index, 'deadline', event.target.value)} /></label></fieldset>)}</div>
        <div className="gx-controls"><button type="submit">Apply jobs</button><button type="button" onClick={reset}>Reset jobs</button></div>
      </form>
      {error && <p role="alert" className="gx-error">{error}</p>}
      <p className="gx-note">Processing times 1–8, deadlines 0–30. Edits apply only with Apply jobs. All displayed strips and quantities describe the applied jobs. Scroll a long schedule locally; even its shortest job keeps readable labels.</p>
      {before && <ScheduleStrip schedule={before} title="Before the last exchange" />}
      <ScheduleStrip schedule={schedule} title="Current order" />
      <div className="gx-controls"><label>Adjacent pair<select aria-label="Adjacent pair" value={pair} onChange={event => setPair(Number(event.target.value))}>{schedule.order.slice(0, -1).map((id, index) => <option key={index} value={index}>{id} then {schedule.order[index + 1]}</option>)}</select></label><button onClick={swap}>Exchange adjacent jobs</button><button onClick={orderByDeadline}>Order all by deadline</button></div>
      <p className="gx-status" aria-live="polite">{message}</p>
      <table><thead><tr><th>Job</th><th>Completion</th><th>Deadline</th><th>C − d</th></tr></thead><tbody>{schedule.scheduled.map(job => <tr key={job.id}><th>{job.id}</th><td>{job.completion}</td><td>{job.deadline}</td><td>{job.lateness}</td></tr>)}</tbody></table>
      <p className="gx-note">Later jobs start at the same time after an adjacent exchange because the pair's total processing time is unchanged. The code below sorts once; repeated swaps here expose its proof rather than its production implementation.</p>
    </section>;
}
export function FractionalCapacityLab() {
  const heading = useId();
  const [capacity, setCapacity] = useState(50);
  const [divisible, setDivisible] = useState(true);
  const model = allocateCapacity(capacityItems, capacity, divisible);
  const whole = wholeCapacityOptimum(capacityItems, capacity);
  return <section className="gx-lab" data-gx-lab="capacity" aria-labelledby={heading}>
      <p className="gx-eyebrow">EXCHANGE · SAME WEIGHT, DIFFERENT BENEFIT</p>
      <h3 id={heading}>Taking part of an item changes which exchanges are legal</h3>
      <p>Items A, B and C have weights 10, 20 and 30; full values 60, 100 and 120. Compare their value per unit of capacity: 6, 5 and 4. Predict the result at capacity 50 before changing divisibility.</p>
      <div className="gx-controls"><label>Capacity: {capacity}<input aria-label="Capacity" type="range" min="0" max="60" value={capacity} onChange={event => setCapacity(Number(event.target.value))} /></label><label className="gx-check"><input type="checkbox" checked={divisible} onChange={event => setDivisible(event.target.checked)} />Allow fractional items</label><button onClick={() => {
        setCapacity(50);
        setDivisible(true);
      }}>Reset capacity</button></div>
      <p className="gx-note">Changes apply immediately. The order stays A→B→C. Fractional mode takes as much as fits; whole-item mode skips an item that does not fit and continues.</p>
      <div className="gx-capacity-track" aria-label={`Used capacity ${capacity - model.remaining} of ${capacity}`}>
        {model.taken.filter(item => item.amount > 0).map(item => <div key={item.id} className={`gx-job--${item.id}`} style={{
        flex: item.amount
      }} title={`${item.id}: ${item.amount} weight`}>{item.amount / capacity >= .05 && <><strong>{item.id}</strong><span>{item.amount}</span></>}</div>)}
        {model.remaining > 0 && <div className="gx-unused" style={{
        flex: model.remaining
      }} title={`Unused capacity: ${model.remaining}`}>{model.remaining / capacity >= .18 && <span>unused {model.remaining}</span>}</div>}
        {capacity === 0 && <span className="gx-empty-capacity">Zero capacity: nothing fits.</span>}
      </div>
      <p className="gx-note">Unused capacity: {model.remaining}. Segment width is the amount of occupied capacity. The exact amounts and fractions remain in the table; a thin segment is not rounded to a whole item.</p>
      <table><thead><tr><th>Item</th><th>Weight taken</th><th>Fraction</th><th>Value added</th></tr></thead><tbody>{model.taken.map(item => <tr key={item.id}><th>{item.id}</th><td>{item.amount}/{item.weight}</td><td>{formatRational(item.fraction)}</td><td>{formatRational(item.contribution)}</td></tr>)}</tbody></table>
      <p className="gx-status" aria-live="polite">Density-first value: <strong>{formatRational(model.total)}</strong>. {divisible ? 'This is optimal under divisible, linear value and capacity-only constraints.' : `The exact whole-item optimum is ${whole.value}, using ${whole.selected.join(', ') || 'nothing'}. ${model.total.numerator / model.total.denominator < whole.value ? 'Density-first loses: its fractional exchange is no longer legal.' : 'This particular input matches the optimum; that does not give a general whole-item guarantee.'}`}</p>
      <p className="gx-note">These are exact integer/rational calculations for a constructed allocation problem. No throughput, cost or performance measurement is implied. Fixed setup costs, dependencies, minimum batch sizes and nonlinear benefits require a different model.</p>
    </section>;
}
export function HuffmanMergeFigure() {
  const model = huffmanTree();
  return <figure className="gx-inline" data-gx-figure="huffman">
      <figcaption>Merge the two smallest remaining frequencies: 2, 3, 7, 9</figcaption>
      <div className="gx-scroll" tabIndex={0} role="region" aria-label="Huffman frequency tree; scroll locally if necessary">
        <svg viewBox={`0 0 ${model.width} ${model.height}`} style={{
        width: model.width
      }} role="img" aria-label="Root 21 has leaf D frequency 9 and subtree 12; subtree 12 has C frequency 7 and subtree 5; subtree 5 has leaves A frequency 2 and B frequency 3.">
          {model.nodes.filter(node => node.parent !== null).map(node => {
          const parent = model.nodes.find(item => item.id === node.parent);
          return <line key={`edge-${node.id}`} x1={parent.x} y1={parent.y + 17} x2={node.x} y2={node.y - 17} className="gx-tree-edge" />;
        })}
          {model.nodes.map(node => <g key={node.id} className={node.symbol ? 'gx-tree-leaf' : 'gx-tree-join'}><circle cx={node.x} cy={node.y} r="19" /><text x={node.x} y={node.y + 5}>{node.frequency}</text>{node.symbol && <text x={node.x} y={node.y + 38} className="gx-tree-symbol">{node.symbol}</text>}</g>)}
        </svg>
      </div>
      <p>Merge sums: {model.merges.map(item => `${item.left}+${item.right}=${item.sum}`).join('; ')}. Their total is {model.cost}. Left edge means 0 and right edge means 1; the table reads each full root-to-leaf path.</p>
      <table><thead><tr><th>Symbol</th><th>Frequency</th><th>Code</th><th>Bits contributed</th></tr></thead><tbody>{model.codes.map(item => <tr key={item.symbol}><th>{item.symbol}</th><td>{item.frequency}</td><td>{item.code}</td><td>{item.frequency} × {item.depth} = {item.frequency * item.depth}</td></tr>)}</tbody></table>
      <p>Circles carry subtree frequencies, not heap array positions. Only leaves are symbols. Geometry shows ancestry; code length is the number of edges, not the drawn line length.</p>
    </figure>;
}
export function ReachablePrefixFigure() {
  const model = reachablePrefix([2, 3, 0, 0, 1, 0]);
  const stages = model.frames.filter(frame => [0, 1, 4].includes(frame.index));
  const column = index => 27 + index * 45;
  return <figure className="gx-inline" data-gx-figure="reach">
      <figcaption>Reach is a certificate for a whole prefix, not a committed jump</figcaption>
      <svg className="gx-reach-figure" viewBox="0 0 280 315" role="img" aria-label="Jump limits 2, 3, 0, 0, 1, 0. After inspecting index 0, the reachable prefix ends at 2; after index 1, at 4; after index 4, at 5. The amber dot is the inspected index, not a committed landing choice.">
        <text x="0" y="14" className="gx-reach-label">Index → maximum jump length</text>
        {model.values.map((value, index) => <g key={index}>
          <text x={column(index)} y="37" className="gx-reach-index">{index}</text>
          <text x={column(index)} y="59" className="gx-reach-value">{value}</text>
        </g>)}
        {stages.map((stage, row) => {
          const y = 88 + row * 80;
          return <g key={stage.index} data-reach={stage.reach} data-inspected={stage.index}>
            <text x="0" y={y} className="gx-reach-label">Inspect {stage.index}: certify indices 0–{stage.reach}</text>
            <line x1={column(0)} x2={column(5)} y1={y + 25} y2={y + 25} className="gx-time-grid" />
            <rect x="11" y={y + 10} width={stage.reach * 45 + 32} height="30" rx="3" className="gx-reach-band" />
            {model.values.map((_, index) => <g key={index}>
              <circle cx={column(index)} cy={y + 25} r="5" className={index === stage.index ? 'gx-reach-inspected' : index <= stage.reach ? 'gx-reach-known' : 'gx-reach-unknown'} />
              <text x={column(index)} y={y + 53} className="gx-reach-index">{index}</text>
            </g>)}
          </g>;
        })}
      </svg>
      <p className="gx-note">Green band = every index certified reachable. Amber dot = index being inspected. The band can expand from a shorter possible jump; it is not one chosen path.</p>
      <div className="gx-reach-events"><p><strong>Inspect 0:</strong> reach 0→2. Indices 1 and 2 both become possible.</p><p><strong>Inspect 1:</strong> reach 2→4. Using index 1 beats committing directly to the zero at index 2.</p><p><strong>Inspect 4:</strong> reach 4→5. The last index is reachable.</p></div>
      <p>Each value is the maximum allowed forward jump; shorter jumps are allowed. The code does not have to store or try each path to maintain this reachable-prefix invariant.</p>
    </figure>;
}
