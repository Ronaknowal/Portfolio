import { useId, useState } from 'react';
import { MARKOV_PRESETS, finiteProcessLaw, markovState, absorptionState, arrivalState, integratedRate, jumpClockState, brownianState, bridgeState, parseProcessNumber } from '../../data/stochastic-processes-models.js';
import './stochastic-processes-labs.css';
function number(value, digits = 4) {
  if (value === null) return 'not identified';
  if (value !== 0 && Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  return String(Number(value.toFixed(digits)));
}
function Lab({
  name,
  title,
  children
}) {
  return <section className="process-lab" aria-label={name}>
    <p className="process-eyebrow">Investigate the mechanism</p><h3>{title}</h3>{children}
  </section>;
}
function Figure({
  title,
  height = 220,
  width = 320,
  children,
  wide = false
}) {
  const id = useId();
  return <div className={'process-figure-scroll' + (wide ? ' process-wide' : '')} tabIndex={wide ? 0 : undefined} role={wide ? 'region' : undefined} aria-label={wide ? title : undefined}>
    <svg viewBox={'0 0 ' + width + ' ' + height} className="process-figure" style={wide ? {
      minWidth: width
    } : undefined} role="img" aria-labelledby={id}>
      <title id={id}>{title}</title>{children}
    </svg>
  </div>;
}
function Table({
  caption,
  headers,
  rows,
  compact = false
}) {
  return <div className={'process-table-scroll' + (compact ? ' process-compact-table' : '')} role="region" aria-label={caption} tabIndex={0}>
    <table><caption>{caption}</caption><thead><tr>{headers.map(h => <th scope="col" key={h}>{h}</th>)}</tr></thead>
      <tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => j === 0 ? <th key={j} scope="row">{cell}</th> : <td key={j}>{cell}</td>)}</tr>)}</tbody></table>
  </div>;
}
function Select({
  label,
  value,
  onChange,
  children,
  disabled = false
}) {
  return <label>{label}<select value={value} disabled={disabled} onChange={event => onChange(event.target.value)}>{children}</select></label>;
}
function Plot({
  title,
  points,
  xmax,
  ymin = 0,
  ymax = 1,
  axis = 'Step',
  ordinate = 'Probability',
  children
}) {
  const x = value => 45 + value / xmax * 250;
  const y = value => 170 - (value - ymin) / (ymax - ymin) * 135;
  return <Figure title={title} height={235}>
    <text x="45" y="18">{ordinate}</text>
    {[ymin, (ymin + ymax) / 2, ymax].map(value => <g key={value}>
      <line x1="45" x2="295" y1={y(value)} y2={y(value)} className="process-grid" />
      <text x="38" y={y(value) + 5} textAnchor="end">{number(value, 2)}</text>
    </g>)}
    {[0, xmax / 2, xmax].map(value => <text key={value} x={x(value)} y="196" textAnchor="middle">{number(value, 2)}</text>)}
    <text x="170" y="227" textAnchor="middle">{axis}</text>
    {points && <polyline points={points.map(point => x(point[0]) + ',' + y(point[1])).join(' ')} className="process-line" />}
    {children?.({
      x,
      y
    })}
  </Figure>;
}
export function ProcessSliceFigure() {
  const families = [['fresh', 'Redraw each time', [[0, 1, 1, 0], [1, 0, 0, 1]]], ['frozen', 'Copy one initial draw', [[0, 0, 0, 0], [1, 1, 1, 1]]], ['alternating', 'Alternate after a fair start', [[0, 1, 0, 1], [1, 0, 1, 0]]]];
  return <figure className="process-inline">
    <div className="process-path-families">{families.map(([kind, title, paths]) => {
        const law = finiteProcessLaw(kind);
        return <div key={kind}><h4>{title}</h4>
        <p className="process-small">{kind === 'fresh' ? 'Two examples from 16 equally likely paths' : 'The two paths, each with probability ½'}</p>
        {paths.map((path, index) => <div className="process-symbol-row" key={index}>
          {path.map((value, time) => <span className={time === 2 ? 'process-selected' : ''} key={time} aria-label={'Time ' + time + ', value ' + value}>{value}</span>)}
        </div>)}
        <p>At each time: P(1)=½.<br />P(next equals current)={number(law.adjacentEqual)}.</p>
      </div>;
      })}</div>
    <figcaption>Read across for a path; read down a fixed time column for a distribution. All three have fair one-time marginals. Their relationships across time differ.</figcaption>
  </figure>;
}
export function StateAggregationFigure() {
  return <figure className="process-inline">
    <Figure title="A deterministic hidden cycle A to B to C to A; reported A and B both become zero." height={245}>
      <text x="160" y="24" textAnchor="middle">Hidden state: enough memory</text>
      {[['A', 50], ['B', 160], ['C', 270]].map(([name, x]) => <g key={name}>
        <circle cx={x} cy="70" r="23" className="process-node" />
        <text x={x} y="76" textAnchor="middle">{name}</text>
        <path d={'M' + x + ' 100 L' + x + ' 130'} className="process-connector" />
      </g>)}
      <text x="105" y="76" textAnchor="middle">→</text>
      <text x="215" y="76" textAnchor="middle">→</text>
      <path d="M270 45 L270 34 L50 34 L50 45" className="process-connector" />
      <text x="160" y="42" textAnchor="middle">←</text>
      {[['0', 50], ['0', 160], ['1', 270]].map(([value, x]) => <text x={x} y="155" textAnchor="middle" key={x}>report {value}</text>)}
      <text x="160" y="193" textAnchor="middle">previous 1, current 0 → next 0</text>
      <text x="160" y="224" textAnchor="middle">previous 0, current 0 → next 1</text>
    </Figure>
    <figcaption>C returns to A along the upper connector. Reporting only 0 or 1 loses which zero-state you occupy. The previous report restores information that the current report alone lacks.</figcaption>
  </figure>;
}
export function MarkovPropagationLab() {
  const [draftA, setDraftA] = useState('0.2');
  const [draftB, setDraftB] = useState('0.3');
  const [config, setConfig] = useState({
    a: 0.2,
    b: 0.3,
    initialSunny: 1
  });
  const [step, setStep] = useState(1);
  const [selectedFlow, setSelectedFlow] = useState(0);
  const [error, setError] = useState('');
  const result = markovState({
    ...config,
    steps: 60
  });
  const current = result.history[step];
  const previous = result.history[Math.max(0, step - 1)];
  const names = ['Sunny', 'Rainy'];
  const source = Math.floor(selectedFlow / 2);
  const destination = selectedFlow % 2;
  function preset(key) {
    const entry = MARKOV_PRESETS[key];
    setConfig({
      a: entry.a,
      b: entry.b,
      initialSunny: 1
    });
    setDraftA(String(entry.a));
    setDraftB(String(entry.b));
    setStep(1);
    setError('');
  }
  function apply(event) {
    event.preventDefault();
    try {
      const next = {
        ...config,
        a: parseProcessNumber(draftA, 0, 1, 'Sunny to Rainy'),
        b: parseProcessNumber(draftB, 0, 1, 'Rainy to Sunny')
      };
      markovState(next);
      setConfig(next);
      setStep(1);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  return <Lab name="Markov probability propagation" title="Move probability, then ask whether it settles">
    <p>Predict: the sticky chain switches ten times less often. Will its final balance also change?</p>
    <div className="process-controls">
      <Select label="Transition preset" value="" onChange={preset}>
        <option value="" disabled>Choose a comparison</option>
        {Object.entries(MARKOV_PRESETS).map(([key, entry]) => <option key={key} value={key}>{entry.name}</option>)}
      </Select>
      <Select label="Initial sunny probability" value={config.initialSunny} onChange={value => {
        setConfig({
          ...config,
          initialSunny: Number(value)
        });
        setStep(1);
      }}>
        <option value="1">1 — start sunny</option><option value="0.5">½ — fair start</option>
        <option value="0">0 — start rainy</option>
      </Select>
    </div>
    <form className="process-controls" onSubmit={apply}>
      <label>Sunny → Rainy, a<input value={draftA} onChange={event => setDraftA(event.target.value)} inputMode="decimal" /></label>
      <label>Rainy → Sunny, b<input value={draftB} onChange={event => setDraftB(event.target.value)} inputMode="decimal" /></label>
      <button type="submit">Apply transition edits</button>
    </form>
    {error && <p role="alert" className="process-error">{error} The prior applied matrix remains active.</p>}
    <div className="process-actions">
      <button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous step</button>
      <span aria-live="polite">Step {step} of 60</span>
      <button disabled={step === 60} onClick={() => setStep(step + 1)}>Next step</button>
      <button onClick={() => preset('weather')}>Reset weather</button>
    </div>
    <div className="process-pair">
      <Figure title={'Applied transitions: Sunny to Rainy ' + config.a + ', Rainy to Sunny ' + config.b} height={250}>
        <path d="M102 92 Q160 44 218 92" className="process-connector" />
        <text x="160" y="64" textAnchor="middle">a={number(config.a)} →</text>
        <path d="M218 123 Q160 172 102 123" className="process-connector" />
        <text x="160" y="164" textAnchor="middle">← b={number(config.b)}</text>
        {[70, 250].map((x, i) => <g key={i}>
          <circle cx={x} cy="107" r="34" className={'process-node process-state-' + i} />
          <text x={x} y="113" textAnchor="middle">{names[i]}</text>
          <text x={x} y="207" textAnchor="middle">mass {number(current.distribution[i])}</text>
          <text x={x} y="234" textAnchor="middle">stay {number(result.matrix[i][i])}</text>
        </g>)}
      </Figure>
      <div>
        <Table caption="Applied transition matrix: row now, column next" headers={['From / to', 'Sunny', 'Rainy']} rows={result.matrix.map((row, i) => [names[i], ...row.map(value => number(value))])} />
        <p>Choose a contribution to inspect:</p>
        <div className="process-flow-options">{[0, 1, 2, 3].map(index => <button key={index} aria-pressed={index === selectedFlow} onClick={() => setSelectedFlow(index)}>
            {names[Math.floor(index / 2)]} → {names[index % 2]}
          </button>)}</div>
      </div>
    </div>
    {step === 0 ? <p>At time zero this is the chosen initial law. No transition has happened.</p> : <div className="process-mass-flow">
        <p aria-live="polite">{names[source]} mass {number(previous.distribution[source])} × transition {number(result.matrix[source][destination])}
          {' = '}{number(current.flows[source][destination])} arriving in {names[destination]}.</p>
        {current.flows.flatMap((row, i) => row.map((mass, j) => <div className={i * 2 + j === selectedFlow ? 'process-flow-row process-active' : 'process-flow-row'} key={i + '-' + j}>
            <span>{names[i][0]} → {names[j][0]}</span><div><i style={{
            width: mass * 100 + '%'
          }} /></div><strong>{number(mass)}</strong>
          </div>))}
        <p>Destination sums: Sunny {number(current.flows[0][0])}+{number(current.flows[1][0])}={number(current.distribution[0])};
          Rainy {number(current.flows[0][1])}+{number(current.flows[1][1])}={number(current.distribution[1])}.</p>
      </div>}
    <Plot title="Calculated sunny probability for steps zero to sixty" xmax={60} points={result.history.map(state => [state.step, state.distribution[0]])}>
      {({
        x,
        y
      }) => <>
        {result.stationary && <line x1={x(0)} x2={x(60)} y1={y(result.stationary[0])} y2={y(result.stationary[0])} className="process-reference" />}
        <circle cx={x(step)} cy={y(current.distribution[0])} r="5" className="process-point" />
      </>}
    </Plot>
    <p>{result.stationary ? 'The dashed stationary sunny probability is ' + number(result.stationary[0]) + '.' : 'Every distribution is stationary for the identity matrix. There is no unique dashed target.'}
      {' '}{result.periodTwo ? 'This chain is irreducible but periodic: a nonstationary start alternates forever.' : result.a + result.b === 0 ? 'The initial law stays exactly where it began.' : 'The plotted law converges to its stationary distribution.'}</p>
    <p className="process-small">These are computed probabilities, not one simulated weather sequence. Exact zero/one transitions are deliberate; other probabilities stay at least 10⁻⁶ from either endpoint in this calculator.</p>
  </Lab>;
}
export function ClassStructureFigure() {
  return <figure className="process-inline">
    <div className="process-class-diagrams">
      <div><h4>A closed singleton</h4><p className="process-diagram-text">A → B ↻</p><p>B absorbs. A is transient.</p></div>
      <div><h4>A closed cycle</h4><p className="process-diagram-text">A ⇄ B</p><p>Always switch: irreducible, period 2.</p></div>
      <div><h4>Two closed classes</h4><p className="process-diagram-text">A ↻　 B ↻</p><p>No switching: initial mass never mixes.</p></div>
    </div>
    <figcaption>Edges mean positive-probability transitions. A stationary distribution can exist in all three cases; the limiting behavior and dependence on the start differ.</figcaption>
  </figure>;
}
export function AbsorptionLab() {
  const [boundary, setBoundary] = useState(4);
  const [start, setStart] = useState(2);
  const [upward, setUpward] = useState(0.5);
  const [step, setStep] = useState(4);
  const [selected, setSelected] = useState(2);
  const result = absorptionState({
    boundary,
    start,
    upward,
    steps: 60
  });
  const current = result.history[step];
  const width = (boundary + 1) * 52 + 20;
  function reset() {
    setBoundary(4);
    setStart(2);
    setUpward(0.5);
    setStep(4);
    setSelected(2);
  }
  return <Lab name="First passage and absorption" title="Watch moving probability reach its first boundary">
    <p>A reserve moves one unit up or down per step. At 0 or the upper boundary it stops. Predict what a lower upward probability does to success <em>and</em> time to either boundary.</p>
    <div className="process-controls">
      <Select label="Upper reserve boundary" value={boundary} onChange={value => {
        const next = Number(value);
        setBoundary(next);
        setStart(Math.min(start, next));
        setSelected(Math.min(selected, next));
      }}>{[3, 4, 5, 6, 7, 8].map(value => <option key={value}>{value}</option>)}</Select>
      <Select label="Starting reserve" value={start} onChange={value => setStart(Number(value))}>
        {Array.from({
          length: boundary + 1
        }, (_, i) => <option key={i}>{i}</option>)}
      </Select>
      <Select label="Upward probability" value={upward} onChange={value => setUpward(Number(value))}>
        {[0.1, 0.25, 0.5, 0.75, 0.9].map(value => <option key={value}>{value}</option>)}
      </Select>
    </div>
    <div className="process-actions">
      <button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous step</button>
      <span aria-live="polite">Step {step}</span>
      <button disabled={step === 60} onClick={() => setStep(step + 1)}>Next step</button>
      <button onClick={reset}>Reset reserve walk</button>
    </div>
    <Figure title={'Reserve probability at step ' + step + '. Boundaries zero and ' + boundary + ' absorb.'} width={width} height={170} wide={width > 310}>
      {Array.from({
        length: boundary + 1
      }, (_, state) => {
        const x = 36 + state * 52;
        return <g key={state}>
          {state < boundary && <line x1={x + 18} x2={x + 34} y1="54" y2="54" className="process-connector" />}
          <circle cx={x} cy="54" r="18" className={state === 0 || state === boundary ? 'process-node process-boundary' : 'process-node'} />
          <text x={x} y="60" textAnchor="middle">{state}</text>
          <rect x={x - 15} y={120 - 40 * current.distribution[state]} width="30" height={40 * current.distribution[state]} className="process-bar" />
          <text x={x} y="147" textAnchor="middle">{number(current.distribution[state], 3)}</text>
        </g>;
      })}
    </Figure>
    {width > 310 && <p className="process-small">Scroll the reserve line horizontally to inspect all states. The labels below the nodes are their current probabilities.</p>}
    <div className="process-mass-strip" aria-label="Absorbed lower, still moving, absorbed upper">
      <span style={{
        flex: current.distribution[0]
      }} className="process-lower" />
      <span style={{
        flex: current.surviving
      }} className="process-surviving" />
      <span style={{
        flex: current.distribution[boundary]
      }} className="process-upper" />
    </div>
    <p aria-live="polite">Lower boundary: {number(current.distribution[0])}; still moving: {number(current.surviving)};
      upper boundary: {number(current.distribution[boundary])}. Total probability: {number(current.distribution.reduce((a, b) => a + b, 0))}.</p>
    <p>First hits <em>at this step</em>: lower {number(current.firstLower)}, upper {number(current.firstUpper)}.
      The rest of the absorbed mass arrived earlier.</p>
    <Plot title="Surviving transient probability versus step, calculated from the finite chain" xmax={60} points={result.history.map(state => [state.step, state.surviving])} />
    <div className="process-controls">
      <Select label="Inspect first-step equations at reserve" value={selected} onChange={value => setSelected(Number(value))}>
        {Array.from({
          length: boundary + 1
        }, (_, i) => <option key={i}>{i}</option>)}
      </Select>
    </div>
    {selected === 0 || selected === boundary ? <p>This is already a boundary: time to either boundary is 0; upper-boundary success is {selected === boundary ? 1 : 0}.</p> : <div className="process-equation-lines">
        <p>h({selected}) = {number(1 - upward)} h({selected - 1}) + {number(upward)} h({selected + 1}) = {number(result.success[selected])}</p>
        <p>t({selected}) = 1 + {number(1 - upward)} t({selected - 1}) + {number(upward)} t({selected + 1}) = {number(result.meanSteps[selected])} steps</p>
      </div>}
    <p>From the chosen start {start}: eventual success {number(result.success[start])};
      mean time to <em>either</em> boundary {number(result.meanSteps[start])} steps.
      A still-moving path at step {step} is censored, not a permanent failure.</p>
    <details><summary>Inspect exact-model probabilities and expected transient visits</summary>
      <Table caption="Solved first-step system" headers={['Reserve', 'Upper success h', 'Mean steps t']} rows={result.success.map((h, i) => [i, number(h, 6), number(result.meanSteps[i], 6)])} />
      <Table caption="Fundamental matrix: expected visits before absorption, including time zero" headers={['Start / visit', ...Array.from({
        length: boundary - 1
      }, (_, i) => String(i + 1))]} rows={result.visits.map((row, i) => [i + 1, ...row.map(value => number(value))])} />
      <Table caption="First-hit probability and survival at every displayed step" headers={['Step', 'First lower', 'First upper', 'Surviving']} rows={result.history.slice(0, step + 1).map(state => [state.step, number(state.firstLower), number(state.firstUpper), number(state.surviving)])} />
    </details>
  </Lab>;
}
export function PoissonArrivalLab() {
  const [rates, setRates] = useState([2.5, 2.5]);
  const [draft, setDraft] = useState(['2.5', '2.5']);
  const [window, setWindow] = useState('0,1');
  const [routing, setRouting] = useState('independent');
  const [split, setSplit] = useState(0.5);
  const [seed, setSeed] = useState(11);
  const [selected, setSelected] = useState(0);
  const [error, setError] = useState('');
  const interval = window.split(',').map(Number);
  const result = arrivalState({
    rates,
    interval,
    seed,
    routing,
    splitProbability: split
  });
  const event = result.events[Math.min(selected, result.events.length - 1)];
  const x = time => 30 + time / 3 * 260;
  function apply(event) {
    event.preventDefault();
    try {
      const next = draft.map((value, index) => parseProcessNumber(value, 0, 6, 'Rate ' + (index + 1)));
      arrivalState({
        rates: next
      });
      setRates(next);
      setSelected(0);
      setError('');
    } catch (issue) {
      setError(issue.message);
    }
  }
  function reset() {
    setRates([2.5, 2.5]);
    setDraft(['2.5', '2.5']);
    setWindow('0,1');
    setRouting('independent');
    setSplit(0.5);
    setSeed(11);
    setSelected(0);
    setError('');
  }
  const staircase = [[0, 0]];
  result.events.forEach((arrival, index) => staircase.push([arrival.time, index], [arrival.time, index + 1]));
  staircase.push([result.observedUntil, result.events.length]);
  return <Lab name="Arrival clocks and event counts" title="One event stream, several connected views">
    <p>Predict first: if only the routing rule changes, should any arrival time move?
      The clock and the marks use separate seeded draws. Rates are events per minute; the horizon is 3 minutes.</p>
    <div className="process-controls">
      <Select label="Arrival rate preset" value="" onChange={value => {
        const next = value.split(',').map(Number);
        setRates(next);
        setDraft(next.map(String));
        setSelected(0);
        setError('');
      }}>
        <option value="" disabled>Choose a rate schedule</option>
        <option value="2.5,2.5">Constant: 2.5 then 2.5</option>
        <option value="1,4">Scheduled: 1 then 4</option>
        <option value="0,4">No arrivals before minute 2</option>
        <option value="0,0">Zero rate throughout</option>
      </Select>
      <Select label="Count interval, left open and right closed" value={window} onChange={setWindow}>
        <option value="0,1">(0, 1]</option><option value="1,2">(1, 2]</option>
        <option value="2,3">(2, 3]</option><option value="0,3">(0, 3]</option>
      </Select>
    </div>
    <form onSubmit={apply} className="process-controls">
      {draft.map((value, index) => <label key={index}>{index === 0 ? 'Rate before minute 2' : 'Rate after minute 2'}
        <input type="text" inputMode="decimal" value={value} onChange={event => setDraft(draft.map((old, i) => i === index ? event.target.value : old))} />
      </label>)}
      <button type="submit">Apply rate edits</button>
    </form>
    <p className="process-small">Each rate is 0 or 0.01–6. Empty or invalid edits leave the last valid stream in place.</p>
    {error && <p role="alert">{error}</p>}
    <div className="process-controls">
      <Select label="Routing rule" value={routing} onChange={setRouting}>
        <option value="independent">Independent A/B marks</option><option value="alternating">Alternate A, B, A, B</option>
      </Select>
      <Select label="Independent probability of A" value={split} disabled={routing !== 'independent'} onChange={value => setSplit(Number(value))}>
        {[0, 0.25, 0.5, 0.75, 1].map(value => <option key={value}>{value}</option>)}
      </Select>
      <button onClick={() => {
        setSeed(seed + 1);
        setSelected(0);
      }}>Draw another event stream</button>
      <button onClick={reset}>Reset arrival clock</button>
    </div>
    <Figure title="Actual event times on A and B routing lanes; highlighted interval is left open, right closed" height={240}>
      <rect x={x(interval[0])} y="35" width={x(interval[1]) - x(interval[0])} height="135" className="process-window" />
      {[0, 1].map(mark => <g key={mark}>
        <text x="12" y={75 + mark * 70}>{mark === 0 ? 'A' : 'B'}</text>
        <line x1="30" x2="290" y1={70 + mark * 70} y2={70 + mark * 70} className="process-grid" />
      </g>)}
      {result.events.map((arrival, index) => <g key={index}>
        <line x1={x(arrival.time)} x2={x(arrival.time)} y1={55 + arrival.mark * 70} y2={85 + arrival.mark * 70} className={arrival === event ? 'process-event-selected' : 'process-event'} />
      </g>)}
      {[0, 1, 2, 3].map(time => <text key={time} x={x(time)} y="195" textAnchor="middle">{time}</text>)}
      <text x="160" y="229" textAnchor="middle">Time (minutes)</text>
    </Figure>
    <div className="process-controls">
      <Select label="Inspect arrival" value={event ? result.events.indexOf(event) : ''} disabled={!event} onChange={value => setSelected(Number(value))}>
        {!event && <option value="">No arrivals in this realization</option>}
        {result.events.map((arrival, i) => <option key={i} value={i}>Event {i + 1}: minute {number(arrival.time, 3)}</option>)}
      </Select>
    </div>
    {event && <p aria-live="polite">Event {event.index}: time {number(event.time)} min,
      preceding gap {number(event.gap)} min, accumulated intensity Λ(t)={number(event.unitTime)},
      route {event.mark === 0 ? 'A' : 'B'}. Its tick is highlighted above.</p>}
    <p>Selected interval: mean count {number(result.intervalMean)}, observed count {result.intervalCount ?? 'unknown beyond work cap'}.
      {result.markedCounts && <> A: {result.markedCounts[0]}; B: {result.markedCounts[1]}.</>}
      {' '}P(no events)={number(result.zeroProbability, 6)}.</p>
    <div className="process-plot-pair">
      <div><h4>Count the jumps</h4>
        <Plot title="Right-continuous cumulative event count on this realized path" points={staircase} xmax={3} ymax={Math.max(1, result.events.length)} axis="Time (minutes)" ordinate="Event count N(t)" />
        <p className="process-small">N(t) includes an event at t. N(t)−N(s) counts (s,t]. The connecting horizontal lines mean no new event.</p>
      </div>
      <div><h4>Accumulate the rate</h4>
        <Plot title="Exact integrated piecewise constant rate; slope changes at minute 2" points={[0, 2, 3].map(time => [time, integratedRate(time, rates, 2)])} xmax={3} ymax={Math.max(1, result.totalIntensity)} axis="Time (minutes)" ordinate="Expected count Λ(t)" />
        <p className="process-small">Λ(3)={number(result.totalIntensity)} expected events.
          The slope is {rates[0]} before minute 2 and {rates[1]} after it.
          A flat section contributes no event-clock time.</p>
      </div>
    </div>
    <p>{routing === 'independent' ? 'Independent marking gives independent Poisson output processes. Given a positive total and a routing probability strictly between 0 and 1, their counts become dependent. At probability 0 or 1, or given total zero, the conditional counts are deterministic.' : 'Alternating marks are a counterexample: the labels depend on event order. Successive A waits add two original waits, so the thinned streams are not Poisson.'}</p>
    {!result.complete && <p role="status">The 120-event work cap was reached at minute {number(result.observedUntil)}.
      Later counts are unknown; the drawing is deliberately incomplete.</p>}
    <details><summary>Inspect the events and count distribution</summary>
      <p className="process-small">Seed {seed}; seeded illustration, not measured traffic. The final gap to the horizon is censored: it is not a completed exponential wait.</p>
      <Table caption="Realized arrivals" headers={['Event', 'Minute', 'Gap', 'Λ(time)', 'Route']} rows={result.events.map(arrival => [arrival.index, number(arrival.time), number(arrival.gap), number(arrival.unitTime), arrival.mark ? 'B' : 'A'])} />
      <Table caption="Theoretical count probabilities for the selected interval" headers={['Count', 'Probability']} rows={result.pmf.map((value, count) => [count, number(value, 6)])} />
      <p>Approximate probability above 40: {number(result.pmfTail, 6)} (summed through 160).
        {result.tailBelowFloatingPointRange && ' This positive tail is below this calculation’s floating-point range.'}</p>
    </details>
  </Lab>;
}
export function JumpClockLab() {
  const [rates, setRates] = useState([0.5, 2]);
  const [draft, setDraft] = useState(['0.5', '2']);
  const [horizon, setHorizon] = useState(4);
  const [initial, setInitial] = useState(0);
  const [seed, setSeed] = useState(11);
  const [error, setError] = useState('');
  const state = jumpClockState({
    alpha: rates[0],
    beta: rates[1],
    horizon,
    initial,
    seed
  });
  const x = time => 40 + time / horizon * 250;
  return <Lab name="Continuous time holding clocks" title="Count visits, then measure how long they last">
    <p>The device alternates On and Off whenever it jumps. Equal numbers of visits do not imply equal time spent in each state.</p>
    <form className="process-controls" onSubmit={event => {
      event.preventDefault();
      try {
        const next = draft.map((value, i) => parseProcessNumber(value, 0.05, 6, i ? 'Off exit rate' : 'On exit rate'));
        setRates(next);
        setError('');
      } catch (issue) {
        setError(issue.message);
      }
    }}>
      {draft.map((value, i) => <label key={i}>{i ? 'Off → On rate β per hour' : 'On → Off rate α per hour'}
        <input inputMode="decimal" value={value} onChange={event => setDraft(draft.map((old, j) => i === j ? event.target.value : old))} />
      </label>)}
      <button type="submit">Apply holding rates</button>
    </form>
    {error && <p role="alert">{error}</p>}
    <div className="process-controls">
      <Select label="Observation horizon in hours" value={horizon} onChange={value => setHorizon(Number(value))}>
        {[1, 4, 12].map(value => <option key={value}>{value}</option>)}
      </Select>
      <Select label="Initial device state" value={initial} onChange={value => setInitial(Number(value))}>
        <option value={0}>On</option><option value={1}>Off</option>
      </Select>
      <button onClick={() => setSeed(seed + 1)}>Draw another holding path</button>
      <button onClick={() => {
        setRates([0.5, 2]);
        setDraft(['0.5', '2']);
        setHorizon(4);
        setInitial(0);
        setSeed(11);
        setError('');
      }}>Reset holding clock</button>
    </div>
    <Figure title="Realized On and Off holding intervals; horizontal length is time spent in that state" height={230}>
      <text x="2" y="75">On</text><text x="2" y="145">Off</text>
      {[70, 140].map(y => <line key={y} x1="40" x2="290" y1={y} y2={y} className="process-grid" />)}
      {state.segments.map((segment, index) => <g key={index}>
        <line x1={x(segment.start)} x2={x(segment.end)} y1={70 + segment.state * 70} y2={70 + segment.state * 70} className={segment.state ? 'process-holding-off' : 'process-holding-on'} />
        {segment.jumped && <line x1={x(segment.end)} x2={x(segment.end)} y1="70" y2="140" className="process-connector" />}
      </g>)}
      {[0, horizon / 2, horizon].map(time => <text key={time} x={x(time)} y="190" textAnchor="middle">{time}</text>)}
      <text x="170" y="221" textAnchor="middle">Time (hours)</text>
    </Figure>
    <Table caption="One realized path versus model expectations" headers={['Quantity', 'On', 'Off']} rows={[['Mean full holding time (hours)', number(1 / rates[0]), number(1 / rates[1])], ['Observed time (hours)', number(state.exposure[0]), number(state.exposure[1])], ['Observed departures', state.departures[0], state.departures[1]], ['Long-run time fraction', number(state.stationaryOn), number(1 - state.stationaryOn)], ['Long-run jump-visit fraction', '½', '½']]} />
    <p>Expected On time over this finite horizon: {number(state.expectedOnExposure)} hours.
      Observed On fraction over the simulated portion: {number(state.exposure[0] / state.observedUntil)}.
      One short path need not resemble either expectation or long-run proportion.</p>
    <Plot title="Exact probability that the device is On as a function of time; not the single realized path" points={state.history.map(row => [row.time, row.probabilityOn])} xmax={horizon} axis="Time (hours)">
      {({
        x,
        y
      }) => <line x1={x(0)} x2={x(horizon)} y1={y(state.stationaryOn)} y2={y(state.stationaryOn)} className="process-equilibrium" />}
    </Plot>
    <p className="process-small">Solid curve: ensemble P(On at t). Dashed line: equilibrium {number(state.stationaryOn)}.
      Holding intervals above are a seeded realization. Rates 0.05–6 per hour keep this finite investigation bounded.</p>
    {!state.complete && <p role="status">The 120-jump cap ended the path at hour {number(state.observedUntil)}.
      Remaining exposure is unknown.</p>}
    <details><summary>Inspect generator and holding intervals</summary>
      <Table caption="Generator: rates, not transition probabilities" headers={['From / to', 'On', 'Off']} rows={state.generator.map((row, i) => [i ? 'Off' : 'On', ...row.map(value => number(value))])} />
      <Table caption="Seeded holding intervals; final interval is clipped at the observation horizon" headers={['State', 'Start', 'Observed end', 'Full drawn hold', 'Departure seen?']} rows={state.segments.map(segment => [segment.state ? 'Off' : 'On', number(segment.start), number(segment.end), number(segment.holdingTime), segment.jumped ? 'Yes' : 'No'])} />
    </details>
  </Lab>;
}
export function BrownianPathLab() {
  const [horizon, setHorizon] = useState(1);
  const [drift, setDrift] = useState(0);
  const [scale, setScale] = useState(1);
  const [level, setLevel] = useState(4);
  const [pathIndex, setPathIndex] = useState(0);
  const [seed, setSeed] = useState(11);
  const [increment, setIncrement] = useState(0);
  const result = brownianState({
    horizon,
    drift,
    scale,
    level,
    pathIndex,
    seed
  });
  const i = Math.min(increment, result.count - 1);
  const extent = Math.max(5 * Math.sqrt(horizon) + Math.abs(drift) * horizon, ...result.paths.flat().map(value => Math.abs(value)), 1.96 * scale * Math.sqrt(horizon) + Math.abs(drift) * horizon);
  const band = time => 1.96 * scale * Math.sqrt(time);
  return <Lab name="Brownian paths and coupled refinement" title="Refine the grid without replacing the path">
    <p>Predict: doubling the number of intervals should preserve every old point.
      Each seed stores 256 increments per path; coarser views sum those same increments.</p>
    <div className="process-controls">
      <Select label="Brownian horizon T" value={horizon} onChange={value => setHorizon(Number(value))}>
        {[1, 2, 4].map(value => <option key={value}>{value}</option>)}
      </Select>
      <Select label="Drift μ, units per time" value={drift} onChange={value => setDrift(Number(value))}>
        {[-0.5, 0, 0.5].map(value => <option key={value}>{value}</option>)}
      </Select>
      <Select label="Diffusion σ, units per square root time" value={scale} onChange={value => setScale(Number(value))}>
        {[0.5, 1, 2].map(value => <option key={value}>{value}</option>)}
      </Select>
      <Select label="Number of grid intervals" value={level} onChange={value => {
        setLevel(Number(value));
        setIncrement(0);
      }}>
        {[2, 3, 4, 5, 6, 7, 8].map(value => <option key={value} value={value}>{2 ** value}</option>)}
      </Select>
      <Select label="Highlighted Brownian path" value={pathIndex} onChange={value => setPathIndex(Number(value))}>
        {Array.from({
          length: 8
        }, (_, index) => <option key={index} value={index}>Path {index + 1}</option>)}
      </Select>
      <button onClick={() => setSeed(seed + 1)}>Draw another Brownian family</button>
      <button onClick={() => {
        setHorizon(1);
        setDrift(0);
        setScale(1);
        setLevel(4);
        setPathIndex(0);
        setSeed(11);
        setIncrement(0);
      }}>Reset Brownian paths</button>
    </div>
    <Plot title="Eight exact-at-grid Brownian paths with drift; highlighted interval and pointwise normal bands" xmax={horizon} ymin={-extent} ymax={extent} axis="Time" ordinate="Position X(t)">
      {({
        x,
        y
      }) => <>
        <polygon points={[...result.times.map(time => [time, drift * time + band(time)]), ...[...result.times].reverse().map(time => [time, drift * time - band(time)])].map(([t, value]) => x(t) + ',' + y(value)).join(' ')} className="process-band" />
        {result.paths.map((path, index) => <polyline key={index} points={path.map((value, j) => x(result.times[j]) + ',' + y(value)).join(' ')} className={index === pathIndex ? 'process-line' : 'process-ghost-path'} />)}
        <line x1={x(result.times[i])} x2={x(result.times[i + 1])} y1={y(result.selected[i])} y2={y(result.selected[i + 1])} className="process-highlight-increment" />
        <circle cx={x(result.times[i])} cy={y(result.selected[i])} r="4" className="process-point" />
        <circle cx={x(result.times[i + 1])} cy={y(result.selected[i + 1])} r="4" className="process-point" />
      </>}
    </Plot>
    <p className="process-small">Shading is a 95% interval for X(t) at each fixed t under this model;
      it is not a band containing 95% of entire paths. Lines between grid points only connect sampled values.
      The vertical scale may expand to keep all displayed points and bands visible.</p>
    <div className="process-controls">
      <Select label="Inspect Brownian increment" value={i} onChange={value => setIncrement(Number(value))}>
        {result.increments.map((_, index) => <option key={index} value={index}>Interval {index + 1}: {number(result.times[index], 3)} to {number(result.times[index + 1], 3)}</option>)}
      </Select>
    </div>
    <p aria-live="polite">Δt={number(result.dt)}; observed ΔX={number(result.increments[i])}.
      Its model mean is μΔt={number(drift * result.dt)} and standard deviation is
      {' '}σ√Δt={number(scale * Math.sqrt(result.dt))}.
      The endpoint X(T)={number(result.selected.at(-1))} is retained when you change only grid resolution.</p>
    <p>Raw sums square each ΔX. Centered sums first subtract μΔt from each increment, then square.</p>
    <Table compact caption="Squared increments: realized value and exact finite-grid moments" headers={['Sum', 'Observed', 'Expected']} rows={[['Raw', number(result.rawVariation), number(result.rawVariationExpectation)], ['Centered', number(result.centeredVariation), number(result.centeredVariationExpectation)]]} />
    <p>Variance of the centered sum: {number(result.centeredVariationVariance)}.
      Refinement reduces its mean-square error around σ²T, but the realized sums need not move monotonically.</p>
    <details><summary>Inspect covariance and grid values</summary>
      <Table caption="Exact covariance σ² min(s,t), unaffected by drift" headers={['s / t', ...result.covarianceTimes.map(time => number(time))]} rows={result.covariance.map((row, index) => [number(result.covarianceTimes[index]), ...row.map(value => number(value))])} />
      <Table caption="Selected grid path and its increments" headers={['Time', 'X(t)', 'Increment ending here']} rows={result.times.map((time, index) => [number(time), number(result.selected[index]), index ? number(result.increments[index - 1]) : 'Start'])} />
      <p>Seed {seed}; synthetic seeded normal draws. Terminal mean {number(result.terminalMean)},
        variance {number(result.terminalVariance)}. The grid calculations come from the displayed model, not measurements.</p>
    </details>
  </Lab>;
}
export function BrownianCovarianceFigure() {
  return <figure className="process-inline">
    <Figure title="Shared increments cause Brownian values at two times to be correlated" height={210}>
      <text x="12" y="30">W(s)</text>
      <rect x="75" y="9" width="105" height="30" className="process-bar" />
      <text x="127" y="31" textAnchor="middle" className="process-on-fill">Shared part</text>
      <text x="12" y="90">W(t)</text>
      <rect x="75" y="69" width="105" height="30" className="process-bar" />
      <rect x="180" y="69" width="120" height="30" className="process-independent-fill" />
      <text x="240" y="91" textAnchor="middle">New part</text>
      <line x1="75" x2="300" y1="145" y2="145" className="process-connector" />
      <text x="75" y="171" textAnchor="middle">0</text>
      <text x="180" y="171" textAnchor="middle">s</text>
      <text x="300" y="171" textAnchor="middle">t</text>
      <text x="160" y="204" textAnchor="middle">W(t) = W(s) + [W(t)−W(s)]</text>
    </Figure>
    <figcaption>For 0≤s≤t, the new increment is independent of the shared part.
      Therefore Cov(W(s),W(t))=Var(W(s))=s. Independent increments do not mean independent levels.</figcaption>
  </figure>;
}
export function BrownianBridgeFigure() {
  const state = bridgeState();
  return <figure className="process-inline">
    <Figure title="Conditional Brownian bridge with endpoints zero and a barrier at one; crossing remains possible between observations" height={250}>
      <line x1="30" x2="290" y1="50" y2="50" className="process-barrier" />
      <text x="160" y="29" textAnchor="middle">Barrier b=1</text>
      <line x1="30" x2="290" y1="175" y2="175" className="process-grid" />
      <path d="M30 175 C75 160,90 15,140 40 S235 135,290 175" className="process-schematic" />
      {[30, 290].map(x => <circle key={x} cx={x} cy="175" r="5" className="process-point" />)}
      <text x="30" y="223" textAnchor="middle">0</text>
      <text x="160" y="238" textAnchor="middle">Time; Δ=1, σ=1</text>
      <text x="290" y="223" textAnchor="middle">1</text>
    </Figure>
    <figcaption>The curved line is a schematic possibility, not a sampled path or confidence band.
      Both observed endpoints are 0. At the midpoint, the conditional mean is {state.mean} and variance {state.variance}.
      The exact probability of crossing 1 somewhere between them is e⁻²≈{number(state.crossingProbability, 6)}.</figcaption>
  </figure>;
}
