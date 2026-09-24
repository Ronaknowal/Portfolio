import { useEffect, useRef, useState } from 'react';
import { defaultProbeProgram, denseSetTrace, hashFamilyState, probeProgramTrace, resizeScenarios, resizeTrace } from '../../data/hashing-amortized-models.js';
import './hashing-amortized-labs.css';
function TraceControls({
  position,
  count,
  setPosition,
  reset
}) {
  return <div className="hash-controls">
    <button disabled={!position} onClick={() => setPosition(position - 1)}>Previous state</button>
    <button disabled={position === count - 1} onClick={() => setPosition(position + 1)}>Next state</button>
    <button disabled={position === count - 1} onClick={() => setPosition(count - 1)}>Finish trace</button>
    <button onClick={reset}>Reset</button><span>State {position + 1} / {count}</span>
  </div>;
}
function SlotStrip({
  slots,
  visited = [],
  active = null,
  label
}) {
  const strip = useRef(null);
  useEffect(() => {
    const region = strip.current;
    if (!region) return;
    if (active === null) {
      region.scrollLeft = 0;
      return;
    }
    const cell = region.querySelector('.hash-active');
    if (!cell) return;
    const bounds = region.getBoundingClientRect();
    const current = cell.getBoundingClientRect();
    if (current.left < bounds.left + 4) region.scrollLeft -= bounds.left + 4 - current.left;else if (current.right > bounds.right - 4) region.scrollLeft += current.right - bounds.right + 4;
  }, [active, slots.length]);
  return <div ref={strip} className="hash-strip" role="region" tabIndex={0} aria-label={label}>
    <div className="hash-slots">{slots.map((slot, index) => <div key={index} className={`hash-slot ${slot === null ? 'hash-empty' : slot.deleted ? 'hash-deleted' : 'hash-occupied'} ${visited.includes(index) ? 'hash-visited' : ''} ${active === index ? 'hash-active' : ''}`}>
      <small>slot {index}</small>
      {slot === null ? <span>EMPTY</span> : slot.deleted ? <span>DELETED</span> : <><strong>{slot.key}</strong><span>→ {slot.value}</span></>}
    </div>)}</div>
  </div>;
}
const resultText = result => {
  if (result === null) return 'Operation not yet complete';
  if (typeof result === 'object' && !Array.isArray(result)) return result.found ? `found value ${result.value}` : 'missing';
  return JSON.stringify(result);
};
export function ProbeChainLab() {
  const [draft, setDraft] = useState(defaultProbeProgram);
  const [program, setProgram] = useState(defaultProbeProgram);
  const [capacity, setCapacity] = useState(8);
  const [faulty, setFaulty] = useState(false);
  const [position, setPosition] = useState(0);
  const [error, setError] = useState('');
  const trace = probeProgramTrace(program, capacity, faulty);
  const state = trace.states[position];
  function apply(event) {
    event.preventDefault();
    try {
      probeProgramTrace(draft, capacity, faulty);
      setProgram(draft);
      setPosition(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setDraft(defaultProbeProgram);
    setProgram(defaultProbeProgram);
    setCapacity(8);
    setFaulty(false);
    setPosition(0);
    setError('');
  }
  return <section className="hash-lab" data-lab="probe-chain" aria-label="Hash probe and deletion investigation">
    <h3>Follow the evidence that lookup is allowed to stop</h3>
    <p>Inspect where key 17 will be found after deleting key 1. Each step exposes a probe or commits an operation. Integer keys/values are limited to −999…999 and the script to 24 lines; an empty script is valid.</p>
    <form onSubmit={apply} className="hash-script-form">
      <label>Operation script<textarea aria-label="Operation script" rows={8} value={draft} onChange={event => setDraft(event.target.value)} spellCheck={false} /></label>
      <div className="hash-controls"><label>Initial capacity<select aria-label="Initial capacity" value={capacity} onChange={event => {
            setCapacity(Number(event.target.value));
            setPosition(0);
          }}>{[4, 8, 16].map(value => <option key={value}>{value}</option>)}</select></label><button type="submit">Apply script</button></div>
    </form>
    {error && <p role="alert">{error} The last applied script remains active.</p>}
    <label className="hash-checkbox"><input type="checkbox" checked={faulty} onChange={event => {
        setFaulty(event.target.checked);
        setPosition(0);
      }} />Try incorrect deletion: erase to EMPTY</label>
    <p className="hash-operation">{state.operationIndex < 0 ? 'Initialization' : `Operation ${state.operationIndex + 1}: ${state.operation}`}</p>
    <SlotStrip slots={state.slots} visited={state.visited} active={state.active} label="Probe table with empty, deleted and live slots" />
    <p className="hash-probe-path">Visited this operation: {state.visited.length ? state.visited.join(' → ') : 'none yet'}</p>
    <p data-result="probe-action">{state.action}</p>
    <dl className="hash-readout"><dt>Live / deleted / capacity</dt><dd data-result="probe-load">{state.live} / {state.deleted} / {state.capacity}</dd><dt>Live load / used load</dt><dd>{(state.live / state.capacity).toFixed(3)} / {((state.live + state.deleted) / state.capacity).toFixed(3)}</dd><dt>Total slot inspections</dt><dd>{state.totalProbes}</dd></dl>
    {state.phase === 'commit' && <div className={state.agrees ? 'hash-verdict' : 'hash-verdict hash-failure'}><p data-result="probe-result">Actual: {resultText(state.result)}. Required: {resultText(state.expected)}.</p><p>{state.agrees ? 'The operation result agrees with the reference mapping.' : 'The operation violates the reference mapping contract.'}</p></div>}
    <div data-result="probe-invariant" className={state.issues.length ? 'hash-verdict hash-failure' : 'hash-verdict'}>{state.issues.length ? <ul>{state.issues.map(issue => <li key={issue}>{issue}</li>)}</ul> : <p>✓ Every live key is reachable before an EMPTY slot, and each key occurs once.</p>}</div>
    <TraceControls position={position} count={trace.states.length} setPosition={setPosition} reset={reset} />
    <details><summary>Try full-table and duplicate-update cases</summary><p>Use capacity 4 and put four distinct keys before looking up an absent key: the explicit scan bound still terminates. Delete one key, then insert another without an EMPTY slot. Finally place a tombstone before an existing equal key and replace that key; the scan must continue past the reusable slot. A rebuild to a capacity smaller than the live count is rejected without changing storage.</p></details>
    <p className="hash-caption">Home = key modulo current capacity; probing advances one slot with wraparound. Inspections include reinsertion probes during explicit rebuilds. This bounded model is not Python dict's internal layout. The reference mapping does not excuse an invalid probe invariant.</p>
  </section>;
}
export function HashFamilyLab() {
  const [draft, setDraft] = useState('1, 5, 9, 13');
  const [keys, setKeys] = useState('1, 5, 9, 13');
  const [query, setQuery] = useState(9);
  const [multiplier, setMultiplier] = useState(1);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState('');
  const state = hashFamilyState(keys, multiplier, offset, query);
  const colors = ['#a5d6b3', '#86b7cf', '#cbb979', '#de8f75', '#bf8ca7', '#c89be2', '#e3c383', '#ded6bf'];
  function apply(event) {
    event.preventDefault();
    try {
      const nextQuery = Number(draft.split(',')[0]);
      hashFamilyState(draft, multiplier, offset, nextQuery);
      setKeys(draft);
      setQuery(nextQuery);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setDraft('1, 5, 9, 13');
    setKeys('1, 5, 9, 13');
    setQuery(9);
    setMultiplier(1);
    setOffset(0);
    setError('');
  }
  return <section className="hash-lab" data-lab="hash-family" aria-label="Exact finite hash family investigation">
    <h3>Keep the keys fixed; vary the chosen hash function</h3>
    <p>Every cell below is one of the 272 possible (a,b) choices. This is complete enumeration of a tiny family, not a random sample or benchmark. Keep p=17 and m=4; keys must be distinct integers from 0 to 16, at most eight.</p>
    <form onSubmit={apply} className="hash-controls"><label>Fixed keys<input aria-label="Fixed keys" value={draft} onChange={event => setDraft(event.target.value)} /></label><button type="submit">Apply keys</button></form>
    {error && <p role="alert">{error} The previous keys remain active.</p>}
    <div className="hash-controls"><label>Query key<select aria-label="Query key" value={query} onChange={event => setQuery(Number(event.target.value))}>{state.keys.map(key => <option key={key}>{key}</option>)}</select></label><label>Multiplier a<select aria-label="Multiplier a" value={multiplier} onChange={event => setMultiplier(Number(event.target.value))}>{Array.from({
            length: 16
          }, (_, index) => <option key={index + 1}>{index + 1}</option>)}</select></label><label>Offset b<select aria-label="Offset b" value={offset} onChange={event => setOffset(Number(event.target.value))}>{Array.from({
            length: 17
          }, (_, index) => <option key={index}>{index}</option>)}</select></label><button onClick={reset}>Reset</button></div>
    <p className="hash-equation">h(k) = (({multiplier} × k + {offset}) mod 17) mod 4</p>
    <div className="hash-buckets">{state.buckets.map((bucket, index) => <div key={index}><strong>Bucket {index}</strong><span>{bucket.length ? bucket.join(' → ') : 'empty'}</span></div>)}</div>
    <div className="hash-linked"><figure><svg viewBox="0 0 330 248" role="img" aria-label="All 272 hash choices, colored by query candidate-chain length">
      <title>Complete hash-family outcomes for the fixed query</title>
      <text x="42" y="18">b = 0 … 16 across columns</text>
      {state.family.map(row => <rect key={`${row.a}-${row.b}`} x={42 + row.b * 15} y={30 + (row.a - 1) * 12} width={13} height={10} fill={colors[row.length - 1]} stroke={row.a === multiplier && row.b === offset ? '#fff' : 'none'} strokeWidth="3" />)}
      <text x="8" y="38">a=1</text><text x="5" y="218">16</text><text x="42" y="242">White outline = selected function</text>
    </svg><figcaption>Each equal-area cell is one equally likely hash choice; its color records the query's full candidate-bucket length.</figcaption></figure>
      <div><h4>How many choices yield each length?</h4>{state.distribution.map(row => <div className="hash-frequency" key={row.length}><span>Length {row.length}</span><div><span style={{
              width: `${100 * row.count / 272}%`,
              background: colors[row.length - 1]
            }} /></div><strong>{row.count} / 272</strong></div>)}<p>Bars share a 0–272 scale. A long bucket still occurs for some choices; an expected bound is not an individual worst-case bound.</p></div></div>
    <dl className="hash-readout" aria-live="polite"><dt>Selected candidate length</dt><dd data-result="family-selected">{state.selectedLength}</dd><dt>Exact mean over the family</dt><dd data-result="family-mean">{state.totalLength} / 272 ≈ {(state.totalLength / 272).toFixed(6)}</dd><dt>Bound 1 + (n−1)/4</dt><dd>{state.theoreticalBound}</dd></dl>
    <p>Pairwise collision counts with key {query}: {state.pairCounts.length ? state.pairCounts.map(pair => `${pair.key}: ${pair.count}/272`).join('; ') : 'no other stored key'}. These count candidates, not exact early-stop equality comparisons. Expected unsuccessful-search candidates use a separate fixed absent key.</p>
  </section>;
}
function CostPlot({
  states,
  position,
  maximum,
  label
}) {
  const selected = states[position];
  const width = 252 / Math.max(1, states.length - 1);
  return <svg viewBox="0 0 330 230" role="img" aria-label={label}>
    <title>{label}</title><text x="44" y="18">counted work per operation</text>
    <path d="M42,34 V184 H302" className="hash-axis" />
    <text x="34" y="188" textAnchor="end">0</text><text x="34" y="39" textAnchor="end">{maximum}</text>
    {states.slice(1).map((state, index) => <rect key={index} x={46 + index * width} y={184 - state.cost / maximum * 148} width={Math.max(1, width - 2)} height={state.cost / maximum * 148} className={index + 1 === position ? 'hash-cost-current' : index + 1 < position ? 'hash-cost-done' : 'hash-cost-future'} />)}
    <text x="46" y="205">1</text><text x="297" y="205" textAnchor="end">{states.length - 1}</text><text x="166" y="224" textAnchor="middle">operation number</text>
    {position > 0 && <text x="166" y="31" textAnchor="middle">selected cost: {selected.cost}</text>}
  </svg>;
}
export function ResizeAccountingLab() {
  const [mode, setMode] = useState('growth');
  const [scenario, setScenario] = useState('boundary');
  const [position, setPosition] = useState(0);
  const operations = mode === 'growth' ? resizeScenarios.append : resizeScenarios[scenario];
  const left = resizeTrace(operations, mode === 'growth' ? 'never' : 'half', 'double');
  const right = resizeTrace(operations, mode === 'growth' ? 'never' : 'quarter', mode === 'growth' ? 'one' : 'double');
  const labels = mode === 'growth' ? ['Double capacity', 'Add one slot'] : ['Shrink at half occupancy', 'Shrink at quarter occupancy'];
  const maximum = Math.max(1, ...left.map(row => row.cost), ...right.map(row => row.cost));
  return <section className="hash-lab" data-lab="resize-accounting" aria-label="Resize policy and counted cost investigation">
    <h3>Place the expensive steps on the same operation timeline</h3>
    <p>Start empty with capacity 1. Count one unit for a successful append/pop plus one for each retained item copied. Allocation/initialization and the initial reserved slot are excluded. These are exact model counts, not elapsed time.</p>
    <div className="hash-controls"><label>Comparison<select aria-label="Comparison" value={mode} onChange={event => {
          setMode(event.target.value);
          setPosition(0);
        }}><option value="growth">Growth: double vs +1</option><option value="shrink">Shrink: half vs quarter</option></select></label>{mode === 'shrink' && <label>Operation sequence<select aria-label="Operation sequence" value={scenario} onChange={event => {
          setScenario(event.target.value);
          setPosition(0);
        }}><option value="boundary">8 fills + 8 append/pop pairs</option><option value="drain">16 appends, then 16 pops</option></select></label>}</div>
    <p className="hash-operation">{position ? `Operation ${position}: ${operations[position - 1] === '+' ? 'append one item' : 'pop the last item'}` : 'Before the first operation'}</p>
    <div className="hash-linked">{[left, right].map((states, index) => {
        const current = states[position];
        return <div className="hash-policy" key={index}><h4>{labels[index]}</h4><CostPlot states={states} position={position} maximum={maximum} label={`${labels[index]} counted operation costs`} />
        <dl className="hash-readout"><dt>Length / capacity</dt><dd>{current.length} / {current.capacity}</dd><dt>Copies / operation cost</dt><dd>{current.copies} / {current.cost}</dd><dt>Cumulative work</dt><dd data-result={`resize-total-${index}`}>{current.total}</dd></dl>
        <p>{current.event}</p>
        {mode === 'growth' && <p>Charging 3 per append leaves {current.credits} units. {current.credits < 0 ? 'This proposed constant charge cannot pay this sequence.' : 'Stored analysis credit can pay later copies; it is not real memory.'}</p>}
      </div>;
      })}</div>
    <TraceControls position={position} count={left.length} setPosition={setPosition} reset={() => {
      setMode('growth');
      setScenario('boundary');
      setPosition(0);
    }} />
    <p className="hash-caption">Both plots have the same vertical scale and preview the full known sequence; the selected bar is highlighted. Shrinking halves capacity when n≤C/2 or n≤C/4 respectively, never below 1. Growth happens only before an append to a full allocation. Hash-table collision scans are a separate cost.</p>
  </section>;
}
export function DenseSetLab() {
  const values = [10, 30, 20, 40];
  const [removed, setRemoved] = useState(30);
  const [position, setPosition] = useState(0);
  const states = denseSetTrace(values, removed);
  const state = states[position];
  return <section className="hash-lab" data-lab="dense-set" aria-label="Dense set and reverse index investigation">
    <h3>Delete one value without shifting the suffix</h3>
    <p>Inspect which reverse-index entry must change. The two representations must agree at operation boundaries. During the displayed write sequence a temporary duplicate can exist; this is not a concurrent atomic operation.</p>
    <div className="hash-controls"><label>Value to remove<select aria-label="Value to remove" value={removed} onChange={event => {
          setRemoved(Number(event.target.value));
          setPosition(0);
        }}>{[10, 30, 20, 40, 99].map(value => <option key={value}>{value}</option>)}</select></label></div>
    <div className="hash-dense-array">{state.items.map((value, index) => <div key={index}><small>index {index}</small><strong>{value}</strong></div>)}</div>
    <p>Reverse index: {state.positions.map(([value, index]) => `${value} → ${index}`).join(' · ') || 'empty'}</p>
    <p data-result="dense-action">{state.action}</p>
    <TraceControls position={position} count={states.length} setPosition={setPosition} reset={() => {
      setRemoved(30);
      setPosition(0);
    }} />
    <p>After the removal completes, drawing a uniformly chosen array index gives each remaining value probability 1/{states.at(-1).items.length}. This depends on unique values and a uniform index source, not on map iteration order.</p>
  </section>;
}
export function CollisionLayoutFigure() {
  const keys = [1, 9, 17];
  return <figure className="hash-inline" aria-label="Same colliding keys in chaining and linear probing">
    <p>Keys 1, 9 and 17 all have home 1 modulo 8.</p>
    <div className="hash-buckets"><div><strong>Chaining · bucket 1</strong><span>{keys.join(' → ')}</span></div></div>
    <SlotStrip slots={Array.from({
      length: 8
    }, (_, index) => index >= 1 && index <= 3 ? {
      key: keys[index - 1],
      value: keys[index - 1] * 10
    } : null)} label="Linear probing stores the same three keys in neighboring slots" />
    <figcaption>Chaining keeps a candidate collection at the home bucket. Linear probing stores entries directly in the slot array and searches successive slots. Both retain the original keys for equality checks; neither replaces key 1 with key 9 merely because their homes match.</figcaption>
  </figure>;
}
export function RehashLocationsFigure() {
  return <figure className="hash-inline" aria-label="Rebuilding changes each key's home instead of copying old slot positions">
    <div className="hash-rehome"><span>key</span><strong>modulo 8</strong><strong>modulo 16</strong>{[1, 9, 17, 25].map(key => <div className="hash-rehome-row" key={key}><strong>{key}</strong><span>home {key % 8}</span><span>→ home {key % 16}</span></div>)}</div>
    <figcaption>Physical destinations still depend on collisions and reinsertion order. The required result is the same key/value mapping under the new lookup rule. Copying slot positions without rebuilding the search structure can leave a live key behind an EMPTY stopping point.</figcaption>
  </figure>;
}
export function HashCostAxesFigure() {
  return <figure className="hash-inline" aria-label="Expected cost and amortized cost describe different axes">
    <div className="hash-cost-axes"><div><strong>Across hash choices</strong><p>Fix keys and a query. Compare its cost under randomly chosen h.</p><span>Expectation answers: average over which randomness?</span></div><div><strong>Across an operation sequence</strong><p>Fix a permitted sequence. Add cheap updates and occasional rebuilds.</p><span>Amortization answers: how much total work over the sequence?</span></div></div>
    <figcaption>A data structure can require both analyses. A randomized map with geometric resizing often offers expected amortized bounds under its stated distribution and sequence assumptions. Neither axis turns a single resize or a worst-case collision chain into constant work.</figcaption>
  </figure>;
}
