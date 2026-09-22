import { useEffect, useMemo, useRef, useState } from 'react';
import { rewardTrace, defaultGrid, gridPlan, lcsPlan, capacityTrace, routeNames, routeCosts, subsetRoutePlan } from '../../data/dynamic-programming-models.js';
import './dynamic-programming-labs.css';
function Steps({
  step,
  length,
  setStep,
  noun = 'state'
}) {
  return <div className="lesson-controls dp-controls">
    <button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous {noun}</button>
    <button disabled={step === length - 1} onClick={() => setStep(step + 1)}>Next {noun}</button>
    <button onClick={() => setStep(length - 1)}>Finish {noun}s</button>
    <span>{step + 1} / {length}</span>
  </div>;
}
export function StateHistoryFigure() {
  return <figure className="dp-inline" aria-label="Two histories leave different legal choices at the same session index">
    <div className="dp-history-row"><span>Earlier choice</span><span>Now: session 3</span><span>Best future</span></div>
    <div className="dp-history-row"><span className="dp-chosen">Session 2 taken</span><span className="dp-forbidden">9 · blocked</span><strong>0</strong></div>
    <div className="dp-history-row"><span>Session 2 skipped</span><span className="dp-chosen">9 · available</span><strong>9</strong></div>
    <figcaption>Same remaining index, different permission. A key of index alone merges answers 0 and 9 unless the function's entry contract guarantees that its first session is free.</figcaption>
  </figure>;
}
export function CoinOrderFigure() {
  return <figure className="dp-inline" aria-label="Amount four with denominations one and three has three ordered sequences and two unordered combinations">
    <div className="dp-coins"><span><b>Ordered · 3</b><i>1 → 1 → 1 → 1</i><i>1 → 3</i><i>3 → 1</i></span><span><b>Unordered · 2</b><i>1 + 1 + 1 + 1</i><i>1 + 3</i><small>Both orders become one multiset.</small></span></div>
    <figcaption>Every row totals 4. Choosing a last coin separates sequences; introducing denominations one at a time separates combinations.</figcaption>
  </figure>;
}
export function TailDominanceFigure() {
  return <figure className="dp-inline" aria-label="Minimum tails are a summary, not necessarily an input subsequence">
    <div className="dp-tail-row"><span>Input order</span>{[3, 5, 6, 2].map((value, index) => <b key={index} className={index === 3 ? 'dp-selected' : ''}>{value}<small>index {index}</small></b>)}</div>
    <div className="dp-tail-row"><span>Before last 2</span>{[3, 5, 6].map((value, index) => <b key={index}>{value}<small>length {index + 1}</small></b>)}</div>
    <div className="dp-tail-row"><span>After last 2</span>{[2, 5, 6].map((value, index) => <b key={index} className={index === 0 ? 'dp-chosen' : ''}>{value}<small>length {index + 1}</small></b>)}</div>
    <figcaption>The best tail for length 1 improves from 3 to 2. The length-3 witness 3→5→6 still exists, but combining tails 2→5→6 violates input order. Each summary cell belongs to its own best-length candidate.</figcaption>
  </figure>;
}
export function WeightedIntervalFigure() {
  return <figure className="dp-inline" aria-label="One long interval worth ten versus two touching intervals worth eight">
    <div className="dp-interval-scale"><span>time 0</span><span>2</span><span>5</span></div>
    <div className="dp-interval-lane"><b style={{ gridColumn: '1 / 6' }} className="dp-chosen">A [0,5) · value 10</b></div>
    <div className="dp-interval-lane"><b style={{ gridColumn: '1 / 3' }}>B · value 4</b><b style={{ gridColumn: '3 / 6' }}>C · value 4</b></div>
    <figcaption>Horizontal extent encodes the stated interval times. B and C can touch at 2 without overlap. More jobs gives value 8; choosing A gives 10. A finish-ordered DP retains both possibilities until their full values are compared.</figcaption>
  </figure>;
}
export function RewardDependencyLab() {
  const [draft, setDraft] = useState('4, 7, 2, 9');
  const [rewards, setRewards] = useState([4, 7, 2, 9]);
  const [method, setMethod] = useState('memo');
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const model = useMemo(() => rewardTrace(rewards, method), [rewards, method]);
  const state = model.frames[step];
  const graphRegion = useRef(null);
  const spacing = 52;
  const width = Math.max(310, (rewards.length + 2) * spacing);
  useEffect(() => {
    const region = graphRegion.current;
    if (region) region.scrollLeft = Math.max(0, (state.index ?? 0) * spacing - 30);
  }, [state.index, rewards]);
  function apply(event) {
    event.preventDefault();
    try {
      const values = draft.trim() ? draft.split(',').map(part => {
        if (!/^\s*-?\d+\s*$/.test(part)) throw new Error('Use comma-separated integers.');
        return Number(part);
      }) : [];
      rewardTrace(values, method);
      setRewards(values);
      setStep(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setDraft('4, 7, 2, 9');
    setRewards([4, 7, 2, 9]);
    setMethod('memo');
    setStep(0);
    setError('');
  }
  return <section className="dp-lab" aria-label="Reward dependency investigation">
    <h3>Follow a request; reuse an answer</h3>
    <p>Inspect which suffix will be requested twice. Arrows point from a question to its dependencies; table evaluation must run in the opposite direction.</p>
    <form className="lesson-controls" onSubmit={apply}><label>Session rewards<input value={draft} onChange={event => setDraft(event.target.value)} /></label><button>Apply rewards</button></form>
    <div className="lesson-controls"><label>Evaluation order<select value={method} onChange={event => {
          setMethod(event.target.value);
          setStep(0);
        }}><option value="memo">Requests + memoization</option><option value="table">Reverse tabulation</option></select></label><button onClick={reset}>Reset reward lab</button></div>
    {error && <p role="alert">{error}</p>}
    <p className="dp-note">Active rewards: [{rewards.join(', ')}]. Amber outline: current state. Green fill: stored answer. Blank dash: not computed.</p>
    <div ref={graphRegion} className="dp-scroll" tabIndex={0} role="region" aria-label="Suffix dependency graph">
      <svg width={width} height="230" role="img" aria-label={`Suffix graph: F(i) depends on F(i+1) and F(i+2). ${state.message}`}>
        {rewards.map((reward, index) => [1, 2].map(distance => {
          const start = 26 + index * spacing;
          const end = start + distance * spacing;
          const level = distance === 1 ? 42 : 185;
          return <g key={`${index}-${distance}`} className={state.index === index ? 'dp-edge-active' : 'dp-edge'}>
            <path d={`M${start},${distance === 1 ? 86 : 136} Q${(start + end) / 2},${level} ${end},${distance === 1 ? 86 : 136}`} fill="none" />
            <path d={`M${end - 6},${distance === 1 ? 77 : 144} L${end},${distance === 1 ? 86 : 136} L${end + 3},${distance === 1 ? 77 : 145}`} fill="none" />
          </g>;
        }))}
        {state.cache.map((value, index) => <g key={index} transform={`translate(${5 + index * spacing},86)`}>
          <rect width="42" height="50" rx="4" className={`${value !== null ? 'dp-node-known' : 'dp-node'} ${state.index === index ? 'dp-node-active' : ''}`} />
          <text x="21" y="18" textAnchor="middle">F({index})</text><text x="21" y="39" textAnchor="middle">{value ?? '—'}</text>
          <text x="21" y="118" textAnchor="middle">{index < rewards.length ? `r=${rewards[index]}` : 'empty'}</text>
        </g>)}
      </svg>
    </div>
    <p className="dp-note">r is the session reward. Short upper arc: skip → i+1. Lower arc: take reward[i] → i+2. Shared destination nodes are the same subproblem, even when several calls request them. Longer graphs scroll locally; focus the graph and use arrow keys to inspect them.</p>
    {method === 'memo' && <div className="dp-stack"><span>Active calls →</span>{state.stack.map((index, position) => <span key={position}>F({index})</span>)}{!state.stack.length && <span>none</span>}</div>}
    <p role="status" data-dp-status>{state.message}</p>
    <Steps step={step} length={model.frames.length} setStep={setStep} />
    <p className="dp-note">Try all negative rewards. Explain why a legitimate zero must still be cached. This is an exact small trace, not a recursion-performance benchmark.</p>
  </section>;
}
export function GridDependencyLab() {
  const [blocked, setBlocked] = useState([]);
  const [step, setStep] = useState(0);
  const [showPath, setShowPath] = useState(false);
  const model = useMemo(() => gridPlan(defaultGrid, blocked), [blocked]);
  const current = model.steps[step];
  const completed = step === model.steps.length - 1;
  const pathCells = new Set(showPath && completed ? model.path.map(cell => cell.join(',')) : []);
  function toggle(row, column) {
    const key = `${row},${column}`;
    setBlocked(blocked.includes(key) ? blocked.filter(cell => cell !== key) : [...blocked, key]);
    setStep(0);
    setShowPath(false);
  }
  return <section className="dp-lab" aria-label="Grid dependency investigation">
    <h3>Grow an answer across the actual grid</h3>
    <p>Move only right or down; pay each visited cell, including the start. Inspect the cheapest route. Activate a cell to block or reopen it; that immediately restarts evaluation.</p>
    <div className="dp-grid" aria-label="Grid costs and minimum prefix totals">
      {defaultGrid.flatMap((row, rowIndex) => row.map((cost, columnIndex) => {
        const key = `${rowIndex},${columnIndex}`;
        const evaluated = rowIndex * row.length + columnIndex <= step;
        const forbidden = blocked.includes(key);
        const active = rowIndex === current.row && columnIndex === current.column;
        const predecessor = current.candidates.some(cell => cell.row === rowIndex && cell.column === columnIndex);
        return <button key={key} onClick={() => toggle(rowIndex, columnIndex)} aria-pressed={forbidden} aria-label={`Cell ${key}, cost ${cost}, ${forbidden ? 'blocked' : 'open'}; toggle obstacle`} className={`${forbidden ? 'dp-forbidden' : ''} ${active ? 'dp-selected' : ''} ${predecessor ? 'dp-source' : ''} ${pathCells.has(key) ? 'dp-chosen' : ''}`}>
          <small>{key}</small><span>{forbidden ? '×' : `cost ${cost}`}</span><strong>{evaluated ? current.costs[rowIndex][columnIndex] ?? '∞' : '—'}</strong><small>{evaluated ? `ways ${current.ways[rowIndex][columnIndex]}` : 'waiting'}</small>
        </button>;
      }))}
    </div>
    <p className="dp-note">Large number: cheapest prefix total. ways: count of all valid paths, not only cheapest ones. × blocked; ∞ unreachable; — unevaluated. Amber: current cell. Blue border: available predecessors. Green: final witness.</p>
    <p role="status" data-dp-status>Cell ({current.row},{current.column}): {current.costs[current.row][current.column] === null ? 'unreachable.' : current.row === 0 && current.column === 0 ? 'start cost = 1; one empty move sequence.' : `min(${current.candidates.map(candidate => candidate.cost).join(', ')}) + ${defaultGrid[current.row][current.column]} = ${current.costs[current.row][current.column]}.`} {completed && `Destination: ${model.result ?? 'unreachable'}; ${model.ways[2][3]} paths.`}</p>
    <Steps step={step} length={model.steps.length} setStep={value => {
      setStep(value);
      setShowPath(false);
    }} noun="cell" />
    <div className="lesson-controls"><button disabled={!completed || model.result === null} onClick={() => setShowPath(!showPath)}>{showPath ? 'Hide witness' : 'Trace cheapest witness'}</button><button onClick={() => {
        setBlocked([]);
        setStep(0);
        setShowPath(false);
      }}>Reset grid lab</button></div>
    {showPath && <p>Parent trail: {model.path.map(cell => `(${cell.join(',')})`).join(' → ')}. Cost {model.result}. On equal predecessor costs, choose above.</p>}
    <p className="dp-note">Block both exits from the start. Why must the remaining cells stay unreachable even when their own cost is small? Then reopen one exit and follow the changed route.</p>
  </section>;
}
export function SequenceAlignmentLab() {
  const [draftFirst, setDraftFirst] = useState('CABAC');
  const [draftSecond, setDraftSecond] = useState('ABC');
  const [inputs, setInputs] = useState(['CABAC', 'ABC']);
  const [selected, setSelected] = useState([3, 2]);
  const [step, setStep] = useState(0);
  const [tracing, setTracing] = useState(false);
  const [error, setError] = useState('');
  const model = useMemo(() => lcsPlan(...inputs), [inputs]);
  const state = model.trace[step];
  const [row, column] = tracing ? [state.row, state.column] : selected;
  const matching = row > 0 && column > 0 && inputs[0][row - 1] === inputs[1][column - 1];
  const dependencies = row && column ? matching ? [[row - 1, column - 1]] : [[row - 1, column], [row, column - 1]] : [];
  function apply(event) {
    event.preventDefault();
    try {
      lcsPlan(draftFirst, draftSecond);
      setInputs([draftFirst, draftSecond]);
      setSelected([draftFirst.length, draftSecond.length]);
      setStep(0);
      setTracing(false);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setDraftFirst('CABAC');
    setDraftSecond('ABC');
    setInputs(['CABAC', 'ABC']);
    setSelected([3, 2]);
    setStep(0);
    setTracing(false);
    setError('');
  }
  return <section className="dp-lab" aria-label="Sequence alignment investigation">
    <h3>Match prefixes, then walk the answer backward</h3>
    <p>Inspect the common sequence before tracing. Select a length cell to see the prefixes and the dependencies that justify it. Empty prefixes have length zero.</p>
    <form onSubmit={apply} className="lesson-controls"><label>First sequence<input value={draftFirst} onChange={event => setDraftFirst(event.target.value)} /></label><label>Second sequence<input value={draftSecond} onChange={event => setDraftSecond(event.target.value)} /></label><button>Apply sequences</button></form>
    {error && <p role="alert">{error}</p>}
    <div className="dp-scroll" tabIndex={0} role="region" aria-label="LCS prefix table">
      <table className="dp-lcs-table"><caption>Prefix lengths · row i uses first[:i], column j uses second[:j]</caption><thead><tr><th>i ↓ / j →</th>{Array.from({
              length: inputs[1].length + 1
            }, (_, index) => <th key={index}>{index}<small>{index ? inputs[1][index - 1] : '∅'}</small></th>)}</tr></thead><tbody>
        {model.lengths.map((values, rowIndex) => <tr key={rowIndex}><th>{rowIndex}<small>{rowIndex ? inputs[0][rowIndex - 1] : '∅'}</small></th>{values.map((length, columnIndex) => <td key={columnIndex}><button aria-label={`LCS prefix ${rowIndex},${columnIndex} length ${length}`} className={`${row === rowIndex && column === columnIndex ? 'dp-selected' : ''} ${dependencies.some(pair => pair[0] === rowIndex && pair[1] === columnIndex) ? 'dp-source' : ''}`} onClick={() => {
                setSelected([rowIndex, columnIndex]);
                setTracing(false);
              }}>{length}</button></td>)}</tr>)}
      </tbody></table>
    </div>
    <p role="status" data-dp-status>L({row},{column}) for “{inputs[0].slice(0, row) || '∅'}” and “{inputs[1].slice(0, column) || '∅'}”: {row === 0 || column === 0 ? 'one prefix is empty, so zero.' : matching ? `equal final letters ${inputs[0][row - 1]}: 1 + diagonal ${model.lengths[row - 1][column - 1]} = ${model.lengths[row][column]}.` : `different final letters: max(above ${model.lengths[row - 1][column]}, left ${model.lengths[row][column - 1]}) = ${model.lengths[row][column]}.`}</p>
    <div className="lesson-controls"><button onClick={() => {
        setTracing(true);
        setStep(0);
      }}>Start witness trace</button><button onClick={reset}>Reset sequence lab</button></div>
    {tracing && <><Steps step={step} length={model.trace.length} setStep={setStep} noun="choice" /><div className="dp-alignment">{inputs.map((text, inputIndex) => <div key={inputIndex}><span>{inputIndex === 0 ? 'First' : 'Second'}</span>{[...text].map((character, index) => <b key={index} className={state.pairs.some(pair => pair[inputIndex] === index) ? 'dp-chosen' : ''}>{character}<small>{index}</small></b>)}{!text && <b>∅</b>}</div>)}</div><p>{state.action === 'done' ? `Finished: ${model.witness || 'empty sequence'}, length ${model.result}.` : state.action === 'match' ? `Next choice: keep ${inputs[0][state.row - 1]}, then move diagonally.` : `Next choice: move ${state.action}; no character is copied.`} Kept suffix so far: {state.pairs.map(pair => inputs[0][pair[0]]).join('') || '∅'}.</p></>}
    <p className="dp-note">Try AB versus BA. There are tied best answers; preferring up gives one deterministic witness, not necessarily a lexicographically smallest one. Inputs are case-sensitive ASCII letters, up to six each.</p>
  </section>;
}
export function CapacityGenerationLab() {
  const [capacity, setCapacity] = useState(6);
  const [direction, setDirection] = useState('descending');
  const [step, setStep] = useState(0);
  const model = useMemo(() => capacityTrace(undefined, capacity, direction), [capacity, direction]);
  const state = model.frames[step];
  const capacityRegion = useRef(null);
  useEffect(() => {
    const region = capacityRegion.current;
    const destination = region?.querySelector('.dp-selected');
    if (region && destination) {
      const left = destination.offsetLeft - destination.parentElement.offsetLeft;
      region.scrollLeft = Math.max(0, left + destination.offsetWidth - region.clientWidth + 5);
    } else if (region) region.scrollLeft = 0;
  }, [step, capacity, direction]);
  return <section className="dp-lab" aria-label="Capacity generation investigation">
    <h3>Which version of this cell are you reading?</h3>
    <p>Item 0 weighs 2 and is worth 3; item 1 weighs 3 and is worth 4. Inspect whether changing direction can use an item twice. The logical row is an item prefix even when both rows share one array.</p>
    <div className="lesson-controls"><label>Capacity<select aria-label="Capacity" value={capacity} onChange={event => {
          setCapacity(Number(event.target.value));
          setStep(0);
        }}>{Array.from({
            length: 9
          }, (_, value) => <option key={value}>{value}</option>)}</select></label><label>Capacity order<select aria-label="Capacity order" value={direction} onChange={event => {
          setDirection(event.target.value);
          setStep(0);
        }}><option value="descending">Descending · each item once</option><option value="ascending">Ascending · reusable items</option></select></label><button onClick={() => {
        setCapacity(6);
        setDirection('descending');
        setStep(0);
      }}>Reset capacity lab</button></div>
    <div ref={capacityRegion} className="dp-scroll" tabIndex={0} role="region" aria-label="Capacity array and update generations"><div className="dp-capacities">{state.best.map((value, budget) => <div key={budget} className={`${state.destination === budget ? 'dp-selected' : ''} ${state.source === budget ? 'dp-source' : ''}`}><small>cap {budget}</small><strong>{value}</strong><span>{state.generations[budget] < 0 ? 'initial' : `item ${state.generations[budget]} row`}</span><small>{state.witnesses[budget].length ? state.witnesses[budget].map(index => `#${index}`).join('+') : 'empty'}</small></div>)}</div></div>
    {state.item !== null && <div className="dp-read-arrow">Read cap {state.source} ({state.sourceValue}, {state.sourceGeneration === state.item ? 'current item row' : 'earlier item row'}) <span>→ add item #{state.item} →</span> write cap {state.destination} ({state.best[state.destination]})</div>}
    <p role="status" data-dp-status>{state.message}</p>
    <Steps step={step} length={model.frames.length} setStep={setStep} noun="update" />
    <p className="dp-note">Amber: written cell; blue: source cell. The written cell stays in view; focus the strip and use arrow keys to inspect other capacities. Each # is an item occurrence in the displayed witness. Repeated #0 is valid only under the reusable-item contract. Witness copies and generation labels are teaching diagnostics, not part of the O(capacity) value-only algorithm.</p>
  </section>;
}
export function SubsetEndpointLab() {
  const [mask, setMask] = useState(7);
  const [endpoint, setEndpoint] = useState(2);
  const model = useMemo(() => subsetRoutePlan(), []);
  const present = routeNames.flatMap((name, index) => mask & 1 << index ? [name] : []);
  const path = model.witness(mask, endpoint);
  const cost = model.best[mask][endpoint];
  function toggle(index) {
    setMask(mask ^ 1 << index);
  }
  return <section className="dp-lab" aria-label="Subset endpoint investigation">
    <h3>One subset can contain several different states</h3>
    <p>Start at A and visit selected vertices once. Toggle membership bits, then choose an endpoint. Inspect the cost of continuing from B versus C when the selected set is exactly A,B,C.</p>
    <div className="dp-bits">{[3, 2, 1, 0].map(index => <button key={index} aria-pressed={Boolean(mask & 1 << index)} aria-label={`Toggle ${routeNames[index]} membership bit ${index}`} onClick={() => toggle(index)}><small>{routeNames[index]} · bit {index}</small><strong>{mask & 1 << index ? 1 : 0}</strong><span>weight {1 << index}</span></button>)}</div>
    <p>Binary {mask.toString(2).padStart(4, '0')} = {mask}. Selected set {'{'}{present.join(', ')}{'}'}. A is bit 0 on the right. Full mask 1111 = 15.</p>
    <div className="lesson-controls"><label>Endpoint<select aria-label="Endpoint" value={endpoint} onChange={event => setEndpoint(Number(event.target.value))}>{routeNames.map((name, index) => <option value={index} key={name}>{name}</option>)}</select></label><button onClick={() => {
        setMask(7);
        setEndpoint(2);
      }}>Reset subset lab</button><button onClick={() => setMask(15)}>Use full subset</button></div>
    <div className="dp-route-map"><div className="dp-route-prefix"><small>Cheapest prefix for this state</small><strong>{path.length ? path.map(index => routeNames[index]).join(' → ') : 'No valid prefix'}</strong><span>{cost === null ? '∞' : `cost ${cost}`}</span></div><div className="dp-next-edges">{routeNames.map((name, next) => !(mask & 1 << next) && <div key={name}><span>{routeNames[endpoint]} → {name}</span><b>edge {routeCosts[endpoint][next]}</b><span>{cost === null ? 'unavailable' : `candidate ${cost + routeCosts[endpoint][next]}`}</span></div>)}{mask === 15 && <p>No unused vertices; this prefix is already a full path.</p>}</div></div>
    <p role="status" data-dp-status>{cost === null ? 'This state is unreachable: the prefix must contain A and its endpoint, and cannot revisit A.' : `D(${mask},${routeNames[endpoint]}) = ${cost}. The previous cost and next-edge cost are different quantities.`}</p>
    <div className="dp-scroll" tabIndex={0} role="region" aria-label="Exact route edge-cost matrix"><table className="dp-cost-table"><caption>From row → to column · fixed costs, arbitrary units</caption><thead><tr><th>From</th>{routeNames.map(name => <th key={name}>{name}</th>)}</tr></thead><tbody>{routeCosts.map((row, index) => <tr key={index}><th>{routeNames[index]}</th>{row.map((value, column) => <td key={column}>{index === column ? '—' : value}</td>)}</tr>)}</tbody></table></div>
    <p className="dp-note">The graph is complete and symmetric for this example; the algorithm also accepts directed or missing edges. D(7,B)=D(7,C)=2, but B→D costs 9 while C→D costs 1. Merging those states by mask alone loses a necessary fact. Full optimum: A→B→C→D, cost 3; no return to A is required.</p>
  </section>;
}
