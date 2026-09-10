import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { parseBoundedIntegers, queenConflicts, subsetTree, summarizeSubarray, traceQueens, traceSubsetChoices } from '../../data/backtracking-divide-models.js';
import './backtracking-divide-labs.css';
function TraceControls({
  frames,
  step,
  setStep,
  solutions = false
}) {
  const nextSolution = frames.findIndex((frame, index) => index > step && frame.phase === 'solution');
  return <div className="bd-controls bd-trace-controls"><button type="button" disabled={step === 0} onClick={() => setStep(step - 1)}>Previous event</button><span>Event {step + 1} / {frames.length}</span><button type="button" disabled={step === frames.length - 1} onClick={() => setStep(step + 1)}>Next event</button>{solutions && <button type="button" disabled={nextSolution < 0} onClick={() => setStep(nextSolution)}>Next solution</button>}<button type="button" disabled={step === frames.length - 1} onClick={() => setStep(frames.length - 1)}>Finish exploration</button></div>;
}
function DecisionTree({
  trace,
  frame
}) {
  const scroll = useRef(null);
  const layout = useMemo(() => subsetTree(trace.values), [trace.values]);
  const active = layout.nodes.find(node => node.id === frame.prefix);
  useEffect(() => {
    if (scroll.current && active) scroll.current.scrollLeft = Math.max(0, active.x - scroll.current.clientWidth / 2);
  }, [active]);
  return <div className="bd-diagram-scroll" tabIndex={0} role="region" aria-label="Binary choice tree; scroll horizontally to inspect other branches" ref={scroll}>
    <svg viewBox={`0 0 ${layout.width} ${layout.height}`} style={{
      width: layout.width
    }} role="img" aria-label="A tree of subset decisions. Node numbers are running sums. Left branches take the next position; right branches skip it.">
      {layout.nodes.filter(node => node.parent).map(node => {
        const parent = layout.nodes.find(item => item.id === node.parent);
        return <line key={`edge-${node.id}`} x1={parent.x} y1={parent.y + 18} x2={node.x} y2={node.y - 18} className="bd-choice-edge" />;
      })}
      {layout.nodes.map(node => (
        <g key={node.id} className={`bd-choice-node bd-choice-node--${frame.visited[node.id] || 'unseen'}${node.id === frame.prefix ? ' bd-choice-node--current' : ''}`}>
          <circle cx={node.x} cy={node.y} r="18" />
          <text x={node.x} y={node.y + 5}>{node.sum}</text>
          {node.parent && (
            <text x={node.x} y={node.y + 34} className="bd-choice-label">
              {node.id.endsWith('1') ? 'take' : 'skip'} {trace.values[node.depth - 1]}
            </text>
          )}
        </g>
      ))}
    </svg>
  </div>;
}
export function SubsetChoicesLab() {
  const heading = useId();
  const [values, setValues] = useState('2, 4, 5');
  const [target, setTarget] = useState('5');
  const [prune, setPrune] = useState(true);
  const [trace, setTrace] = useState(() => traceSubsetChoices());
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const frame = trace.frames[step];
  function apply(event) {
    event.preventDefault();
    try {
      if (!/^\d+$/.test(target)) throw new Error('Target must be a whole number from 0 through 30.');
      const input = parseBoundedIntegers(values, {
        maximumLength: 3,
        minimum: 1,
        maximum: 9,
        allowEmpty: true
      });
      setTrace(traceSubsetChoices(input, Number(target), prune));
      setStep(0);
      setError('');
    } catch (issue) {
      setError(issue.message);
    }
  }
  return <section className="bd-lab" data-bd-lab="subsets" aria-labelledby={heading}>
    <p className="bd-eyebrow">INVESTIGATION · CHOOSE, EXPLORE, UNDO</p><h3 id={heading}>A temporary path is not a saved answer</h3>
    <p>Find subsets of positions whose values reach the target. Repeated values are separate occurrences. Predict the next choice, then watch the shared path change while saved answers remain intact.</p>
    <form className="bd-controls" onSubmit={apply}><label>Values · at most three<input value={values} onChange={event => setValues(event.target.value)} maxLength={30} /></label><label>Target sum<input value={target} onChange={event => setTarget(event.target.value)} inputMode="numeric" maxLength={3} /></label><label className="bd-checkbox"><input type="checkbox" checked={prune} onChange={event => setPrune(event.target.checked)} />Use positive-value pruning</label><button type="submit">Apply search</button><button type="button" onClick={() => {
        setValues('2, 4, 5');
        setTarget('5');
        setPrune(true);
        setTrace(traceSubsetChoices());
        setStep(0);
        setError('');
      }}>Reset example</button></form>
    {error && <p className="bd-error" role="alert">{error}</p>}
    <p className="bd-note">Applied values [{trace.values.join(', ')}], target {trace.target}; pruning {trace.prune ? 'on' : 'off'}. Input edits take effect with Apply. An empty list is allowed. Left = take; right = skip. Node number = running sum.</p>
    <p className="bd-status" aria-live="polite">{frame.message}</p>
    <div className="bd-path"><strong>Shared path now</strong>{frame.path.length ? frame.path.map(index => <span key={index}><small>position {index}</small><b>{trace.values[index]}</b></span>) : <span className="bd-empty">empty</span>}</div>
    <DecisionTree trace={trace} frame={frame} />
    <p className="bd-note">Amber outline: current event. Green: accepted leaf. Red dashed: pruned branch. Faint nodes were not entered. The diagram centers the current node; scroll to inspect other branches.</p>
    <TraceControls frames={trace.frames} step={step} setStep={setStep} solutions />
    <div className="bd-answer-strip"><strong>{frame.answers.length} saved answers · {frame.entered} calls entered so far</strong><div>{frame.answers.length ? frame.answers.map((answer, index) => <span key={index}>positions [{answer.join(', ')}] → values [{answer.map(item => trace.values[item]).join(', ')}]</span>) : 'No accepted leaf yet.'}</div></div>
    <details><summary>Inspect actual pending call frames</summary><div className="bd-table-scroll" tabIndex={0} role="region" aria-label="Backtracking call frames"><table><thead><tr><th>Frame</th><th>Next index</th><th>Sum on entry</th><th>Decision prefix</th></tr></thead><tbody>{frame.calls.map((call, index) => <tr key={call.prefix}><th>{index === frame.calls.length - 1 ? 'active' : 'waiting'}</th><td>{call.index}</td><td>{call.sum}</td><td>{call.prefix.slice(1) || 'none'}</td></tr>)}</tbody></table></div><p>Prefix 1 means take and 0 means skip. A parent's arguments stay fixed while a child explores; the separate shared path is temporarily extended between choose and undo.</p></details>
  </section>;
}
export function QueensSearchLab() {
  const heading = useId();
  const [size, setSize] = useState('4');
  const [trace, setTrace] = useState(() => traceQueens());
  const [step, setStep] = useState(0);
  const frame = trace.frames[step];
  const candidateVisible = ['try', 'reject'].includes(frame.phase);
  return <section className="bd-lab" data-bd-lab="queens" aria-labelledby={heading}>
    <p className="bd-eyebrow">INVESTIGATION · CONSTRAINTS LIVE ON THE BOARD</p><h3 id={heading}>Reject a threatened square before searching below it</h3>
    <p>One row, one queen. Place only when its column and both diagonals are free. Follow the first dead end, then jump to the next complete solution to see what the undo steps made possible.</p>
    <div className="bd-controls"><label>Board size<select aria-label="Board size" value={size} onChange={event => setSize(event.target.value)}><option value="4">4 × 4</option><option value="5">5 × 5</option></select></label><button type="button" onClick={() => {
        setTrace(traceQueens(Number(size)));
        setStep(0);
      }}>Apply board</button><button type="button" onClick={() => setStep(0)}>Restart trace</button></div>
    <p className="bd-note">Applied board: {trace.n} × {trace.n}. Rows and columns start at zero. This is exhaustive depth-first search in ascending column order; no symmetry elimination is used.</p>
    <p className="bd-status" aria-live="polite">{frame.message}</p>
    <div className="bd-queen-layout"><div className="bd-chessboard" style={{
        gridTemplateColumns: `repeat(${trace.n}, 1fr)`
      }} role="img" aria-label={`Queen board: ${frame.queens.map((column, row) => `(${row},${column})`).join(', ') || 'empty'}. ${candidateVisible ? `Candidate (${frame.row},${frame.column}), ${frame.conflicts.length ? 'conflict' : 'available'}.` : ''}`}>
      {Array.from({
          length: trace.n * trace.n
        }, (_, index) => {
          const row = Math.floor(index / trace.n);
          const column = index % trace.n;
          const queen = frame.queens[row] === column;
          const candidate = candidateVisible && row === frame.row && column === frame.column;
          const attacking = candidateVisible && frame.conflicts.some(conflict => conflict.row === row && conflict.column === column);
          return <div key={index} className={`bd-square${(row + column) % 2 ? ' bd-square--dark' : ''}${queen ? ' bd-square--queen' : ''}${candidate ? ' bd-square--candidate' : ''}${attacking ? ' bd-square--attacker' : ''}`}><small>{row},{column}</small><b>{queen ? 'Q' : candidate ? frame.conflicts.length ? '×' : '?' : '·'}</b></div>;
        })}
      <svg viewBox={`0 0 ${trace.n} ${trace.n}`} aria-hidden="true">{candidateVisible && frame.conflicts.map(conflict => <line key={`${conflict.row},${conflict.column}`} x1={conflict.column + .5} y1={conflict.row + .5} x2={frame.column + .5} y2={frame.row + .5} />)}</svg>
    </div><div className="bd-queen-ledger"><strong>{frame.answers.length} solutions copied</strong><p>{frame.candidates} candidate tests so far</p><p>Columns occupied: {frame.queens.join(', ') || 'none'}</p><p>Down-diagonal keys r−c: {frame.queens.map((column, row) => row - column).join(', ') || 'none'}</p><p>Up-diagonal keys r+c: {frame.queens.map((column, row) => row + column).join(', ') || 'none'}</p></div></div>
    <TraceControls frames={trace.frames} step={step} setStep={setStep} solutions />
    <details><summary>Read saved solutions and current coordinates</summary><p>Current queens: {frame.queens.map((column, row) => `(${row},${column})`).join(', ') || 'none'}. {candidateVisible ? frame.message : ''}</p><ol>{frame.answers.map((answer, index) => <li key={index}>{answer.map((column, row) => `(${row},${column})`).join(', ')}</li>)}</ol></details>
    <p className="bd-note">The browser derives attack lines by checking prior coordinates; the Python solver uses sets for constant expected-time constraint tests. They enumerate the same placements. Snapshot storage and board rendering are extra diagnostic work.</p>
  </section>;
}
function SummaryTable({
  title,
  node
}) {
  return <table className="bd-summary-table"><caption>{title} [{node.low}, {node.high})</caption><tbody><tr><th>Total</th><td>{node.total}</td></tr>{['prefix', 'suffix', 'best'].map(key => <tr key={key}><th>{key}</th><td>{node[key].sum} <small>[{node[key].low}, {node[key].high})</small></td></tr>)}</tbody></table>;
}
export function SubarraySummaryLab() {
  const heading = useId();
  const [draft, setDraft] = useState('-2, 4, -1, 3, -5, 2');
  const [result, setResult] = useState(() => summarizeSubarray());
  const [selected, setSelected] = useState('0:6');
  const [highlight, setHighlight] = useState('best');
  const [error, setError] = useState('');
  const node = result.nodes.find(item => item.id === selected) || result.root;
  const left = result.nodes.find(item => item.id === node.left);
  const right = result.nodes.find(item => item.id === node.right);
  const region = highlight === 'crossing' ? node.crossing : node[highlight];
  const depth = item => {
    let count = 0;
    let parent = result.nodes.find(candidate => candidate.left === item.id || candidate.right === item.id);
    while (parent) {
      count++;
      parent = result.nodes.find(candidate => candidate.left === parent.id || candidate.right === parent.id);
    }
    return count;
  };
  const levels = Math.max(...result.nodes.map(depth)) + 1;
  function apply(event) {
    event.preventDefault();
    try {
      const next = summarizeSubarray(parseBoundedIntegers(draft));
      setResult(next);
      setSelected(next.root.id);
      setError('');
    } catch (issue) {
      setError(issue.message);
    }
  }
  return <section className="bd-lab" data-bd-lab="summary" aria-labelledby={heading}>
    <p className="bd-eyebrow">INVESTIGATION · RETURN ENOUGH TO COMBINE</p><h3 id={heading}>A crossing answer needs a suffix and a prefix</h3>
    <p>Select a real subproblem in the split tree. The four summaries describe its contiguous input region; choosing a parent combines the two children without reading every element again.</p>
    <form className="bd-controls" onSubmit={apply}><label>Signed array · one to eight integers<input value={draft} onChange={event => setDraft(event.target.value)} maxLength={60} /></label><button type="submit">Apply array</button><button type="button" onClick={() => {
        const next = summarizeSubarray();
        setResult(next);
        setSelected(next.root.id);
        setDraft('-2, 4, -1, 3, -5, 2');
        setHighlight('best');
        setError('');
      }}>Reset array</button></form>
    {error && <p className="bd-error" role="alert">{error}</p>}
    <p className="bd-note">Applied [{result.values.join(', ')}]. Ranges [lo, hi) include lo and stop before hi. Equal sums choose the earliest start, then earliest end; the core problem asks only for the sum.</p>
    <div className="bd-diagram-scroll" tabIndex={0} role="region" aria-label="Divide tree and indexed array; scroll horizontally if needed"><div style={{
        width: Math.max(280, result.values.length * 64)
      }}>
      {Array.from({
          length: levels
        }, (_, level) => <div className="bd-split-level" key={level} style={{
          gridTemplateColumns: `repeat(${result.values.length}, 1fr)`
        }}>{result.nodes.filter(item => depth(item) === level).map(item => <button type="button" key={item.id} style={{
            gridColumn: `${item.low + 1} / ${item.high + 1}`
          }} onClick={() => setSelected(item.id)} aria-label={`Inspect range ${item.low} through ${item.high} exclusive`} aria-pressed={item.id === node.id}>{item.high - item.low === 1 ? result.values[item.low] : `[${item.low}, ${item.high})`}</button>)}</div>)}
      <div className="bd-array" style={{
          gridTemplateColumns: `repeat(${result.values.length}, 1fr)`
        }}>{result.values.map((value, index) => <div key={index} className={`${index >= node.low && index < node.high ? 'bd-array-cell' : 'bd-array-cell bd-array-cell--outside'}${region && index >= region.low && index < region.high ? ' bd-array-cell--selected' : ''}`}><small>i={index}</small><b>{value}</b></div>)}</div>
    </div></div>
    <div className="bd-controls"><label>Highlight interval<select aria-label="Highlight interval" value={highlight} onChange={event => setHighlight(event.target.value)}><option value="best">Best anywhere</option><option value="prefix">Best prefix</option><option value="suffix">Best suffix</option><option value="crossing">Crossing candidate</option></select></label></div>
    <p className="bd-status" aria-live="polite">Range [{node.low}, {node.high}): {region ? `${highlight} sum ${region.sum}, interval [${region.low}, ${region.high}).` : 'A singleton has no split or crossing candidate.'}</p>
    {left ? <><div className="bd-summary-children"><SummaryTable title="Left" node={left} /><SummaryTable title="Right" node={right} /></div><p className="bd-cross-formula">Crossing = left suffix <b>{left.suffix.sum}</b> + right prefix <b>{right.prefix.sum}</b> = <b>{node.crossing.sum}</b></p><p>Parent best = max({left.best.sum}, {right.best.sum}, {node.crossing.sum}) = <strong>{node.best.sum}</strong></p></> : <p>A leaf's total, prefix, suffix and best are all its one value, including when negative.</p>}
    <SummaryTable title="Selected result" node={node} />
    <p className="bd-note">This input creates {result.nodes.length} calls and {result.combines} combines. The computation returns constant-size summaries; retaining the whole divide tree here is extra visualizer storage.</p>
  </section>;
}
export function InversionBoundaryFigure() {
  const left = [2, 6, 8];
  const right = [1, 5, 7];
  return <figure className="bd-inline"><h3>A smaller right value settles several pairs at once</h3><div className="bd-inversion-halves"><div><strong>Remaining left, sorted</strong><div>{left.map(value => <span key={value}>{value}</span>)}</div></div><div><strong>Remaining right, sorted</strong><div>{right.map((value, index) => <span key={value} className={index === 0 ? 'bd-inversion-picked' : ''}>{value}</span>)}</div></div></div><p className="bd-inversion-pairs">(2, 1)　(6, 1)　(8, 1)</p><figcaption>The right head 1 is smaller than the left head 2, so it is smaller than all three remaining left values. Every left element originally preceded every right element. Emit 1 and add three cross inversions; move only the right pointer. Equal values do not form a strict inversion.</figcaption></figure>;
}
