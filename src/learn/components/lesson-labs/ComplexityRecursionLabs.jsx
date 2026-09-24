import { useState } from 'react';
import { CodeBlock } from '../content';
import { LessonTable } from './LessonElements';
import { loopPatterns, loopWork, parseSumValues, sumCallTrace, recurrencePatterns, recurrenceLevels } from '../../data/complexity-recursion-models.js';
import './complexity-recursion-labs.css';
export function IterationLatticeLab() {
  const [pattern, setPattern] = useState('triangle');
  const [size, setSize] = useState(6);
  const [selectedRow, setSelectedRow] = useState(0);
  const model = loopWork(pattern, size);
  const doubled = loopWork(pattern, 2 * size);
  function reset() {
    setPattern('triangle');
    setSize(6);
    setSelectedRow(0);
  }
  return <section className="lesson-lab complexity-lab" aria-label="Iteration lattice investigation">
    <h3>See which loop bodies actually execute</h3>
    <p>Inspect the total before changing n. A filled cell is one call to work(i, j); rows are i and columns are j. Select a row to inspect its exact inner indices.</p>
    <div className="lesson-controls">
      <label>Loop pattern<select value={pattern} onChange={event => {
          setPattern(event.target.value);
          setSelectedRow(0);
        }}>{Object.entries(loopPatterns).map(([key, value]) => <option key={key} value={key}>{value.label}</option>)}</select></label>
      <label>Input size n: {size}<input aria-label="Loop input size" type="range" min="0" max="16" value={size} onChange={event => {
          setSize(Number(event.target.value));
          setSelectedRow(0);
        }} /></label>
      <button onClick={reset}>Reset lattice</button>
    </div>
    <CodeBlock language="python">{loopPatterns[pattern].code}</CodeBlock>
    {size > 0 ? <div className="complexity-scroll" tabIndex={0} role="region" aria-label="Execution lattice; rows i and columns j">
      <div className="complexity-lattice" style={{
        '--columns': size
      }}>
        <div className="complexity-lattice-row"><span>i / j</span>{Array.from({
            length: size
          }, (_, index) => <span key={index}>{index}</span>)}</div>
        {model.rows.map(row => <div className={`complexity-lattice-row ${selectedRow === row.outer ? 'is-selected' : ''}`} key={row.outer}>
          <button aria-label={`Inspect outer index ${row.outer}`} aria-pressed={selectedRow === row.outer} onClick={() => setSelectedRow(row.outer)}>{row.outer}</button>
          {Array.from({
            length: size
          }, (_, column) => <span className={row.columns.includes(column) ? 'is-work' : ''} key={column} aria-label={`i ${row.outer}, j ${column}: ${row.columns.includes(column) ? 'executes' : 'not executed'}`}>{row.columns.includes(column) ? '•' : '–'}</span>)}
        </div>)}
      </div>
    </div> : <p>No outer iterations: there are no work cells.</p>}
    <p className="complexity-readout" data-lattice-total={model.total} aria-live="polite">{size ? `Row ${selectedRow}: j = ${model.rows[selectedRow].columns.join(', ') || 'none'}. ` : ''}<strong>{model.total} work calls</strong> at n={size}; {doubled.total} at n={2 * size}.</p>
    <p className="lesson-note">Exact body-call counts, not milliseconds. {loopPatterns[pattern].bound} describes large n; the surrounding explanation includes loop overhead and small/empty cases. Doubling n need not double the work. For n=0 both totals are zero.</p>
  </section>;
}
export function ReturnLadderFigure() {
  return <figure className="complexity-inline" aria-label="Suffix sum calls move to index 3; values return to index 0">
    <figcaption>One shared array [3, 1, 4]. Calls go down; completed values come back.</figcaption>
    <div className="complexity-return-ladder">
      {[['suffix(0)', '3 + suffix(1)', '3 + 5 = 8'], ['suffix(1)', '1 + suffix(2)', '1 + 4 = 5'], ['suffix(2)', '4 + suffix(3)', '4 + 0 = 4'], ['suffix(3)', 'empty → no call', 'return 0']].map(([call, pending, result], index) => <div key={call} style={{
        '--depth': index
      }}><strong>{call}</strong><span>{pending}</span><span>↑ {result}</span></div>)}
    </div>
    <p>The caller retains its own index and pending addition. Entering the last frame has not yet finished the earlier additions.</p>
  </figure>;
}
export function RecursiveFramesLab() {
  const [draft, setDraft] = useState('3, 1, 4');
  const [values, setValues] = useState([3, 1, 4]);
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const trace = sumCallTrace(values);
  const state = trace[step];
  function apply(event) {
    event.preventDefault();
    try {
      setValues(parseSumValues(draft));
      setStep(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setDraft('3, 1, 4');
    setValues([3, 1, 4]);
    setStep(0);
    setError('');
  }
  return <section className="lesson-lab complexity-lab" aria-label="Recursive frames investigation">
    <h3>Follow the waiting additions</h3>
    <p>Before advancing, decide whether another call is needed or a saved addition can finish. Each row below is one live call frame; the top row is the current call.</p>
    <form className="lesson-controls" onSubmit={apply}>
      <label>Values (empty is allowed)<input value={draft} aria-label="Suffix sum values" onChange={event => setDraft(event.target.value)} /></label>
      <button type="submit">Apply values</button><button type="button" onClick={reset}>Reset frames</button>
    </form>
    {error && <p role="alert">{error} The active trace is unchanged.</p>}
    <div className="complexity-array" aria-label="Active shared array">{values.length ? values.map((value, index) => <span key={index}><small>index {index}</small><strong>{value}</strong></span>) : <span>Empty array [ ]</span>}</div>
    <div className="lesson-controls"><button disabled={step === 0} onClick={() => setStep(step - 1)}>Back one event</button><button disabled={step === trace.length - 1} onClick={() => setStep(step + 1)}>Advance one event</button><button onClick={() => setStep(0)}>Restart trace</button></div>
    <p className="complexity-readout" aria-live="polite" data-frame-step={step}>{step + 1} / {trace.length} · {state.message}</p>
    <div className="complexity-frames" data-frame-count={state.frames.length}>
      {[...state.frames].reverse().map((frame, position) => <div className={`complexity-frame ${position === 0 ? 'is-current' : ''}`} key={frame.index}>
        <strong>suffix({frame.index}) <small>{position === 0 ? 'current' : 'suspended caller'}</small></strong>
        <span>{frame.phase === 'enter' ? 'Check the base case' : frame.phase === 'waiting' ? `${values[frame.index]} + ?` : `return ${frame.value}`}</span>
        <small>{frame.phase === 'waiting' ? `Waiting for suffix(${frame.index + 1})` : frame.phase === 'return' && frame.child !== null ? `Child returned ${frame.child}` : `${values.length - frame.index} values remain`}</small>
      </div>)}
      {!state.frames.length && <p>Stack empty. Caller receives <strong>{state.result}</strong>.</p>}
    </div>
    <p>{state.calls} calls entered so far; {state.frames.length} live now. The source list is shared, not copied into each frame.</p>
    <p className="lesson-note">A bounded model of the exact suffix-sum example: at most eight integers from −99 to 99. This shows call/return events, not every Python interpreter instruction. Trace recording itself uses additional storage; it is not included in the uninstrumented algorithm’s auxiliary-space claim.</p>
  </section>;
}
export function RecurrenceLevelsLab() {
  const [pattern, setPattern] = useState('twoHalfLinear');
  const [size, setSize] = useState(8);
  const [selectedLevel, setSelectedLevel] = useState(0);
  const model = recurrenceLevels(pattern, size);
  const selected = model.levels[selectedLevel];
  const maximum = Math.max(...model.levels.map(level => level.work));
  function reset() {
    setPattern('twoHalfLinear');
    setSize(8);
    setSelectedLevel(0);
  }
  return <section className="lesson-lab complexity-lab" aria-label="Recurrence level investigation">
    <h3>Count work across the whole recursion tree</h3>
    <p>Inspect which change affects depth and which affects work. These are exact toy recurrences with T(1)=1. Every leaf costs one unit; the bound describes growth, not seconds.</p>
    <div className="lesson-controls">
      <label>Recurrence<select value={pattern} onChange={event => {
          setPattern(event.target.value);
          setSelectedLevel(0);
        }}>{Object.entries(recurrencePatterns).map(([key, value]) => <option value={key} key={key}>{value.label}</option>)}</select></label>
      <label>Starting size<select aria-label="Recurrence input size" value={size} onChange={event => {
          setSize(Number(event.target.value));
          setSelectedLevel(0);
        }}>{[1, 2, 4, 8, 16, 32].map(value => <option key={value}>{value}</option>)}</select></label>
      <button onClick={reset}>Reset recurrence</button>
    </div>
    <p><strong>{recurrencePatterns[pattern].formula}</strong>; {recurrencePatterns[pattern].bound} as n grows.</p>
    <div className="complexity-levels">{model.levels.map(level => <button className={selectedLevel === level.depth ? 'is-selected' : ''} key={level.depth} aria-label={`Inspect recurrence depth ${level.depth}`} aria-pressed={selectedLevel === level.depth} onClick={() => setSelectedLevel(level.depth)}>
      <span>Depth {level.depth}<small>{level.nodes} × size {level.size}</small></span>
      <span className="complexity-work-bar" aria-hidden="true"><i style={{
            width: `${100 * level.work / maximum}%`
          }} /></span>
      <strong>{level.work}<small>work units</small></strong>
    </button>)}</div>
    <div className="complexity-node-strip" aria-label={`Selected level: ${selected.nodes} nodes, each with local work ${selected.localWork}`}>{Array.from({
        length: selected.nodes
      }, (_, index) => <span key={index}>size {selected.size}<strong>{selected.localWork} work</strong></span>)}</div>
    <p className="complexity-readout" aria-live="polite" data-recurrence-total={model.totalWork}>Depth {selected.depth}: {selected.nodes} calls × {selected.localWork} local work = {selected.work}. <strong>{model.totalWork} total work units</strong>; {model.totalCalls} total calls; at most {model.peakFrames} live frames in a sequential depth-first evaluation with constant-sized frames.</p>
    <LessonTable caption="Exact level accounting" headers={['Depth', 'Calls', 'Size / call', 'Work / call', 'Level work']} rows={model.levels.map(level => [level.depth, level.nodes, level.size, level.localWork, `${level.work}${level.leaf ? ' (leaves)' : ''}`])} />
    <p className="lesson-note">Bars share a linear work scale within this selection. They rescale when the recurrence changes: compare the numbers across selections. A level is not a simultaneous execution wave; this picture does not model parallel speedup, allocations or Python timings.</p>
  </section>;
}
export function SliceStorageFigure() {
  return <figure className="complexity-inline" aria-label="Retained suffix copies at deepest call for a four item list">
    <figcaption>At the deepest call, parents still retain their lists.</figcaption>
    <div className="complexity-slices">{[4, 3, 2, 1, 0].map((length, index) => <div key={length}><span>{index === 0 ? 'Caller’s input' : `Copy at depth ${index}`}</span><span className="complexity-slot-row">{Array.from({
            length
          }, (_, slot) => <i key={slot} aria-label="one reference slot" />)}{length === 0 && 'empty'}</span><strong>{length} slots</strong></div>)}</div>
    <p>The original has 4 slots. Copies retain 3 + 2 + 1 + 0 = 6 additional slots, plus list headers and call frames. These are copied references, not deep-copied elements or measured bytes.</p>
  </figure>;
}
