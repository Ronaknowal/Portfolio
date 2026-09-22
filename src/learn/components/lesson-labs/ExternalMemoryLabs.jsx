import { useId, useMemo, useState } from 'react';
import { bplusRange, btreeTrace, bufferTrace, mergePlan, shadowCommitTrace } from '../../data/external-memory-models.js';
import './external-memory-labs.css';
function StepControls({
  step,
  count,
  setStep,
  name
}) {
  return <div className="external-actions">
    <button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous {name}</button>
    <button disabled={step >= count - 1} onClick={() => setStep(step + 1)}>Next {name}</button>
    <button onClick={() => setStep(count - 1)}>Finish {name}</button>
    <button onClick={() => setStep(0)}>Restart {name}</button>
  </div>;
}
function Facts({
  values
}) {
  return <dl className="external-facts">{values.map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}</dl>;
}
function PageKeys({
  keys
}) {
  return <span className="external-keys">{keys.length ? keys.map((key, index) => <span key={index}>{key}</span>) : <em>empty</em>}</span>;
}
function PageTree({
  root,
  active = [],
  plus = false
}) {
  return <ul className="external-tree"><li>
    <div className={`external-page ${active.includes(root.id) ? 'is-active' : ''}`}>
      <strong>{root.id} · {root.children.length ? plus ? 'separator copies' : 'internal keys' : plus ? 'leaf records' : 'leaf keys'}</strong>
      <PageKeys keys={root.keys} />
      {plus && !root.children.length && <small>next → {root.next || 'end'}</small>}
      {root.children.length > 0 && <small>children in key order: {root.children.map(child => child.id).join(' · ')}</small>}
    </div>
    {root.children.length > 0 && <div className="external-branches">{root.children.map((child, index) => <div className="external-branch" key={child.id}>
      <span className="external-child-label">↳ child {index + 1}</span>
      <PageTree root={child} active={active} plus={plus} />
    </div>)}</div>}
  </li></ul>;
}
export function PageBufferLab() {
  const [pageSize, setPageSize] = useState(4);
  const [capacity, setCapacity] = useState(2);
  const [pattern, setPattern] = useState('sequential');
  const [step, setStep] = useState(0);
  const state = useMemo(() => bufferTrace(pageSize, capacity, pattern), [pageSize, capacity, pattern]);
  const frame = state.frames[Math.min(step, state.frames.length - 1)];
  const choose = (setter, value) => {
    setter(value);
    setStep(0);
  };
  const id = useId();
  return <section className="external-lab" aria-label="Pages and buffer residency">
    <h3>Follow the page, not just the record</h3>
    <p>Inspect the next load. Sequential and strided cases visit exactly the same 16 record addresses. The write case adds dirty-page eviction and a final flush.</p>
    <div className="external-controls">
      <label htmlFor={`${id}-size`}>Records per page<select id={`${id}-size`} value={pageSize} onChange={event => choose(setPageSize, Number(event.target.value))}>{[1, 2, 4, 8].map(value => <option key={value}>{value}</option>)}</select></label>
      <label htmlFor={`${id}-frames`}>Buffer frames<select id={`${id}-frames`} value={capacity} onChange={event => choose(setCapacity, Number(event.target.value))}>{[1, 2, 4].map(value => <option key={value}>{value}</option>)}</select></label>
      <label htmlFor={`${id}-pattern`}>Request pattern<select id={`${id}-pattern`} value={pattern} onChange={event => choose(setPattern, event.target.value)}><option value="sequential">Sequential addresses</option><option value="strided">Column-first addresses</option><option value="reuse">Repeated working set</option><option value="writes">Reads and writes</option></select></label>
    </div>
    <figure className="external-buffer-map">
      <div><h4>Storage layout · 16 record addresses</h4><div className="external-storage-pages">{Array.from({
            length: Math.ceil(16 / pageSize)
          }, (_, page) => <div className={`external-page ${frame.page === page ? 'is-active' : ''}`} key={page}><strong>P{page}</strong><PageKeys keys={Array.from({
              length: Math.min(pageSize, 16 - page * pageSize)
            }, (_, index) => page * pageSize + index)} /></div>)}</div></div>
      <div className="external-transfer" aria-hidden="true">↕ page-sized transfers</div>
      <div><h4>Resident frames · LRU → MRU</h4><div className="external-resident">{Array.from({
            length: capacity
          }, (_, index) => {
            const page = frame.resident[index];
            return <div className={`external-page ${page?.dirty ? 'is-dirty' : ''}`} key={index}>{page ? <><strong>P{page.id}</strong><span>{page.dirty ? 'dirty · write owed' : 'clean'}</span></> : <em>free frame</em>}</div>;
          })}</div></div>
      <figcaption>Page identity stays the same in storage and memory. A dirty badge means the resident version differs from storage. LRU is the leftmost occupied frame; every read or write makes its page most recent.</figcaption>
    </figure>
    <p className="external-current" aria-live="polite">Step {Math.min(step, state.frames.length - 1) + 1}/{state.frames.length}: {frame.action}.</p>
    <Facts values={[["Page reads", frame.reads], ['Page writes', frame.writes], ['Resident hits', frame.hits], ['Evicted page', frame.evicted ? `P${frame.evicted.id}, ${frame.evicted.dirty ? 'dirty: written' : 'clean: discarded'}` : 'none in this step']]} />
    <StepControls step={step} count={state.frames.length} setStep={setStep} name="request" />
    <button onClick={() => {
      setPageSize(4);
      setCapacity(2);
      setPattern('sequential');
      setStep(0);
    }}>Reset buffer experiment</button>
    <p>Cold fully associative LRU, write-allocate and write-back; one page transfer per miss or dirty write. Counters exclude CPU work and do not measure physical disk requests, time or durability. Finish both read orders, then increase the buffer from two to four frames.</p>
  </section>;
}
export function BTreePageLab() {
  const [degree, setDegree] = useState(2);
  const [preset, setPreset] = useState('insert');
  const [extra, setExtra] = useState([]);
  const [step, setStep] = useState(0);
  const [key, setKey] = useState(8);
  const [operation, setOperation] = useState('insert');
  const [lookup, setLookup] = useState(null);
  const state = useMemo(() => btreeTrace(degree, preset, extra), [degree, preset, extra]);
  const frame = state.frames[Math.min(step, state.frames.length - 1)];
  const id = useId();
  const reset = (nextPreset = 'insert', nextDegree = 2) => {
    setPreset(nextPreset);
    setDegree(nextDegree);
    setExtra([]);
    setStep(0);
    setLookup(null);
  };
  return <section className="external-lab" aria-label="B-tree page repairs">
    <h3>Keep the page tree valid as keys move</h3>
    <p>First follow the split sequence. Then load deletion to see keys rotate through a parent or pages merge. Page IDs retain identity; a removed page disappears from the reachable tree.</p>
    <div className="external-controls"><label htmlFor={`${id}-degree`}>Minimum degree t<select id={`${id}-degree`} value={degree} onChange={event => reset(preset, Number(event.target.value))}>{[2, 3, 4].map(value => <option key={value}>{value}</option>)}</select></label><label htmlFor={`${id}-scenario`}>Tree scenario<select id={`${id}-scenario`} value={preset} onChange={event => reset(event.target.value, degree)}><option value="insert">Insertion and splits</option><option value="delete">Deletion to empty</option><option value="duplicates">Duplicate and missing keys</option></select></label></div>
    <Facts values={[["Nonroot key range", `${degree - 1}…${2 * degree - 1}`], ['Maximum children', 2 * degree], ['Current event', `${Math.min(step, state.frames.length - 1) + 1}/${state.frames.length}`]]} />
    <p className="external-current" aria-live="polite">{frame.action}.</p>
    <StepControls step={step} count={state.frames.length} setStep={value => {
      setStep(value);
      setLookup(null);
    }} name="tree event" />
    <figure><PageTree root={frame.root} active={lookup ? lookup.visited : frame.pages} /><figcaption>Each outlined group is one abstract page. Child branches preserve key order; on narrow screens they stack with explicit parent/child labels. Gold marks pages involved in this event or the selected final-tree search. This is a textbook B-tree: internal keys are records, not copied leaf separators.</figcaption></figure>
    <p>A replacement step temporarily shows two occurrences before its recursive leaf removal. A merge can temporarily leave an empty root before promotion. “Complete” events restore the full public invariant; those intermediate states are not separate committed updates.</p>
    <div className="external-controls"><label htmlFor={`${id}-key`}>Key to try<input id={`${id}-key`} type="range" min="0" max="99" value={key} onChange={event => {
          setKey(Number(event.target.value));
          setLookup(null);
        }} /><output>{key}</output></label><label htmlFor={`${id}-operation`}>Operation<select id={`${id}-operation`} value={operation} onChange={event => setOperation(event.target.value)}><option value="insert">Insert</option><option value="delete">Delete</option></select></label></div>
    <div className="external-actions"><button disabled={extra.length >= 12} onClick={() => {
        const next = [...extra, [operation, key]];
        const nextState = btreeTrace(degree, preset, next);
        setExtra(next);
        setStep(nextState.frames.length - 1);
        setLookup(null);
      }}>Apply to final tree</button><button onClick={() => {
        setLookup(state.search(key));
        setStep(state.frames.length - 1);
      }}>Search final tree</button><button onClick={() => {
        reset();
        setKey(8);
        setOperation('insert');
      }}>Reset tree experiment</button></div>
    {lookup && <p role="status">{key} is {lookup.found ? 'present' : 'absent'}; final-tree path {lookup.visited.join(' → ')} ({lookup.visited.length} page visits). Cached pages would reduce actual reads.</p>}
    <p>Added operations apply after the preset's complete sequence, at most twelve per experiment. Search counts visited page identities, not cache misses; the page-buffer investigation explains that separate step. The model runs in memory and does not implement disk locking or recovery.</p>
  </section>;
}
export function BPlusContrastFigure() {
  return <figure className="external-inline external-contrast">
    <div><h4>Textbook B-tree · record 20 moves up</h4><div className="external-page"><strong>parent</strong><PageKeys keys={[20]} /></div><div className="external-pair"><div className="external-page"><strong>left child</strong><PageKeys keys={[5, 10]} /></div><div className="external-page"><strong>right child</strong><PageKeys keys={[25, 30]} /></div></div><p>20 is found in the parent; it is not repeated below.</p></div>
    <div><h4>B+ tree · separator 20 is copied</h4><div className="external-page"><strong>internal guide</strong><PageKeys keys={[20]} /></div><div className="external-pair"><div className="external-page"><strong>left leaf</strong><PageKeys keys={[5, 10]} /><small>next → right leaf</small></div><div className="external-page"><strong>right leaf</strong><PageKeys keys={[20, 25, 30]} /></div></div><p>Equality follows the right child to record 20; leaves carry the records.</p></div>
    <figcaption>Both store the same five records. These are schematic record-placement conventions, not a claim that their occupancy parameters or byte layouts are identical.</figcaption>
  </figure>;
}
export function BPlusRangeLab() {
  const [low, setLow] = useState(10),
    [high, setHigh] = useState(27);
  const [capacity, setCapacity] = useState(3),
    [fanout, setFanout] = useState(3),
    [step, setStep] = useState(0);
  const state = useMemo(() => bplusRange(low, high, capacity, fanout), [low, high, capacity, fanout]);
  const frame = state.frames[Math.min(step, state.frames.length - 1)];
  const choose = (setter, value) => {
    setter(value);
    setStep(0);
  };
  const id = useId();
  return <section className="external-lab" aria-label="B-plus linked leaf range">
    <h3>Descend once, then walk the leaf records</h3>
    <p>Inspect which separator handles equality at 20. Change the range to [20,20], then compare it with a range spanning several leaves.</p>
    <div className="external-controls">{[['Start', low, setLow], ['End', high, setHigh]].map(([label, value, setter]) => <label key={label} htmlFor={`${id}-${label}`}>{label} of inclusive range: {value}<input id={`${id}-${label}`} aria-label={`${label} of inclusive range`} type="range" min="0" max="40" value={value} onChange={event => choose(setter, Number(event.target.value))} /></label>)}<label htmlFor={`${id}-capacity`}>Records per leaf<select id={`${id}-capacity`} value={capacity} onChange={event => choose(setCapacity, Number(event.target.value))}>{[2, 3, 4].map(value => <option key={value}>{value}</option>)}</select></label><label htmlFor={`${id}-fanout`}>Internal fanout<select id={`${id}-fanout`} value={fanout} onChange={event => choose(setFanout, Number(event.target.value))}>{[2, 3, 4].map(value => <option key={value}>{value}</option>)}</select></label></div>
    <p className="external-current" aria-live="polite">{frame.action}.</p>
    <StepControls step={step} count={state.frames.length} setStep={setStep} name="range page" />
    <figure><PageTree root={state.root} active={frame.page ? [frame.page] : []} plus /><figcaption>The tree is bulk-built from sorted unique records. Internal separators copy the minimum of the right child's subtree. The next links connect leaves in record order; they need not be adjacent disk addresses.</figcaption></figure>
    <Facts values={[["Records returned so far", frame.result.join(', ') || 'none'], ['Cold query page reads so far', frame.page ? Math.min(step + 1, state.visited.length) : 0], ['Final query page path', state.visited.join(' → ') || 'none; reversed range']]} />
    <button onClick={() => {
      setLow(10);
      setHigh(27);
      setCapacity(3);
      setFanout(3);
      setStep(0);
    }}>Reset range experiment</button>
    <p>The first leaf can contain no qualifying record when the lower bound falls in a gap; one extra leaf can prove the upper bound is passed. Each displayed query begins cold. This bulk-layout model does not implement dynamic B+ insertion/deletion.</p>
  </section>;
}
export function ExternalMergeLab() {
  const [records, setRecords] = useState(32),
    [pageSize, setPageSize] = useState(4),
    [memoryPages, setMemoryPages] = useState(3),
    [step, setStep] = useState(0);
  const state = useMemo(() => mergePlan(records, pageSize, memoryPages), [records, pageSize, memoryPages]);
  const stage = state.stages[Math.min(step, state.stages.length - 1)];
  const id = useId();
  const choose = (setter, value) => {
    setter(value);
    setStep(0);
  };
  return <section className="external-lab" aria-label="External merge runs and transfers">
    <h3>Spend memory on more input runs—or on bigger pages?</h3>
    <p>Inspect the number of merge passes while changing the buffer budget. Every colored band below is an actual sorted run, grouped into separately materialized output files.</p>
    <div className="external-controls"><label htmlFor={`${id}-records`}>Record count<select id={`${id}-records`} value={records} onChange={event => choose(setRecords, Number(event.target.value))}>{[0, 7, 12, 20, 32, 48].map(value => <option key={value}>{value}</option>)}</select></label><label htmlFor={`${id}-page`}>Records per transfer page<select id={`${id}-page`} value={pageSize} onChange={event => choose(setPageSize, Number(event.target.value))}>{[2, 4, 8].map(value => <option key={value}>{value}</option>)}</select></label><label htmlFor={`${id}-memory`}>Available record-buffer pages<select id={`${id}-memory`} value={memoryPages} onChange={event => choose(setMemoryPages, Number(event.target.value))}>{[3, 4, 5, 8].map(value => <option key={value}>{value}</option>)}</select></label></div>
    <figure><div className="external-buffer-budget">{Array.from({
          length: memoryPages
        }, (_, index) => <div className={`external-page ${index === memoryPages - 1 ? 'is-output' : ''}`} key={index}><strong>{index === memoryPages - 1 ? 'output' : `input ${index + 1}`}</strong><span>{pageSize} records</span></div>)}</div><figcaption>During a merge: at most {state.fanIn} input page buffers and one output page buffer. Heap keys, cursors and language overhead require additional space in an implementation. During run formation the record workspace holds up to {state.memoryRecords} records.</figcaption></figure>
    {stage ? <><h4>{stage.name}</h4><div className="external-runs">{stage.runs.map((run, index) => <div className="external-run" key={index}><strong>Output run {index + 1} · {run.length} records · {Math.ceil(run.length / pageSize)} pages</strong>{stage.groups[index] && <small>inputs {stage.groups[index].inputIndices.map(value => value + 1).join(' + ')} → this run</small>}<div className="external-run-pages">{Array.from({
              length: Math.ceil(run.length / pageSize)
            }, (_, page) => <PageKeys key={page} keys={run.slice(page * pageSize, (page + 1) * pageSize)} />)}</div></div>)}</div><Facts values={[["This pass: page reads", stage.reads], ['This pass: page writes', stage.writes], ['Materialized passes in complete sort', state.stages.length], ['Whole sort: reads + writes', `${state.totalReads} + ${state.totalWrites} = ${state.totalReads + state.totalWrites}`]]} /><StepControls step={step} count={state.stages.length} setStep={setStep} name="merge pass" /></> : <p role="status">Empty input: zero runs, zero transfers and no merge pass.</p>}
    <button onClick={() => {
      setRecords(32);
      setPageSize(4);
      setMemoryPages(3);
      setStep(0);
    }}>Reset merge experiment</button>
    <p>Input is the deterministic reverse order 0…N−1. The browser stores tiny runs to expose their contents; it is not an external-memory engine. The native file program below performs real bounded-fan-in merges. Counts include every materialized singleton merge group and partial page, exclude input creation/output inspection, and are not device timings.</p>
  </section>;
}
export function ShadowCommitLab() {
  const [early, setEarly] = useState(false),
    [step, setStep] = useState(0);
  const state = shadowCommitTrace(early);
  const frame = state.frames[Math.min(step, state.frames.length - 1)];
  const id = useId();
  return <section className="external-lab" aria-label="Shadow page crash recovery">
    <h3>Interrupt the update before the next write</h3>
    <p>Assume one writer, immutable old pages, completed page writes that are durable, and an atomic durable metadata-root switch. These are the model's assumptions, not promises about an ordinary file write.</p>
    <label htmlFor={id}>Publication order<select id={id} value={early ? 'early' : 'safe'} onChange={event => {
        setEarly(event.target.value === 'early');
        setStep(0);
      }}><option value="safe">Durable pages first (safe)</option><option value="early">Root first (broken)</option></select></label>
    <figure className="external-durability"><div className="external-root-pointer"><strong>Durable metadata</strong><span>root → {frame.committedRoot}</span></div><div><h4>Durable page set</h4><div className="external-pair">{frame.durable.map(name => <div className={`external-page ${frame.committedRoot === name ? 'is-active' : ''}`} key={name}><strong>{name}</strong><span>{state.pages[name].child ? `child → ${state.pages[name].child}` : `value ${state.pages[name].value}`}</span></div>)}</div></div><div><h4>Volatile prepared pages</h4><div className="external-pair">{frame.pending.length ? frame.pending.map(name => <div className="external-page is-pending" key={name}><strong>{name}</strong><span>lost on crash</span></div>) : <p>none pending</p>}</div></div><figcaption>A crash discards volatile preparation and follows only the durable metadata/root/page chain. Old pages stay allocated here; reclaiming them safely is a separate concern.</figcaption></figure>
    <p className="external-current" aria-live="polite">{frame.action}.</p>
    <p className="external-recovery" role="status">Crash now: {frame.missing.length ? `invalid reachable root; missing ${frame.missing.join(', ')}` : `recover value ${frame.recoveredValue}`}.</p>
    <StepControls step={step} count={state.frames.length} setStep={setStep} name="commit stage" />
    <button onClick={() => {
      setEarly(false);
      setStep(0);
    }}>Reset commit experiment</button>
    <p>Safe ordering recovers either the old value 5 or complete new value 9. Early publication exposes a root whose page does not exist durably. This is an abstract shadow-page protocol, not SQLite's rollback journal, a real crash test or an implementation of fsync.</p>
  </section>;
}
