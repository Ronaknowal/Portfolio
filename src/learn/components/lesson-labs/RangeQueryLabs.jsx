import { useId, useState } from 'react';
import { displayRangeValue, fenwickAdd, fenwickBlocks, fenwickPrefix, fenwickState, lazyRangeOperation, lazyState, parseRangeValues, rangeReadings, segmentAssign, segmentGeometry, segmentQuery, segmentState, sparseMinimum, weightedIncreasingPlan } from '../../data/range-query-models.js';
import './range-query-labs.css';
export function rangeInteger(text) {
  if (!/^-?\d+$/.test(String(text).trim())) throw new Error('Enter a whole number in each numeric field; a blank is not zero.');
  const value = Number(text);
  if (!Number.isSafeInteger(value)) throw new Error('Use an integer within the displayed teaching bounds.');
  return value;
}
export function RangeField({
  label,
  value,
  onChange,
  type = 'number',
  children
}) {
  const id = useId();
  return <div className="range-field"><label htmlFor={id}>{label}</label>{children ? <select id={id} value={value} onChange={event => onChange(event.target.value)}>{children}</select> : <input id={id} type={type} value={value} onChange={event => onChange(event.target.value)} />}</div>;
}
export function RangeSteps({
  step,
  setStep,
  frames
}) {
  return <div className="range-steps" aria-label="Trace controls">
    <button type="button" disabled={!step} onClick={() => setStep(step - 1)}>Previous</button>
    <span>Event {step + 1} of {frames.length}</span>
    <button type="button" disabled={step === frames.length - 1} onClick={() => setStep(step + 1)}>Next event</button>
    <button type="button" disabled={step === frames.length - 1} onClick={() => setStep(frames.length - 1)}>Finish operation</button>
  </div>;
}
export function RangeCells({
  values,
  active = [],
  label = 'Array values',
  indices = true
}) {
  return <>{values.length > 4 && <p className="range-pan-hint">{values.length} positions · swipe or focus and use arrow keys to see the full row.</p>}<div className="range-scroll" tabIndex="0" role="region" aria-label={`${label}; scroll horizontally if needed`}>
    <div className="range-cells" aria-label={label}>{values.length ? values.map((value, index) => <div key={index} className={`range-cell ${active.includes(index) ? 'is-active' : ''}`}>
      {indices && <small>i={index}</small>}<strong>{value}</strong>
    </div>) : <p>Empty array · no element positions.</p>}</div>
  </div></>;
}
function RangeTree({
  state,
  current,
  selected = [],
  dirty = [],
  lazy = false
}) {
  const titleId = useId();
  const geometry = segmentGeometry(state);
  const pending = node => state.multipliers?.[node.id] !== 1 || state.additions?.[node.id] !== 0;
  const mapLabel = node => state.multipliers[node.id] === 0 ? `set ${state.additions[node.id]}` : `add ${state.additions[node.id]}`;
  return <>
    <div className="range-scroll" tabIndex="0" role="region" aria-label="Interval tree; scroll horizontally to retain readable labels">
      <svg className="range-tree" width={geometry.width} height={geometry.height + (lazy ? 20 : 0)} viewBox={`0 0 ${geometry.width} ${geometry.height + (lazy ? 20 : 0)}`} role="img" aria-labelledby={titleId}>
        <title id={titleId}>Ordered interval tree. Each node shows its external half-open range and stored {state.operation || 'sum'}. {lazy ? 'Pending child updates appear below nodes.' : ''}</title>
        {geometry.nodes.filter(node => node.parent).map(node => {
          const parent = geometry.nodes.find(item => item.id === node.parent);
          return <line key={`edge-${node.id}`} x1={parent.x} y1={parent.y + 48} x2={node.x} y2={node.y} />;
        })}
        {geometry.nodes.map(node => <g key={node.id} className={`${selected.includes(node.id) ? 'is-selected' : ''} ${current === node.id ? 'is-current' : ''} ${node.padding ? 'is-padding' : ''} ${dirty.includes(node.id) ? 'is-dirty' : ''}`}>
          <rect x={node.x - 27} y={node.y} width="54" height="48" rx="3" />
          <text x={node.x} y={node.y + 16} className="range-node-interval">[{node.left},{node.right})</text>
          <text x={node.x} y={node.y + 37} className="range-node-value">{displayRangeValue(node.value)}</text>
          {lazy && pending(node) && <text className="range-node-tag" x={node.x} y={node.y + 64}>{mapLabel(node)}</text>}
          {dirty.includes(node.id) && <text className="range-node-tag" x={node.x} y={node.y + 64}>repair</text>}
        </g>)}
      </svg>
    </div>
    <p className="range-key">Amber outline: current event. Green: selected query interval. Dashed: padding{lazy ? ' or a parent awaiting repair. Tags are already included in that node’s sum' : ''}.</p>
    <details className="range-data"><summary>Read the exact node table</summary><div className="range-scroll" tabIndex="0" role="region" aria-label="Exact interval node table"><table><thead><tr><th>Node ID</th><th>Range</th><th>Stored summary</th>{lazy && <th>Pending map for children</th>}</tr></thead><tbody>{geometry.nodes.map(node => <tr key={node.id}><td>{node.id}</td><td>[{node.left},{node.right}){node.padding ? ' padding' : ''}</td><td>{displayRangeValue(node.value)}{dirty.includes(node.id) ? ' · awaiting repair' : ''}</td>{lazy && <td>{pending(node) ? mapLabel(node) : 'identity'}</td>}</tr>)}</tbody></table></div></details>
  </>;
}
export function SegmentRangeLab() {
  const [draft, setDraft] = useState('2, 1, 3, 4');
  const [operation, setOperation] = useState('sum');
  const [state, setState] = useState(() => segmentState());
  const [trace, setTrace] = useState(() => segmentQuery(segmentState(), 1, 4));
  const [step, setStep] = useState(0);
  const [left, setLeft] = useState('1');
  const [right, setRight] = useState('4');
  const [index, setIndex] = useState('2');
  const [value, setValue] = useState('8');
  const [error, setError] = useState('');
  const frame = trace.frames[step];
  const busy = step < trace.frames.length - 1;
  function run(action) {
    try {
      const next = action();
      setState(next.state);
      setTrace(next);
      setStep(0);
      setError('');
    } catch (problem) {
      setError(problem.message);
    }
  }
  function load(text = draft) {
    run(() => {
      const next = segmentState(parseRangeValues(text), operation);
      setRight(String(next.values.length));
      setLeft('0');
      return segmentQuery(next, 0, next.values.length);
    });
  }
  return <section className="range-lab" aria-label="Segment range investigation">
    <div className="range-eyebrow">Investigate · interval summaries</div>
    <h3>Cover the range once, in array order</h3>
    <p>Inspect which nodes cover [1,4). Then assign index 2 to 8: which ancestors must change? Edit drafts and apply to restart; finish an operation before starting another.</p>
    <div className="range-controls"><RangeField label="Array draft · up to 8 integers, −99…99" type="text" value={draft} onChange={setDraft} /><RangeField label="Combine draft" value={operation} onChange={setOperation}><option value="sum">Sum</option><option value="min">Minimum</option><option value="max">Maximum</option></RangeField><button type="button" onClick={() => load()}>Apply array and combine</button><button type="button" onClick={() => {
        setDraft('2, 1, 3, 4');
        setOperation('sum');
        const next = segmentState();
        setState(next);
        setTrace(segmentQuery(next, 1, 4));
        setStep(0);
        setLeft('1');
        setRight('4');
        setIndex('2');
        setValue('8');
        setError('');
      }}>Reset segment lab</button></div>
    <div className="range-controls"><RangeField label="Left boundary" value={left} onChange={setLeft} /><RangeField label="Right boundary · excluded" value={right} onChange={setRight} /><button type="button" disabled={busy} onClick={() => run(() => segmentQuery(state, rangeInteger(left), rangeInteger(right)))}>Query range</button><RangeField label="Assignment index" value={index} onChange={setIndex} /><RangeField label="New value" value={value} onChange={setValue} /><button type="button" disabled={busy} onClick={() => run(() => segmentAssign(state, rangeInteger(index), rangeInteger(value)))}>Assign point</button></div>
    {error && <p role="alert" className="range-error">{error}</p>}
    <RangeCells values={frame.state.values} active={trace.kind === 'query' ? frame.state.values.map((_, i) => i).filter(i => trace.left <= i && i < trace.right) : [trace.index]} />
    <RangeTree state={frame.state} current={frame.current} selected={[...frame.leftNodes, ...frame.rightNodes]} />
    {trace.kind === 'query' && <div className="range-accumulators"><div><small>Left · append</small><strong>{displayRangeValue(frame.leftValue)}</strong><span>nodes {frame.leftNodes.join(' → ') || 'none'}</span></div><span>then</span><div><small>Right · prepend</small><strong>{displayRangeValue(frame.rightValue)}</strong><span>nodes {frame.rightNodes.join(' → ') || 'none'}</span></div></div>}
    <p className="range-event" role="status">{frame.message}</p><RangeSteps step={step} setStep={setStep} frames={trace.frames} />
    <p className="range-transfer">Try five values: padded leaves must use the selected identity. Query an empty range. Why would combining the right intervals in the opposite order matter for text, though not for sum?</p>
  </section>;
}
export function FenwickBlockLab() {
  const [draft, setDraft] = useState(rangeReadings.join(', '));
  const [state, setState] = useState(() => fenwickState());
  const [trace, setTrace] = useState(() => fenwickPrefix(fenwickState(), 7));
  const [step, setStep] = useState(0);
  const [end, setEnd] = useState('7');
  const [index, setIndex] = useState('4');
  const [delta, setDelta] = useState('3');
  const [error, setError] = useState('');
  const frame = trace.frames[step];
  const blocks = fenwickBlocks(frame.state);
  const busy = step < trace.frames.length - 1;
  function run(action) {
    try {
      const next = action();
      setState(next.state);
      setTrace(next);
      setStep(0);
      setError('');
    } catch (problem) {
      setError(problem.message);
    }
  }
  return <section className="range-lab" aria-label="Fenwick block investigation">
    <div className="range-eyebrow">Investigate · binary prefix blocks</div><h3>Read backward; update containing blocks forward</h3>
    <p>For prefix length 7, follow 7→6→4→0. Each green lane supplies one disjoint block. An increment at external index 4 instead visits internal positions 5→6→8.</p>
    <div className="range-controls"><RangeField label="Array draft · up to 8 integers, −99…99" type="text" value={draft} onChange={setDraft} /><button type="button" onClick={() => run(() => {
        const next = fenwickState(parseRangeValues(draft));
        setEnd(String(next.values.length));
        return fenwickPrefix(next, next.values.length);
      })}>Apply array</button><button type="button" onClick={() => {
        setDraft(rangeReadings.join(', '));
        setEnd('7');
        setIndex('4');
        setDelta('3');
        run(() => fenwickPrefix(fenwickState(), 7));
      }}>Reset Fenwick lab</button></div>
    <div className="range-controls"><RangeField label="Prefix end · excluded" value={end} onChange={setEnd} /><button type="button" disabled={busy} onClick={() => run(() => fenwickPrefix(state, rangeInteger(end)))}>Read prefix</button><RangeField label="External element index" value={index} onChange={setIndex} /><RangeField label="Increment · −20…20" value={delta} onChange={setDelta} /><button type="button" disabled={busy} onClick={() => run(() => fenwickAdd(state, rangeInteger(index), rangeInteger(delta)))}>Add at point</button></div>
    {error && <p role="alert" className="range-error">{error}</p>}
    <p className="range-pan-hint">Pan horizontally to see every block. Original indices and binary labels retain their size.</p>
    <div className="range-scroll" tabIndex="0" role="region" aria-label="Binary block lanes and original values; scroll horizontally if needed"><div className="fenwick-lanes" style={{
        width: Math.max(360, 130 + frame.state.values.length * 48)
      }}>
      <div className="fenwick-heading"><span>Internal index<br />binary · lowbit</span><div className="fenwick-array">{frame.state.values.map((item, i) => <div key={i}><small>i={i}</small><strong>{item}</strong></div>)}</div></div>
      {blocks.map(block => <div key={block.internal} className={`fenwick-lane ${frame.visited.includes(block.internal) ? 'is-selected' : ''} ${frame.current === block.internal ? 'is-current' : ''}`}><span><b>{block.internal}</b> · {block.binary}<small>lowbit {block.lowbit}</small></span><div className="fenwick-track"><div style={{
              left: block.left * 48,
              width: (block.right - block.left) * 48
            }}><strong>{block.value}</strong><small>[{block.left},{block.right})</small></div></div></div>)}
      {!blocks.length && <p>Empty tree · prefix 0 is zero.</p>}
    </div></div>
    <p className="range-key">Green: visited block. Amber outline: current write/read. During an update, other containing blocks await their writes.</p>
    <div className="range-readout"><span>Visited internal indices <b>{frame.visited.join(' → ') || 'none'}</b></span><span>Next internal index <b>{frame.cursor}</b></span>{trace.kind === 'prefix' && <span>Accumulated sum <b>{frame.result}</b></span>}</div>
    <p className="range-event" role="status">{frame.message}</p><RangeSteps step={step} setStep={setStep} frames={trace.frames} />
    <p className="range-transfer">Read prefix 0, then read the whole array. After adding 3 at index 4, read prefix 7 again. What delta would assign that element to 2?</p>
  </section>;
}
export function LazyRangeLab() {
  const [draft, setDraft] = useState('2, 1, 3, 4');
  const [trace, setTrace] = useState(() => lazyRangeOperation(lazyState(), 'add', 0, 4, 3));
  const [step, setStep] = useState(0);
  const [kind, setKind] = useState('set');
  const [left, setLeft] = useState('1');
  const [right, setRight] = useState('3');
  const [amount, setAmount] = useState('5');
  const [error, setError] = useState('');
  const frame = trace.frames[step];
  const busy = step < trace.frames.length - 1;
  function run(action) {
    try {
      const next = action();
      setTrace(next);
      setStep(0);
      setError('');
    } catch (problem) {
      setError(problem.message);
    }
  }
  // The initial operation's completed state is the next operation's starting state.
  const completedState = trace.state;
  return <section className="range-lab" aria-label="Lazy range propagation investigation">
    <div className="range-eyebrow">Investigate · deferred child work</div><h3>A current parent can have stale children</h3>
    <p>First finish adding 3 to [0,4). The root sum becomes 22 while its children retain old sums. Then run the prepared assignment on [1,3): watch the pending addition move before any partial descent.</p>
    <div className="range-controls"><RangeField label="Array draft · up to 8 integers, −99…99" type="text" value={draft} onChange={setDraft} /><button type="button" onClick={() => run(() => {
        const next = lazyState(parseRangeValues(draft));
        setLeft('0');
        setRight(String(next.values.length));
        return lazyRangeOperation(next, 'query', 0, next.values.length);
      })}>Apply array</button><button type="button" onClick={() => {
        setDraft('2, 1, 3, 4');
        setKind('set');
        setLeft('1');
        setRight('3');
        setAmount('5');
        run(() => lazyRangeOperation(lazyState(), 'add', 0, 4, 3));
      }}>Reset lazy lab</button></div>
    <div className="range-controls"><RangeField label="Next operation" value={kind} onChange={setKind}><option value="set">Set values</option><option value="add">Add values</option><option value="query">Query sum</option></RangeField><RangeField label="Left boundary" value={left} onChange={setLeft} /><RangeField label="Right boundary · excluded" value={right} onChange={setRight} />{kind !== 'query' && <RangeField label="Amount · −20…20" value={amount} onChange={setAmount} />}<button type="button" disabled={busy} onClick={() => run(() => lazyRangeOperation(completedState, kind, rangeInteger(left), rangeInteger(right), kind === 'query' ? 0 : rangeInteger(amount)))}>Run range operation</button></div>
    {error && <p role="alert" className="range-error">{error}</p>}
    <p><strong>Logical array:</strong> all pending maps included. Intermediate update frames can show a partly applied operation.</p><RangeCells values={frame.effective} active={frame.effective.map((_, index) => index).filter(index => trace.left <= index && index < trace.right)} />
    <p><strong>Stored sums below:</strong> a node's own tag is already included in its sum. A tag on an ancestor may still be absent from this node.</p><RangeTree state={frame.state} lazy current={frame.current} selected={frame.taken} dirty={frame.dirty} />
    <div className="range-readout"><span>Action <b>{trace.kind} [{trace.left},{trace.right}){trace.kind !== 'query' ? ` ${trace.amount}` : ''}</b></span>{trace.kind === 'query' && <span>Consumed sum <b>{frame.result}</b></span>}<span>Parents awaiting repair <b>{frame.dirty.join(', ') || 'none'}</b></span></div>
    <p className="range-event" role="status">{frame.message}</p><RangeSteps step={step} setStep={setStep} frames={trace.frames} />
    <p className="range-transfer">Compare “assign all to 4, then add 3” with “add 3, then assign all to 4.” Query a singleton to force pending work downward. Why must a full-cover query use the sum without adding its tag again?</p>
  </section>;
}
export function PrefixCancellationFigure() {
  return <figure className="range-figure"><figcaption>Reuse a prefix, then cancel what lies before the range</figcaption><RangeCells values={rangeReadings} active={[1, 2, 3, 4, 5, 6]} /><div className="range-equation"><span>P[7] = 2 + <b>15</b></span><span>− P[1] = −2</span><strong>sum [1,7) = 15</strong></div><p>Changing index 4 changes P[5], P[6], P[7] and P[8]. A static prefix cache saves query work by accepting expensive changes.</p></figure>;
}
export function DifferenceBoundaryFigure() {
  const deltas = [0, 3, 0, 0, -3, 0];
  return <figure className="range-figure"><figcaption>Adding 3 on [1,4) changes only two difference boundaries</figcaption><RangeCells values={deltas} label="Boundary deltas" /><p>Boundary 1 turns +3 on. Boundary 4 turns it off. Reconstructing prefixes gives the additions below; the final boundary is outside the five-element array.</p><RangeCells values={[0, 3, 3, 3, 0]} active={[1, 2, 3]} label="Resulting element increments" /><div className="range-equation"><span>In a prefix of length t, boundary j contributes</span><strong>(t − j) × d[j], for j &lt; t</strong></div></figure>;
}
export function SparseOverlapFigure() {
  const plan = sparseMinimum(rangeReadings, 1, 7);
  return <figure className="range-figure"><figcaption>Two length-four blocks cover a length-six query</figcaption><p>Query [1,7). The overlap [3,5) is deliberate.</p><RangeCells values={rangeReadings} active={[1, 2, 3, 4]} label="First block [1,5)" /><RangeCells values={rangeReadings} active={[3, 4, 5, 6]} label="Second block [3,7)" /><div className="range-equation"><span>min(0,0) = <b>{plan.result}</b> · correct</span><span>8 + 11 = <b>19</b> · double counts 4 + 0</span><strong>The real range sum is 15.</strong></div></figure>;
}
export function WeightedRankFigure() {
  const plan = weightedIncreasingPlan();
  return <figure className="range-figure"><figcaption>Keep input time order; query by value rank</figcaption><p>Values [3,1,2,2,4], weights [4,2,5,20,3]. Sorted coordinate ranks map 1→0, 2→1, 3→2, 4→3. Two input positions share rank 1 but remain different occurrences.</p><div className="range-scroll" tabIndex="0" role="region" aria-label="Weighted increasing subsequence trace"><table><thead><tr><th>Input index</th><th>Value / rank</th><th>Weight</th><th>Best smaller-value score</th><th>Candidate</th><th>Rank score retained</th></tr></thead><tbody>{plan.frames.map(frame => <tr key={frame.index}><td>{frame.index}</td><td>{frame.value} / {frame.rank}</td><td>{frame.weight}</td><td>{frame.prior.score}</td><td>{frame.candidate.score}</td><td>{frame.stored.score}</td></tr>)}</tbody></table></div><p><strong>Witness indices {plan.witness.join(' → ')}:</strong> values 1→2→4, weights 2+20+3={plan.score}. At the second value 2, querying ranks below 1 excludes the earlier equal value.</p></figure>;
}
