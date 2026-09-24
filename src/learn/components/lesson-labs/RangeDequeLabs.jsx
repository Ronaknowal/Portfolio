import { useId, useState } from 'react';
import { movingMaximumTrace, parseDequeValues, signedShortestTrace } from '../../data/range-deque-models.js';
import { RangeCells, RangeField, RangeSteps, rangeInteger } from './RangeQueryLabs.jsx';
import './range-deque-labs.css';
function CandidateDeque({
  indices,
  values,
  prefix = false
}) {
  return <div className="range-scroll" tabIndex="0" role="region" aria-label="Candidate deque in front-to-back order"><div className="range-candidates"><span>front</span>{indices.length ? indices.map(index => <div key={index}><small>{prefix ? 'boundary' : 'index'} {index}</small><strong>{values[index]}</strong></div>) : <em>empty</em>}<span>back</span></div></div>;
}
export function MovingMaximumLab() {
  const initial = [4, 2, 2, 5, 1, 3, 0, 2];
  const [draft, setDraft] = useState(initial.join(', '));
  const [width, setWidth] = useState('3');
  const [trace, setTrace] = useState(() => movingMaximumTrace());
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const frame = trace.frames[step];
  function apply(values = parseDequeValues(draft), nextWidth = rangeInteger(width)) {
    setTrace(movingMaximumTrace(values, nextWidth));
    setStep(0);
    setError('');
  }
  return <section className="range-lab" aria-label="Moving maximum investigation">
    <div className="range-eyebrow">Investigate · candidate lifetimes</div><h3>Expired is different from dominated</h3>
    <p>Inspect what happens to the older value 2 when another 2 arrives. Later, value 5 removes several candidates at once. Follow indices as well as values.</p>
    <div className="range-controls"><RangeField label="Array draft · up to 8 integers, −20…20" type="text" value={draft} onChange={setDraft} /><RangeField label="Window width · 1…8" value={width} onChange={setWidth} /><button type="button" onClick={() => {
        try {
          apply();
        } catch (problem) {
          setError(problem.message);
        }
      }}>Apply maximum inputs</button><button type="button" onClick={() => {
        setDraft(initial.join(', '));
        setWidth('3');
        apply(initial, 3);
      }}>Reset maximum lab</button></div>
    {error && <p className="range-error" role="alert">{error}</p>}
    <p><strong>{frame.index < 0 ? 'No arrival yet' : `Current range [${frame.left},${frame.right})`}</strong>{frame.right < trace.width ? ' · not yet a full window' : ' · full window'}. Green cells show the current range; the deque below contains only surviving maximum candidates.</p>
    <RangeCells values={trace.values} active={trace.values.map((_, index) => index).filter(index => frame.left <= index && index < frame.right)} />
    <CandidateDeque indices={frame.deque} values={trace.values} />
    <div className={`range-removal ${['expire', 'dominate'].includes(frame.phase) ? 'is-removing' : ''}`}><strong>{frame.phase === 'expire' ? 'EXPIRE · from the front' : frame.phase === 'dominate' ? 'DOMINATE · from the back' : 'Candidate rule'}</strong><span>{frame.removed !== null ? `Removed original index ${frame.removed}, value ${trace.values[frame.removed]}.` : 'Keep indices increasing and values strictly decreasing after each append. Newest equal maximum wins.'}</span></div>
    <p className="range-event" role="status">{frame.message}</p><RangeSteps step={step} setStep={setStep} frames={trace.frames} />
    <div className="range-scroll" tabIndex="0" role="region" aria-label="Emitted moving maxima"><table><thead><tr><th>Completed window</th><th>Maximum value</th><th>Original argmax index</th></tr></thead><tbody>{frame.answers.map(answer => <tr key={answer.left}><td>[{answer.left},{answer.right})</td><td>{answer.value}</td><td>{answer.index}</td></tr>)}{!frame.answers.length && <tr><td colSpan="3">No complete window emitted yet.</td></tr>}</tbody></table></div>
    <p className="range-key">So far: {frame.pushes} appends, {frame.pops} removals. Every index can leave only once. Trace snapshots copy small arrays; the Python deque uses endpoint operations.</p>
    <p className="range-transfer">Try [2,2,2] with width 2, then width 1 and a width larger than the input. To keep the oldest tied maximum instead, which comparison must become strict?</p>
  </section>;
}
function PrefixBoundaryPlot({
  trace,
  frame
}) {
  const titleId = useId();
  const width = Math.max(250, 66 + trace.values.length * 62);
  const height = 250;
  const minimum = Math.min(0, ...trace.prefixes, ...trace.prefixes.map(value => value - trace.target)) - 1;
  const maximum = Math.max(1, ...trace.prefixes) + 1;
  const x = index => 42 + index * (width - 62) / Math.max(1, trace.values.length);
  const y = value => 24 + (maximum - value) / (maximum - minimum) * 182;
  const tickValues = [...new Set([minimum, 0, maximum])].sort((a, b) => a - b);
  return <div className="range-scroll" tabIndex="0" role="region" aria-label="Computed prefix totals versus boundary index; scroll horizontally if needed"><svg className="prefix-boundary-plot" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={titleId}>
    <title id={titleId}>Each point is a computed prefix total P at its boundary index. Green rings mark candidate starts; amber marks the current end. The dashed threshold is current prefix minus target, {frame.threshold}.</title>
    {tickValues.map(value => <g key={value}><line className="prefix-grid" x1="38" x2={width - 12} y1={y(value)} y2={y(value)} /><text className="prefix-y-label" x="32" y={y(value) + 4}>{value}</text></g>)}
    <line className="prefix-threshold" x1="38" x2={width - 12} y1={y(frame.threshold)} y2={y(frame.threshold)} />
    <path className="prefix-history" d={trace.prefixes.map((value, index) => `${index ? 'L' : 'M'}${x(index)},${y(value)}`).join(' ')} />
    {trace.prefixes.map((value, index) => <g key={index} className={`${index > frame.index ? 'is-future' : ''} ${frame.deque.includes(index) ? 'is-candidate' : ''} ${index === frame.index ? 'is-current' : ''}`}><circle cx={x(index)} cy={y(value)} r="6" /><text className="prefix-point-label" x={x(index)} y={y(value) - 12}>{value}</text><text className="prefix-x-label" x={x(index)} y="224">{index}</text></g>)}
    <text x={width / 2} y="244" className="prefix-axis-label">prefix boundary index</text>
  </svg></div>;
}
export function SignedShortestRangeLab() {
  const [draft, setDraft] = useState('1, -1, 5');
  const [target, setTarget] = useState('5');
  const [trace, setTrace] = useState(() => signedShortestTrace());
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const frame = trace.frames[step];
  function apply(values = parseDequeValues(draft), nextTarget = rangeInteger(target)) {
    setTrace(signedShortestTrace(values, nextTarget));
    setStep(0);
    setError('');
  }
  return <section className="range-lab" aria-label="Signed shortest range investigation">
    <div className="range-eyebrow">Investigate · ordered prefix candidates</div><h3>A later, lower prefix is a better start</h3>
    <p>The prefix totals for [1,−1,5] are [0,1,0,5]. Investigate why boundary 2 replaces both old starts before the final value arrives, then returns range [2,3).</p>
    <div className="range-controls"><RangeField label="Array draft · up to 8 integers, −20…20" type="text" value={draft} onChange={setDraft} /><RangeField label="Positive target · 1…100" value={target} onChange={setTarget} /><button type="button" onClick={() => {
        try {
          apply();
        } catch (problem) {
          setError(problem.message);
        }
      }}>Apply shortest inputs</button><button type="button" onClick={() => {
        setDraft('1, -1, 5');
        setTarget('5');
        apply([1, -1, 5], 5);
      }}>Reset shortest lab</button></div>
    {error && <p className="range-error" role="alert">{error}</p>}
    <PrefixBoundaryPlot trace={trace} frame={frame} />
    <p className="range-key">Vertical position: computed prefix sum. Horizontal position: boundary index. Green ring: candidate start. Amber ring: current end. Dashed line: P[end]−target = {frame.threshold}; eligible starts lie at or below it. Faded points are known input prefixes not processed yet.</p>
    <CandidateDeque indices={frame.deque} values={trace.prefixes} prefix />
    <div className={`range-removal ${['record-front', 'dominate-back'].includes(frame.phase) ? 'is-removing' : ''}`}><strong>{frame.phase === 'record-front' ? 'RECORD THEN RETIRE · front' : frame.phase === 'dominate-back' ? 'DOMINATED START · back' : 'Two different retirement proofs'}</strong><span>{frame.phase === 'record-front' ? 'This start has a valid ending now; any later ending is longer.' : frame.phase === 'dominate-back' ? 'This later start has no larger prefix, giving future ends at least as much sum in less length.' : 'Check valid earlier starts before inserting the current boundary.'}</span></div>
    <p className="range-event" role="status">{frame.message}</p><RangeSteps step={step} setStep={setStep} frames={trace.frames} />
    <p><strong>Best actual range:</strong> {frame.best ? `[${frame.best.left},${frame.best.right}), length ${frame.best.length}, sum ${frame.best.sum}.` : 'none yet.'} Highlighted elements below belong to the best range.</p><RangeCells values={trace.values} active={trace.values.map((_, index) => index).filter(index => frame.best && frame.best.left <= index && index < frame.best.right)} />
    <details className="range-data"><summary>Read the exact prefix table</summary><div className="range-scroll" tabIndex="0" role="region" aria-label="Exact prefix totals"><table><thead><tr><th>Boundary</th><th>Prefix total</th><th>Candidate now?</th></tr></thead><tbody>{trace.prefixes.map((value, index) => <tr key={index}><td>{index}</td><td>{value}</td><td>{frame.deque.includes(index) ? 'yes' : 'no'}</td></tr>)}</tbody></table></div></details>
    <p className="range-transfer">Try [2,−1,2] with target 3, then [−2,−1] with target 1. Explain why “remove from the front” means something different here than expiry in the fixed-width lab.</p>
  </section>;
}
