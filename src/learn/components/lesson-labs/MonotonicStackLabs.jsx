import { useMemo, useState } from 'react';
import { histogramState, nextGreaterTrace, parseStackValues } from '../../data/monotonic-stack-models.js';
import './monotonic-stack-labs.css';
function Bars({
  values,
  current = null,
  resolved = null,
  candidate = null,
  label
}) {
  if (!values.length) return <p className="monostack-empty">Empty input: no bar, unresolved index or positive rectangle.</p>;
  const minimum = Math.min(0, ...values);
  const maximum = Math.max(1, ...values);
  const y = value => 25 + 145 * (maximum - value) / (maximum - minimum);
  const x = index => 36 + index * 52;
  const width = Math.max(300, values.length * 52 + 72);
  return <div className="monostack-scroll" role="region" tabIndex="0" aria-label={label}>
    <svg width={width} height={resolved === null ? 236 : 292} viewBox={`0 0 ${width} ${resolved === null ? 236 : 292}`} role="img" aria-label={label}>
      <path d={`M30 ${y(0)}H${x(values.length) + 6}`} stroke="#878d79" />
      <text x="8" y={y(0) + 5} className="monostack-small">0</text>
      {values.map((value, index) => <g key={index} data-bar={index}>
        <rect x={x(index)} y={Math.min(y(value), y(0))} width="52" height={Math.abs(y(value) - y(0))} fill={index === current ? '#b29338' : '#314936'} stroke={candidate && [candidate.left, candidate.right].includes(index) ? '#e8a38b' : '#81917b'} strokeWidth="2" />
        <text x={x(index) + 26} y={value >= 0 ? y(value) - 7 : y(value) + 18} textAnchor="middle">{value}</text>
        <text x={x(index) + 26} y="213" textAnchor="middle" className="monostack-small">i={index}</text>
      </g>)}
      {candidate && candidate.height > 0 && <rect data-rectangle="candidate" x={x(candidate.start)} y={y(candidate.height)} width={candidate.width * 52} height={y(0) - y(candidate.height)} fill="#e7ba43" fillOpacity=".22" stroke="#f4c850" strokeWidth="3" strokeDasharray="6 3" />}
      {resolved !== null && <g><path d={`M${x(resolved) + 26} 229Q${(x(resolved) + x(current)) / 2 + 26} 282 ${x(current) + 26} 229`} fill="none" stroke="#edc75c" strokeWidth="2" /><text x={(x(resolved) + x(current)) / 2 + 26} y="283" textAnchor="middle" className="monostack-small">answer distance {current - resolved}</text></g>}
    </svg>
  </div>;
}
function StackView({
  stack,
  values
}) {
  return <div className="monostack-memory"><p>Stack: bottom → top</p><ol aria-label="Stack bottom to top">{stack.map(index => <li key={index}><strong>index {index}</strong><span>value {values[index]}</span></li>)}</ol>{!stack.length && <p>Empty stack</p>}</div>;
}
function ResultRow({
  caption,
  values
}) {
  return <div className="monostack-scroll" role="region" tabIndex="0" aria-label={caption}><table><caption>{caption}</caption><thead><tr><th scope="col">Index</th>{values.map((_, index) => <th scope="col" key={index}>{index}</th>)}</tr></thead><tbody><tr><th scope="row">Result</th>{values.map((value, index) => <td key={index}>{value === null ? '?' : value}</td>)}</tr></tbody></table></div>;
}
export function NextGreaterLab() {
  const [values, setValues] = useState([6, 6, 4, 7, 5, 8]);
  const [draft, setDraft] = useState('6, 6, 4, 7, 5, 8');
  const [inclusive, setInclusive] = useState(false);
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const trace = useMemo(() => nextGreaterTrace(values, inclusive), [values, inclusive]);
  const frame = trace.frames[step];
  const replace = next => {
    setValues(next);
    setDraft(next.join(', '));
    setStep(0);
    setError('');
  };
  const apply = event => {
    event.preventDefault();
    try {
      replace(parseStackValues(draft));
    } catch (problem) {
      setError(problem.message);
    }
  };
  return <section className="monostack-lab" data-monostack-lab="greater" aria-label="Next greater unresolved-index investigation">
    <h3>One new reading can answer several earlier questions</h3>
    <p>Predict which indices the next arrival will settle. A stack entry keeps its original position and value. A question mark means “not resolved yet”; zero becomes final only when the input ends.</p>
    <form onSubmit={apply}><label>Readings<input aria-label="Next-greater readings" value={draft} onChange={event => setDraft(event.target.value)} /></label><p>Up to 12 integers from −20 to 20; blank means empty. Drafts apply on the button. Invalid input preserves the active calculation.</p><button type="submit">Apply readings</button></form>
    {error && <p role="alert">{error}</p>}
    <label>Qualifying future reading<select aria-label="Future comparison" value={inclusive ? 'inclusive' : 'strict'} onChange={event => {
        setInclusive(event.target.value === 'inclusive');
        setStep(0);
      }}><option value="strict">Strictly greater</option><option value="inclusive">Greater or equal</option></select></label>
    <p>{inclusive ? 'Changed contract: an equal later reading now qualifies. The original strictly-warmer question uses the other setting.' : 'Original contract: an equal later reading does not answer the question.'}</p>
    <div className="monostack-buttons"><button onClick={() => replace([5, 5, 6])}>Equal readings</button><button onClick={() => replace([9, 8, 7, 6, 10])}>Many pops at once</button><button onClick={() => replace([])}>Empty readings</button><button onClick={() => {
        replace([6, 6, 4, 7, 5, 8]);
        setInclusive(false);
      }}>Reset future search</button></div>
    <p>Current event: <strong>{frame.phase}</strong>{frame.current !== null && ` at index ${frame.current}`}. Scroll the bars sideways if needed.</p>
    <Bars values={values} current={frame.current} resolved={frame.resolved} label="Readings by original index and resolved future distance" />
    <StackView stack={frame.stack} values={values} />
    <ResultRow caption="Distances to the first qualifying future reading" values={frame.distances} />
    <p role="status">{frame.note}</p>
    <p>Pushes {frame.pushes}; pops {frame.pops}; total {frame.pushes + frame.pops}. An arrival may do several pops, but every index can leave the stack only once.</p>
    <div className="monostack-buttons"><button disabled={!step} onClick={() => setStep(step - 1)}>Previous stack event</button><button disabled={step === trace.frames.length - 1} onClick={() => setStep(step + 1)}>Next stack event</button><button onClick={() => setStep(trace.frames.length - 1)}>Finish future search</button></div>
    <p>Frame {step + 1} of {trace.frames.length}. The illustration retains history for stepping; the algorithm needs only its current stack and output.</p>
  </section>;
}
export function HistogramBoundaryLab() {
  const [heights, setHeights] = useState([2, 2, 1, 4, 4, 3]);
  const [draft, setDraft] = useState('2, 2, 1, 4, 4, 3');
  const [direction, setDirection] = useState('left');
  const [step, setStep] = useState(0);
  const [selected, setSelected] = useState(5);
  const [error, setError] = useState('');
  const state = useMemo(() => histogramState(heights), [heights]);
  const trace = direction === 'left' ? state.leftTrace : state.rightTrace;
  const frame = trace.frames[step];
  const candidate = state.candidates[selected] ?? null;
  const replace = values => {
    setHeights(values);
    setDraft(values.join(', '));
    setSelected(Math.max(0, values.length - 1));
    setStep(0);
    setError('');
  };
  const apply = event => {
    event.preventDefault();
    try {
      replace(parseStackValues(draft, true));
    } catch (problem) {
      setError(problem.message);
    }
  };
  return <section className="monostack-lab" data-monostack-lab="histogram" aria-label="Nearest smaller boundaries and rectangle investigation">
    <h3>Find the blockers; then measure the rectangle</h3>
    <form onSubmit={apply}><label>Unit-width bar heights<input aria-label="Histogram heights" value={draft} onChange={event => setDraft(event.target.value)} /></label><p>Up to 12 nonnegative integers, each at most 20. Blank means empty. The figure uses actual height and unit-width geometry.</p><button type="submit">Apply heights</button></form>
    {error && <p role="alert">{error}</p>}
    <div className="monostack-buttons"><button onClick={() => replace([2, 2])}>Equal plateau</button><button onClick={() => replace([3, 1, 3, 3, 2])}>Tied maxima</button><button onClick={() => replace([0, 0])}>Zero heights</button><button onClick={() => replace([])}>Empty histogram</button><button onClick={() => {
        replace([2, 2, 1, 4, 4, 3]);
        setDirection('left');
      }}>Reset boundaries</button></div>
    <h4>Build one directional boundary array</h4>
    <label>Boundary direction<select aria-label="Smaller boundary direction" value={direction} onChange={event => {
        setDirection(event.target.value);
        setStep(0);
      }}><option value="left">Nearest smaller on left</option><option value="right">Nearest smaller on right</option></select></label>
    <Bars values={heights} current={frame.current} label="Histogram bars and the active directional scan" />
    <StackView stack={frame.stack} values={heights} />
    <ResultRow caption={`Computed ${direction} boundaries so far`} values={frame.boundaries} />
    <p role="status">{frame.note}</p>
    <div className="monostack-buttons"><button disabled={!step} onClick={() => setStep(step - 1)}>Previous boundary event</button><button disabled={step === trace.frames.length - 1} onClick={() => setStep(step + 1)}>Next boundary event</button><button onClick={() => setStep(trace.frames.length - 1)}>Finish boundary scan</button></div>
    <h4>Audit a completed rectangle</h4>
    <p>This separate view uses both completed scans. Equal bars can belong to the same rectangle; only strictly smaller bars block it. Scroll locally to inspect every bar.</p>
    <label>Limiting bar<select aria-label="Limiting histogram bar" disabled={!heights.length} value={selected} onChange={event => setSelected(Number(event.target.value))}>{!heights.length && <option value="0">No bar</option>}{heights.map((height, index) => <option key={index} value={index}>Index {index}: height {height}</option>)}</select></label>
    <Bars values={heights} candidate={candidate} label="Chosen histogram rectangle and its excluded smaller boundary bars" />
    {candidate && <p className="monostack-candidate">L={candidate.left}, R={candidate.right}. Include indices {candidate.start} through {candidate.end - 1}. Width = {candidate.right} − ({candidate.left}) − 1 = {candidate.width}; area = {candidate.height} × {candidate.width} = {candidate.area}.</p>}
    <ResultRow caption="Completed nearest strictly smaller left indices" values={state.leftTrace.boundaries} /><ResultRow caption="Completed nearest strictly smaller right indices" values={state.rightTrace.boundaries} />
    <p>Largest area: <strong>{state.area}</strong>. {state.best ? `One witness spans indices ${state.best.start} through ${state.best.end - 1} at height ${state.best.height}. Ties need not give a unique rectangle.` : 'No positive-area rectangle exists.'} Missing boundaries −1 and n are outside the array, not extra bars.</p>
  </section>;
}
