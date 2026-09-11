import { useId, useMemo, useState } from 'react';
import { chainSuccessors, linkedCycleTrace, middleSplitState } from '../../data/linked-traversal-models.js';
import './linked-traversal-labs.css';
const nodeName = index => index === null ? 'None' : `n${index}`;
function ChainDrawing({
  next,
  slow = null,
  fast = null,
  repeated = true,
  cutAfter = null,
  label
}) {
  const marker = useId().replaceAll(':', '');
  if (!next.length) return <p className="traversal-empty">head → None · no node exists</p>;
  const width = Math.max(310, next.length * 82 + 60);
  const x = index => 42 + index * 82;
  return <div className="traversal-scroll" role="region" tabIndex="0" aria-label={label}>
    <svg width={width} height="240" viewBox={`0 0 ${width} 240`} role="img" aria-label={label}>
      <defs><marker id={marker} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#d6b65e" /></marker></defs>
      {next.map((target, index) => {
        if (target === null) return cutAfter === index && index + 1 < next.length ? <g key={index}><path d={`M${x(index) + 23} 105H${x(index + 1) - 23}`} className="traversal-cut" /><text x={x(index) + 41} y="151" textAnchor="middle" className="traversal-small">cut</text></g> : <text key={index} x={x(index) + 29} y="109" className="traversal-small">∅</text>;
        const path = target === index + 1 ? `M${x(index) + 23} 105H${x(target) - 24}` : target === index ? `M${x(index) + 19} 117C${x(index) + 68} 168 ${x(index) - 65} 170 ${x(index) - 19} 121` : `M${x(index)} 129V181H${x(target)}V131`;
        return <path key={index} d={path} className="traversal-link" markerEnd={`url(#${marker})`} />;
      })}
      {next.map((_, index) => <g key={index} data-node={index}>
        <circle cx={x(index)} cy="105" r="22" fill="#171c15" stroke={index === slow || index === fast ? '#efbd51' : '#7c806e'} strokeWidth="2" />
        <text x={x(index)} y="111" textAnchor="middle">n{index}</text>
        <text x={x(index)} y="216" textAnchor="middle" className="traversal-small">value {repeated ? 7 : index + 4}</text>
        {index === slow && <text x={x(index)} y="42" textAnchor="middle" className="traversal-slow">slow</text>}
        {index === fast && <text x={x(index)} y="68" textAnchor="middle" className="traversal-fast">fast</text>}
      </g>)}
    </svg>
  </div>;
}
export function LinkedCycleLab() {
  const [length, setLength] = useState(7);
  const [entry, setEntry] = useState(2);
  const [repeated, setRepeated] = useState(true);
  const [step, setStep] = useState(0);
  const trace = useMemo(() => linkedCycleTrace(chainSuccessors(length, entry)), [length, entry]);
  const frame = trace.frames[step];
  const preset = (size, target) => {
    setLength(size);
    setEntry(target);
    setStep(0);
  };
  return <section className="traversal-lab" data-traversal-lab="cycle" aria-label="Cycle meeting and entry investigation">
    <h3>Meet inside the loop; then find its entrance</h3>
    <p>Predict the first positive meeting and the entry separately. The tail choice constructs the fixture; the algorithm receives only head. Node names are identities, not their stored values. Scroll the chain sideways when needed.</p>
    <div className="traversal-controls"><label>Number of nodes: {length}<input aria-label="Cycle node count" type="range" min="0" max="9" value={length} onChange={event => preset(Number(event.target.value), Math.min(entry, Number(event.target.value) - 1))} /></label>
      <label>Tail points to<select aria-label="Cycle tail target" value={entry} onChange={event => {
          setEntry(Number(event.target.value));
          setStep(0);
        }}><option value="-1">None: no cycle</option>{Array.from({
            length
          }, (_, index) => <option key={index} value={index}>n{index}</option>)}</select></label>
      <label className="traversal-checkbox"><input type="checkbox" checked={repeated} onChange={event => setRepeated(event.target.checked)} />All values equal 7</label></div>
    <div className="traversal-buttons"><button onClick={() => preset(0, -1)}>Empty chain</button><button onClick={() => preset(1, 0)}>Self-loop</button><button onClick={() => preset(6, -1)}>No cycle</button><button onClick={() => {
        preset(7, 2);
        setRepeated(true);
      }}>Reset cycle</button></div>
    <p className="traversal-state">Phase: <strong>{frame.phase}</strong> · detection rounds {frame.rounds} · reset-walk steps {frame.entrySteps}</p>
    <ChainDrawing next={trace.next} slow={frame.slow} fast={frame.fast} repeated={repeated} label="Successor chain and active slow and fast references" />
    <p>slow → <strong>{nodeName(frame.slow)}</strong>; fast → <strong>{nodeName(frame.fast)}</strong>. {frame.phase === 'detect' && <>The fast pointer's intermediate hop was {nodeName(frame.via)}.</>}</p>
    <p role="status">{frame.note}</p>
    <div className="traversal-buttons"><button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous cycle step</button><button disabled={step === trace.frames.length - 1} onClick={() => setStep(step + 1)}>Next cycle step</button><button onClick={() => setStep(trace.frames.length - 1)}>Show cycle result</button></div>
    <p>Frame {step + 1} of {trace.frames.length}. The drawing stores nodes and earlier frames to explain the run. The runnable entry finder keeps only a constant number of working references.</p>
  </section>;
}
export function LinkedMiddleLab() {
  const [length, setLength] = useState(6);
  const [policy, setPolicy] = useState('second');
  const [cut, setCut] = useState(false);
  const state = middleSplitState(length, policy, cut);
  const final = state.frames.at(-1);
  return <section className="traversal-lab" data-traversal-lab="middle" aria-label="Middle convention and split investigation">
    <h3>Which middle, and which link should be cut?</h3>
    <p>The query below uses the original acyclic chain. Choose what “middle” means on an even length before making a split. The optional cut always uses the separate left-heavy policy. Scroll either chain sideways when needed; its regions also accept the arrow keys.</p>
    <div className="traversal-controls"><label>Acyclic length: {length}<input aria-label="Middle chain length" type="range" min="0" max="9" value={length} onChange={event => setLength(Number(event.target.value))} /></label><label>Middle query<select aria-label="Middle policy" value={policy} onChange={event => setPolicy(event.target.value)}><option value="second">Second on even lengths</option><option value="first">First on even lengths</option></select></label></div>
    <ChainDrawing next={chainSuccessors(length, -1)} slow={final.slow} fast={final.fast} label="Original acyclic chain with final middle-query references" />
    <p role="status">{policy === 'second' ? 'Second-middle' : 'First-middle'} query returns {nodeName(state.middle)}. Pointer pairs after successive rounds: {state.frames.map(row => `(${nodeName(row.slow)}, ${nodeName(row.fast)})`).join(' → ')}.</p>
    <label className="traversal-checkbox"><input type="checkbox" checked={cut} onChange={event => setCut(event.target.checked)} />Show the left-heavy cut</label>
    {cut && <><ChainDrawing next={state.next} cutAfter={state.splitAfter} label="Two disjoint chains after the left-heavy cut" /><p>Return left head {nodeName(state.leftHead)} and right head {nodeName(state.rightHead)}: {state.leftSize} and {state.rightSize} original nodes. {length > 1 ? `Save n${state.splitAfter}.next as the right head before replacing that link with None.` : 'There is no nonempty right part.'} The selected middle query does not change this split policy.</p></>}
    <button onClick={() => {
      setLength(6);
      setPolicy('second');
      setCut(false);
    }}>Reset middle and split</button>
  </section>;
}
