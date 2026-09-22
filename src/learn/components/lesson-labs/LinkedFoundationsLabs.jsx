import { useId, useState } from "react";
import { Investigation, Stepper } from "./LessonInvestigation.jsx";
import { reverseTrace, bracketTrace, ringTrace } from "../../data/linked-foundations-model";
import './systems-structures.css';
import { QueueRingFigure } from "./SystemsMechanismFigures";
export function LinkedReversalLab() {
  const uid = useId(),
    [size, setSize] = useState(3),
    [broken, setBroken] = useState(false),
    [step, setStep] = useState(0),
    trace = reverseTrace(size, broken),
    s = trace[step];
  const x = id => 52 + ['A', 'B', 'C'].indexOf(id) * 126;
  return <Investigation id="linked-reversal" kicker="SAVE THE ROUTE BEFORE REWIRING IT" title="How do you reverse links without losing the rest of the list?">
    <p>Node identities A, B and C remain fixed on the page; their arrows determine traversal order. A reference stores which node to reach, not the node's screen position.</p><p className="lesson-note">Compare the correct and faulty pointer-update orders. Watch the saved successor to see which links remain reachable after reversal.</p>
    <div className="nt-controls"><label>List length<select value={size} onChange={e => {
          setSize(Number(e.target.value));
          setStep(0);
        }}><option value="3">Three nodes</option><option value="1">One node</option><option value="0">Empty list</option></select></label><label>Rewiring order<select value={String(broken)} onChange={e => {
          setBroken(e.target.value === 'true');
          setStep(0);
        }}><option value="false">Save successor before overwriting next</option><option value="true">Bug: overwrite next before saving it</option></select></label></div>
    <div className="foundation-reference">{['head', 'previous', 'current', 'saved'].map(name => <span key={name}>{name} → <strong>{s[name] ?? '∅'}</strong></span>)}</div>
    <svg className="nt-diagram foundation-diagram" viewBox="0 0 360 210" role="img" aria-label={s.nodes.length ? s.nodes.map(n => `${n.id} value ${n.value} points to ${n.next ?? 'empty'}`).join('; ') : 'Empty list: no nodes'}>
      <defs><marker id={uid} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#dabb79" /></marker></defs>
      {s.nodes.filter(n => n.next !== null).map(n => {
        const direction = x(n.next) > x(n.id) ? 1 : -1;
        return <path key={n.id} d={`M${x(n.id) + direction * 29} 87 Q${(x(n.id) + x(n.next)) / 2} ${direction === 1 ? 42 : 137},${x(n.next) - direction * 31} 87`} fill="none" stroke="#dabb79" strokeWidth="2" markerEnd={`url(#${uid})`} />;
      })}
      {s.nodes.map(n => <g key={n.id}><circle cx={x(n.id)} cy="87" r="27" fill={s.current === n.id ? '#3e3017' : '#19221a'} stroke={s.current === n.id ? '#e9bf63' : '#8eac92'} strokeWidth="2" /><text x={x(n.id)} y="93" textAnchor="middle" className="nt-emphasis">{n.id}</text><text x={x(n.id)} y="148" textAnchor="middle">value {n.value}</text><text x={x(n.id)} y="177" textAnchor="middle" className="nt-small">next: {n.next ?? '∅'}</text></g>)}
      {!s.nodes.length && <text x="80" y="96">head → ∅ · no nodes to visit</text>}
    </svg>
    <p className="foundation-state">{s.phase}</p><p className="nt-feedback" aria-live="polite">{s.note}</p><Stepper step={step} count={trace.length} setStep={setStep} />
    <p className="lesson-note">The diagram keeps all original nodes visible to expose lost reachability. In a garbage-collected runtime, unreachable objects can later be collected; a low-level language may leak them. Returning the new head is separate from updating the caller's head variable.</p>
  </Investigation>;
}
export function BracketStackLab() {
  const [text, setText] = useState('([])'),
    [step, setStep] = useState(0),
    trace = bracketTrace(text),
    s = trace[step];
  return <Investigation id="bracket-stack" kicker="MOST RECENT UNMATCHED OPENER CLOSES FIRST" title="Why are equal bracket counts not enough?">
    <p>A stack keeps unmatched openers in the order they must close. The next closer must match the top; it cannot jump past a newer opener to find a convenient match below it.</p><p className="lesson-note">Step through properly and improperly nested brackets. Follow the stack top and the exact character that conflicts with it.</p>
    <div className="nt-controls"><label>Bracket input<select value={text} onChange={e => {
          setText(e.target.value);
          setStep(0);
        }}>{['([])', '([)]', ')', '(()', ''].map(v => <option key={v} value={v}>{v || '(empty input)'}</option>)}</select></label></div>
    <div className="foundation-units" aria-label="Input and current position">{Array.from(text).map((char, i) => <span className="nt-chip" style={{
        background: i === s.index ? '#423319' : 'transparent'
      }} key={i}>{char}<small> {i}</small>{i === s.index ? ' ←' : ''}</span>)}{!text && <span>empty input</span>}</div>
    <h4>Unmatched stack · top is at the top</h4><div className="foundation-stack">{s.stack.length ? s.stack.map((entry, i) => <span key={entry.index}>{entry.char}<small>input {entry.index}{i === s.stack.length - 1 ? ' · TOP' : ''}</small></span>) : <span>empty stack</span>}</div>
    <p className="foundation-rule">Invariant: after a valid processed prefix, the stack contains exactly its unmatched opening brackets, oldest at the bottom and newest at the top.</p>
    <p className="foundation-state">{s.result === null ? 'Still scanning' : s.result ? 'Accept: nested and closed' : 'Reject: mismatch or missing partner'}</p><p className="nt-feedback" aria-live="polite">{s.note}</p><Stepper step={step} count={trace.length} setStep={setStep} />
    <p className="lesson-note">This recognizes bracket nesting only. Real source code needs tokenization so brackets inside strings or comments are interpreted correctly.</p>
  </Investigation>;
}
export function CircularQueueLab() {
  const [capacity, setCapacity] = useState(4),
    [step, setStep] = useState(0),
    trace = ringTrace(capacity),
    s = trace[step];
  return <Investigation id="circular-queue" kicker="PHYSICAL SLOTS WRAP · LOGICAL ORDER DOES NOT" title="How can a queue reuse the front without shifting every item?">
    <p>Jobs arrive as A, B, C, D, E and F. Head identifies the oldest stored job; size counts occupied slots. The next insertion position is (head + size) remainder capacity.</p><p className="lesson-note">Fill, remove from and wrap the queue. Follow head, tail and occupied slots to distinguish physical position from logical order.</p>
    <div className="nt-controls"><label>Buffer capacity<select value={capacity} onChange={e => {
          setCapacity(Number(e.target.value));
          setStep(0);
        }}><option value="3">3 jobs</option><option value="4">4 jobs</option></select></label></div>
    <p className="foundation-state">{s.event ?? 'Initialize'} · head {s.head} · size {s.size} · next insertion slot {s.tail}</p>
    <QueueRingFigure state={s} capacity={capacity} />
    <details><summary>Inspect exact slot contents</summary><div className="foundation-buffer" style={{
        gridTemplateColumns: `repeat(${capacity},minmax(0,1fr))`
      }}>{s.cells.map((value, i) => <div key={i} className={(i === s.head ? 'is-head ' : '') + (i === s.changed ? 'is-changed' : '')}><small>slot {i}</small><strong>{value ?? '·'}</strong><small>{[i === s.head ? 'HEAD' : '', i === s.tail ? 'NEXT' : ''].filter(Boolean).join(' + ') || '—'}</small></div>)}</div></details>
    <p>After slot {capacity - 1} → wrap to slot 0. A full buffer has no writable next slot until a dequeue frees capacity.</p>
    <p><strong>Logical FIFO order:</strong> {s.logical.join(' → ') || 'empty'}</p><p><strong>Already served:</strong> {s.output.join(' → ') || 'none'}</p>
    <p className="nt-feedback" aria-live="polite">{s.note}</p><Stepper step={step} count={trace.length} setStep={setStep} />
    <p className="lesson-note">Fixed capacity, one producer/consumer acting in sequence, explicit rejection when full. It is not a thread-safe queue. Head and next can coincide when either empty or full; size disambiguates them.</p>
  </Investigation>;
}
