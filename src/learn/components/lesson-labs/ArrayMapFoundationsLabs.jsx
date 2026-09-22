import { useId, useState } from "react";
import { Investigation, Stepper } from "./LessonInvestigation.jsx";
import { arrayMovementTrace, textModel, hashTrace } from "../../data/array-map-foundations-model";
import './systems-structures.css';
export function ArrayMovementLab() {
  const uid = useId(),
    [operation, setOperation] = useState('insert'),
    [full, setFull] = useState(false),
    [step, setStep] = useState(0);
  const trace = arrayMovementTrace(operation, full),
    s = trace[step],
    newY = s.oldCells ? 147 : 92;
  const row = (cells, y, old = false) => cells.map((value, i) => <g key={i}><rect x={26 + i * 50} y={y} width="46" height="43" fill={!old && s.to === i ? '#423116' : '#181b15'} stroke={!old && s.to === i ? '#f0c66d' : '#83977b'} /><text x={49 + i * 50} y={y + 29} textAnchor="middle" className="nt-emphasis">{value ?? '·'}</text><text x={49 + i * 50} y={y + 61} textAnchor="middle" className="nt-small">{i}</text></g>);
  return <Investigation id="array-movement" kicker="LOGICAL LENGTH ≠ ALLOCATED CAPACITY" title="Why can inserting one item move several others?">
    <p>Keep A, B and C in order. Insert X or remove B while following each slot write. Empty dots are unused capacity, not stored items.</p><p className="lesson-note">Step through insertion and removal to follow every slot write. Compare appending with spare capacity against appending to full storage.</p>
    <div className="nt-controls"><label>Sequence operation<select value={operation} onChange={e => {
          setOperation(e.target.value);
          setStep(0);
        }}><option value="front">Insert X at index 0</option><option value="insert">Insert X at index 1</option><option value="append">Append X at index 3</option><option value="delete">Delete B at index 1</option></select></label><label>Starting capacity<select value={String(full)} onChange={e => {
          setFull(e.target.value === 'true');
          setStep(0);
        }}><option value="false">4 slots · one spare</option><option value="true">3 slots · full</option></select></label></div>
    <svg className="nt-diagram foundation-diagram" viewBox={`0 0 360 ${s.oldCells ? 230 : 180}`} role="img" aria-label={`Backing cells ${s.cells.map(v => v ?? 'unused').join(', ')}; ${s.writes} element writes${s.oldCells ? '; original storage remains during copying' : ''}`}>
      <defs><marker id={uid} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#e5bc68" /></marker></defs>
      {s.oldCells && <><text x="26" y="20" className="nt-small">Old backing · still preserves the sequence</text>{row(s.oldCells, 33, true)}</>}
      <text x={s.oldCells ? 220 : 26} y={s.oldCells ? 132 : 21} className="nt-small">{s.oldCells ? 'New backing' : 'Active backing storage'}</text>{row(s.cells, newY)}
      {s.from !== null && s.to !== null && <path d={s.oldCells ? `M${49 + s.from * 50} 98 C${49 + s.from * 50} 114,${49 + s.to * 50} 119,${49 + s.to * 50} 143` : `M${49 + s.from * 50} ${newY - 4} C${49 + s.from * 50} ${newY - 55},${49 + s.to * 50} ${newY - 55},${49 + s.to * 50} ${newY - 4}`} fill="none" stroke="#e5bc68" strokeWidth="2" markerEnd={`url(#${uid})`} />}
    </svg>
    <p className="foundation-state">Logical length {s.length} · backing capacity {s.capacity} · element writes {s.writes}</p>
    <p className="nt-feedback" aria-live="polite">{s.note}</p><Stepper step={step} count={trace.length} setStep={setStep} />
    <p className="lesson-note">This model doubles a full backing array and counts copying, shifting and inserting elements. It excludes allocation/clearing costs and temporarily duplicated slots. Python lists use their own growth policy; their slots hold object references, not fixed-width copies of arbitrary Python objects.</p>
  </Investigation>;
}
export function TextUnitsLab() {
  const [kind, setKind] = useState('decomposed'),
    [normalize, setNormalize] = useState(false),
    [selected, setSelected] = useState(3);
  const m = textModel(kind, normalize),
    index = Math.min(selected, m.units.length - 1),
    u = m.units[index];
  return <Investigation id="text-units" kicker="VISIBLE TEXT · CODE POINTS · ENCODED BYTES" title="Can two identical-looking names have different lengths?">
    <p>Python strings are sequences of Unicode code points. UTF-8 encodes those points into bytes for storage or transmission. A mark you perceive as one character can contain several code points.</p><p className="lesson-note">Compare café with e plus a separate accent. Inspect code points, UTF-8 bytes, equality and normalization to distinguish appearance from representation.</p>
    <div className="nt-controls"><label>Text sample<select value={kind} onChange={e => {
          setKind(e.target.value);
          setSelected(0);
        }}><option value="ascii">cat · ASCII</option><option value="composed">café · precomposed é</option><option value="decomposed">cafe + combining acute accent</option><option value="emoji">🙂 · one emoji code point</option></select></label><label>Text operation<select value={String(normalize)} onChange={e => {
          setNormalize(e.target.value === 'true');
          setSelected(0);
        }}><option value="false">Keep original code points</option><option value="true">Create NFC-normalized string</option></select></label></div>
    <p>Original: <strong>{m.original}</strong> → current result: <strong>{m.text}</strong></p><p>{m.length} code points · {m.bytes.length} UTF-8 bytes · {m.changed ? 'a different code-point sequence' : 'same code-point sequence as input'}</p>
    <div className="foundation-units" aria-label="Select a code point">{m.units.map((unit, i) => <button type="button" aria-pressed={index === i} key={i} onClick={() => setSelected(i)}><small>index {i}</small><strong>{unit.code === 'U+0301' ? '◌́' : unit.char}</strong><small>{unit.code}</small></button>)}</div>
    <p>Encoded bytes (hexadecimal):</p><div className="foundation-units">{m.bytes.map((byte, i) => <span key={i} className="nt-chip" style={{
        borderColor: i >= u.byteOffset && i < u.byteOffset + u.bytes.length ? '#edc268' : '#61533b'
      }}>{byte.toString(16).toUpperCase().padStart(2, '0')}</span>)}</div>
    <p className="nt-feedback" aria-live="polite">Code point {u.code} at index {index} occupies byte position{u.bytes.length === 1 ? '' : 's'} {u.byteOffset}{u.bytes.length > 1 ? '–' + (u.byteOffset + u.bytes.length - 1) : ''}. {normalize ? 'NFC composes canonically equivalent sequences where possible; it does not erase accents or make all similar-looking symbols equal.' : 'Encoding changes representation, not which code points the string contains.'}</p>
    <button type="button" onClick={() => {
      setKind('decomposed');
      setNormalize(false);
      setSelected(3);
    }}>Reset text</button>
    <p className="lesson-note">The dotted circle displays an otherwise hard-to-see combining mark; it is not an extra code point in the data. This lab does not implement grapheme segmentation or language-aware matching. A substring cut at an arbitrary byte can break UTF-8; a cut at a code-point boundary can still split a displayed character.</p>
  </Investigation>;
}
export function HashBucketLab() {
  const [key, setKey] = useState(18),
    [capacity, setCapacity] = useState(4),
    [operation, setOperation] = useState('get'),
    [step, setStep] = useState(0);
  const trace = hashTrace(key, capacity, operation),
    s = trace[step];
  return <Investigation id="hash-buckets" kicker="HASH NARROWS · EQUALITY DECIDES" title="What stops a collision from returning the wrong value?">
    <p>Event IDs 10, 14 and 18 map to counts 2, 5 and 1. This teaching table chooses a bucket using ID remainder bucket-count, then compares the complete keys inside it.</p><p className="lesson-note">Change bucket count and lookup key. Follow the full-key comparisons after a collision to see why sharing a bucket does not make two keys equal.</p>
    <div className="nt-controls"><label>Event key<select value={key} onChange={e => {
          setKey(Number(e.target.value));
          setStep(0);
        }}>{[10, 14, 18, 22].map(k => <option key={k}>{k}</option>)}</select></label><label>Bucket count<select value={capacity} onChange={e => {
          setCapacity(Number(e.target.value));
          setStep(0);
        }}><option>4</option><option>5</option></select></label><label>Map operation<select value={operation} onChange={e => {
          setOperation(e.target.value);
          setStep(0);
        }}><option value="get">Look up value</option><option value="set">Set value to 99</option></select></label></div>
    {s.buckets.map((bucket, b) => <div className={'foundation-bucket ' + (b === s.bucket ? 'is-active' : '')} key={b}><strong>Bucket {b}{b === s.bucket ? ' ←' : ''}</strong><ol aria-label={`Entries in bucket ${b}`}>{bucket.length ? bucket.map((entry, i) => <li className={b === s.bucket && i === s.selected ? 'is-selected' : ''} key={entry.key}>{entry.key} → {entry.value}</li>) : <li>empty</li>}</ol></div>)}
    <p className="foundation-state">Key comparisons: {s.comparisons} · result: {s.result ?? 'not decided'}</p><p className="nt-feedback" aria-live="polite">{s.note}</p><Stepper step={step} count={trace.length} setStep={setStep} />
    <p className="lesson-note">Separate chaining with a deliberately simple hash. Python dict has a different representation and handles collisions internally. Correct equality preserves distinct keys even when their hashes collide; good distribution matters for cost, not for changing that correctness rule.</p>
  </Investigation>;
}
