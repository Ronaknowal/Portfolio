import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import { useEffect, useState } from 'react';
import { NeuralLab, NeuralNumber, NeuralSelect, NeuralTable, NeuralPlot, formatNeural as f } from './NeuralLessonElements.jsx';
import { longContextDefaults, retainedRead, recurrenceTrace, latentRead } from '../../data/long-context-models.js';
import './long-context-labs.css';

export const longContextAsset = '/learn-code/long-context-sequence-models-transformer-xl-griffin-perceiver/';
export function useLongContextResource(file, open, json = false) {
  const [state, setState] = useState({ data: null, error: null }), [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setState({ data: null, error: null });
    fetch(longContextAsset + file, { signal: controller.signal }).then(response => {
      if (!response.ok) throw new Error('This saved lesson resource could not be loaded.');
      return json ? response.json() : response.text();
    }).then(data => { if (!controller.signal.aborted) setState({ data, error: null }); })
      .catch(error => { if (!controller.signal.aborted) setState({ data: null, error: error.message }); });
    return () => controller.abort();
  }, [file, open, json, attempt]);
  return { ...state, retry: () => setAttempt(value => value + 1) };
}
export function LongContextProgram({ file, title }) {
  return <>
    <RemoteCodeBlock source={longContextAsset + file} language="python" filename={file} title={(title)} />
    <p><a href={longContextAsset + file} download>Download {file}</a></p>
  </>;
}
const updateAt = (items, index, field, value) => items.map((item, i) => i === index ? { ...item, [field]: value } : item);
export function MemoryCacheLab() {
  const [records, setRecords] = useState(longContextDefaults), [segment, setSegment] = useState(2), [memory, setMemory] = useState(2), [query, setQuery] = useState(4), [beta, setBeta] = useState(0), [edit, setEdit] = useState(0), [excluded, setExcluded] = useState([]);
  const selected = Math.min(query, records.length - 1), editing = Math.min(edit, records.length - 1);
  const read = retainedRead(records, segment, memory, selected, beta, excluded);
  const all = records.map((_, i) => retainedRead(records, segment, memory, i, beta));
  const mutate = (field, value) => setRecords(previous => updateAt(previous, editing, field, value));
  const structural = action => { action(); setExcluded([]); };
  const reset = () => { setRecords(longContextDefaults()); setSegment(2); setMemory(2); setQuery(4); setBeta(0); setEdit(0); setExcluded([]); };
  return <NeuralLab id="long-context-cache" title="Which earlier records can this query still reach?">
    <p>Scalar q = 1. Move the cache boundary or change a record: the legal set, softmax denominator and answer update together. A future record stays excluded even when its storage is present.</p>
    <div className="neural-controls"><NeuralNumber label="Segment length" value={segment} min={1} max={8} step={1} integer onChange={value => structural(() => setSegment(value))} /><NeuralNumber label="Retained memory positions" value={memory} min={0} max={16} step={1} integer onChange={value => structural(() => setMemory(value))} /><NeuralNumber label="Query position (zero-based)" value={selected} min={0} max={records.length - 1} step={1} integer onChange={value => structural(() => setQuery(value))} /><NeuralNumber label="Recency penalty beta" value={beta} min={0} max={1} onChange={setBeta} /></div>
    <ol className="long-record-strip">{read.rows.map(row => <li key={row.id} className={row.status === 'legal' ? 'is-legal' : ''}><small>position {row.position} · {row.id}</small><strong>{f(row.value)}</strong><span>{row.status}</span><span>{row.status === 'legal' ? `weight ${f(row.weight)}` : 'weight 0'}</span></li>)}</ol>
    <p className="long-result">Current output <strong>{f(read.output, 8)}</strong>. Legal records: {read.legalIds.join(', ')}. Sum of exp(scores): {f(read.denominator, 8)}.</p>
    <p>Segment [{read.start}, {read.stop - 1}]; retained tail starts at position {read.first}. The selected query only uses positions ≤ {selected}. Greyed records are absent from this read, not values changed to zero.</p>
    <div className="neural-controls"><NeuralSelect label="Edit a record" value={editing} onChange={value => setEdit(Number(value))} options={records.map((r, i) => [i, `${r.id} at position ${i}`])} /><NeuralNumber label="Selected record key" value={records[editing].key} min={-4} max={4} onChange={value => mutate('key', value)} /><NeuralNumber label="Selected record value" value={records[editing].value} min={-10} max={10} onChange={value => mutate('value', value)} /></div>
    <div className="neural-buttons"><button disabled={editing === 0} onClick={() => structural(() => { setRecords(previous => { const next = [...previous]; [next[editing - 1], next[editing]] = [next[editing], next[editing - 1]]; return next; }); setEdit(editing - 1); })}>Move record earlier</button><button disabled={editing === records.length - 1} onClick={() => structural(() => { setRecords(previous => { const next = [...previous]; [next[editing], next[editing + 1]] = [next[editing + 1], next[editing]]; return next; }); setEdit(editing + 1); })}>Move record later</button><button disabled={records.length === 16} onClick={() => structural(() => setRecords(previous => [...previous, { id: `R${Math.max(...previous.map(r => Number(r.id.slice(1)))) + 1}`, key: 0, value: 0 }]))}>Add record</button><button disabled={records.length === 1} onClick={() => structural(() => setRecords(previous => previous.filter((_, i) => i !== editing)))}>Remove selected record</button><button onClick={reset}>Reset cache</button></div>
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Exact score, mask and contribution table</h4><NeuralTable caption="Current selected read; optional exclusions add a stricter mask" headers={['Record / position', 'Included', 'Score', 'Weight', 'Contribution']} rows={read.rows.map(row => [row.id + ' / ' + row.position, ['legal', 'excluded by edit'].includes(row.status) ? <input aria-label={`Include ${row.id} in selected read`} type="checkbox" checked={row.status === 'legal'} disabled={row.status === 'legal' && read.legalIds.length === 1} onChange={() => setExcluded(previous => previous.includes(row.id) ? previous.filter(id => id !== row.id) : [...previous, row.id])} /> : row.status, f(row.score), f(row.weight ?? 0), f(row.contribution ?? 0)])} /><NeuralTable caption="All query outputs under the cache rule, before optional selected-row exclusions" headers={['Query position', 'Output']} rows={all.map((result, i) => [i, f(result.output)])} /></section>
    <p>Try moving R0 just outside the retained tail, then recover it by increasing memory. For a causal null, edit a future value while inspecting an earlier query. Equal legal values always return that same value, regardless of the weights.</p>
  </NeuralLab>;
}

const impulse = () => Array.from({ length: 6 }, (_, i) => ({ x: i === 0 ? 1 : 0, input: 1, recurrence: 1 / 8 }));
function SignedAmount({ label, value, bound }) {
  return <div className="long-signed"><span>{label}</span><div className="long-signed-track"><i style={{ left: `${value < 0 ? 50 + 50 * value / bound : 50}%`, width: `${50 * Math.abs(value) / bound}%` }} /></div><output>{f(value, 7)}</output></div>;
}
export function RecurrentMemoryLab() {
  const [events, setEvents] = useState(impulse), [base, setBase] = useState(.8), [initial, setInitial] = useState(0), [step, setStep] = useState(0);
  const current = Math.min(step, events.length - 1), trace = recurrenceTrace(events, base, initial), baseline = recurrenceTrace(impulse()), state = trace[current];
  const values = [...trace.flatMap(r => [r.state, r.retained, r.injection]), ...baseline.map(r => r.state), initial, 0], bound = Math.max(.65, ...values.map(Math.abs)) * 1.05;
  const edit = (field, value) => setEvents(previous => updateAt(previous, current, field, value));
  const reset = () => { setEvents(impulse()); setBase(.8); setInitial(0); setStep(0); };
  return <NeuralLab id="long-context-recurrence" title="Preserving the old state is different from blocking new input">
    <p>Edit the selected event or gate. The final state is always visible; the event selector inspects the retained and injected terms along the way. Gate endpoints are mathematical teaching settings.</p>
    <div className="neural-controls"><NeuralNumber label="Inspect event" value={current + 1} min={1} max={events.length} step={1} integer onChange={value => setStep(value - 1)} /><NeuralNumber label="Decay base a" value={base} min={.05} max={.999} onChange={setBase} /><NeuralNumber label="Initial state" value={initial} min={-3} max={3} onChange={setInitial} /><NeuralNumber label="Event input x" value={events[current].x} min={-3} max={3} onChange={value => edit('x', value)} /><NeuralNumber label="Event input gate i" value={events[current].input} min={0} max={1} onChange={value => edit('input', value)} /><NeuralNumber label="Event recurrence gate r" value={events[current].recurrence} min={0} max={1} onChange={value => edit('recurrence', value)} /></div>
    <NeuralPlot title="State after each event" xLabel="event number" yLabel="state (feature units)" xDomain={[1, Math.max(6, events.length)]} yDomain={[Math.min(0, ...values) * 1.05, Math.max(.65, ...values) * 1.05]} series={[{ label: 'Current edited sequence', color: '#e2b55a', values: trace.map((r, i) => [i + 1, r.state]) }, { label: 'Fixed six-event impulse baseline', color: '#bcbcbc', dashed: true, values: baseline.map((r, i) => [i + 1, r.state]) }]} />
    <div className="long-contributions"><h4>Event {current + 1}: effective decay {f(state.decay, 7)}</h4><SignedAmount label="Retained" value={state.retained} bound={bound} /><SignedAmount label="Injected" value={state.injection} bound={bound} /><SignedAmount label="Their sum" value={state.state} bound={bound} /><small>Shared signed scale: −{f(bound)} to +{f(bound)}, centered at zero.</small></div>
    <p className="long-result">Final state <strong>{f(trace.at(-1).state, 9)}</strong>. Fixed baseline: 0.196608. Closing i stops a new write; changing r also changes the multiplier on existing state.</p>
    <div className="neural-buttons"><button onClick={() => setEvents(previous => previous.map((e, i) => ({ ...e, recurrence: i ? .001 : 1 / 8 })))}>Near-hold after first event</button><button onClick={() => setEvents(previous => previous.map((e, i) => ({ ...e, input: i ? 0 : 1 })))}>Close later input gates only</button><button onClick={() => setEvents(previous => previous.map(e => ({ ...e, x: 0 })))}>Set inputs to zero</button><button onClick={() => setEvents(previous => [...previous].reverse())}>Reverse events and gates</button><button disabled={events.length === 16} onClick={() => setEvents(previous => [...previous, { x: 0, input: 1, recurrence: .125 }])}>Add event</button><button disabled={events.length === 1} onClick={() => setEvents(previous => previous.filter((_, i) => i !== current))}>Remove event</button><button onClick={reset}>Reset recurrence</button></div>
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Read every update and each final contribution</h4><NeuralTable caption="Exact recurrence trace" headers={['Event', 'Input', 'i / r', 'Retained', 'Injected', 'State']} rows={trace.map((r, i) => [i + 1, f(events[i].x), `${f(events[i].input)} / ${f(events[i].recurrence)}`, f(r.retained), f(r.injection), f(r.state)])} /><NeuralTable caption="Contributions to the final state, with these gates held fixed" headers={['Source', 'Remaining contribution']} rows={[[ 'Initial state', f(trace.at(-1).initialContribution)], ...trace.at(-1).contributions.map((v, i) => [`Event ${i + 1}`, f(v)])]} /></section>
    <p>Try preserving the first event through two distractors, then replacing it deliberately. A negative input can cancel an old contribution; the state is not a probability.</p>
  </NeuralLab>;
}

const latentDefaults = () => [-1, 0, 1].map((position, i) => ({ id: `P${i}`, position, value: 2 + i * 4, comparison: i * 6 }));
export function LatentWorkspaceLab() {
  const [records, setRecords] = useState(latentDefaults), [queries, setQueries] = useState([-Math.log(2), Math.log(2)]), [edit, setEdit] = useState(0), [selected, setSelected] = useState(0);
  const point = Math.min(edit, records.length - 1), latent = Math.min(selected, queries.length - 1), output = latentRead(records, queries), comparison = latentRead(records.map(r => ({ ...r, value: r.comparison })), queries);
  const difference = Math.max(...output.map((r, i) => Math.abs(r.output - comparison[i].output)));
  const change = (field, value) => setRecords(previous => updateAt(previous, point, field, value));
  return <NeuralLab id="long-context-latent" title="Choose what the latent workspace can distinguish">
    <p>Each query reads the same position-tagged records. Ribbons encode weights, not causal importance. Array B is editable beside A; identical compressed outputs cannot be separated by a later deterministic classifier.</p>
    <div className="neural-controls"><NeuralSelect label="Edit position-tagged record" value={point} onChange={value => setEdit(Number(value))} options={records.map((r, i) => [i, r.id])} /><NeuralNumber label="Record position tag" value={records[point].position} min={-2} max={2} onChange={value => change('position', value)} /><NeuralNumber label="Array A value" value={records[point].value} min={-12} max={12} onChange={value => change('value', value)} /><NeuralNumber label="Array B comparison value" value={records[point].comparison} min={-12} max={12} onChange={value => change('comparison', value)} /><NeuralSelect label="Inspect latent" value={latent} onChange={value => setSelected(Number(value))} options={queries.map((_, i) => [i, `Latent ${i + 1}`])} /><NeuralNumber label="Latent query" value={queries[latent]} min={-3} max={3} onChange={value => setQueries(previous => previous.map((q, i) => i === latent ? value : q))} /></div>
    <div className="long-weight-read">{records.map((r, i) => <div key={r.id}><span>{r.id} · tag {f(r.position)}<br />A {f(r.value)} / B {f(r.comparison)}</span><div className="long-weight-track"><i style={{ width: `${output[latent].weights[i] * 100}%` }} /></div><span>α {f(output[latent].weights[i])}<br />A contribution {f(output[latent].contributions[i])}</span></div>)}</div>
    <div className="long-latent-outputs">{output.map((r, i) => <div key={i} className={i === latent ? 'is-active' : ''}><small>Latent {i + 1} · q {f(r.query)}</small><strong>A: {f(r.output, 8)}</strong><span>B: {f(comparison[i].output, 8)}</span></div>)}</div>
    <p className="long-result">Largest representation difference: <strong>{f(difference, 9)}</strong>. {difference < 1e-10 ? 'A and B collide within numerical tolerance (1e−10).' : 'The current query set distinguishes these two arrays.'}</p>
    <div className="neural-buttons"><button onClick={() => { setQueries([0]); setSelected(0); }}>Use one uniform query</button><button disabled={queries.length === 4} onClick={() => setQueries(previous => [...previous, 1])}>Add latent query</button><button disabled={queries.length === 1} onClick={() => setQueries(previous => previous.filter((_, i) => i !== latent))}>Remove latent</button><button onClick={() => setRecords(previous => [previous.at(-1), ...previous.slice(0, -1)])}>Rotate complete tagged records</button><button onClick={() => setRecords(previous => previous.map((r, i) => ({ ...r, value: previous[(i + previous.length - 1) % previous.length].value, comparison: previous[(i + previous.length - 1) % previous.length].comparison })))}>Reassign values to fixed tags</button><button disabled={records.length === 8} onClick={() => setRecords(previous => [...previous, { id: `P${Math.max(...previous.map(r => Number(r.id.slice(1)))) + 1}`, position: 0, value: 0, comparison: 0 }])}>Add record</button><button disabled={records.length === 1} onClick={() => setRecords(previous => previous.filter((_, i) => i !== point))}>Remove record</button><button onClick={() => { setRecords(latentDefaults()); setQueries([-Math.log(2), Math.log(2)]); setEdit(0); setSelected(0); }}>Reset latents</button></div>
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Exact weights and contributions</h4><NeuralTable caption="Selected latent read" headers={['Record', 'Weight', 'A term', 'B term']} rows={records.map((r, i) => [r.id, f(output[latent].weights[i], 8), f(output[latent].contributions[i], 8), f(comparison[latent].contributions[i], 8)])} /></section>
    <p>Try a uniform-query collision, then find a query that separates it. Moving complete tagged records preserves the representation; reassigning their values changes the input question.</p>
  </NeuralLab>;
}
