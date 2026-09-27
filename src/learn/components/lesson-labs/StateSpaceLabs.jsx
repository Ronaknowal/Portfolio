import useLessonViewport from './useLessonViewport.js';
import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import { useEffect, useMemo, useRef, useState } from 'react';
import { NeuralLab, NeuralNumber, NeuralSelect, NeuralTable, NeuralPlot, formatNeural as f } from './NeuralLessonElements.jsx';
import { systemDefault, linearSystem, selectionDefault, selectiveMemory, ssdDefault, ssdOperator, trajectoryForward } from '../../data/state-space-models.js';
import './state-space-labs.css';

export const stateSpaceAsset = '/learn-code/state-space-models-s4-mamba-mamba-2/';
export const movementLabels = ['curved swing', 'horizontal swing', 'vertical swing', 'anti-clockwise arc', 'clockwise arc', 'circle', 'horizontal straight-line', 'vertical straight-line', 'horizontal zigzag', 'vertical zigzag', 'horizontal wavy', 'vertical wavy', 'face-up curve', 'face-down curve', 'tremble'];
export const stateColors = ['#e2b55a', '#eeeeee', '#b692ce', '#999999'];
const vector = values => `[${values.map(value => f(value)).join(', ')}]`;
const domain = values => {
  const low = Math.min(0, ...values), high = Math.max(0, ...values);
  const margin = Math.max(.1, .08 * (high - low));
  return [low - margin, high + margin];
};
export function StateTrace({ title, series, yLabel = 'arbitrary state units' }) {
  return <NeuralPlot title={title} xLabel="step (zero-based)" yLabel={yLabel} xDomain={[0, Math.max(1, series[0].values.length - 1)]} yDomain={domain(series.flatMap(line => line.values))} series={series.map((line, i) => ({ ...line, color: stateColors[i], values: line.values.map((value, t) => [t, value]) }))} />;
}
export function StateMatrix({ title, matrix, causal = false, selected = -1, axes = 'Row / column' }) {
  const extent = Math.max(1e-12, ...matrix.flat().map(Math.abs));
  return <div className="ssm-matrix" style={{ '--ssm-matrix-width': `${70 + matrix[0].length * 64}px` }}><NeuralTable caption={title} headers={[axes, ...matrix[0].map((_, i) => i)]} rows={matrix.map((row, i) => [i, ...row.map((value, j) => <span className={selected === i ? 'ssm-selected-cell' : ''} style={causal && j > i ? {} : { backgroundColor: value < 0 ? `rgba(190,175,204,${.12 + .42 * Math.abs(value) / extent})` : `rgba(226,181,90,${.42 * Math.abs(value) / extent})` }} key={j}>{causal && j > i ? '—' : f(value, 4)}</span>)])} /></div>;
}
function Investigation({ title, children }) {
  const [teachingSection, ready] = useLessonViewport();
  return <section className="ssm-investigation lesson-teaching-section"  data-lesson-teaching="" ref={teachingSection}><h4 className="lesson-teaching-section__title">{title}</h4>{ready && children}</section>;
}
function useStateResource(filename, open, json = false) {
  const [resource, setResource] = useState({ data: null, error: null });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!open) { setResource({ data: null, error: null }); return; }
    const controller = new AbortController();
    setResource({ data: null, error: null });
    fetch(stateSpaceAsset + filename, { signal: controller.signal }).then(response => {
      if (!response.ok) throw new Error('This saved resource could not be loaded.');
      return json ? response.json() : response.text();
    }).then(data => { if (!controller.signal.aborted) setResource({ data, error: null }); }).catch(error => { if (!controller.signal.aborted) setResource({ data: null, error: error.message }); });
    return () => controller.abort();
  }, [filename, open, json, attempt]);
  return { ...resource, retry: () => setAttempt(value => value + 1) };
}
export function StateSpaceProgram({ file, title }) {
  return <>
    <RemoteCodeBlock source={stateSpaceAsset + file} language="python" filename={file} title={(title)} />
    <p><a href={stateSpaceAsset + file} download>Download {file}</a></p>
  </>;
}

function SystemLaboratory() {
  const [system, setSystem] = useState(systemDefault), [selected, setSelected] = useState(2);
  const result = linearSystem(system), index = Math.min(selected, system.inputs.length - 1);
  const change = (key, value) => setSystem(previous => ({ ...previous, [key]: value }));
  const entry = (key, i, value) => setSystem(previous => ({ ...previous, [key]: previous[key].map((item, j) => i === j ? value : item) }));
  return <NeuralLab id="ssm-system" title="Where did this output come from?">
    <p>Exact held-input discretization, two diagonal modes. Edit an input or a system coefficient and reconcile three independently evaluated answers. All quantities use arbitrary units.</p>
    <div className="neural-controls"><NeuralNumber label="Inspect input / output step" value={index} min={0} max={system.inputs.length - 1} integer step={1} onChange={setSelected} /><NeuralNumber label={`Input u at step ${index}`} value={system.inputs[index]} min={-10} max={10} onChange={value => entry('inputs', index, value)} /><NeuralNumber label="Held interval Δ" value={system.interval} min={.05} max={2} onChange={value => change('interval', value)} /><NeuralNumber label="Direct path D" value={system.direct} min={-2} max={2} onChange={value => change('direct', value)} /></div>
    <details><summary>Edit both memory modes and initial state</summary><div className="ssm-two">{[0, 1].map(mode => <fieldset key={mode}><legend>Memory mode {mode + 1}</legend><NeuralNumber label={`Continuous rate A${mode + 1}`} value={system.rates[mode]} min={-3} max={0} onChange={value => entry('rates', mode, value)} /><NeuralNumber label={`Write B${mode + 1}`} value={system.write[mode]} min={-2} max={2} onChange={value => entry('write', mode, value)} /><NeuralNumber label={`Read C${mode + 1}`} value={system.read[mode]} min={-2} max={2} onChange={value => entry('read', mode, value)} /><NeuralNumber label={`Initial state ${mode + 1}`} value={system.initial[mode]} min={-5} max={5} onChange={value => entry('initial', mode, value)} /><p>Discrete retention {f(result.transitions[mode])}; write coefficient {f(result.injections[mode])}.</p></fieldset>)}</div></details>
    <div className="neural-buttons"><button disabled={system.inputs.length >= 16} onClick={() => change('inputs', [...system.inputs, 0])}>Append zero input</button><button disabled={system.inputs.length <= 1} onClick={() => change('inputs', system.inputs.slice(0, -1))}>Remove last input</button><button onClick={() => setSystem({ ...systemDefault(), rates: [0, -2], write: [1, 0], read: [1, 0], direct: 0, interval: .5, initial: [3, 0], inputs: [2, -1] })}>Integrator</button><button onClick={() => setSystem(previous => ({ ...previous, read: [0, 0], direct: 1 }))}>Direct path only</button><button onClick={() => setSystem(previous => ({ ...previous, inputs: previous.inputs.map(() => 0), initial: [0, 0] }))}>Zero input and state</button><button onClick={() => { setSystem(systemDefault()); setSelected(2); }}>Reset system</button></div>
    <div className="ssm-two"><StateTrace title="Input and actual output" series={[{ label: 'Input u', values: system.inputs }, { label: 'Output y', values: result.recurrent }]} /><StateTrace title="Two retained state coordinates" series={[{ label: 'State 1', values: result.states.map(row => row[0]) }, { label: 'State 2', values: result.states.map(row => row[1]) }]} /></div>
    <p className="ssm-result" data-result="system">Step {index}: memory read {f(result.memory[index])} + direct input {f(system.direct * system.inputs[index])} = <strong>{f(result.recurrent[index])}</strong>. Maximum recurrent / direct / FFT difference: <strong>{result.difference.toExponential(2)}</strong>.</p>
    <StateTrace title="Kernel: what survives at each lag" series={[{ label: 'K at lag', values: result.kernel }]} yLabel="output per unit input" />
    <StateMatrix title="Input contribution ledger: birth row → output column" matrix={result.contributions} selected={index} />
    <NeuralTable caption="Full reconciliation; initial state and direct path counted once" headers={['Step', 'Initial response', 'Recurrent', 'Direct convolution', 'FFT convolution']} rows={system.inputs.map((_, t) => [t, f(result.initialResponse[t]), f(result.recurrent[t]), f(result.directConvolution[t]), f(result.frequencyConvolution[t])])} />
    <p>The column sum of the contribution ledger, plus the initial response and Du, gives the output. A final-input edit has no path to earlier columns. Changing the initial state can affect every output.</p>
  </NeuralLab>;
}
export function StateSpaceSystemLab() { return <Investigation title="Open the system and impulse laboratory"><SystemLaboratory /></Investigation>; }

function SelectionLaboratory() {
  const [settings, setSettings] = useState(selectionDefault), [selected, setSelected] = useState(2);
  const trace = selectiveMemory(settings), index = Math.min(selected, trace.length - 1), step = trace[index];
  const change = (key, value) => setSettings(previous => ({ ...previous, [key]: value }));
  const entry = (key, value) => setSettings(previous => ({ ...previous, [key]: previous[key].map((item, i) => i === index ? value : item) }));
  return <NeuralLab id="ssm-selection" title="Retain a marked value through distractions">
    <p>Markers define the target: the latest marked input. You control the gate schedule; a learned model would have to infer useful coefficients from its features.</p>
    <div className="ssm-event-strip">{trace.map((row, i) => <button key={i} className={i === index ? 'is-selected' : ''} onClick={() => setSelected(i)} aria-label={`Select event ${i}`}><small>{i} · {settings.marked[i] ? 'marked' : 'distractor'}</small><strong>{f(row.input)}</strong><span>g {f(settings.gates[i])}</span></button>)}</div>
    <div className="neural-controls"><NeuralNumber label="Selected event" value={index} min={0} max={trace.length - 1} step={1} integer onChange={setSelected} /><NeuralNumber label="Selected event value" value={settings.inputs[index]} min={-10} max={10} onChange={value => entry('inputs', value)} /><NeuralNumber label="Selected write gate g" value={settings.gates[index]} min={0} max={1} onChange={value => entry('gates', value)} /><NeuralNumber label="Constant comparison gate" value={settings.constant} min={0} max={1} onChange={value => change('constant', value)} /><NeuralNumber label="Selection initial state" value={settings.initial} min={-5} max={5} onChange={value => change('initial', value)} /></div>
    <div className="neural-buttons"><button aria-pressed={settings.marked[index]} onClick={() => entry('marked', !settings.marked[index])}>{settings.marked[index] ? 'Unmark selected event' : 'Mark selected event'}</button><button onClick={() => change('gates', settings.marked.map(marked => marked ? .99 : .01))}>Set gates from markers</button><button aria-pressed={settings.independent} onClick={() => change('independent', !settings.independent)}>{settings.independent ? 'Use coupled retention 1−g' : 'Separate retention from write'}</button></div>
    {settings.independent && <NeuralNumber label="Independent retention a" value={settings.retention} min={0} max={1} onChange={value => change('retention', value)} />}
    {settings.independent ? <div className="ssm-independent-bands"><p>Independent retention a = {f(settings.retention)}</p><div><span style={{ width: `${100 * settings.retention}%` }} /></div><p>Write g = {f(settings.gates[index])}</p><div><span style={{ width: `${100 * settings.gates[index]}%` }} /></div></div> : <><div className="ssm-bands"><div style={{ flexGrow: 1 - settings.gates[index] }} /><div style={{ flexGrow: settings.gates[index] }} /></div><p>Gray retention 1−g = {f(1 - settings.gates[index])}; amber write g = {f(settings.gates[index])}. Their widths sum to one.</p></>}
    <p>{settings.independent ? 'Each independent bar spans 0…1. The two coefficients need not sum to one.' : 'These bands are coefficients, not probabilities of successful recall.'}</p>
    <StateTrace title="Actual state after each event" series={[{ label: 'Selected gate schedule', values: trace.map(row => row.state) }, { label: 'Constant gate', values: trace.map(row => row.fixed) }]} />
    <p className="ssm-result" data-result="selection">At event {index}: retained {f(step.retained)} + new write {f(step.incoming)} = <strong>{f(step.state)}</strong>. {step.target === null ? 'No marked target exists yet.' : <>Latest marked value {f(step.target)}; absolute error {f(step.error)} versus constant-gate error {f(step.fixedError)}.</>}</p>
    <NeuralTable caption="Each update, including signs" headers={['Event', 'Retained', 'Written', 'State', 'Constant state', 'Target']} rows={trace.map((row, i) => [i, f(row.retained), f(row.incoming), f(row.state), f(row.fixed), row.target === null ? 'none' : f(row.target)])} />
    <div className="neural-buttons"><button disabled={trace.length >= 16} onClick={() => setSettings(previous => ({ ...previous, inputs: [...previous.inputs, 0], marked: [...previous.marked, false], gates: [...previous.gates, .01] }))}>Add distraction</button><button disabled={trace.length <= 2} onClick={() => setSettings(previous => ({ ...previous, inputs: previous.inputs.slice(0, -1), marked: previous.marked.slice(0, -1), gates: previous.gates.slice(0, -1) }))}>Remove last event</button><button onClick={() => setSettings(previous => ({ ...previous, inputs: previous.inputs.map(() => 0), initial: 0 }))}>Zero signal</button><button onClick={() => setSettings(previous => ({ ...previous, gates: previous.gates.map(() => 0), initial: 4 }))}>Close writes; start at 4</button><button onClick={() => { setSettings(selectionDefault()); setSelected(2); }}>Reset selective memory</button></div>
  </NeuralLab>;
}
export function StateSpaceSelectionLab() { return <Investigation title="Open the selective memory challenge"><SelectionLaboratory /></Investigation>; }

function SSDLaboratory() {
  const [settings, setSettings] = useState(ssdDefault), [selected, setSelected] = useState(2);
  const index = Math.min(selected, settings.decay.length - 1), result = ssdOperator(settings);
  const change = (key, value) => setSettings(previous => ({ ...previous, [key]: value }));
  const entry = (key, coordinate, value) => setSettings(previous => ({ ...previous, [key]: previous[key].map((row, i) => i === index ? row.map((item, j) => j === coordinate ? value : item) : row) }));
  return <NeuralLab id="ssm-ssd" title="Change the partition, preserve the operator">
    <div className="neural-controls"><NeuralNumber label="SSD selected position" value={index} min={0} max={settings.decay.length - 1} integer step={1} onChange={setSelected} /><NeuralNumber label="Chunk size q" value={settings.chunkSize} min={1} max={8} integer step={1} onChange={value => change('chunkSize', value)} /><NeuralNumber label="Selected decay a" value={settings.decay[index]} min={0} max={1} onChange={value => change('decay', settings.decay.map((item, i) => i === index ? value : item))} /></div>
    <div className="ssm-three">{[['write', 'Write b'], ['read', 'Read c'], ['values', 'Value v']].map(([key, label]) => <fieldset key={key}><legend>{label} at position {index}</legend>{[0, 1].map(coordinate => <NeuralNumber key={coordinate} label={`${label} coordinate ${coordinate + 1}`} value={settings[key][index][coordinate]} min={-8} max={8} onChange={value => entry(key, coordinate, value)} />)}</fieldset>)}</div>
    <details><summary>Edit the initial 2 × 2 matrix state</summary><div className="neural-controls">{settings.initial.flatMap((row, i) => row.map((value, j) => <NeuralNumber key={`${i}-${j}`} label={`Initial S row ${i} column ${j}`} value={value} min={-5} max={5} onChange={next => change('initial', settings.initial.map((other, k) => k === i ? other.map((item, l) => l === j ? next : item) : other))} />))}</div></details>
    <p className="ssm-result" data-result="ssd">Output {index}: <strong>{vector(result.recurrent[index])}</strong>. Maximum recurrent / matrix / chunk difference: <strong>{result.difference.toExponential(2)}</strong>. The chunk size changes grouping; its output should remain equal within 10⁻¹⁰.</p>
    <section open data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Separate the content factor, decay mask and their product</h4><StateMatrix title="Content factor CBᵀ (signed)" matrix={result.content} selected={index} /><StateMatrix title="Decay mask L (future entries absent)" matrix={result.mask} causal selected={index} /><StateMatrix title="Influence M = CBᵀ ⊙ L" matrix={result.influence} causal selected={index} /><p>Amber = positive; pale violet = negative; neutral = zero. Exact signs and values are printed. These weights are unnormalized coefficients, not probabilities.</p></section>
    <NeuralTable caption={`Contributions of writes to selected output ${index}; initial state is additional`} headers={['Write position', 'Influence', 'Value vector', 'Contribution']} rows={result.influence[index].map((coefficient, j) => [j, f(coefficient), vector(settings.values[j]), vector(settings.values[j].map(value => coefficient * value))])} />
    <div className="ssm-chunks">{result.chunks.map(chunk => <section key={chunk.start}><h4>Chunk {chunk.start}–{chunk.stop - 1}</h4><p>Boundary decay {f(chunk.totalDecay)}. Incoming state {chunk.carry.map(vector).join('; ')}.</p><NeuralTable caption="Local + incoming = output" headers={['Step', 'Local', 'Incoming', 'Sum']} rows={chunk.local.map((row, i) => [chunk.start + i, vector(row), vector(chunk.incoming[i]), vector(chunk.total[i])])} /><p>Own final write {chunk.ownFinal.map(vector).join('; ')}. Pass boundary decay × incoming state + own write onward.</p></section>)}</div>
    <NeuralTable caption="Independent evaluation comparison" headers={['Step', 'Recurrent', 'Matrix', 'Chunked']} rows={result.recurrent.map((row, i) => [i, vector(row), vector(result.matrix[i]), vector(result.chunked[i])])} />
    <div className="neural-buttons"><button disabled={settings.decay.length >= 8} onClick={() => setSettings(previous => ({ ...previous, decay: [...previous.decay, .5], write: [...previous.write, [1, 0]], read: [...previous.read, [1, 0]], values: [...previous.values, [0, 0]] }))}>Append position</button><button disabled={settings.decay.length <= 1} onClick={() => setSettings(previous => ({ ...previous, decay: previous.decay.slice(0, -1), write: previous.write.slice(0, -1), read: previous.read.slice(0, -1), values: previous.values.slice(0, -1) }))}>Remove final position</button><button onClick={() => change('write', settings.write.map(() => [0, 0]))}>Zero writes</button><button onClick={() => change('decay', settings.decay.map((value, i) => i === index ? 0 : value))}>Erase incoming state here</button><button onClick={() => { setSettings(ssdDefault()); setSelected(2); }}>Reset SSD workshop</button></div>
  </NeuralLab>;
}
export function StateSpaceSSDLab() { return <Investigation title="Open the SSD matrix and chunk workshop"><SSDLaboratory /></Investigation>; }

export function TrajectoryPlot({ title, points, baseline, selected = -1, onChange }) {
  const drag = useRef(null);
  const position = (svg, event) => {
    const coordinates = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    return [(coordinates.x - 35) / 250, (285 - coordinates.y) / 250];
  };
  const start = event => {
    if (!onChange || selected < 0) return;
    const svg = event.currentTarget.ownerSVGElement;
    const pointer = position(svg, event);
    drag.current = { offset: [points[selected][0] - pointer[0], points[selected][1] - pointer[1]], pointerId: event.pointerId };
    svg.setPointerCapture(event.pointerId);
    event.preventDefault();
  };
  const move = event => {
    if (!onChange || !drag.current || drag.current.pointerId !== event.pointerId) return;
    const pointer = position(event.currentTarget, event);
    onChange(pointer.map((value, axis) => Math.max(0, Math.min(1, value + drag.current.offset[axis]))));
  };
  const stop = event => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    drag.current = null;
  };
  return <figure className="ssm-path"><figcaption>{title}</figcaption><svg viewBox="0 0 320 320" role="img" aria-label={`${title}. Equal-scale x and y coordinates, 0 to 1. Start circle and end square show traversal order.`} onPointerMove={move} onPointerUp={stop} onPointerCancel={stop} onLostPointerCapture={() => { drag.current = null; }}>
    <rect x="35" y="35" width="250" height="250" fill="#0a0a0a" stroke="#555" />
    {[0, .5, 1].map(value => <g key={value}><text x={35 + value * 250} y="305" textAnchor="middle">{value}</text><text x="26" y={289 - value * 250} textAnchor="end">{value}</text></g>)}
    {baseline && <polyline points={baseline.map(([x, y]) => `${35 + x * 250},${285 - y * 250}`).join(' ')} fill="none" stroke="#777" strokeDasharray="5 4" strokeWidth="2" />}
    <polyline points={points.map(([x, y]) => `${35 + x * 250},${285 - y * 250}`).join(' ')} fill="none" stroke="#e2b55a" strokeWidth="2" />
    {points.map(([x, y], i) => i % 10 === 0 || i === selected ? <g key={i}><circle cx={35 + x * 250} cy={285 - y * 250} r={i === selected ? 8 : 4} fill={i === selected ? '#e2b55a' : '#ddd'} stroke="#050505" strokeWidth="2" /><title>Point {i + 1}: ({f(x)}, {f(y)})</title></g> : null)}
    <circle cx={35 + points[0][0] * 250} cy={285 - points[0][1] * 250} r="6" fill="#fff" stroke="#050505" strokeWidth="2" />
    <rect x={31 + points.at(-1)[0] * 250} y={281 - points.at(-1)[1] * 250} width="8" height="8" fill="#e2b55a" stroke="#fff" />
    {onChange && selected >= 0 && <circle data-ssm-drag-handle="true" cx={35 + points[selected][0] * 250} cy={285 - points[selected][1] * 250} r="22" fill="transparent" stroke="#e2b55a" strokeDasharray="3 4" onPointerDown={start} style={{ touchAction: 'none', cursor: 'grab' }}><title>Drag selected point {selected + 1}</title></circle>}
  </svg><p>Horizontal x; vertical y. White circle = start, amber square = end. {onChange && 'Drag the dashed ring around the selected point; the numeric controls are equivalent.'}</p></figure>;
}
function FittedTrajectory({ data }) {
  const [recordId, setRecordId] = useState(7), [kind, setKind] = useState('diagonal'), [edited, setEdited] = useState(null), [selected, setSelected] = useState(22), [channel, setChannel] = useState(0);
  const record = data.records.find(row => row.id === recordId), points = edited || record.points;
  const baseline = useMemo(() => trajectoryForward(record.points, data.models[kind], kind), [record, data, kind]);
  const current = useMemo(() => trajectoryForward(points, data.models[kind], kind), [points, data, kind]);
  const coordinate = value => setEdited(points.map((point, i) => i === selected ? value : point));
  return <NeuralLab id="ssm-trajectory" title="Move a point; watch the fitted state and class probabilities change">
    <p>Retained seed-17 weights; no browser training. Only the 50 validation records are selectable. Original source label: <strong>{record.label} · {movementLabels[record.label - 1]}</strong>. An edited trajectory is hypothetical and has no newly certified label.</p>
    <div className="neural-controls"><NeuralSelect label="Validation source row" value={recordId} options={data.records.map(row => [row.id, `${row.id} · ${movementLabels[row.label - 1]}`])} onChange={value => { setRecordId(Number(value)); setEdited(null); }} /><NeuralSelect label="Frozen temporal mixer" value={kind} options={['diagonal', 'selective'].map(value => [value, `${value} · seed 17`])} onChange={setKind} /><NeuralNumber label="Trajectory point (one-based)" value={selected + 1} min={1} max={45} integer step={1} onChange={value => setSelected(value - 1)} /></div>
    <div className="ssm-two"><TrajectoryPlot title={`Source row ${recordId} · current solid / original dashed`} points={points} baseline={record.points} selected={selected} onChange={coordinate} /><div><NeuralNumber label="Selected point x" value={points[selected][0]} min={0} max={1} onChange={value => coordinate([value, points[selected][1]])} /><NeuralNumber label="Selected point y" value={points[selected][1]} min={0} max={1} onChange={value => coordinate([points[selected][0], value])} /><div className="neural-buttons"><button onClick={() => coordinate([1 - points[selected][0], points[selected][1]])}>Reflect selected x</button><button onClick={() => setEdited([...points].reverse())}>Reverse traversal order</button><button onClick={() => setEdited(null)}>Reset path</button></div></div></div>
    <p className="ssm-result" data-result="trajectory">Original top class: {baseline.top} ({movementLabels[baseline.top - 1]}). Current top class: <strong>{current.top} ({movementLabels[current.top - 1]})</strong>. Probability of original source label: {f(baseline.probabilities[record.label - 1], 7)} → <strong>{f(current.probabilities[record.label - 1], 7)}</strong>.</p>
    <NeuralNumber label="Inspect mixer channel" value={channel} min={0} max={15} integer step={1} onChange={setChannel} /><StateTrace title={`First temporal mixer · channel ${channel}`} series={[{ label: 'Original', values: baseline.traces[0].map(row => row[channel]) }, { label: 'Edited', values: current.traces[0].map(row => row[channel]) }]} yLabel="internal response (not importance)" />
    <div className="ssm-probabilities">{movementLabels.map((label, i) => <div key={label}><span>{i + 1}. {label}</span><div><span className="ssm-prob-original" style={{ width: `${100 * baseline.probabilities[i]}%` }} /><span className="ssm-prob-current" style={{ width: `${100 * current.probabilities[i]}%` }} /></div><small>{f(current.probabilities[i], 5)} ({f(current.probabilities[i] - baseline.probabilities[i], 5)} change)</small></div>)}</div><p>Full-width bars span probability 0…1. Pale = original, amber = current. Classes stay in the original fixed order.</p>
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Exact logits, probabilities and trajectory coordinates</h4><NeuralTable caption="Whole-record output" headers={['Class', 'Original probability', 'Current probability', 'Current logit']} rows={movementLabels.map((label, i) => [`${i + 1}. ${label}`, f(baseline.probabilities[i], 8), f(current.probabilities[i], 8), f(current.logits[i], 8)])} /><NeuralTable caption="Current coordinates" headers={['Point', 'x', 'y']} rows={points.map((point, i) => [i + 1, ...point.map(value => f(value, 8))])} /></section>
    <p>Changing a late point can change the final mean-pooled class while earlier causal mixer responses remain fixed. The internal response is an inspection view, not a causal attribution score.</p>
  </NeuralLab>;
}
export function StateSpaceTrajectoryLab() {
  const [teachingSection, ready] = useLessonViewport(), resource = useStateResource('trajectory-inference.json', ready, true);
  return <section className="ssm-investigation lesson-teaching-section"  data-lesson-teaching="" ref={teachingSection}><h4 className="lesson-teaching-section__title">Fitted trajectory investigation</h4>{ready && (resource.error ? <p role="alert">{resource.error} <button onClick={resource.retry}>Retry fitted models</button></p> : resource.data ? <FittedTrajectory data={resource.data} /> : <p role="status">Loading two small retained models and validation trajectories…</p>)}</section>;
}
