import { useId, useMemo, useRef, useState } from 'react';
import { NeuralLab, NeuralNumber, NeuralSelect, NeuralTable, NeuralPlot, formatNeural as f } from './NeuralLessonElements.jsx';
import { trajectoryForward, meanTrajectory } from '../../data/long-context-models.js';
import { longContextAsset, useLongContextResource } from './LongContextLabs.jsx';

const names = ['curved swing', 'horizontal swing', 'vertical swing', 'anti-clockwise arc', 'clockwise arc', 'circle', 'horizontal straight-line', 'vertical straight-line', 'horizontal zigzag', 'vertical zigzag', 'horizontal wavy', 'vertical wavy', 'face-up curve', 'face-down curve', 'tremble'];
const copyRecords = source => source.map(record => ({ ...record }));
function TrajectoryWorkbench({ data }) {
  const initial = data.specimens.find(item => item.sourceRow === 7);
  const [sourceRow, setSourceRow] = useState(7), [model, setModel] = useState('latents4_seed29'), [records, setRecords] = useState(() => copyRecords(initial.records)), [pointId, setPointId] = useState(1), [latent, setLatent] = useState(0), [message, setMessage] = useState('');
  const grab = useRef(null);
  const arrowId = useId();
  const specimen = data.specimens.find(item => item.sourceRow === sourceRow), parameters = data.models[model];
  const current = useMemo(() => trajectoryForward(parameters, records), [parameters, records]);
  const original = useMemo(() => trajectoryForward(parameters, specimen.records), [parameters, specimen]);
  const mean = useMemo(() => meanTrajectory(data.mean, records), [data.mean, records]);
  const index = records.findIndex(r => r.id === pointId), point = records[index], selectedLatent = Math.min(latent, parameters.latent.length - 1);
  const byTime = [...records].sort((a, b) => a.position - b.position), validCount = records.filter(r => r.valid).length;
  const updatePoint = (field, value) => { setRecords(previous => previous.map(r => r.id === pointId ? { ...r, [field]: value } : r)); setMessage(''); };
  const reset = () => { setSourceRow(7); setRecords(copyRecords(initial.records)); setModel('latents4_seed29'); setPointId(1); setLatent(0); setMessage(''); };
  const coordinate = value => 36 + 248 * value;
  const beginDrag = (event, record) => {
    const svg = event.currentTarget.ownerSVGElement, matrix = svg.getScreenCTM();
    if (!matrix) return;
    const cursor = svg.createSVGPoint(); cursor.x = event.clientX; cursor.y = event.clientY;
    const local = cursor.matrixTransform(matrix.inverse());
    grab.current = { x: local.x - coordinate(record.x), y: local.y - coordinate(1-record.y) };
    event.currentTarget.setPointerCapture(event.pointerId); setPointId(record.id);
  };
  const endDrag = event => {
    grab.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };
  const drag = event => {
    if (!grab.current || event.buttons !== 1) return;
    const svg = event.currentTarget.ownerSVGElement, matrix = svg.getScreenCTM();
    if (!matrix) return;
    const pointer = svg.createSVGPoint(); pointer.x = event.clientX; pointer.y = event.clientY;
    const local = pointer.matrixTransform(matrix.inverse());
    const x = (local.x - grab.current.x - 36) / 248, y = 1 - (local.y - grab.current.y - 36) / 248;
    // Keep the previous valid coordinate when dragged beyond the declared domain.
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    const id = Number(event.currentTarget.dataset.point);
    setPointId(id); setRecords(previous => previous.map(r => r.id === id ? { ...r, x, y } : r)); setMessage('');
  };
  return <>
    <div className="neural-controls long-model-controls"><NeuralSelect label="Validation trajectory" value={sourceRow} onChange={value => { const next = data.specimens.find(item => item.sourceRow === Number(value)); setSourceRow(next.sourceRow); setRecords(copyRecords(next.records)); setPointId(1); setMessage(''); }} options={data.specimens.map(item => [item.sourceRow, `Source row ${item.sourceRow} · class ${item.label}`])} /><NeuralSelect label="Frozen latent model" value={model} onChange={value => { setModel(value); setLatent(0); }} options={Object.keys(data.models).map(key => [key, key.replace('latents', 'Latents ').replace('_seed', ', seed ')])} /></div>
    <p>Original label: {specimen.label}, {names[specimen.label - 1]}. {validCount}/45 measured points retained. Edited geometry has no automatically certified new label.</p>
    <p className="long-result">Current model class <strong>{current.predictedClass}: {names[current.predictedClass - 1]}</strong>. P(class 1) {f(current.probabilities[0], 9)}; change from this source's full original input {f(current.probabilities[0] - original.probabilities[0], 9)}. Mean-coordinate baseline class {mean.predictedClass}.</p>
    <div className="long-trajectory-grid"><figure className="long-trajectory"><figcaption>Measured path in unit coordinates · equal x/y scale</figcaption><svg viewBox="0 0 320 320" role="img" aria-label="Editable hand path. Select a point and edit x and y using the numeric controls; dragging is also available.">
      <rect x="36" y="36" width="248" height="248" fill="none" stroke="#626262" />
      {[0, .5, 1].map(t => <g key={t}><text x={coordinate(t)} y="306" textAnchor="middle">{t}</text><text x="26" y={coordinate(1 - t) + 4} textAnchor="end">{t}</text></g>)}
      <defs><marker id={arrowId} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#eee" /></marker></defs>
      <polyline fill="none" stroke="#9a9a9a" strokeWidth="1.5" strokeDasharray="3 2" markerEnd={`url(#${arrowId})`} points={byTime.map(r => `${coordinate(r.x)},${coordinate(1-r.y)}`).join(' ')} />
      {records.map(r => <circle key={r.id} data-point={r.id} cx={coordinate(r.x)} cy={coordinate(1-r.y)} r={r.id === pointId ? 6 : 3.5} fill={r.valid ? '#e2b55a' : '#101010'} stroke={r.id === pointId ? '#fff' : '#8b8b8b'} strokeWidth={r.id === pointId ? 2 : 1} onClick={() => setPointId(r.id)}><title>{`Point ${r.id}; x ${f(r.x)}, y ${f(r.y)}; tag ${f(r.position)}; ${r.valid ? 'retained' : 'masked'}`}</title></circle>)}
      <g pointerEvents="none"><circle cx={coordinate(byTime[0].x)} cy={coordinate(1-byTime[0].y)} r="10" fill="none" stroke="#eee" strokeWidth="2" /><rect x={coordinate(byTime.at(-1).x)-8} y={coordinate(1-byTime.at(-1).y)-8} width="16" height="16" fill="none" stroke="#eee" strokeWidth="2" /><text x="36" y="21">Selected point {pointId} · tag {f(point.position, 3)}</text></g>
      <circle className="long-drag-handle" data-point={point.id} cx={coordinate(point.x)} cy={coordinate(1-point.y)} r="22" fill="transparent" stroke="#e2b55a" strokeDasharray="3 4" onPointerDown={event => beginDrag(event, point)} onPointerMove={drag} onPointerUp={endDrag} onPointerCancel={endDrag} onLostPointerCapture={() => { grab.current = null; }}><title>Drag selected point {point.id}</title></circle>
    </svg><p>White ring: start, point {byTime[0].id}. White square and arrow: end, point {byTime.at(-1).id}. Hollow small marks are masked. Drag the amber dashed ring or use the x/y controls. The dashed path follows increasing position tags; complete storage reordering does not change it.</p></figure>
    <div><div className="neural-controls"><NeuralNumber label="Inspect point number" value={pointId} min={1} max={45} step={1} integer onChange={setPointId} /><NeuralNumber label="Selected point x" value={point.x} min={0} max={1} onChange={value => updatePoint('x', value)} /><NeuralNumber label="Selected point y" value={point.y} min={0} max={1} onChange={value => updatePoint('y', value)} /><NeuralSelect label="Inspect second-read latent" value={selectedLatent} onChange={value => setLatent(Number(value))} options={parameters.latent.map((_, i) => [i, `Latent ${i + 1}`])} /></div>
      <label className="long-check"><input type="checkbox" checked={point.valid} onChange={() => { if (point.valid && validCount === 1) setMessage('Keep at least one point so the attention denominator is defined.'); else updatePoint('valid', !point.valid); }} />Retain selected point in the read</label>{message && <p role="status">{message}</p>}
      <p>Point {point.id}: normalized ({f(2*point.x-1)}, {f(2*point.y-1)}), ordinal tag {f(point.position)}. Latent weight {f(current.weights[selectedLatent][index], 8)}.</p>
    </div></div>
    <div className="neural-buttons"><button onClick={() => { setRecords(previous => previous.map(r => ({ ...r, valid: r.id === 1 }))); setPointId(1); setMessage(''); }}>Retain only point 1</button><button onClick={() => { setRecords(previous => previous.map(r => ({ ...r, valid: true }))); setMessage(''); }}>Restore all points</button><button onClick={() => { setRecords(previous => previous.map(r => r.id === 23 ? { ...r, x: 1-r.x } : r)); setPointId(23); }}>Reflect point 23 x</button><button onClick={() => setRecords(previous => [...previous].reverse())}>Reverse whole tagged records</button><button onClick={() => setRecords(previous => { const ordered = [...previous].sort((a, b) => a.position - b.position), positions = ordered.map(r => r.position); return ordered.reverse().map((r, i) => ({ ...r, position: positions[i] })); })}>Reverse path under increasing tags</button><button onClick={() => { setRecords(copyRecords(specimen.records)); setPointId(1); setMessage(''); }}>Restore current trajectory</button><button onClick={reset}>Reset learned read</button></div>
    <NeuralPlot title="Coordinates along ordinal position" xLabel="ordinal tag (not elapsed time)" yLabel="raw coordinate" xDomain={[-1,1]} yDomain={[0,1]} series={[{ label: 'x', color: '#e2b55a', values: byTime.map(r => [r.position, r.x]) }, { label: 'y', color: '#bcbcbc', dashed: true, values: byTime.map(r => [r.position, r.y]) }]} />
    <figure className="long-attention-bars"><figcaption>Second-read weights, latent {selectedLatent + 1}; weights sum to one</figcaption><div>{byTime.map(r => <button key={r.id} aria-label={`Inspect attention for point ${r.id}`} aria-pressed={pointId === r.id} onClick={() => setPointId(r.id)}><i style={{ height: `${72 * current.weights[selectedLatent][records.findIndex(item => item.id === r.id)] / Math.max(...current.weights[selectedLatent])}px` }} /><span>{r.id}</span></button>)}</div><p>Shared bar scale: 0 to {f(Math.max(...current.weights[selectedLatent]), 7)} (current largest weight). Exact values are in the point inspector and table. These read weights describe the mechanism, not a full causal attribution.</p></figure>
    <div className="long-probabilities">{current.probabilities.map((probability, i) => <div key={i}><span>{i+1}. {names[i]}</span><div><i style={{ width: `${probability*100}%` }} /></div><output>{f(probability, 7)}</output></div>)}</div>
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Complete input rows, read weights and logits</h4><NeuralTable caption="Current records, with the original position tags unless explicitly reassigned" headers={['Point / storage row', 'x / y', 'Tag', 'Retained', 'Selected latent weight']} rows={records.map((r, i) => [r.id + ' / ' + i, `${f(r.x)} / ${f(r.y)}`, f(r.position), r.valid ? 'yes' : 'no', f(current.weights[selectedLatent][i], 9)])} /><NeuralTable caption="Current logits and probabilities; original output remains tied to the source and chosen model" headers={['Class', 'Logit', 'Current P', 'Original P', 'Mean baseline P']} rows={current.logits.map((logit, i) => [i+1, f(logit, 8), f(current.probabilities[i], 9), f(original.probabilities[i], 9), f(mean.probabilities[i], 9)])} /></section>
    <p>Try source row 7, four latents, seed 29: retaining point 1 changes class 1 to 10. Restoring all points and reflecting point 23 changes probability while keeping class 1. A full tagged-record reversal is the unchanged case. This workbench exposes only the predeclared validation rows.</p>
  </>;
}
export function LongContextTrajectoryLab() {
  const [open, setOpen] = useState(false), resource = useLongContextResource('trajectory-models.json', open, true);
  return <NeuralLab id="long-context-trajectory" title="Change an actual hand path and inspect a learned latent read"><p>Frozen fitted models; no browser training. Open the workbench to load 50 validation trajectories and four small model states. Their original collection, class names and CC BY 4.0 attribution are in the <a href={longContextAsset + 'data-provenance.md'}>Libras data record</a>.</p>{!open ? <button onClick={() => setOpen(true)}>Open learned trajectory workbench</button> : resource.error ? <p role="alert">{resource.error} <button onClick={resource.retry}>Retry model data</button></p> : !resource.data ? <p role="status">Loading frozen model and validation points…</p> : <TrajectoryWorkbench data={resource.data} />}</NeuralLab>;
}
