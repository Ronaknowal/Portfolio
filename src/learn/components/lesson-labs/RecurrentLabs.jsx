import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { NeuralLab, NeuralNumber, NeuralSelect, NeuralTable, NeuralPlot, formatNeural as fmt } from './NeuralLessonElements.jsx';
import { boundaryExperiment, lstmAccounting, paddingExperiment, pointStatistics, recurrentSequence, resetPlacement, retentionPath, scalarRecurrence, streamOwnership } from '../../data/recurrent-models.js';
import './recurrent-labs.css';

const assetBase = '/learn-assets/rnns-lstms-grus/';
const colors = ['#e4b752', '#c8c8c8', '#94afd4'];
const defaultPoints = [[.4, 1], [1, .94], [.4, .62], [-.1, .3], [-.4, -.02], [-.6, -.34], [-1, -.68], [-1, -1]];
const vector = values => '[' + values.map(value => fmt(value, 5)).join(', ') + ']';

function useRecurrentData(file) {
  const ref = useRef(null), [visible, setVisible] = useState(false), [data, setData] = useState(null);
  const [failed, setFailed] = useState(false), [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { setVisible(true); observer.disconnect(); }
    }, { rootMargin: '300px' });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible || data) return;
    const controller = new AbortController();
    setFailed(false);
    fetch(assetBase + file, { signal: controller.signal }).then(response => {
      if (!response.ok) throw new Error('Could not load lab data');
      return response.json();
    }).then(value => { if (!controller.signal.aborted) setData(value); })
      .catch(() => { if (!controller.signal.aborted) setFailed(true); });
    return () => controller.abort();
  }, [visible, file, attempt, data]);
  return { ref, data, fallback: failed ? <p role="alert">The retained data could not load. <button onClick={() => setAttempt(value => value + 1)}>Retry lab data</button></p> : <p role="status">Loading this investigation’s retained data…</p> };
}

export function RecurrentProgram({ file = 'recurrent-mechanics.py', title = 'Read the complete scratch recurrent mechanism', start, end }) {
  const [open, setOpen] = useState(false), [code, setCode] = useState(null), [failed, setFailed] = useState(false), [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!open || code) return;
    const controller = new AbortController();
    setFailed(false);
    fetch(assetBase + file, { signal: controller.signal }).then(response => {
      if (!response.ok) throw new Error('Source unavailable');
      return response.text();
    }).then(value => { if (!controller.signal.aborted) setCode(value); }).catch(() => { if (!controller.signal.aborted) setFailed(true); });
    return () => controller.abort();
  }, [open, code, attempt, file]);
  const from = code && start ? code.indexOf(start) : 0;
  const to = code && end ? code.indexOf(end, from + 1) : code?.length;
  const excerpt = code && from >= 0 && to > from ? code.slice(from, to).trim() : null;
  return <details className="neural-program" onToggle={event => setOpen(event.currentTarget.open)}><summary>{title}</summary>
    <p><a href={assetBase + file} download>Download {file}</a> · This view comes from that exact program.</p>
    {open && (excerpt ? <pre className="neural-program-source" tabIndex={0} role="region" aria-label={title}><code>{excerpt}</code></pre> : failed || code ? <p role="alert">This source view is unavailable. <button onClick={() => { setCode(null); setAttempt(value => value + 1); }}>Retry source</button></p> : <p role="status">Loading source…</p>)}
  </details>;
}

function PointPath({ points, selected, onSelect, onMove, title = 'Numbered pen trajectory' }) {
  const arrow = useId().replaceAll(':', ''), svg = useRef(null), drag = useRef(null);
  const xy = point => [35 + (point[0] + 1) * 125, 275 - (point[1] + 1) * 125];
  const move = event => {
    if (drag.current === null || !onMove) return;
    const p = svg.current.createSVGPoint(); p.x = event.clientX; p.y = event.clientY;
    const local = p.matrixTransform(svg.current.getScreenCTM().inverse());
    const bounded = value => Math.max(-1, Math.min(1, value));
    onMove(drag.current.index, [bounded((local.x - drag.current.dx - 35) / 125 - 1), bounded((275 - local.y + drag.current.dy) / 125 - 1)]);
  };
  return <figure className="recurrent-path"><figcaption>{title}</figcaption><svg ref={svg} viewBox="0 0 330 315" role="img" aria-label={`${title}. Point numbers show order, not elapsed time. Exact coordinates follow.`} onPointerMove={move} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
    <defs><marker id={arrow} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#939393" /></marker></defs>
    <path d="M35 25 V275 H285" fill="none" stroke="#666" />
    {points.slice(1).map((point, i) => <line key={i} x1={xy(points[i])[0]} y1={xy(points[i])[1]} x2={xy(point)[0]} y2={xy(point)[1]} stroke="#939393" strokeWidth="2" markerEnd={`url(#${arrow})`} />)}
    {points.map((point, i) => <g key={i}><circle cx={xy(point)[0]} cy={xy(point)[1]} r={i === selected ? 12 : 10} fill={i === selected ? '#e4b752' : '#171717'} stroke={i === selected ? '#e4b752' : '#aaa'} onPointerDown={event => {
      if (!onMove) return;
      const p = svg.current.createSVGPoint(); p.x = event.clientX; p.y = event.clientY;
      const local = p.matrixTransform(svg.current.getScreenCTM().inverse());
      drag.current = { index: i, dx: local.x - xy(point)[0], dy: local.y - xy(point)[1] };
      onSelect(i); event.currentTarget.setPointerCapture(event.pointerId);
    }} /><text className="recurrent-point-label" style={{ fill: i === selected ? '#111' : '#eee' }} x={xy(point)[0]} y={xy(point)[1] + 5} textAnchor="middle">{i + 1}</text></g>)}
    <text x="35" y="298">−1</text><text x="285" y="298" textAnchor="end">1 · x</text><text x="8" y="32">1</text><text x="4" y="270">−1</text>
  </svg><p>Normalized coordinates; positions are not timestamps.{onMove && ' Drag a point or use the point selector and numeric controls.'}</p></figure>;
}

export function PenRepresentationLab() {
  const [points, setPoints] = useState(defaultPoints);
  const statistics = pointStatistics(points);
  return <NeuralLab id="pen-representation" title="Same points, different information">
    <p>Real specimen pendigits.tes:6. Reverse or swap its positions: ordered inputs change, while orderless statistics remain fixed.</p>
    <div className="neural-buttons"><button onClick={() => setPoints([...points].reverse())}>Reverse positions</button><button onClick={() => setPoints(points.map((point, i) => i === 2 ? points[3] : i === 3 ? points[2] : point))}>Swap positions 3 and 4</button><button onClick={() => setPoints(defaultPoints)}>Reset representation</button></div>
    <div className="neural-two"><PointPath points={points} /><div><p><strong>Ordered input slots</strong></p><ol className="recurrent-slots">{points.map((point, i) => <li key={i}><b>Position {i + 1}</b><span>{vector(point)}</span></li>)}</ol></div></div>
    <NeuralTable caption="Orderless features discard these permutations" headers={['Axis', 'Mean', 'Std. deviation', 'Minimum', 'Maximum']} rows={statistics.map((row, i) => [i ? 'y' : 'x', ...Object.values(row).map(value => fmt(value))])} />
  </NeuralLab>;
}

const scalarDefault = { inputs: [.3, -.5, .2], inputWeight: .7, recurrentWeight: .4, bias: -.05, initial: .1, target: -.1, rate: .1 };
export function RecurrentCreditLab() {
  const [state, setState] = useState(scalarDefault), [step, setStep] = useState(2), [revision, setRevision] = useState(0);
  const result = scalarRecurrence(state), update = (name, value) => setState(current => ({ ...current, [name]: value }));
  return <NeuralLab id="recurrent-credit" title="Follow state forward and shared-weight credit backward">
    <p>Constructed scalar cell. Change a value to recompute the entire forward pass, gradient and simultaneous proposed update. Step through the trace without hiding its result.</p>
    <div className="neural-controls" key={revision}>{state.inputs.map((value, i) => <NeuralNumber key={i} label={`Observation ${i + 1}`} value={value} min={-1} max={1} step="any" onChange={next => update('inputs', state.inputs.map((old, j) => i === j ? next : old))} />)}
      {[['inputWeight', 'Input weight', -2, 2], ['recurrentWeight', 'Recurrent weight', -2, 2], ['bias', 'Bias', -2, 2], ['initial', 'Initial state', -1, 1], ['target', 'Target', -1, 1], ['rate', 'Learning rate', 0, .2]].map(([name, label, min, max]) => <NeuralNumber key={name} label={label} value={state[name]} min={min} max={max} step="any" onChange={value => update(name, value)} />)}</div>
    <div className="recurrent-unrolled">{result.steps.map((row, i) => <button key={i} aria-pressed={step === i} onClick={() => setStep(i)}><b>State h{i + 1}</b><span>{fmt(row.hidden)}</span><small>same wₓ, wₕ, b</small></button>)}</div>
    <div className="neural-buttons"><button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous step</button><button disabled={step === 2} onClick={() => setStep(step + 1)}>Next step</button><button onClick={() => { setState(scalarDefault); setStep(2); setRevision(revision + 1); }}>Reset recurrence</button></div>
    <p className="neural-result">Step {step + 1}: tanh(input {fmt(result.steps[step].incoming)} + old-state {fmt(result.steps[step].retained)} + bias {fmt(state.bias)}) = {fmt(result.steps[step].hidden)}. Backward credit at its preactivation: {fmt(result.contributions[step].delta)}.</p>
    <NeuralTable caption="Add contributions because all three positions reuse each parameter" headers={['Position', '∂L/∂wₓ contribution', '∂L/∂wₕ contribution', '∂L/∂b contribution']} rows={result.contributions.map((row, i) => [i + 1, fmt(row.inputWeight), fmt(row.recurrentWeight), fmt(row.bias)])} />
    <p data-result="credit">Current final state {fmt(result.states.at(-1), 7)}; loss {fmt(result.loss, 7)}. Summed gradient {vector(Object.values(result.gradient))}. Updated parameters {vector(Object.values(result.updated))}; recomputed loss {fmt(result.updatedLoss, 7)}.</p>
    <p>Changing only the target leaves forward states fixed. Rate 0 leaves parameters unchanged. A positive rate can still increase loss; the display reports that result.</p>
  </NeuralLab>;
}

const cellDefault = { cell: -.4, forget: .85, input: .3, candidate: .6, output: .7 };
export function LstmMemoryLab() {
  const [state, setState] = useState(cellDefault), [revision, setRevision] = useState(0);
  const result = lstmAccounting(state);
  return <NeuralLab id="lstm-memory" title="Retain, write, add, then expose">
    <p>Constructed gate intervention: real gates are learned affine functions of input and previous hidden state. These controls isolate their different jobs.</p>
    <div className="neural-controls" key={revision}>{[['cell', 'Old cell', -2, 2], ['forget', 'Forget gate', 0, 1], ['input', 'Input gate', 0, 1], ['candidate', 'Candidate write', -1, 1], ['output', 'Output gate', 0, 1]].map(([name, label, min, max]) => <NeuralNumber key={name} label={label} value={state[name]} min={min} max={max} step="any" onChange={value => setState({ ...state, [name]: value })} />)}</div>
    <div className="recurrent-accounting"><div><small>OLD × FORGET</small><strong>{fmt(result.retained)}</strong><span>Retained signed value</span></div><b aria-hidden="true">+</b><div><small>INPUT × CANDIDATE</small><strong>{fmt(result.written)}</strong><span>New signed write</span></div><b aria-hidden="true">→</b><div><small>NEW CELL</small><strong>{fmt(result.cell)}</strong><span>Sum, before reading</span></div><b aria-hidden="true">→</b><div><small>OUTPUT × TANH(CELL)</small><strong>{fmt(result.hidden)}</strong><span>Hidden output</span></div></div>
    <figure className="recurrent-signed"><figcaption>Signed memory accounting · one common −3 to +3 scale</figcaption>{[['Retained', result.retained], ['Written', result.written], ['New cell', result.cell], ['Readout', result.hidden]].map(([name, value]) => <div key={name}><span>{name}</span><div className="recurrent-signed-track"><span style={{ left: `${50 + Math.min(0, value) / 6 * 100}%`, width: `${Math.abs(value) / 6 * 100}%` }} /></div><output>{fmt(value)}</output></div>)}<p>Left of the center line is negative; right is positive. Opposite signs cancel when the retained value and new write are added. The readout applies a separate nonlinear compression and output gate.</p></figure>
    <p data-result="cell">Cell {fmt(result.cell, 7)}; hidden output {fmt(result.hidden, 7)}. Changing only the output gate changes this readout, not this step’s cell. Input gate 0 makes candidate edits ineffective for this step.</p>
    <button onClick={() => { setState(cellDefault); setRevision(revision + 1); }}>Reset cell</button>
  </NeuralLab>;
}

export function LstmRetentionLab() {
  const [fraction, setFraction] = useState(.65), [horizon, setHorizon] = useState(80), [writeStep, setWriteStep] = useState(25), [write, setWrite] = useState(0), [revision, setRevision] = useState(0);
  const result = retentionPath(fraction, horizon, writeStep, write);
  return <NeuralLab id="lstm-retention" title="Design a direct memory path">
    <p>Fixed-gate cell path, starting at 1. This is a controlled recurrence, not a measured full LSTM gradient.</p>
    <div className="neural-controls" key={revision}><NeuralNumber label="Fraction retained at horizon" value={fraction} min={.05} max={.99} step="any" onChange={setFraction} /><NeuralNumber label="Horizon in steps" value={horizon} min={1} max={200} step={1} integer onChange={value => { setHorizon(value); setWriteStep(Math.min(writeStep, value)); }} /><NeuralNumber label="Write position" value={writeStep} min={1} max={horizon} step={1} integer onChange={setWriteStep} /><NeuralNumber label="One signed write" value={write} min={-.5} max={.5} step="any" onChange={setWrite} /></div>
    <NeuralPlot title="Retention with and without one write" xLabel="Step" yLabel="Cell value" xDomain={[0, horizon]} yDomain={[Math.min(0, ...result.changed.map(row => row[1])), Math.max(1.05, ...result.changed.map(row => row[1]))]} series={[{ label: 'No writes · fᵗ', color: colors[1], dashed: true, values: result.plain }, { label: 'Current path', color: colors[0], values: result.changed }]} />
    <p data-result="retention">Forget factor {fmt(result.forget, 9)}; sigmoid preactivation {fmt(result.bias, 6)}; half-life {fmt(result.halfLife, 6)} steps. End without writes: {fmt(result.plain.at(-1)[1])}; with write: {fmt(result.changed.at(-1)[1])}.</p>
    <p>The required forget factor may exceed a convenient slider’s rounded endpoint; it is calculated exactly from the requested fraction and horizon. No clamping changes that relationship.</p>
    <button onClick={() => { setFraction(.65); setHorizon(80); setWriteStep(25); setWrite(0); setRevision(revision + 1); }}>Reset retention</button>
  </NeuralLab>;
}

const resetDefault = { hidden: [-.5, 1.5], reset: [.7, .3], matrix: [[.4, -1.2], [1.1, .5]], bias: [.2, -.1] };
function ResetEdges({ state, after }) {
  const marker = useId().replaceAll(':', '');
  return <figure className="recurrent-edges"><figcaption>{after ? 'Reset each destination after summing' : 'Reset each source before its outgoing edges'}</figcaption><svg viewBox="0 0 360 290" role="img" aria-label={after ? 'Four mixing edges. Each output gate multiplies all incoming contributions and its bias.' : 'Four mixing edges. Each source gate multiplies both outgoing contributions. Bias is added afterward.'}>
    <defs><marker id={marker} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#aaa" /></marker></defs>
    {state.matrix.flatMap((row, i) => row.map((weight, j) => <line key={`${i}${j}`} x1="79" y1={65 + j * 145} x2="273" y2={65 + i * 145} stroke={weight < 0 ? '#aaa' : '#e4b752'} strokeDasharray={weight < 0 ? '5 4' : undefined} strokeWidth={1.5 + Math.min(4, Math.abs(weight))} markerEnd={`url(#${marker})`} />))}
    {[0, 1].map(i => <g key={i}><circle cx="64" cy={65 + i * 145} r="16" fill="#171717" stroke="#aaa" /><text x="64" y={71 + i * 145} textAnchor="middle">h{i + 1}</text><circle cx="294" cy={65 + i * 145} r="16" fill="#171717" stroke="#aaa" /><text x="294" y={71 + i * 145} textAnchor="middle">Σ{i + 1}</text><text x={after ? 292 : 64} y={100 + i * 145} textAnchor="middle" fill="#e4b752">× r{i + 1}={fmt(state.reset[i], 2)}</text></g>)}
    <text x="178" y="49" textAnchor="middle">W₁₁ = {fmt(state.matrix[0][0])}</text><text x="178" y="238" textAnchor="middle">W₂₂ = {fmt(state.matrix[1][1])}</text>
    <rect x="143" y="105" width="108" height="25" fill="#101010" /><text x="197" y="124" textAnchor="middle">W₁₂={fmt(state.matrix[0][1])}</text><rect x="101" y="147" width="108" height="25" fill="#101010" /><text x="155" y="166" textAnchor="middle">W₂₁={fmt(state.matrix[1][0])}</text>
    <text x="180" y="282" textAnchor="middle">Solid: positive · dashed: negative</text>
  </svg><p>{after ? 'Each rᵢ scales destination i’s entire mixed affine value, including bᵢ.' : 'Each rⱼ scales source j for both destinations; bᵢ is added after mixing.'} Exact signed edge contributions are tabulated below.</p></figure>;
}
export function GruResetLab() {
  const [state, setState] = useState(resetDefault), [revision, setRevision] = useState(0);
  const result = resetPlacement(state.hidden, state.reset, state.matrix, state.bias);
  const setEntry = (name, i, value) => setState({ ...state, [name]: state[name].map((old, j) => i === j ? value : old) });
  return <NeuralLab id="gru-reset" title="A gate before mixing is different from a gate after mixing">
    <p>Compare the recurrent candidate contribution, before its input term and tanh. The separate update gate still controls the final old/new blend.</p>
    <div className="neural-controls" key={revision}>{state.reset.map((value, i) => <NeuralNumber key={`r${i}`} label={`Reset coordinate ${i + 1}`} value={value} min={0} max={1} step="any" onChange={next => setEntry('reset', i, next)} />)}{state.bias.map((value, i) => <NeuralNumber key={`b${i}`} label={`Hidden bias ${i + 1}`} value={value} min={-1} max={1} step="any" onChange={next => setEntry('bias', i, next)} />)}{state.matrix.flatMap((row, i) => row.map((value, j) => <NeuralNumber key={`${i}${j}`} label={`Mixing weight W${i + 1}${j + 1}`} value={value} min={-3} max={4} step="any" onChange={next => setState({ ...state, matrix: state.matrix.map((old, index) => index === i ? old.map((v, k) => j === k ? next : v) : old) })} />))}</div>
    <div className="neural-two"><div className="recurrent-lane"><b>Reset source coordinates → mix → bias</b><p>h = {vector(state.hidden)}</p><p>r ⊙ h = {vector(state.hidden.map((v, i) => v * state.reset[i]))}</p><p>W(r ⊙ h) + b = <strong>{vector(result.before)}</strong></p></div><div className="recurrent-lane"><b>Mix → bias → reset mixed outputs</b><p>h = {vector(state.hidden)}</p><p>Wh + b, then scale each output</p><p>r ⊙ (Wh + b) = <strong>{vector(result.after)}</strong></p></div></div>
    <div className="neural-two"><ResetEdges state={state} after={false} /><ResetEdges state={state} after /></div>
    <NeuralTable caption="Each edge contributes to a different reset placement" headers={['Edge', 'Before: Wᵢⱼ rⱼ hⱼ', 'After: rᵢ Wᵢⱼ hⱼ']} rows={state.matrix.flatMap((row, i) => row.map((w, j) => [`h${j + 1} → output ${i + 1}`, fmt(w * state.reset[j] * state.hidden[j]), fmt(state.reset[i] * w * state.hidden[j])] ))} />
    <p data-result="reset">Before {vector(result.before)}; after {vector(result.after)}. All-one reset agrees. A diagonal matrix agrees for arbitrary reset only when its hidden bias contribution also agrees, for example with zero bias.</p>
    <div className="neural-buttons"><button onClick={() => setState({ ...state, reset: [1, 1] })}>Set reset to ones</button><button onClick={() => setState({ ...state, matrix: [[.4, 0], [0, .5]], bias: [0, 0] })}>Diagonal, zero bias</button><button onClick={() => { setState(resetDefault); setRevision(revision + 1); }}>Reset comparison</button></div>
  </NeuralLab>;
}

function FrozenPenExplorer({ data }) {
  const [kind, setKind] = useState('gru'), [sample, setSample] = useState(0), [points, setPoints] = useState(data.specimens[0].points);
  const [point, setPoint] = useState(2), [feature, setFeature] = useState(0), [selectedClass, setSelectedClass] = useState(1), [revision, setRevision] = useState(0);
  const baseline = useMemo(() => recurrentSequence(kind, data.specimens[sample].points, data.weights[kind]), [data, kind, sample]);
  const current = useMemo(() => recurrentSequence(kind, points, data.weights[kind]), [data, kind, points]);
  const original = baseline.at(-1).probabilities, probabilities = current.at(-1).probabilities;
  const winner = values => values.indexOf(Math.max(...values));
  const move = (i, next) => setPoints(old => old.map((value, j) => i === j ? next : value));
  const gates = current[point].gates;
  return <>
    <p>Fixed seed-1 fitted weights; actual specimen {data.specimens[sample].sourceId}, digit {data.specimens[sample].digit}. Valid edits run bounded NumPy-equivalent cell arithmetic in this browser. No training happens here.</p>
    <div className="neural-controls" key={revision}><NeuralSelect label="Recurrent model" value={kind} onChange={setKind} options={['rnn', 'lstm', 'gru'].map(v => [v, v.toUpperCase()])} /><NeuralSelect label="Pen specimen" value={sample} onChange={value => { const n = Number(value); setSample(n); setPoints(data.specimens[n].points); }} options={data.specimens.map((row, i) => [i, `${row.sourceId} · digit ${row.digit}`])} /><NeuralSelect label="Selected point" value={point} onChange={value => setPoint(Number(value))} options={points.map((_, i) => [i, `Position ${i + 1}`])} /><NeuralSelect label="Hidden feature" value={feature} onChange={value => setFeature(Number(value))} options={Array.from({ length: 32 }, (_, i) => [i, `Feature ${i + 1}`])} /><NeuralNumber label="Point x coordinate" value={points[point][0]} min={-1} max={1} step="any" onChange={value => move(point, [value, points[point][1]])} /><NeuralNumber label="Point y coordinate" value={points[point][1]} min={-1} max={1} step="any" onChange={value => move(point, [points[point][0], value])} /><NeuralSelect label="Probability to inspect" value={selectedClass} onChange={value => setSelectedClass(Number(value))} options={probabilities.map((_, i) => [i, `Digit ${i}`])} /></div>
    <div className="neural-two"><PointPath points={points} selected={point} onSelect={setPoint} onMove={move} title="Edit the completed pen trace" /><NeuralPlot title={`Hidden feature ${feature + 1} across the same trace`} xLabel="Position" yLabel="Hidden value" xDomain={[1, 8]} yDomain={[-1, 1]} series={[{ label: 'Original input', color: colors[1], dashed: true, values: baseline.map((row, i) => [i + 1, row.hidden[feature]]) }, { label: 'Current input', color: colors[0], values: current.map((row, i) => [i + 1, row.hidden[feature]]) }]} /></div>
    <p className="neural-result" data-result="pen">Original top digit {winner(original)}; current top digit {winner(probabilities)}. P(digit {selectedClass}) {fmt(original[selectedClass], 8)} → {fmt(probabilities[selectedClass], 8)}; change {fmt(probabilities[selectedClass] - original[selectedClass], 8)}.</p>
    <div className="recurrent-probabilities" aria-label="All ten current class probabilities">{probabilities.map((value, i) => <div key={i}><span>Digit {i}</span><meter aria-label={`Probability of digit ${i}`} min={0} max={1} value={value} /><output>{fmt(value, 7)}</output></div>)}</div>
    <NeuralTable caption={`Position ${point + 1}, feature ${feature + 1}: computed state and gates`} headers={['Quantity', 'Current value']} rows={[[ 'Hidden', fmt(current[point].hidden[feature], 7)], ...Object.entries(gates).map(([name, values]) => [name.replaceAll('_', ' '), fmt(values[feature], 7)])]} />
    <div className="neural-buttons"><button onClick={() => setPoints([...points].reverse())}>Reverse current trace</button><button onClick={() => setPoints(points.map((value, i) => i === 2 ? points[3] : i === 3 ? points[2] : value))}>Swap current points 3 and 4</button><button onClick={() => { setKind('gru'); setSample(0); setPoints(data.specimens[0].points); setPoint(2); setFeature(0); setSelectedClass(1); setRevision(revision + 1); }}>Reset fitted trace</button></div>
    <p>These are completed, resampled trajectories. Intermediate hidden readouts do not establish validated early recognition. A feature index has no assigned human meaning, and a probability change need not change the top class.</p>
    <details><summary>All current coordinates and selected state values</summary><NeuralTable caption="Full eight-position trace" headers={['Position', 'x', 'y', 'Selected hidden']} rows={current.map((row, i) => [i + 1, fmt(points[i][0]), fmt(points[i][1]), fmt(row.hidden[feature], 7)])} /></details>
  </>;
}

export function RecurrentPenLab() {
  const { ref, data, fallback } = useRecurrentData('pen-models.json');
  return <div ref={ref}><NeuralLab id="recurrent-pen" title="Change a real pen trace and inspect the fitted model">{data ? <FrozenPenExplorer data={data} /> : fallback}</NeuralLab></div>;
}

function BoundaryExplorer({ data }) {
  const [boundary, setBoundary] = useState(3), [mode, setMode] = useState('carry'), [swap, setSwap] = useState(false), [revision, setRevision] = useState(0);
  const sequence = data.sequence[0];
  const result = useMemo(() => boundaryExperiment(sequence, data.weights, boundary, mode), [sequence, data, boundary, mode]);
  const owners = streamOwnership(sequence, data.weights, boundary, swap);
  return <><div className="neural-controls" key={revision}><NeuralNumber label="Chunk boundary after position" value={boundary} min={1} max={4} integer step={1} onChange={setBoundary} /><NeuralSelect label="Boundary operation" value={mode} onChange={setMode} options={['carry', 'detach', 'reset'].map(value => [value, value])} /></div>
    <div className="recurrent-boundary"><div>Prefix · positions 1–{boundary}</div><div><b>Forward value {mode === 'reset' ? 'replaced by zero' : 'carried →'}</b><br /><b>Backward credit {mode === 'carry' ? '← connected' : 'cut at boundary'}</b></div><div>Suffix · positions {boundary + 1}–5</div></div>
    <p data-result="boundary">Maximum final-state change versus the full pass: {fmt(result.finalError, 9)}. Loss is the sum of squared suffix hidden values: {fmt(result.loss, 7)}.</p>
    <NeuralTable caption="Current input credit, independently estimated by central differences" headers={['Position', '∂L/∂x', '∂L/∂y', 'Current hidden state']} rows={result.gradients.map((row, i) => [i + 1, fmt(row[0], 7), fmt(row[1], 7), vector(result.current[i].hidden)])} />
    <p>Detach carries the same numerical state but treats it as constant for later credit. Reset changes the forward values too. These effects use fixed weights; an optimizer update creates a different comparison.</p>
    <label className="recurrent-checkbox"><input type="checkbox" checked={swap} onChange={e => setSwap(e.target.checked)} /> Give each stream the other stream’s prefix state</label>
    <NeuralTable caption="Stream identity owns state, even when batch rows move" headers={['Stream', 'Correct final', 'Current final']} rows={owners.current.map((row, i) => [i ? 'B · reverse order' : 'A · original order', vector(owners.correct[i]), vector(row)])} />
    <p data-result="ownership">Maximum final-state change from ownership: {fmt(owners.difference, 9)}. Reordering both inputs and their owned states preserves meaning; exchanging state alone does not.</p>
    <button onClick={() => { setBoundary(3); setMode('carry'); setSwap(false); setRevision(revision + 1); }}>Reset boundaries</button></>;
}
export function RecurrentBoundaryLab() {
  const { ref, data, fallback } = useRecurrentData('recurrent-evidence.json');
  return <div ref={ref}><NeuralLab id="recurrent-boundary" title="Carry values and backward credit across a boundary">{data ? <BoundaryExplorer data={data.boundaries} /> : fallback}</NeuralLab></div>;
}

function PaddingExplorer({ data }) {
  const [length, setLength] = useState(2), [padding, setPadding] = useState(-3.5), [revision, setRevision] = useState(0);
  const result = paddingExperiment(data.sequence[0], data.padding.True.weights, length, padding);
  return <><p>The same five-position storage contains a shorter valid sequence. The forward path visits left to right; the backward path visits padding first. Trimming to the true length models what packing excludes.</p>
    <div className="neural-controls" key={revision}><NeuralNumber label="Valid sequence length" value={length} min={1} max={5} integer step={1} onChange={setLength} /><NeuralNumber label="Value in each padded coordinate" value={padding} min={-4} max={4} step="any" onChange={setPadding} /></div>
    <ol className="recurrent-padding">{result.padded.map((point, i) => <li key={i} data-valid={i < length}><b>{i + 1} · {i < length ? 'valid' : 'padding'}</b><span>{vector(point)}</span></li>)}</ol>
    <p className="neural-result" data-result="padding">Changing padding from zero: valid forward change {fmt(result.forwardEditError, 8)}; valid backward change {fmt(result.backwardEditError, 8)}. Padded final forward state versus true final state: {fmt(result.paddedFinalError, 8)}.</p>
    <NeuralTable caption="Forward → and backward ← states at each valid position" headers={['Position', 'Forward, padded', 'Backward, padded', 'Backward, true length']} rows={result.forward.slice(0, length).map((row, i) => [i + 1, vector(row.hidden), vector(result.backward[i].hidden), vector(result.packedBackward[i].hidden)])} />
    <p>True-length states ignore the padded values. Changing those values cannot change the packed result. Reading the last padded state or the backward half at the last valid index answers a different question from the two directional final states.</p>
    <p>For a batch with lengths 5, 4 and {length}, there are {9 + length} valid target positions. Dividing a summed token loss by all 15 storage positions scales it by {fmt((9 + length) / 15)} relative to the valid-position mean.</p>
    <button onClick={() => { setLength(2); setPadding(-3.5); setRevision(revision + 1); }}>Reset padding</button></>;
}
export function RecurrentPaddingLab() {
  const { ref, data, fallback } = useRecurrentData('recurrent-evidence.json');
  return <div ref={ref}><NeuralLab id="recurrent-padding" title="A padded zero is still an input unless you exclude it">{data ? <PaddingExplorer data={data.boundaries} /> : fallback}</NeuralLab></div>;
}

function MeasuredExplorer({ data }) {
  const [kind, setKind] = useState('gru'), [seed, setSeed] = useState(1);
  const run = data.runs.find(row => row.kind === kind && row.seed === seed);
  return <><p>Recorded CPU fits, 500 updates each. These controls select saved evidence; they do not train or interpolate a model.</p><div className="neural-controls"><NeuralSelect label="Recorded architecture" value={kind} onChange={setKind} options={['rnn', 'lstm', 'gru'].map(v => [v, v.toUpperCase()])} /><NeuralSelect label="Recorded seed" value={seed} onChange={value => setSeed(Number(value))} options={[1, 2, 3].map(v => [v, String(v)])} /></div>
    <NeuralPlot title="Loss through the measured training run" xLabel="Optimizer updates" yLabel="Cross-entropy" xDomain={[0, 500]} yDomain={[0, Math.max(...run.trajectory.flatMap(row => [row.train.cross_entropy, row.development.cross_entropy])) * 1.05]} series={['train', 'development'].map((key, i) => ({ label: key, color: colors[i], dashed: i === 1, values: run.trajectory.map(row => [row.step, row[key].cross_entropy]) }))} />
    <NeuralTable caption="Every retained checkpoint, with exact counts" headers={['Update', 'Train loss', 'Dev loss', 'Train correct / 600', 'Dev correct / 300']} rows={run.trajectory.map(row => [row.step, fmt(row.train.cross_entropy, 7), fmt(row.development.cross_entropy, 7), row.train.correct, row.development.correct])} />
    <p>Original: {run.final.correct}/300; reversed: {run.reversed.correct}/300; swap points 3/4: {run.swapped_points_3_4.correct}/300. Clipped updates: {run.clipped_updates}/500. Parameter count: {run.parameters}.</p>
    <p>Reversal and swapping are fixed-weight distribution probes, not independently sampled tests. The inspected set is development evidence; these curves establish no speed ranking.</p></>;
}
export function RecurrentMeasuredLab() {
  const { ref, data, fallback } = useRecurrentData('recurrent-evidence.json');
  return <div ref={ref}><NeuralLab id="recurrent-measured" title="Inspect all nine measured runs">{data ? <MeasuredExplorer data={data} /> : fallback}</NeuralLab></div>;
}
