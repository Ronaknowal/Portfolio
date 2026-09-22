import { useEffect, useState } from 'react';
import { NeuralLab, NeuralNumber, NeuralSelect, NeuralTable, formatNeural as number } from './NeuralLessonElements.jsx';
import { activationMoments, directionalGeometry, gatedIdentity, idealSignal, initializationSchemes, symmetryState, widthConfiguration } from '../../data/weight-initialization-models.js';
import records from '../../data/weight-initialization-measurements.json';
import './weight-initialization.css';

const amber = '#e2b55a', pale = '#dedede';
const seedOptions = ['1', '2', '3'].map(value => [value, `Seed ${value}`]);
export const initializationAsset = '/learn-assets/weight-initialization-xavier-kaiming-p/';

export function InitializationProgram({ file = 'initialization-experiments.py', title = 'Read the complete CPU experiment' }) {
  const [open, setOpen] = useState(false), [source, setSource] = useState(null), [error, setError] = useState(false), [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!open || source) return;
    const controller = new AbortController();
    setError(false);
    fetch(initializationAsset + file, { signal: controller.signal }).then(response => { if (!response.ok) throw new Error('Source request failed'); return response.text(); }).then(setSource).catch(reason => { if (reason.name !== 'AbortError') setError(true); });
    return () => controller.abort();
  }, [open, source, file, attempt]);
  return <details className="neural-program" onToggle={event => setOpen(event.currentTarget.open)}><summary>{title}</summary>
    {open && (source ? <pre className="neural-program-source" tabIndex={0} role="region" aria-label={`${file}; scroll code horizontally`}><code>{source}</code></pre> : error ? <p role="alert">The source could not load. <button onClick={() => setAttempt(value => value + 1)}>Retry source</button></p> : <p role="status">Loading source…</p>)}
    <p><a href={initializationAsset + file} download={file}>Download {file}</a></p>
  </details>;
}

function Plot({ title, series, logarithmic = false, xLabel = 'Layer', yLabel, xDomain, yDomain }) {
  const positive = series.flatMap(line => line.values.map(([, value]) => value)).filter(value => !logarithmic || value > 0);
  const hasZero = logarithmic && series.some(line => line.values.some(([, value]) => value === 0));
  const allX = series.flatMap(line => line.values.map(([value]) => value));
  const [xmin, xmax] = xDomain || [Math.min(...allX), Math.max(...allX)];
  let [ymin, ymax] = yDomain || (logarithmic ? [Math.floor(Math.log10(Math.min(...positive))), Math.ceil(Math.log10(Math.max(...positive)))] : [0, Math.max(...positive) * 1.08]);
  if (ymax === ymin) { ymin -= 1; ymax += 1; }
  // Reserve room for signed power labels in the site's monospaced font, including on phones.
  const left = 80, right = 322, top = 24, bottom = hasZero ? 185 : 205;
  const x = value => left + (value - xmin) / (xmax - xmin || 1) * (right - left);
  const y = value => value === 0 && logarithmic ? 213 : bottom - ((logarithmic ? Math.log10(value) : value) - ymin) / (ymax - ymin) * (bottom - top);
  const middleTick = logarithmic ? Math.round((ymin + ymax) / 2) : (ymin + ymax) / 2;
  const ticks = [...new Set([ymin, middleTick, ymax])];
  return <figure><figcaption>{title}</figcaption><svg className="init-chart" viewBox="0 0 340 250" role="img" aria-label={`${title}. ${xLabel}; ${yLabel}. Exact values in the adjacent table.`}>
    <line className="init-axis" x1={left} x2={left} y1={top} y2={bottom} /><line className="init-axis" x1={left} x2={right} y1={bottom} y2={bottom} />
    {ticks.map(tick => <g key={tick}><text x={left - 7} y={bottom - (tick - ymin) / (ymax - ymin) * (bottom - top) + 4} textAnchor="end">{logarithmic ? `10^${number(tick, 1)}` : number(tick, 2)}</text></g>)}
    {[xmin, (xmin + xmax) / 2, xmax].map(tick => <text key={tick} x={x(tick)} y="240" textAnchor="middle">{number(tick, 1)}</text>)}
    {hasZero && <><line x1={left} x2={right} y1="213" y2="213" stroke="#505050" strokeDasharray="2 4" /><text x={left - 7} y="217" textAnchor="end">zero</text></>}
    {series.map((line, index) => <g key={line.name}>{line.values.map(([a, b], i) => {
      const next = line.values[i + 1];
      return <g key={i}>{next && !(logarithmic && ((b === 0) !== (next[1] === 0))) && <line x1={x(a)} y1={y(b)} x2={x(next[0])} y2={y(next[1])} stroke={index ? pale : amber} strokeWidth="2" strokeDasharray={index ? '5 4' : undefined} />}<circle cx={x(a)} cy={y(b)} r="2.8" fill={index ? pale : amber}><title>{line.name}: {a}, {number(b, 7)}</title></circle></g>;
    })}</g>)}
  </svg><p className="init-reading">Horizontal: {xLabel}. Vertical: {yLabel}{logarithmic ? '; logarithmic positive scale' : ''}.{hasZero && ' Exact zeros use a separate rail; no artificial epsilon.'}</p><div className="init-legend">{series.map((line, index) => <span key={line.name} data-dashed={Boolean(index)} style={{ '--series-color': index ? pale : amber }}>{line.name}</span>)}</div></figure>;
}

export function InitializationSignalLab() {
  const [scheme, setScheme] = useState('small'), [seed, setSeed] = useState('1'), [variance, setVariance] = useState(.01), [depth, setDepth] = useState(20);
  const current = records.propagation.find(row => row.seed === Number(seed) && row.scheme === scheme), baseline = records.propagation.find(row => row.seed === Number(seed) && row.scheme === 'kaiming');
  const ratio = current.layers.at(-1).q / current.layers[0].q, theory = idealSignal(variance, depth);
  return <NeuralLab title="Follow a signal through twenty layers" id="initialization-signal"><p>Saved float64 probes: 128 fixed inputs, width 64, input/cotangent seed 71. Selections read recorded runs; they do not train a model here. Dashed Kaiming uses the same selected seed.</p>
    <div className="neural-controls"><NeuralSelect label="Recorded initialization" value={scheme} onChange={setScheme} options={initializationSchemes} /><NeuralSelect label="Propagation seed" value={seed} onChange={setSeed} options={seedOptions} /></div>
    <div className="neural-two">{[['q', 'Forward second moment', 'pooled mean(h²)'], ['g', 'Backward sensitivity', 'RMS ∂probe/∂h']].map(([key, title, unit]) => <Plot key={key} title={title} yLabel={unit} logarithmic series={[{ name: initializationSchemes.find(([id]) => id === scheme)[1], values: current.layers.map(row => [row.layer, row[key]]) }, { name: 'Kaiming, same seed', values: baseline.layers.map(row => [row.layer, row[key]]) }]} />)}</div>
    <p className="neural-result" data-result="signal">Layer 20 / input mean square: {number(ratio)} — {ratio < .1 ? 'below one tenth' : ratio > 10 ? 'above tenfold' : 'within one tenth to tenfold'}. This descriptive range does not certify trainability. Input gradient RMS: {number(current.layers[0].g)}.</p>
    <p>The external output cotangent has RMS {number(records.cotangentRms)} for every scheme. Even a zero activation can have a nonzero derivative with respect to that output; the gates can still block its route back to the input.</p>
    <details><summary>Exact forward and backward measurements</summary><NeuralTable caption="Current and reference measurements" headers={['Layer', 'Current mean square', 'Kaiming mean square', 'Current gradient RMS', 'Kaiming gradient RMS']} rows={current.layers.map((row, i) => [row.layer, number(row.q, 8), number(baseline.layers[i].q, 8), number(row.g, 8), number(baseline.layers[i].g, 8)])} /></details>
    <h4>Construct an idealized recurrence</h4><p>A separate assumption-based calculation: fan-in 100, symmetric ReLU inputs, one-layer factor = 100 × weight variance / 2. No new measured curve is implied.</p>
    <div className="neural-controls"><NeuralNumber label="Weight variance" value={variance} onChange={setVariance} min={0} max={.1} step={.001} /><NeuralNumber label="Idealized depth" value={depth} onChange={setDepth} min={1} max={30} step={1} integer /></div>
    <p className="neural-result" data-result="recurrence">Per-layer factor {number(theory.factor)}; after {depth} layers, q / q₀ = {number(theory.ratio)}. Variance 0.02 preserves this idealized scale at every depth.</p><button onClick={() => { setScheme('small'); setSeed('1'); setVariance(.01); setDepth(20); }}>Reset signal investigation</button>
  </NeuralLab>;
}

export function InitializationMomentsLab() {
  const [values, setValues] = useState([-2, -1, 1, 2]), [activation, setActivation] = useState('relu'), [origin, setOrigin] = useState('zero');
  const result = activationMoments(values, activation), maximum = Math.max(1, ...[result.before, result.after].flatMap(row => row.values.map(value => (value - (origin === 'mean' ? row.mean : 0)) ** 2)));
  const edit = (index, value) => setValues(previous => previous.map((entry, i) => i === index ? value : entry));
  return <NeuralLab title="Four values, two different squared distances" id="initialization-moments"><p>Move each value. The same named observation stays in its row before and after the activation. The bars measure squared distance; they are not sliders.</p>
    <div className="neural-controls">{values.map((value, i) => <NeuralNumber key={i} label={`Value ${'ABCD'[i]}`} value={value} onChange={next => edit(i, next)} min={-5} max={5} step={.1} />)}<NeuralSelect label="Activation" value={activation} onChange={setActivation} options={[['relu', 'ReLU'], ['identity', 'Identity · no clipping']]} /><NeuralSelect label="Squared distance from" value={origin} onChange={setOrigin} options={[['zero', 'Zero · second moment'], ['mean', 'Each row’s own mean · variance']]} /></div>
    <div className="neural-two">{[['Before', result.before], ['After', result.after]].map(([label, row]) => <div key={label}><h4>{label}: {row.values.map(value => number(value)).join(', ')}</h4><svg className="init-chart" viewBox="0 0 340 125" role="img" aria-label={`${label} observation positions on minus five to five. Mean ${number(row.mean)}.`}><line x1="30" x2="310" y1="94" y2="94" className="init-axis" /><line x1="170" x2="170" y1="10" y2="97" stroke="#505050" strokeDasharray="3 4" /><line x1={170 + 28 * row.mean} x2={170 + 28 * row.mean} y1="10" y2="97" stroke={amber} strokeDasharray="5 4" />{row.values.map((value, i) => <g key={i}><text x="12" y={24 + 19 * i}>{'ABCD'[i]}</text><circle cx={170 + 28 * value} cy={20 + 19 * i} r="4" fill={pale} /></g>)}{[-5, 0, 5].map(value => <text key={value} x={170 + 28 * value} y="115" textAnchor="middle">{value}</text>)}</svg><p className="init-reading">Dashed amber: mean {number(row.mean)}. Neutral center line: zero.</p>
      {row.values.map((value, i) => { const squared = (value - (origin === 'mean' ? row.mean : 0)) ** 2; return <div className="init-moment-row" key={i}><span>{'ABCD'[i]}: {number(value)}</span><div className="init-distance"><span style={{ width: `${100 * squared / maximum}%` }} /></div><span>{number(squared)}</span></div>; })}
      <div className="init-stats"><div>Mean<strong>{number(row.mean)}</strong></div><div>Mean square<strong>{number(row.secondMoment)}</strong></div><div>Variance<strong>{number(row.variance)}</strong></div></div></div>)}</div>
    <p className="neural-result" data-result="moments">After: q = {number(result.after.secondMoment)}, variance = q − mean² = {number(result.after.variance)}. {activation === 'identity' ? 'Identity leaves every statistic unchanged.' : 'ReLU can move the mean. Half of a symmetric input’s squared mass is removed; its variance need not halve.'}</p><div className="neural-buttons"><button onClick={() => { setValues([-3, -1, 1, 3]); setActivation('relu'); }}>Outer pair ±3</button><button onClick={() => { setValues([-2, -1, 1, 2]); setActivation('relu'); setOrigin('zero'); }}>Reset moment investigation</button></div>
  </NeuralLab>;
}

export function InitializationGeometryLab() {
  const [smallerSquared, setSmallerSquared] = useState(.1), [vector, setVector] = useState([0, 1]), [depth, setDepth] = useState(1), [gate, setGate] = useState(false), [base, setBase] = useState([-1, 1]);
  const smaller = Math.sqrt(smallerSquared);
  const geometry = directionalGeometry(smaller, vector, depth), gated = gatedIdentity(base, vector);
  const gains = gate ? gated.diagonal : geometry.gains, output = gate ? gated.output : geometry.output;
  const radius = Math.max(1.2, ...gains, ...vector.map(Math.abs), ...output.map(Math.abs)) * 1.15, scale = 125 / radius;
  const x = value => 170 + scale * value, y = value => 155 - scale * value;
  const points = Array.from({ length: 129 }, (_, i) => { const angle = i * Math.PI / 64; return `${x(gains[0] * Math.cos(angle))},${y(gains[1] * Math.sin(angle))}`; }).join(' ');
  const gain = Math.hypot(...vector) === 0 ? null : Math.hypot(...output) / Math.hypot(...vector);
  return <NeuralLab title="Average stability can hide a lost direction" id="initialization-geometry"><p>The unit circle becomes an ellipse under a diagonal map. Both axes use the same physical scale. Depth repeats the same map; it is not a random-network claim.</p>
    <div className="neural-controls"><NeuralNumber label="Smaller one-layer gain squared" value={smallerSquared} onChange={setSmallerSquared} min={.0025} max={1} step={.0025} /><NeuralNumber label="Repeated layers" value={depth} onChange={setDepth} min={1} max={8} step={1} integer />{vector.map((value, i) => <NeuralNumber key={i} label={`Perturbation ${i ? 'vertical' : 'horizontal'}`} value={value} onChange={next => setVector(previous => previous.map((entry, j) => i === j ? next : entry))} min={-2} max={2} step={.1} />)}</div>
    <p>The slider edits the squared gain s²; the actual smaller gain is s = √{number(smallerSquared)} = {number(smaller)}. The other squared gain is 2 − s², so their average remains one.</p>
    <label><input type="checkbox" checked={gate} onChange={event => setGate(event.target.checked)} /> Compare the local Jacobian of ReLU(√2 x)</label>
    {gate && <><p>This switches to a separate one-layer counterexample. The ellipse now shows the Jacobian at the base point, applied to a small perturbation; depth and smaller-gain controls belong to the linear comparison.</p><div className="neural-controls">{base.map((value, i) => <NeuralNumber key={i} label={`Base point ${i ? 'vertical' : 'horizontal'}`} value={value} onChange={next => setBase(previous => previous.map((entry, j) => i === j ? next : entry))} min={-2} max={2} step={.1} />)}</div>{gated.atKink && <p>At a zero base coordinate the derivative is not unique. This display uses the PyTorch convention: derivative zero at zero.</p>}</>}
    <svg className="init-chart" viewBox="0 0 340 320" role="img" aria-label={`Equal-axis circle and transformed ellipse; gains ${gains.map(value => number(value)).join(', ')}. Exact vectors below.`}><line className="init-axis" x1="20" x2="320" y1="155" y2="155" /><line className="init-axis" x1="170" x2="170" y1="10" y2="300" /><circle cx="170" cy="155" r={scale} fill="none" stroke={pale} strokeDasharray="4 4" /><polyline points={points} fill="none" stroke={amber} strokeWidth="2" /><line x1="170" y1="155" x2={x(vector[0])} y2={y(vector[1])} stroke={pale} strokeWidth="3" strokeDasharray="4 3" /><circle cx={x(vector[0])} cy={y(vector[1])} r="4" fill={pale} /><line x1="170" y1="155" x2={x(output[0])} y2={y(output[1])} stroke={amber} strokeWidth="3" /><circle cx={x(output[0])} cy={y(output[1])} r="4" fill={amber} /><text x="318" y="175" textAnchor="end">x</text><text x="180" y="16">y</text></svg>
    <p className="init-reading">Neutral dashed: input circle/vector. Amber solid: transformed circle/vector. Visible radius = {number(radius)} in both axes; this view rescales together as depth changes.</p>
    <div className="neural-result" data-result="geometry"><p>One-layer linear gains: {number(geometry.larger)} and {number(smaller)}; average squared gain = {number(geometry.averageSquaredGain)}.</p><p>Displayed map gains: {gains.map(value => number(value)).join(', ')}. Input [{vector.map(value => number(value)).join(', ')}] → output [{output.map(value => number(value)).join(', ')}]. Norm gain: {gain === null ? 'undefined: the zero input has no direction; choose a nonzero perturbation' : number(gain)}.</p></div>
    <div className="neural-buttons"><button onClick={() => { setSmallerSquared(.04); setDepth(5); setVector([0, 1]); setGate(false); }}>Five layers · smaller gain 0.2</button><button onClick={() => { setSmallerSquared(1); setDepth(1); setGate(false); }}>Identity comparison</button><button onClick={() => { setSmallerSquared(.1); setDepth(1); setVector([0, 1]); setBase([-1, 1]); setGate(false); }}>Reset geometry investigation</button></div>
  </NeuralLab>;
}

export function InitializationSpectrumFigure() {
  const values = records.fixtures.gaussian_singular_values, bins = Array.from({ length: 10 }, () => 0);
  values.forEach(value => { bins[Math.min(9, Math.floor(value / .2))]++; });
  return <figure><figcaption>One saved 64 × 64 Gaussian draw, seed 19</figcaption><div className="init-spectrum" role="img" aria-label={`Computed histogram of 64 singular values, ten bins from zero to two. Counts ${bins.join(', ')}.`}>{bins.map((count, i) => <span key={i} style={{ height: `${100 * count / Math.max(...bins)}%` }} title={`${number(i * .2, 1)}–${number((i + 1) * .2, 1)}: ${count}`} />)}</div><p>Ten equal bins from 0 to 2. Minimum {number(Math.min(...values))}; maximum {number(Math.max(...values))}. These are a different 64-dimensional map, not the ellipse above.</p><details><summary>Inspect every singular value and histogram bin</summary><p>{values.map(value => number(value, 7)).join(', ')}</p><NeuralTable caption="Bins recomputed from the actual saved singular values" headers={['Interval [left, right)', 'Count']} rows={bins.map((count, i) => [`${number(i * .2, 1)}–${number((i + 1) * .2, 1)}`, count])} /></details></figure>;
}

export function InitializationSymmetryLab() {
  const initial = { weights: [.2, .2], outgoing: [.3, .3] };
  const [state, setState] = useState(initial), [rate, setRate] = useState(.1), [steps, setSteps] = useState(0);
  const result = symmetryState(state.weights, state.outgoing);
  const setCase = (weights, outgoing) => { setState({ weights, outgoing }); setSteps(0); };
  return <NeuralLab title="Which parameter can learn on the first step?" id="initialization-symmetry"><p>Two tanh units; input 1, target 1, loss ½(output − 1)². Gradients below are evaluated from the current state. A step updates every weight simultaneously.</p><div className="neural-buttons"><button onClick={() => setCase([.2, .2], [.3, .3])}>Identical units</button><button onClick={() => setCase([.1, .3], [.3, .3])}>Distinct units</button><button onClick={() => setCase([.1, .3], [0, 0])}>Distinct features, zero head</button></div>
    <div className="init-matrix-flow">{state.weights.map((weight, i) => <div key={i}><strong>Unit {i + 1}: input 1 → weight {number(weight)} → tanh {number(result.features[i])} → outgoing {number(state.outgoing[i])}</strong><span>Incoming gradient {number(result.hiddenGradient[i])}; outgoing gradient {number(result.headGradient[i])}</span></div>)}</div>
    <NeuralNumber label="Symmetry step size" value={rate} onChange={setRate} min={0} max={1} step={.01} /><p className="neural-result" data-result="symmetry">After {steps} {steps === 1 ? 'step' : 'steps'}: output {number(result.output)}, loss {number(result.loss)}. Incoming difference {number(state.weights[1] - state.weights[0])}. {rate === 0 ? 'A zero step leaves both parameters unchanged.' : 'Equal units receive equal gradients; distinct features can update a zero head before their own first change.'}</p>
    <div className="neural-buttons"><button disabled={steps >= 30} onClick={() => { setState({ weights: state.weights.map((value, i) => value - rate * result.hiddenGradient[i]), outgoing: state.outgoing.map((value, i) => value - rate * result.headGradient[i]) }); setSteps(value => value + 1); }}>Apply one gradient step</button><button onClick={() => { setState(initial); setSteps(0); setRate(.1); }}>Reset symmetry investigation</button></div>{steps >= 30 && <p>This trace is bounded to 30 steps. Reset or choose a new starting case.</p>}
  </NeuralLab>;
}

export function InitializationTrainingLab() {
  const [scheme, setScheme] = useState('small'), [seed, setSeed] = useState('1');
  const current = records.digitFits.find(row => row.scheme === scheme && row.seed === Number(seed));
  return <NeuralLab title="A fitted training set can still give poor probabilities" id="initialization-training"><p>Recorded 64 → 32 → 32 → 32 → 32 → 10 digit model; 280 training / 120 validation rows. Adam 0.003, 300 full-batch updates. This is not the twenty-layer Gaussian probe.</p><div className="init-pixels">{records.specimens.map(sample => <figure key={sample.id}><div className="neural-pixels" role="img" aria-label={`Actual handwritten digit ${sample.digit}, source row ${sample.id}`}>{sample.pixels.map((pixel, i) => <span key={i} style={{ '--pixel': `rgb(${255 * pixel / 16} ${255 * pixel / 16} ${255 * pixel / 16})` }} />)}</div><figcaption>Digit {sample.digit}<br />Source {sample.id}</figcaption></figure>)}</div><div className="neural-controls"><NeuralSelect label="Training initialization" value={scheme} onChange={setScheme} options={initializationSchemes} /><NeuralSelect label="Training seed" value={seed} onChange={setSeed} options={seedOptions} /></div>
    <Plot title="Recorded objective checkpoints" xLabel="Full-batch updates" yLabel="Cross-entropy" logarithmic xDomain={[0, 300]} series={['train', 'validation'].map(key => ({ name: key === 'train' ? 'Training' : 'Validation', values: current.trace.map(row => [row.step, row[key].ce]) }))} />
    <p>Markers are actual recorded checkpoints. Straight connections show their order; the intervening trajectory was not recorded.</p><p className="neural-result" data-result="training">Final training CE {number(current.trace.at(-1).train.ce, 6)}; validation CE {number(current.trace.at(-1).validation.ce, 6)}; validation correct {current.trace.at(-1).validation.correct}/120. Validation is development evidence, not a final test estimate.</p>
    <NeuralTable caption="All recorded checkpoints for the current selection" headers={['Update', 'Training CE', 'Validation CE', 'Correct / 120']} rows={current.trace.map(row => [row.step, number(row.train.ce, 6), number(row.validation.ce, 6), row.validation.correct])} /><button onClick={() => { setScheme('small'); setSeed('1'); }}>Reset training comparison</button>
  </NeuralLab>;
}

export function InitializationWidthLab() {
  const [width, setWidth] = useState(128), [rate, setRate] = useState(.003), [mode, setMode] = useState('mu'), [recordRate, setRecordRate] = useState('.01'), [seed, setSeed] = useState('1'), [step, setStep] = useState('0'), [coordinate, setCoordinate] = useState('4');
  const config = widthConfiguration(width, rate, mode), names = ['First preactivation', 'First ReLU', 'Second preactivation', 'Second ReLU', 'Output logits'];
  const selectedRecords = records.widthFits.filter(row => row.mode === mode && row.seed === Number(seed) && row.rate === Number(recordRate));
  const means = ['standard', 'mu'].flatMap(parameterization => [32, 64, 128].map(n => [parameterization, n, ...[.001, .003, .01].map(lr => records.widthFits.filter(row => row.mode === parameterization && row.width === n && row.rate === lr).reduce((sum, row) => sum + row.trace.at(-1).validation.ce, 0) / 3)]));
  return <NeuralLab title="Width changes the forward path and the update together" id="initialization-width"><p>Exact arithmetic for the restricted bias-free model. Base width stays 32; input/output sizes stay 64/10. These controls calculate a recipe; they do not run training.</p><div className="neural-controls"><NeuralNumber label="Target hidden width" value={width} onChange={setWidth} min={16} max={512} step={1} integer /><NeuralNumber label="Base Adam learning rate" value={rate} onChange={setRate} min={0} max={.02} step={.0001} /><NeuralSelect label="Width parametrization" value={mode} onChange={setMode} options={[['mu', 'μP · matching forward + Adam rules'], ['standard', 'Standard comparison']]} /></div>
    <div className="init-matrix-flow"><div><strong>Input matrix: {width} × 64 → ReLU</strong>Fixed input axis 64; changing hidden axis {width}. Std {number(config.inputStd)}; Adam rate {number(config.inputRate)}.</div><div><strong>Hidden matrix: {width} × {width} → ReLU</strong>Two changing axes. Std {number(config.hiddenStd)}; Adam rate {number(config.hiddenRate)}.</div><div><strong>Readout input × {number(config.readoutMultiplier)} → raw matrix 10 × {width}</strong>Fixed output axis 10. Raw std {number(config.readoutStd)}; Adam rate {number(config.readoutRate)}.</div></div>
    <p className="neural-result" data-result="width">m = {width}/32 = {number(config.multiplier)}. Hidden rate = {number(config.hiddenRate)}; readout input multiplier = {number(config.readoutMultiplier)}. {width === 32 ? 'At base width both full procedures coincide.' : mode === 'mu' ? `Omitting the forward division would multiply the same raw readout’s output by ${number(config.multiplier)}. This is an algebraic contrast, not new accuracy evidence.` : 'Standard comparison uses the same rate in all groups and no forward division.'}</p>
    <h4>Inspect the saved coordinate experiment separately</h4><p>Only widths 32, 64 and 128 were trained. The plot below uses its own recorded rate, seed and checkpoint. Editing recipe width/rate above does not relabel these measurements. The same parametrization selector applies to both panels.</p>
    <div className="neural-controls"><NeuralSelect label="Recorded base rate" value={recordRate} onChange={setRecordRate} options={[['.001', '0.001'], ['.003', '0.003'], ['.01', '0.01']]} /><NeuralSelect label="Coordinate seed" value={seed} onChange={setSeed} options={seedOptions} /><NeuralSelect label="Coordinate checkpoint" value={step} onChange={setStep} options={[0, 1, 2, 5, 150].map(value => [String(value), `Step ${value}`])} /><NeuralSelect label="Measured coordinate" value={coordinate} onChange={setCoordinate} options={names.map((name, i) => [String(i), name])} /></div>
    <Plot title={`${names[Number(coordinate)]} · mean absolute value on the same 32 training rows`} xLabel="Recorded hidden width" yLabel="Mean absolute coordinate" xDomain={[32, 128]} series={[{ name: `${mode === 'mu' ? 'μP' : 'Standard'} · seed ${seed} · rate ${recordRate} · step ${step}`, values: selectedRecords.map(row => [row.width, row.trace.find(value => value.step === Number(step)).coordinate_mean_abs[Number(coordinate)]]) }]} />
    <NeuralTable caption="Exact selected coordinates" headers={['Width', 'Mean absolute value']} rows={selectedRecords.map(row => [row.width, number(row.trace.find(value => value.step === Number(step)).coordinate_mean_abs[Number(coordinate)], 7)])} />
    <p>A μP random readout can shrink with width at step zero. Compare the same coordinate at the same update; neither perfect flatness nor one final validation grid is a theorem about all architectures.</p>
    <details><summary>All three-seed validation means and within-row minima</summary><NeuralTable caption="Final validation CE; only these three candidate rates were tested" headers={['Parametrization', 'Width', 'LR .001', 'LR .003', 'LR .01']} rows={means.map(row => [...row.slice(0, 2), ...row.slice(2).map(value => <span key={value}>{number(value, 6)}{value === Math.min(...row.slice(2)) ? ' · row minimum' : ''}</span>)])} /></details>
    <button onClick={() => { setWidth(128); setRate(.003); setMode('mu'); setRecordRate('.01'); setSeed('1'); setStep('0'); setCoordinate('4'); }}>Reset width investigation</button>
  </NeuralLab>;
}

export function InitializationPrecisionFigure() {
  const precision = records.fixtures.precision;
  return <aside><h4>Same requested normal scale, different retained distribution</h4><p>Distribution illustration: the central amber band below is the retained interval ±2σ. The dashed normal tail continues outside it; rejected draws are resampled, not moved to the boundary.</p><svg className="init-chart" viewBox="0 0 340 150" role="img" aria-label="Schematic normal density; retained interval from minus two to two standard deviations, tails outside."><rect x="100" width="140" y="18" height="100" fill="#e2b55a" opacity=".12" /><path d="M30 115 C68 115 81 107 102 93 C126 75 142 23 170 20 C198 23 214 75 238 93 C259 107 272 115 310 115" fill="none" stroke={pale} strokeWidth="2" strokeDasharray="4 3" /><line className="init-axis" x1="30" x2="310" y1="118" y2="118" />{[[-2, 100], [0, 170], [2, 240]].map(([value, position]) => <text key={value} x={position} y="140" textAnchor="middle">{value}σ</text>)}</svg><NeuralTable caption="Actual 100,000-value samples; requested std 0.02" headers={['Absolute bounds', 'Measured std', 'Maximum absolute sample']} rows={Object.values(records.fixtures.truncation).map(row => [row.bounds.join(' to '), number(row.std, 8), number(row.max_abs, 8)])} />
    <NeuralTable caption="Actual precision conversion, back to float32 for display" headers={['Original float32', 'BF16 → float32', 'Float16 → float32']} rows={precision.float32.map((value, i) => [value.toExponential(7), precision.bfloat16_back_to_float32[i].toExponential(7), precision.float16_back_to_float32[i].toExponential(7)])} /><p>Representation of a small value is separate from losing its addition to a much larger weight.</p></aside>;
}
