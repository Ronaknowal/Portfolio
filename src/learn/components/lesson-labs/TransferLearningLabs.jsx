import { useEffect, useId, useRef, useState } from 'react';
import { CodeBlock } from '../content';
import experiment from '../../data/transfer-learning-experiment.json';
import specimens from '../../data/transfer-learning-specimens.json';
import { TRANSFER_METHODS, TRANSFER_LABELS, finiteIn, loraFixture, loraStep, batchNormFixture, batchNormForward, adapterBudget, selectTransferCandidate } from '../../data/transfer-learning-model.js';
import programUrl from '../../assets/transfer-learning/transfer-experiments.py?url';
import dataUrl from '../../assets/transfer-learning/digits-400.csv?url';
import provenanceUrl from '../../assets/transfer-learning/data-provenance.md?url';
import './transfer-learning.css';

export { programUrl, dataUrl, provenanceUrl };
const number = (value, digits = 4) => Number.isFinite(value) ? Number(value.toFixed(digits)).toLocaleString('en-US', { maximumFractionDigits: digits }) : 'Unavailable';
const vector = values => `[${values.map(value => number(value)).join(', ')}]`;
const matrix = rows => rows.map(vector).join(' · ');

// Match SVG user units to CSS pixels so axis labels remain readable on phones.
function usePlotWidth() {
  const container = useRef(null);
  const [width, setWidth] = useState(490);
  useEffect(() => {
    const observer = new ResizeObserver(entries => {
      const next = entries[0]?.contentRect.width;
      if (next > 0) setWidth(next);
    });
    if (container.current) observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  return [container, width];
}

function Numeric({ label, value, onChange, min, max, integer = false, slider = false }) {
  const id = useId();
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const valid = text.trim() !== '' && finiteIn(Number(text), min, max, integer);
  return <div className="transfer-field">
    <label htmlFor={id}>{label}</label>
    <input id={id} type="number" min={min} max={max} step={integer ? 1 : 'any'} value={text} aria-invalid={!valid} aria-describedby={!valid ? `${id}-error` : undefined} onChange={event => {
      const next = event.target.value; setText(next);
      if (next.trim() !== '' && finiteIn(Number(next), min, max, integer)) onChange(Number(next));
    }} />
    {slider && <input type="range" aria-label={`${label} slider`} min={min} max={max} step={integer ? 1 : 'any'} value={value} onChange={event => { setText(event.target.value); onChange(Number(event.target.value)); }} />}
    {!valid && <span id={`${id}-error`} className="transfer-error" role="alert">Enter {integer ? 'an integer' : 'a finite number'} from {min} to {max}. Results still use {number(value)}.</span>}
  </div>;
}

function Toggle({ children, checked, onChange }) {
  return <label className="transfer-toggle"><input type="checkbox" checked={checked} onChange={event => onChange(event.target.checked)} /> <span>{children}</span></label>;
}

function Lab({ id, title, kind = 'Live exploration', children }) {
  return <section className="transfer-lab" aria-labelledby={`${id}-title`} data-testid={id}>
    <p className="transfer-eyebrow">{kind}</p><h3 id={`${id}-title`}>{title}</h3>{children}
  </section>;
}

function PixelDigit({ row }) {
  return <figure className="transfer-digit"><svg className="transfer-pixels" viewBox="0 0 80 80" role="img" aria-label={`Recorded digit ${row.digit}; source ID ${row.source_id}; 8 by 8 pixels, intensity 0 to 16`}>
    {row.pixels.map((value, i) => <rect key={i} x={10 * (i % 8)} y={10 * Math.floor(i / 8)} width="10" height="10" fill={`rgb(${Math.round(16 + value * 13.5)},${Math.round(16 + value * 10.3)},${Math.round(16 + value * 4.6)})`} />)}
  </svg><figcaption>Digit {row.digit} · ID {row.source_id}<br />Pixel intensity: 0–16</figcaption></figure>;
}

export function TransferReuseFigure() {
  const [sourceDigit, setSourceDigit] = useState(0);
  const [targetDigit, setTargetDigit] = useState(5);
  const [stage, setStage] = useState(1);
  const [oldHead, setOldHead] = useState(false);
  const shapes = [ ['Input', '64 values; row-major 8×8 pixels / 16. No trainable input weights.'], ['Lower tanh', '64 → 32. Weight shape 32 × 64; bias shape 32.'], ['Upper tanh', '32 → 16. Weight shape 16 × 32; bias shape 16.'], ['Head', '16 → 5 logits. Weight shape 5 × 16; bias shape 5.'] ];
  return <Lab id="transfer-reuse" title="Reuse the instrument; check what its outputs mean" kind="Recorded pixels · explanatory architecture">
    <p>Choose real training specimens and inspect a stage. The boxes show dimensions and ownership; they do not show computed activations or model predictions.</p>
    <div className="transfer-columns">
      <div><label className="transfer-select">Source training digit<select value={sourceDigit} onChange={event => setSourceDigit(Number(event.target.value))}>{[0, 1, 2, 3, 4].map(value => <option key={value}>{value}</option>)}</select></label><PixelDigit row={specimens.training.find(row => row.digit === sourceDigit)} /><p>Source head meanings: <strong>0 · 1 · 2 · 3 · 4</strong></p></div>
      <div><label className="transfer-select">Target training digit<select value={targetDigit} onChange={event => setTargetDigit(Number(event.target.value))}>{[5, 6, 7, 8, 9].map(value => <option key={value}>{value}</option>)}</select></label><PixelDigit row={specimens.training.find(row => row.digit === targetDigit)} /><p>Target needs meanings: <strong>5 · 6 · 7 · 8 · 9</strong></p></div>
    </div>
    <div className="transfer-stage-route" aria-label="Inspect architecture stages">{shapes.map(([label], i) => <button key={label} aria-pressed={stage === i} onClick={() => setStage(i)}>{label}<span>{[64, 32, 16, 5][i]} values</span></button>)}</div>
    <p className="transfer-readout">{shapes[stage][1]} Weight dimensions are <strong>outputs × inputs</strong>. These are single-specimen shapes; a batch adds a leading N dimension.</p>
    <div className="transfer-flow"><div>Source backbone<br /><strong>64 → tanh 32 → tanh 16</strong></div><span aria-hidden="true">↓</span><div>Copy learned backbone weights<br /><strong>64 → tanh 32 → tanh 16</strong></div><span aria-hidden="true">↓</span><div>{oldHead ? 'Original source head' : 'New target head'}<br /><strong>{oldHead ? '0 · 1 · 2 · 3 · 4' : '5 · 6 · 7 · 8 · 9'}</strong></div></div>
    <Toggle checked={oldHead} onChange={setOldHead}>Attach the original head to target features</Toggle>
    <p data-testid="transfer-head-meaning">{oldHead ? 'The five scores still mean digits 0–4. A matching tensor shape cannot turn them into scores for digits 5–9.' : 'A new head learns the meanings 5–9. The original source head is retained separately to measure source-task retention.'} Changing a displayed specimen does not change either head’s ownership.</p>
  </Lab>;
}

export function TransferFreezeLab() {
  const [state, setState] = useState(batchNormFixture);
  const [epoch, setEpoch] = useState(0);
  const [forwards, setForwards] = useState(0);
  const result = batchNormForward(state);
  const edit = patch => setState(current => ({ ...current, ...patch }));
  const reset = () => { setState(batchNormFixture()); setForwards(0); setEpoch(value => value + 1); };
  return <Lab id="transfer-freeze" title="A frozen weight and a moving running mean are different things">
    <p>Exact one-channel BatchNorm calculation, ε = 0.00001. Affine output = γ × normalized input + β. Controls immediately update the current forward calculation. Stored buffers change only when you apply a forward.</p>
    <p className="transfer-note">Pinned starting case: input [1, 3], momentum 0.1, mean 0, variance 1, γ = 1 and β = 0; frozen affine parameters, no optimizer ownership, no gradient recording, training mode.</p>
    <div key={epoch} className="transfer-columns"><Numeric label="Batch value 1" value={state.input[0]} min={-20} max={20} onChange={value => edit({ input: [value, state.input[1]] })} /><Numeric label="Batch value 2" value={state.input[1]} min={-20} max={20} onChange={value => edit({ input: [state.input[0], value] })} /><Numeric label="Running-statistic momentum" value={state.momentum} min={0} max={1} slider onChange={value => edit({ momentum: value })} /></div>
    <div className="transfer-control-lanes">
      <Toggle checked={state.trainable} onChange={value => edit({ trainable: value })}>Affine parameters require gradients</Toggle>
      <Toggle checked={state.optimizer} onChange={value => edit({ optimizer: value })}>Optimizer owns the affine parameters</Toggle>
      <Toggle checked={state.recording} onChange={value => edit({ recording: value })}>Record gradients (input also requires gradients)</Toggle>
      <Toggle checked={state.training} onChange={value => edit({ training: value })}>Training mode (unchecked = evaluation)</Toggle>
    </div>
    <div className="transfer-columns">
      <div className="transfer-readout"><h4>Which statistics?</h4><p>Batch mean {number(result.batchMean)}; biased variance {number(result.batchVariance)}; unbiased variance {number(result.unbiasedVariance)}.</p><p>Normalization uses <strong>{state.training ? 'this batch' : 'stored buffers'}</strong>: mean {number(result.usedMean)}, variance {number(result.usedVariance)}.</p><p data-testid="transfer-bn-output">Current output {vector(result.output)}</p></div>
      <div className="transfer-readout"><h4>Stored buffers → next forward</h4><p data-testid="transfer-bn-state">Running mean: {number(state.mean)} → {number(result.nextMean)}<br />Running variance: {number(state.variance)} → {number(result.nextVariance)}</p><p>{state.training ? 'Training updates running variance with the unbiased batch variance. Gradient recording does not govern this update.' : 'Evaluation reads the stored buffers and leaves them unchanged.'}</p><button onClick={() => { edit({ mean: result.nextMean, variance: result.nextVariance }); setForwards(value => value + 1); }}>Apply one forward state transition</button><p>{forwards} forwards applied. The next preview uses the now-current buffers.</p></div>
    </div>
    <div className="transfer-readout"><h4>Parameter values and derivatives</h4><p>Stored γ = {number(state.gamma)}; β = {number(state.beta)}. For the separate teaching loss, mean squared output against zero:</p><p data-testid="transfer-bn-gradients">Affine gradients: {result.gradGamma === null ? 'absent' : `γ ${number(result.gradGamma, 6)}, β ${number(result.gradBeta, 6)}`}. Input gradient: {result.inputGradient ? vector(result.inputGradient) : 'not recorded'}.</p><p>{!state.recording ? 'no_grad: outputs and running-state updates still exist; there is no backward graph.' : !state.trainable ? 'Frozen affine values have no parameter gradients. A derivative through them can still reach an upstream input.' : !state.optimizer ? 'Affine gradients exist, but this optimizer does not own the parameters, so it cannot update them.' : 'Both affine gradients and optimizer membership are present; a fresh SGD step can change γ and β.'}</p><button disabled={!result.canUpdate || ![result.nextGamma, result.nextBeta].every(value => Number.isFinite(value) && Math.abs(value) <= 100)} onClick={() => edit({ gamma: result.nextGamma, beta: result.nextBeta })}>Apply affine SGD step (rate 0.1)</button><p>Gradients here are freshly calculated analytically, with no stale-gradient storage. Steps outside |γ|, |β| ≤ 100 are disabled; reset to restore the bounded example.</p></div>
    <p className="transfer-note">Frozen matrix connection: with W = I₂, x = [2, 1] and mean squared loss to zero, ∂L/∂x = [2, 1] while the frozen weight has no stored gradient. Disabling the whole graph would also cut this useful input derivative.</p>
    <button onClick={reset}>Reset freeze experiment</button>
  </Lab>;
}

const partitionNames = { source_train: 'Source training', source_holdout: 'Source holdout', target_train: 'Target training', target_validation: 'Target validation', target_test: 'Target test' };
const partitionPurposes = { source_train: 'Learn backbone and original head', source_holdout: 'Describe retention; never select target candidate', target_train: 'Fit all adaptation candidates', target_validation: 'Choose the exact seed-1 artifact', target_test: 'Already reported once: selected artifact only' };

export function TransferPartitionsFigure() {
  const [open, setOpen] = useState(false);
  return <Lab id="transfer-partitions" title="Five bins; five distinct jobs" kind="Recorded fixed partitions">
    <div className="transfer-partitions">{Object.entries(experiment.splits_source_ids).map(([key, ids]) => <div key={key}><h4>{partitionNames[key]}</h4><strong>{ids.length} specimens</strong><p>Digits {key.startsWith('source') ? '0–4' : '5–9'}</p><p>{partitionPurposes[key]}</p></div>)}</div>
    <p>Source training → learned backbone. Target training → six candidates. Target validation → one choice. Exact selected artifact → target test report. Source holdout → separate retention report.</p>
    <p>All 400 IDs belong to exactly one bin. These are fixed row blocks, not random writer groups or the official UCI benchmark. The test is already observed; browsing this saved report does not make a new untouched test.</p>
    <details onToggle={event => setOpen(event.currentTarget.open)}><summary>Inspect all 400 specimen assignments</summary>{open && <div className="transfer-table-scroll" role="region" tabIndex={0} aria-label="All specimen IDs and partitions"><table><caption>Source IDs in saved experiment order</caption><thead><tr><th>Partition</th><th>Count</th><th>Source IDs</th></tr></thead><tbody>{Object.entries(experiment.splits_source_ids).map(([key, ids]) => <tr key={key}><th scope="row">{partitionNames[key]}</th><td>{ids.length}</td><td className="transfer-id-list">{ids.join(', ')}</td></tr>)}</tbody></table></div>}</details>
    <p><strong>Why does target digit 5 have class index 0?</strong> The program computes digit − 5 for target labels. Index 0 of the new head means 5; index 0 of the retained source head means 0.</p>
  </Lab>;
}

export function TransferProgram() {
  const [open, setOpen] = useState(false);
  const [code, setCode] = useState(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!open || code) return;
    let active = true; setFailed(false);
    import('../../assets/transfer-learning/transfer-experiments.py?raw').then(module => { if (active) setCode(module.default); }).catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, [open, code, attempt]);
  return <section className="transfer-downloads" aria-label="Complete reproducible transfer experiment">
    <h3>Run the complete experiment</h3><p><a href={programUrl} download="transfer-experiments.py">Download complete Python program</a> · <a href={dataUrl} download="digits-400.csv">Download all 400 digit rows</a> · <a href={provenanceUrl} download="data-provenance.md">Dataset provenance and license</a></p>
    <p>Keep the program and CSV together. The program performs all 18 target fits, all source fits, the predeclared selection and the one test report. Browser labs use saved observations or exact small arithmetic; they do not run these fits.</p>
    <details onToggle={event => setOpen(event.currentTarget.open)}><summary>Read the full executable Python program</summary>{open && (code ? <div className="transfer-code" role="region" tabIndex={0} aria-label="Complete Python program; scroll horizontally when needed"><CodeBlock language="python">{code}</CodeBlock></div> : failed ? <p role="alert">The in-page program could not load. The download remains available. <button onClick={() => setAttempt(value => value + 1)}>Retry code view</button></p> : <p role="status">Loading full program…</p>)}</details>
  </section>;
}

function MatrixEditor({ label, values, onChange }) {
  return <fieldset className="transfer-matrix"><legend>{label} · {values.length} × {values[0].length}</legend><div className="transfer-matrix-cells" style={{ gridTemplateColumns: `repeat(${values[0].length}, minmax(0, 1fr))` }}>{values.flatMap((row, i) => row.map((value, j) => <Numeric key={`${i}-${j}`} label={`${label} row ${i + 1}, column ${j + 1}`} value={value} min={-5} max={5} onChange={next => onChange(values.map((source, rowIndex) => source.map((cell, columnIndex) => rowIndex === i && columnIndex === j ? next : cell)))} />))}</div></fieldset>;
}

export function TransferLoraLab() {
  const [state, setState] = useState(loraFixture);
  const [epoch, setEpoch] = useState(0);
  const [steps, setSteps] = useState(0);
  const result = loraStep(state);
  const edit = patch => setState(current => ({ ...current, ...patch }));
  const reset = () => { setState(loraFixture()); setSteps(0); setEpoch(value => value + 1); };
  const hasA = result.before.gradA.flat().some(value => Math.abs(value) > 1e-10);
  const hasB = result.before.gradB.flat().some(value => Math.abs(value) > 1e-10);
  return <Lab id="transfer-lora" title="Build a low-rank correction and watch one genuine update">
    <p>W stays I₂. Current output and one-step preview are visible as you edit. Loss is the mean of the two squared errors; all factor gradients use the same pre-update state.</p>
    <p className="transfer-note">Pinned baseline: rank 1, A = [1, −1], B = [0, 0]ᵀ, x = [2, 1], target = [0, 0], α = 1, rate = 0.1. Output [2, 1], loss 2.5; one step gives [1.8, 0.9], loss 2.025.</p>
    <div className="transfer-lora-route"><div>Direct path: x → frozen W = I₂ → x</div><div>Added path: x → A (2 → {state.rank}) → B ({state.rank} → 2) → × {number(result.before.scale)}</div><strong>Add both paths → y = {vector(result.before.output)}</strong></div>
    <label className="transfer-select">Factor rank<select value={state.rank} onChange={event => { setState(loraFixture(Number(event.target.value))); setSteps(0); setEpoch(value => value + 1); }}><option value="1">1 measurement</option><option value="2">2 measurements</option></select></label><p className="transfer-note">Changing rank restores declared factors for that shape and α = rank, so the starting scale stays 1.</p>
    <div key={epoch}>
      <div className="transfer-columns"><MatrixEditor label="A" values={state.A} onChange={value => edit({ A: value })} /><MatrixEditor label="B" values={state.B} onChange={value => edit({ B: value })} /></div>
      <div className="transfer-columns"><MatrixEditor label="Input x" values={[state.x]} onChange={value => edit({ x: value[0] })} /><MatrixEditor label="Target" values={[state.target]} onChange={value => edit({ target: value[0] })} /></div>
      <div className="transfer-columns"><Numeric label="LoRA alpha" value={state.alpha} min={0} max={4} slider onChange={value => edit({ alpha: value })} /><Numeric label="SGD learning rate" value={state.rate} min={0} max={0.5} slider onChange={value => edit({ rate: value })} /></div>
    </div>
    <div className="transfer-columns">
      <div className="transfer-readout"><h4>Current forward and gradients</h4><p>Scale α/r = {number(result.before.scale)}; Ax = {vector(result.before.bottleneck)}<br />Correction = {vector(result.before.correction)}</p><p data-testid="transfer-lora-current">Output {vector(result.before.output)} · loss {number(result.before.loss, 6)}</p><p data-testid="transfer-lora-gradients">∂L/∂A: {matrix(result.before.gradA)}<br />∂L/∂B: {matrix(result.before.gradB)}<br />∂L/∂x: {vector(result.before.inputGradient)}</p><p>{hasA && hasB ? 'Both factors have a gradient at this state.' : hasA ? 'Only A has a nonzero factor gradient here.' : hasB ? 'Only B has a nonzero factor gradient. Zero B blocks the current gradient into A.' : result.before.loss > 1e-6 ? 'Neither factor receives a gradient, even though loss is positive. Zero signal through the factors is not optimization success.' : 'Both factor gradients are zero and the current loss is zero.'}</p></div>
      <div className="transfer-readout"><h4>Exact one-SGD-step preview</h4><p>A next: {matrix(result.next.A)}<br />B next: {matrix(result.next.B)}</p><p data-testid="transfer-lora-next">Output next {vector(result.after.output)} · loss next {number(result.after.loss, 6)}</p><p>ΔA = {matrix(result.next.A.map((row, i) => row.map((value, j) => value - state.A[i][j])))}<br />ΔB = {matrix(result.next.B.map((row, i) => row.map((value, j) => value - state.B[i][j])))}</p><button disabled={!result.withinEditorBounds || steps >= 20} onClick={() => { setState(result.next); setSteps(value => value + 1); }}>Apply this SGD step</button><p>{steps}/20 steps applied. {result.withinEditorBounds ? 'Current values become the previewed state; the next preview is recomputed.' : 'This step leaves the matrix editor’s [−5, 5] range. Lower the learning rate before applying it.'}</p></div>
    </div>
    <p data-testid="transfer-lora-merge">Merged W + sBA: {matrix(result.before.merged)}. Merged output {vector(result.before.mergedOutput)}. Maximum absolute path difference {result.before.mergeError.toExponential(2)} (tolerance 10⁻⁶ for this small browser calculation).</p>
    <p>Try x = [1, 3] for a changed gradient direction, then x = [1, 1] with the default factors: Ax becomes zero while loss stays positive. With both factors zero, neither can start learning here. A zero rate or zero scale is another useful unchanged-output case. These are mechanism comparisons, not a claim that every step improves loss.</p>
    <button onClick={reset}>Reset LoRA fixture</button>
  </Lab>;
}

export function TransferAdapterBudget() {
  const [dimension, setDimension] = useState(16);
  const [width, setWidth] = useState(4);
  const [head, setHead] = useState(true);
  const [unit, setUnit] = useState('bytes');
  const [loraD, setLoraD] = useState(4);
  const [loraK, setLoraK] = useState(6);
  const [rank, setRank] = useState(2);
  const [epoch, setEpoch] = useState(0);
  const budget = adapterBudget(dimension, width, head);
  const factorCount = rank * (loraD + loraK), fullCount = loraD * loraK;
  const reset = () => { setDimension(16); setWidth(4); setHead(true); setUnit('bytes'); setLoraD(4); setLoraK(6); setRank(2); setEpoch(value => value + 1); };
  return <Lab id="transfer-budget" title="A feature detour and an honest parameter budget">
    <div className="transfer-lora-route"><div>Direct h: {dimension} values ─────────→ addition</div><div>Detour: {dimension} → down {width} → tanh → up {dimension} → addition</div><strong>At zero-up initialization: correction = 0, so h′ = h.</strong></div>
    <p>The up weight and bias start at zero; the down path starts random. This identity is exact for any feature vector at initialization. The nonlinear detour generally cannot be merged into one fixed linear weight.</p>
    <div key={epoch}>
      <div className="transfer-columns"><Numeric label="Adapter feature dimension d" value={dimension} min={2} max={1024} integer onChange={setDimension} /><Numeric label="Adapter bottleneck b" value={width} min={1} max={64} integer onChange={setWidth} /></div>
      <Toggle checked={head} onChange={setHead}>Include the new five-output head</Toggle>
      <p data-testid="transfer-adapter-count">Down: {number(budget.down)} · Up: {number(budget.up)} · Adapter: {number(budget.adapter)} · Head: {number(budget.head)}{head ? ' included' : ' excluded'}<br /><strong>Total trainable values: {number(budget.count)}</strong></p>
      <label className="transfer-select">Memory display unit<select value={unit} onChange={event => setUnit(event.target.value)}><option value="bytes">Bytes</option><option value="GB">Decimal GB</option><option value="GiB">Binary GiB</option></select></label>
      <p data-testid="transfer-adapter-bytes">FP32 gradients + two FP32 moments: {unit === 'bytes' ? number(budget.bytes) : (budget.bytes / (unit === 'GB' ? 1e9 : 2 ** 30)).toFixed(9)} {unit} = {number(budget.bytes)} exact bytes.</p>
      <p className="transfer-note">12 bytes per trainable value. Excludes weights, master copies, activations, allocator overhead and all other state. This is arithmetic, not measured peak memory or speed. Changing the unit does not change the stored values.</p>
      <h4>Does low rank actually save parameters at this shape?</h4>
      <div className="transfer-columns"><Numeric label="LoRA output width d" value={loraD} min={2} max={1024} integer onChange={setLoraD} /><Numeric label="LoRA input width k" value={loraK} min={2} max={1024} integer onChange={setLoraK} /><Numeric label="LoRA factor width r" value={rank} min={1} max={64} integer onChange={setRank} /></div>
    </div>
    <p data-testid="transfer-lora-count">Full weights: d × k = {number(fullCount)}. Factors: r(d + k) = {number(factorCount)}. <strong>{factorCount < fullCount ? `${number(fullCount - factorCount)} fewer factor values.` : factorCount > fullCount ? `${number(factorCount - fullCount)} more factor values; this shape does not save parameters.` : 'Equal counts; no parameter saving.'}</strong> Biases and the head are excluded in this comparison. The product’s rank is at most min(r, d, k).</p>
    <button onClick={reset}>Reset parameter budget</button>
  </Lab>;
}

function TransferTrace({ record }) {
  const [container, width] = usePlotWidth();
  const ceiling = Math.max(...record.trace.flatMap(row => [row.train.ce, row.validation.ce])) * 1.05;
  const right = width - 16;
  const px = value => 48 + value / 300 * (right - 48);
  const py = value => 190 - value / ceiling * 160;
  return <figure ref={container} className="transfer-trace"><figcaption><strong>{TRANSFER_LABELS[record.method]} · seed {record.seed}</strong><br />Mean CE (nats) against full-batch updates. Lines connect only five saved samples.</figcaption>
    <svg className="transfer-plot" viewBox={`0 0 ${width} 230`} role="img" aria-label={`Saved training and validation cross-entropy for ${TRANSFER_LABELS[record.method]}, seed ${record.seed}; exact values are in the table below`}>
      {[0, ceiling / 2, ceiling].map(value => <g key={value}><line x1="48" x2={right} y1={py(value)} y2={py(value)} stroke="#35443b" /><text x="41" y={py(value) + 4} textAnchor="end">{number(value, 2)}</text></g>)}
      {[0, 100, 300].map(value => <text key={value} x={px(value)} y="211" textAnchor="middle">{value}</text>)}
      {['train', 'validation'].map((split, i) => <g key={split}><polyline fill="none" stroke={i ? '#e2b55a' : '#9fc8b4'} strokeWidth="2" strokeDasharray={i ? '5 4' : undefined} points={record.trace.map(row => `${px(row.step)},${py(row[split].ce)}`).join(' ')} />{record.trace.map(row => <circle key={row.step} cx={px(row.step)} cy={py(row[split].ce)} r="3" fill={i ? '#e2b55a' : '#9fc8b4'} />)}</g>)}
    </svg><p>Solid green = training (40 rows); dashed amber = validation (60 rows).</p>
    <details><summary>Read all saved trace values</summary><div className="transfer-table-scroll" role="region" tabIndex={0} aria-label="Exact recorded trace"><table><thead><tr><th>Update</th><th>Train CE</th><th>Train correct / 40</th><th>Validation CE</th><th>Validation correct / 60</th></tr></thead><tbody>{record.trace.map(row => <tr key={row.step}><th scope="row">{row.step}</th><td>{number(row.train.ce, 6)}</td><td>{row.train.correct}</td><td>{number(row.validation.ce, 6)}</td><td>{row.validation.correct}</td></tr>)}</tbody></table></div></details>
  </figure>;
}

function Retention({ record }) {
  return <div className="transfer-readout"><h4>Original source head: retention on 50 source holdout rows</h4>{record.source_after ? <>
    {[['Before adaptation', record.source_before], ['After adaptation', record.source_after]].map(([label, metric]) => <div key={label} className="transfer-retention"><span>{label}: <strong>{metric.correct}/50</strong> · CE {number(metric.ce, 6)} nats</span><div className="transfer-bar" aria-hidden="true"><i style={{ width: `${2 * metric.correct}%` }} /></div></div>)}
    <p>These are recorded bars, not controls. {record.method === 'probe' ? 'The fixed backbone and original head preserve both accuracy and loss exactly.' : record.method === 'lora2' ? 'The active low-rank paths change the function even though base weights remain frozen.' : 'Identical accuracy counts do not imply identical probabilities; compare the CE too.'}</p>
  </> : <p>Not applicable: this scratch candidate uses the saved original random backbone, not the learned source representation. Its target scores do not supply an original-source retention result.</p>}</div>;
}

export function TransferEvidenceLab() {
  const [seed, setSeed] = useState(1);
  const [method, setMethod] = useState('scratch');
  const [budget, setBudget] = useState(400);
  const [epoch, setEpoch] = useState(0);
  const records = TRANSFER_METHODS.map(key => experiment.runs.find(row => row.seed === seed && row.method === key));
  const selected = records.find(row => row.method === method);
  const hypothetical = selectTransferCandidate(experiment.runs, budget, 1);
  return <Lab id="transfer-evidence" title="Inspect the actual evidence; keep hypothetical choices separate" kind="Saved measurements · no browser fitting">
    <div className="transfer-test-report" data-testid="transfer-original-test"><h4>Original predeclared selection — unchanged by these controls</h4><p>Seed 1 · scratch · minimum final validation CE 0.123143, 58/60 correct. The exact selected model, without refitting, was then reported on test: <strong>77/100 correct; CE 0.757101</strong>.</p><p>The test gap is retained. These controls inspect existing evidence; they do not produce a fresh untouched test result.</p></div>
    <div className="transfer-columns"><label className="transfer-select">Recorded seed<select value={seed} onChange={event => setSeed(Number(event.target.value))}>{[1, 2, 3].map(value => <option key={value} value={value}>Seed {value}{value === 1 ? ' — original selection' : ' — validation sensitivity only'}</option>)}</select></label><label className="transfer-select">Inspect method<select value={method} onChange={event => setMethod(event.target.value)}>{TRANSFER_METHODS.map(key => <option key={key} value={key}>{TRANSFER_LABELS[key]}</option>)}</select></label></div>
    <p>{seed === 1 ? 'The table shows the original candidate set. Highlighting a different method is inspection, not a new final selection.' : `Seed ${seed} is a sensitivity view. It did not participate in the declared seed-1 choice and has no retained target test result.`}</p>
    <div className="transfer-evidence-cards">{records.map(row => <article key={row.method} className={row.method === method ? 'is-selected' : ''}><h4>{TRANSFER_LABELS[row.method]}</h4><dl><dt>Trainable / total values</dt><dd>{number(row.trainable_parameters)} / {number(row.total_parameters)}</dd><dt>Train correct / 40</dt><dd>{row.train.correct}/40</dd><dt>Validation CE (nats)</dt><dd>{number(row.validation.ce, 6)}</dd><dt>Validation correct / 60</dt><dd>{row.validation.correct}/60</dd></dl><button aria-pressed={row.method === method} onClick={() => setMethod(row.method)}>Inspect {TRANSFER_LABELS[row.method]}</button></article>)}</div>
    <TransferTrace record={selected} /><Retention record={selected} />
    <div className="transfer-budget-decision"><h4>Hypothetical budget applied to the saved seed-1 candidates</h4><p>This separate exercise reinterprets existing validation records; no new model was trained or tested. Seed sensitivity controls above do not change this seed-1 exercise.</p>
      <div key={epoch}><Numeric label="Maximum trainable parameters" value={budget} min={0} max={5000} integer slider onChange={setBudget} /></div>
      <p data-testid="transfer-budget-winner">Eligible: {hypothetical.candidates.length ? hypothetical.candidates.map(row => TRANSFER_LABELS[row.method]).join(', ') : 'none'}.<br /><strong>{hypothetical.winner ? `Hypothetical validation winner: ${TRANSFER_LABELS[hypothetical.winner.method]} (${hypothetical.winner.trainable_parameters} values; CE ${number(hypothetical.winner.validation.ce, 6)})` : 'No eligible candidate.'}</strong></p>
      <p>At 400, LoRA is eligible and wins. At 250, the probe beats the eligible adapter; at 84, none fits. Budgets 400 and 500 leave the same eligible set and winner. These alternatives have no newly measured final test result.</p>
      <button onClick={() => { setSeed(1); setMethod('scratch'); setBudget(400); setEpoch(value => value + 1); }}>Reset evidence workspace</button>
    </div>
  </Lab>;
}

export function TransferCheckpointLab() {
  const [index, setIndex] = useState(0);
  const [reverse, setReverse] = useState(false);
  const [preprocessingChanged, setPreprocessingChanged] = useState(false);
  const probabilities = experiment.selection.test_probabilities[index];
  const labelOrder = reverse ? [9, 8, 7, 6, 5] : [5, 6, 7, 8, 9];
  const largest = probabilities.indexOf(Math.max(...probabilities));
  const row = specimens.test[index];
  return <Lab id="transfer-checkpoint" title="A checkpoint needs meaning as well as numbers" kind="Recorded replay · semantic comparison">
    <dl className="transfer-checkpoint-fields"><dt>Architecture</dt><dd>64 → tanh 32 → tanh 16 → 5</dd><dt>Base and selected artifact</dt><dd>Seed-1 scratch; exact post-300-update weights, no refit</dd><dt>Adaptation configuration</dt><dd>No adapter on this winner; a LoRA artifact additionally needs targeted layers, rank, α and scaling convention</dd><dt>Weights and buffers</dt><dd>Every state-dictionary value; stateful architectures also need running buffers</dd><dt>Preprocessing</dt><dd>8×8 row-major pixels, divide by 16</dd><dt>Output label order</dt><dd>Index 0–4 means digits 5–9</dd></dl>
    <label className="transfer-select">Recorded test specimen<select value={index} onChange={event => setIndex(Number(event.target.value))}>{specimens.test.map((sample, i) => <option key={sample.source_id} value={i}>Row {i + 1}/100 · source ID {sample.source_id} · digit {sample.digit}</option>)}</select></label>
    <div className="transfer-columns"><PixelDigit row={row} /><div><Toggle checked={reverse} onChange={setReverse}>Reverse the declared output labels</Toggle><Toggle checked={preprocessingChanged} onChange={setPreprocessingChanged}>Change the preprocessing version</Toggle><p data-testid="transfer-checkpoint-meaning">Saved argmax index {largest}. Its declared meaning is digit {labelOrder[largest]}. Observed digit: {row.digit}. Relabeling keeps all numerical probabilities unchanged.</p></div></div>
    <div className="transfer-probabilities" aria-label="Saved probabilities and declared meanings">{probabilities.map((value, i) => <div key={i}><span>Index {i} → digit {labelOrder[i]}: {number(value, 6)}</span><div className="transfer-bar" aria-hidden="true"><i style={{ width: `${100 * value}%` }} /></div></div>)}</div>
    <p data-testid="transfer-replay-status">{preprocessingChanged ? 'Replay fixture invalid for the changed preprocessing. The values above remain the original saved probabilities; no new inference has run.' : reverse ? 'Numerical probabilities are preserved, but the changed label order no longer reproduces the original prediction meanings. The original configuration-preserving replay difference was 0.' : 'Recorded state-dictionary round trip: maximum probability difference 0, with architecture, preprocessing and label configuration held fixed.'}</p>
    <p>All 100 test rows are selectable, including mistakes. No remapped aggregate accuracy is claimed. This local list-based replay checks stored values with fixed configuration; it is not a complete deployment artifact.</p>
    <button onClick={() => { setIndex(0); setReverse(false); setPreprocessingChanged(false); }}>Reset checkpoint comparison</button>
  </Lab>;
}

export function TransferScheduleFigure() {
  const [container, width] = usePlotWidth();
  const px = time => 60 + time / 100 * (width - 76);
  const py = rate => 130 - rate / 0.01 * 110;
  const samples = [[0, 0.01 / 32], [10, 0.01], [100, 0.01 / 32]];
  return <figure ref={container} className="transfer-schedule"><figcaption><strong>Specified teaching schedule; not a measured learning curve</strong><br />T = 100, peak c = 10, floor = 1/32 of peak. Rates are sampled at the stated t.</figcaption><svg className="transfer-plot" viewBox={`0 0 ${width} 180`} role="img" aria-label="Rate rises from 0.0003125 at t zero to 0.01 at t ten, then decreases to 0.0003125 at t one hundred"><line x1="60" y1="130" x2={width - 16} y2="130" stroke="#647d6c" /><polyline points={samples.map(([time, rate]) => `${px(time)},${py(rate)}`).join(' ')} fill="none" stroke="#e2b55a" strokeWidth="3" /><text x="52" y="24" textAnchor="end">0.01</text>{(width >= 360 ? [0, 10, 100] : [0, 100]).map(time => <text key={time} x={px(time)} y="151" textAnchor="middle">{time}</text>)}</svg><p>t = 0: 0.0003125 · t = 10: 0.01 · t = 100: 0.0003125. This is the manuscript’s exact teaching triangle, not the original ULMFiT schedule or an observed loss trajectory.</p></figure>;
}
