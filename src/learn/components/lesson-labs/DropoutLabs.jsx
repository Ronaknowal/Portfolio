import { useState } from 'react';
import { NeuralLab, NeuralNumber, NeuralSelect, NeuralTable, NeuralPlot } from './NeuralLessonElements.jsx';
import { formatDropout as fmt, maskedUpdate, maskOutcomes, maskShapes, geometryOutput, residualMask, depthRates, expectedActive, executeDepth, normalizationStep, monteCarloSummary } from '../../data/dropout-models.js';
import measurements from '../../data/dropout-measurements.json';
import './dropout-labs.css';

const vector = values => '[' + values.map(value => fmt(value)).join(', ') + ']';
const number = (label, value, onChange, min = -3, max = 3, step = 'any') => <NeuralNumber {...{ label, value, onChange, min, max, step }} />;
function Bit({ label, value, onChange }) {
  return <button type="button" aria-pressed={Boolean(value)} onClick={() => onChange(1 - value)}>{label}: {value ? '1 · keep' : '0 · drop'}</button>;
}
function Reset({ onClick, children = 'Reset' }) {
  return <button type="button" onClick={onClick}>{children}</button>;
}

export function DropoutUpdateLab() {
  const initial = { features: [1, 2], weights: [1, -.5], mask: [1, 0], probability: .5, target: 1, learningRate: .1, training: true };
  const [state, setState] = useState(initial);
  const [steps, setSteps] = useState(0);
  const result = maskedUpdate(state);
  const edit = (key, value) => { setState(current => ({ ...current, [key]: value })); setSteps(0); };
  const editVector = (key, index, value) => edit(key, state[key].map((old, i) => i === index ? value : old));
  return <NeuralLab title="A sampled mask leaves a visible gradient route" id="dropout-update">
    <p>Two fixed feature lanes join at one scalar output. Edit the values or keep bits; the mask stays fixed during the backward pass and SGD update. The controls edit a mask fixture, not a fresh random draw.</p>
    <div className="dropout-lanes">{state.features.map((value, index) => <section key={index} className={result.factors[index] === 0 ? 'dropout-lane is-dropped' : 'dropout-lane'}>
      <h4>Feature {index + 1}</h4>
      {number('Feature ' + (index + 1), value, next => editVector('features', index, next))}
      {number('Weight ' + (index + 1), state.weights[index], next => editVector('weights', index, next), -30, 30)}
      <Bit label={'Mask ' + (index + 1)} value={state.mask[index]} onChange={next => editVector('mask', index, next)} />
      <ol className="dropout-flow"><li>Input <b>{fmt(value)}</b></li><li>Mask × scale <b>{fmt(result.factors[index])}</b></li><li>Value <b>{fmt(result.masked[index])}</b></li><li>Weighted contribution <b>{fmt(result.contributions[index])}</b></li></ol>
      <p>Backward: dL/dw = <b>{fmt(result.weightGradient[index])}</b>; dL/dh = <b>{fmt(result.inputGradient[index])}</b>.</p>
    </section>)}</div>
    <div className="neural-controls">
      {number('Update drop probability', state.probability, value => edit('probability', value), 0, 1)}
      {number('Target', state.target, value => edit('target', value))}
      {number('SGD learning rate', state.learningRate, value => edit('learningRate', value), 0, .25)}
      <NeuralSelect label="Update mode" value={state.training ? 'train' : 'eval'} onChange={value => edit('training', value === 'train')} options={[['train', 'Training mask'], ['eval', 'Evaluation identity']]} />
    </div>
    <p className="neural-result" data-result="dropout-update">Output {fmt(result.output)} · loss {fmt(result.loss)} · weight gradient {vector(result.weightGradient)}. Next SGD weights {vector(result.nextWeights)} give output {fmt(result.nextOutput)} and loss {fmt(result.nextLoss)} with this same mask.</p>
    <p>{!state.training || state.probability === 0 ? 'Identity mode: mask bits are ignored.' : state.probability === 1 ? 'All values and local gradients are zero; target error can remain nonzero.' : state.mask.every(bit => bit === 0) ? 'This all-dropped sample has zero local gradients, even though its loss may remain nonzero.' : 'A zero bit removes this example’s gradient through that coordinate; it does not delete a parameter.'}</p>
    <div className="neural-buttons"><button type="button" disabled={result.nextWeights.some(value => Math.abs(value) > 30 || !Number.isFinite(value))} onClick={() => { setState(current => ({ ...current, weights: result.nextWeights })); setSteps(value => value + 1); }}>Apply one SGD update</button><Reset onClick={() => { setState(initial); setSteps(0); }} /></div>
    <p className="dropout-small">{steps} updates applied. The button changes real weights; all results are visible before it. Updates beyond the displayed ±30 weight budget are disabled; reset or reduce the rate.</p>
  </NeuralLab>;
}

export function DropoutExpectationLab() {
  const [probability, setProbability] = useState(.5);
  const [scaling, setScaling] = useState('inverted');
  const [custom, setCustom] = useState(.75);
  const multiplier = scaling === 'inverted' ? probability === 1 ? 0 : 1 / (1 - probability) : custom;
  const result = maskOutcomes(probability, multiplier);
  return <NeuralLab title="Four outcomes explain the mean—and what it misses" id="dropout-expectation">
    <p>These leaves enumerate every independent two-bit mask. Their probabilities are exact; they are not averages from a short random animation. Inputs stay [1, 2].</p>
    <div className="neural-controls">
      {number('Outcome drop probability', probability, setProbability, 0, 1)}
      <NeuralSelect label="Survivor scaling" value={scaling} onChange={setScaling} options={[['inverted', 'Inverted: 1 / keep probability'], ['custom', 'Edit the survivor multiplier']]} />
      {scaling === 'custom' && number('Survivor multiplier', custom, setCustom, 0, 5)}
    </div>
    <div className="dropout-outcomes">{result.outcomes.map(row => <div key={row.mask.join('')}>
      <span className="dropout-small">Mask {vector(row.mask)}</span><strong>{vector(row.values)}</strong>
      <span>Probability {fmt(row.probability, 6)}</span><div className="dropout-probability"><span style={{ width: (100 * row.probability) + '%' }} /></div>
      <small>Weighted values {vector(row.values.map(value => value * row.probability))}</small>
    </div>)}</div>
    <p className="neural-result" data-result="dropout-expectation">Survivor multiplier {fmt(multiplier)} · mean {vector(result.mean)} · variances {vector(result.variance)}.</p>
    <p>{probability === 1 ? 'At p = 1 only the all-zero mask has probability one. Inverted scaling explicitly returns zero; expectation preservation no longer applies.' : scaling === 'inverted' ? 'Both means stay at the input values. Variance changes with drop probability because survivors must grow.' : 'Compare each weighted mean with its input. At p = 0.25, multiplier 4/3 restores both means; multiplying by 0.75 does not.'}</p>
    <div className="dropout-nonlinear"><h4>Pass the same masks through a nonlinear fork</h4><p>Use signed contributions [1, −1], then ReLU after their sum. Leaf outputs: {vector(result.outcomes.map(row => row.relu))}.</p><p>Expected ReLU = <b>{fmt(result.expectedRelu)}</b>; ReLU of the expected signed sum = <b>0</b>. {result.expectedRelu > 0 ? 'The positive and negative fluctuations no longer cancel after clipping negatives to zero.' : 'This is a null comparison: the two calculations agree here, not for all masks and rates.'}</p></div>
    <div className="neural-buttons"><button type="button" onClick={() => { setProbability(.25); setScaling('custom'); setCustom(.75); }}>Inspect a scaling mistake</button><Reset onClick={() => { setProbability(.5); setScaling('inverted'); setCustom(.75); }} /></div>
  </NeuralLab>;
}

export function DropoutGeometryLab() {
  const [mode, setMode] = useState('channel');
  const [bits, setBits] = useState([1, 0, 0, 1]);
  const [probability, setProbability] = useState(.5);
  const [training, setTraining] = useState(true);
  const [offset, setOffset] = useState(0);
  const result = geometryOutput({ mode, bits, probability, training, offset });
  const shape = maskShapes[mode];
  function changeMode(next) { setMode(next); setBits(Array(shapeProduct(maskShapes[next])).fill(1)); }
  function reset() { setMode('channel'); setBits([1, 0, 0, 1]); setProbability(.5); setTraining(true); setOffset(0); }
  return <NeuralLab title="Choose which cells share one random decision" id="dropout-geometry">
    <p>Two examples, two channels per example, and four spatial cells per channel. Each cell shows input → output. Crossed cells have an effective multiplier of zero. The stored fixture bit is labeled separately because evaluation and p = 0/1 override it; a zero input alone does not mean a value was dropped.</p>
    <div className="neural-controls"><NeuralSelect label="Mask geometry" value={mode} onChange={changeMode} options={[['element', 'Element: every cell'], ['channel', 'Channel: one feature map'], ['row', 'Row: one example'], ['batch', 'Batch: everything together']]} />
      {number('Geometry drop probability', probability, setProbability, 0, 1)}
      {number('Add to every input', offset, setOffset, -2, 2)}
      <NeuralSelect label="Geometry mode" value={training ? 'train' : 'eval'} onChange={value => setTraining(value === 'train')} options={[['train', 'Training'], ['eval', 'Evaluation']]} /></div>
    <p className="neural-result" data-result="dropout-geometry">Mask shape {vector(shape)} · {shapeProduct(shape)} independent bits. {training && probability > 0 && probability < 1 ? 'Surviving values scale by ' + fmt(1 / (1 - probability)) + '.' : 'Boundary/evaluation rule overrides the editable fixture bits.'}</p>
    <div className="neural-buttons">{bits.map((bit, index) => <Bit key={index} label={'Bit ' + (index + 1)} value={bit} onChange={value => setBits(bits.map((old, i) => i === index ? value : old))} />)}</div>
    <div className="dropout-examples">{[0, 1].map(example => <section key={example}><h4>Example {example + 1}</h4>{[0, 1].map(channel => <div className="dropout-channel" key={channel}><p>Channel {channel + 1}</p><div className="dropout-grid">{result.slice(example * 8 + channel * 4, example * 8 + channel * 4 + 4).map((cell, index) => <div key={index} data-mask={cell.factor === 0 ? 0 : 1}><span>Fixture bit {cell.maskIndex + 1}: {cell.bit}</span><span>{cell.factor === 0 ? '× drop' : cell.factor === 1 ? 'Identity' : 'Keep and scale'} · multiplier {fmt(cell.factor)}</span><b>{fmt(cell.input)} → {fmt(cell.output)}</b></div>)}</div></div>)}</section>)}</div>
    <p>For fixed [1, 2] at p = 0.5, independent masks have covariance 0; a shared mask has covariance 2. Sharing a bit changes which values fluctuate together. It does not establish correlations in a dataset.</p>
    <div className="neural-buttons"><button type="button" onClick={() => { setMode('channel'); setBits([1, 0, 1, 1]); }}>Hide only example 1 / channel 2</button><button type="button" onClick={() => setBits(bits.map(() => 1))}>Keep every bit</button><Reset onClick={reset} /></div>
    <p className="dropout-small">Keeping every bit in a training fixture still applies survivor scaling when 0 &lt; p &lt; 1. Select evaluation or p = 0 for the identity operation.</p>
  </NeuralLab>;
}
const shapeProduct = shape => shape.reduce((product, value) => product * value, 1);

export function DropoutBranchLab() {
  const [input, setInput] = useState([2, -1]);
  const [correction, setCorrection] = useState([.5, 1]);
  const [probability, setProbability] = useState(.5);
  const [bit, setBit] = useState(0);
  const [placement, setPlacement] = useState('branch');
  const result = residualMask({ input, correction, probability, bit, placement });
  return <NeuralLab title="Move the mask and inspect what survives" id="dropout-branch">
    <div className="neural-controls">{input.map((value, i) => number('Residual input ' + (i + 1), value, next => setInput(input.map((old, j) => i === j ? next : old))))}{correction.map((value, i) => number('Correction ' + (i + 1), value, next => setCorrection(correction.map((old, j) => i === j ? next : old))))}
      {number('Branch drop probability', probability, setProbability, 0, 1)}
      <NeuralSelect label="Mask placement" value={placement} onChange={setPlacement} options={[['branch', 'On the correction F only'], ['whole', 'After the whole sum x + F']]} /></div>
    <div className="neural-buttons"><Bit label="Branch bit" value={bit} onChange={setBit} /></div>
    <div className="dropout-circuit">
      <div><span>Direct lane</span><b>x = {vector(input)}</b><p>{placement === 'branch' ? 'Untouched path into the join' : 'Will be masked after the join'}</p></div>
      <div><span>Correction lane</span><b>F = {vector(correction)}</b><p>{placement === 'branch' ? 'Multiply only this lane by ' + fmt(result.factor) : 'Join before applying mask scale ' + fmt(result.factor)}</p></div>
      <div className="dropout-join"><span>{placement === 'branch' ? 'x + ' + fmt(result.factor) + 'F' : fmt(result.factor) + '(x + F)'}</span><b>{vector(result.output)}</b></div>
    </div>
    <p className="neural-result" data-result="dropout-branch">{result.preservesInput ? 'Output equals x for this case.' : 'Output differs from x.'} {result.factor === 0 ? placement === 'branch' ? 'Dropping the correction preserves the direct path.' : 'Dropping the complete sum removes the direct path too.' : 'A surviving correction may add, amplify or cancel: a skip is not a gradient guarantee.'} {probability === 0 || probability === 1 ? 'The p = ' + probability + ' boundary overrides the stored fixture bit.' : ''}</p>
    <p>The conditional derivative is I + factor × J_F when only the branch is masked. For 0 &lt; p &lt; 1, factor is m/(1 − p); the endpoints use their explicit identity/zero rules. The displayed correction values are a local fixture; the lab does not pretend to know their input Jacobian.</p>
    <div className="neural-buttons"><button type="button" onClick={() => setInput([1, 3])}>Change the residual input</button><Reset onClick={() => { setInput([2, -1]); setCorrection([.5, 1]); setProbability(.5); setBit(0); setPlacement('branch'); }} /></div>
  </NeuralLab>;
}

export function DropoutDepthLab() {
  const [length, setLength] = useState(4);
  const [endpoint, setEndpoint] = useState(.5);
  const [convention, setConvention] = useState('zero-first');
  const [rates, setRates] = useState(depthRates(4, .5));
  const [bits, setBits] = useState([1, 0, 1, 0]);
  const [execution, setExecution] = useState('eager');
  function schedule(nextLength, nextEndpoint, nextConvention) {
    setLength(nextLength); setEndpoint(nextEndpoint); setConvention(nextConvention);
    setRates(depthRates(nextLength, nextEndpoint, nextConvention)); setBits(Array.from({ length: nextLength }, (_, i) => Number(i % 2 === 0)));
  }
  const executed = executeDepth({ rates, bits, strategy: execution });
  const active = executed.trace.map(row => row.active);
  return <NeuralLab title="Active branches and executed work are separate counters" id="dropout-depth">
    <p>Set a schedule, then edit individual block probabilities. Bits are one batchwise fixture, shared across the batch for each block. To isolate executed work, each candidate branch independently receives scalar x = 1 and computes F(x) = 2x. The call counter increments inside that function. These small jobs test the execution strategy, not a fitted residual network.</p>
    <div className="neural-controls"><NeuralNumber label="Number of blocks" value={length} onChange={value => schedule(value, endpoint, convention)} min={1} max={12} integer step={1} />
      {number('Schedule endpoint', endpoint, value => schedule(length, value, convention), 0, 1)}
      <NeuralSelect label="Schedule convention" value={convention} onChange={value => schedule(length, endpoint, value)} options={[['zero-first', 'Zero-first: first block never dropped'], ['original', 'Original index: pmax × l / L']]} />
      <NeuralSelect label="Execution strategy" value={execution} onChange={setExecution} options={[['eager', 'Compute F, then multiply by mask'], ['lazy', 'Check batch bit, then call F if kept']]} /></div>
    <div className="dropout-blocks">{rates.map((rate, index) => <section key={index}><h4>Block {index + 1}</h4>{number('Block ' + (index + 1) + ' drop rate', rate, value => setRates(rates.map((old, i) => i === index ? value : old)), 0, 1)}<Bit label={'Block ' + (index + 1)} value={bits[index]} onChange={value => setBits(bits.map((old, i) => i === index ? value : old))} /><p>{active[index] ? 'Active contribution' : 'No correction contribution'} · {executed.trace[index].called ? 'F called' : 'F skipped'}</p><p>Raw F: {executed.trace[index].raw === null ? 'not computed' : fmt(executed.trace[index].raw)}<br />Masked correction: {fmt(executed.trace[index].correction)}</p></section>)}</div>
    <p className="neural-result" data-result="dropout-depth">Expected active branches: {fmt(expectedActive(rates))} / {length}. Current active fixture: {active.reduce((sum, value) => sum + value, 0)}. Branch calls: {executed.calls} / {length}. Sum of the independently computed masked corrections: {fmt(executed.totalCorrection)}.</p>
    <p>Expectation sums 1 − p for each block, regardless of independence. The current bits describe one outcome. The call count establishes skipped functions in this declared strategy; it is not elapsed time or a GPU speedup. At p = 0/1, the deterministic boundary overrides a conflicting editable bit.</p>
    <Reset onClick={() => { schedule(4, .5, 'zero-first'); setExecution('eager'); }} />
  </NeuralLab>;
}

export function DropoutModeLab() {
  const [batchTraining, setBatchTraining] = useState(true);
  const [dropoutTraining, setDropoutTraining] = useState(true);
  const [recordGradients, setRecordGradients] = useState(false);
  const [values, setValues] = useState([0, 2, 0, 6]);
  const [memory, setMemory] = useState({ runningMean: 0, runningVariance: 1, batches: 0 });
  const next = normalizationStep({ values, ...memory, batchTraining, recordGradients });
  const masked = next.output.map((value, index) => dropoutTraining ? value * (index % 2) * 2 : value);
  function reset() { setBatchTraining(true); setDropoutTraining(true); setRecordGradients(false); setValues([0, 2, 0, 6]); setMemory({ runningMean: 0, runningVariance: 1, batches: 0 }); }
  return <NeuralLab title="Changing gradient recording does not change module mode" id="dropout-mode">
    <p>This small state machine mirrors BatchNorm(momentum=1, epsilon=10⁻⁵, no affine parameters), followed by p=0.5 dropout. The dropout bits alternate drop/keep and stay fixed for inspection. The forward button commits the displayed buffer transition; output is already visible.</p>
    <div className="neural-controls"><NeuralSelect label="BatchNorm mode" value={batchTraining ? 'train' : 'eval'} onChange={value => setBatchTraining(value === 'train')} options={[['train', 'Training: use current batch'], ['eval', 'Evaluation: use stored buffers']]} />
      <NeuralSelect label="Dropout mode" value={dropoutTraining ? 'train' : 'eval'} onChange={value => setDropoutTraining(value === 'train')} options={[['train', 'Training: mask and scale'], ['eval', 'Evaluation: identity']]} />
      <NeuralSelect label="Gradient recording" value={recordGradients ? 'on' : 'off'} onChange={value => setRecordGradients(value === 'on')} options={[['off', 'Off: no_grad'], ['on', 'On: record operations']]} />
      {values.map((value, index) => number('Batch value ' + (index + 1), value, nextValue => setValues(values.map((old, i) => i === index ? nextValue : old)), -8, 8))}</div>
    <NeuralTable caption="Current buffers and the next forward transition" headers={['Buffer', 'Stored now', 'After next forward']} rows={[
      ['Running mean', fmt(memory.runningMean), fmt(next.runningMean)],
      ['Running variance (corrected estimate)', fmt(memory.runningVariance), fmt(next.runningVariance)],
      ['Batches tracked', memory.batches, next.batches],
    ]} />
    <p className="neural-result" data-result="dropout-mode">BatchNorm output {vector(next.output)} → dropout output {vector(masked)}. {recordGradients ? 'Gradient recording enabled.' : 'No gradient recording.'} {batchTraining ? 'Running buffers still update in training mode.' : 'Evaluation leaves running buffers unchanged.'}</p>
    <div className="neural-buttons"><button type="button" onClick={() => setMemory({ runningMean: next.runningMean, runningVariance: next.runningVariance, batches: next.batches })}>Apply this forward pass</button><button type="button" onClick={() => { setMemory({ runningMean: 2, runningVariance: 8, batches: 1 }); setValues([1, 3]); setBatchTraining(false); setDropoutTraining(false); setRecordGradients(false); }}>Inspect clean ordinary evaluation</button><button type="button" onClick={() => { setBatchTraining(false); setDropoutTraining(true); setRecordGradients(false); }}>Select MC mode only</button><Reset onClick={reset} /></div>
    <p>For the noisy values [0, 2, 0, 6], population variance is 6 while the stored n−1 estimate is 8. Clean evaluation after that update maps [1, 3] to approximately [−0.353553, 0.353553]. Enabling only dropout for MC preserves the stored BatchNorm values.</p>
  </NeuralLab>;
}

function Digit({ specimen }) {
  return <figure className="dropout-digit"><div className="neural-pixels" role="img" aria-label={'Real UCI digit ' + specimen.target + ', source ID ' + specimen.id}>{specimen.pixels.map((value, index) => <span key={index} style={{ '--pixel': 'rgb(' + Array(3).fill(Math.round(value / 16 * 255)).join(',') + ')' }} />)}</div><figcaption>Source {specimen.id} · digit {specimen.target}</figcaption></figure>;
}

export function DropoutMeasuredLab() {
  const [family, setFamily] = useState('mlp');
  const [seed, setSeed] = useState('1');
  const [variantKey, setVariantKey] = useState('0.2-row');
  const options = measurements.fits.filter(run => run.family === family && run.seed === Number(seed));
  const baseline = options.find(run => run.drop_probability === 0);
  const variant = options.find(run => run.drop_probability + '-' + run.mode === variantKey) || baseline;
  const last = run => run.trace.at(-1);
  const gap = last(variant).validation.cross_entropy - last(baseline).validation.cross_entropy;
  const series = [baseline, variant].flatMap((run, index) => ['train', 'validation'].map(split => ({
    label: (index ? 'Variant' : 'Baseline') + ' ' + split,
    color: index ? '#e4b752' : '#dddddd', dashed: split === 'train',
    values: run.trace.map(row => [row.step, row[split].cross_entropy]),
  })));
  const maxY = Math.max(...series.flatMap(item => item.values.map(point => point[1]))) * 1.08;
  return <NeuralLab title="Compare recorded fits without inventing a regularization win" id="dropout-measured">
    <p>These controls select the 27 retained CPU fits; they do not train a model in the browser. Each comparison keeps architecture family, initial seed and 400-update budget matched. White is the unmasked baseline; amber is the selected variant; dashed lines show training CE.</p>
    <div className="neural-controls"><NeuralSelect label="Architecture family" value={family} onChange={value => { setFamily(value); setVariantKey('0.2-row'); }} options={[['mlp', 'MLP · 8,970 parameters'], ['residual', 'Residual MLP · 21,450 parameters']]} /><NeuralSelect label="Recorded seed" value={seed} onChange={setSeed} options={['1', '2', '3'].map(value => [value, value])} /><NeuralSelect label="Recorded masking configuration" value={variant.drop_probability + '-' + variant.mode} onChange={setVariantKey} options={options.map(run => [run.drop_probability + '-' + run.mode, run.drop_probability === 0 ? 'No masking' : (family === 'mlp' ? 'Element' : run.mode) + ' · ' + run.drop_probability])} /></div>
    <NeuralPlot title="Actual recorded loss points" xLabel="optimizer updates" yLabel="cross-entropy (nats / example)" xDomain={[0, 400]} yDomain={[0, maxY]} series={series} points={series.flatMap((item, index) => item.values.map(([x, y], i) => ({ id: index + '-' + i, x, y, color: item.color, label: item.label + ': step ' + x + ', CE ' + fmt(y, 6) })))} />
    <p className="dropout-small">Line segments connect six measured checkpoints at 0, 1, 25, 100, 200 and 400; intermediate values were not measured. The table preserves the final small differences hidden by the full loss scale.</p>
    <details><summary>Exact values at every plotted checkpoint</summary><NeuralTable caption="All six checkpoints for the displayed baseline and variant" headers={['Update', 'Baseline train CE', 'Baseline validation CE', 'Variant train CE', 'Variant validation CE']} rows={baseline.trace.map((row, index) => [row.step, fmt(row.train.cross_entropy, 9), fmt(row.validation.cross_entropy, 9), fmt(variant.trace[index].train.cross_entropy, 9), fmt(variant.trace[index].validation.cross_entropy, 9)])} /></details>
    <NeuralTable caption="Final matched comparison" headers={['Run', 'Train CE', 'Validation CE', 'Validation correct']} rows={[['Unmasked baseline', fmt(last(baseline).train.cross_entropy, 6), fmt(last(baseline).validation.cross_entropy, 6), last(baseline).validation.correct + ' / 120'], ['Selected variant', fmt(last(variant).train.cross_entropy, 6), fmt(last(variant).validation.cross_entropy, 6), last(variant).validation.correct + ' / 120']]} />
    <p className="neural-result" data-result="dropout-measured">Variant − baseline validation CE = {fmt(gap, 9)}. {Math.abs(gap) < 1e-10 ? 'Identical comparison within 10⁻¹⁰.' : gap > 0 ? 'The masking variant has worse validation loss in this recorded run.' : 'The masking variant has better validation loss in this recorded run.'} This is validation evidence for this setup, not a universal ranking.</p>
    <div className="dropout-specimens">{measurements.specimens.map(specimen => <Digit specimen={specimen} key={specimen.id} />)}</div>
    <p className="dropout-small">Real 8×8 UCI Optical Recognition digits, Alpaydin and Kaynak, CC BY 4.0. Pixel values are grayscale measurements, not a decorative theme colour.</p>
    <Reset onClick={() => { setFamily('mlp'); setSeed('1'); setVariantKey('0.2-row'); }} />
  </NeuralLab>;
}

export function DropoutMonteCarloLab() {
  const [specimen, setSpecimen] = useState('0');
  const [count, setCount] = useState(10);
  const index = Number(specimen);
  const rows = measurements.samples.map(draw => draw[index]);
  const current = monteCarloSummary(rows.slice(0, count));
  const full = monteCarloSummary(rows);
  const item = measurements.specimens[index];
  return <NeuralLab title="Inspect a prefix of actual Monte Carlo predictions" id="dropout-monte-carlo">
    <p>The model and 100 recorded draws stay fixed. Select how many of those draws to average, for the three specimens whose complete individual predictions were retained. Changing the prefix estimates the same saved experiment with fewer samples; it does not retrain or draw new masks.</p>
    <div className="neural-controls"><NeuralSelect label="Recorded specimen" value={specimen} onChange={setSpecimen} options={measurements.specimens.map((entry, i) => [String(i), 'Source ' + entry.id + ' · true digit ' + entry.target])} /><NeuralNumber label="Saved draws in prefix" value={count} onChange={setCount} min={1} max={100} integer step={1} /></div>
    <div className="dropout-mc-layout"><Digit specimen={item} /><div className="dropout-class-bars">{current.mean.map((value, digit) => <div key={digit}><span>Digit {digit}</span><div className="dropout-probability"><span style={{ width: value * 100 + '%' }} /></div><b>{fmt(value, 5)}</b></div>)}</div></div>
    <NeuralTable caption="Probability moments for the current prefix" headers={['Digit', 'Mean probability', 'Sample standard deviation']} rows={current.mean.map((value, digit) => [digit, fmt(value, 7), current.std[digit] === null ? 'Undefined for one draw' : fmt(current.std[digit], 7)])} />
    <p className="neural-result" data-result="dropout-monte-carlo">Mean chooses {current.choice}; true digit {item.target}. Entropy of mean {fmt(current.predictiveEntropy, 7)} nats; mean entropy {fmt(current.meanEntropy, 7)}; disagreement {fmt(current.disagreement, 7)}. Entropy difference versus all 100 = {fmt(current.predictiveEntropy - full.predictiveEntropy, 7)}.</p>
    <p>Entropy need not decrease monotonically as more masks are included. With one draw, between-mask disagreement is zero by construction, while sample standard deviation is undefined. A stable wrong answer can also have low disagreement.</p>
    <p>Hypothetical comparison: [.9, .1] / [.1, .9] and two [.5, .5] draws both average to [.5, .5]. Their disagreement is 0.368064 and 0 nats respectively. This separates masks that disagree from a model that stays ambiguous on each pass; neither number certifies correctness.</p>
    <Reset onClick={() => { setSpecimen('0'); setCount(10); }} />
  </NeuralLab>;
}

export function DropoutProgram({ file = 'dropout-experiments.py', title = 'Read the complete experiment program', start, end }) {
  const [source, setSource] = useState(null);
  const [failed, setFailed] = useState(false);
  const [loading, setLoading] = useState(false);
  const url = '/learn-assets/dropout-droppath-stochastic-depth/' + file;
  async function load() {
    setLoading(true); setFailed(false);
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error('Source unavailable');
      const text = await response.text();
      const first = start ? text.indexOf(start) : 0;
      const last = end ? text.indexOf(end, first + (start?.length || 0)) : text.length;
      if (first < 0 || last < first) throw new Error('Source boundary missing');
      setSource(text.slice(first, last));
    } catch { setFailed(true); }
    finally { setLoading(false); }
  }
  return <details className="neural-program" onToggle={event => { if (event.currentTarget.open && source === null && !loading && !failed) load(); }}><summary>{title}</summary>
    {loading && <p role="status">Loading the canonical source…</p>}
    {failed && <p role="alert">The source could not load. <button type="button" onClick={load}>Retry source</button></p>}
    {source !== null && <pre className="neural-program-source" tabIndex={0} aria-label={title}><code>{source}</code></pre>}
    <p><a href={url} download>Download {file}</a></p>
  </details>;
}
