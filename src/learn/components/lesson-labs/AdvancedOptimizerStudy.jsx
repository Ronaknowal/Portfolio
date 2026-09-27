import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import { useEffect, useRef, useState } from 'react';
import { NeuralLab, NeuralSelect, NeuralTable, formatNeural as f } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { digitUpdate, digitProbabilities } from '../../data/advanced-optimizer-models.js';
import { optimizerDigits } from '../../data/advanced-optimizer-study.js';
import { OptimizerPlot, vector } from './AdvancedOptimizerLabs.jsx';
import { OptimizerFigure } from './AdvancedOptimizerDiagrams.jsx';
const base = '/learn-assets/advanced-optimizers-lion-sophia-prodigy-schedule-free/';
const methods = [['adamw', 'AdamW constant'], ['adamw_cosine', 'AdamW warmup + cosine'], ['lion', 'Lion'], ['sophia_g', 'Sophia-G paper-scaled'], ['prodigy', 'Prodigy paper Algorithm 4'], ['schedule_free', 'Schedule-Free AdamW at x']];
function useAsset(file) {
  const [value, setValue] = useState(null), [failed, setFailed] = useState(false), [attempt, setAttempt] = useState(0), [visible, setVisible] = useState(false), container = useRef(null);
  useEffect(() => { const observer = new IntersectionObserver(entries => { if (entries.some(e => e.isIntersecting)) { setVisible(true); observer.disconnect(); } }, { rootMargin: '240px' }); if (container.current) observer.observe(container.current); return () => observer.disconnect(); }, []);
  useEffect(() => {
    if (!visible) return undefined;
    const controller = new AbortController(); setValue(null); setFailed(false);
    fetch(base + file, { signal: controller.signal }).then(r => { if (!r.ok) throw new Error('Unavailable asset'); return r.json(); }).then(data => { if (!controller.signal.aborted) setValue({ file, data }); }).catch(() => { if (!controller.signal.aborted) setFailed(file); });
    return () => controller.abort();
  }, [visible, file, attempt]);
  return { value: value?.file === file ? value.data : null, failed: failed === file, container, retry: () => setAttempt(n => n + 1) };
}
function AssetState({ resource, children }) {
  if (resource.failed) return <p role="alert">The saved experiment could not be loaded. Check your connection, then <button onClick={resource.retry}>retry the experiment</button>.</p>;
  if (!resource.value) return <p role="status">Loading the selected saved experiment…</p>;
  return children(resource.value);
}
function DigitGrid({ pixels, selected, onSelect }) {
  return <div className="optimizer-digit-scroll" tabIndex={0} role="region" aria-label="Digit pixel grid, horizontally scrollable"><div className="optimizer-digit" role="group" aria-label="Actual 8 by 8 digit pixels; select a pixel, then edit its intensity">{pixels.map((p, i) => <button key={i} type="button" aria-label={`Pixel ${i}, row ${Math.floor(i / 8)}, column ${i % 8}, intensity ${p}`} aria-pressed={selected === i} onClick={() => onSelect(i)} style={{ background: `rgb(${Math.round(22 + p / 16 * 223)},${Math.round(22 + p / 16 * 223)},${Math.round(22 + p / 16 * 223)})`, color: p > 8 ? '#111' : '#fff' }}>{p}</button>)}</div></div>;
}
function ProbabilityBars({ before, after }) {
  return <figure className="optimizer-probabilities"><figcaption>All ten probabilities on the same 0–1 scale</figcaption>{before.map((p, i) => <div key={i}><b>{i}</b><div><span style={{ width: `${100 * p}%` }} /><i style={{ width: `${100 * after[i]}%` }} /></div><small>{f(p, 4)} → {f(after[i], 4)}</small></div>)}<p>Gray upper bar: before; amber lower bar: after one copied-state update.</p></figure>;
}
export function OptimizerDigitLab() {
  const [method, setMethod] = useState('adamw'), [seed, setSeed] = useState(11), [pixels, setPixels] = useState(optimizerDigits[312].pixels), [selected, setSelected] = useState(28), [target, setTarget] = useState(0), [mode, setMode] = useState('evaluation');
  const resource = useAsset(`model-${method}-${seed}.json`);
  const reset = () => { setPixels(optimizerDigits[312].pixels); setSelected(28); setTarget(0); setMode('evaluation'); };
  return <div ref={resource.container}><NeuralLab id="optimizer-digit" title="One edited digit, one genuine optimizer update">
    <p>Validation source 312, recorded class 0. Every diagnostic begins from the saved update-400 state. Editing a pixel or target recalculates one temporary update; it does not retrain or alter the reported experiment.</p>
    <div className="optimizer-controls"><NeuralSelect label="Saved method" value={method} onChange={setMethod} options={methods} /><NeuralSelect label="Saved seed" value={seed} onChange={v => setSeed(Number(v))} options={[[11, '11'], [29, '29']]} /><NeuralSelect label="Diagnostic target label" value={target} onChange={v => setTarget(Number(v))} options={Array.from({ length: 10 }, (_, i) => [i, i])} /></div>
    <div className="optimizer-two"><DigitGrid pixels={pixels} selected={selected} onSelect={setSelected} /><div className="optimizer-controls"><NeuralNumber label="Selected zero-based pixel" value={selected} min={0} max={63} step={1} integer range={false} onChange={setSelected} /><NeuralNumber label="Raw pixel intensity" value={pixels[selected]} min={0} max={16} integer step={1} onChange={v => setPixels(old => old.map((p, i) => i === selected ? v : p))} /><p>Row {Math.floor(selected / 8)}, column {selected % 8}; normalized intensity {f(pixels[selected] / 16, 4)}. Bias is a separate feature with value one.</p></div></div>
    <AssetState resource={resource}>{snapshot => {
      const result = digitUpdate(snapshot, pixels, target), evaluation = mode === 'evaluation';
      const before = evaluation ? result.before.probabilities : result.training.probabilities;
      const after = evaluation ? result.after.probabilities : digitProbabilities(pixels, result.next.parameters).probabilities;
      const delta = after.map((v, i) => v - before[i]);
      const largestDelta = Math.max(...delta.map(Math.abs)), deltaExponent = largestDelta === 0 ? 0 : Math.floor(Math.log10(largestDelta));
      const deltaUnit = 10 ** deltaExponent, scaledDelta = delta.map(v => v / deltaUnit);
      return <><p>Selected scale {snapshot.rate}; gradient at training parameters, update {snapshot.state.step + 1}. Active rate {f(result.next.diagnostics.rate, 8)}. {method === 'adamw_cosine' && 'The frozen cosine schedule is at zero: parameters and probabilities remain unchanged, while moments can still update.'}</p>
        {method === 'schedule_free' && <NeuralSelect label="Displayed parameter mode" value={mode} onChange={setMode} options={[["evaluation", 'Evaluation x: method-defined report'], ['training', 'Training y: separate diagnostic']]} />}
        <ProbabilityBars before={before} after={after} /><NeuralTable caption={`${evaluation ? 'Evaluation' : 'Training'} predictions and precise changes`} headers={['Class', 'Before', 'After', 'Delta']} rows={before.map((p, i) => [i, f(p, 9), f(after[i], 9), delta[i].toExponential(5)])} />
        <OptimizerPlot title="Probability change, enlarged difference scale" xLabel="class ID" yLabel={`Δprobability / 10^${deltaExponent}`} xDomain={[0, 9]} yDomain={[Math.min(-1, ...scaledDelta) * 1.1, Math.max(1, ...scaledDelta) * 1.1]} series={[{ label: 'Diagnostic change in stated units', values: scaledDelta.map((v, i) => [i, v]), color: '#e4b65a' }]} /><p>One vertical-axis unit represents {deltaUnit.toExponential(0)} probability. Multiply the plotted value by that unit to recover after minus before; the table retains unscaled exact changes. {largestDelta === 0 ? 'Every change is zero in this state.' : 'The axis rescales to keep small changes readable.'}</p>
        <NeuralTable caption={`Pixel ${selected}: intensity × training residual for each class`} headers={['Class', 'Training P', 'P − target indicator', 'Pixel gradient', 'Old weight', 'New weight']} rows={result.training.probabilities.map((p, i) => [i, f(p, 7), f(p - Number(i === target), 7), f(result.gradient[selected][i], 7), f(snapshot.parameters[selected][i], 7), f(result.next.parameters[selected][i], 7)])} />
        <p>Before-update probabilities depend on pixels and weights, so changing only the diagnostic label leaves them unchanged. The label changes the residual and hence the update. Stored momentum can oppose the new gradient; the chosen class need not increase.</p>
        {method === 'sophia_g' && <p>This diagnostic refresh uses the exact expected-label GGN diagonal for this one-example linear model: x²P(1−P). It is distinct from the independently sampled-label refresh used during the fit. Clipped fraction {f(result.next.diagnostics.clipped_fraction, 6)}.</p>}
        {method === 'prodigy' && <p>Old d {f(result.next.diagnostics.distance_used, 8)}; next d {f(result.next.diagnostics.distance_next, 8)}. Initial parameters, numerator, displacement history and scaled moments were all restored.</p>}
        <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect this pixel’s persistent state before and after</h4><NeuralTable caption="Parameter row state; scalar statistics retain their exact role" headers={['State', 'Before', 'After']} rows={Object.keys(snapshot.state).map(key => [key, Array.isArray(snapshot.state[key]) ? vector(snapshot.state[key][selected]) : String(snapshot.state[key]), Array.isArray(result.next.state[key]) ? vector(result.next.state[key][selected]) : String(result.next.state[key])])} /></section>
      </>;
    }}</AssetState><div className="optimizer-actions"><button onClick={() => setPixels(old => old.map((p, i) => i === selected ? 16 - p : p))}>Reflect selected intensity</button><button onClick={reset}>Reset diagnostic image</button></div>
  </NeuralLab></div>;
}
function PixelContributionDiagram({ intensity, weights, probabilities, target }) {
  const terms = weights.map(w => intensity * w), gradients = probabilities.map((p, c) => intensity * (p - Number(c === target)));
  const scoreScale = 60 / Math.max(1e-12, ...terms.map(Math.abs)), gradientScale = 60 / Math.max(1e-12, ...gradients.map(Math.abs));
  return <OptimizerFigure title="One normalized input fans into ten parameter paths" width={720} height={435} description="Each row uses the same selected normalized intensity. Amber signed bars show its score contribution; blue bars show its parameter gradient. The score and gradient columns use separately stated scales so small gradients remain visible; their bar lengths are not comparable across columns. The full class score additionally sums the other features and bias.">{marker => <>
    <text x="12" y="23">selected x = {f(intensity, 5)}</text><text x="290" y="23">x × weight → score term</text><text x="535" y="23">x × (P − Y) → gradient</text>
    <path d="M35 37 V392" stroke="#aaa" /><path d="M386 37 V397 M623 37 V397" stroke="#777" strokeDasharray="3 3" />
    {terms.map((v, c) => <g key={c}><path d={`M35 ${57 + 36 * c} H82`} stroke="#aaa" markerEnd={marker} /><text x="91" y={61 + 36 * c}>class {c}: w={f(weights[c], 3)}, P−Y={(probabilities[c] - Number(c === target)).toExponential(2)}</text><rect x={386 + Math.min(0, v) * scoreScale} y={47 + 36 * c} width={Math.abs(v) * scoreScale} height="13" fill="#e4b65a" /><text x="452" y={61 + 36 * c}>{f(v, 4)}</text><rect x={623 + Math.min(0, gradients[c]) * gradientScale} y={47 + 36 * c} width={Math.abs(gradients[c]) * gradientScale} height="13" fill="#9ebde4" /><text x="553" y={78 + 36 * c}>{gradients[c].toExponential(3)}</text></g>)}
    <text x="20" y="422">Score: {scoreScale.toExponential(3)} pixels/unit; gradient: {gradientScale.toExponential(3)} pixels/unit; dashed rails are zero</text>
  </>}</OptimizerFigure>;
}
export function OptimizerPixelFigure() {
  const [selected, setSelected] = useState(18), resource = useAsset('model-adamw-11.json');
  const digit = optimizerDigits[277];
  return <figure className="optimizer-figure" ref={resource.container}><figcaption>One real pixel contributes to all ten class scores and gradients</figcaption><p>Worked source 277, class 0, selected AdamW seed 11. The initial pixel 18 has intensity 16, so its class paths are visible immediately. Select any pixel to follow its raw intensity through the fixed classifier; a zero pixel removes that pixel’s direct score and current-gradient contribution.</p><DigitGrid pixels={digit.pixels} selected={selected} onSelect={setSelected} /><AssetState resource={resource}>{snapshot => { const r = digitUpdate(snapshot, digit.pixels, digit.label); return <><p>Pixel {selected}: raw {digit.pixels[selected]} → divide by 16 → {f(digit.pixels[selected] / 16)} → multiply each class weight → add into its score. The gradient instead multiplies that intensity by the class residual.</p><PixelContributionDiagram intensity={digit.pixels[selected] / 16} weights={snapshot.parameters[selected]} probabilities={r.training.probabilities} target={digit.label} /><NeuralTable caption="One pixel’s score contribution and objective derivative" headers={['Class', 'Pixel weight', 'Pixel × weight', 'P − Y', 'Pixel gradient']} rows={r.training.probabilities.map((p, c) => [c, f(snapshot.parameters[selected][c], 6), f(digit.pixels[selected] / 16 * snapshot.parameters[selected][c], 6), f(p - Number(c === digit.label), 6), f(r.gradient[selected][c], 6)])} /><NeuralTable caption="Separate bias feature: intensity is one" headers={['Class', 'Bias weight = score term', 'Bias gradient = P − Y', 'Full score', 'Probability']} rows={r.training.probabilities.map((p, c) => [c, f(snapshot.parameters[64][c], 6), f(r.gradient[64][c], 6), f(r.training.scores[c], 6), f(p, 6)])} /><p>The full score sums contributions from all 64 pixels and the bias. This selected-pixel table is one term, not the complete prediction.</p></>; }}</AssetState></figure>;
}
export function OptimizerHistoryFigure() {
  const resource = useAsset('histories.json'), [index, setIndex] = useState(0), [log, setLog] = useState(false), [seed, setSeed] = useState(11);
  return <div ref={resource.container}><AssetState resource={resource}>{data => {
    const run = data.runs[index], transform = v => log ? Math.log10(v) : v;
    const selected = data.runs.filter(r => r.seed === seed && r.selected_by_validation);
    const pair = ['fit', 'validation'].map((key, i) => ({ label: key, color: ['#e4b65a', '#9ebde4'][i], values: run.curve.map(r => [r.step, transform(r[key].cross_entropy)]) }));
    const values = pair.flatMap(r => r.values.map(p => p[1]));
    const comparison = selected.map((r, i) => ({ label: `${r.method} scale ${r.rate}`, color: ['#e4b65a', '#9ebde4', '#b7a1d4', '#e8a999', '#ddd', '#d597ad'][i], values: r.curve.map(point => [point.step, transform(point.validation.cross_entropy)]) }));
    const comparativeValues = comparison.flatMap(r => r.values.map(p => p[1]));
    return <figure className="optimizer-figure"><figcaption>Every declared candidate remains visible</figcaption><NeuralSelect label="Inspect one measured candidate" value={index} onChange={v => setIndex(Number(v))} options={data.runs.map((r, i) => [i, `${r.method}, scale ${r.rate}, seed ${r.seed}${r.selected_by_validation ? ' · selected' : ''}`])} /><label className="optimizer-toggle"><input type="checkbox" checked={log} onChange={e => setLog(e.target.checked)} />Use a log10 cross-entropy axis</label>
      <OptimizerPlot title="Actual fitting and validation measurements" xLabel="optimizer update" yLabel={log ? 'log10 cross-entropy (nats/example)' : 'cross-entropy (nats/example)'} xDomain={[1, 400]} yDomain={[log ? Math.min(...values) - .1 : 0, Math.max(...values) + .1]} series={pair} />
      <NeuralSelect label="Paired seed for all selected methods" value={seed} onChange={v => setSeed(Number(v))} options={[[11, '11'], [29, '29']]} /><OptimizerPlot title="Selected settings: common validation-loss axes" xLabel="optimizer update" yLabel={log ? 'log10 validation CE' : 'validation CE (nats/example)'} xDomain={[1, 400]} yDomain={[log ? Math.min(...comparativeValues) - .1 : 0, Math.max(...comparativeValues) + .1]} series={comparison} />
      <p>Lines connect the recorded updates 1, 10 and every twentieth update. They do not supply additional measurements or elapsed-time evidence. Selection uses the mean final validation loss across both seeds, not the most favorable point on these curves.</p>
      {run.method === 'prodigy' && <OptimizerPlot title="Recorded Prodigy scale" xLabel="update" yLabel="distance d retained for next step" xDomain={[1, 400]} yDomain={[0, Math.max(...run.curve.map(r => r.distance_next)) * 1.1]} series={[{ label: 'd next', color: '#e4b65a', values: run.curve.map(r => [r.step, r.distance_next]) }]} />}
      {run.method === 'sophia_g' && <OptimizerPlot title="Recorded Sophia clipping" xLabel="update" yLabel="fraction of coordinates clipped" xDomain={[1, 400]} yDomain={[0, 1]} series={[{ label: 'clipped fraction', color: '#e4b65a', values: run.curve.map(r => [r.step, r.clipped_fraction]) }]} />}
      <NeuralTable caption="Both candidates and seeds, final validation and selected-only assessment" headers={['Method', 'Scale', 'Seed', 'Validation CE', 'Selected?', 'Assessment CE / correct']} rows={data.runs.map(r => [r.method, r.rate, r.seed, f(r.validation.cross_entropy, 6), r.selected_by_validation ? 'yes' : 'no', r.assessment ? `${f(r.assessment.cross_entropy, 6)} / ${r.assessment.correct} of 80` : 'not used for selection'])} />
      {run.method === 'schedule_free' && <p>Current run final validation CE at x: {f(run.validation.cross_entropy, 7)}; at y: {f(run.curve.at(-1).training_iterate_validation.cross_entropy, 7)}. The method defines x as the evaluation point even when y happens to have lower loss.</p>}
      <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Exact measured curve values for this candidate</h4><NeuralTable caption="Recorded points only" headers={['Step', 'Fit CE', 'Validation CE', 'Training-iterate validation CE']} rows={run.curve.map(r => [r.step, f(r.fit.cross_entropy, 8), f(r.validation.cross_entropy, 8), f(r.training_iterate_validation.cross_entropy, 8)])} /></section>
    </figure>;
  }}</AssetState></div>;
}
export function OptimizerProgram({ file }) {
  return <>
    <RemoteCodeBlock source={base + file} language="python" filename={file} title={"Read complete " + (file)} />
  </>;
}
