import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import { useEffect, useMemo, useRef, useState } from 'react';
import { NeuralLab, NeuralTable } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';

import { XFigure, XArrow, XNode, XValues, XPlot, XMatrix, XImage, f, vec } from './XlstmPrimitives.jsx';
import { readerForward } from '../../data/xlstm-memory-models.js';
const base = '/learn-code/xlstm-extended-lstm/';
export const readerIds = ['lstm', 'slstm', 'mlstm'].flatMap(kind => [19, 43].map(seed => `${kind}_seed${seed}`));
function useNear() {
  const ref = useRef(null),
    [near, setNear] = useState(false);
  useEffect(() => {
    if (!ref.current) return;
    if (!globalThis.IntersectionObserver) {
      setNear(true);
      return;
    }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(e => e.isIntersecting)) {
        setNear(true);
        observer.disconnect();
      }
    }, {
      rootMargin: '350px'
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, near];
}
function useAsset(name, enabled, retry) {
  const [state, setState] = useState({});
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    setState({
      name
    });
    fetch(base + name, {
      signal: controller.signal
    }).then(r => {
      if (!r.ok) throw Error('asset');
      return r.json();
    }).then(data => setState({
      name,
      data
    })).catch(error => {
      if (error.name !== 'AbortError') setState({
        name,
        error: true
      });
    });
    return () => controller.abort();
  }, [name, enabled, retry]);
  return state.name === name ? state : {};
}
function ModelChoice({
  id,
  onChange,
  label = 'Selected fitted row reader'
}) {
  return <label className="xl-select">{label}<select value={id} onChange={e => onChange(e.target.value)}>{readerIds.map(name => <option key={name} value={name}>{name.replace('_seed', ' · seed ')}</option>)}</select></label>;
}
const maxDelta = (a, b) => Math.max(...a.flat(Infinity).map((v, i) => Math.abs(v - b.flat(Infinity)[i])));
export const initialReaderView = {
  prefix: 7,
  channel: 0,
  key: 0,
  scoreClass: 4,
  split: 3,
  execution: 'full',
  reversed: false
};
export function executeReader(rows, model, view) {
  if (view.execution === 'carry') {
    const first = readerForward(rows.slice(0, view.split), model),
      last = readerForward(rows.slice(view.split), model, {
        initial: first.state
      });
    return {
      trace: [...first.trace, ...last.trace],
      state: last.state
    };
  }
  const noNormalizer = view.execution === 'reset-n' && model.kind === 'lstm';
  return readerForward(rows, model, {
    resetAfter: view.execution.startsWith('reset') && !noNormalizer ? view.split : -1,
    resetNormalizer: view.execution === 'reset-n'
  });
}
export function RowReaderWorkspace({
  model,
  original,
  raw,
  view = initialReaderView,
  onView = () => {},
  pinned = null
}) {
  const rows = view.reversed ? [...raw].reverse() : raw,
    result = useMemo(() => executeReader(rows, model, view), [rows, model, view]),
    reference = useMemo(() => readerForward(original.pixels, model), [original, model]),
    unsplit = useMemo(() => readerForward(rows, model), [rows, model]),
    current = result.trace[view.prefix],
    detail = current.cellDetails,
    channel = view.channel,
    firstChanged = rows.findIndex((row, i) => row.some((v, j) => v !== original.pixels[i][j])),
    prefixDifferences = result.trace.map((r, i) => maxDelta(r.logits, reference.trace[i].logits)),
    scheduleDifference = Math.max(...result.trace.map((r, i) => maxDelta(r.logits, unsplit.trace[i].logits))),
    stateDifference = maxDelta(result.state, unsplit.state),
    change = (key, value) => onView({
      ...view,
      [key]: value
    });
  return <><p className="xl-result" aria-live="polite">{model.id}, selected epoch {model.selectedEpoch}: current final predicted class {result.trace[7].prediction}; original-image reference class {reference.trace[7].prediction}. Source row {original.sourceId} has evaluation label {original.label}, which is never passed into the network. Maximum logit difference from an unsplit execution of these same ordered pixels: {scheduleDifference.toExponential(5)}.</p>
 <div className="neural-controls"><NeuralNumber label="Inspect scan step" value={view.prefix + 1} min={1} max={8} integer onChange={n => change('prefix', n - 1)} /><NeuralNumber label="Track digit class score" value={view.scoreClass} min={0} max={9} integer onChange={n => change('scoreClass', n)} /><NeuralNumber label="Inspect hidden/value channel" value={channel + 1} min={1} max={16} integer onChange={n => change('channel', n - 1)} /><NeuralNumber label="Split after scan step" value={view.split} min={1} max={7} integer onChange={n => change('split', n)} /></div>
 <div className="xl-two"><label className="xl-select">Execution and state ownership<select value={view.execution} onChange={e => change('execution', e.target.value)}><option value="full">Single complete scan</option><option value="carry">Split, carry every state component</option><option value="reset">Split, reset all state</option><option value="reset-n">Split, reset only normalizer n (LSTM: no n)</option></select></label><label className="xl-select">Row order<select value={view.reversed ? 'reverse' : 'natural'} onChange={e => change('reversed', e.target.value === 'reverse')}><option value="natural">Original top-to-bottom order</option><option value="reverse">Reverse complete row records</option></select></label></div>
 <XFigure title="The selected pixel row enters one chronological recurrence" width={640} height={185}>{rows.map((row, i) => <g key={i}><rect x={12 + i * 77} y="35" width="61" height="67" fill={i === view.prefix ? '#48391d' : '#222'} stroke={i === view.prefix ? '#e6bb60' : '#777'} /><text x={42 + i * 77} y="58" textAnchor="middle">step {i + 1}</text><text x={42 + i * 77} y="81" textAnchor="middle">row {view.reversed ? 8 - i : i + 1}</text>{i < 7 && <XArrow x1={73 + i * 77} y1={69} x2={89 + i * 77} y2={69} />}</g>)}<line x1={5 + view.split * 77} x2={5 + view.split * 77} y1="14" y2="132" stroke="#87bdf1" strokeDasharray="4 3" /><text x="12" y="156">Boundary after step {view.split}: {view.execution === 'full' ? 'marked only; single scan' : view.execution === 'carry' ? 'all state carried' : view.execution === 'reset-n' ? model.kind === 'lstm' ? 'LSTM has no n; no reset' : 'only n deliberately reset' : 'all state deliberately reset'}.</text></XFigure>
 <div className="xl-images"><XImage title="Original reference image" pixels={original.pixels} selectedRow={view.prefix} /><XImage title="Current editable image" pixels={raw} selectedRow={view.reversed ? 7 - view.prefix : view.prefix} /></div><p>Selected input row in scan order: {vec(rows[view.prefix])}; normalized by16: {vec(rows[view.prefix].map(x => x / 16))}. {firstChanged < 0 ? 'Pixels and row order match the original reference.' : `First changed input is scan step ${firstChanged + 1}.`} {firstChanged > 0 && `Maximum earlier-prefix logit change is ${Math.max(...prefixDifferences.slice(0, firstChanged)).toExponential(5)}; a deliberate state reset before that input can also alter the trace.`}</p>
 <XPlot title={`Class ${view.scoreClass} score after each observed row`} xLabel="observed scan step" yLabel="logit (before softmax)" series={[{
      label: 'same-model original pixels, natural order',
      color: '#87bdf1',
      points: reference.trace.map((r, i) => [i + 1, r.logits[view.scoreClass]])
    }, {
      label: 'current pixels and execution',
      color: '#e6bb60',
      points: result.trace.map((r, i) => [i + 1, r.logits[view.scoreClass]])
    }, {
      label: 'selected current prefix',
      color: '#e9979f',
      points: [[view.prefix + 1, current.logits[view.scoreClass]]]
    }]} />
 <NeuralTable caption={`All ten scores at selected prefix ${view.prefix + 1}`} headers={['Digit class', 'Current logit', 'Current probability', 'Original logit', 'Logit change']} rows={current.logits.map((x, i) => [i, f(x, 8), f(current.probabilities[i], 8), f(reference.trace[view.prefix].logits[i], 8), f(x - reference.trace[view.prefix].logits[i], 8)])} />
 <p>At this prefix the current argmax is {current.prediction}. The training loss was applied only after all eight rows; intermediate classes expose the current readout and were not separately supervised. Final state-coordinate difference from the complete scan: {stateDifference.toExponential(5)}.</p>
 {model.kind === 'mlstm' ? <><NeuralNumber label="Inspect matrix key coordinate" value={view.key + 1} min={1} max={8} integer onChange={n => change('key', n - 1)} /><XMatrix title={`Actual carried C at prefix ${view.prefix + 1}`} values={current.state[0]} selected={[view.key, channel]} /><p>C has eight key rows and sixteen value columns. Its selected entry is {f(current.state[0][view.key][channel], 9)}. The query is projected and divided by√8 exactly once; no probability interpretation is imposed on signed key alignments.</p><XValues caption="Follow the selected matrix read coordinate through normalization and exposure" rows={[[`q coordinate ${view.key + 1}`, f(detail.query[view.key], 9)], [`k coordinate ${view.key + 1}`, f(detail.key[view.key], 9)], [`new value coordinate ${channel + 1}`, f(detail.value[channel], 9)], ['Entire carried n', vec(current.state[1])], ['Carried log-scale m', f(current.state[2], 9)], ['Effective write / retention', `${f(detail.write, 9)} / ${f(detail.retain, 9)}`], ['Signed nᵀq', f(detail.mass, 9)], ['Scaled floor exp(−m)', f(detail.floor, 9)], ['One denominator max(|nᵀq|,exp(−m))', f(detail.denominator, 9)], ['Selected Cᵀq numerator', f(detail.numerator[channel], 9)], ['Memory read before RMSNorm', f(detail.read[channel], 9)], ['Read after learned RMSNorm', f(detail.readNorm[channel], 9)], ['Output gate', f(detail.outputGate[channel], 9)], ['Exposed cell output', f(current.mixed[channel], 9)]]} /></> : <><XValues caption={`Carried ${model.kind === 'slstm' ? 'h,c,n,m' : 'h,c'} for channel ${channel + 1}`} rows={current.state.map((values, i) => [['h', 'c', 'n', 'm'][i], f(values[channel], 9)])} /><XPlot title={`Channel ${channel + 1}: carried hidden output and content`} xLabel="scan step" yLabel="state coordinate" series={['h', 'c'].map((name, j) => ({
        label: name,
        color: j ? '#87bdf1' : '#e6bb60',
        points: result.trace.map((r, i) => [i + 1, r.state[j][channel]])
      }))} />{model.kind === 'slstm' ? <><XPlot title={`Channel ${channel + 1}: mass and separate log-scale`} xLabel="scan step" yLabel="stored coordinate (distinct units)" series={['n', 'm'].map((name, j) => ({
          label: name,
          color: j ? '#e9979f' : '#87bdf1',
          points: result.trace.map((r, i) => [i + 1, r.state[j + 2][channel]])
        }))} /><NeuralTable caption="Input and recurrent mixing meet before each gate nonlinearity" headers={['Gate field', 'Current input projection', 'Previous hidden projection', 'Sum']} rows={['write log', 'forget preactivation', 'output preactivation', 'candidate preactivation'].map((name, i) => {
          const k = i * 16 + channel;
          return [name, f(detail.inputGates[k], 8), f(detail.recurrent[k], 8), f(detail.inputGates[k] + detail.recurrent[k], 8)];
        })} /><XValues caption="Actual stabilized gate and candidate values" rows={['writeLog', 'forgetLog', 'write', 'retain', 'candidate', 'outputGate'].map(name => [name, f(detail[name][channel], 9)])} /></> : <XValues caption="Native LSTM gate order is input, forget, candidate, output" rows={['inputGate', 'forgetGate', 'candidate', 'outputGate'].map(name => [name, f(detail[name][channel], 9)])} />}</>}
 <XFigure title="The inspected recurrent output rejoins the complete residual block" width={650} height={210}><XNode x={5} y={25} width={165} lines={['embedded input', f(current.stages.input_projection[channel], 6)]} /><XNode x={5} y={119} width={165} lines={['cell output', f(current.mixed[channel], 6)]} /><XArrow x1={170} y1={47} x2={215} y2={87} /><XArrow x1={170} y1={141} x2={215} y2={87} /><circle cx="225" cy="87" r="13" fill="#222" stroke="#e6bb60" /><text x="225" y="92" textAnchor="middle">+</text><XArrow x1={238} y1={87} x2={270} y2={87} /><XNode x={270} y={65} width={155} lines={['first residual', f(current.residual[channel], 6)]} /><XArrow x1={425} y1={87} x2={464} y2={87} /><XNode x={464} y={65} width={175} lines={['+ gated FFN output', f(current.output[channel], 6)]} /><text x="15" y="191">Shown coordinate {channel + 1}; the classifier uses all16 final coordinates.</text></XFigure>
 <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Every prefix, full state and full block-stage vectors</h4><NeuralTable caption="All eight chronological decisions" headers={['Step', 'Original class', 'Current class', 'Maximum logit change']} rows={result.trace.map((r, i) => [i + 1, reference.trace[i].prediction, r.prediction, f(prefixDifferences[i], 9)])} /><XValues caption="All carried components at selected prefix" rows={current.state.map((state, i) => [`state component ${i + 1}`, JSON.stringify(state)])} /><XValues caption="Every saved operation at the selected prefix" rows={Object.entries(current.stages).map(([name, values]) => [name, vec(values)])} /></section>
 {pinned && <div className="xl-result xl-pinned"><p>Pinned {pinned.modelId} epoch {pinned.epoch}, source {pinned.sourceId}, {pinned.view.reversed ? 'reversed' : 'natural'} order, {pinned.view.execution} after step {pinned.view.split}: final class {pinned.prediction}. Current minus pinned class-{view.scoreClass} logit: {f(result.trace[7].logits[view.scoreClass] - pinned.logits[view.scoreClass], 9)}. A fit change keeps this saved reference intact.</p><details><summary>Complete pinned input and settings</summary><p>{JSON.stringify(pinned.view)}</p><NeuralTable caption="Pinned unnormalized pixel rows" headers={['Image row', ...Array.from({
          length: 8
        }, (_, i) => `column ${i + 1}`)]} rows={pinned.pixels.map((row, i) => [i + 1, ...row])} /><p>All ten pinned logits: {vec(pinned.logits)}</p></details></div>}
 </>;
}
export function RowReaderLab() {
  const [ref, near] = useNear(),
    [modelId, setModelId] = useState('slstm_seed19'),
    [retry, setRetry] = useState(0),
    [sourceIndex, setSourceIndex] = useState(142),
    [raw, setRaw] = useState(null),
    [selected, setSelected] = useState(40),
    [view, setView] = useState(initialReaderView),
    [pinned, setPinned] = useState(null),
    bank = useAsset('digit-rows.json', near, retry),
    fit = useAsset(`model-${modelId}.json`, near, retry),
    original = bank.data?.rows[sourceIndex],
    pixels = raw || original?.pixels,
    activeView = view;
  const edit = (index, value) => setRaw(old => (old || original.pixels).map((row, i) => row.map((v, j) => i * 8 + j === index ? value : v))),
    keyboard = (event, index) => {
      const offsets = {
        ArrowLeft: -1,
        ArrowRight: 1,
        ArrowUp: -8,
        ArrowDown: 8
      };
      if (!(event.key in offsets)) return;
      event.preventDefault();
      const next = Math.max(0, Math.min(63, index + offsets[event.key]));
      setSelected(next);
      event.currentTarget.parentElement.querySelector(`[data-xl-pixel="${next}"]`)?.focus();
    };
  return <NeuralLab id="xlstm-digit" title="Edit an image and inspect what each row leaves in memory"><div ref={ref}><p>Start with source row187, then change actual pixel values and inspect the first affected prefix. Model switches preserve your image, scan settings, selected state coordinate and pin. A blank image still runs the learned biases; it need not give a uniform class distribution.</p><div className="xl-two"><ModelChoice id={modelId} onChange={setModelId} /><NeuralNumber label="Validation image index (zero-based)" value={sourceIndex} min={0} max={299} integer onChange={n => {
          setSourceIndex(n);
          setRaw(null);
        }} /></div>
 {pixels && <><p>Selected pixel: image row {Math.floor(selected / 8) + 1}, column {selected % 8 + 1}. Arrow keys move inside the grid; the exact number control edits its intensity from0 to16.</p><div className="xl-two"><div className="xl-pixels" role="group" aria-label="Editable eight by eight input pixels">{pixels.flat().map((value, i) => <button type="button" data-xl-pixel={i} key={i} tabIndex={i === selected ? 0 : -1} aria-pressed={i === selected} aria-label={`Pixel row ${Math.floor(i / 8) + 1}, column ${i % 8 + 1}, intensity ${value}`} onClick={() => setSelected(i)} onKeyDown={event => keyboard(event, i)} style={{
              background: `rgb(${Math.round(value / 16 * 255)},${Math.round(value / 16 * 255)},${Math.round(value / 16 * 255)})`,
              color: value >= 8 ? '#000' : '#fff'
            }}>{value}</button>)}</div><NeuralNumber label={`Pixel row ${Math.floor(selected / 8) + 1}, column ${selected % 8 + 1}`} value={pixels.flat()[selected]} min={0} max={16} integer onChange={n => edit(selected, n)} /></div><div className="xl-controls"><button onClick={() => setRaw(old => (old || original.pixels).map((row, i) => i >= 5 ? row.map(() => 0) : [...row]))}>Zero current bottom three rows</button><button onClick={() => setRaw(Array.from({
            length: 8
          }, () => Array(8).fill(0)))}>Blank image</button><button onClick={() => {
            setRaw(structuredClone(original.pixels));
            setView(old => ({
              ...old,
              reversed: false
            }));
          }}>Restore original image and order</button><button disabled={!fit.data} onClick={() => {
            const result = executeReader(activeView.reversed ? [...pixels].reverse() : pixels, fit.data, activeView);
            setPinned({
              modelId,
              epoch: fit.data.selectedEpoch,
              sourceId: original.sourceId,
              pixels: structuredClone(pixels),
              view: {
                ...activeView
              },
              logits: [...result.trace[7].logits],
              prediction: result.trace[7].prediction
            });
          }}>Pin image, model and execution</button><button onClick={() => {
            setModelId('slstm_seed19');
            setSourceIndex(142);
            setRaw(null);
            setSelected(40);
            setView(initialReaderView);
            setPinned(null);
          }}>Reset digit investigation</button></div></>}
 {bank.error || fit.error ? <div role="alert"><p>The selected saved reader or image data could not be loaded. Your edits and pin are retained. Retry to continue the same experiment.</p><button onClick={() => setRetry(n => n + 1)}>Retry saved row reader</button></div> : !bank.data || !fit.data ? <p role="status">{near ? 'Loading the selected saved reader and validation images…' : 'The selected reader loads when this investigation is near the viewport.'}</p> : <RowReaderWorkspace model={fit.data} original={original} raw={pixels} view={activeView} onView={setView} pinned={pinned} />}</div></NeuralLab>;
}
export function RowLearningCurve() {
  const [ref, near] = useNear(),
    [id, setId] = useState('slstm_seed19'),
    [retry, setRetry] = useState(0),
    asset = useAsset(`model-${id}.json`, near, retry);
  return <div ref={ref}><ModelChoice id={id} onChange={setId} label="Inspect the complete retained training curve" />{asset.error ? <div role="alert"><p>The saved learning curve could not be loaded.</p><button onClick={() => setRetry(n => n + 1)}>Retry learning curve</button></div> : !asset.data ? <p role="status">{near ? 'Loading all150 recorded epochs…' : 'The curve loads near this figure.'}</p> : <LearningCurveView model={asset.data} />}</div>;
}
export function LearningCurveView({
  model
}) {
  const selected = model.trainingCurve.find(row => row.epoch === model.selectedEpoch);
  return <><XPlot title={`${model.id}: every recorded epoch and the selected checkpoint`} xLabel="epoch" yLabel="mean cross-entropy" series={[{
      label: 'fit before this epoch’s update',
      color: '#87bdf1',
      points: model.trainingCurve.map(row => [row.epoch, row.fit_loss_before_update])
    }, {
      label: 'validation after this epoch’s update',
      color: '#e6bb60',
      points: model.trainingCurve.map(row => [row.epoch, row.validation_loss_after_update])
    }, {
      label: 'minimum validation checkpoint',
      color: '#e9979f',
      points: [[model.selectedEpoch, selected.validation_loss_after_update]]
    }]} /><p>The selected epoch is {model.selectedEpoch}; validation cross-entropy {f(selected.validation_loss_after_update, 9)}. The blue and gold values within one epoch use different parameter snapshots. Selection uses validation only; the original test file stays outside fitting and selection.</p><NeuralTable caption="Executed selected-checkpoint measurements" headers={['Role/condition', 'Examples', 'Errors', 'Cross-entropy']} rows={Object.entries(model.metrics).filter(([, value]) => typeof value === 'object').map(([name, row]) => [name, row.count, row.errors, f(row.cross_entropy, 8)])} /></>;
}
export function XlstmProgram({
  filename
}) {
  return <>
    <RemoteCodeBlock source={base + filename} language="python" filename={filename} title={"Read the complete " + (filename) + " program"} />
    <p><a href={base + filename} download>Download {filename}</a></p>
  </>;
}
