import { useEffect, useMemo, useRef, useState } from 'react';
import { NeuralLab, NeuralSelect, NeuralTable, NeuralPlot as BaseNeuralPlot, formatNeural as f } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { BlockTaskDiagrams, BlockWriteStacks, BlockSystemsFigure, BlockPairedScores } from './TransformerBlockMechanisms.jsx';
import { gatedFeedforwardTrace, blockCosts, blockDefault, feedforwardTrace, movementBlockForward, normalizationProbe, traceBlock } from '../../data/transformer-block-models.js';
import { layerNorm, mean, maxDifference } from '../../data/sequence-tensor-operations.js';
import './transformer-block-labs.css';
import './neural-lesson-neutral.css';
import { BlockCommunicationDiagram, BlockWiringDiagram, BlockVariantDependencies } from './TransformerBlockDiagrams.jsx';
const base = '/learn-code/transformer-block-architecture/';
const colors = ['#e6b854', '#eeeeee', '#b8a0d2', '#999999'];
function NeuralPlot(props) {
  return <div className="block-plot-scroll" role="region" aria-label={props.title} tabIndex={0}><div className="block-plot-size"><BaseNeuralPlot {...props} /></div></div>;
}
const vector = values => `[${values.map(value => f(value, 5)).join(', ')}]`;
function VectorEditor({
  label,
  values,
  onChange,
  min = -10,
  max = 10
}) {
  return <fieldset className="block-vector-editor"><legend>{label}</legend><div className="neural-controls">{values.map((value, i) => <NeuralNumber key={i} label={`${label} ${i + 1}`} value={value} min={min} max={max} range={false} onChange={next => onChange(values.map((item, j) => i === j ? next : item))} />)}</div></fieldset>;
}
function SignedBars({
  title,
  rows
}) {
  const extent = Math.max(.01, ...rows.flatMap(row => row.values).map(Math.abs));
  return <figure className="block-bars"><figcaption>{title}</figcaption>{rows.map((row, index) => <div key={row.label}><h4>{row.label}</h4><div className="block-bar-grid">{row.values.map((value, i) => <div className="block-bar-cell" key={i}><span>{i + 1}</span><div className="block-bar-track"><i style={{
              left: `${value < 0 ? 50 + 50 * value / extent : 50}%`,
              width: `${50 * Math.abs(value) / extent}%`,
              background: colors[index % colors.length]
            }} /></div><strong>{f(value, 4)}</strong></div>)}</div></div>)}<p>Shared scale −{f(extent, 3)} to +{f(extent, 3)}; center line is zero. Bars and printed signs show direction.</p></figure>;
}
function useAsset(file) {
  const [resource, setResource] = useState({
    data: null,
    error: null
  });
  const [attempt, setAttempt] = useState(0);
  const [visible, setVisible] = useState(false);
  const container = useRef(null);
  useEffect(() => {
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        setVisible(true);
        observer.disconnect();
      }
    }, {
      rootMargin: '240px'
    });
    if (container.current) observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible) return undefined;
    const controller = new AbortController();
    setResource({
      data: null,
      error: null
    });
    fetch(base + file, {
      signal: controller.signal
    }).then(response => {
      if (!response.ok) throw new Error('The saved experiment could not be loaded.');
      return response.json();
    }).then(data => {
      if (!controller.signal.aborted) setResource({
        data,
        error: null
      });
    }).catch(error => {
      if (!controller.signal.aborted) setResource({
        data: null,
        error: error.message
      });
    });
    return () => controller.abort();
  }, [file, attempt, visible]);
  return {
    ...resource,
    container,
    retry: () => setAttempt(value => value + 1)
  };
}
function ResourceState({
  resource,
  children
}) {
  if (resource.error) return <p role="alert">{resource.error} <button onClick={resource.retry}>Retry experiment</button></p>;
  if (!resource.data) return <p role="status">Loading measured experiment…</p>;
  return children(resource.data);
}
export function BlockCommunicationFigure() {
  return <figure className="block-circuit"><figcaption>One representation per position; a shared feature function</figcaption><BlockCommunicationDiagram /><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Read the operations in each lane</h4><div className="block-lanes">{['Position 1', 'Position 2', 'Position 3'].map((label, i) => <div key={label}><strong>{label}</strong><span>4 input features</span><span>↘ ↔ ↗<br />reads permitted positions</span><span>+ saved row</span><span>same FFN<br />4 → f → 4</span><span>+ saved row</span><strong>4 output features</strong></div>)}</div></section><p>Sequence axis: the three rows. Feature axis: columns within each row. Attention mixes positions; the FFN changes each contextual row separately with shared parameters. Entering and leaving shape: (B,L,d).</p></figure>;
}
export function BlockWiringFigure() {
  const [zero, setZero] = useState(false);
  const fixture = blockDefault();
  return <figure className="block-circuit"><figcaption>Where the bypass meets normalization</figcaption><div className="block-two">{[true, false].map(pre => <section key={String(pre)}><h4>{pre ? 'Pre-norm' : 'Post-norm'}</h4><div className="block-diagram-scroll" role="region" aria-label={pre ? "Pre-norm wiring" : "Post-norm wiring"} tabIndex={0}><BlockWiringDiagram pre={pre} zero={zero} /></div><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Read the exact operation sequence</h4><ol className="block-flow"><li><strong>X</strong> saved on bypass</li><li>{pre ? 'Norm₁(X) → Attention' : 'Attention(X)'} → {zero ? '0 update' : 'U'}</li><li>X + U {pre ? '= Z' : '→ Norm₁ → Z'}</li><li>{pre ? 'Norm₂(Z) → FFN' : 'FFN(Z)'} → {zero ? '0 update' : 'V'}</li><li>Z + V {pre ? '= Y' : '→ Norm₂ → Y'}</li></ol></section><p>First output row: {vector(traceBlock({
            ...fixture,
            pre,
            branch: zero ? 0 : 1
          }).output[0])}</p></section>)}</div><button aria-pressed={zero} onClick={() => setZero(value => !value)}>{zero ? 'Restore branch updates' : 'Set both branch updates to zero'}</button><p>The bypass starts at X, then at the already updated Z. With zero updates, only pre-norm preserves X exactly.</p></figure>;
}
export function BlockNormalizationLab() {
  const [input, setInput] = useState([1, 2, 5, 8]);
  const [offset, setOffset] = useState(5),
    [multiplier, setMultiplier] = useState(1),
    [epsilon, setEpsilon] = useState(1e-5);
  const changed = input.map(value => multiplier * value + offset);
  const denominators = row => {
    const variance = mean(row.map(x => (x - mean(row)) ** 2));
    const squareMean = mean(row.map(x => x * x));
    return { layer: Math.sqrt(variance + epsilon), rms: Math.sqrt(squareMean + epsilon) };
  };
  const originalDenominator = denominators(input), changedDenominator = denominators(changed);
  const rms = row => row.map(value => value / Math.sqrt(mean(row.map(item => item * item)) + epsilon));
  const lnOriginal = layerNorm(input, [], [], epsilon),
    ln = layerNorm(changed, [], [], epsilon);
  const rmsOriginal = rms(input),
    rmsChanged = rms(changed);
  return <NeuralLab id="block-normalization" title="Choose the normalizer’s reference frame">
    <p>LayerNorm measures spread around the row’s mean. RMSNorm measures distance from zero. Edit one feature, then apply the common shift and scale.</p>
    <VectorEditor label="Original feature" values={input} onChange={setInput} />
    <div className="neural-controls"><NeuralNumber label="Common offset" value={offset} min={-10} max={10} onChange={setOffset} /><NeuralNumber label="Common scale" value={multiplier} min={0} max={4} onChange={setMultiplier} /><NeuralNumber label="Normalization epsilon" value={epsilon} min={1e-8} max={.1} range={false} onChange={setEpsilon} /></div>
    <SignedBars title="Original and transformed results" rows={[{
      label: 'Original input',
      values: input
    }, {
      label: 'Transformed input',
      values: changed
    }, {label: 'Original deviations from mean', values: input.map(x => x - mean(input))}, {label: 'Transformed deviations from mean', values: changed.map(x => x - mean(changed))}]} />
    <SignedBars title="Same gain 1; LayerNorm offset 0" rows={[{
      label: 'Original LayerNorm',
      values: lnOriginal
    }, {
      label: 'Transformed LayerNorm',
      values: ln
    }, {
      label: 'Original RMSNorm',
      values: rmsOriginal
    }, {
      label: 'Transformed RMSNorm',
      values: rmsChanged
    }]} />
    <NeuralTable caption="The actual denominators and lengths" headers={['Quantity', 'Original', 'Transformed']} rows={[['Mean', f(mean(input)), f(mean(changed))], ['Mean squared deviation', f(mean(input.map(value => (value - mean(input)) ** 2))), f(mean(changed.map(value => (value - mean(changed)) ** 2)))], ['Mean square from zero', f(mean(input.map(value => value * value))), f(mean(changed.map(value => value * value)))], ['LayerNorm denominator: sqrt(variance + epsilon)', f(originalDenominator.layer), f(changedDenominator.layer)], ['RMSNorm denominator: sqrt(mean square + epsilon)', f(originalDenominator.rms), f(changedDenominator.rms)], ['LN L2 length', f(Math.hypot(...lnOriginal)), f(Math.hypot(...ln))], ['LN RMS', f(Math.hypot(...lnOriginal) / 2), f(Math.hypot(...ln) / 2)]]} />
    <p data-result="normalization">Maximum LayerNorm change {f(maxDifference(lnOriginal, ln))}; RMSNorm change {f(maxDifference(rmsOriginal, rmsChanged))}. An offset is removed only by centering.</p>
    <div className="neural-buttons"><button onClick={() => {
        setInput([-1, -1, 1, 1]);
        setOffset(0);
        setMultiplier(1);
      }}>Zero-mean comparison</button><button onClick={() => {
        setInput([1, 2, 5, 8]);
        setOffset(5);
        setMultiplier(1);
        setEpsilon(1e-5);
      }}>Reset normalizers</button></div>
  </NeuralLab>;
}
export function BlockFeedforwardLab() {
  const fixture = blockDefault();
  const [input, setInput] = useState([1, 2, -1, -2]),
    [up, setUp] = useState(fixture.up),
    [down, setDown] = useState(fixture.down);
  const [activation, setActivation] = useState('relu');
  const [second, setSecond] = useState([-1, 0, 1, 0]);
  const [gateMatrix, setGateMatrix] = useState([[1,0,0],[0,-1,0],[0,0,1],[0,0,0]]);
  const [detector,setDetector] = useState(0);
  const result = feedforwardTrace(input, up, down, activation);
  const gated = gatedFeedforwardTrace(input,up,gateMatrix,down);
  return <NeuralLab id="block-feature-write" title="A response reads a pattern, then writes a direction">
    <VectorEditor label="Direct FFN input" values={input} onChange={setInput} />
    <NeuralSelect label="Activation" value={activation} onChange={setActivation} options={['relu', 'gelu', 'silu'].map(name => [name, name.toUpperCase()])} />
    <SignedBars title="The feature circuit" rows={[{
      label: 'Dot-product responses',
      values: result.responses
    }, {
      label: 'Activated responses',
      values: result.hidden
    }, {
      label: 'Written update',
      values: result.output
    }]} />
    <VectorEditor label="Second position direct input" values={second} onChange={setSecond} />
    <p>Second position update under the same matrices: {vector(feedforwardTrace(second, up, down, activation).output)}. Editing this position alone leaves the first position’s direct update unchanged. Editing a shared matrix can affect both.</p>
    <NeuralSelect label="Inspect hidden detector and write row" value={detector} onChange={v=>setDetector(Number(v))} options={result.hidden.map((_,i)=>[i,`Hidden coordinate ${i+1}`])} />
    <p>Detector {detector+1}: input {vector(input)} dotted with up column {vector(up.map(row=>row[detector]))} gives {f(result.responses[detector])}; {activation.toUpperCase()} gives {f(result.hidden[detector])}. Multiply the entire down row {vector(down[detector])} by this scalar to obtain {vector(result.contributions[detector])}.</p>
    <BlockWriteStacks contributions={result.contributions} output={result.output} selected={detector} />
    <NeuralTable caption="Signed writes sum to the output" headers={['Hidden response', 'Activation', 'Write contribution']} rows={result.contributions.map((row, i) => [i + 1, f(result.hidden[i]), vector(row)])} />
    <details><summary>Edit the detector and write matrices</summary>{[['Up matrix input row', up, setUp], ['Down matrix hidden row', down, setDown]].map(([label, matrix, setter]) => <div key={label}>{matrix.map((row, i) => <VectorEditor key={i} label={`${label} ${i + 1}`} values={row} onChange={next => setter(matrix.map((other, j) => i === j ? next : other))} min={-3} max={3} />)}</div>)}</details>
    <h4>A gate reads the same current input through a second projection</h4>
    <p>The up matrix supplies value responses u; the gate matrix supplies logits g. Each hidden coordinate becomes u × SiLU(g), then the same down rows write the result. No bias is used in this small circuit.</p>
    <details><summary>Edit the gate projection</summary>{gateMatrix.map((row,i)=><VectorEditor key={i} label={`Gate matrix input row ${i+1}`} values={row} min={-3} max={3} onChange={next=>setGateMatrix(previous=>previous.map((r,j)=>i===j?next:r))} />)}</details>
    <NeuralTable caption="Current input through all three SwiGLU maps" headers={['Hidden coordinate','u = x·up column','g = x·gate column','SiLU(g)','u × SiLU(g)']} rows={gated.hidden.map((value,i)=>[i+1,f(gated.value[i]),f(gated.gateLogits[i]),f(gated.gateResponse[i]),f(value)])} />
    <BlockWriteStacks contributions={gated.contributions} output={gated.output} selected={detector} />
    <p>Current gated output {vector(gated.output)}. A negative SiLU response can reverse a write; it is not a probability.</p>
    <button onClick={() => {
      setInput([1, 2, -1, -2]);
      setSecond([-1, 0, 1, 0]);
      setUp(fixture.up);
      setDown(fixture.down);
      setActivation('relu');
      setGateMatrix([[1,0,0],[0,-1,0],[0,0,1],[0,0,0]]);
      setDetector(0);
    }}>Reset feature circuit</button>
  </NeuralLab>;
}
export function BlockTraceLab() {
  const [settings, setSettings] = useState(blockDefault),
    [position, setPosition] = useState(0);
  const result = traceBlock(settings),
    comparison = traceBlock({
      ...settings,
      pre: !settings.pre
    });
  return <NeuralLab id="block-trace" title="Follow every junction in a complete block">
    <p>Exact two-position fixture, not a fitted model: Q=K=H, V=H/4, one head, ReLU FFN, no biases/dropout. Moving normalization changes which H attention reads.</p>
    {settings.inputs.map((row, i) => <VectorEditor key={i} label={`Position ${i + 1} input`} values={row} onChange={next => setSettings(previous => ({
      ...previous,
      inputs: previous.inputs.map((other, j) => i === j ? next : other)
    }))} />)}
    <div className="neural-controls"><NeuralSelect label="Normalization placement" value={settings.pre ? 'pre' : 'post'} options={['pre', 'post'].map(name => [name, `${name}-norm`])} onChange={value => setSettings(previous => ({
        ...previous,
        pre: value === 'pre'
      }))} /><NeuralNumber label="Both branch multipliers λ" value={settings.branch} min={0} max={2} onChange={branch => setSettings(previous => ({
        ...previous,
        branch
      }))} /><NeuralNumber label="Inspect position" value={position + 1} min={1} max={2} integer step={1} onChange={value => setPosition(value - 1)} /></div>
    <SignedBars title="Carried state versus branch update" rows={[{
      label: 'Saved X',
      values: settings.inputs[position]
    }, {
      label: 'Attention update',
      values: result.update[position]
    }, {
      label: 'Context Z',
      values: result.context[position]
    }, {
      label: 'FFN update',
      values: result.ffnUpdate[position]
    }, {
      label: 'Output Y',
      values: result.output[position]
    }]} />
    <NeuralTable caption={`Exact trace for position ${position + 1}`} headers={['Stage', 'Vector']} rows={Object.entries(result).filter(([, value]) => Array.isArray(value)).map(([name, values]) => [name, vector(values[position])])} />
    <p data-result="block">Same input and weights with the other placement: {vector(comparison.output[position])}. λ=0 leaves both branch outputs zero, yet the post-norm path still normalizes.</p>
    <details><summary>Edit the FFN weights used by this whole block</summary>{[['up', settings.up], ['down', settings.down]].map(([name, matrix]) => matrix.map((row, i) => <VectorEditor key={`${name}${i}`} label={`${name} row ${i + 1}`} values={row} min={-3} max={3} onChange={next => setSettings(previous => ({
        ...previous,
        [name]: matrix.map((other, j) => i === j ? next : other)
      }))} />))}</details>
    <button onClick={() => {
      setSettings(blockDefault());
      setPosition(0);
    }}>Reset whole block</button>
  </NeuralLab>;
}
export function BlockProbeLab() {
  const [input, setInput] = useState([1, 2, 5, 8]),
    [probe, setProbe] = useState([1, 1, 1, 1]),
    [step, setStep] = useState(1e-5);
  const result = normalizationProbe(input, probe, 1e-5, step);
  return <NeuralLab id="block-probe" title="Choose an output question that can change">
    <p>The measured scalar is pᵀLayerNorm(x). The input gradient describes this particular weighted sum; it is not a generic score of a layer.</p>
    <VectorEditor label="Probe input x" values={input} onChange={setInput} /><VectorEditor label="Output probe p" values={probe} onChange={setProbe} />
    <NeuralNumber label="Finite-difference step" value={step} min={1e-8} max={.1} range={false} onChange={setStep} />
    <SignedBars title="Analytic and numerical sensitivities" rows={[{
      label: 'Analytic Jᵀp',
      values: result.gradient
    }, {
      label: 'Central difference',
      values: result.finite
    }]} />
    <p data-result="probe">Scalar {f(result.scalar)}; gradient L2 {f(Math.hypot(...result.gradient))}; maximum finite-difference discrepancy {maxDifference(result.gradient, result.finite).toExponential(3)}.</p>
    <p>Equal probe weights sum centered features to zero. A contrast can change. Small input spread can amplify a contrast rather than contract it.</p>
    <div className="neural-buttons"><button onClick={() => setProbe([1, -1, 0, 0])}>Contrast probe</button><button onClick={() => setInput([-.03, -.01, .01, .03])}>Small-spread input</button><button onClick={() => {
        setInput([1, 2, 5, 8]);
        setProbe([1, 1, 1, 1]);
        setStep(1e-5);
      }}>Reset probe</button></div>
  </NeuralLab>;
}
function MovementExplorer({
  data
}) {
  const [placement, setPlacement] = useState('pre-norm'),
    [points, setPoints] = useState(data['pre-norm'].points),
    [times, setTimes] = useState(data['pre-norm'].times);
  const [selected, setSelected] = useState(22),
    [layer, setLayer] = useState(0),
    [paddingMode, setPaddingMode] = useState('none');
  const source = data[placement];
  const result = useMemo(() => {
    const padded = paddingMode !== 'none';
    return movementBlockForward(source, padded ? [...points, ...Array.from({
      length: 5
    }, () => [.75, .75])] : points, padded ? [...times, 0, 0, 0, 0, 0] : times, paddingMode === 'masked' ? [...points.map(() => false), true, true, true, true, true] : []);
  }, [source, points, times, paddingMode]);
  const trace = result.traces[layer];
  const predicted = result.probabilities.indexOf(Math.max(...result.probabilities)) + 1;
  const path = values => values.map(([x, y]) => `${35 + x * 250},${285 - y * 250}`).join(' ');
  return <>
    <p>Saved seed-101 parameters; live inference. Source row 77 is class 4 (anticlockwise arc). Both original models misclassify it. Edited paths retain no newly measured label.</p>
    <div className="neural-controls"><NeuralSelect label="Fitted placement" value={placement} onChange={setPlacement} options={['pre-norm', 'post-norm'].map(name => [name, name])} /><NeuralNumber label="Selected movement point" value={selected + 1} min={1} max={45} step={1} integer onChange={value => setSelected(value - 1)} /><NeuralNumber label="Point x" value={points[selected][0]} min={0} max={1} onChange={value => setPoints(previous => previous.map((row, i) => i === selected ? [value, row[1]] : row))} /><NeuralNumber label="Point y" value={points[selected][1]} min={0} max={1} onChange={value => setPoints(previous => previous.map((row, i) => i === selected ? [row[0], value] : row))} /><NeuralNumber label="Point time tag" value={times[selected]} min={-2} max={2} onChange={value => setTimes(previous => previous.map((item, i) => i === selected ? value : item))} /><NeuralSelect label="Five appended padding records" value={paddingMode} onChange={setPaddingMode} options={['none', 'masked', 'unmasked'].map(name => [name, name])} /></div>
    <div className="block-two"><figure className="block-path" tabIndex={0}><figcaption>Observed movement and current edited path</figcaption><svg viewBox="0 0 320 320" role="img" aria-label="Equal-scale x and y coordinates, zero to one. Gray dashed source path; amber edited path."><rect x="35" y="35" width="250" height="250" fill="none" stroke="#777" />{[0, .5, 1].map(value => <g key={value}><text x={35 + value * 250} y="307" textAnchor="middle">{value}</text><text x="27" y={289 - value * 250} textAnchor="end">{value}</text></g>)}<polyline points={path(source.points)} fill="none" stroke="#888" strokeDasharray="4 3" strokeWidth="2" /><polyline points={path(points)} fill="none" stroke="#e6b854" strokeWidth="2" /><circle cx={35 + points[0][0] * 250} cy={285 - points[0][1] * 250} r="5" fill="#fff" /><rect x={31 + points[44][0] * 250} y={281 - points[44][1] * 250} width="8" height="8" fill="#e6b854" /><circle cx={35 + points[selected][0] * 250} cy={285 - points[selected][1] * 250} r="8" fill="none" stroke="#fff" strokeWidth="2" /></svg><p>x horizontal; y vertical. White start circle; amber end square. Ring: selected point.</p></figure><div><p data-result="movement">Current predicted class <strong>{predicted}</strong>; original class-4 probability {f(source.original.probabilities[3], 6)} → <strong>{f(result.probabilities[3], 6)}</strong>. Maximum logit change {f(maxDifference(result.logits, source.original.logits))}.</p><NeuralTable caption="All class probabilities" headers={['Class', 'Original', 'Current']} rows={result.probabilities.map((value, i) => [i + 1, f(source.original.probabilities[i], 4), f(value, 4)])} /></div></div>
    <div className="neural-buttons"><button onClick={() => setPoints(previous => [...previous].reverse())}>Reverse coordinates only</button><button onClick={() => {
        setPoints(previous => [...previous].reverse());
        setTimes(previous => [...previous].reverse());
      }}>Reverse complete point/time records</button><button onClick={() => {
        setPoints(source.points);
        setTimes(source.times);
        setPaddingMode('none');
        setSelected(22);
      }}>Reset movement</button></div>
    <NeuralSelect label="Inspect block" value={String(layer)} onChange={value => setLayer(Number(value))} options={[[0, 'Block 1'], [1, 'Block 2']]} />
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect all feature stages for this position</h4><SignedBars title={`Actual feature values, block ${layer + 1}, position ${selected + 1}`} rows={['input', 'attentionInput', 'update', 'context', 'ffnInput', 'ffnUpdate', 'output'].map(name => ({
      label: name,
      values: trace[name][selected]
    }))} /></section>
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect both attention heads and hidden FFN responses</h4><NeuralTable caption="One receiver, 45 valid keys" headers={['Key position', 'Head 1', 'Head 2']} rows={points.map((_, i) => [i + 1, f(trace.weights[0][selected][i], 5), f(trace.weights[1][selected][i], 5)])} /><p>Hidden responses: {vector(trace.hidden[selected])}</p></section>
  </>;
}
export function BlockMovementLab() {
  const resource = useAsset('movement-runtime.json');
  return <div ref={resource.container}><NeuralLab id="block-movement" title="What did two fitted blocks use?"><ResourceState resource={resource}>{data => <MovementExplorer data={data} />}</ResourceState></NeuralLab></div>;
}
export function BlockEvidenceFigure({
  gradients = false
}) {
  const resource = useAsset('evidence-runtime.json');
  const [selection, setSelection] = useState(0);
  return <div ref={resource.container}><ResourceState resource={resource}>{data => {
        if (gradients) {
          const depth = [1, 4, 12][selection % 3],
            rows = data.gradientDepth.filter(row => row.depth === depth);
          return <figure className="block-circuit"><figcaption>Measured sensitivity to a unit random output probe</figcaption><NeuralSelect label="Measured stack depth" value={String(selection % 3)} onChange={value => setSelection(Number(value))} options={[[0, '1 block'], [1, '4 blocks'], [2, '12 blocks']]} />{[['state_gradient_l2', 'State gradient L2'], ['state_rms', 'Carried-state RMS']].map(([key, title]) => <NeuralPlot key={key} title={title} xLabel="carried state index" yLabel={title} xDomain={[0, depth]} yDomain={[0, 1.1 * Math.max(...rows.flatMap(row => row[key]))]} series={rows.map((row, i) => ({
              label: row.pre_norm ? 'pre-norm' : 'post-norm',
              color: colors[i],
              values: row[key].map((value, index) => [index, value])
            }))} />)}<NeuralTable caption="Every measured state" headers={['State', 'Pre gradient L2', 'Post gradient L2', 'Pre RMS', 'Post RMS']} rows={rows[0].state_rms.map((_, i) => [i, ...['state_gradient_l2', 'state_rms'].flatMap(key => rows.map(row => f(row[key][i])))])} /><p>CPU initialization, seeds 53/97; width 8, 2 heads, final affine-free LayerNorm, unit random probe. These are measured vector-Jacobian products, not singular-value bounds or trained quality.</p></figure>;
        }
        const fit = data.fits[selection % 6];
        return <figure className="block-circuit"><figcaption>Six observed fits on one fixed 220/50/60 split</figcaption><NeuralSelect label="Inspect measured learning history" value={String(selection % 6)} onChange={value => setSelection(Number(value))} options={data.fits.map((row, i) => [i, `${row.placement}, seed ${row.seed}`])} /><NeuralPlot title={`Cross entropy; selected epoch ${fit.selected_epoch} by validation F1`} xLabel="epoch" yLabel="nats / example" xDomain={[1, 180]} yDomain={[0, 1.05 * Math.max(...fit.history.flatMap(row => [row.train.loss, row.validation.loss]))]} series={['train', 'validation'].map((key, i) => ({
            label: key,
            color: colors[i],
            values: fit.history.map(row => [row.epoch, row[key].loss])
          }))} points={[{
            id: 'selected',
            x: fit.selected_epoch,
            y: fit.history[fit.selected_epoch - 1].validation.loss,
            label: `Selected epoch ${fit.selected_epoch}`,
            selected: true
          }]} /><BlockPairedScores fits={data.fits} /><NeuralTable caption="Validation-selected checkpoint; all six outcomes" headers={['Placement', 'Seed', 'Epoch', 'Test correct / 60', 'Test macro F1']} rows={data.fits.map(row => [row.placement, row.seed, row.selected_epoch, row.test.correct, f(row.test.macro_f1, 6)])} /><p>The recorded training/validation traces support checkpoint inspection. No per-epoch test curve was collected. Seed 103 reverses the direction of the placement comparison seen in seeds 101 and 102.</p></figure>;
      }}</ResourceState></div>;
}
export function BlockTaskFigure() {
 return <figure className="block-circuit"><figcaption>Which information can reach the prediction?</figcaption><BlockTaskDiagrams /><p>Rows in each mask are queries; columns are stored keys and values. A check marks an allowed read. Encoder reads can connect all valid source positions. Decoder input rows are shifted relative to their next-token targets. Cross-attention draws Q from the decoder and K/V from the encoder, with its own source padding mask.</p></figure>;
}
export { BlockSystemsFigure };
export function BlockVariantsFigure() {
  return <figure className="block-circuit"><figcaption>Branch inputs are dependencies, not interchangeable labels</figcaption><BlockVariantDependencies /><div className="block-two"><section><h4>Sequential</h4><ol className="block-flow"><li>x → N₁ → A → +x = z</li><li>z → N₂ → F → +z = y</li></ol><p>The FFN reads this block’s attention update through z.</p></section><section><h4>Parallel</h4><ol className="block-flow"><li>x → N → A</li><li>the same N(x) → F</li><li>x + A(N(x)) + F(N(x)) = y</li></ol><p>Both branches read the incoming state; the FFN cannot read this block’s attention output.</p></section></div><NeuralTable caption="Locate every additional normalizer" headers={['Wiring', 'Path from saved x to y']} rows={[['Branch-output normalization', 'x + [F(x) → N]'], ['Norms on both sides of branch', 'x + [x → N_in → F → N_out]'], ['Scaled post-norm', '[αx + F(x)] → N']]} /><p>These are real alternative dependency structures. Their published recipes include initialization and other choices; the equations expose the local difference without ranking their quality.</p></figure>;
}
export function BlockCostsLab() {
  const [length, setLength] = useState(512),
    [width, setWidth] = useState(512);
  const result = blockCosts(length, width, 4 * width);
  return <NeuralLab id="block-cost" title="Which cost grows when the sequence grows?"><p>Derived arithmetic: B=1, 8 heads, ordinary equal-total-width attention, f=4d, biased two-map FFN, two affine LayerNorms. No measured latency is implied.</p><div className="neural-controls"><NeuralNumber label="Sequence length L" value={length} min={64} max={8192} integer step={64} onChange={setLength} /><NeuralNumber label="Feature width d" value={width} min={64} max={1024} integer step={64} onChange={setWidth} /></div><NeuralPlot title="Linear maps and pairwise interactions" xLabel="sequence length" yLabel="billion MACs" xDomain={[64, 8192]} yDomain={[0, Math.max(blockCosts(8192, width, 4 * width).maps, blockCosts(8192, width, 4 * width).pairs) / 1e9 * 1.05]} series={['maps', 'pairs'].map((key, i) => ({
      label: key === 'maps' ? 'Shared projections + FFN' : 'Attention score + mixture',
      color: colors[i],
      values: Array.from({
        length: 33
      }, (_, j) => {
        const value = 64 + (8192 - 64) * j / 32;
        return [value, blockCosts(value, width, 4 * width)[key] / 1e9];
      })
    }))} /><NeuralTable caption="Current derived quantities" headers={['Quantity', 'Value', 'Unit']} rows={Object.entries(result).map(([name, value]) => [name, value.toLocaleString(), name === 'parameters' ? 'scalars' : ['maps', 'pairs'].includes(name) ? 'MACs' : 'entries'])} /><p>Increasing L leaves parameter count fixed. Materialized attention grows as L²; the ordinary per-layer K/V cache grows as L. Float32 bytes equal four times the respective entry count, before runtime overhead.</p><button onClick={() => {
      setLength(512);
      setWidth(512);
    }}>Reset cost model</button></NeuralLab>;
}
