import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import { useEffect, useMemo, useRef, useState } from 'react';
import { NeuralLab, NeuralTable } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';

import { Figure, Node, Arrow, Plot, Stems, Vector, Values, DnaStrip, Probabilities, f, vec, positionLabel } from './HyenaPrimitives.jsx';
import { alphabet, classes, generateFilter, prepareHyenaModel, spliceForward } from '../../data/hyena-convolution-models.js';
import { hyenaExamples as examples } from '../../data/hyena-example-inputs.js';
const base = '/learn-code/hyena-long-convolution-models/';
export const hyenaModelIds = ['linear_29', 'gated_29', 'ungated_29', 'gated_71'];
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
function Loading({
  error,
  onRetry
}) {
  return <div role={error ? 'alert' : 'status'}>{error ? <><p>The saved calculation could not load. Your edits are retained. Check the connection and retry.</p><button onClick={onRetry}>Retry saved Hyena data</button></> : <p>Loading the selected saved fit…</p>}</div>;
}
export function FilterReadView({
  model,
  block = 0,
  channel = 0
}) {
  const filter = useMemo(() => generateFilter(model, block), [model, block]);
  return <><Figure title="H08 · The actual fitted coordinate-to-filter map" height={250} description={`Saved ${model.id}, block ${block}, channel ${channel}. All60 lag coordinates use the fixed denominator59; plotted coefficients are fitted, not a hand-drawn curve.`}><Node x={10} y={15} width={155} height={95} lines={['lag r=0…59', 's=r/59', '1+4sin+4cos features']} /><Arrow x1={165} y1={62} x2={205} y2={62} /><Node x={205} y={15} width={180} height={95} lines={['Linear9→32', 'sin activation', 'Linear32→16']} /><Arrow x1={385} y1={62} x2={425} y2={62} /><Node x={425} y={15} width={205} height={95} lines={['raw × positive envelope', 'exp(−s softplus(a))', '16 channel coefficients']} /><text x="15" y="151">One parameter set is reused at every lag. Per filter:288+32+512+16+16=864.</text><text x="15" y="186">The9 features are s plus sin/cos pairs at frequencies1,2,4,8.</text><text x="15" y="225">This complete saved classifier uses two such filters, one in each block.</text></Figure><Plot title={`Saved ${model.id} · block ${block} · channel ${channel}`} xLabel="Lag r, fixed coordinate r/59" yLabel="Coefficient / envelope" series={[{
      label: 'Raw fitted filter network',
      values: filter.raw.map((row, r) => [r, row[channel]])
    }, {
      label: 'Positive fitted decay envelope',
      values: filter.window.map((row, r) => [r, row[channel]])
    }, {
      label: 'Product used as long filter',
      values: filter.kernel.map((row, r) => [r, row[channel]])
    }]} /></>;
}
export function LearnedFilterFigure() {
  const [ref, near] = useNear(),
    [retry, setRetry] = useState(0),
    [block, setBlock] = useState(0),
    [channel, setChannel] = useState(0),
    state = useAsset('model-gated_29.json', near, retry);
  return <div ref={ref}><div className="neural-controls"><NeuralNumber label="Fitted filter block" value={block} min={0} max={1} integer onChange={setBlock} /><NeuralNumber label="Fitted filter channel" value={channel} min={0} max={15} integer onChange={setChannel} /></div>{state.data ? <FilterReadView model={state.data} block={block} channel={channel} /> : <Loading error={state.error} onRetry={() => setRetry(v => v + 1)} />}</div>;
}
const initial = examples.freshSequence;
export const initialDnaState = () => ({
  rowIndex: 1,
  original: initial.sequence,
  sequence: initial.sequence,
  raw: initial.sequence,
  sourceId: initial.source_id,
  label: 'EI',
  selected: 30,
  spanEnd: 31,
  fill: 'A',
  gatesOff: false,
  kernelLimit: 60,
  block: 0,
  channel: 0,
  pinned: null
});
export function DnaReadView({
  model,
  state
}) {
  const prepared = useMemo(() => prepareHyenaModel(model), [model]),
    options = {
      gatesOff: state.gatesOff,
      kernelLimit: state.kernelLimit,
      preparedFilters: prepared
    },
    current = useMemo(() => spliceForward(state.sequence, model, options), [state.sequence, model, prepared, state.gatesOff, state.kernelLimit]),
    original = useMemo(() => spliceForward(state.original, model, options), [state.original, model, prepared, state.gatesOff, state.kernelLimit]),
    pinned = useMemo(() => state.pinned ? spliceForward(state.pinned.sequence, model, {
      ...options,
      gatesOff: state.pinned.gatesOff,
      kernelLimit: state.pinned.kernelLimit
    }) : null, [state.pinned, model, prepared]),
    changed = [...state.sequence].flatMap((v, i) => v !== state.original[i] ? [i] : []),
    prefix = changed.length ? Math.min(...changed) : 60,
    hiddenDelta = current.hidden && original.hidden ? Math.max(0, ...current.hidden.slice(0, prefix).flatMap((row, t) => row.map((v, c) => Math.abs(v - original.hidden[t][c])))) : null,
    block = current.blocks[state.block];
  return <><p className="hy-result" aria-live="polite">Saved {model.id}, selected epoch {model.selectedEpoch}: predicted class {classes[current.prediction]}. {changed.length} edited positions. {model.kind === 'linear' ? 'The linear baseline has neither gates nor a long convolution.' : `Current computation retains lags0–${state.kernelLimit - 1}; ${state.gatesOff ? 'gates overridden to1' : 'saved gate behavior'}.`}</p><Probabilities title="Current sequence and exact original through the same current fit" current={current.probabilities} reference={original.probabilities} /><NeuralTable caption="Current fit at controlled inputs" headers={['Class', 'Original logit', 'Current logit', 'Original probability', 'Current probability']} rows={classes.map((label, i) => [label, f(original.logits[i], 8), f(current.logits[i], 8), f(original.probabilities[i], 8), f(current.probabilities[i], 8)])} /><p>Original source {state.sourceId} has observed label {state.label}. {changed.length ? 'The edited string is synthetic and has no newly measured biological label.' : 'The current string exactly restores that original source.'} {model.kind === 'ungated' ? 'This fit was trained with unit gates, so the gates-off control is an exact no-op.' : ''}</p>{pinned && <><Probabilities title="Same selected fit: current versus pinned sequence/settings" current={current.probabilities} reference={pinned.probabilities} referenceLabel="Pinned sequence and computation settings, recomputed with current fit" /><p className="hy-result hy-pinned">Pin created while viewing {state.pinned.modelId}; it contains sequence and computation settings. It is recomputed through {model.id} for a controlled model comparison. Pinned gates {state.pinned.gatesOff ? 'unit' : 'saved'}, retained taps{state.pinned.kernelLimit}; pinned logits{vec(pinned.logits)}.</p></>}{block && <><Figure title={`Current block ${state.block}, channel ${state.channel}, final receiving position 59`} height={220} description="The final received contribution sums causal transmitted values, adds a learned current-position skip and then applies the receiving gate."><Node x={10} y={15} width={180} height={65} lines={['sender stream k×v', `selected current=${f(block.transmitted[state.selected][state.channel], 5)}`]} /><Arrow x1={190} y1={47} x2={230} y2={47} /><Node x={230} y={15} width={180} height={65} lines={['Σ h₅₉₋ⱼ kⱼvⱼ', f(block.convolved[59][state.channel], 6)]} /><Arrow x1={410} y1={47} x2={450} y2={47} /><Node x={450} y={15} width={180} height={65} lines={['+ current skip, then ×q', f(block.mixed[59][state.channel], 6)]} /><text x="15" y="121">Selected sender{state.selected}: k={f(block.key[state.selected][state.channel], 5)}, v={f(block.value[state.selected][state.channel], 5)}.</text><text x="15" y="155">Final receiving gate q₅₉={f(block.query[59][state.channel], 6)}.</text><text x="15" y="192">Output projection and residual/FFN paths then continue through the full classifier.</text></Figure><Plot title={`Current transmitted values and long-filter contributions to receiver59`} xLabel="Sender position j" yLabel="Signed channel value" points={[{
        x: state.selected,
        y: block.transmitted[state.selected][state.channel],
        label: 'Selected sender'
      }]} series={[{
        label: 'Transmitted kⱼvⱼ',
        values: block.transmitted.map((row, j) => [j, row[state.channel]])
      }, {
        label: 'h₅₉₋ⱼ kⱼvⱼ (before skip and q)',
        values: block.transmitted.map((row, j) => [j, row[state.channel] * block.kernel[59 - j][state.channel]])
      }]} /><p>Before the earliest edit (array index{prefix}), maximum final hidden-vector difference is {f(hiddenDelta, 12)} across {prefix} positions. {prefix === 0 ? 'There are no earlier positions in this comparison.' : 'Those positions cannot see the changed suffix.'} The final class reads all 60 observed symbols, so its score can still change.</p><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">All final hidden vectors and selected block intermediates</h4><NeuralTable caption="Current full forward pass" headers={['Array position', 'Selected q', 'Selected k', 'Selected v', 'Selected convolved', 'Final hidden vector']} rows={current.hidden.map((row, t) => [t, f(block.query[t][state.channel], 7), f(block.key[t][state.channel], 7), f(block.value[t][state.channel], 7), f(block.convolved[t][state.channel], 7), vec(row)])} /></section><FilterReadView model={model} block={state.block} channel={state.channel} /></>}</>;
}
export function DnaStudy() {
  const [ref, near] = useNear(),
    [modelId, setModelId] = useState('gated_29'),
    [retry, setRetry] = useState(0),
    [state, setState] = useState(initialDnaState),
    model = useAsset('model-' + modelId + '.json', near, retry),
    data = useAsset('validation-sequences.json', near, retry),
    update = patch => setState(s => ({
      ...s,
      ...patch
    })),
    setSequence = sequence => update({
      sequence,
      raw: sequence
    }),
    invalid = state.raw !== state.sequence,
    chooseRow = index => {
      const row = data.data?.rows[index];
      if (row) update({
        rowIndex: index,
        sourceId: row.sourceId,
        label: row.label,
        original: row.sequence,
        sequence: row.sequence,
        raw: row.sequence
      });
    },
    editBase = (index, base) => setSequence([...state.sequence].map((v, i) => i === index ? base : v).join('')),
    spanStart = Math.min(state.selected, state.spanEnd),
    spanEnd = Math.max(state.selected, state.spanEnd);
  return <NeuralLab id="hyena-dna" title="HC · Which observed context changes this saved model’s answer?"><div ref={ref}><p>All60 characters are observed input to this classifier. Change any supported symbol and compare actual logits from a full saved network. Changing the fit keeps your current sequence, view settings and pin.</p><label className="hy-select">Saved fit<select value={modelId} onChange={e => setModelId(e.target.value)}>{hyenaModelIds.map(id => <option key={id}>{id}</option>)}</select></label>{data.data ? <NeuralNumber label="Validation row index (0–459)" value={state.rowIndex} min={0} max={459} integer range={false} onChange={chooseRow} /> : <Loading error={data.error} onRetry={() => setRetry(v => v + 1)} />}<p>Current original: source row{state.sourceId}, observed label {state.label}. Validation examples support inspection; assessment rows remain in the fixed reported evaluation.</p><label className="hy-select">Editable complete60-character sequence<textarea className="hy-sequence-input" value={state.raw} aria-invalid={invalid} onChange={e => {
          const raw = e.target.value.toUpperCase().replace(/\s/g, ''),
            valid = raw.length === 60 && [...raw].every(c => alphabet.includes(c));
          update(valid ? {
            raw,
            sequence: raw
          } : {
            raw
          });
        }} /></label>{invalid && <p className="hy-error" role="alert">Enter exactly 60 symbols from A,C,G,T,D,N,R,S. Outputs retain the last valid sequence below.</p>}<div className="hy-base-grid" aria-label="Select an observed position">{[...state.sequence].map((v, i) => <button key={i} aria-label={`Array${i}, biological${positionLabel(i)}, ${v}`} aria-pressed={state.selected === i} data-changed={v !== state.original[i]} data-boundary={i === 29 || i === 30} onClick={() => update({
          selected: i
        })}><span>{positionLabel(i) > 0 ? '+' : ''}{positionLabel(i)}</span>{v}</button>)}</div><div className="neural-controls"><NeuralNumber label="Selected array index" value={state.selected} min={0} max={59} integer onChange={selected => update({
          selected
        })} /><label className="hy-select">Selected symbol at biological{positionLabel(state.selected)}<select value={state.sequence[state.selected]} onChange={e => editBase(state.selected, e.target.value)}>{[...alphabet].map(c => <option key={c}>{c}</option>)}</select></label><NeuralNumber label="Fill span endpoint (inclusive)" value={state.spanEnd} min={0} max={59} integer onChange={spanEnd => update({
          spanEnd
        })} /><label className="hy-select">Fill symbol<select value={state.fill} onChange={e => update({
            fill: e.target.value
          })}>{[...alphabet].map(c => <option key={c}>{c}</option>)}</select></label></div><div className="hy-controls"><button onClick={() => setSequence([...state.sequence].map((v, i) => i >= spanStart && i <= spanEnd ? state.fill : v).join(''))}>Fill indices{spanStart}–{spanEnd}</button><button onClick={() => setSequence(state.original.slice(0, 30) + 'AA' + state.original.slice(32))}>Original with boundary30–31→AA</button><button onClick={() => setSequence('AA' + state.original.slice(2))}>Original with distant0–1→AA</button><button onClick={() => setSequence('N'.repeat(60))}>All ambiguous N symbols</button><button onClick={() => setSequence(state.original)}>Restore exact original</button><button onClick={() => update({
          pinned: {
            sequence: state.sequence,
            gatesOff: state.gatesOff,
            kernelLimit: state.kernelLimit,
            modelId
          }
        })}>Pin sequence and settings</button><button onClick={() => update({
          pinned: null
        })}>Clear pin</button><button onClick={() => {
          setState(initialDnaState());
          setModelId('gated_29');
        }}>Reset DNA investigation</button></div><div className="neural-controls"><NeuralNumber label="Retain learned filter taps (1–60)" value={state.kernelLimit} min={1} max={60} integer onChange={kernelLimit => update({
          kernelLimit
        })} /><NeuralNumber label="Inspect network block" value={state.block} min={0} max={1} integer onChange={block => update({
          block
        })} /><NeuralNumber label="Inspect channel" value={state.channel} min={0} max={15} integer onChange={channel => update({
          channel
        })} /></div><label><input type="checkbox" checked={state.gatesOff} onChange={e => update({
          gatesOff: e.target.checked
        })} /> Override sending and receiving gates to1</label><p>These are fixed-weight interventions, not new trained architectures. Linear baseline controls for gates, taps, block and channel have no mathematical effect; switching back preserves their values.</p><DnaStrip title="Current sequence aligned to the original" sequence={state.sequence} reference={state.original} selected={state.selected} observedLabel={state.sequence === state.original ? state.label : null} />{model.data ? <DnaReadView model={model.data} state={state} /> : <Loading error={model.error} onRetry={() => setRetry(v => v + 1)} />}</div></NeuralLab>;
}
export function HyenaProgram({
  filename
}) {
  return <>
    <RemoteCodeBlock source={base + filename} language="python" filename={filename} title={"Complete program: " + (filename)} />
    <p><a href={base + filename} download>Download {filename}</a></p>
    <p role="alert">The program could not load. Retry or use its download link.</p>
  </>;
}
