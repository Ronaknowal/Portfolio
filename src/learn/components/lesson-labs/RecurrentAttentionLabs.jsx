import useLessonViewport from './useLessonViewport.js';
import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import { useEffect, useMemo, useState } from 'react';
import { NeuralLab, NeuralNumber, NeuralSelect, NeuralTable, NeuralPlot, formatNeural as format } from './NeuralLessonElements.jsx';
import { attentionTokens, initialKeys, initialValues, memoryRead, cancellationRead, localAttention, copyDistribution, encodeAttention, traceAttention } from '../../data/recurrent-attention-models.js';
import measurements from '../../data/recurrent-attention-measurements.json';
import './recurrent-attention-labs.css';

const assetBase = '/learn-code/attention-mechanism-bahdanau-luong/';
const names = ['A', 'B', 'C'];
const palette = ['#e2b55a', '#eeeeee', '#ac9b7b'];
const vectorText = vector => `(${vector.map(value => format(value, 6)).join(', ')})`;

function TokenStrip({ title, tokens, selected = -1 }) {
  return <figure className="attention-token-strip"><figcaption>{title}</figcaption><ol>{tokens.map((token, index) => <li key={index} className={index === selected ? 'is-selected' : ''}><small>{index}</small><span>{token}</span></li>)}</ol></figure>;
}

function Shares({ labels, values, title }) {
  return <figure className="attention-shares"><figcaption>{title}</figcaption>{labels.map((label, index) => <div className="attention-share-row" key={`${index}-${label}`}><span>{label}</span><span className="attention-bar"><span style={{ width: `${100 * values[index]}%` }} /></span><output>{format(values[index], 6)}</output></div>)}<small>Bar scale: 0 to 1, fixed across all rows.</small></figure>;
}

function ValuePlane({ values, context }) {
  const position = value => 24 + (value + 3) * 42;
  const vertical = value => 276 - (value + 3) * 42;
  return <figure className="attention-plane"><figcaption>Value space · context is the weighted mixture</figcaption><svg viewBox="0 0 300 300" role="img" aria-label={`Both axes range from minus 3 to 3 with equal scale. Context ${vectorText(context)}. Exact point coordinates are below.`}>
    <path d="M24 150H276M150 24V276" stroke="#777" fill="none" />
    <polygon points={values.map(value => `${position(value[0])},${vertical(value[1])}`).join(' ')} fill="#e2b55a" fillOpacity=".08" stroke="#999" strokeDasharray="4 4" />
    {values.map((value, index) => <g key={index}><line x1={position(value[0])} y1={vertical(value[1])} x2={position(context[0])} y2={vertical(context[1])} stroke={palette[index]} strokeOpacity=".45" />{index === 0 ? <circle cx={position(value[0])} cy={vertical(value[1])} r="5" fill={palette[index]} /> : index === 1 ? <rect x={position(value[0]) - 5} y={vertical(value[1]) - 5} width="10" height="10" fill={palette[index]} /> : <path d={`M${position(value[0])} ${vertical(value[1]) - 6}l6 12h-12z`} fill={palette[index]} />}</g>)}
    <path d={`M${position(context[0])} ${vertical(context[1]) - 7}l7 7-7 7-7-7z`} fill="#090909" stroke="#e2b55a" strokeWidth="2.5" />
    <text x="24" y="168">−3</text><text x="265" y="168">3</text><text x="160" y="29">3</text><text x="158" y="281">−3</text>
  </svg><p>● A · ■ B · ▲ C · ◇ context. Horizontal = coordinate 0; vertical = coordinate 1. Overlapping markers mean coincident coordinates.</p></figure>;
}

export function AttentionMemoryShelf() {
  return <figure className="attention-shelf" data-figure="memory-shelf"><TokenStrip title="Saved source positions: one learned 64-coordinate vector per token" tokens={['<past>', ...'lactate', '<eos>']} /><div className="attention-branch"><p><strong>Final state h₈</strong><span>↓ initialize once</span><span>Decoder state s₀ (64 coordinates)</span></p><p><strong>All states H (9 × 64)</strong><span>↓ score → normalize → read, each step</span><span>Current context cₜ (64 coordinates)</span></p></div><div className="attention-result">Decoder state + current read → next output probability</div><TokenStrip title="A separate output timeline" tokens={[...'lactated', '<eos>']} /><figcaption>The two paths survive together. The encoder can discard information; keeping its vectors makes those learned representations available again. A forward state sees its source prefix; a bidirectional state would also have a backward path from later source positions.</figcaption></figure>;
}

export function AttentionWorkedRead() {
  const result = memoryRead([1, 0]);
  return <figure data-figure="weighted-read"><div className="attention-two-column"><ValuePlane values={initialValues} context={result.context} /><Shares title="Worked read: score 1, 0, −1 → attention" labels={names} values={result.attention} /></div><NeuralTable caption="The same context from three signed contributions" headers={['Memory', 'Value', 'Weight × value']} rows={initialValues.map((value, index) => [names[index], vectorText(value), vectorText(value.map(coordinate => coordinate * result.attention[index]))])} /><figcaption>The contributions add to {vectorText(result.context)}. The plane and the vector sum show the same calculation. Negative value coordinates do not imply negative attention.</figcaption></figure>;
}

export function AttentionReadLab({ learning = false }) {
  const [query, setQuery] = useState([.3, -.4]);
  const [keys, setKeys] = useState(initialKeys);
  const [values, setValues] = useState(initialValues);
  const [valid, setValid] = useState([true, true, true]);
  const [selected, setSelected] = useState(1);
  const [rate, setRate] = useState(.1);
  const current = memoryRead(query, keys, values, valid, rate);
  const baseline = memoryRead([.3, -.4]);
  const changeCoordinate = (setter, row, coordinate, value) => setter(previous => previous.map((item, index) => index === row ? item.map((old, axis) => axis === coordinate ? value : old) : item));
  const reset = () => {
    setQuery([.3, -.4]); setKeys(initialKeys); setValues(initialValues); setValid([true, true, true]); setSelected(1); setRate(.1);
  };
  return <NeuralLab id={learning ? 'attention-learning' : 'attention-read'} title={learning ? 'Follow signed credit into a query update' : 'Change the question, the address or the returned information'}>
    <p>Constructed vectors. Keys choose the weights; values supply the returned coordinates. {learning ? 'Context coordinates are also two-class logits in this small experiment. The desired class is class 2.' : 'Begin by changing a returned value, then a key. Compare which quantities move in each case; the context is the endpoint of this investigation.'}</p>
    <div className="neural-controls">{query.map((value, axis) => <NeuralNumber key={axis} label={`Query coordinate ${axis}`} min={-3} max={3} value={value} onChange={next => setQuery(previous => previous.map((old, coordinate) => coordinate === axis ? next : old))} />)}<NeuralSelect label="Memory to edit" value={selected} onChange={value => setSelected(Number(value))} options={names.map((name, index) => [index, name])} /></div>
    <div className="attention-two-column"><fieldset><legend>Key {names[selected]}: changes relevance</legend>{keys[selected].map((value, axis) => <NeuralNumber key={axis} label={`Key ${names[selected]} coordinate ${axis}`} min={-3} max={3} value={value} onChange={next => changeCoordinate(setKeys, selected, axis, next)} />)}</fieldset><fieldset><legend>Value {names[selected]}: changes returned information</legend>{values[selected].map((value, axis) => <NeuralNumber key={axis} label={`Value ${names[selected]} coordinate ${axis}`} min={-3} max={3} value={value} onChange={next => changeCoordinate(setValues, selected, axis, next)} />)}</fieldset></div>
    {learning && <><div className="attention-mask-controls">{names.map((name, index) => <label key={name}><input type="checkbox" checked={valid[index]} disabled={valid[index] && valid.filter(Boolean).length === 1} onChange={event => setValid(previous => previous.map((value, position) => position === index ? event.target.checked : value))} />Memory {name} is valid</label>)}</div><p>At least one memory must remain valid; softmax over no legal memories is undefined.</p></>}
    <div className="attention-two-column"><ValuePlane values={values} context={current.context} /><Shares title="Attention: one denominator over valid memories" labels={names} values={current.attention} /></div>
    {learning
      ? <p className="attention-result">Context <strong>{vectorText(current.context)}</strong> → P(class 2) <strong>{format(current.probabilities[1])}</strong> → loss <strong>{format(current.loss)}</strong> nats. Original: context {vectorText(baseline.context)}, P(class 2) {format(baseline.probabilities[1])}, loss {format(baseline.loss)}.</p>
      : <p className="attention-result">Returned context <strong>{vectorText(current.context)}</strong>. Original context: {vectorText(baseline.context)}. The source shares above say how much of each value enters this mixture.</p>}
    <NeuralTable caption="Current read, with exact signed contributions" headers={['Memory', 'Key', 'Value', 'Score', 'Contribution']} rows={names.map((name, index) => [name, vectorText(keys[index]), vectorText(values[index]), valid[index] ? format(current.scores[index]) : 'masked', vectorText(values[index].map(value => value * current.attention[index]))])} />
    {learning && <><NeuralNumber label="Query update rate" value={rate} min={0} max={.2} onChange={setRate} /><NeuralTable caption="Backpropagation through the mixture" headers={['Memory', 'Weight', 'Loss gradient of score']} rows={names.map((name, index) => [name, format(current.attention[index]), format(current.scoreGradient[index])])} /><div className="attention-gradient-chain"><p>Context gradient<br /><strong>{vectorText(current.contextGradient)}</strong></p><span>→</span><p>Query gradient<br /><strong>{vectorText(current.queryGradient)}</strong></p><span>→</span><p>Query minus rate × gradient<br /><strong>{vectorText(current.nextQuery)}</strong></p></div><p className="attention-result">After the complete query update: context {vectorText(current.nextContext)}, loss <strong>{format(current.nextLoss)}</strong> nats; change {format(current.nextLoss - current.loss, 8)}. {rate === 0 ? 'Zero rate leaves the query and result unchanged.' : values.every(value => value.every((coordinate, axis) => coordinate === values[0][axis])) ? 'All values coincide, so changing their weights cannot change the read or supply a score gradient.' : 'This recomputes the read with the new query; keys and values stay fixed.'}</p></>}
    <div className="neural-buttons"><button onClick={() => setValues(previous => previous.map((value, index) => index === 1 ? [1, 2] : value))}>Change B’s returned value</button><button onClick={() => setValues([[.5, .5], [.5, .5], [.5, .5]])}>Make all values identical</button><button onClick={reset}>Reset {learning ? 'learning' : 'read'}</button></div>
    <p>Try a value edit with query and keys fixed, then a key edit. The first leaves attention unchanged; the second can redistribute every weight. Equal values reveal a different limit: attention can move while the returned information stays fixed.</p>
  </NeuralLab>;
}

export function AttentionCancellationLab() {
  const [query, setQuery] = useState(.2);
  const linear = cancellationRead(query), nonlinear = cancellationRead(query, true);
  return <NeuralLab id="attention-cancellation" title="Does the scorer actually use its question?"><p>Keys are −0.7, 0.1 and 1.3. Move q: the shared 2q term cancels from the linear scorer’s softmax. Applying tanh before normalizing allows relative scores to change.</p><NeuralNumber label="Scorer query q" min={-2} max={2} value={query} onChange={setQuery} /><div className="attention-two-column"><Shares title="Linear: softmax(2q + k)" labels={names} values={linear.attention} /><Shares title="Nonlinear: softmax(tanh(2q + k))" labels={names} values={nonlinear.attention} /></div><NeuralTable caption="Scores can change even when weights do not" headers={['Memory', 'Linear score', 'Nonlinear score']} rows={names.map((name, index) => [name, format(linear.scores[index]), format(nonlinear.scores[index])])} /><button onClick={() => setQuery(.2)}>Reset scorer</button></NeuralLab>;
}

export function AttentionSchedules() {
  return <figure data-figure="decoder-schedules"><div className="attention-two-column"><div className="attention-timeline"><h4>Bahdanau-style order</h4><ol><li>Previous state sₜ₋₁ → query</li><li>Query + saved memory → read cₜ</li><li>Previous token (24) + context (64) → GRU input (88)</li><li>New state sₜ + context cₜ → next-token probabilities</li></ol></div><div className="attention-timeline"><h4>Luong-style order</h4><ol><li>Previous token (24) → GRU input</li><li>Updated state sₜ → query</li><li>Query + saved memory → read cₜ</li><li>New state + context → combined vector s̃ₜ → output</li></ol></div></div><p className="attention-result">Optional input feeding: previous s̃ₜ₋₁ (64) joins the previous token (24) at the next GRU input (88). It starts at zero. It never feeds the current answer into its own prediction.</p><figcaption>These are dependency timelines, not measured durations. The score function belongs inside the read; its choice and the read’s position are separate design decisions.</figcaption></figure>;
}

export function AttentionGradientFlow() {
  return <figure data-figure="gradient-path"><div className="attention-gradient-chain"><p>Output loss<br /><strong>∂L/∂c = p − target</strong></p><span>→</span><p>Memory j versus mixture<br /><strong>αⱼ gᵀ(vⱼ − c)</strong></p><span>→</span><p>Query sensitivity<br /><strong>Σⱼ (∂L/∂eⱼ) kⱼ</strong></p></div><figcaption>Backward dependencies. A positive score derivative means increasing that score locally increases loss; descent subtracts it. The value is compared with the current mixture, so a large weight alone does not determine the sign.</figcaption></figure>;
}

export function AttentionLearningCurves() {
  return <figure data-figure="learning-curves"><div className="attention-curve-grid">{['fixed', 'additive', 'general'].map(kind => <NeuralPlot key={kind} title={`${kind === 'fixed' ? 'Fixed context' : kind === 'additive' ? 'Additive attention' : 'General attention'} · measured development exact match`} xLabel="optimizer updates" yLabel="exact forms / 447" xDomain={[0, 1200]} yDomain={[0, 1]} series={[...measurements.runs.filter(run => run.kind === kind).map((run, index) => ({ label: `seed ${run.seed}`, color: palette[index], dashed: index === 2, values: run.checkpoints.map(row => [row.update, row.exact / 447]) })), { label: 'suffix rules: 407/447', color: '#888888', dashed: true, values: [[0, 407 / 447], [1200, 407 / 447]] }]} points={measurements.runs.filter(run => run.kind === kind).flatMap((run, index) => run.checkpoints.map(row => ({ x: row.update, y: row.exact / 447, color: palette[index], label: `seed ${run.seed}, update ${row.update}: ${row.exact}/447` })))} />)}</div><figcaption>Actual saved checkpoints; connecting lines guide the eye between measurements. All panels use the same 0–1 scale and retain late dips. The gray rule baseline is unchanged; it is not a fitted curve.</figcaption><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Exact checkpoint counts for all nine runs</h4><NeuralTable caption="Measured development exact counts, denominator 447" headers={['Architecture', 'Seed', '0', '100', '400', '800', '1200']} rows={measurements.runs.map(run => [run.kind, run.seed, ...run.checkpoints.map(row => row.exact)])} /></section></figure>;
}

function useAttentionAsset(file, enabled, json = false) {
  const [state, setState] = useState({ key: '', data: null, error: '' });
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    setState({ key: file, data: null, error: '' });
    fetch(assetBase + file, { signal: controller.signal }).then(response => {
      if (!response.ok) throw new Error('The lesson asset could not be loaded.');
      return json ? response.json() : response.text();
    }).then(data => {
      if (!controller.signal.aborted) setState({ key: file, data, error: '' });
    }).catch(error => {
      if (!controller.signal.aborted) setState({ key: file, data: null, error: error.message });
    });
    return () => controller.abort();
  }, [file, enabled, json, retry]);
  return { data: enabled && state.key === file ? state.data : null, error: state.key === file ? state.error : '', retry: () => setRetry(value => value + 1) };
}

export function AttentionProgram({ file, title }) {
  return <>
    <RemoteCodeBlock source={assetBase + file} language="python" filename={file} title={(title)} />
    <p><a href={assetBase + file} download={file}>Download {file}</a>. The source shown here is the exact downloadable file.</p>
  </>;
}

function AlignmentMatrix({ rows, sourceIds, selected, onSelect, label }) {
  return <div className="attention-matrix-scroll" tabIndex={0} role="region" aria-label={`${label}; scroll horizontally for all source positions`}><table className="attention-matrix"><caption>{label} · brightness 0 to 1 · select an output row to inspect it</caption><thead><tr><th scope="col">Output step</th>{sourceIds.map((token, index) => <th scope="col" key={index}><small>{index}</small>{attentionTokens[token]}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index} className={selected === index ? 'is-selected' : ''}><th scope="row"><button aria-pressed={selected === index} onClick={() => onSelect(index)}>{index + 1}: {attentionTokens[row.emittedToken ?? row.emitted]}</button></th>{row.attention.map((weight, position) => <td key={position} style={{ background: `rgba(226, 181, 90, ${weight * .72})` }}>{format(weight, 3)}</td>)}</tr>)}</tbody></table></div>;
}

export function AttentionWorkedAlignment() {
  const [selected, setSelected] = useState(0);
  const sourceIds = [3, ...[...'lactate'].map(character => attentionTokens.indexOf(character)), 2];
  return <figure data-figure="worked-alignment"><AlignmentMatrix rows={measurements.worked} sourceIds={sourceIds} selected={selected} onSelect={setSelected} label="Recorded additive seed 1 · lactate + past" /><Shares title={`Recorded attention for output ${selected + 1}: ${attentionTokens[measurements.worked[selected].emitted]}`} labels={sourceIds.map((token, index) => `${index}: ${attentionTokens[token]}`)} values={measurements.worked[selected].attention} /><figcaption>These are measured rows from one frozen model. The live model below exposes scores, context and output probabilities and allows a new source or prefix. Source positions identify repeated letters separately.</figcaption></figure>;
}

function AttentionTextControl({ label, value, onChange, min = 0, max = 8 }) {
  const [draft, setDraft] = useState(null);
  return <label className="attention-text-control">{label}<input aria-label={label} value={draft ?? value} aria-invalid={draft !== null} onChange={event => {
    const text = event.target.value;
    if (new RegExp(`^[a-z]{${min},${max}}$`).test(text)) { setDraft(null); onChange(text); } else setDraft(text);
  }} onBlur={() => setDraft(null)} />{draft !== null && <small>Use {min}–{max} lowercase a–z letters. Views retain “{value}”.</small>}</label>;
}

function FittedAttentionView({ weights, kind, mode }) {
  const paddingMode = mode === 'padding';
  const [lemma, setLemma] = useState('cash');
  const [feature, setFeature] = useState('past');
  const [prefix, setPrefix] = useState('');
  const [step, setStep] = useState(paddingMode ? 2 : 0);
  const [coordinate, setCoordinate] = useState(0);
  const [padding, setPadding] = useState(paddingMode ? 2 : 0);
  const [admitted, setAdmitted] = useState([false, false]);
  const [cap, setCap] = useState(16);
  const encoded = useMemo(() => encodeAttention(weights, lemma, feature), [weights, lemma, feature]);
  const trace = useMemo(() => traceAttention(weights, kind, encoded, { prefix, padding, admitted, cap }), [weights, kind, encoded, prefix, padding, admitted, cap]);
  const baseline = useMemo(() => traceAttention(weights, kind, encodeAttention(weights, 'cash', 'past')), [weights, kind]);
  const clean = useMemo(() => traceAttention(weights, kind, encoded, { prefix, padding, cap }), [weights, kind, encoded, prefix, padding, cap]);
  const selected = Math.min(step, trace.rows.length - 1), row = trace.rows[selected];
  const original = baseline.rows[selected];
  const cleanRow = clean.rows[selected];
  const ranked = row.probabilities.map((probability, token) => ({ probability, token })).sort((left, right) => right.probability - left.probability).slice(0, 6);
  const padMass = row.attention.slice(encoded.memory.length).reduce((sum, value) => sum + value, 0);
  const reset = () => {
    setLemma('cash'); setFeature('past'); setPrefix(''); setStep(paddingMode ? 2 : 0); setCoordinate(0); setPadding(paddingMode ? 2 : 0); setAdmitted([false, false]); setCap(16);
  };
  return <>
    <div className="neural-controls"><AttentionTextControl label="Source spelling" value={lemma} onChange={setLemma} min={3} /><NeuralSelect label="Requested form" value={feature} onChange={setFeature} options={['past', 'participle', 'third_person'].map(value => [value, value.replace('_', ' ')])} /><AttentionTextControl label="Forced emitted prefix (optional)" value={prefix} onChange={setPrefix} /><NeuralNumber label="Output token cap (includes EOS)" min={1} max={16} step={1} integer value={cap} onChange={setCap} /></div>
    {paddingMode && <><NeuralNumber label="Extra zero PAD memories" min={0} max={2} step={1} integer value={padding} onChange={next => { setPadding(next); setAdmitted([false, false]); }} /><div className="attention-mask-controls">{Array.from({ length: padding }, (_, index) => <label key={index}><input type="checkbox" checked={admitted[index]} onChange={event => setAdmitted(previous => previous.map((value, position) => position === index ? event.target.checked : value))} />Fault: admit PAD slot {index + 1}</label>)}</div><p>Source lengths still protect the encoder. This deliberate fault admits zero-valued storage after encoding. Newly inserted slots start masked.</p></>}
    <p className="attention-result">Current output: <strong>{trace.prediction || '(no letters)'}</strong> · {trace.ended ? 'natural EOS' : 'capped before EOS'}. Reference starting model: cash + past → <strong>{baseline.prediction}</strong>. Edited spellings are queries, not automatically labeled examples.</p>
    <TokenStrip title="Encoded source: position identities" tokens={trace.sourceIds.map(token => attentionTokens[token])} />
    <AlignmentMatrix rows={trace.rows} sourceIds={trace.sourceIds} selected={selected} onSelect={setStep} label="Current computed attention over source positions" />
    <div className="neural-controls"><NeuralNumber label="Inspect output step (one-based)" min={1} max={trace.rows.length} step={1} integer value={selected + 1} onChange={value => setStep(value - 1)} /><NeuralNumber label="Inspect hidden coordinate" min={0} max={63} step={1} integer value={coordinate} onChange={setCoordinate} /></div>
    <p>Step {selected + 1} receives <strong>{attentionTokens[row.previous]}</strong>. Argmax is <strong>{attentionTokens[row.argmaxToken]}</strong>; emitted <strong>{attentionTokens[row.emittedToken]}</strong>{selected < prefix.length ? ' is forced after this distribution was computed' : ' is chosen from this distribution'}. The next step consumes that emission.</p>
    <div className="attention-two-column"><Shares title="Source weights" labels={trace.sourceIds.map((token, index) => `${index}: ${attentionTokens[token]}`)} values={row.attention} /><Shares title="Six most probable next outputs" labels={ranked.map(item => attentionTokens[item.token])} values={ranked.map(item => item.probability)} /></div>
    <p className="attention-result">Coordinate {coordinate}: query {format(row.query[coordinate])} → context <strong>{format(row.context[coordinate])}</strong> → post-update state {format(row.state[coordinate])}. {paddingMode && <>PAD mass <strong>{format(padMass, 8)}</strong>; {cleanRow ? <>correctly masked context <strong>{format(cleanRow.context[coordinate], 8)}</strong></> : <>the correctly masked run has already ended</>}. Same-source correct masking produces <strong>{clean.prediction}</strong>.</>}</p>
    <NeuralTable caption={`Output step ${selected + 1}: how memory forms context coordinate ${coordinate}`} headers={['Position', 'Score', 'Weight', 'Memory value', 'Contribution']} rows={trace.sourceIds.map((token, index) => [`${index}: ${attentionTokens[token]}`, trace.valid[index] ? format(row.scores[index]) : 'masked', format(row.attention[index]), format(trace.memory[index][coordinate]), format(row.attention[index] * trace.memory[index][coordinate])])} />
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect all output probabilities and the fixed starting reference</h4><NeuralTable caption="Current probability versus cash + past at the same step" headers={['Output', 'Current', 'Original', 'Difference']} rows={attentionTokens.map((token, index) => [token, format(row.probabilities[index], 8), original ? format(original.probabilities[index], 8) : 'original ended', original ? format(row.probabilities[index] - original.probabilities[index], 8) : '—'])} /></section>
    <div className="neural-buttons"><button onClick={() => { setLemma('cask'); setPrefix(''); }}>Change cash → cask</button><button onClick={() => { setPrefix('b'); setStep(1); }}>Force first output b</button><button onClick={() => { setLemma('lactate'); setFeature('past'); setPrefix(''); }}>Inspect worked lactate</button><button onClick={reset}>Reset fitted inputs</button></div>
    <p>Source/request edits rebuild the encoder and projected keys. Prefix edits reuse that source memory and replay the decoder. Forcing the first emission changes later reads but leaves the first distribution unchanged. Numeric hidden coordinates are learned features; their indices are not linguistic labels.</p>
  </>;
}

export function AttentionFittedLab({ mode = 'source' }) {
  const [teachingSection, ready] = useLessonViewport();
  const [kind, setKind] = useState('additive');
  const resource = useAttentionAsset(`${kind}-seed-one.json`, ready, true);
  return <NeuralLab id={`attention-${mode}`} title={mode === 'padding' ? 'Zero-valued storage can steal attention mass' : 'Edit a real input and follow the complete decoder'}><p>Frozen seed-one parameters; inference runs locally when this investigation comes into view. No training starts in the browser. Additive and general are different fitted architectures, so switching them is not a scorer-only ablation.</p><section   data-lesson-teaching="" ref={teachingSection} className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">The {mode === 'padding' ? 'padding' : 'source and prefix'} investigation</h4><NeuralSelect label="Frozen attention model" value={kind} onChange={setKind} options={[["additive", "Additive · read before update"], ["general", "General · read after update"]]} />{resource.error ? <p role="alert">{resource.error} <button onClick={resource.retry}>Retry model</button></p> : resource.data ? <FittedAttentionView key={kind} mode={mode} weights={resource.data.weights} kind={kind} /> : ready ? <p role="status">Loading this model’s parameters…</p> : null}</section></NeuralLab>;
}

export function AttentionAmbiguityFigure() {
  return <figure data-figure="same-context"><div className="attention-two-column"><Shares title="First distribution" labels={names} values={[.4, .4, .2]} /><Shares title="Second distribution" labels={names} values={[.2, .2, .6]} /></div><p className="attention-result">Values (1, 0), (0, 1), (0.5, 0.5) → both return <strong>(0.5, 0.5)</strong>.</p><figcaption>The third value is itself the midpoint of the first two. Moving probability onto it preserves the context. Different heatmaps can therefore deliver exactly the same downstream input.</figcaption></figure>;
}

export function AttentionWindowLab() {
  const [center, setCenter] = useState(2.5), [radius, setRadius] = useState(1.5);
  const [renormalize, setRenormalize] = useState(false);
  const [scores, setScores] = useState([0, .5, 1, -.5, 2]);
  const [selected, setSelected] = useState(4);
  const result = localAttention(center, radius, renormalize, scores);
  const reset = () => { setCenter(2.5); setRadius(1.5); setRenormalize(false); setScores([0, .5, 1, -.5, 2]); setSelected(4); };
  return <NeuralLab id="attention-window" title="Move the read window; keep its magnitude visible"><p>Constructed five-position source. Values equal position numbers. Move the center or boundary and watch legal donors, Gaussian weights and the final scalar read change immediately.</p><div className="neural-controls"><NeuralNumber label="Window center" min={1} max={5} value={center} onChange={setCenter} /><NeuralNumber label="Window radius" min={1} max={3} value={radius} onChange={setRadius} /><NeuralSelect label="Source score to edit" value={selected} onChange={value => setSelected(Number(value))} options={[1, 2, 3, 4, 5].map((position, index) => [index, `Position ${position}`])} /><NeuralNumber label={`Score at position ${selected + 1}`} min={-3} max={3} value={scores[selected]} onChange={next => setScores(previous => previous.map((value, index) => index === selected ? next : value))} /></div>
    <p>Each card shows a source position and its distance d from the center. Amber “in” cards are eligible; gray “out” cards are excluded.</p>
    <div className="attention-window-ruler">{result.positions.map((position, index) => <div key={position} className={result.valid[index] ? 'is-included' : ''}><strong>{position}</strong><span>{result.valid[index] ? 'in' : 'out'}</span><small>d {format(Math.abs(position - center), 2)}</small></div>)}</div>
    <label className="attention-toggle"><input type="checkbox" checked={renormalize} onChange={event => setRenormalize(event.target.checked)} />Explicitly renormalize after Gaussian multiplication</label>
    <NeuralTable caption="Four distinct operations: eligibility, softmax, Gaussian, optional normalization" headers={['Position', 'Valid?', 'Softmax', 'Gaussian', 'Final weight']} rows={result.positions.map((position, index) => [position, result.valid[index] ? 'yes' : 'no', format(result.base[index]), format(result.gaussian[index]), format(result.weights[index])])} />
    <Shares title="Final weights: magnitude matters" labels={result.positions.map(position => `Position ${position}`)} values={result.weights} /><p className="attention-result">Weight sum <strong>{format(result.sum, 8)}</strong>; context <strong>{format(result.context, 8)}</strong>. {renormalize ? 'The second normalization makes a convex weighted average.' : 'Original Gaussian multiplication retains a smaller total mass; this read is not necessarily a convex average.'} Position {selected + 1} {result.valid[selected] ? 'contributes to the current read.' : 'is excluded; changing its score cannot change this read.'}</p><div className="neural-buttons"><button onClick={() => { setCenter(3); setRadius(2); setScores([0, .5, 1, -.5, 2]); setRenormalize(false); }}>Inspect worked center 3, radius 2</button><button onClick={reset}>Reset window</button></div><p>This edits a declared center; it does not train the network that predicts centers. Window membership changes at boundaries. A local window can move backward, and does not by itself enforce monotonic or streaming behavior.</p></NeuralLab>;
}

export function AttentionCopyFlow() {
  const [pGenerate, setPGenerate] = useState(.4);
  const output = copyDistribution(['Ada', 'met', 'Ada'], [.2, .3, .5], { Ada: .1, met: .6, left: .3 }, pGenerate);
  return <NeuralLab id="attention-copy" title="Two source positions can name one output word"><TokenStrip title="Source-position probabilities: Ada .2 → met .3 → Ada .5" tokens={['Ada', 'met', 'Ada']} /><p className="attention-result">Positions 0 and 2 both point to Ada: copy mass 0.2 + 0.5 = <strong>0.7</strong>. Position 1 gives met mass 0.3. Copying groups by the actual word, not by whichever occurrence is brightest.</p><NeuralNumber label="Vocabulary-route probability" min={0} max={1} value={pGenerate} onChange={setPGenerate} /><div className="attention-copy-output">{output.map(item => <div key={item.word}><strong>{item.word}: {format(item.probability)}</strong><div className="attention-stacked-bar"><span style={{ width: `${100 * item.generated}%` }} /><span style={{ width: `${100 * item.copied}%` }} /></div><small>Vocabulary {format(item.generated)} + copy {format(item.copied)}</small></div>)}</div><p>Amber = vocabulary contribution; light gray = copying. Each full track means probability 1. Vocabulary-only “left” loses mass when copying dominates.</p><button onClick={() => setPGenerate(.4)}>Reset mixture</button></NeuralLab>;
}

export function AttentionLocationFlow() {
  return <figure data-figure="location-features"><div className="attention-gradient-chain"><p>Previous attention<br /><strong>… 0.1, 0.7, 0.2 …</strong></p><span>→</span><p>Convolve around each position<br /><strong>F * αₜ₋₁ = local features</strong></p><span>→</span><p>Add inside current scorer<br /><strong>tanh(query + key + location)</strong></p></div><figcaption>Conceptual dependency diagram, not measured speech. Similar acoustic content can occur twice; the previous read supplies a location cue. A backward encoder can already depend on future audio, which this extra cue cannot remove.</figcaption></figure>;
}

export function AttentionScratchRoute() {
  return <section className="attention-scratch"><h3>Rebuild one complete saved-model read in NumPy</h3><p>The scratch route implements matrix projections, GRU state updates, masked softmax, weighted reads, the two decoder orders and greedy generation. NumPy supplies array arithmetic; it supplies no attention layer or sequence model. Download <a href={assetBase + 'attention-inference.py'}>attention-inference.py</a> and the <a href={assetBase + 'calculated-inputs.json'}>same saved weights</a> into one directory, then run <code>python attention-inference.py</code>. The default prints <code>lactated True</code>, matching the PyTorch program above.</p><p>Read <code>gru</code> as the reused recurrent primitive: reset and update gates use PyTorch’s r,z,n order. <code>encode</code> builds and caches source memory once; <code>decode</code> locates the read before or after the recurrent update, normalizes over valid source positions, and feeds each emitted token back at the next step. The <code>context = attention @ memory</code> line is the same weighted sum drawn in section 2.</p><AttentionProgram file="attention-inference.py" title="Read the complete NumPy mechanism and saved-model inference" /><p>For learning derivatives and the optional local/copy calculations, use the <a href={assetBase + 'attention-calculations.py'}>constructed calculation program</a>. Its analytic query derivative is compared with native autograd and finite differences. The full training program composes the same owned attention operations with maintained PyTorch embeddings, GRU, optimizer and loss. Reuse the <a href="/learn/path/full-curriculum/rnns-lstms-grus?module=deep-learning-fundamentals">recurrent lesson’s gate derivation</a>; no second autograd engine is needed here.</p><AttentionProgram file="attention-calculations.py" title="Read the exact attention-gradient, window and copy calculations" /><p>The supplementary <code>attention-mechanics.py</code> report also reads the <a href="/learn-code/sequence-to-sequence-encoder-decoder/calculated-inputs.json">earlier encoder–decoder report</a>. To reproduce that comparison, keep sibling directories named <code>attention-mechanism-bahdanau-luong</code> and <code>sequence-to-sequence-encoder-decoder</code>, with each report saved as <code>calculated-inputs.json</code> in its own directory. The standalone NumPy inference above needs only its own saved weights.</p></section>;
}
