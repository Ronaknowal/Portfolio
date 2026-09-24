import { useEffect, useState } from 'react';
import { projectAssets } from './project-elements.jsx';
import { attentionKeys, attentionValues, attentionMixture, candidateFeatures, encodeRequestTrace, scoringHeadStep } from './mechanism-models.js';

const names = ['Billing', 'Access', 'Delivery'];
const vector = values => `[${values.map(value => Number(value).toFixed(3)).join(', ')}]`;
let fixtureRequest;

function Slider({ id, label, value, min, max, step = 0.1, onChange }) {
  return <label className="tdp-range" htmlFor={id}><span>{label}<output htmlFor={id}>{value.toFixed(2)}</output></span>
    <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}

export function EncodingLab() {
  const [fixture, setFixture] = useState(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState('please help with my refund');
  const [includeDelivery, setIncludeDelivery] = useState(true);
  const [rotated, setRotated] = useState(false);
  useEffect(() => {
    let active = true;
    setFailed(false);
    if (!fixtureRequest) fixtureRequest = fetch(`${projectAssets}/trace-fixture.json`).then(response => {
      if (!response.ok) throw new Error('Trace fixture unavailable');
      return response.json();
    }).catch(error => { fixtureRequest = null; throw error; });
    fixtureRequest.then(value => { if (active) setFixture(value); }, () => { if (active) setFailed(true); });
    return () => { active = false; };
  }, [attempt]);
  if (failed) return <p role="alert">The tokenizer fixture could not load. <button type="button" onClick={() => setAttempt(value => value + 1)}>Retry tokenizer trace</button></p>;
  if (!fixture) return <p role="status">Loading the executable tokenizer’s vocabulary…</p>;
  const original = fixture.request.options.filter(option => includeDelivery || option.id !== 'delivery');
  const options = rotated ? [...original.slice(1), original[0]] : original;
  const result = encodeRequestTrace({ ...fixture.request, state, options }, fixture.vocabulary);
  return <section className="tdp-explorer tdp-encoding-lab" aria-label="Trace a request into tokens and candidate positions">
    <div className="tdp-explorer-heading"><div><span className="tdp-eyebrow">Live input trace</span><h3>A label keeps its identity when its position moves</h3></div>
      <button type="button" onClick={() => { setState('please help with my refund'); setIncludeDelivery(true); setRotated(false); }}>Reset trace</button></div>
    <p className="tdp-small">The vocabulary is exported from the real Python program’s training data. This browser trace runs the same word and marker rules; it does not execute the transformer.</p>
    <label className="tdp-text-control">Request text<textarea aria-label="Request text" maxLength={4000} rows={2} value={state} onChange={event => setState(event.target.value)} /></label>
    <div className="tdp-controls"><button type="button" onClick={() => setState('money left my bank twice')}>Try an unseen paraphrase</button>
      <button type="button" aria-pressed={rotated} onClick={() => setRotated(value => !value)}>Move first candidate to the end</button>
      <label className="tdp-check"><input type="checkbox" checked={includeDelivery} onChange={event => setIncludeDelivery(event.target.checked)} />Include Delivery</label></div>
    {result.error ? <p className="tdp-trace-error" role="status">{result.error}</p> : <>
      <p className="tdp-trace-summary"><b>{result.cells.length} tokens</b> · marker positions [{result.markers.join(', ')}] · target index {result.target} for semantic ID <code>billing</code> · {result.unknownCount} unknown words</p>
      <p className="tdp-small">Each cell shows position → token ID → original word. Outlined OPTION cells mark the vectors the scorer will read. A word tagged UNK has ID 1, even when its original spelling differs.</p>
      <ol className="tdp-token-grid" aria-label="Token positions and vocabulary IDs">{result.cells.map(cell => <li key={cell.position} className={cell.word === '<option>' ? 'tdp-marker-cell' : ''}>
        <span>pos {cell.position} · ID {cell.id}</span><code>{cell.word}</code><small>{cell.id === 1 ? 'UNK · ' : ''}{cell.section}</small>
      </li>)}</ol>
      <div className="tdp-marker-rows">{options.map((option, index) => <div key={option.id}>
        <code>hidden[{result.markers[index]}]</code><span aria-hidden="true">→</span><strong>{option.id}</strong><span>candidate {index}{index === result.target ? ' · target' : ''}</span>
      </div>)}</div>
    </>}
    <p className="tdp-small">The target remains Billing for this fixed routing exercise. Editing the text does not create a new human label. Notice that reordering options changes positions and the target index together.</p>
  </section>;
}

export function AttentionLab() {
  const [queryFirst, setQueryFirst] = useState(1);
  const [maskPadding, setMaskPadding] = useState(true);
  const result = attentionMixture(queryFirst, maskPadding);
  return <section className="tdp-explorer tdp-attention-lab" aria-label="Explore an attention row and padding mask">
    <div className="tdp-explorer-heading"><div><span className="tdp-eyebrow">One query, four keys</span><h3>Attention mixes values; a mask excludes evidence</h3></div><button type="button" onClick={() => { setQueryFirst(1); setMaskPadding(true); }}>Reset attention</button></div>
    <p className="tdp-small">Constructed two-dimensional vectors isolate the exact dot-product, scaling and softmax operations. These are not learned weights or an explanation of a trained model’s reasoning.</p>
    <Slider id="tdp-query" label="First query coordinate" value={queryFirst} min={-2} max={2} onChange={setQueryFirst} />
    <p>Query = [{queryFirst.toFixed(2)}, 0]. Divide each dot product by √2 before softmax.</p>
    <label className="tdp-check"><input type="checkbox" checked={maskPadding} onChange={event => setMaskPadding(event.target.checked)} />Mask the padding key</label>
    <div className="tdp-table-wrap tdp-live-table" tabIndex={0} aria-label="Attention weights and weighted values"><table><thead><tr><th>Key / value</th><th>Scaled score</th><th>Attention weight</th><th>Weighted value</th></tr></thead><tbody>
      {attentionKeys.map((key, index) => <tr key={index}><td data-label="Key / value"><b>{index === 3 ? 'Padding' : `Token ${index + 1}`}</b><br />K {vector(key)}<br />V {vector(attentionValues[index])}</td>
        <td data-label="Scaled score">{maskPadding && index === 3 ? '−∞ (masked)' : result.scores[index].toFixed(3)}</td>
        <td data-label="Attention weight">{(100 * result.probabilities[index]).toFixed(1)}%<div className="tdp-bar-track"><span style={{ width: `${100 * result.probabilities[index]}%` }} /></div></td>
        <td data-label="Weighted value">{vector(result.contributions[index])}</td></tr>)}
    </tbody></table></div>
    <p className="tdp-attention-output"><strong>Output = {vector(result.output)}</strong> · sum of the weighted value rows.</p>
    <p>{maskPadding ? 'Padding contributes exactly zero. At query [0, 0], the three valid tokens each receive one third.' : 'The mask is deliberately disabled: the padding vector now contaminates the output. This is an error demonstration, not an alternative training recommendation.'}</p>
  </section>;
}

export function ScorerTrainingLab() {
  const [weights, setWeights] = useState([2, 1]);
  const [target, setTarget] = useState(0);
  const [learningRate, setLearningRate] = useState(0.5);
  const [steps, setSteps] = useState(0);
  const result = scoringHeadStep(weights, target, learningRate);
  const reset = () => { setWeights([2, 1]); setTarget(0); setLearningRate(0.5); setSteps(0); };
  return <section className="tdp-explorer tdp-training-lab" aria-label="Train a shared scoring head one step at a time">
    <div className="tdp-explorer-heading"><div><span className="tdp-eyebrow">From loss to weights</span><h3>One shared head, one visible update</h3></div><button type="button" onClick={reset}>Reset training</button></div>
    <p className="tdp-small">Freeze three illustrative candidate vectors and fit the shared linear scorer with plain SGD. This isolates the chain rule; the complete program trains the encoder too and uses AdamW.</p>
    <label className="tdp-text-control">Correct candidate<select aria-label="Correct candidate" value={target} onChange={event => { setTarget(Number(event.target.value)); setSteps(0); }}>{names.map((name, index) => <option key={name} value={index}>{name}</option>)}</select></label>
    <Slider id="tdp-learning-rate" label="SGD learning rate" value={learningRate} min={0} max={2} onChange={setLearningRate} />
    <div className="tdp-table-wrap tdp-live-table" tabIndex={0} aria-label="Shared head scores and gradients"><table><thead><tr><th>Candidate / frozen h</th><th>Score w·h</th><th>Probability p</th><th>Score gradient p − y</th></tr></thead><tbody>
      {names.map((name, index) => <tr key={name}><td data-label="Candidate / frozen h">{name}{index === target ? ' · target' : ''}<br /><code>{vector(candidateFeatures[index])}</code></td><td data-label="Score w·h">{result.scores[index].toFixed(3)}</td><td data-label="Probability p">{result.probabilities[index].toFixed(4)}</td><td data-label="Score gradient p − y">{result.scoreGradients[index].toFixed(4)}</td></tr>)}
    </tbody></table></div>
    <ol className="tdp-update-flow"><li><span>Current weights</span><code>{vector(weights)}</code></li><li><span>Weight gradient Σ(p − y)h</span><code>{vector(result.gradient)}</code></li><li><span>Proposed weights: w − η∇L</span><code>{vector(result.nextWeights)}</code></li></ol>
    <p className="tdp-step-loss">Cross-entropy: <b>{result.loss.toFixed(5)}</b> now → <b>{result.nextLoss.toFixed(5)}</b> after the proposed step.</p>
    <button type="button" disabled={learningRate === 0 || steps >= 40} onClick={() => { setWeights(result.nextWeights); setSteps(value => value + 1); }}>Apply one SGD step</button>
    <p className="tdp-small">{steps} {steps === 1 ? 'step' : 'steps'} applied to this target. {learningRate === 0 ? 'A zero learning rate leaves every weight unchanged.' : steps >= 40 ? 'Reset to start another bounded investigation.' : 'Change the learning rate to inspect the proposed update immediately; apply it to make those weights current.'}</p>
    <p>A shared bias would add the same constant to all three scores and cancel in softmax. Changing one shared weight, however, moves different scores according to their candidate vectors. This is why “increase only the correct logit” is not a complete description of a parameter update.</p>
  </section>;
}
