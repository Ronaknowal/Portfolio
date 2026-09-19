import { useState } from 'react';
import { LessonTable } from './LessonElements.jsx';
import { BIO_LABELS, chainDistribution, initialFactors, legalBio, legalBioPath } from '../../data/crf-models.js';
import data from '../../data/crf-data.json';
import './crf.css';

export const number = value => Number(value.toFixed(6)).toString();
export function CrfBar({ value, maximum = 1, label }) {
  return <div className="crf-bar-row"><span>{label}</span><div className="crf-track"><span style={{ width: `${100 * value / maximum}%` }} /></div><strong>{number(value)}</strong></div>;
}
export function CrfFactorFigure() {
  const words = ['Maya', 'Chen', 'joined', 'Cedar', 'Labs'];
  const labels = ['B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG'];
  return <figure className="crf-figure" aria-label="Observed words and the output factor chain">
    <figcaption><strong>All five words are observed.</strong> Each input score may inspect this whole sentence.</figcaption>
    <div className="crf-chain-head"><span>Observed word</span><span>Input factor</span><span>Output label</span></div>
    {words.map((word, index) => <div className="crf-chain-unit" key={word}>
      <div className="crf-chain-row"><strong>{word}</strong><span className="crf-factor">e{index + 1}(y)</span><span className="crf-node">y{index + 1}</span><span className="crf-tag">{labels[index]}</span></div>
      {index < 4 && <div className="crf-pair-join"><span aria-hidden="true">│</span><span className="crf-factor">pair<br />{index + 1}–{index + 2}</span><span aria-hidden="true">│</span></div>}
    </div>)}
    <p className="lesson-note">Circles are unknown labels; boxes are score factors. Each pair factor involves only adjacent labels. The right-hand strip is one legal labeling of this invented sentence. Lines show factor scope, not causal direction.</p>
  </figure>;
}
export function CrfPathLedger({ model = chainDistribution(initialFactors), selected, onSelect }) {
  return <figure className="crf-figure" aria-label="Complete path masses and probabilities">
    <figcaption><strong>Four paths compete for one total: Z = {number(model.partition)}.</strong> Every bar uses the same 0–{number(model.partition)} mass scale.</figcaption>
    {model.paths.map(path => <div className="crf-path-row" key={path.name} data-selected={selected === path.name}>
      {onSelect ? <button type="button" aria-pressed={selected === path.name} onClick={() => onSelect(path.name)}>Trace {path.name}</button> : <strong>{path.name}</strong>}
      <span>{path.factors.map(number).join(' × ')} = {number(path.mass)}</span>
      <CrfBar label={path.name} value={path.mass} maximum={model.partition} />
      <span className="lesson-note">Probability {number(path.mass)}/{number(model.partition)} = {number(path.probability)}</span>
    </div>)}
    <p>Group by first label: A has mass {number(model.paths[0].mass + model.paths[1].mass)} and probability {number(model.nodes[0][0])}; B has mass {number(model.paths[2].mass + model.paths[3].mass)} and probability {number(model.nodes[0][1])}.</p>
  </figure>;
}
export function CrfMessagesFigure() {
  return <figure className="crf-figure" aria-label="Prefix and suffix messages meet">
    <figcaption><strong>Cut at position 1: multiply compatible prefixes and suffixes.</strong></figcaption>
    {[['A', 3, 9, 27, '.9'], ['B', 1, 3, 3, '.1']].map(([label, prefix, suffix, product, probability]) => <div className="crf-message" key={label}>
      <span>prefix α: <strong>{prefix}</strong></span><span aria-hidden="true">→</span><span className="crf-node">{label}</span><span aria-hidden="true">←</span><span>suffix β: <strong>{suffix}</strong></span>
      <strong>{prefix} × {suffix} = {product}; divide by 30 → {probability}</strong>
    </div>)}
    <LessonTable caption="Both cuts recover the same normalizer" headers={['Position', 'Forward A, B', 'Backward A, B', 'Marginals A, B']} rows={[[1, '3, 1', '9, 3', '.9, .1'], [2, '4, 26', '1, 1', '4/30, 26/30']]} />
    <p className="lesson-note">These are the same 27 and 3 obtained by grouping whole paths. At position 2, the forward B value 26 includes AB (24) and BB (2).</p>
  </figure>;
}
export function CrfCountFigure() {
  const [gold, setGold] = useState('AB');
  const observed = gold === 'AB' ? 1 : 0;
  return <figure className="crf-figure" aria-label="Observed and expected AB feature counts">
    <figcaption><strong>Change the gold labeling; hold the parameters fixed.</strong></figcaption>
    <label>Gold path <select value={gold} onChange={event => setGold(event.target.value)}>{['AA', 'AB', 'BA', 'BB'].map(path => <option key={path}>{path}</option>)}</select></label>
    <CrfBar label="Observed AB count" value={observed} /><CrfBar label="Expected AB count" value={.8} />
    <p aria-live="polite">Log-likelihood ascent direction: {observed} − 0.8 = <strong>{number(observed - .8)}</strong>. {observed ? 'Increase' : 'Decrease'} the AB log weight. Both count bars use 0–1.</p>
  </figure>;
}
export function CrfBioFigure() {
  const [target, setTarget] = useState('I-PER');
  return <figure className="crf-figure" aria-label="BIO predecessor rules and hard start mask">
    <figcaption><strong>Which labels may precede this continuation?</strong></figcaption>
    <label>Continuation <select value={target} onChange={event => setTarget(event.target.value)}><option>I-PER</option><option>I-ORG</option></select></label>
    <div className="crf-bio-routes">{['START', ...BIO_LABELS].map(previous => <div key={previous}><span className="crf-tag">{previous}</span><span>{legalBio(previous, target) ? '→ allowed' : '× forbidden'}</span><span className="crf-tag">{target}</span></div>)}</div>
    {[[ 'B-PER', 'I-PER', 'O'], ['O', 'I-PER', 'O']].map(path => <p key={path.join()}>{path.join(' → ')}: <strong>{legalBioPath(path) ? 'valid' : 'invalid: I-PER has no person-span predecessor'}</strong>.</p>)}
    <LessonTable caption="One-token scores: a preference versus a hard start rule" headers={['Label', 'Raw log score', 'Allowed at start', 'Masked mass']} rows={[[ 'O', 0, 'yes', 1], ['B-PER', 0, 'yes', 1], ['I-PER', 10, 'no', 0]]} />
    <p>Unconstrained: I-PER wins with mass exp(10). With the start mask: Z = 2, O and B-PER each have probability 1/2; I-PER has probability 0. If all paths were forbidden, there would be no distribution to normalize.</p>
  </figure>;
}
export function CrfRealFigure() {
  const defaultIndex = data.development.findIndex(row => row.gold.some((label, index) => label !== row.prediction[index]));
  const [sentenceIndex, setSentenceIndex] = useState(defaultIndex);
  const [tokenIndex, setTokenIndex] = useState(0);
  const row = data.development[sentenceIndex];
  const names = ['NOUN', 'VERB', 'OTHER'];
  const selected = Math.min(tokenIndex, row.tokens.length - 1);
  return <figure className="crf-figure" aria-label="Inspect real EWT development tagging errors">
    <figcaption><strong>Real development errors, one token at a time.</strong> Recorded outputs of the downloadable program; selecting a sentence does not retrain it.</figcaption>
    <label>Development sentence <select value={sentenceIndex} onChange={event => {setSentenceIndex(Number(event.target.value)); setTokenIndex(0);}}>{data.development.map((sentence, index) => <option value={index} key={sentence.id}>{sentence.id}</option>)}</select></label>
    <ol className="crf-token-list">{row.tokens.map((token, index) => <li key={index} data-error={row.gold[index] !== row.prediction[index]}>
      <button type="button" aria-pressed={selected === index} onClick={() => setTokenIndex(index)}>{index + 1}. {token}</button><span>Gold: {names[row.gold[index]]}</span><span>Predicted: {names[row.prediction[index]]} {row.gold[index] !== row.prediction[index] ? '× error' : '✓'}</span>
    </li>)}</ol>
    <div className="crf-real-detail" aria-live="polite"><strong>Token {selected + 1}: {row.tokens[selected]}</strong>
      {names.map((name, index) => <CrfBar key={name} label={name} value={row.marginals[selected][index]} />)}
      <p>Marginal probability scale: 0–1. {selected ? <>Selected decoded edge: {names[row.prediction[selected - 1]]} → {names[row.prediction[selected]]}; transition log score <strong>{number(data.transitions[row.prediction[selected - 1]][row.prediction[selected]])}</strong>. This is one contribution to the full path score.</> : 'The first token has no preceding pair contribution.'}</p>
    </div>
    <details><summary>Original annotation and data provenance</summary><p>Sentence ID: {row.id}. Original UPOS: {row.upos.join(', ')}.</p><p>UD English EWT r2.16; first 40 short development sentences, coarse NOUN/VERB/OTHER mapping. <a href="/learn-assets/crf/data-provenance.md">Attribution and extraction</a> · <a href="/learn-assets/crf/ewt-sequences.json" download>Download the offline data</a>.</p></details>
    <div className="crf-real-metrics"><strong>Development token accuracy (common 0–1 scale)</strong><CrfBar label="Independent · 269/341" value={269 / 341} /><CrfBar label="Chain · 281/341" value={281 / 341} /><p>Exact sentences: independent 11/40; chain 12/40. Selected-chain final test: 293/370 tokens and 7/40 complete sentences. The small extraction is not a representative benchmark.</p></div>
  </figure>;
}
export function CrfNeuralFigure() {
  return <figure className="crf-figure" aria-label="Neural encoder and CRF training versus decoding">
    <figcaption><strong>One likelihood loss trains two parts.</strong> B: batch, T: positions, H: encoder width, K: labels.</figcaption>
    <div className="crf-two-panel"><div><h4>Input-score branch</h4><ol className="crf-neural-flow"><li>Observed token IDs</li><li>↓ encoder produces (B, T, H)</li><li>↓ linear layer produces scores (B, T, K)</li></ol><p>↑ Loss gradient returns through both layers.</p></div><div><h4>Pair-score branch</h4><p>Learned matrix A has shape (K, K).</p><p>Rows: previous label i.<br />Columns: current label j.</p><p>↓ A<sub>ij</sub> contributes whenever that adjacent label pair occurs.</p><p>↑ Loss gradient updates the pair matrix.</p></div></div>
    <p className="crf-neural-join"><strong>↓ Both score branches enter the CRF ↓</strong><br />Valid-position mask selects the real tokens.</p>
    <div className="crf-two-panel"><div><h4>Training</h4><p>Gold-path score and log Z → loss = log Z − gold score.</p><p>Gradients return through input scores to the encoder, and separately into pair scores.</p></div><div><h4>Prediction</h4><p>Maxima + backpointers → one decoded label sequence.</p><p>The likelihood does not differentiate through this discrete path.</p></div></div>
    <p className="lesson-note">Mask padded positions in the loss and decoder. A bidirectional encoder needs its own padding handling upstream. Pair rows are previous labels; columns are current labels.</p>
  </figure>;
}
