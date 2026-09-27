import { useId, useState } from 'react';
import { chainDistribution, initialFactors } from '../../data/crf-models.js';
import './crf-intuition.css';

export function CrfPairMemoryFigure() {
  return <figure className="crf-intuition" data-concept="second-order-pair-memory">
    <figcaption><strong>A second-order score needs the last two labels before choosing the next one.</strong></figcaption>
    <p>At a prefix ending in A,B, retain the pair (A,B). A new label completes a three-label score; then drop the oldest label.</p>
    <div className="crf-pair-memory">
      <div><span className="crf-memory-token">A</span><span className="crf-memory-token is-shared">B</span><strong>retained pair</strong></div>
      <div className="crf-memory-options"><p>Add A → score (A,B,A) → retain <b>(B,A)</b></p><p>Add B → score (A,B,B) → retain <b>(B,B)</b></p></div>
    </div>
    <p>The next pair must begin with B because that is the label both windows share. A transition from pair (A,B) directly to (A,A) is inconsistent: it would replace the shared label rather than slide the window.</p>
    <p>There are K² possible remembered pairs, but only K consistent extensions per pair. That gives K³ candidate triple updates per position, explaining O(TK³), instead of K⁴ transitions between arbitrary pairs. This counts dense second-order label work; it excludes input encoding.</p>
  </figure>;
}

export function CrfPartialLabelLab() {
  const id = useId();
  const [observation, setObservation] = useState('first');
  const [pairFactor, setPairFactor] = useState(4);
  const factors = [...initialFactors];
  factors[5] = pairFactor;
  const model = chainDistribution(factors);
  const compatible = path => observation === 'none' || (observation === 'first' ? path.i === 0 : path.name === 'AB');
  const retained = model.paths.filter(compatible).reduce((sum, path) => sum + path.mass, 0);
  const probability = retained / model.partition;
  return <figure className="crf-intuition" data-concept="partial-label-likelihood">
    <figcaption><strong>Incomplete labels teach a set of possible paths, not a guessed completion.</strong></figcaption>
    <p>Use the same two-position model. Change what is observed, then move the A→B factor. All other factors stay fixed; the model is recomputed immediately without fitting or sampling.</p>
    <label className="crf-intuition-control" htmlFor={`${id}-observation`}>Observed labels
      <select id={`${id}-observation`} value={observation} onChange={event => setObservation(event.target.value)}><option value="both">Both: AB</option><option value="first">First is A; second unknown</option><option value="none">Neither label observed</option></select>
    </label>
    <label className="crf-intuition-control" htmlFor={`${id}-factor`}>A→B positive factor: {pairFactor}
      <input id={`${id}-factor`} type="range" min="0.125" max="16" step="0.125" value={pairFactor} onChange={event => setPairFactor(Number(event.target.value))} />
    </label>
    <div className="crf-partial-paths">{model.paths.map(path => <div key={path.name} className={compatible(path) ? 'is-compatible' : ''}><strong>{path.name}</strong><span>mass {Number(path.mass.toFixed(3))}</span><span>{compatible(path) ? 'Matches known labels' : 'Contradicts known labels'}</span></div>)}</div>
    <p aria-live="polite" className="crf-intuition-result">Compatible mass {Number(retained.toFixed(3))} / all-path mass {Number(model.partition.toFixed(3))} = {probability.toFixed(6)}<br />Negative log-likelihood = {(probability === 1 ? 0 : -Math.log(probability)).toFixed(6)}</p>
    <p>At the original factor 4, fully observed AB gives 24/30=.8; only the first A gives (3+24)/30=.9; no labels gives 30/30=1. In that last mode the loss stays zero while the path probabilities still change: ordinary conditional likelihood receives no labeling information from this input alone.</p>
    <button type="button" onClick={() => { setObservation('first'); setPairFactor(4); }}>Reset worked case</button>
  </figure>;
}

export function CrfBackwardSamplingFigure() {
  return <figure className="crf-intuition" data-concept="backward-path-sampling">
    <figcaption><strong>Sample a whole path by keeping its label dependencies intact.</strong></figcaption>
    <p>Original factor-4 model. First draw the final label from [4/30,26/30]. Then draw the preceding label conditional on that result.</p>
    <div className="crf-sampling-branches">
      <section><strong>Final label A: chance 4/30</strong><p>Earlier-label weights: A has 3·1=3; B has 1·1=1.</p><p>A first: 3/4 → path AA<br />B first: 1/4 → path BA</p><p className="crf-intuition-result">AA: (4/30)(3/4)=3/30<br />BA: (4/30)(1/4)=1/30</p></section>
      <section><strong>Final label B: chance 26/30</strong><p>Earlier-label weights: A has 3·4=12; B has 1·1=1.</p><p>A first: 12/13 → path AB<br />B first: 1/13 → path BB</p><p className="crf-intuition-result">AB: (26/30)(12/13)=24/30<br />BB: (26/30)(1/13)=2/30</p></section>
    </div>
    <p>Every path regains its original probability. Drawing the first label independently from its marginal .9 would instead give AB probability .9·26/30=.78, changing the joint distribution. These branches describe exact sampling probabilities, not observed frequencies from a finite run.</p>
  </figure>;
}
