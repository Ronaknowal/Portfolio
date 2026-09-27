import { useId, useState } from 'react';
import './semi-supervised-intuition.css';

export function DegreeNormalizedAgreementFigure() {
  const degrees = [1, 2, 1];
  const cases = [[1, 1, 1], [1, Math.SQRT2, 1]];
  return <figure className="ssl-intuition">
    <h4>Agreement depends on the coordinates being compared</h4>
    <p>A—B—C has unit edge weights and degrees 1, 2, 1. These are constructed score columns, not fitted class probabilities.</p>
    <div className="ssl-intuition-panels">{cases.map((scores, index) => {
      const normalized = scores.map((value, i) => value / Math.sqrt(degrees[i]));
      const energy = (normalized[0] - normalized[1]) ** 2 + (normalized[1] - normalized[2]) ** 2;
      return <section key={index}><h5>{index === 0 ? 'Equal raw scores' : 'Equal normalized scores'}</h5>
        <svg viewBox="0 0 300 225" role="img" aria-label={`${index === 0 ? 'Equal raw' : 'Equal normalized'} scores; normalized energy ${energy.toFixed(6)}. Exact values below.`}>
          {[0, 1, 2].map(i => <g key={i}>
            <rect x={40 + i * 85} y={145 - 65 * scores[i]} width="22" height={65 * scores[i]} fill="#d9d6cf" />
            <rect x={64 + i * 85} y={145 - 65 * normalized[i]} width="22" height={65 * normalized[i]} fill="#e9b949" />
            <text x={63 + i * 85} y="170" textAnchor="middle">{'ABC'[i]}</text>
          </g>)}
          <line x1="25" y1="145" x2="280" y2="145" stroke="currentColor" />
          <text x="150" y="205" textAnchor="middle">Common bar-height scale in both panels</text>
        </svg>
        <p>Raw: {scores.map(x => x.toFixed(3)).join(', ')}<br />Normalized: {normalized.map(x => x.toFixed(3)).join(', ')}<br /><strong>Normalized energy: {energy.toFixed(6)}</strong></p>
      </section>;
    })}</div>
    <figcaption>White bars show fᵢ; amber bars show fᵢ / √dᵢ. The normalized smoothness term compares amber bars across edges. Zero smoothness cost alone does not make these scores the fitted solution: label fidelity still contributes to the full objective.</figcaption>
  </figure>;
}

export function ConsistencyTargetLab() {
  const id = useId();
  const [weak, setWeak] = useState(0.98);
  const [strong, setStrong] = useState(0.4);
  const [threshold, setThreshold] = useState(0.95);
  const accepted = weak >= threshold;
  const loss = accepted ? -Math.log(strong) : 0;
  const derivative = accepted ? strong - 1 : 0;
  return <figure className="ssl-intuition">
    <h4>Explore which branch supplies the target and which receives the gradient</h4>
    <p>One constructed binary example. The weak view always favors class 1 here. Controls change supplied model outputs; this does not simulate image augmentation or train a network.</p>
    <div className="ssl-intuition-controls">{[
      ['weak', 'Weak-view class-1 probability', weak, setWeak, 0.51],
      ['threshold', 'Acceptance threshold', threshold, setThreshold, 0.5],
      ['strong', 'Strong-view class-1 probability', strong, setStrong, 0.01],
    ].map(([name, label, value, setter, min]) => <label key={name} htmlFor={`${id}-${name}`}>{label}: <output>{value.toFixed(2)}</output><input id={`${id}-${name}`} type="range" min={min} max="0.99" step="0.01" value={value} onChange={event => setter(Number(event.target.value))} /></label>)}</div>
    <div className="ssl-intuition-panels">
      <section><h5>1 · Freeze the proposed target</h5><p>Weak output: [{(1 - weak).toFixed(2)}, {weak.toFixed(2)}]</p><p>{accepted ? 'Accepted → target [0, 1]' : 'Rejected → mask 0'}</p><p>No gradient through the hard target or threshold decision in this update.</p></section>
      <section><h5>2 · Score the strong view</h5><p>Strong output: [{(1 - strong).toFixed(2)}, {strong.toFixed(2)}]</p><p>Masked loss: <strong>{loss.toFixed(4)}</strong><br />Derivative w.r.t. strong class-1 logit: <strong>{derivative.toFixed(4)}</strong></p><p>{accepted ? 'A negative derivative asks gradient descent to raise that logit.' : 'This unlabeled example contributes no gradient. The supervised loss still trains the model.'}</p></section>
    </div>
    <button type="button" onClick={() => { setWeak(0.98); setStrong(0.4); setThreshold(0.95); }}>Reset example</button>
    <figcaption>Move the weak score across the threshold, then change the strong score. Once accepted, making the weak score more confident does not further scale this hard-label loss. Confidence selects a target; it does not establish that the target is correct.</figcaption>
  </figure>;
}
